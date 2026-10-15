"""Prepare all existing CTC cases while retaining the original numerical acceptance bound."""
import argparse
import json
from pathlib import Path
import sys
import wave

import numpy as np
import onnxruntime as ort

from export import ROOT, digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


def prepare_cases(output):
    weight_digest = digest(output / "ctc.weights")
    reference_path = ROOT / "crates/models/tests/data/ctc-reference.json"
    reference = json.loads(reference_path.read_text())
    wav_path = ROOT / "crates/models/tests/data/two-voice.wav"
    assert digest(wav_path) == reference["wav_sha256"]
    with wave.open(str(wav_path)) as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output / "ctc.onnx"), options, providers=["CPUExecutionProvider"])
    ids = np.asarray(reference["token_ids"], dtype=np.int64)
    cases = [pcm, pcm[16000:16400], pcm[16000:16719], pcm[16000:16720], pcm[16000:16549], np.zeros(720, np.float32)]
    results = []
    for audio, expected in zip(cases, reference["cases"]):
        normalized = ((audio - audio.mean()) / np.sqrt(audio.var() + np.float32(1e-5)))[None, :]
        scores, = session.run(None, {"x": normalized, "token_ids": ids})
        wanted = np.asarray(expected["gathered"], dtype=np.float32)[None, :, :]
        assert scores.shape == wanted.shape
        error = float(np.max(np.abs(scores - wanted)))
        results.append(dict(case=expected["name"], max_absolute_error=error, passed=error < 1e-4))
        print(results[-1], flush=True)
        case_dir = output / expected["name"]
        case_dir.mkdir()
        normalized.astype("<f4").tofile(case_dir / "audio.f32")
        ids.astype("<i8").tofile(case_dir / "ids.i64")
        wanted.astype("<f4").tofile(case_dir / "expected.f32")
        case = dict(model="../ctc.onnx", model_files={"../ctc.weights": weight_digest}, repeat=2, inputs=[
            dict(name="x", dtype="f32", shape=list(normalized.shape), path="audio.f32"),
            dict(name="token_ids", dtype="i64", shape=list(ids.shape), path="ids.i64"),
        ])
        (case_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    result = dict(passed=all(row["passed"] for row in results), cases=results,
                  reference_sha256=digest(reference_path), graph_sha256=digest(output / "ctc.onnx"),
                  weights_sha256=weight_digest, runtime=ort.__version__, provider="CPUExecutionProvider")
    (output / "cpu-qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_native(case, output, provider):
    record = verify(case, output, provider)
    specification = json.loads((case / "case.json").read_text())
    wanted = np.fromfile(case / "expected.f32", "<f4")
    frames = (specification["inputs"][0]["shape"][1] - 400) // 320 + 1
    shape = [1, frames, specification["inputs"][1]["shape"][0]]
    if wanted.size != int(np.prod(shape)) or len(record["outputs"]) != 2:
        raise ValueError("incomplete CTC scores")
    reports, previous = [], None
    for run, row in enumerate(record["outputs"]):
        path = output / row["file"]
        if (row["name"] != "scores" or row["shape"] != shape or row["dtype"] != "f32"
                or row["run"] != run or row["sha256"] != digest(path)):
            raise ValueError("native CTC output identity differs")
        scores = np.fromfile(path, "<f4")
        if scores.shape != wanted.shape or not np.isfinite(scores).all():
            raise ValueError("invalid native CTC scores")
        error = float(np.max(np.abs(scores - wanted)))
        repeat = 0.0 if previous is None else float(np.max(np.abs(scores - previous)))
        reports.append(dict(run=run, max_absolute_error=error, repeat_error=repeat,
                            passed=error < 1e-4 and repeat == 0))
        previous = scores
    result = dict(passed=all(row["passed"] for row in reports), reports=reports)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--native-result", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"))
    args = parser.parse_args()
    result = (compare_native(args.directory, args.native_result, args.provider)
              if args.native_result else prepare_cases(args.directory))
    raise SystemExit(0 if result["passed"] else 1)
