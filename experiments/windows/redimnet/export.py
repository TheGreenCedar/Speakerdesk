"""Build-only export of the original pinned ReDimNet2 B6 graph; no app Python."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import wave

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[3]
PINS = json.loads(Path(__file__).with_name("source-pins.json").read_text())
IDENTITY_REUSE_COSINE = 0.9999
INPUT_SAMPLES = 96000
DIMENSION = 192


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def source_class(source):
    source = source.resolve()
    actual = {str(path.relative_to(source)).replace("\\", "/")
              for path in (source / "redimnet2").rglob("*.py")}
    assert actual == set(PINS["files"]) - {"LICENSE"}, "unexpected upstream Python inventory"
    for name, expected in PINS["files"].items():
        assert sha256(source / name) == expected, ("upstream source differs", name)
    sys.path.insert(0, str(source))
    module = importlib.import_module("redimnet2.redimnet2")
    assert Path(module.__file__).resolve() == source / "redimnet2/redimnet2.py"
    return module.ReDimNet2Wrap


def prepared_audio(pcm):
    pcm = np.asarray(pcm, dtype=np.float32)
    assert pcm.ndim == 1 and 32000 <= pcm.size <= 160000 and np.isfinite(pcm).all()
    if pcm.size >= INPUT_SAMPLES:
        start = (pcm.size - INPUT_SAMPLES) // 2
        return pcm[start:start + INPUT_SAMPLES].copy()
    return np.resize(pcm, INPUT_SAMPLES)


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    assert vector.shape == (DIMENSION,) and np.isfinite(vector).all()
    norm = np.linalg.norm(vector)
    assert np.isfinite(norm) and norm > 0
    return vector / norm


class HalfBackbone(torch.nn.Module):
    """Keep the existing frontend/head precision boundary around the maintained backbone."""
    def __init__(self, backbone):
        super().__init__()
        self.graph = backbone.half()

    def forward(self, features):
        return self.graph(features.half()).float()


class Embedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.graph = model
        self.graph.backbone = HalfBackbone(model.backbone)
        self.eval()

    def forward(self, audio):
        return torch.nn.functional.normalize(self.graph(audio), dim=-1)


def export_graph(model, example, output):
    path = output / "redimnet.onnx"
    torch.onnx.export(model, (example,), path, input_names=["audio"], output_names=["embedding"],
                      opset_version=17, dynamo=False, do_constant_folding=False,
                      training=torch.onnx.TrainingMode.EVAL)
    graph = onnx.load(path)
    onnx.save_model(graph, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="redimnet.weights", size_threshold=1024)
    onnx.checker.check_model(str(path))


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    assert sha256(args.checkpoint) == PINS["checkpoint_sha256"], "unpinned ReDimNet checkpoint"
    model_type = source_class(args.source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"model_config", "state_dict"}
    config = checkpoint["model_config"]
    assert config["feat_type"] == "tf" and config["embed_dim"] == DIMENSION
    assert config["spec_params"] == {"do_preemph": True, "do_spec_aug": False, "norm_signal": True}
    model = model_type(**config).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = Embedding(model)
    reference_path = ROOT / "crates/models/tests/data/redimnet-mlx-reference.json"
    reference = json.loads(reference_path.read_text())
    wav_path = ROOT / "crates/models/tests/data/two-voice.wav"
    assert sha256(wav_path) == reference["wav_sha256"]
    with wave.open(str(wav_path)) as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    clips = (pcm[:64000], pcm[80000:])
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, clip in enumerate(clips):
        expected = reference["cases"][index]
        assert clip.size == expected["samples"] and expected["speaker"] == index
        audio = torch.from_numpy(prepared_audio(clip)[None, :])
        start = time.perf_counter()
        result = model(audio)
        vector = normalized(result.numpy())
        cosine = float(np.dot(vector, expected["vector"]))
        row = dict(speaker=index, samples=clip.size, legacy_cosine=cosine,
                   seconds=time.perf_counter() - start, vector=vector.tolist())
        rows.append(row)
        json_write(args.output / "eager-reference-comparison.json", rows)
        assert cosine >= IDENTITY_REUSE_COSINE, ("saved-profile compatibility gate failed", row)
        case = args.output / f"voice-{index}-case"
        case.mkdir()
        audio.numpy().tofile(case / "audio.f32")
        json_write(case / "case.json", dict(model="../redimnet.onnx", repeat=3,
                    inputs=[dict(name="audio", dtype="f32", shape=[1, INPUT_SAMPLES], path="audio.f32")]))
        json_write(case / "expected.json", expected)
    print(json.dumps({"independent_legacy_cosines": [row["legacy_cosine"] for row in rows]}), flush=True)
    export_graph(model, torch.from_numpy(prepared_audio(clips[0])[None, :]), args.output)
    weight_hash = sha256(args.output / "redimnet.weights")
    for index in range(len(clips)):
        path = args.output / f"voice-{index}-case/case.json"
        case = json.loads(path.read_text())
        case["model_files"] = {"../redimnet.weights": weight_hash}
        json_write(path, case)
    shutil.copyfile(reference_path, args.output / "independent-reference.json")
    shutil.copyfile(args.source / "LICENSE", args.output / "LICENSE")
    for name in PINS["files"]:
        destination = args.output / "upstream-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / name, destination)
    files = [path for path in args.output.rglob("*") if path.is_file()]
    provenance = dict(source=PINS, model_config=config,
        precision="FP32 waveform normalization, convolutional DFT/mel/log/features and pooling/head; FP16 backbone; FP32 L2 output",
        independent_reference_sha256=sha256(reference_path), wav_sha256=sha256(wav_path),
        exporter_sha256=sha256(__file__), identity_reuse_cosine=IDENTITY_REUSE_COSINE,
        packages={name: importlib.metadata.version(name) for name in
                  ("torch", "torchaudio", "numpy", "scipy", "onnx")},
        outputs={str(path.relative_to(args.output)).replace("\\", "/"):
                 dict(size=path.stat().st_size, sha256=sha256(path)) for path in files})
    json_write(args.output / "provenance.json", provenance)
    print(json.dumps({"output": str(args.output), "weights_sha256": weight_hash,
                      "legacy_cosines": [row["legacy_cosine"] for row in rows]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
