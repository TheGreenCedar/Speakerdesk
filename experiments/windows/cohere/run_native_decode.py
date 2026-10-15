"""Build-only qualification: drive native ort-probe through the full retained Cohere decode."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnx
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_probe(args, label, model, inputs):
    case_dir = args.output / (label + "-case")
    case_dir.mkdir()
    case = case_dir / "case.json"
    provenance = json.loads((args.artifacts / "provenance.json").read_text())
    weights = "encoder.weights" if model == "encoder.onnx" else "decoder.weights"
    model_files = {os.path.relpath(args.artifacts / weights, case_dir):
                   provenance["outputs"][weights]["sha256"]}
    write_json(case, dict(model=str((args.artifacts / model).resolve()), repeat=1,
                         inputs=inputs, model_files=model_files))
    output = args.output / label
    with (args.output / (label + ".log")).open("w", encoding="utf-8") as stream:
        subprocess.run([str(args.probe), str(args.runtime), args.provider, str(case), str(output),
                        str(args.adapter)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = verify(case_dir, output, args.provider)
    records = {value["name"]: {**value, "path": str((output / value["file"]).resolve())}
               for value in report["outputs"] if value["run"] == 0}
    return records, report


def bind(name, record):
    return dict(name=name, dtype=record["dtype"], shape=record["shape"], path=record["path"])


def tensor(directory, name, array):
    dtype = {np.dtype("float32"): "f32", np.dtype("int64"): "i64"}[array.dtype]
    path = directory / (name + "." + dtype)
    array.tofile(path)
    return dict(name=name, dtype=dtype, shape=list(array.shape), path=str(path.resolve()))


def run(args):
    args.probe, args.runtime, args.artifacts, args.output = [path.resolve() for path in
        (args.probe, args.runtime, args.artifacts, args.output)]
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.artifacts / "independent-reference.json").read_text())
    prompt = json.loads((args.artifacts / "prompts.json").read_text())["en"]
    source_case = args.artifacts / "encoder-case"
    inputs = json.loads((source_case / "case.json").read_text())["inputs"]
    for value in inputs:
        value["path"] = str((source_case / value["path"]).resolve())
        if value["name"] == "features" and args.features:
            value["path"] = str(args.features.resolve())
    encoded, encoder_report = run_probe(args, "encoder", "encoder.onnx", inputs)
    encoder_mask = np.fromfile(encoded["encoder_mask"]["path"], dtype="<i8").reshape(
        encoded["encoder_mask"]["shape"])
    assert encoder_mask.sum() == (reference["valid_frames"] + 7) // 8
    signatures = {label: [value.name for value in onnx.load(args.artifacts / f"decoder-{label}.onnx",
                                                           load_external_data=False).graph.input]
                  for label in ("init", "step")}
    emitted, offset, ids, states, steps = [], 0, prompt, {}, []
    started = time.perf_counter()
    for index in range(449):
        label = "init" if index == 0 else "step"
        directory = args.output / f"inputs-{index:03}"
        directory.mkdir()
        positions = np.arange(offset, offset + len(ids), dtype=np.int64)[None, :]
        allowed = np.arange(offset + len(ids))[None, :] <= positions.reshape(-1, 1)
        cross = np.where(encoder_mask[:, None, None, :], 0.0, np.finfo(np.float32).min).astype(np.float32)
        cross = np.broadcast_to(cross, (1, 1, len(ids), encoder_mask.shape[1])).copy()
        available = {"encoded": bind("encoded", encoded["encoded"])}
        for name, value in [("tokens", np.asarray([ids], dtype=np.int64)), ("positions", positions),
                            ("self_mask", np.where(allowed, 0.0, np.finfo(np.float32).min)
                             .astype(np.float32)[None, None, :, :]), ("cross_mask", cross)]:
            available[name] = tensor(directory, name, value)
        for name, state in states.items():
            past_name = name.replace("present_", "past_", 1)
            available[past_name] = bind(past_name, state)
        inputs = [available[name] for name in signatures[label]]
        states, report = run_probe(args, f"decode-{index:03}", f"decoder-{label}.onnx", inputs)
        logits = np.fromfile(states.pop("logits")["path"], dtype="<f4")
        assert logits.size == 16384 and np.isfinite(logits).all()
        token = int(logits.argmax())
        steps.append(dict(index=index, token=token, run_ms=report["run_ms"], load_ms=report["load_ms"],
                          executed_provider_events=report["executed_provider_events"]))
        write_json(args.output / "progress.json", dict(tokens=emitted, next_token=token, steps=steps))
        if token == 3:
            break
        assert index < len(reference["tokens"]) and token == reference["tokens"][index], (
            "native decode diverged from independent reference", index, token, reference["tokens"])
        emitted.append(token)
        offset += len(ids)
        ids = [token]
    assert emitted == reference["tokens"], ("native decode did not complete reference", emitted)
    text = Tokenizer.from_file(str(args.artifacts / "tokenizer.json")).decode(emitted, skip_special_tokens=True)
    assert text == reference["text"], text
    result = dict(passed=True, tokens=emitted, text=text, elapsed_s=time.perf_counter() - started,
                  encoder=encoder_report, steps=steps)
    write_json(args.output / "decode-result.json", result)
    print(json.dumps(dict(passed=True, tokens=emitted, text=text, elapsed_s=result["elapsed_s"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("provider", choices=("cpu", "dml"))
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapter", type=int, default=0)
    parser.add_argument("--features", type=Path, help="Qualify a native frontend tensor against the same retained transcript")
    run(parser.parse_args())
