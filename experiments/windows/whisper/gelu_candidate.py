"""One causal control: compiled Half GELU boundaries in the frozen precise graph."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort

from export import sha256
from gelu_boundary import HALF_RECIPROCAL, operator_model, preserve_gelu_boundaries

PRECISE_SHA256 = "41027f0ee2ca1b6c63476dbc7f484b1506382170fa4a2a37cc3bed57efd0dc7c"


def prepare(original, output):
    if sha256(original / "whisper.onnx") != PRECISE_SHA256:
        raise ValueError("expected the frozen precise-softmax Whisper graph")
    source = onnx.load(original / "whisper.onnx")
    model, changed = preserve_gelu_boundaries(source)
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "whisper.onnx")
    shutil.copyfile(original / "languages.json", output / "languages.json")
    cases = []
    for case in sorted((original / "cases").glob("neural-*")):
        target = output / "cases" / case.name
        target.mkdir(parents=True)
        for name in ["input.f32", "expected.f32", "expected.json"]:
            shutil.copyfile(case / name, target / name)
        metadata = json.loads((case / "case.json").read_text())
        metadata["disable_optimizations"] = True
        metadata["model_files"] = {"../../whisper.onnx": sha256(output / "whisper.onnx")}
        (target / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(str(target.relative_to(output) / "case.json"))
    # Every original constant remains byte-for-byte; new scalar constants belong only to GELU.
    old_constants = {value.name: value.SerializeToString() for value in source.graph.initializer}
    new_constants = {value.name: value.SerializeToString() for value in model.graph.initializer}
    if any(new_constants.get(name) != value for name, value in old_constants.items()):
        raise ValueError("a source constant changed")
    provenance = {"source_graph_sha256": PRECISE_SHA256,
                  "model_sha256": sha256(output / "whisper.onnx"),
                  "script_sha256": sha256(Path(__file__)),
                  "transform_sha256": sha256(Path(__file__).with_name("gelu_boundary.py")),
                  "changed_gelu_inputs": changed, "half_reciprocal": HALF_RECIPROCAL,
                  "rule": "GELU only: Half reciprocal multiply and explicit Half rounding after each primitive; Half result",
                  "source_constants_unchanged": True, "disable_optimizations": True,
                  "cases": cases, "qualified": False}
    (output / "gelu.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def check_capture(input, capture, output):
    x, wanted = np.load(input), np.load(capture)
    if x.shape != wanted.shape or not np.isfinite(x).all() or not np.isfinite(wanted).all():
        raise ValueError("invalid retained operator input/output")
    original = operator_model(list(x.shape))
    corrected, _ = preserve_gelu_boundaries(original, expected_count=1)
    reports = []
    for name, model in [("frozen_arithmetic", original), ("compiled_half_boundaries", corrected)]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        actual, = session.run(None, {"x": x.astype(np.float32)})
        reports.append({"name": name, "max_error": float(np.max(np.abs(actual - wanted))),
                        "nonzero": int(np.count_nonzero(actual != wanted))})
    receipt = {"input_sha256": sha256(input), "capture_sha256": sha256(capture),
               "runtime": ort.__version__, "provider": "CPUExecutionProvider",
               "diagnostic_only": True, "reports": reports}
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))
    if reports[-1]["nonzero"] != 0:
        raise ValueError("corrected operator did not reproduce the retained source capture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["original", "output"]:
        build.add_argument(name, type=Path)
    check = commands.add_parser("check-capture")
    for name in ["input", "capture", "output"]:
        check.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (prepare if command == "prepare" else check_capture)(**args)
