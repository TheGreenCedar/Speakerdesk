"""Prepare fixed, stateful graph calls for the shared native Windows ORT probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from export import ROOT, sha256
from qualify import Session, read_pcm, frame_windows


def prepare(artifact):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(artifact / "nemotron.onnx"), options, providers=["CPUExecutionProvider"])
    run = lambda inputs: runtime.run(["chunk", "high", "low"], inputs)
    frontend = ort.InferenceSession(str(artifact / "frontend.onnx"), options, providers=["CPUExecutionProvider"])
    pcm = read_pcm(ROOT / "crates/models/tests/data/two-voice.wav")
    cases = []

    def write_case(directory, model, inputs, outputs):
        directory.mkdir(parents=True, exist_ok=True)
        case = {"model": model, "repeat": 2, "inputs": []}
        for name, array in inputs.items():
            dtype = "i64" if array.dtype == np.int64 else "f32"
            filename = f"{name}.{dtype}"
            array.astype("<i8" if dtype == "i64" else "<f4").tofile(directory / filename)
            case["inputs"].append({"name": name, "dtype": dtype, "shape": list(array.shape), "path": filename})
        (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        expected = {}
        for name, array in outputs.items():
            filename = f"expected-{name}.f32"
            array.astype("<f4").tofile(directory / filename)
            expected[name] = {"shape": list(array.shape), "file": filename, "sha256": sha256(directory / filename)}
        (directory / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
        cases.append(str(directory.relative_to(artifact) / "case.json"))

    def capture(preset):
        def save(step, inputs, outputs):
            directory = artifact / "cases" / f"{preset}-{step}"
            write_case(directory, "../../nemotron.onnx", inputs, outputs)
            windows = frame_windows(session.pcm, session.sample_offset, session.received,
                                    session.frames, inputs["features"].shape[1])
            front_inputs = {"audio_windows": windows, "valid_frames": inputs["valid_frames"]}
            front_outputs = {"features": frontend.run(["features"], front_inputs)[0]}
            write_case(artifact / "cases" / f"frontend-{preset}-{step}", "../../frontend.onnx",
                       front_inputs, front_outputs)
        return save

    for preset in ["offline", "low"]:
        session = Session(run, artifact, preset, capture(preset))
        session.feed(pcm, True)
    manifest = {"model_sha256": sha256(artifact / "nemotron.onnx"), "cases": cases,
                "reference": "Expected tensors are this graph's macOS ORT CPU output; independent acoustic references are checked separately by qualify.py."}
    (artifact / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    prepare(parser.parse_args().artifact)
