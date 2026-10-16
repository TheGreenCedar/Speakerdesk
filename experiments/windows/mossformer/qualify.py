"""Run all dynamic MossFormer stages against sealed independent raw references."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from export import ROOT, SOURCE_FILES, SOURCE_REVISION, json_write, sha256


def waveform_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("invalid waveform shape or values")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference), initial=0))
    relative = float(np.sqrt(np.sum(difference**2) / max(float(np.sum(expected.astype(np.float64)**2)), 1e-20)))
    return {"max_abs_error": maximum, "normalized_rms_error": relative,
            "within_original_bounds": maximum <= 2e-3 and relative <= 2e-3}


def read_float(root, record, samples):
    path = root / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError("independent waveform hash changed")
    data = np.fromfile(path, "<f4")
    if data.shape != (samples,) or not np.isfinite(data).all():
        raise ValueError("invalid independent raw Float32 waveform")
    return data[None]


def write_case(artifact, name, graph, inputs, outputs, weights_hash, independent=None):
    directory = artifact / "cases" / name
    directory.mkdir(parents=True, exist_ok=True)
    case = {"model": "../../" + graph, "repeat": 2, "inputs": [],
            "model_files": {"../../mossformer.weights": weights_hash}}
    for key, data in inputs.items():
        data.astype("<f4").tofile(directory / f"{key}.f32")
        case["inputs"].append({"name": key, "dtype": "f32", "shape": list(data.shape), "path": f"{key}.f32"})
    expected = {}
    for key, data in outputs.items():
        filename = f"expected-{key}.f32"
        data.astype("<f4").tofile(directory / filename)
        expected[key] = {"file": filename, "shape": list(data.shape), "sha256": sha256(directory / filename)}
    json_write(directory / "case.json", case)
    json_write(directory / "expected.json", expected)
    if independent:
        targets = {}
        for index, data in enumerate(independent):
            filename = f"independent-source{index}.f32"
            data.astype("<f4").tofile(directory / filename)
            targets[f"source{index}"] = {"file": filename, "shape": list(data.shape), "sha256": sha256(directory / filename)}
        json_write(directory / "independent.json", targets)
    return str((directory / "case.json").relative_to(artifact))


def execute(program, sessions, audio, capture=None):
    state = {"audio": audio}
    phases = []
    for graph in program["graphs"]:
        inputs = {name: state[name] for name in graph["inputs"]}
        started = time.perf_counter()
        values = sessions[graph["name"]].run(graph["outputs"], inputs)
        outputs = dict(zip(graph["outputs"], values))
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError(("nonfinite MossFormer output", graph["name"]))
        phases.append({"name": graph["name"], "seconds": time.perf_counter() - started})
        if capture:
            capture(graph, inputs, outputs)
        if "x_out" in outputs:
            state["x"] = outputs["x_out"]
        else:
            state.update(outputs)
        print(json.dumps(phases[-1]), flush=True)
    return [state["source0"], state["source1"]], phases


def qualify(artifact, references, output, case_id=None):
    import onnxruntime as ort
    program = json.loads((artifact / "program.json").read_text())
    reference = json.loads((references / "references.json").read_text())
    pin = json.loads((ROOT / "crates/host/src/separation-artifact.json").read_text())
    if (reference["source_revision"] != SOURCE_REVISION or reference["source_sha256"] != SOURCE_FILES
            or reference["checkpoint_sha256"] != pin["files"]["last_best_checkpoint.pt"]["sha256"]):
        raise ValueError("independent reference identity changed")
    names = [graph["name"] for graph in program["graphs"]]
    expected_names = ["encoder"] + [name for index in range(24) for name in [f"flash-{index:02d}", f"fsmn-{index:02d}"]] + ["finish"]
    if names != expected_names:
        raise ValueError("invalid staged MossFormer program")
    for name, digest in program["model_files"].items():
        if sha256(artifact / name) != digest:
            raise ValueError("external model weights changed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    # Fifty yielding stages must release their temporary buffers between calls.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    sessions = {}
    for graph in program["graphs"]:
        if sha256(artifact / graph["model"]) != graph["sha256"]:
            raise ValueError("exported graph changed")
        sessions[graph["name"]] = ort.InferenceSession(str(artifact / graph["model"]), options,
                                                      providers=["CPUExecutionProvider"])
    selected_cases = [case for case in reference["cases"] if case_id is None or case["id"] == case_id]
    if not selected_cases:
        raise ValueError("unknown independent reference case")
    selected = min(selected_cases, key=lambda case: case["samples"])["id"]
    reports, cases = [], []
    for case in selected_cases:
        if not 16 <= case["samples"] <= 392000:
            raise ValueError("invalid separation window")
        audio = read_float(references, case["input"], case["samples"])
        expected = [read_float(references, record, case["samples"]) for record in case["outputs"]]

        def capture(graph, inputs, outputs):
            if case["id"] != selected or graph["name"] not in {"encoder", "flash-00", "fsmn-00", "finish"}:
                return
            cases.append(write_case(artifact, case["id"] + "-" + graph["name"], graph["model"], inputs,
                                    outputs, program["model_files"]["mossformer.weights"],
                                    expected if graph["name"] == "finish" else None))

        actual, phases = execute(program, sessions, audio, capture)
        channels = [waveform_error(got, wanted) for got, wanted in zip(actual, expected)]
        report = {"id": case["id"], "samples": case["samples"], "channels": channels, "phases": phases}
        reports.append(report)
        print(json.dumps({"id": case["id"], "channels": channels}), flush=True)
    result = {"runtime": ort.__version__, "provider": "CPUExecutionProvider", "cases": reports,
              "cpu_mem_arena": False, "memory_pattern": False,
              "references_manifest_sha256": sha256(references / "references.json"),
              "program_sha256": sha256(artifact / "program.json")}
    json_write(output, result)
    json_write(artifact / "cases.json", {"cases": cases})
    if not all(channel["within_original_bounds"] for report in reports for channel in report["channels"]):
        raise ValueError("MossFormer waveforms exceed original independent bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", help="Run one retained reference without changing it")
    args = parser.parse_args()
    qualify(args.artifact, args.references, args.output, args.case)
