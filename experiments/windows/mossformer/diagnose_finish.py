"""Finish-only source/ONNX comparison on captured full-window state; never regenerates references."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from export import ROOT, SOURCE_FILES, SOURCE_REVISION, json_write, load_source, sha256, verify_checkpoint
from qualify import waveform_error


def mapped_float(path, shape, expected_hash=None):
    if (not shape or any(type(n) is not int or n < 1 for n in shape)
            or path.stat().st_size != int(np.prod(shape, dtype=object)) * 4):
        raise ValueError(f"invalid captured Float32 shape: {path}")
    digest = sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"captured tensor identity differs: {path}")
    # Copy-on-write mapping allows Torch's array view without mutating captured files.
    data = np.memmap(path, dtype="<f4", mode="c", shape=tuple(shape))
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite captured Float32 tensor: {path}")
    return data, digest


def bind_inputs(case_directory, reference):
    case = json.loads((case_directory / "case.json").read_text())
    rows = case["inputs"]
    if len(rows) != 4 or {row["name"] for row in rows} != {"audio", "encoded", "residual", "x"}:
        raise ValueError("expected captured finish audio/encoded/residual/x inputs")
    audio_row = next(row for row in rows if row["name"] == "audio")
    shape = audio_row["shape"]
    if len(shape) != 2 or shape[0] != 1 or not 16 <= shape[1] <= 392000:
        raise ValueError("invalid raw full-window audio shape")
    samples, frames = shape[1], (shape[1] - 16) // 8 + 1
    shapes = {"audio": [1, samples], "encoded": [1, 512, frames],
              "residual": [1, 512, frames], "x": [1, frames, 512]}
    inputs, identities = {}, {}
    for row in rows:
        if row["dtype"] != "f32" or row["shape"] != shapes[row["name"]]:
            raise ValueError("captured finish tensor type/shape differs")
        inputs[row["name"]], digest = mapped_float(case_directory / row["path"], row["shape"])
        identities[row["name"]] = {"shape": row["shape"], "sha256": digest}
    matches = [row for row in reference["cases"] if row["samples"] == samples
               and row["input"]["sha256"] == identities["audio"]["sha256"]]
    if len(matches) != 1:
        raise ValueError("captured audio does not identify exactly one independent reference")
    return case, inputs, identities, matches[0]


def attribution(torch_vs_ort, torch_vs_reference, ort_vs_reference):
    # Source finish consumes captured ONNX state; it is not a new independent full forward.
    source_matches = all(row["within_original_bounds"] for row in torch_vs_reference)
    ort_matches = all(row["within_original_bounds"] for row in ort_vs_reference)
    same_finish = all(row["within_original_bounds"] for row in torch_vs_ort)
    if source_matches and not ort_matches:
        return "finish_path_sufficient_for_original_bound_failure"
    if not source_matches and same_finish:
        return "captured_state_or_source_runtime_difference_remains"
    if not same_finish:
        return "finish_path_difference_present_upstream_contribution_not_isolated"
    return "captured_finish_outputs_within_original_bounds_only"


def diagnose(artifact, case_directory, checkpoint, source_parent, references, output,
             threads=4, native_result=None, provider=None):
    pin = verify_checkpoint(checkpoint)
    program_path = artifact / "program.json"
    program = json.loads(program_path.read_text())
    reference_path = references / "references.json"
    reference = json.loads(reference_path.read_text())
    checkpoint_hash = pin["files"]["last_best_checkpoint.pt"]["sha256"]
    if (reference["checkpoint_sha256"] != checkpoint_hash or reference["source_revision"] != SOURCE_REVISION
            or reference["source_sha256"] != SOURCE_FILES or program["checkpoint"] != pin
            or program["source_revision"] != SOURCE_REVISION or program["source_sha256"] != SOURCE_FILES
            or program["stages_sha256"] != sha256(Path(__file__).with_name("stages.py"))):
        raise ValueError("source, wrapper, checkpoint, or independent provenance differs")
    finish = program["graphs"][-1]
    if (finish["name"] != "finish" or finish["model"] != "finish.onnx"
            or finish["inputs"] != ["audio", "encoded", "residual", "x"]
            or finish["outputs"] != ["source0", "source1"]):
        raise ValueError("incompatible finish stage")
    case, inputs, input_identities, original = bind_inputs(case_directory, reference)
    graph_path = (case_directory / case["model"]).resolve()
    if graph_path != (artifact / "finish.onnx").resolve() or sha256(graph_path) != finish["sha256"]:
        raise ValueError("captured case is not the frozen finish graph")
    for name, digest in program["model_files"].items():
        if sha256(artifact / name) != digest:
            raise ValueError("external finish weights differ")
    for name, digest in case.get("model_files", {}).items():
        if sha256(case_directory / name) != digest:
            raise ValueError("captured external weights differ")
    samples = inputs["audio"].shape[1]
    independent = [mapped_float(references / row["path"], [1, samples], row["sha256"])[0]
                   for row in original["outputs"]]
    if len(independent) != 2:
        raise ValueError("expected two independent source channels")
    capture_record = json.loads((case_directory / "expected.json").read_text())
    if set(capture_record) != {"source0", "source1"}:
        raise ValueError("expected both captured CPU finish outputs")
    captured = [mapped_float(case_directory / capture_record[name]["file"], [1, samples],
                            capture_record[name]["sha256"])[0] for name in ["source0", "source1"]]
    independent_case_path = case_directory / "independent.json"
    if independent_case_path.exists():
        copied = json.loads(independent_case_path.read_text())
        if any(copied[f"source{index}"]["sha256"] != original["outputs"][index]["sha256"] for index in range(2)):
            raise ValueError("captured independent outputs belong to a different reference")
    if native_result is not None and provider not in {"cpu", "dml"}:
        raise ValueError("native result requires explicit --provider cpu|dml")
    if native_result is None and provider is not None:
        raise ValueError("--provider applies only to an existing native result")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from stages import Finish
    source, _ = load_source(source_parent)
    torch.set_num_threads(threads)
    model = source.MossFormer().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    finish_module = Finish(model).eval()
    del state, model
    started = time.perf_counter()
    with torch.inference_mode():
        tensor_inputs = [torch.from_numpy(inputs[name]) for name in ["audio", "encoded", "residual", "x"]]
        torch_outputs = [value.detach().cpu().numpy() for value in finish_module(*tensor_inputs)]
    torch_seconds = time.perf_counter() - started
    for index, values in enumerate(torch_outputs):
        waveform_error(values, independent[index])  # Validate shape and finiteness before saving.
        values.astype("<f4", copy=False).tofile(output / f"torch-source{index}.f32")
    del tensor_inputs, finish_module
    if native_result is None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
        started = time.perf_counter()
        ort_outputs = session.run(["source0", "source1"], inputs)
        runtime = {"kind": "Python ONNX Runtime CPU control", "version": ort.__version__,
                   "seconds": time.perf_counter() - started, "cpu_mem_arena": False, "memory_pattern": False}
        runs = [(0, ort_outputs)]
    else:
        sys.path.insert(0, str(ROOT / "experiments/windows"))
        from verify_probe import verify
        native = verify(case_directory, native_result, provider)
        rows = native["outputs"]
        if len(rows) != 4 or {(r["run"], r["name"]) for r in rows} != {
                (run, name) for run in [0, 1] for name in ["source0", "source1"]}:
            raise ValueError("expected both native source channels and repetitions")
        runs = []
        for run in [0, 1]:
            values = []
            for name in ["source0", "source1"]:
                row = next(r for r in rows if r["run"] == run and r["name"] == name)
                if row["dtype"] != "f32" or row["shape"] != [1, samples]:
                    raise ValueError("native finish type or shape differs")
                values.append(mapped_float(native_result / row["file"], row["shape"], row["sha256"])[0])
            runs.append((run, values))
        runtime = {"kind": "verified native Windows ML result", "provider": provider,
                   "result_sha256": sha256(native_result / "result.json"), "runtime_sha256": native["runtime_sha256"]}
    comparisons = []
    source_reference = [waveform_error(got, wanted) for got, wanted in zip(torch_outputs, independent)]
    for run, values in runs:
        source_ort = [waveform_error(actual, source) for source, actual in zip(torch_outputs, values)]
        ort_reference = [waveform_error(actual, wanted) for actual, wanted in zip(values, independent)]
        for index, channel in enumerate(values):
            channel.astype("<f4", copy=False).tofile(output / f"ort-run{run}-source{index}.f32")
        comparisons.append({"run": run, "torch_vs_ort_same_captured_state": source_ort,
            "ort_vs_independent_full_forward": ort_reference,
            "ort_vs_existing_cpu_capture": [waveform_error(actual, wanted) for actual, wanted in zip(values, captured)],
            "attribution": attribution(source_ort, source_reference, ort_reference)})
    # A diagnostic observation never substitutes its outputs for the retained full-forward reference.
    result = {"diagnostic_only": True, "full_forward_qualification": "not performed by a finish-only diagnosis",
        "samples": samples, "original_bounds": {"max_abs": .002, "normalized_rms": .002},
        "torch_finish_vs_independent_full_forward": source_reference,
        "torch_finish_vs_existing_cpu_capture": [waveform_error(actual, wanted) for actual, wanted in zip(torch_outputs, captured)],
        "ort_runs": comparisons, "torch_seconds": torch_seconds, "ort_runtime": runtime,
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy"]},
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_build": torch.__config__.show(),
        "export_tool_versions": program["tool_versions"], "inputs": input_identities,
        "checkpoint_sha256": checkpoint_hash, "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_FILES,
        "program_sha256": sha256(program_path), "finish_graph_sha256": finish["sha256"],
        "external_weights": program["model_files"], "reference_case": original["id"],
        "original_reference_record_sha256": original["original_record_sha256"],
        "references_manifest_sha256": sha256(reference_path), "captured_case_sha256": sha256(case_directory / "case.json"),
        "script_sha256": sha256(Path(__file__)),
        "outputs": {path.name: sha256(path) for path in sorted(output.glob("*.f32"))},
        "limits": "Torch finish consumes captured ONNX state. Residual reference error does not uniquely locate an upstream stage; original/current Torch runtime differences remain possible."}
    json_write(output / "diagnosis.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["artifact", "case_directory", "checkpoint", "source_parent", "references", "output"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--native-result", type=Path, help="reuse a verified native finish call instead of running Python ORT")
    parser.add_argument("--provider", choices=["cpu", "dml"])
    args = vars(parser.parse_args())
    if args["threads"] < 1:
        parser.error("--threads must be positive")
    diagnose(**args)
