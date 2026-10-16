"""Rounding-only controls; no model graph or independent expectation is changed."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from export import Graph, ROOT, sha256, value

sys.path.insert(0, str(ROOT / "experiments/windows"))
from verify_probe import verify

RUNTIME_SHA256 = "0c9db159aedd2951bb77610c8b028c4ef79bc64ccc4c73b07a4452a60612ac03"
METALLIB_SHA256 = "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2"
CONV_CAPTURE_SHA256 = "b98f3d64abc45644cff8136ea509a8883adf7c355f25a3a85e17b0886de58332"
SOURCES = {
    "ort_cast": {
        "url": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorCast.cpp",
        "sha256": "4420d01c94d964b165aad9e8bdb7b2d8712808396b006e52b2ab4b47a134b77a",
        "finding": "ONNX Cast directly selects DML_OPERATOR_CAST; the descriptor has no rounding option."},
    "ort_round": {
        "url": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorElementWise.cpp",
        "sha256": "1c3ddf600757438bccbc76c8d82c186663e1edb343244515b1d41eaa7bf30d71",
        "finding": "ONNX Round selects DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN."},
    "dml_round": "https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_element_wise_round_operator_desc",
    "conversion_rules": "https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#3.2.2%20Floating%20Point%20Conversion",
}


def half_grid(g, x):
    """F32 -> nearest-even representable Half value, returned as F32.

    The log estimates the bin only. Exact bounds repair a neighboring-bin
    estimate, and Gather supplies exact binary powers; no approximate Pow
    result is used as a grid. Float32 subnormal inputs round to signed zero.
    Finite overflow rounds to signed infinity at 65520, as in IEEE binary16.
    NaNs remain NaN; payload identity is not part of this diagnostic contract.
    """
    scalar = lambda number: g.const(np.asarray(number, np.float32))
    absolute = g.op("Abs", x)
    bounded = g.op("Min", g.op("Max", absolute, scalar(2**-14)), scalar(65504))
    bounded = g.op("Where", g.op("IsNaN", absolute), scalar(1), bounded)
    estimate = g.op("Floor", g.op("Div", g.op("Log", bounded), scalar(np.log(2))))
    estimate = g.op("Clip", estimate, scalar(-14), scalar(15))
    indices = g.op("Cast", g.op("Add", estimate, scalar(14)), to=TensorProto.INT64)
    exponents = np.arange(-14, 16, dtype=np.int32)
    steps = np.exp2(exponents - 10).astype(np.float32)
    scales = np.exp2(10 - exponents).astype(np.float32)
    step = g.op("Gather", g.const(steps), indices, axis=0)
    scale = g.op("Gather", g.const(scales), indices, axis=0)
    below = g.op("Less", bounded, g.op("Mul", step, scalar(1024)))
    above = g.op("GreaterOrEqual", bounded, g.op("Mul", step, scalar(2048)))
    step = g.op("Where", below, g.op("Mul", step, scalar(.5)),
                g.op("Where", above, g.op("Mul", step, scalar(2)), step))
    scale = g.op("Where", below, g.op("Mul", scale, scalar(2)),
                 g.op("Where", above, g.op("Mul", scale, scalar(.5)), scale))
    rounded = g.op("Mul", g.op("Round", g.op("Mul", x, scale)), step)
    infinity = g.op("Where", g.op("Less", x, scalar(0)), scalar(-np.inf), scalar(np.inf))
    return g.op("Where", g.op("GreaterOrEqual", absolute, scalar(65520)), infinity, rounded)


def operator_model():
    g = Graph({})
    half = g.op("Cast", "x", to=TensorProto.FLOAT16)
    g.op("Cast", half, name="native", to=TensorProto.FLOAT)
    grid = half_grid(g, "x")
    g.op("Identity", grid, name="grid")
    g.op("Cast", g.op("Cast", grid, to=TensorProto.FLOAT16), name="corrected", to=TensorProto.FLOAT)
    return g.model("Whisper-Half-rounding-only-control", [value("x", ["elements"])],
                   [value(name, ["elements"]) for name in ["native", "grid", "corrected"]])


def hostile_values():
    positive = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float32)
    midpoints = (positive[:-1] + positive[1:]) * np.float32(.5)
    neighbors = np.concatenate([np.nextafter(midpoints, -np.inf, dtype=np.float32),
                                midpoints, np.nextafter(midpoints, np.inf, dtype=np.float32)])
    powers = np.exp2(np.arange(-149, 128, dtype=np.float64)).astype(np.float32)
    boundaries = np.concatenate([np.nextafter(powers, -np.inf, dtype=np.float32), powers,
                                 np.nextafter(powers, np.inf, dtype=np.float32)])
    overflow = np.asarray([65504, 65520, 65536], np.float32)
    overflow = np.concatenate([np.nextafter(overflow, -np.inf, dtype=np.float32), overflow,
                              np.nextafter(overflow, np.inf, dtype=np.float32)])
    special = np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                          0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                          0x7fc00000, 0x7fa00001, 0xffc12345], np.uint32).view(np.float32)
    random = np.random.default_rng(0x48414C46).integers(0, 2**32, 131072, dtype=np.uint32).view(np.float32)
    return np.concatenate([positive, -positive, neighbors, -neighbors, boundaries, -boundaries,
                           overflow, -overflow, special, random])


def expectations(x):
    with np.errstate(all="ignore"):
        nearest_half = x.astype(np.float16)
        nearest = nearest_half.astype(np.float32)
        truncated = nearest_half.copy()
        too_far = np.isfinite(x) & (np.abs(nearest) > np.abs(x))
        truncated[too_far] = np.nextafter(nearest_half[too_far], np.float16(0), dtype=np.float16)
    return nearest, truncated.astype(np.float32)


def native_values(x):
    # The native probe intentionally requires finite inputs and outputs.
    # At 65520, nearest-even binary16 overflows; keep that class local-only.
    with np.errstate(all="ignore"):
        return x[np.isfinite(x) & (np.abs(x) < np.float32(65520))]


def differing(actual, expected):
    # Preserve infinity and signed-zero semantics; NaN payloads may canonicalize.
    return (actual.view(np.uint32) != expected.view(np.uint32)) & ~(np.isnan(actual) & np.isnan(expected))


def report(actual, nearest, truncated):
    finite = np.isfinite(actual) & np.isfinite(nearest)
    return {"rne_differences": int(np.count_nonzero(differing(actual, nearest))),
            "rtz_differences": int(np.count_nonzero(differing(actual, truncated))),
            "finite_max_rne_error": float(np.max(np.abs(actual[finite] - nearest[finite]), initial=0))}


def prepare(capture, output):
    receipt = json.loads((capture / "receipt.json").read_text())
    if (receipt["runtime_sha256"] != RUNTIME_SHA256 or
            receipt["colocated_metallib_sha256"] != METALLIB_SHA256 or
            receipt["independent_max_error"] != 0 or
            any(row["nonzero"] for row in receipt["comparisons"])):
        raise ValueError("the source capture lacks the production-runtime baseline")
    path = capture / "conv1-pre.npy"
    if sha256(path) != CONV_CAPTURE_SHA256:
        raise ValueError("the source convolution capture differs")
    convolution = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
    if convolution.size != 1_152_000 or not np.isfinite(convolution).all():
        raise ValueError("invalid captured Whisper convolution tensor")
    hostile = hostile_values()
    inputs = {"hostile-finite": native_values(hostile), "captured-conv1": convolution,
              "gelu-first-product": convolution * np.float32(.70703125)}
    model = operator_model()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    # Preserve the complete IEEE matrix as a local gate, without emitting an
    # inadmissible native case containing nonfinite inputs or outcomes.
    nearest, truncated = expectations(hostile)
    local_only = {name: report(actual, nearest, truncated) for name, actual in zip(
        ["native", "grid", "corrected"], session.run(None, {"x": hostile}))}
    if any(row["rne_differences"] for row in local_only.values()):
        raise ValueError(("local full IEEE matrix failed", local_only))
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "rounding.onnx")
    cases = []
    for name, x in inputs.items():
        nearest, truncated = expectations(x)
        if not np.isfinite(x).all() or not np.isfinite(nearest).all():
            raise ValueError("native rounding case contains a nonfinite input or expected output")
        local = dict(zip(["native", "grid", "corrected"], session.run(None, {"x": x})))
        reports = {key: report(actual, nearest, truncated) for key, actual in local.items()}
        if any(row["rne_differences"] for row in reports.values()):
            raise ValueError(("local operator does not implement exact nearest-even", name, reports))
        directory = output / name
        directory.mkdir()
        for filename, array in [("input.f32", x), ("nearest.f32", nearest), ("toward-zero.f32", truncated)]:
            array.astype("<f4").tofile(directory / filename)
        metadata = {"model": "../rounding.onnx", "repeat": 2, "disable_optimizations": True,
                    "model_files": {"../rounding.onnx": sha256(output / "rounding.onnx")},
                    "inputs": [{"name": "x", "dtype": "f32", "shape": [len(x)], "path": "input.f32"}]}
        (directory / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (directory / "expected.json").write_text(json.dumps({"shape": [len(x)],
            "nearest": {"file": "nearest.f32", "sha256": sha256(directory / "nearest.f32")},
            "toward_zero": {"file": "toward-zero.f32", "sha256": sha256(directory / "toward-zero.f32")},
            "cpu_control": reports}, indent=2) + "\n")
        cases.append(str(directory.relative_to(output) / "case.json"))
    provenance = {"graph_sha256": sha256(output / "rounding.onnx"), "script_sha256": sha256(Path(__file__)),
                  "production_capture_receipt_sha256": sha256(capture / "receipt.json"),
                  "convolution_capture_sha256": CONV_CAPTURE_SHA256, "sources": SOURCES,
                  "local_cpu": ort.__version__, "cases": cases,
                  "local_only_full_ieee_matrix": {"elements": len(hostile), "reports": local_only,
                      "scope": "NaNs, infinities and finite overflow are local CPU evidence only; no native case includes them."},
                  "native_hostile_elements": len(inputs["hostile-finite"]),
                  "native_input_contract": "Finite Float32 values with abs(x)<65520; all expected Half outcomes are finite. Source cases are within [-16,16].",
                  "contract": "All finite F32 inputs round to nearest-even Half (including subnormal and signed-zero values); finite overflow at 65520 becomes signed infinity. Infinities and NaN classification are preserved; NaN payload is unspecified.",
                  "whole_model_changed": False, "native_qualified": False}
    (output / "rounding.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def compare(case, result, provider):
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("rounding control used CPU model work; grid behavior cannot be attributed to DirectML")
    metadata = json.loads((case / "expected.json").read_text())
    expected = {}
    for name in ["nearest", "toward_zero"]:
        path = case / metadata[name]["file"]
        if sha256(path) != metadata[name]["sha256"]:
            raise ValueError("rounding expectation identity differs")
        expected[name] = np.fromfile(path, "<f4")
    rows = record["outputs"]
    if len(rows) != 6 or {(row["run"], row["name"]) for row in rows} != {
            (run, name) for run in [0, 1] for name in ["native", "grid", "corrected"]}:
        raise ValueError("rounding output inventory differs")
    reports, first, passed = [], {}, True
    for row in rows:
        path = result / row["file"]
        if row["dtype"] != "f32" or row["shape"] != metadata["shape"] or sha256(path) != row["sha256"]:
            raise ValueError("rounding output identity differs")
        actual = np.fromfile(path, "<f4")
        if actual.shape != expected["nearest"].shape:
            raise ValueError("rounding tensor length differs")
        first.setdefault(row["name"], actual)
        report_row = {"run": row["run"], "name": row["name"],
                      **report(actual, expected["nearest"], expected["toward_zero"]),
                      "repeat_differences": int(np.count_nonzero(differing(actual, first[row["name"]])))}
        reports.append(report_row)
        passed &= report_row["repeat_differences"] == 0
        if row["name"] in ["grid", "corrected"]:
            passed &= report_row["rne_differences"] == 0
    response = {"diagnostic_only": True, "grid_exact_nearest_even": bool(passed), "reports": reports}
    print(json.dumps(response, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["capture", "output"]:
        build.add_argument(name, type=Path)
    comparison = commands.add_parser("compare")
    for name in ["case", "result"]:
        comparison.add_argument(name, type=Path)
    comparison.add_argument("--provider", choices=["cpu", "dml"], required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    else:
        raise SystemExit(0 if compare(**args) else 1)
