"""Compile pinned MLX Half GELU over its complete binary16 input domain."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from export import Graph, ROOT, sha256, value
from implicit_control import EXTRA_CAPTURES
from materialization_control import pinned_runtime

sys.path.insert(0, str(ROOT / "experiments/windows"))
from verify_probe import verify

ACTIVATIONS_SHA256 = "153660ac19d4fe93d6ca15f67ad8527ed03b0e860e36192f3815e1d3daf73ed1"


def half_index(g, x):
    """Exact binary16 index for a Half-valued F32 tensor.

    The repaired logarithm selects an exact power-of-two scale. Every finite
    Half significand is then an exact F32 integer. Signed zero uses reciprocal
    sign; infinities have their normal indices and NaNs use canonical 0x7e00.
    No NaN payload or sign identity is promised.
    """
    scalar = lambda number: g.const(np.asarray(number, np.float32))
    zero = scalar(0)
    absolute = g.op("Abs", x)
    bounded = g.op("Min", g.op("Max", absolute, scalar(2**-14)), scalar(65504))
    bounded = g.op("Where", g.op("IsNaN", x), scalar(1), bounded)
    exponent = g.op("Clip", g.op("Floor", g.op("Div", g.op("Log", bounded), scalar(np.log(2)))),
                    scalar(-14), scalar(15))
    bin_index = g.op("Cast", g.op("Add", exponent, scalar(14)), to=TensorProto.INT64)
    powers = np.exp2(np.arange(-14, 16, dtype=np.float64)).astype(np.float32)
    lower = g.op("Gather", g.const(powers), bin_index, axis=0)
    scale = g.op("Gather", g.const(np.exp2(np.arange(24, -6, -1)).astype(np.float32)), bin_index, axis=0)
    below = g.op("Less", bounded, lower)
    above = g.op("GreaterOrEqual", bounded, g.op("Mul", lower, scalar(2)))
    exponent = g.op("Add", exponent, g.op("Where", below, scalar(-1), g.op("Where", above, scalar(1), zero)))
    scale = g.op("Where", below, g.op("Mul", scale, scalar(2)),
                 g.op("Where", above, g.op("Mul", scale, scalar(.5)), scale))
    magnitude = g.op("Add", g.op("Mul", g.op("Add", exponent, scalar(14)), scalar(1024)),
                     g.op("Mul", absolute, scale))
    negative = g.op("Or", g.op("Less", x, zero),
                    g.op("And", g.op("Equal", x, zero), g.op("Less", g.op("Reciprocal", x), zero)))
    sign = g.op("Where", negative, scalar(32768), zero)
    index = g.op("Add", sign, magnitude)
    index = g.op("Where", g.op("Greater", absolute, scalar(65504)), g.op("Add", sign, scalar(0x7c00)), index)
    return g.op("Where", g.op("IsNaN", x), scalar(0x7e00), index)


def lookup(g, x, table):
    return g.op("Gather", table, g.op("Cast", half_index(g, x), to=TensorProto.INT64), axis=0)


def classification(array):
    result = np.zeros(array.shape, np.float32)
    result[np.isposinf(array)] = 1
    result[np.isneginf(array)] = 2
    result[np.isnan(array)] = 3
    result[(array == 0) & ~np.signbit(array)] = 4
    result[(array == 0) & np.signbit(array)] = 5
    return result


def control_model(table, specials):
    g = Graph({})
    x = g.op("Concat", "x", g.const(specials), axis=0)
    index = half_index(g, x)
    result = g.op("Gather", g.const(table, "compiled_half_gelu"), g.op("Cast", index, to=TensorProto.INT64), axis=0)
    scalar = lambda number: g.const(np.asarray(number, np.float32))
    zero = scalar(0)
    is_nan = g.op("IsNaN", result)
    is_infinite = g.op("Greater", g.op("Abs", result), scalar(np.finfo(np.float32).max))
    classes = g.op("Where", is_nan, scalar(3),
        g.op("Where", is_infinite, g.op("Where", g.op("Less", result, zero), scalar(2), scalar(1)),
            g.op("Where", g.op("Equal", result, zero),
                g.op("Where", g.op("Less", g.op("Reciprocal", result), zero), scalar(5), scalar(4)), zero)))
    finite = g.op("Where", g.op("Or", is_nan, is_infinite), zero, result)
    for name, tensor in [("index", index), ("finite", finite), ("classification", classes)]:
        g.op("Identity", tensor, name=name)
    return g.model("Whisper-complete-Half-GELU-compiler-control", [value("x", ["inputs"])],
                   [value(name, ["outputs"]) for name in ["index", "finite", "classification"]])


def expected_indices(bits):
    indices = bits.astype(np.float32)
    indices[(bits & 0x7fff) > 0x7c00] = 0x7e00
    return indices


def differences(actual, wanted):
    return int(np.count_nonzero(actual.view(np.uint32) != wanted.view(np.uint32)))


def prepare(extra, output):
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.gpu)
    runtime = pinned_runtime()
    source = Path(inspect.getsourcefile(nn.gelu))
    if sha256(source) != ACTIVATIONS_SHA256:
        raise ValueError("compiled GELU requires the exact MLX 0.32.2 activation source")
    for name in ["pre", "post"]:
        if sha256(extra / (name + ".npy")) != EXTRA_CAPTURES[name]:
            raise ValueError("retained Conv2 source capture differs")
    bits = np.arange(65536, dtype=np.uint16)
    half = bits.view(np.float16)
    with np.errstate(all="ignore"):
        domain = half.astype(np.float32)
    table = np.asarray(nn.gelu(mx.array(half)).astype(mx.float32))
    repeated = np.asarray(nn.gelu(mx.array(half)).astype(mx.float32))
    if (differences(table[np.isfinite(table)], repeated[np.isfinite(table)])
            or np.any(classification(table) != classification(repeated))):
        raise ValueError("pinned compiled GELU is not deterministic")
    canonical = bits.copy()
    canonical[(bits & 0x7fff) > 0x7c00] = 0x7e00
    canonical_table = table[canonical]
    if (np.any(classification(table) != classification(canonical_table))
            or differences(table[np.isfinite(table)], canonical_table[np.isfinite(table)])):
        raise ValueError("canonical NaN lookup changes source value classification")
    pre = np.load(extra / "pre.npy", allow_pickle=False).reshape(-1)
    post = np.load(extra / "post.npy", allow_pickle=False).reshape(-1)
    if np.any(pre != pre.astype(np.float16).astype(np.float32)):
        raise ValueError("retained GELU input is not Half-valued")
    pre_bits = pre.astype(np.float16).view(np.uint16)
    if differences(table[pre_bits], post):
        raise ValueError("complete-domain compiled table differs from retained Conv2 GELU")
    finite_input = np.isfinite(domain)
    specials = domain[~finite_input]
    model = control_model(table, specials)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "gelu-control.onnx")
    table.astype("<f4").tofile(output / "gelu.f32")
    records = []
    for name, x, input_bits in [("complete-domain", domain[finite_input], bits[finite_input]),
                                 ("captured-conv2", pre, pre_bits)]:
        directory = output / name
        directory.mkdir()
        all_bits = np.concatenate([input_bits, bits[~finite_input]])
        wanted = table[all_bits].copy()
        expected = {"index": expected_indices(all_bits), "classification": classification(wanted),
                    "finite": np.where(np.isfinite(wanted), wanted, np.float32(0))}
        actual = dict(zip(["index", "finite", "classification"], session.run(None, {"x": x})))
        local = {key: differences(actual[key], array) for key, array in expected.items()}
        if any(local.values()) or not all(np.isfinite(array).all() for array in actual.values()):
            raise ValueError(("local complete-domain lookup failed", name, local))
        x.astype("<f4").tofile(directory / "input.f32")
        declarations = {}
        for key, array in expected.items():
            filename = key + ".expected.f32"
            array.astype("<f4").tofile(directory / filename)
            declarations[key] = {"shape": list(array.shape), "file": filename, "sha256": sha256(directory / filename)}
        (directory / "expected.json").write_text(json.dumps(declarations, indent=2) + "\n")
        case = {"model": "../gelu-control.onnx", "repeat": 2, "disable_optimizations": True,
                "model_files": {"../gelu-control.onnx": sha256(output / "gelu-control.onnx"),
                                "expected.json": sha256(directory / "expected.json")},
                "inputs": [{"name": "x", "dtype": "f32", "shape": list(x.shape), "path": "input.f32"}]}
        (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        records.append({"case": name + "/case.json", "case_sha256": sha256(directory / "case.json"),
                        "source_patterns": len(all_bits), "local_bit_differences": local})
    provenance = {"diagnostic_only": True, **runtime, "graph_sha256": sha256(output / "gelu-control.onnx"),
        "script_sha256": sha256(Path(__file__)), "table_sha256": sha256(output / "gelu.f32"),
        "table_bytes": table.nbytes, "table_elements": table.size, "table_dtype": "float32",
        "source": {"url": "https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/activations.py",
                   "sha256": sha256(source)}, "conv2_capture_pins": {name: EXTRA_CAPTURES[name] for name in ["pre", "post"]},
        "finite_inputs": int(finite_input.sum()), "finite_inputs_with_nonfinite_results": int((finite_input & ~np.isfinite(table)).sum()),
        "output_classes": {str(code): int(np.count_nonzero(classification(table) == code)) for code in range(6)},
        "signed_zero_bits": table[[0, 0x8000]].view(np.uint32).tolist(),
        "contract": "Complete compiled binary16 GELU, including source intermediate overflow. Every finite Half input is indexed exactly, including signed zero/subnormals. Infinities preserve source results; NaN inputs use canonical 0x7e00 (payload/sign unspecified).",
        "native_io": "All inputs/outputs finite. The graph appends all 2048 nonfinite Half input patterns internally; outputs retain finite values, classify +/-Inf/NaN and +/-zero separately. Sanitization is diagnostic only; the compiled table is unmodified.",
        "class_codes": {"0": "finite nonzero", "1": "+Inf", "2": "-Inf", "3": "NaN", "4": "+zero", "5": "-zero"},
        "local_ort": ort.__version__, "cases": records, "whole_model_changed": False, "native_qualified": False}
    (output / "gelu-table.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def compare(case, result, provider):
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("complete-domain GELU lookup used CPU model work")
    expected = json.loads((case / "expected.json").read_text())
    if len(record["outputs"]) != 2 * len(expected) or {(row["run"], row["name"]) for row in record["outputs"]} != {
            (run, name) for run in [0, 1] for name in expected}:
        raise ValueError("lookup output inventory differs")
    first, reports, passed = {}, [], True
    for row in record["outputs"]:
        spec = expected[row["name"]]
        if sha256(case / spec["file"]) != spec["sha256"]:
            raise ValueError("compiled source expectation identity differs")
        if row["dtype"] != "f32" or row["shape"] != spec["shape"] or sha256(result / row["file"]) != row["sha256"]:
            raise ValueError("native lookup output identity differs")
        actual = np.fromfile(result / row["file"], "<f4")
        wanted = np.fromfile(case / spec["file"], "<f4")
        if actual.size != wanted.size:
            raise ValueError("lookup output length differs")
        first.setdefault(row["name"], actual)
        report = {"run": row["run"], "name": row["name"], "bit_differences": differences(actual, wanted),
                  "repeat_bit_differences": differences(actual, first[row["name"]])}
        reports.append(report)
        passed &= report["bit_differences"] == 0 and report["repeat_bit_differences"] == 0
    print(json.dumps({"diagnostic_only": True, "compiled_half_gelu_exact": bool(passed),
                      "all_model_work_accelerated": provider == "dml", "reports": reports}, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["extra", "output"]:
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
