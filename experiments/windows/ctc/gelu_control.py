"""One retained-input Float32 GELU diagnostic; no full-model candidate or gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
SHAPE = [1, 96, 512]
SOURCE_GRAPH_SHA256 = "4e9c26adc08f23712374a98b7a8056fffac4f1646fa6b51ca5ee7b1c0dbb81f1"
CAPTURE_SHA256 = "c7f9b7e73b5417bc756b18c0dae977bb09fc9e5c03a66ad517049af4d60e3d79"
INPUT_SHA256 = "740869f4dd5e02ea8a727ffaf1b2a7bb26019d19c788b975c577494afb1aacc1"
EXPECTED_SHA256 = "c89feb24f384acb7ce35e94af25100fdbc9653db6a68edce77ecd3b476c5ddbb"
SQRT2 = np.float32(1.4142135381698608)
OUTPUTS = ("fused_gelu", "bias_gelu", "scaled", "native_erf", "one_plus_erf", "product", "explicit_gelu")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_graph():
    nodes = [
        helper.make_node("Gelu", ["x"], ["fused_gelu"], domain="com.microsoft"),
        helper.make_node("BiasGelu", ["x", "zero_bias"], ["bias_gelu"], domain="com.microsoft"),
        helper.make_node("Div", ["x", "sqrt2"], ["scaled"]),
        helper.make_node("Erf", ["scaled"], ["native_erf"]),
        helper.make_node("Add", ["native_erf", "one"], ["one_plus_erf"]),
        helper.make_node("Mul", ["x", "one_plus_erf"], ["product"]),
        helper.make_node("Div", ["product", "two"], ["explicit_gelu"]),
    ]
    weights = [numpy_helper.from_array(np.asarray(value, np.float32), name)
               for name, value in [("sqrt2", SQRT2), ("one", 1), ("two", 2),
                                   ("zero_bias", np.zeros(SHAPE[-1], np.float32))]]
    value_info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, "same-source-input-f32-gelu", [value_info("x")],
                              [value_info(name) for name in OUTPUTS], weights)
    model = helper.make_model(graph, ir_version=7,
                              opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
    onnx.checker.check_model(model)
    return model


def prepare(source_graph, captures, output):
    graph_bytes = source_graph.read_bytes()
    capture_bytes = (captures / "capture.json").read_bytes()
    if digest(graph_bytes) != SOURCE_GRAPH_SHA256 or digest(capture_bytes) != CAPTURE_SHA256:
        raise ValueError("original graph or unchanged source capture identity differs")
    source = onnx.load_model_from_string(graph_bytes)
    constants = {node.output[0]: numpy_helper.to_array(node.attribute[0].t)
                 for node in source.graph.node if node.op_type == "Constant"}
    prefix = "/model/encoder/layers.0/ffn/inner_activation/"
    values = [constants[prefix + name].item() for name in
              ("Constant_output_0", "Constant_1_output_0", "Constant_2_output_0")]
    if values != [float(SQRT2), 1.0, 0.5]:
        raise ValueError("retained Float32 GELU constants differ")
    data = [(captures / "conv0_norm.f32").read_bytes(), (captures / "conv0_gelu.f32").read_bytes()]
    if [digest(value) for value in data] != [INPUT_SHA256, EXPECTED_SHA256]:
        raise ValueError("retained same-input GELU capture differs")
    if any(len(value) != 4 * np.prod(SHAPE) or not np.isfinite(np.frombuffer(value, "<f4")).all()
           for value in data):
        raise ValueError("retained capture is not the finite Float32 shape")
    model_bytes = make_graph().SerializeToString()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in zip(("input.f32", "expected.f32"), data):
        (output / name).write_bytes(value)
    (output / "gelu.onnx").write_bytes(model_bytes)
    case = dict(model="gelu.onnx", repeat=2, disable_optimizations=True,
                model_files={"gelu.onnx": digest(model_bytes)},
                inputs=[dict(name="x", dtype="f32", shape=SHAPE, path="input.f32")])
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    provenance = dict(source_graph_sha256=SOURCE_GRAPH_SHA256, capture_sha256=CAPTURE_SHA256,
                      input_sha256=INPUT_SHA256, expected_sha256=EXPECTED_SHA256,
                      graph_sha256=digest(model_bytes), tool_sha256=digest(Path(__file__).read_bytes()),
                      source_library_sha256=json.loads(capture_bytes)["actual_libmlx_sha256"],
                      limit="Diagnostic source capture only; original independent whole-score gate remains 1e-4")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def metrics(actual, expected):
    if actual.dtype != np.float32 or actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("diagnostic tensor dtype, shape or finiteness differs")
    bits = lambda value: value.view(np.int32).astype(np.int64)
    ordered = lambda value: np.where(bits(value) < 0, -0x80000000 - bits(value), bits(value))
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(max_absolute_error=float(error.max(initial=0)),
                mean_absolute_error=float(error.mean()),
                unequal_bits=int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
                maximum_ulp_distance=int(np.abs(ordered(actual) - ordered(expected)).max(initial=0)))


def summarize(x, expected, runs):
    if len(runs) != 2 or any(set(run) != set(OUTPUTS) for run in runs):
        raise ValueError("both complete diagnostic output sets are required")
    reports = []
    for index, run in enumerate(runs):
        for name in OUTPUTS:
            metrics(run[name], expected)
            if index and metrics(run[name], runs[0][name])["unequal_bits"]:
                raise ValueError("diagnostic output did not repeat bit-exactly")
        # Reassemble only the native Erf result using source operation order in Float32.
        reassembled = x * (np.float32(1) + run["native_erf"]) / np.float32(2)
        reports.append(dict(run=index, versus_retained_source={name: metrics(run[name], expected)
                            for name in ("fused_gelu", "bias_gelu", "explicit_gelu")},
                            native_erf_reassembled_vs_source=metrics(reassembled, expected),
                            explicit_vs_reassembled=metrics(run["explicit_gelu"], reassembled),
                            scaled_vs_ieee_division=metrics(run["scaled"], x / SQRT2),
                            bias_vs_fused=metrics(run["bias_gelu"], run["fused_gelu"])))
    return reports


def compare(case, result, provider):
    sys.path.insert(0, str(ROOT / "experiments/windows"))
    from verify_probe import verify
    record = verify(case, result, provider)
    if (digest((case / "gelu.onnx").read_bytes()) != digest(make_graph().SerializeToString())
            or digest((case / "input.f32").read_bytes()) != INPUT_SHA256
            or digest((case / "expected.f32").read_bytes()) != EXPECTED_SHA256):
        raise ValueError("diagnostic graph/input/source-output identity differs")
    if record.get("graph_optimizations_disabled") is not True:
        raise ValueError("this control requires disabled ORT graph optimizations")
    events = json.loads((result / PureWindowsPath(record["profile"]).name).read_text())
    placements = Counter((event.get("args", {}).get("op_name"), event.get("args", {}).get("provider"))
                         for event in events if event.get("args", {}).get("provider"))
    if provider == "dml" and any(ep == "CPUExecutionProvider" for _, ep in placements):
        raise ValueError("a same-input diagnostic branch fell back to CPU")
    runs = [{}, {}]
    for row in record["outputs"]:
        index, name = row["run"], row["name"]
        if (index not in (0, 1) or name not in OUTPUTS or name in runs[index]
                or row["shape"] != SHAPE or row["dtype"] != "f32"):
            raise ValueError("unexpected or duplicate diagnostic output")
        data = (result / row["file"]).read_bytes()
        if digest(data) != row["sha256"]:
            raise ValueError("native diagnostic output hash differs")
        runs[index][name] = np.frombuffer(data, "<f4").reshape(SHAPE)
    x = np.fromfile(case / "input.f32", "<f4").reshape(SHAPE)
    expected = np.fromfile(case / "expected.f32", "<f4").reshape(SHAPE)
    report = dict(provider=provider, reports=summarize(x, expected, runs),
                  executed_operators=[dict(operator=op, provider=ep, events=count)
                                      for (op, ep), count in placements.items()],
                  limit="No source-Erf intermediate exists: a residual explicit-chain difference does not uniquely distinguish source polynomial from compiler arithmetic. No full-model qualification or new numerical gate.")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("prepare")
    build.add_argument("source_graph", type=Path)
    build.add_argument("captures", type=Path)
    build.add_argument("output", type=Path)
    check = modes.add_parser("compare")
    check.add_argument("case", type=Path)
    check.add_argument("result", type=Path)
    check.add_argument("--provider", choices=("cpu", "dml"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.source_graph, args.captures, args.output)
    else:
        compare(args.case, args.result, args.provider)
