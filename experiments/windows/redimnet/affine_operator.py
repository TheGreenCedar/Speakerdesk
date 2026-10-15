"""Isolate the captured ReDimNet GroupNorm affine from whole-graph fusion.

Prepare with: SOURCE_GRAPH WEIGHTS CAPTURE_CASE CAPTURE_RESULT OUTPUT_CASE.
Compare with: CASE CPU_RESULT DML_RESULT. No model execution occurs here.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from compare_taps import difference, read_outputs, verify
from precision_candidate import SOURCE_SHA256
from prepare_taps import sha256


WEIGHTS_SHA256 = "37693ccbbf2d80a83647f8357e1645effa19357397923a642208f31778ecf242"
RAW_SHAPE = [1, 64, 42912]
RESTORED_SHAPE = [1, 4608, 596]
OUTPUTS = ["affine_f32", "affine_half_as_f32"]


def build_model(gamma, beta):
    if any(value.dtype != np.dtype("float16") or value.shape != (4608,)
           or not np.isfinite(value).all() for value in (gamma, beta)):
        raise ValueError("expected the finite original FP16 affine vectors")
    parameters = [numpy_helper.from_array(value, name) for name, value in [
        ("shape", np.array(RESTORED_SHAPE, np.int64)),
        ("gamma", gamma.astype(np.float32).reshape(4608, 1)),
        ("beta", beta.astype(np.float32).reshape(4608, 1))]]
    nodes = [helper.make_node("Reshape", ["raw_norm", "shape"], ["restored"]),
             helper.make_node("Mul", ["restored", "gamma"], ["scaled"]),
             helper.make_node("Add", ["scaled", "beta"], ["affine_f32"]),
             helper.make_node("Cast", ["affine_f32"], ["affine_half"], to=TensorProto.FLOAT16),
             helper.make_node("Cast", ["affine_half"], ["affine_half_as_f32"], to=TensorProto.FLOAT)]
    model = helper.make_model(helper.make_graph(nodes, "isolated_stage0_affine", [
        helper.make_tensor_value_info("raw_norm", TensorProto.FLOAT, RAW_SHAPE)], [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, RESTORED_SHAPE) for name in OUTPUTS],
        parameters), opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def captured_output(record, directory, name, shape):
    rows = [row for row in record["outputs"] if row["name"] == name]
    value = read_outputs(dict(outputs=rows), directory, [name], 1)[0, name]
    if list(value.shape) != shape:
        raise ValueError(("captured tensor shape differs", name))
    return value, Path(directory) / rows[0]["file"]


def prepare(source, weights, capture_case, capture_result, output):
    source, weights, capture_case, capture_result, output = map(
        Path, (source, weights, capture_case, capture_result, output))
    if output.exists():
        raise ValueError("output case must be new")
    if sha256(source) != SOURCE_SHA256 or sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("source graph or external parameters differ from original pins")
    provenance = json.loads((capture_case / "taps.json").read_text())
    if (provenance["source_graph_sha256"] != SOURCE_SHA256 or
            provenance["tap_set"] != "stage0_norm" or
            provenance["weights_sha256"] != WEIGHTS_SHA256):
        raise ValueError("capture is not the original stage-0 normalization diagnostic")
    record = verify(capture_case, capture_result, "dml")
    if record["model_sha256"] != provenance["graph_sha256"]:
        raise ValueError("capture metadata belongs to another tapped graph")
    _, raw_path = captured_output(record, capture_result, "tap_stage0_unit_norm", RAW_SHAPE)
    _, post_path = captured_output(record, capture_result, "tap_stage0", RESTORED_SHAPE)
    graph = onnx.load(source, load_external_data=False)
    tensors = {tensor.name: tensor for tensor in graph.graph.initializer}
    parameters, slices = [], []
    with weights.open("rb") as stream:
        for suffix, offset in (("weight", 1973184), ("bias", 1982400)):
            name = "graph.backbone.graph.stage0.10." + suffix
            tensor = tensors[name]
            external = {item.key: item.value for item in tensor.external_data}
            if (tensor.data_type != TensorProto.FLOAT16 or list(tensor.dims) != [4608] or
                    external != dict(location="redimnet.weights", offset=str(offset), length="9216")):
                raise ValueError(("affine external tensor identity differs", name))
            stream.seek(offset)
            data = stream.read(9216)
            if len(data) != 9216:
                raise ValueError("truncated external affine parameters")
            parameters.append(np.frombuffer(data, "<f2"))
            slices.append(dict(name=name, offset=offset, length=9216))
    model = build_model(*parameters)
    output.mkdir(parents=True, exist_ok=False)
    (output / "affine.onnx").write_bytes(model.SerializeToString())
    shutil.copyfile(raw_path, output / "raw_norm.f32")
    shutil.copyfile(post_path, output / "captured_post.f32")
    case = dict(model="affine.onnx", repeat=1, inputs=[dict(
        name="raw_norm", dtype="f32", shape=RAW_SHAPE, path="raw_norm.f32")])
    metadata = dict(diagnostic_only=True, source_graph_sha256=SOURCE_SHA256,
                    weights_sha256=WEIGHTS_SHA256, parameter_slices=slices,
                    capture_case_sha256=sha256(capture_case / "case.json"),
                    capture_result_sha256=sha256(capture_result / "result.json"),
                    captured_graph_sha256=record["model_sha256"],
                    graph_sha256=sha256(output / "affine.onnx"),
                    input_sha256=sha256(output / "raw_norm.f32"),
                    captured_post_sha256=sha256(output / "captured_post.f32"))
    for name, value in (("case.json", case), ("affine.json", metadata)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def arithmetic_oracle(raw, gamma, beta):
    """F64 oracle plus bounds for F32 arithmetic and one final FP16 rounding."""
    product = raw.reshape(RESTORED_SHAPE).astype(np.float64) * gamma.astype(np.float64)
    oracle = product + beta.astype(np.float64)
    unit = np.finfo(np.float32).eps / 2
    arithmetic_bound = (2 * unit / (1 - 2 * unit)) * (np.abs(product) + np.abs(beta))
    half = oracle.astype(np.float16)
    upper = np.nextafter(half, np.float16(np.inf)).astype(np.float64)
    lower = np.nextafter(half, np.float16(-np.inf)).astype(np.float64)
    half_bound = np.maximum(upper - half, half - lower) / 2 + arithmetic_bound
    return oracle, [arithmetic_bound, half_bound]


def compare(case, cpu, dml):
    case = Path(case)
    metadata = json.loads((case / "affine.json").read_text())
    for name, key in (("affine.onnx", "graph_sha256"), ("raw_norm.f32", "input_sha256"),
                      ("captured_post.f32", "captured_post_sha256")):
        if sha256(case / name) != metadata[key]:
            raise ValueError(("diagnostic artifact identity differs", name))
    model = onnx.load(case / "affine.onnx")
    parameters = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    raw = np.fromfile(case / "raw_norm.f32", "<f4").reshape(RAW_SHAPE)
    oracle, bounds = arithmetic_oracle(raw, parameters["gamma"], parameters["beta"])
    captured = np.fromfile(case / "captured_post.f32", "<f4").reshape(RESTORED_SHAPE)
    report = dict(diagnostic_only=True, captured_whole_graph_vs_oracle=difference(captured, oracle),
                  providers={})
    for provider, directory in (("cpu", Path(cpu)), ("dml", Path(dml))):
        values = read_outputs(verify(case, directory, provider), directory, OUTPUTS, 1)
        rows = []
        for name, bound in zip(OUTPUTS, bounds):
            value = values[0, name]
            if list(value.shape) != RESTORED_SHAPE:
                raise ValueError("standalone output shape differs")
            rows.append(dict(name=name, **difference(value, oracle),
                elements_outside_arithmetic_bound=int(np.count_nonzero(np.abs(value - oracle) > bound)),
                maximum_arithmetic_bound=float(bound.max())))
        report["providers"][provider] = rows
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("prepare")
    for name in ("source", "weights", "capture_case", "capture_result", "output"):
        builder.add_argument(name, type=Path)
    comparator = commands.add_parser("compare")
    for name in ("case", "cpu", "dml"):
        comparator.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (prepare if command == "prepare" else compare)(**args)
