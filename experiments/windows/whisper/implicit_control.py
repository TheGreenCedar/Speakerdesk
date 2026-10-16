"""Small controls for implicit Half boundaries in the frozen Whisper candidate."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from export import Graph, ROOT, sha256, value
from rounding_control import half_grid, RUNTIME_SHA256, METALLIB_SHA256

sys.path.insert(0, str(ROOT / "experiments/windows"))
from verify_probe import verify

INSERT_CAST_SOURCE = {
    "url": "https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/optimizer/insert_cast_transformer.cc",
    "sha256": "555fef17a3aa36c1e6408c9d255274a649638c556099c633b19cd6c5649e9424",
    "finding": "CPU fallback promotes unsupported Half operators; its cast cleanup treats InsertedPrecisionFreeCast conversions as removable without precision loss.",
}
CANDIDATE_SHA256 = "6791343d682d18e252d655128522ad3e605215e8ece31efc4d847268ef12b10f"
CAPTURES = {
    "conv1-pre": "b98f3d64abc45644cff8136ea509a8883adf7c355f25a3a85e17b0886de58332",
    "conv1": "33315b619e59ac149c6fd251c61af111d426c023e86f25dee96297e9be8e37c7",
    "conv2": "2ab248309743f4596dc1c281322d8e122f966f963a7ec17084311c7afdc06a32",
    "added_positions": "9a7309b2c76dff6a713659129a9c00a3dfdc400e326f04b7d0cd80f943a2cfc5",
}
EXTRA_CAPTURES = {
    "pre": "488254285b5588ba8926c565257a90efc75002e8cc895ad132fb4233baa6fc3d",
    "post": CAPTURES["conv2"],
    "first_norm": "dc0b04b80001831ef0a4f7d2490f61d32c3202bc6c14ea7ef36633f457e36290",
}


def bias_model(bias, rows=3):
    """Retain the original Half Add; repair its promoted F32 result separately."""
    g = Graph({})
    source = g.op("Cast", "x", to=TensorProto.FLOAT16)
    added = g.op("Add", source, g.const(np.asarray(bias, np.float16)))
    g.op("Cast", added, name="native", to=TensorProto.FLOAT)
    g.op("Identity", half_grid(g, "native"), name="corrected")
    return g.model("Whisper-conv2-bias-boundary", [value("x", [rows, len(bias)])],
                   [value(name, [rows, len(bias)]) for name in ["native", "corrected"]])


def prefix_model(source):
    targets = {
        "conv1_bias": ("Div_12_gelu_input_f32", [1, 3000, 384]),
        "conv1_gelu": ("Cast_20", [1, 3000, 384]),
        "conv2_bias": ("Div_30_gelu_input_f32", [1, 1500, 384]),
        "conv2_gelu": ("Cast_38", [1, 1500, 384]),
        "positions": ("Cast_42", [1, 1500, 384]),
        "first_norm": ("LayerNormalization_43", [1, 1500, 384]),
    }
    needed = {name for name, _ in targets.values()}
    nodes = []
    for node in reversed(source.graph.node):
        if needed.intersection(node.output):
            nodes.append(copy.deepcopy(node))
            needed.difference_update(node.output)
            needed.update(node.input)
    constants = [copy.deepcopy(tensor) for tensor in source.graph.initializer if tensor.name in needed]
    nodes.reverse()
    # Float32 observers satisfy the native probe contract. They do not assert
    # that CPU promotion preserves every implicit Half boundary internally.
    nodes.extend(helper.make_node("Cast", [tensor], [name], name="observe_" + name,
                                  to=TensorProto.FLOAT) for name, (tensor, _) in targets.items())
    model = helper.make_model(helper.make_graph(nodes, "Whisper-frozen-prefix",
        [copy.deepcopy(source.graph.input[0])],
        [value(name, shape) for name, (_, shape) in targets.items()], constants),
        opset_imports=source.opset_import, ir_version=source.ir_version)
    onnx.checker.check_model(model, full_check=True)
    return model


def write_case(output, name, model, inputs, expected):
    directory = output / name
    directory.mkdir()
    onnx.save(model, directory / "control.onnx")
    declarations = []
    for key, array in inputs.items():
        array = np.asarray(array, np.float32)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite diagnostic input")
        array.astype("<f4").tofile(directory / (key + ".f32"))
        declarations.append({"name": key, "dtype": "f32", "shape": list(array.shape), "path": key + ".f32"})
    metadata = {"model": "control.onnx", "model_files": {"control.onnx": sha256(directory / "control.onnx")},
                "repeat": 2, "disable_optimizations": True, "inputs": declarations}
    (directory / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    expectations = {}
    for key, array in expected.items():
        if not np.isfinite(array).all():
            raise ValueError("nonfinite diagnostic expectation")
        filename = key + ".expected.f32"
        np.asarray(array, dtype="<f4").tofile(directory / filename)
        expectations[key] = {"file": filename, "shape": list(array.shape), "sha256": sha256(directory / filename)}
    (directory / "expected.json").write_text(json.dumps(expectations, indent=2) + "\n")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return {output.name: difference(actual, expected[output.name]) for output, actual in
            zip(session.get_outputs(), session.run(None, inputs))}


def difference(actual, expected):
    return {"nonzero": int(np.count_nonzero(actual != expected)),
            "max_error": float(np.max(abs(actual - expected), initial=0))}


def prepare(candidate, capture, extra, output):
    if sha256(candidate / "whisper.onnx") != CANDIDATE_SHA256:
        raise ValueError("not the frozen nearest-half-v1 graph")
    receipt = json.loads((capture / "receipt.json").read_text())
    if (receipt["runtime_sha256"] != RUNTIME_SHA256 or
            receipt["colocated_metallib_sha256"] != METALLIB_SHA256 or
            receipt["independent_max_error"] != 0 or any(row["nonzero"] for row in receipt["comparisons"])):
        raise ValueError("source capture does not reproduce the production-runtime baseline")
    extra_receipt = json.loads((extra / "receipt.json").read_text())
    if (extra_receipt["runtime_sha256"] != RUNTIME_SHA256
            or extra_receipt["colocated_metallib_sha256"] != METALLIB_SHA256
            or extra_receipt["model_sha256"] != receipt["model_sha256"]
            or extra_receipt["source_sha256"] != receipt["source_sha256"]
            or extra_receipt["post_vs_retained_nonzero"] != 0
            or extra_receipt["input_sha256"] != CAPTURES["conv1"]):
        raise ValueError("second-convolution capture identity differs")
    for directory, pins in [(capture, CAPTURES), (extra, EXTRA_CAPTURES)]:
        for name, pin in pins.items():
            if sha256(directory / (name + ".npy")) != pin:
                raise ValueError(("source stage differs", name))
    source = onnx.load(candidate / "whisper.onnx")
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in source.graph.initializer}
    first = np.load(capture / "conv1.npy", allow_pickle=False).astype(np.float64)
    weights = constants["encoder.conv2.weight"].astype(np.float64)
    bias = constants["encoder.conv2.bias"].astype(np.float64)
    # Silence has two padded boundary classes and one repeated interior class.
    # Exact F64 contraction is diagnostic input construction, never a new golden.
    rows = []
    for index in [0, 1, 1499]:
        window = np.zeros((384, 3), np.float64)
        for kernel in range(3):
            location = index * 2 + kernel - 1
            if 0 <= location < 3000:
                window[:, kernel] = first[0, location]
        rows.append(np.einsum("ock,ck->o", weights, window).astype(np.float16).astype(np.float32))
    rows = np.asarray(rows)
    expected_bias = np.load(extra / "pre.npy", allow_pickle=False)[0, [0, 1, 1499]]
    if np.any((rows.astype(np.float64) + bias).astype(np.float16).astype(np.float32) != expected_bias):
        raise ValueError("exact contraction/round/bias control does not match the maintained MLX operator")
    output.mkdir(parents=True, exist_ok=False)
    local_bias = write_case(output, "conv2-bias", bias_model(bias), {"x": rows},
                            {"native": expected_bias, "corrected": expected_bias})
    if local_bias["native"]["nonzero"] == 0 or local_bias["corrected"]["nonzero"] != 0:
        raise ValueError(("the isolated CPU promotion class was not reproduced/corrected", local_bias))
    mel_path = candidate / "cases/neural-silence/input.f32"
    if sha256(mel_path) != receipt["mel_sha256"]:
        raise ValueError("candidate input differs from the pinned source capture")
    mel = np.fromfile(mel_path, "<f4").reshape(1, 3000, 80)
    expected_prefix = {"conv1_bias": np.load(capture / "conv1-pre.npy"),
                       "conv1_gelu": np.load(capture / "conv1.npy"),
                       "conv2_bias": np.load(extra / "pre.npy"),
                       "conv2_gelu": np.load(capture / "conv2.npy"),
                       "positions": np.load(capture / "added_positions.npy"),
                       "first_norm": np.load(extra / "first_norm.npy")}
    local_prefix = write_case(output, "frozen-prefix", prefix_model(source), {"mel": mel}, expected_prefix)
    provenance = {"diagnostic_only": True, "source_graph_sha256": CANDIDATE_SHA256,
        "capture_receipt_sha256": sha256(capture / "receipt.json"),
        "operator_receipt_sha256": sha256(extra / "receipt.json"),
        "script_sha256": sha256(Path(__file__)), "local_ort": ort.__version__,
        "local_bias": local_bias, "local_prefix": local_prefix, "ort_cpu_fallback_source": INSERT_CAST_SOURCE,
        "cases": ["conv2-bias/case.json", "frozen-prefix/case.json"],
        "source_stages": CAPTURES, "operator_stages": EXTRA_CAPTURES,
        "rule": "No complete model changed; prefix retains original nodes/constants and adds Float32 observers only.",
        "observer_limit": "F32 observers themselves change CPU promotion: prefix CPU values do not reproduce the unobserved full graph. Native DML results may localize an earlier source difference but observer effects remain a diagnostic limitation; these controls do not qualify final probabilities.",
        "candidate_gate_unchanged": "strict max language probability error < 1e-4"}
    (output / "implicit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def compare(case, result, provider):
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("implicit-boundary control used CPU model work")
    expected = json.loads((case / "expected.json").read_text())
    if len(record["outputs"]) != len(expected) * 2 or {(row["run"], row["name"]) for row in record["outputs"]} != {
            (run, name) for run in [0, 1] for name in expected}:
        raise ValueError("diagnostic output inventory differs")
    first, reports, deterministic = {}, [], True
    for row in record["outputs"]:
        spec = expected[row["name"]]
        if sha256(case / spec["file"]) != spec["sha256"]:
            raise ValueError("source-stage expectation identity differs")
        if row["dtype"] != "f32" or row["shape"] != spec["shape"] or sha256(result / row["file"]) != row["sha256"]:
            raise ValueError("native diagnostic output identity differs")
        actual = np.fromfile(result / row["file"], "<f4").reshape(spec["shape"])
        wanted = np.fromfile(case / spec["file"], "<f4").reshape(spec["shape"])
        first.setdefault(row["name"], actual)
        repeat = difference(actual, first[row["name"]])["nonzero"]
        deterministic &= repeat == 0
        reports.append({"run": row["run"], "name": row["name"], **difference(actual, wanted), "repeat_nonzero": repeat})
    print(json.dumps({"diagnostic_only": True, "deterministic": bool(deterministic), "reports": reports}, indent=2))
    return deterministic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["candidate", "capture", "extra", "output"]:
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
