"""One diagnostic correction: explicit Float32 attention softmax, Half result."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

from export import ROOT, sha256, value

sys.path.insert(0, str(ROOT / "experiments/windows"))
from verify_probe import verify

ORIGINAL_SHA256 = "aedad8e55602f46a748ef29f4e0ae8e6f1fc9be6626b0e779d10c5bf559ae6d8"


def precise_attention(model, expected_count=12):
    model = copy.deepcopy(model)
    types = {v.name: v.type.tensor_type.elem_type for v in
             onnx.shape_inference.infer_shapes(model).graph.value_info}
    nodes, changed = [], []
    for node in model.graph.node:
        if node.op_type == "Softmax" and types.get(node.input[0]) == TensorProto.FLOAT16:
            promoted, result = node.output[0] + "_f32_input", node.output[0] + "_f32_result"
            nodes.append(helper.make_node("Cast", [node.input[0]], [promoted], to=TensorProto.FLOAT))
            softmax = copy.deepcopy(node)
            softmax.input[0], softmax.output[0] = promoted, result
            nodes.extend([softmax, helper.make_node("Cast", [result], [node.output[0]], to=TensorProto.FLOAT16)])
            changed.append(node.name)
        else:
            nodes.append(node)
    if len(changed) != expected_count:
        raise ValueError(("unexpected Half attention softmax inventory", len(changed), expected_count))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model, full_check=True)
    return model, changed


def operator_model(shape):
    nodes = [helper.make_node("Cast", ["logits"], ["half"], to=TensorProto.FLOAT16),
             helper.make_node("Softmax", ["half"], ["native_half"], axis=-1),
             helper.make_node("Cast", ["native_half"], ["native"], to=TensorProto.FLOAT),
             helper.make_node("Cast", ["half"], ["float"], to=TensorProto.FLOAT),
             helper.make_node("Softmax", ["float"], ["float_result"], axis=-1),
             helper.make_node("Cast", ["float_result"], ["precise_half"], to=TensorProto.FLOAT16),
             helper.make_node("Cast", ["precise_half"], ["precise"], to=TensorProto.FLOAT)]
    model = helper.make_model(helper.make_graph(nodes, "Whisper-precise-softmax-control",
        [value("logits", shape)], [value(name, shape) for name in ["native", "precise"]]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model, full_check=True)
    return model


def prepare(original, oracle, output):
    if sha256(original / "whisper.onnx") != ORIGINAL_SHA256:
        raise ValueError("original Whisper candidate identity differs")
    observation = json.loads((oracle / "oracle.json").read_text())
    for name, expected in observation["files"].items():
        if sha256(oracle / name) != expected:
            raise ValueError("operator observation identity differs")
    model, changed = precise_attention(onnx.load(original / "whisper.onnx"))
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "whisper.onnx")
    shutil.copyfile(original / "languages.json", output / "languages.json")
    cases = []
    for case in sorted((original / "cases").glob("neural-*")):
        target = output / "cases" / case.name
        shutil.copytree(case, target)
        cases.append(str(target.relative_to(output) / "case.json"))
    operator = output / "operator"
    shutil.copytree(oracle, operator)
    onnx.save(operator_model(observation["shape"]), operator / "softmax.onnx")
    (operator / "case.json").write_text(json.dumps({"model": "softmax.onnx", "repeat": 2, "inputs": [
        {"name": "logits", "dtype": "f32", "shape": observation["shape"], "path": "input.f32"}]}, indent=2) + "\n")
    provenance = {"original_model_sha256": ORIGINAL_SHA256, "model_sha256": sha256(output / "whisper.onnx"),
        "script_sha256": sha256(Path(__file__)), "changed_attention_softmax": changed,
        "rule": "Half scores -> Cast Float32 -> Softmax -> Cast Half; all other nodes unchanged",
        "operator_oracle_sha256": sha256(oracle / "oracle.json"),
        "operator_model_sha256": sha256(operator / "softmax.onnx"),
        "cases": cases, "operator_case": "operator/case.json", "qualified": False}
    (output / "precision.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def compare_operator(case, output, provider):
    record = verify(case, output, provider)
    expected_record = json.loads((case / "oracle.json").read_text())
    expected_path = case / "precise.f32"
    if sha256(expected_path) != expected_record["files"][expected_path.name]:
        raise ValueError("operator observation identity differs")
    expected = np.fromfile(expected_path, "<f4").reshape(expected_record["shape"])
    reports, first = [], {}
    rows = record["outputs"]
    if {(r["run"], r["name"]) for r in rows} != {(r, n) for r in [0, 1] for n in ["native", "precise"]} or len(rows) != 4:
        raise ValueError("operator output inventory differs")
    for row in rows:
        path = output / row["file"]
        if row["dtype"] != "f32" or row["shape"] != list(expected.shape) or sha256(path) != row["sha256"]:
            raise ValueError("operator output metadata differs")
        actual = np.fromfile(path, "<f4").reshape(expected.shape)
        if not np.isfinite(actual).all():
            raise ValueError("nonfinite operator output")
        first.setdefault(row["name"], actual)
        reports.append({"run": row["run"], "name": row["name"],
            "max_error_from_mlx_precise": float(np.max(np.abs(actual - expected))),
            "different_from_mlx_precise": int(np.count_nonzero(actual != expected)),
            "repeat_error": float(np.max(np.abs(actual - first[row["name"]])))})
    print(json.dumps({"diagnostic_only": True, "reports": reports}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["original", "oracle", "output"]:
        build.add_argument(name, type=Path)
    compare = commands.add_parser("compare-operator")
    compare.add_argument("case", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("--provider", choices=["cpu", "dml"], required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (prepare if command == "prepare" else compare_operator)(**args)
