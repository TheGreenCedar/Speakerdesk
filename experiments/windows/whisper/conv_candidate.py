"""Replace only the two native Half convolution/bias arithmetic boundaries."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export import Graph, sha256
from implicit_control import CANDIDATE_SHA256
from rounding_control import half_grid

CONTROL_SHA256 = "10eac649b96d4103e291e1d15708c9c8afd714c86c859f71b390f7691dc3fdf4"


def preserve_convolution_boundaries(source, expected_count=2):
    model = copy.deepcopy(source)
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    users = {}
    for node in model.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    additions, before, changed_nodes, changed_types = [], {}, set(), set()
    changes = []

    def single_user(node):
        found = users.get(node.output[0], [])
        if len(found) != 1:
            raise ValueError(("unexpected convolution boundary consumers", node.name))
        return found[0]

    def half_cast(node):
        if node.op_type != "Cast" or next(a.i for a in node.attribute if a.name == "to") != TensorProto.FLOAT16:
            raise ValueError(("missing original Half boundary", node.name))
        return node

    def grid_before(cast, input_name, prefix):
        graph = Graph({})
        result = half_grid(graph, input_name)
        locals_ = {name for node in graph.nodes for name in node.output}
        locals_.update(tensor.name for tensor in graph.initializers)
        rename = lambda name: prefix + name if name in locals_ else name
        for node in graph.nodes:
            node.name = prefix + node.name
            node.input[:] = [rename(name) for name in node.input]
            node.output[:] = [rename(name) for name in node.output]
        for tensor in graph.initializers:
            tensor.name = rename(tensor.name)
        before.setdefault(cast.output[0], []).extend(graph.nodes)
        additions.extend(graph.initializers)
        cast.input[0] = rename(result)
        changed_nodes.add(cast.output[0])

    for conv in model.graph.node:
        if conv.op_type != "Conv":
            continue
        if len(conv.input) != 2 or conv.input[1] not in constants:
            raise ValueError(("unexpected convolution inputs", conv.name))
        weight = constants[conv.input[1]]
        if weight.dtype != np.float16 or weight.ndim != 3 or weight.shape[2] != 3:
            raise ValueError(("unexpected source Half convolution weight", conv.name))
        first = half_cast(single_user(conv))
        second = half_cast(single_user(first))
        transpose = single_user(second)
        if transpose.op_type != "Transpose" or list(next(a.ints for a in transpose.attribute if a.name == "perm")) != [0, 2, 1]:
            raise ValueError(("unexpected convolution layout boundary", conv.name))
        bias = single_user(transpose)
        if bias.op_type != "Add" or len(bias.input) != 2 or bias.input[0] != transpose.output[0]:
            raise ValueError(("unexpected separate convolution bias", conv.name))
        bias_values = constants.get(bias.input[1])
        if bias_values is None or bias_values.dtype != np.float16 or bias_values.shape != (weight.shape[0],):
            raise ValueError(("unexpected source Half bias", conv.name))
        bias_cast = half_cast(single_user(bias))
        prefix = "speakerdesk_f32_" + conv.output[0] + "_"
        weight_name, bias_name = prefix + "weight", prefix + "bias"
        if weight_name in constants or bias_name in constants:
            raise ValueError("convolution correction already exists")
        additions.extend([numpy_helper.from_array(weight.astype(np.float32), weight_name),
                          numpy_helper.from_array(bias_values.astype(np.float32), bias_name)])
        source_input, source_weight, source_bias = conv.input[0], conv.input[1], bias.input[1]
        promoted_input, promoted_bias_input = prefix + "input", prefix + "bias_input"
        before[conv.output[0]] = [helper.make_node("Cast", [source_input], [promoted_input],
                                                     name=promoted_input, to=TensorProto.FLOAT)]
        before[bias.output[0]] = [helper.make_node("Cast", [bias.input[0]], [promoted_bias_input],
                                                     name=promoted_bias_input, to=TensorProto.FLOAT)]
        conv.input[:] = [promoted_input, weight_name]
        bias.input[:] = [promoted_bias_input, bias_name]
        grid_before(first, conv.output[0], prefix + "contraction_")
        grid_before(bias_cast, bias.output[0], prefix + "addition_")
        changed_nodes.update([conv.output[0], bias.output[0]])
        changed_types.update([conv.output[0], bias.output[0]])
        changes.append({"convolution": conv.name, "bias": bias.name, "source_input": source_input,
                        "source_weight": source_weight, "source_bias": source_bias,
                        "convolution_round": first.name, "bias_round": bias_cast.name})
    if len(changes) != expected_count:
        raise ValueError(("unexpected convolution inventory", len(changes), expected_count))
    nodes = [added for node in model.graph.node for added in [*before.get(node.output[0], []), node]]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(additions)
    for info in model.graph.value_info:
        if info.name in changed_types:
            info.type.tensor_type.elem_type = TensorProto.FLOAT
    # Only original input edges on these eight nodes may change. This also
    # checks attributes, order-independent output identity and existing weights.
    actual = {node.output[0]: node for node in model.graph.node}
    for node in source.graph.node:
        check = copy.deepcopy(actual[node.output[0]])
        if node.output[0] in changed_nodes:
            check.input[:] = node.input
        if check.SerializeToString() != node.SerializeToString():
            raise ValueError(("unrelated source node changed", node.name))
    actual_constants = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer}
    if any(actual_constants.get(tensor.name) != tensor.SerializeToString() for tensor in source.graph.initializer):
        raise ValueError("an existing source initializer changed")
    onnx.checker.check_model(model, full_check=True)
    return model, changes


def prepare(original, output):
    if sha256(original / "whisper.onnx") != CANDIDATE_SHA256:
        raise ValueError("expected the frozen nearest-half-v1 graph")
    source = onnx.load(original / "whisper.onnx")
    model, changes = preserve_convolution_boundaries(source)
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "whisper.onnx")
    shutil.copyfile(original / "languages.json", output / "languages.json")
    model_sha = sha256(output / "whisper.onnx")
    cases, copied = [], {}
    for case in sorted((original / "cases").glob("neural-*")):
        destination = output / "cases" / case.name
        destination.mkdir(parents=True)
        copied[case.name] = {}
        for name in ["input.f32", "expected.f32", "expected.json"]:
            shutil.copyfile(case / name, destination / name)
            copied[case.name][name] = sha256(destination / name)
            if copied[case.name][name] != sha256(case / name):
                raise ValueError("input or independent expectation changed")
        metadata = json.loads((case / "case.json").read_text())
        if not metadata.get("disable_optimizations"):
            raise ValueError("source case lost disabled graph optimizations")
        metadata["model_files"] = {"../../whisper.onnx": model_sha}
        (destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(str(destination.relative_to(output) / "case.json"))
    if len(cases) != 4:
        raise ValueError("expected all four retained neural cases")
    provenance = {"source_graph_sha256": CANDIDATE_SHA256, "model_sha256": model_sha,
        "script_sha256": sha256(Path(__file__)),
        "grid_source_sha256": sha256(Path(__file__).with_name("rounding_control.py")),
        "qualified_conv1_operator_graph_sha256": CONTROL_SHA256,
        "rule": "Two Half Conv paths use F32 convolution on identical Half-valued input/weights, nearest-Half grid before original Half Cast, separate F32 bias addition and nearest-Half grid before original bias Half Cast.",
        "source_initializers_unchanged": True, "other_source_nodes_unchanged": True,
        "changed_boundaries": changes, "added_nodes": len(model.graph.node) - len(source.graph.node),
        "copied_case_files": copied, "cases": cases, "disable_optimizations": True,
        "gate": "Original independent max probability error strictly <1e-4, same route and exact repeats",
        "limitation": "Conv1 corrected bias passed full pinned silence capture on native DML. Conv2 uses the same arithmetic class; the complete independent gate decides whether this composition holds. Other implicit Half arithmetic is unchanged.",
        "qualified": False}
    (output / "convolution.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.original, args.output)
