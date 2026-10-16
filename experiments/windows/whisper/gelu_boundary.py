"""Preserve the observed compiled MLX Half GELU arithmetic in portable ONNX."""
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HALF_RECIPROCAL = float(np.float16(1 / np.float16(np.sqrt(2))))


def preserve_gelu_boundaries(model, expected_count=10):
    """Change GELU only; explicit Float32 islands keep each Half rounding visible."""
    model = copy.deepcopy(model)
    producers = {name: node for node in model.graph.node for name in node.output}
    users = {}
    for node in model.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}

    def scalar(name, expected):
        value = constants.get(name)
        return value is not None and value.shape == () and value.dtype == np.float16 and float(value) == expected

    def only_user(node):
        found = users.get(node.output[0], [])
        if len(found) != 1:
            raise ValueError(("unexpected GELU intermediate consumers", node.name))
        return found[0]

    groups = []
    for erf in model.graph.node:
        if erf.op_type != "Erf":
            continue
        first = producers[erf.input[0]]
        if not (first.op_type == "Div" and scalar(first.input[1], float(np.float16(np.sqrt(2))))
                or first.op_type == "Mul" and scalar(first.input[1], HALF_RECIPROCAL)):
            raise ValueError(("unexpected GELU square-root boundary", first.name))
        if only_user(first) != erf:
            raise ValueError("GELU input arithmetic has another consumer")
        add = only_user(erf)
        if add.op_type != "Add" or len(add.input) != 2 or not any(scalar(name, 1.) for name in add.input):
            raise ValueError(("unexpected GELU addition", add.name))
        multiply = only_user(add)
        if multiply.op_type != "Mul" or set(multiply.input) != {first.input[0], add.output[0]}:
            raise ValueError(("unexpected GELU product", multiply.name))
        last = only_user(multiply)
        if last.op_type != "Div" or last.input[0] != multiply.output[0] or not scalar(last.input[1], 2.):
            raise ValueError(("unexpected GELU final scale", last.name))
        groups.append((first, erf, add, multiply, last))
    if len(groups) != expected_count:
        raise ValueError(("unexpected Half GELU inventory", len(groups), expected_count))

    new_constants = {"speakerdesk_gelu_reciprocal_f32": HALF_RECIPROCAL,
                     "speakerdesk_gelu_one_f32": 1., "speakerdesk_gelu_two_f32": 2.}
    if set(new_constants).intersection(constants):
        raise ValueError("GELU precision boundaries are already present")
    for name, value in new_constants.items():
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
    replacements, changed = {}, []
    float_intermediates = set()
    for first, erf, add, multiply, last in groups:
        promoted = first.output[0] + "_gelu_input_f32"
        replacements[first.name] = [helper.make_node("Cast", [first.input[0]], [promoted], to=TensorProto.FLOAT)]
        stages = [
            (first, "Mul", [promoted, "speakerdesk_gelu_reciprocal_f32"]),
            (erf, "Erf", [first.output[0]]),
            (add, "Add", [erf.output[0], "speakerdesk_gelu_one_f32"]),
            (multiply, "Mul", [promoted, add.output[0]]),
            (last, "Div", [multiply.output[0], "speakerdesk_gelu_two_f32"]),
        ]
        for original, kind, inputs in stages:
            result = original.output[0] + "_gelu_f32"
            half = original.output[0] if original == last else original.output[0] + "_gelu_half"
            nodes = replacements.setdefault(original.name, [])
            nodes.append(helper.make_node(kind, inputs, [result], name=original.name))
            nodes.append(helper.make_node("Cast", [result], [half], to=TensorProto.FLOAT16))
            if original != last:
                nodes.append(helper.make_node("Cast", [half], list(original.output), to=TensorProto.FLOAT))
                float_intermediates.update(original.output)
        changed.append(first.name)
    nodes = [replacement for node in model.graph.node
             for replacement in replacements.get(node.name, [node])]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for info in model.graph.value_info:
        if info.name in float_intermediates:
            info.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(model, full_check=True)
    return model, changed


def operator_model(shape):
    """The original exporter GELU in isolation, with Float32 probe I/O."""
    constants = [numpy_helper.from_array(np.asarray(value, np.float16), name)
                 for name, value in [("sqrt_two", np.sqrt(2)), ("one", 1), ("two", 2)]]
    nodes = [helper.make_node("Cast", ["x"], ["half"], to=TensorProto.FLOAT16),
             helper.make_node("Div", ["half", "sqrt_two"], ["divided"], name="divided"),
             helper.make_node("Erf", ["divided"], ["erf"], name="erf"),
             helper.make_node("Add", ["one", "erf"], ["gated"], name="gated"),
             helper.make_node("Mul", ["half", "gated"], ["product"], name="product"),
             helper.make_node("Div", ["product", "two"], ["result"], name="result"),
             helper.make_node("Cast", ["result"], ["y"], to=TensorProto.FLOAT)]
    graph = helper.make_graph(nodes, "Whisper-compiled-Half-GELU-control",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model, full_check=True)
    return model
