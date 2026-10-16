"""Bounded F32 lowering shared by new exports and frozen artifact transforms."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def inventory(model):
    """Distinguish GroupNorm(1,512) lowering from per-channel InstanceNorm."""
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}

    def constant(name):
        node = producers.get(name)
        tensor = initializers.get(name)
        if node is not None and node.op_type == "Constant":
            tensor = next((a.t for a in node.attribute if a.name == "value"), None)
        if tensor is None or tensor.external_data:
            return None
        return numpy_helper.to_array(tensor)

    norms = [node for node in model.graph.node if node.op_type == "InstanceNormalization"]
    if not norms:
        return []
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {value.name: [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                           for dim in value.type.tensor_type.shape.dim]
              for value in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]}
    records = []
    for node in norms:
        epsilon = next(helper.get_attribute_value(a) for a in node.attribute if a.name == "epsilon")
        before = producers.get(node.input[0])
        groups = constant(before.input[1]) if before is not None and before.op_type == "Reshape" else None
        scale, bias = constant(node.input[1]), constant(node.input[2])
        source_shape = shapes.get(before.input[0], []) if before is not None and before.op_type == "Reshape" else []
        matches = (groups is not None and np.array_equal(groups, [0, 1, -1]) and
                   scale is not None and scale.dtype == np.float32 and np.array_equal(scale, [1.]) and
                   bias is not None and bias.dtype == np.float32 and np.array_equal(bias, [0.]) and
                   np.isclose(epsilon, 1e-8, rtol=1e-7, atol=0) and
                   len(source_shape) == 3 and source_shape[0] == 1 and source_shape[1] == 512)
        records.append(dict(name=node.name, epsilon=epsilon, input_shape=shapes.get(node.input[0]),
            source_shape=source_shape, grouped_shape=groups.tolist() if groups is not None else None,
            same_global_groupnorm_class=bool(matches)))
    return records


def apply_stage(model, stage):
    """Only source-owned encoder/finish GroupNorm(1,512), never FSMN InstanceNorm."""
    before = inventory(model)
    expected = {"encoder": ["/norm/InstanceNormalization"],
                "finish": ["/global_norm/InstanceNormalization"]}.get(stage, [])
    actual = [row["name"] for row in before if row["same_global_groupnorm_class"]]
    if actual != expected:
        raise ValueError(("source GroupNorm inventory differs", stage, actual, expected))
    records = [replace_global_norm(model, name.rsplit("/", 1)[0]) for name in actual]
    return before, records


def replace_global_norm(model, prefix="/global_norm"):
    candidate_prefix = prefix + "/bounded_sum/"
    nodes = {node.name: node for node in model.graph.node}
    norm = nodes[prefix + "/InstanceNormalization"]
    if norm.op_type != "InstanceNormalization":
        raise ValueError("expected the frozen global unit normalization")
    epsilon = next(helper.get_attribute_value(a) for a in norm.attribute if a.name == "epsilon")
    if not np.isclose(epsilon, 1e-8, rtol=1e-7, atol=0):
        raise ValueError("expected the frozen global unit normalization")
    for suffix, expected in [("Constant_1", [1.]), ("Constant_2", [0.])]:
        node = nodes[prefix + "/" + suffix]
        tensor = next(a.t for a in node.attribute if a.name == "value")
        if tensor.data_type != TensorProto.FLOAT or not np.array_equal(numpy_helper.to_array(tensor), expected):
            raise ValueError("global InstanceNormalization affine is not unit/zero")
    if list(norm.input[1:]) != [nodes[prefix + "/Constant_1"].output[0], nodes[prefix + "/Constant_2"].output[0]]:
        raise ValueError("unexpected global normalization affine wiring")
    if any(node.name.startswith(candidate_prefix) for node in model.graph.node):
        raise ValueError("normalization candidate is already applied")
    original_node = dict(name=norm.name, operation=norm.op_type, inputs=list(norm.input),
                         outputs=list(norm.output), epsilon=epsilon)
    constants = {
        "shape512": np.array([1, 1, -1, 512], np.int64),
        "shape256": np.array([1, 1, -1, 256], np.int64),
        "axis3": np.array([3], np.int64), "axis2": np.array([2], np.int64),
        "axis0": np.array([0], np.int64), "index2": np.array(2, np.int64),
        "divisor512": np.array(512, np.int64), "divisor256": np.array(256, np.int64),
        "pad_prefix": np.zeros(5, np.int64), "epsilon": np.array(epsilon, np.float32),
    }
    names = {value.name for value in model.graph.initializer}
    names.update(name for node in model.graph.node for name in node.output)
    if any(candidate_prefix + name in names for name in constants):
        raise ValueError("candidate constant identity collision")
    new_nodes = []

    def emit(operation, inputs, output, **attributes):
        name = candidate_prefix + output
        new_nodes.append(helper.make_node(operation, inputs, [name], name=name, **attributes))
        return name

    c = {name: candidate_prefix + name for name in constants}
    shape = emit("Shape", [norm.input[0]], "input_shape")
    count = emit("Gather", [shape, c["index2"]], "count", axis=0)
    partial_count = emit("Div", [count, c["divisor512"]], "partial_count")
    negative_count = emit("Neg", [partial_count], "negative_partial_count")
    pad_count = emit("Mod", [negative_count, c["divisor256"]], "pad_count", fmod=0)
    pad_vector = emit("Unsqueeze", [pad_count, c["axis0"]], "pad_vector")
    pads = emit("Concat", [c["pad_prefix"], pad_vector], "pads", axis=0)
    count_f32 = emit("Cast", [count], "count_f32", to=TensorProto.FLOAT)

    def mean(value, name):
        blocks = emit("Reshape", [value, c["shape512"]], name + "_blocks512")
        partials = emit("ReduceSum", [blocks, c["axis3"]], name + "_partials512", keepdims=0)
        # Only the <=48,999 partials are padded. Centered original values are
        # squared before this operation, so padding never enters the variance.
        padded = emit("Pad", [partials, pads], name + "_padded", mode="constant")
        blocks = emit("Reshape", [padded, c["shape256"]], name + "_blocks256")
        partials = emit("ReduceSum", [blocks, c["axis3"]], name + "_partials256", keepdims=0)
        total = emit("ReduceSum", [partials, c["axis2"]], name + "_total", keepdims=1)
        return emit("Div", [total, count_f32], name)

    average = mean(norm.input[0], "mean")
    centered = emit("Sub", [norm.input[0], average], "centered")
    squared = emit("Mul", [centered, centered], "squared_centered")
    variance = mean(squared, "variance")
    stabilized = emit("Add", [variance, c["epsilon"]], "variance_epsilon")
    denominator = emit("Sqrt", [stabilized], "denominator")
    new_nodes.append(helper.make_node("Div", [centered, denominator], list(norm.output), name=norm.name))
    rewritten = [replacement for node in model.graph.node
                 for replacement in (new_nodes if node.name == norm.name else [node])]
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(numpy_helper.from_array(value, c[name]) for name, value in constants.items())
    return dict(removed_node=original_node,
        added_nodes=[dict(name=node.name, operation=node.op_type, inputs=list(node.input), outputs=list(node.output),
                         attributes={a.name: helper.get_attribute_value(a).decode() if a.type == onnx.AttributeProto.STRING else helper.get_attribute_value(a)
                                     for a in node.attribute}) for node in new_nodes],
        added_constants=[dict(name=c[name], dtype=str(value.dtype), shape=list(value.shape), value=value.tolist())
                         for name, value in constants.items()],
        reduction_lengths=[512, 256, 192], maximum_source_elements=512 * 48999,
        semantics="mean(x); mean((x-mean)^2); divide centered x by sqrt(variance+same epsilon); original learned affine follows unchanged")

