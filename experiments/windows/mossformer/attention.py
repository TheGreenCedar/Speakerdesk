"""Lower the pinned FLASH attention equations to bounded, DML-supported operations."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper

SOURCE_PROGRAM = "e640a6ac4627b7e19fcb781fcf59d1b8c67116786f8d0857fbf81091ea12e5e2"
EQUATIONS = {
    "/block/qk_offset_scale/Einsum": "...d,hd->...hd",
    "/block/Einsum": "...,f->...f",
    **{f"/block/Einsum_{i}": "...,f->...f" for i in range(1, 4)},
    "/block/Einsum_4": "...id,...jd->...ij",
    **{f"/block/Einsum_{i}": "...ij,...jd->...id" for i in (5, 6)},
    **{f"/block/Einsum_{i}": "bgnd,bgne->bde" for i in (7, 9)},
    **{f"/block/Einsum_{i}": "bgnd,bde->bgne" for i in (8, 10)},
}
AXES = {"batch": [0], "offset": [2], "rotary": [1]}


def lower_attention(model):
    candidates = [node for node in model.graph.node if node.op_type == "Einsum"]
    actual = {node.name: next((helper.get_attribute_value(a).decode().replace(" ", "")
              for a in node.attribute if a.name == "equation"), None) for node in candidates}
    if len(candidates) != len(EQUATIONS) or actual != EQUATIONS:
        raise ValueError("expected exactly the twelve source-owned FLASH attention equations")
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    shapes = {v.name: v.type.tensor_type for v in [*inferred.graph.input, *inferred.graph.value_info,
              *inferred.graph.output]}
    shapes.update({t.name: helper.make_tensor_type_proto(t.data_type, t.dims).tensor_type
                   for t in model.graph.initializer})
    source_input = next((v for v in model.graph.input if v.name == "x"), None)
    if source_input is not None and (source_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in source_input.type.tensor_type.shape.dim] != [1, 0, 512]):
        raise ValueError("MossFormer requires fixed batch one, dynamic frames and 512 input channels")

    def require_shape(name, expected):
        tensor = shapes.get(name)
        if tensor is None or tensor.elem_type != onnx.TensorProto.FLOAT or not tensor.HasField("shape"):
            raise ValueError(f"missing Float32 attention tensor shape: {name}")
        dimensions = tensor.shape.dim
        if len(dimensions) != len(expected) or any(
                want is not None and got.HasField("dim_value") and got.dim_value != want
                for got, want in zip(dimensions, expected)):
            raise ValueError(f"incompatible source attention shape: {name}")

    names = {name for node in model.graph.node for name in [*node.input, *node.output]}
    names.update(tensor.name for tensor in model.graph.initializer)
    constants = {label: f"speakerdesk_attention_{label}_axis" for label in AXES}
    if any(name in names for name in constants.values()) or any(
            "/bounded_matmul/" in node.name or "/dml_attention/" in node.name for node in model.graph.node):
        raise ValueError("attention is already lowered or generated names collide")
    # Validate the entire causal class before mutating any node or parameter inventory.
    for node in candidates:
        if len(node.input) != 2 or len(node.output) != 1:
            raise ValueError("unexpected source attention interface")
        equation = actual[node.name]
        if equation == "...d,hd->...hd":
            expected = ([1, None, 128], [4, 128], [1, None, 4, 128])
        elif equation == "...,f->...f":
            expected = ([None], [16], [None, 16])
        elif equation == "...id,...jd->...ij":
            expected = ([1, None, 256, 128], [1, None, 256, 128], [1, None, 256, 256])
        elif equation == "...ij,...jd->...id":
            expected = ([1, None, 256, 256], [1, None, 256, 1024], [1, None, 256, 1024])
        elif equation == "bgnd,bgne->bde":
            expected = ([1, None, 256, 128], [1, None, 256, 1024], [1, 128, 1024])
        else:
            expected = ([1, None, 256, 128], [1, 128, 1024], [1, None, 256, 1024])
        for name, shape in zip([*node.input, *node.output], expected):
            require_shape(name, shape)
    replacements, records = {}, []
    for node in candidates:
        equation = actual[node.name]
        if equation == "bgnd,bde->bgne":
            continue  # These two explicit-label equations already use bounded DML GEMM.
        prefix = node.name + "/dml_attention/"
        left, right = node.input
        if equation == "bgnd,bgne->bde":
            flattened_left, flattened_right, transposed, product = [prefix + suffix
                for suffix in ["left", "right", "transpose", "product"]]
            # Batch is one: collapse the two contracted axes, avoiding a 24 GiB broadcast product.
            nodes = [
                helper.make_node("Flatten", [left], [flattened_left], name=flattened_left, axis=3),
                helper.make_node("Flatten", [right], [flattened_right], name=flattened_right, axis=3),
                helper.make_node("Transpose", [flattened_left], [transposed], name=transposed, perm=[1, 0]),
                helper.make_node("MatMul", [transposed, flattened_right], [product], name=product),
                helper.make_node("Unsqueeze", [product, constants["batch"]], list(node.output), name=prefix + "batch"),
            ]
        elif equation == "...id,...jd->...ij":
            transposed = prefix + "key_transpose"
            nodes = [helper.make_node("Transpose", [right], [transposed], name=transposed, perm=[0, 1, 3, 2]),
                     helper.make_node("MatMul", [left, transposed], list(node.output), name=prefix + "matmul")]
        elif equation == "...ij,...jd->...id":
            nodes = [helper.make_node("MatMul", [left, right], list(node.output), name=prefix + "matmul")]
        else:
            # These outer products have no reduction; only insert the broadcast axis.
            expanded = prefix + "expanded"
            axis = constants["offset" if equation == "...d,hd->...hd" else "rotary"]
            nodes = [helper.make_node("Unsqueeze", [left, axis], [expanded], name=expanded),
                     helper.make_node("Mul", [expanded, right], list(node.output), name=prefix + "multiply")]
        if any(name in names for replacement in nodes for name in replacement.output if name not in node.output):
            raise ValueError("generated attention value name collides")
        replacements[node.name] = nodes
        records.append(dict(name=node.name, equation=equation, inputs=list(node.input),
                            outputs=list(node.output), fixed_batch=1,
                            replacement_operators=[item.op_type for item in nodes]))
    rewritten = [item for node in model.graph.node for item in replacements.get(node.name, [node])]
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(numpy_helper.from_array(np.array(values, np.int64), constants[label])
                                   for label, values in AXES.items())
    return records


def transform(source, output):
    from export import json_write, sha256

    if output.exists():
        raise ValueError("candidate artifact directory must be new")
    if sha256(source / "program.json") != SOURCE_PROGRAM:
        raise ValueError("expected the pinned normalized source program")
    original = json.loads((source / "program.json").read_text())
    for name, digest in {**original["model_files"], **{
            graph["model"]: graph["sha256"] for graph in original["graphs"]}}.items():
        if Path(name).name != name or sha256(source / name) != digest:
            raise ValueError("source artifact identity differs")
    output.mkdir(parents=True, exist_ok=False)
    program = copy.deepcopy(original)
    changed = []
    for graph in program["graphs"]:
        if not graph["name"].startswith("flash-"):
            shutil.copyfile(source / graph["model"], output / graph["model"])
            continue
        model = onnx.load(source / graph["model"], load_external_data=False)
        original_tensors = [tensor.SerializeToString() for tensor in model.graph.initializer]
        original_nodes = {node.name: node.SerializeToString() for node in model.graph.node}
        records = lower_attention(model)
        if original_tensors != [tensor.SerializeToString() for tensor in model.graph.initializer[:len(original_tensors)]]:
            raise ValueError("a learned parameter changed")
        if any(node.SerializeToString() != original_nodes[node.name] for node in model.graph.node
               if node.name in original_nodes):
            raise ValueError("an unrelated operation changed")
        onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        path = output / graph["model"]
        path.write_bytes(model.SerializeToString())
        graph["source_graph_sha256"] = graph["sha256"]
        graph["sha256"] = sha256(path)
        graph["operators"] = sorted({node.op_type for node in model.graph.node})
        graph["attention_lowerings"] = records
        changed.append(graph["name"])
    for name in original["model_files"]:
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "program.json", output / "source-program.json")
    program["attention_sha256"] = sha256(Path(__file__))
    program["frozen_transform"] = dict(source_program_sha256=SOURCE_PROGRAM,
        unchanged_external_weights=True, changed_stages=changed)
    program["source_adaptations"].append("FLASH attention uses bounded MatMul contractions and broadcast Mul outer products; no ellipsis CPU fallback")
    json_write(output / "program.json", program)
    print(json.dumps(dict(program_sha256=sha256(output / "program.json"),
                         changed_stages=changed), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    transform(args.source, args.output)
