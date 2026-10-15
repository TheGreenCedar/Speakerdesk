"""Build a side-by-side ReDimNet stage GroupNorm affine FP32 control.

The proven discrepancy is after raw InstanceNormalization. Leave that operation
and its parameters unchanged; cast its output and the learned FP16 affine
parameters to FP32, then cast the affine result back to its original FP16 edge.
External tensor descriptors and bytes are never rewritten or loaded.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from prepare_taps import STAGE_TAPS, sha256


SOURCE_SHA256 = "89c532e48539bebddc168e773cdcbf014b1941043c50d0a4ff7ccd1139efa5f1"
STAGE_AFFINES = [name.rsplit("/", 1)[0] for name in list(STAGE_TAPS.values())[:6]]
SUFFIX = "__affine_f32"


def promote_affines(model, stage0_only=False):
    """Validate the same exported pattern at every selected stage before mutation."""
    nodes = {node.name: node for node in model.graph.node}
    if len(nodes) != len(model.graph.node):
        raise ValueError("duplicate source node names")
    consumers = Counter(name for node in model.graph.node for name in node.input)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    output_names = {value.name for value in model.graph.output}
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {value.name: value.type.tensor_type.elem_type for value in
             [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]}
    types.update({tensor.name: tensor.data_type for tensor in model.graph.initializer})
    selected = STAGE_AFFINES[:1] if stage0_only else STAGE_AFFINES
    regions = []
    for prefix in selected:
        def get(suffix, operation):
            node = nodes[prefix + "/" + suffix]
            if node.op_type != operation or len(node.output) != 1:
                raise ValueError(("unexpected GroupNorm operation", node.name))
            return node

        norm = get("InstanceNormalization", "InstanceNormalization")
        reshape = get("Reshape_1", "Reshape")
        gamma = get("Unsqueeze", "Unsqueeze")
        beta = get("Unsqueeze_1", "Unsqueeze")
        mul = get("Mul", "Mul")
        add = get("Add", "Add")
        if (reshape.input[0] != norm.output[0] or
                list(mul.input) != [reshape.output[0], gamma.output[0]] or
                list(add.input) != [mul.output[0], beta.output[0]]):
            raise ValueError(("unexpected GroupNorm affine wiring", prefix))
        for node in (reshape, mul):
            if consumers[node.output[0]] != 1 or node.output[0] in output_names:
                raise ValueError(("affine intermediate has another consumer", node.name))
        for node in (norm, reshape, gamma, beta, mul, add):
            if types[node.output[0]] != TensorProto.FLOAT16:
                raise ValueError(("expected original FP16 region", node.name))
        parameters = []
        for node, kind in ((gamma, "weight"), (beta, "bias")):
            tensor = initializers[node.input[0]]
            if (not tensor.name.endswith("." + kind) or
                    tensor.data_type != TensorProto.FLOAT16 or list(tensor.dims) != [4608]):
                raise ValueError(("unexpected learned affine parameter", tensor.name))
            parameters.append(tensor.name)
        regions.append((prefix, norm, reshape, gamma, beta, mul, add, parameters))

    replacements = {}
    records = []
    for prefix, norm, reshape, gamma, beta, mul, add, parameters in regions:
        norm32, gamma32, beta32 = [node.output[0] + SUFFIX for node in (norm, gamma, beta)]
        reshape32, mul32, add32 = [node.output[0] + SUFFIX for node in (reshape, mul, add)]
        new_reshape, new_mul, new_add = [copy.deepcopy(node) for node in (reshape, mul, add)]
        new_reshape.input[0], new_reshape.output[0] = norm32, reshape32
        new_mul.input[:], new_mul.output[0] = [reshape32, gamma32], mul32
        new_add.input[:], new_add.output[0] = [mul32, beta32], add32
        # Keep the original stage boundary node name/output on the final Cast so
        # existing stage taps still observe the downstream FP16 feature map.
        new_add.name = add.name + SUFFIX
        replacements[reshape.name] = [helper.make_node("Cast", [norm.output[0]], [norm32],
            name=prefix + "/Cast_norm_f32", to=TensorProto.FLOAT), new_reshape]
        replacements[mul.name] = [helper.make_node("Cast", [gamma.output[0]], [gamma32],
            name=prefix + "/Cast_gamma_f32", to=TensorProto.FLOAT), new_mul]
        replacements[add.name] = [helper.make_node("Cast", [beta.output[0]], [beta32],
            name=prefix + "/Cast_beta_f32", to=TensorProto.FLOAT), new_add,
            helper.make_node("Cast", [add32], [add.output[0]], name=add.name,
                             to=TensorProto.FLOAT16)]
        records.append(dict(region=prefix, raw_normalization=norm.output[0],
                            learned_parameters=parameters, original_output=add.output[0],
                            changed_operations=[reshape.name, mul.name, add.name]))
    rewritten = [replacement for node in model.graph.node
                 for replacement in replacements.get(node.name, [node])]
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return records


def prepare(source, output, stage0_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata_path = output.with_suffix(".precision.json")
    if output.exists() or metadata_path.exists():
        raise ValueError("candidate graph and metadata paths must be new")
    if source.parent != output.parent:
        raise ValueError("candidate must stay beside the unchanged external weights")
    source_digest = sha256(source)
    if source_digest != SOURCE_SHA256:
        raise ValueError("expected the pinned original embedding graph")
    model = onnx.load(source, load_external_data=False)
    if [value.name for value in model.graph.output] != ["embedding"]:
        raise ValueError("expected the original embedding-only graph")
    initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
    records = promote_affines(model, stage0_only)
    if initializers != [tensor.SerializeToString() for tensor in model.graph.initializer]:
        raise ValueError("candidate changed an initializer")
    # Shape/type inference catches broken Cast/Reshape/Mul/Add wiring without
    # loading external data or executing the model.
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    with output.open("xb") as stream:
        stream.write(model.SerializeToString())
    metadata = dict(candidate_only=True, source_graph_sha256=source_digest,
                    graph_sha256=sha256(output), stage0_only=stage0_only, regions=records,
                    initializers_unchanged=True, raw_normalization_unchanged=True,
                    limitation="Requires native CPU/DML comparison against the independent legacy identity gate; not a qualified graph.")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage0-only", action="store_true",
                        help="limit the same control to the first proven faulty stage")
    args = parser.parse_args()
    prepare(args.source, args.output, args.stage0_only)
