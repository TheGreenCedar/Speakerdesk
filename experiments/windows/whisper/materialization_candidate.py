"""Lower the reviewed source materialization rules in the FloatConv graph."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export import Graph, ROOT, sha256
from materialization_control import half_fma
from rounding_control import half_grid

FLOAT_CONV_SHA256 = "ae89c0ec47f34a4da74c7127c2b83f0a63b8e1fa88033f74380ca4c752d6052a"
MATRIX_SHA256 = "a4fd1e47c1d8081c1b08347c1d98087b9009b15aa94bbbb428bbd55fdbec08b0"
EXPECTED = {"matmul": 89, "encoder_biased": 20, "decoder_biased": 32, "layer_norm": 22}


class SourceGraph(Graph):
    def __init__(self, prefix):
        super().__init__({})
        self.prefix = prefix

    def const(self, value, name=None):
        self.serial += 1
        name = self.prefix + (name or f"constant_{self.serial}")
        self.initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, kind, *inputs, name=None, **attributes):
        self.serial += 1
        name = self.prefix + (name or f"{kind}_{self.serial}")
        self.nodes.append(helper.make_node(kind, list(inputs), [name], name=name, **attributes))
        return name


def source_norm(g, x, weight, bias, epsilon):
    mean = g.op("ReduceMean", x, axes=[-1], keepdims=1)
    minimum = g.op("ReduceMin", x, axes=[-1], keepdims=1)
    maximum = g.op("ReduceMax", x, axes=[-1], keepdims=1)
    # A finite constant row has exactly zero centered values. Mean reciprocal
    # rounding must not turn it into a signal. Nonconstant arithmetic is unchanged.
    centered = g.op("Where", g.op("Equal", minimum, maximum),
                    g.const(np.asarray(0, np.float32)), g.op("Sub", x, mean))
    variance = g.op("ReduceMean", g.op("Mul", centered, centered), axes=[-1], keepdims=1)
    scale = g.op("Div", g.const(np.asarray(1, np.float32)),
                 g.op("Sqrt", g.op("Add", variance, g.const(np.asarray(epsilon, np.float32)))))
    normalized = half_grid(g, g.op("Mul", centered, scale))
    return half_fma(g, normalized, weight, bias)


def preserve_source_materializations(source, expected=EXPECTED):
    model = copy.deepcopy(source)
    inferred = onnx.shape_inference.infer_shapes(source)
    types = {info.name: info.type.tensor_type.elem_type for info in
             [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in source.graph.initializer}
    types.update({tensor.name: tensor.data_type for tensor in source.graph.initializer})
    users = {}
    for node in source.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    counts, groups, encoder_adds = Counter(), {}, {}
    for node in source.graph.node:
        if node.op_type != "MatMul":
            continue
        if types.get(node.output[0]) != TensorProto.FLOAT16:
            raise ValueError(("unexpected non-Half source MatMul", node.name))
        counts["matmul"] += 1
        following = users.get(node.output[0], [])
        biased = (len(following) == 1 and following[0].op_type == "Add"
                  and following[0].input[0] == node.output[0]
                  and node.input[1].endswith(".weight"))
        if biased:
            if node.input[1].startswith("encoder."):
                groups[node.output[0]] = "encoder_biased"
                encoder_adds[following[0].output[0]] = following[0]
                counts["encoder_biased"] += 1
            elif node.input[1].startswith("decoder."):
                groups[node.output[0]] = "decoder_biased"
                counts["decoder_biased"] += 1
            else:
                raise ValueError(("unknown source addmm placement", node.name))
        else:
            groups[node.output[0]] = "unbiased_or_attention"
    counts["layer_norm"] = sum(node.op_type == "LayerNormalization" for node in source.graph.node)
    if dict(counts) != expected:
        raise ValueError(("source operation inventory differs", dict(counts), expected))
    float_values = {name for name, rule in groups.items() if rule == "encoder_biased"}
    copies, new_constants, replacements = {}, [], {}

    def promote(graph, name):
        if name in float_values or types.get(name) == TensorProto.FLOAT:
            return name
        if types.get(name) != TensorProto.FLOAT16:
            raise ValueError(("source arithmetic input is not Half", name))
        if name in constants:
            if name not in copies:
                full_name = "speakerdesk_source_parameter_" + name
                if full_name in types:
                    raise ValueError("source materialization correction already exists")
                copies[name] = full_name
                new_constants.append(numpy_helper.from_array(constants[name].astype(np.float32), full_name))
            return copies[name]
        return graph.op("Cast", name, to=TensorProto.FLOAT)

    changes = []
    for node in source.graph.node:
        target = node.output[0]
        if node.op_type != "MatMul" and node.op_type != "LayerNormalization" and target not in encoder_adds:
            continue
        graph = SourceGraph("speakerdesk_source_" + target + "_")
        if node.op_type == "MatMul":
            result = graph.op("MatMul", *(promote(graph, name) for name in node.input))
            rule = groups[target]
            if rule != "encoder_biased":
                result = graph.op("Cast", half_grid(graph, result), to=TensorProto.FLOAT16)
        elif target in encoder_adds:
            result = graph.op("Add", *(promote(graph, name) for name in node.input))
            result = graph.op("Cast", half_grid(graph, result), to=TensorProto.FLOAT16)
            rule = "encoder_addmm_final_round"
        else:
            attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            if attributes.get("axis", -1) != -1 or len(node.input) != 3:
                raise ValueError(("unsupported source LayerNorm", node.name))
            epsilon = attributes.get("epsilon", 1e-5)
            if not np.isclose(epsilon, 1e-5, rtol=1e-6, atol=0):
                raise ValueError("source LayerNorm epsilon differs")
            result = source_norm(graph, *(promote(graph, name) for name in node.input), epsilon)
            result = graph.op("Cast", result, to=TensorProto.FLOAT16)
            rule = "normalized_half_then_fused_affine"
        replacements[target] = [*graph.nodes, helper.make_node("Identity", [result], [target], name=node.name)]
        new_constants.extend(graph.initializers)
        changes.append({"node": node.name, "output": target, "rule": rule})
    nodes = [replacement for node in source.graph.node for replacement in replacements.get(node.output[0], [copy.deepcopy(node)])]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(new_constants)
    for info in model.graph.value_info:
        if info.name in float_values:
            info.type.tensor_type.elem_type = TensorProto.FLOAT
    actual = {node.output[0]: node for node in model.graph.node}
    if any(actual[node.output[0]].SerializeToString() != node.SerializeToString()
           for node in source.graph.node if node.output[0] not in replacements):
        raise ValueError("a source operation outside the reviewed rules changed")
    actual_constants = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer}
    if any(actual_constants.get(tensor.name) != tensor.SerializeToString() for tensor in source.graph.initializer):
        raise ValueError("an original source parameter changed")
    if ([info.SerializeToString() for info in model.graph.input] != [info.SerializeToString() for info in source.graph.input]
            or [info.SerializeToString() for info in model.graph.output] != [info.SerializeToString() for info in source.graph.output]):
        raise ValueError("the model I/O contract changed")
    onnx.checker.check_model(model, full_check=True)
    return model, {"counts": dict(counts), "changes": changes, "parameter_copies": copies}


def prepare(original, output):
    if sha256(original / "whisper.onnx") != FLOAT_CONV_SHA256:
        raise ValueError("expected the reviewed FloatConv graph")
    source = onnx.load(original / "whisper.onnx")
    model, changes = preserve_source_materializations(source)
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
                raise ValueError("original case input or reference changed")
        metadata = json.loads((case / "case.json").read_text())
        if not metadata.get("disable_optimizations"):
            raise ValueError("source case lost disabled optimizations")
        metadata["model_files"] = {"../../whisper.onnx": model_sha}
        (destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(str(destination.relative_to(output) / "case.json"))
    if len(cases) != 4:
        raise ValueError("expected the four retained neural cases")
    provenance = {"source_graph_sha256": FLOAT_CONV_SHA256, "model_sha256": model_sha,
        "script_sha256": sha256(Path(__file__)),
        "arithmetic_control_source_sha256": sha256(Path(__file__).with_name("materialization_control.py")),
        "reviewed_native_matrix_graph_sha256": MATRIX_SHA256,
        "independent_reference_sha256": sha256(ROOT / "crates/models/tests/data/whisper-reference.json"),
        "source_initializers_unchanged": True, "other_source_nodes_unchanged": True,
        "native_residual_add_scale_mul_preserved": True, "decoder_bias_adds_preserved": 32, "conv_gelu_positions_softmax_preserved": True,
        "constant_normalization_rule": "A finite row with min==max has exactly zero centered values; all other rows retain the reviewed F32 formula.",
        "affine_contract": "Half-valued operands with abs(product+bias)<65504; qualified Half-FMA tie repair.",
        "gate": "Original independent probability error strictly <1e-4, same route and exact repeats; intermediate bitexactness is diagnostic only.",
        "added_nodes": len(model.graph.node) - len(source.graph.node), **changes,
        "copied_case_files": copied, "cases": cases, "qualified": False}
    (output / "materialization.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({key: val for key, val in provenance.items() if key not in ["changes", "parameter_copies", "copied_case_files"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.original, args.output)
