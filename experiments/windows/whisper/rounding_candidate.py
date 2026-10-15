"""Apply the native-qualified RNE grid only at explicit Float32-to-Half edges."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import onnx
from onnx import TensorProto

from export import Graph, sha256
from rounding_control import half_grid

POSITION_SHA256 = "f913bd2696974dbd8ea04c848052446e342faeb2ebb482971690f3a83d645926"


def preserve_half_conversions(model, expected_count=63):
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {value.name: value.type.tensor_type.elem_type for value in
             [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    types.update({value.name: value.data_type for value in inferred.graph.initializer})
    implicit = Counter(node.op_type for node in model.graph.node if node.op_type != "Cast"
                       and any(types.get(name) == TensorProto.FLOAT16 for name in node.output))
    corrected = copy.deepcopy(model)
    nodes, changed = [], []
    for node in corrected.graph.node:
        if node.op_type != "Cast" or next(attribute.i for attribute in node.attribute
                                          if attribute.name == "to") != TensorProto.FLOAT16:
            nodes.append(node)
            continue
        dtype = types.get(node.input[0])
        if dtype is None:
            raise ValueError(("unknown input type at Half conversion", node.name))
        if dtype != TensorProto.FLOAT:
            nodes.append(node)
            continue
        graph = Graph({})
        rounded = half_grid(graph, node.input[0])
        prefix = "speakerdesk_rne_" + node.output[0] + "_"
        local_names = {name for operation in graph.nodes for name in operation.output}
        local_names.update(value.name for value in graph.initializers)
        rename = lambda name: prefix + name if name in local_names else name
        for operation in graph.nodes:
            operation.name = prefix + operation.name
            operation.input[:] = [rename(name) for name in operation.input]
            operation.output[:] = [rename(name) for name in operation.output]
        for value in graph.initializers:
            value.name = rename(value.name)
        changed.append({"name": node.name, "input": node.input[0], "output": node.output[0]})
        node.input[0] = rename(rounded)
        nodes.extend([*graph.nodes, node])
        corrected.graph.initializer.extend(graph.initializers)
    if len(changed) != expected_count:
        raise ValueError(("unexpected Float32-to-Half inventory", len(changed), expected_count))
    del corrected.graph.node[:]
    corrected.graph.node.extend(nodes)
    onnx.checker.check_model(corrected, full_check=True)
    return corrected, {"changed_casts": changed, "implicit_half_outputs_unchanged": dict(implicit)}


def prepare(original, output):
    if sha256(original / "whisper.onnx") != POSITION_SHA256:
        raise ValueError("expected the frozen source-position/GELU Whisper graph")
    source = onnx.load(original / "whisper.onnx")
    model, changes = preserve_half_conversions(source)
    by_output = {node.output[0]: node for node in model.graph.node}
    changed_outputs = {row["output"] for row in changes["changed_casts"]}
    for node in source.graph.node:
        actual = by_output[node.output[0]]
        if node.output[0] in changed_outputs:
            actual = copy.deepcopy(actual)
            actual.input[:] = node.input
        if actual.SerializeToString() != node.SerializeToString():
            raise ValueError(("an unrelated source node changed", node.name))
    old_constants = {value.name: value.SerializeToString() for value in source.graph.initializer}
    new_constants = {value.name: value.SerializeToString() for value in model.graph.initializer}
    if any(new_constants.get(name) != value for name, value in old_constants.items()):
        raise ValueError("a source weight or constant changed")
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "whisper.onnx")
    shutil.copyfile(original / "languages.json", output / "languages.json")
    cases = []
    for case in sorted((original / "cases").glob("neural-*")):
        destination = output / "cases" / case.name
        destination.mkdir(parents=True)
        for name in ["input.f32", "expected.f32", "expected.json"]:
            shutil.copyfile(case / name, destination / name)
        metadata = json.loads((case / "case.json").read_text())
        if not metadata.get("disable_optimizations"):
            raise ValueError("the source case lost disabled graph optimizations")
        metadata["model_files"] = {"../../whisper.onnx": sha256(output / "whisper.onnx")}
        (destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(str(destination.relative_to(output) / "case.json"))
    provenance = {"source_graph_sha256": POSITION_SHA256,
                  "model_sha256": sha256(output / "whisper.onnx"),
                  "script_sha256": sha256(Path(__file__)),
                  "grid_source_sha256": sha256(Path(__file__).with_name("rounding_control.py")),
                  "qualified_operator_graph_sha256": "83b3df6e22caad23437c79503037035f8535d4d571b515201d54d6d9478298ac",
                  "source_constants_unchanged": True, "source_nodes_preserved": True,
                  "added_node_count": len(model.graph.node) - len(source.graph.node),
                  "rule": "Validated F32 nearest-even Half grid before existing F32-to-Half Casts only; original Casts and implicit Half arithmetic preserved",
                  "limitation": "The operator control does not establish internal rounding/accumulation of implicit Half Conv, MatMul, LayerNormalization, Add or Mul. No such arithmetic was changed.",
                  "disable_optimizations": True, **changes, "cases": cases, "qualified": False}
    (output / "rounding.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["original", "output"]:
        parser.add_argument(name, type=Path)
    prepare(**vars(parser.parse_args()))
