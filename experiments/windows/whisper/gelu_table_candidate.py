"""Replace only reviewed Half GELU groups with the admitted compiled table."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export import ROOT, sha256
from gelu_table import lookup
from materialization_candidate import SourceGraph

SOURCE_SHA256 = "b117f7591ffb61f5c6781bf2655c8fc49cb0f2d0cfa50903c98447c58412905a"
CONTROL_SHA256 = "6d8fe0afd738edc05f4b53515a7d64df1caff9430a0e604f751440f09e4d5aa7"
TABLE_SHA256 = "06f8fadc81e36eb98608923316816e151f9e0dcaef19eb1ce1f96e6b72f42c91"
TABLE_NAME = "speakerdesk_compiled_half_gelu_table"


def preserve_compiled_gelu(source, table, expected_count=10):
    if table.shape != (65536,) or table.dtype != np.float32:
        raise ValueError("GELU compiler artifact must cover every binary16 pattern")
    finite = np.isfinite(table)
    if np.any(table[finite] != table[finite].astype(np.float16).astype(np.float32)):
        raise ValueError("compiled Half GELU contains a non-Half finite result")
    constants = {tensor.name for tensor in source.graph.initializer}
    if TABLE_NAME in constants:
        raise ValueError("compiled GELU already exists")
    nodes = list(source.graph.node)
    starts = [i for i, node in enumerate(nodes) if node.op_type == "Cast"
              and len(node.output) == 1 and node.output[0].endswith("_gelu_input_f32")]
    if len(starts) != expected_count:
        raise ValueError(("compiled GELU group inventory differs", len(starts), expected_count))
    inferred = onnx.shape_inference.infer_shapes(source)
    types = {info.name: info.type.tensor_type.elem_type for info in
             [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    types.update({tensor.name: tensor.data_type for tensor in source.graph.initializer})
    graph_outputs = {info.name for info in source.graph.output}
    replacements, changes, removed_outputs = {}, [], set()
    for number, start in enumerate(starts):
        limit = starts[number + 1] if number + 1 < len(starts) else len(nodes)
        first = nodes[start]
        if types.get(first.input[0]) != TensorProto.FLOAT16:
            raise ValueError("compiled GELU input is not Half")
        last_primitive = next((i for i in range(start, limit) if nodes[i].op_type == "Div"
                               and nodes[i].output[0].endswith("_gelu_f32")), None)
        if last_primitive is None:
            raise ValueError("GELU final division is missing")
        target = nodes[last_primitive].output[0].removesuffix("_gelu_f32")
        end = next((i for i in range(last_primitive + 1, limit) if list(nodes[i].output) == [target]), None)
        if end is None or nodes[end].op_type != "Cast" or types.get(target) != TensorProto.FLOAT16:
            raise ValueError("GELU final Half materialization is missing")
        region = nodes[start:end + 1]
        primitives = [node.op_type for node in region if node.output[0].endswith("_gelu_f32")]
        if primitives != ["Mul", "Erf", "Add", "Mul", "Div"]:
            raise ValueError(("GELU source recipe changed", primitives))
        produced = {name for node in region for name in node.output}
        external_inputs = {name for node in region for name in node.input} - produced - constants
        external_users = graph_outputs | {name for i, node in enumerate(nodes)
                                           if not start <= i <= end for name in node.input}
        if external_inputs != {first.input[0]} or produced & external_users != {target}:
            raise ValueError("GELU region has another input or observable intermediate")
        graph = SourceGraph("speakerdesk_compiled_gelu_" + target + "_")
        result = lookup(graph, first.output[0], TABLE_NAME)
        replacement = [copy.deepcopy(first), *graph.nodes,
                       helper.make_node("Cast", [result], [target], name=nodes[end].name, to=TensorProto.FLOAT16)]
        replacements[start] = (end, replacement, graph.initializers)
        removed_outputs.update(produced - {first.output[0], target})
        changes.append({"input": first.input[0], "output": target,
                        "old_nodes": len(region), "new_nodes": len(replacement)})
    model = copy.deepcopy(source)
    del model.graph.node[:]
    cursor = 0
    while cursor < len(nodes):
        if cursor in replacements:
            end, replacement, initializers = replacements[cursor]
            model.graph.node.extend(replacement)
            model.graph.initializer.extend(initializers)
            cursor = end + 1
        else:
            model.graph.node.append(copy.deepcopy(nodes[cursor]))
            cursor += 1
    model.graph.initializer.append(numpy_helper.from_array(table, TABLE_NAME))
    retained_info = [copy.deepcopy(info) for info in model.graph.value_info if info.name not in removed_outputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    old_constants = {tensor.name: tensor.SerializeToString() for tensor in source.graph.initializer}
    new_constants = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer}
    if any(new_constants.get(name) != tensor for name, tensor in old_constants.items()):
        raise ValueError("a source parameter or constant changed")
    untouched = [node for i, node in enumerate(nodes)
                 if not any(start <= i <= end for start, (end, _, _) in replacements.items())]
    actual = {node.output[0]: node for node in model.graph.node}
    if any(actual[node.output[0]].SerializeToString() != node.SerializeToString() for node in untouched):
        raise ValueError("a node outside GELU changed")
    if ([info.SerializeToString() for info in source.graph.input] != [info.SerializeToString() for info in model.graph.input]
            or [info.SerializeToString() for info in source.graph.output] != [info.SerializeToString() for info in model.graph.output]):
        raise ValueError("model I/O changed")
    if any(node.op_type == "Erf" for node in model.graph.node):
        raise ValueError("a GELU Erf was not replaced")
    onnx.checker.check_model(model, full_check=True)
    return model, changes


def prepare(original, compiler, output):
    if sha256(original / "whisper.onnx") != SOURCE_SHA256:
        raise ValueError("expected the frozen source-rules-v1 candidate")
    provenance = json.loads((compiler / "gelu-table.json").read_text())
    if (sha256(compiler / "gelu-control.onnx") != CONTROL_SHA256
            or sha256(compiler / "gelu.f32") != TABLE_SHA256
            or provenance["graph_sha256"] != CONTROL_SHA256
            or provenance["table_sha256"] != TABLE_SHA256):
        raise ValueError("compiled GELU differs from the native-admitted control")
    source = onnx.load(original / "whisper.onnx")
    table = np.fromfile(compiler / "gelu.f32", "<f4")
    model, changes = preserve_compiled_gelu(source, table)
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "whisper.onnx")
    shutil.copyfile(original / "languages.json", output / "languages.json")
    cases, copied = [], {}
    model_sha = sha256(output / "whisper.onnx")
    for case in sorted((original / "cases").glob("neural-*")):
        target = output / "cases" / case.name
        target.mkdir(parents=True)
        copied[case.name] = {}
        for filename in ["input.f32", "expected.f32", "expected.json"]:
            shutil.copyfile(case / filename, target / filename)
            copied[case.name][filename] = sha256(target / filename)
            if copied[case.name][filename] != sha256(case / filename):
                raise ValueError("an original input or independent expectation changed")
        metadata = json.loads((case / "case.json").read_text())
        if not metadata.get("disable_optimizations"):
            raise ValueError("graph optimization setting changed")
        metadata["model_files"]["../../whisper.onnx"] = model_sha
        (target / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(str(target.relative_to(output) / "case.json"))
    if len(cases) != 4:
        raise ValueError("expected the four unchanged independent cases")
    receipt = {"source_graph_sha256": SOURCE_SHA256, "model_sha256": model_sha,
        "script_sha256": sha256(Path(__file__)), "lookup_source_sha256": sha256(Path(__file__).with_name("gelu_table.py")),
        "admitted_native_control_graph_sha256": CONTROL_SHA256, "compiled_table_sha256": TABLE_SHA256,
        "compiler_artifact_receipt_sha256": sha256(compiler / "gelu-table.json"),
        "source_runtime_sha256": provenance["runtime_sha256"], "source_metallib_sha256": provenance["metallib_sha256"],
        "source_activation": provenance["source"], "source_initializers_unchanged": True,
        "outside_gelu_nodes_unchanged": True, "model_io_unchanged": True,
        "rule": "Every source Half GELU uses the complete pinned compiled binary16-domain table. Its Half input and final Half materialization remain; only the five GELU primitives and their rounding scaffolding are replaced.",
        "nonfinite_contract": provenance["contract"],
        "independent_reference_sha256": sha256(ROOT / "crates/models/tests/data/whisper-reference.json"),
        "gate": "Unchanged original strict max probability error <1e-4, same routing and exact repeats.",
        "changed_groups": changes, "source_nodes": len(source.graph.node), "candidate_nodes": len(model.graph.node),
        "copied_case_files": copied, "cases": cases, "qualified": False}
    (output / "compiled-gelu.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "copied_case_files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["original", "compiler", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.original, args.compiler, args.output)
