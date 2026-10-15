"""Inspect or transform the frozen 50-stage artifact without tracing/inference.

Copy unchanged graphs and weights verbatim. Rewrite only the proven source-owned
global GroupNorm lowerings and retain the original program as provenance.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import onnx

from export import ROOT, SOURCE_FILES, SOURCE_REVISION, json_write, sha256
from normalization import apply_stage, inventory


def admitted_program(source):
    program = json.loads((source / "program.json").read_text())
    checkpoint = json.loads((ROOT / "crates/host/src/separation-artifact.json").read_text())
    names = ["encoder"] + [name for index in range(24) for name in [f"flash-{index:02d}", f"fsmn-{index:02d}"]] + ["finish"]
    if (program["source_revision"] != SOURCE_REVISION or program["source_sha256"] != SOURCE_FILES or
            program["checkpoint"] != checkpoint or program["dtype"] != "float32" or
            program["latent_channels"] != 512 or program["sample_bounds"] != [16, 392000] or
            [graph["name"] for graph in program["graphs"]] != names or
            program["stages_sha256"] != sha256(Path(__file__).with_name("stages.py")) or
            "normalization_sha256" in program):
        raise ValueError("expected the original pinned staged program")
    for name in [*[graph["model"] for graph in program["graphs"]], *program["model_files"]]:
        if Path(name).name != name:
            raise ValueError("expected flat original artifact filenames")
    return program


def inspect(source):
    source = Path(source)
    program = admitted_program(source)
    rows = []
    for graph in program["graphs"]:
        path = source / graph["model"]
        if sha256(path) != graph["sha256"]:
            raise ValueError(("source graph differs", graph["name"]))
        model = onnx.load(path, load_external_data=False)
        norms = inventory(model)
        expected = {"encoder": ["/norm/InstanceNormalization"],
                    "finish": ["/global_norm/InstanceNormalization"]}.get(graph["name"], [])
        actual = [row["name"] for row in norms if row["same_global_groupnorm_class"]]
        if actual != expected:
            raise ValueError(("source GroupNorm inventory differs", graph["name"], actual, expected))
        if norms:
            rows.append(dict(stage=graph["name"], graph_sha256=graph["sha256"], normalizations=norms))
    return dict(source_program_sha256=sha256(source / "program.json"), stages=rows,
        global_groupnorm_count=sum(row["same_global_groupnorm_class"] for graph in rows for row in graph["normalizations"]),
        instance_normalization_count=sum(len(graph["normalizations"]) for graph in rows))


def transform(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("normalized artifact directory must be new")
    original = admitted_program(source)
    inspected = inspect(source)
    for name, digest in original["model_files"].items():
        if sha256(source / name) != digest:
            raise ValueError("source external weights differ")
    output.mkdir(parents=True, exist_ok=False)
    program = copy.deepcopy(original)
    changed = []
    for graph in program["graphs"]:
        if graph["name"] not in {"encoder", "finish"}:
            shutil.copyfile(source / graph["model"], output / graph["model"])
            continue
        model = onnx.load(source / graph["model"], load_external_data=False)
        tensors = [tensor.SerializeToString() for tensor in model.graph.initializer]
        before = {node.name: node.SerializeToString() for node in model.graph.node}
        _, records = apply_stage(model, graph["name"])
        replaced = {row["removed_node"]["name"] for row in records}
        if (tensors != [tensor.SerializeToString() for tensor in model.graph.initializer[:len(tensors)]] or
                any(node.SerializeToString() != before[node.name] for node in model.graph.node
                    if node.name in before and node.name not in replaced)):
            raise ValueError("transform modified a learned initializer or unrelated node")
        onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        path = output / graph["model"]
        path.write_bytes(model.SerializeToString())
        graph["source_graph_sha256"] = graph["sha256"]
        graph["sha256"] = sha256(path)
        graph["operators"] = sorted({node.op_type for node in model.graph.node})
        graph["normalization_lowerings"] = records
        changed.append(graph["name"])
    for name in original["model_files"]:
        shutil.copyfile(source / name, output / name)
        if sha256(output / name) != original["model_files"][name]:
            raise ValueError("copied external weights differ")
    shutil.copyfile(source / "program.json", output / "source-program.json")
    program["normalization_sha256"] = sha256(Path(__file__).with_name("normalization.py"))
    program["frozen_transform"] = dict(source_program_sha256=inspected["source_program_sha256"],
        script_sha256=sha256(Path(__file__)), unchanged_external_weights=True, changed_stages=changed)
    program["source_adaptations"] = [*original["source_adaptations"],
        "bounded Float32 centered-variance sums for encoder/finish GroupNorm(1,512), original epsilon/affine"]
    json_write(output / "program.json", program)
    json_write(output / "normalization-inventory.json", inspected)
    print(json.dumps(dict(output=str(output), program_sha256=sha256(output / "program.json"),
        changed_stages=changed, unchanged_stages=50-len(changed), inventory=inspected), indent=2))
    return program


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reader = commands.add_parser("inventory")
    reader.add_argument("source", type=Path)
    writer = commands.add_parser("transform")
    writer.add_argument("source", type=Path)
    writer.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "inventory":
        print(json.dumps(inspect(arguments.source), indent=2))
    else:
        transform(arguments.source, arguments.output)
