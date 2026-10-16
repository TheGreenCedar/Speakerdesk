"""Finish-only bounded Float32 reduction tree for the proven global-norm error.

Replace one unit-affine InstanceNormalization, preserving its output edge and
every learned parameter. No Float64 operators or acoustic-window segmentation.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnx

from diagnose_finish import mapped_float
from diagnose_global_norm import SOURCE_GRAPH
from export import ROOT, json_write, sha256
from normalization import replace_global_norm
from qualify import waveform_error


WEIGHTS_SHA256 = "37277bcd8ecc7a6c4792de49d7ae726c09db8d8042b19b45c523540f6605d6bf"




def prepare(source_graph, source_case, graph, case):
    source_graph, source_case, graph, case = map(Path, (source_graph, source_case, graph, case))
    descriptor = graph.with_suffix(".globalnorm.json")
    if graph.exists() or case.exists() or descriptor.exists() or graph.parent.resolve() != source_graph.parent.resolve():
        raise ValueError("use new graph/case paths beside the original weights")
    if sha256(source_graph) != SOURCE_GRAPH:
        raise ValueError("expected the frozen original finish graph")
    original_case = json.loads((source_case / "case.json").read_text())
    if (source_case / original_case["model"]).resolve() != source_graph.resolve():
        raise ValueError("source case belongs to another graph")
    rows = original_case["inputs"]
    if len(rows) != 4 or {row["name"] for row in rows} != {"audio", "encoded", "residual", "x"}:
        raise ValueError("expected the captured full-window finish inputs")
    samples = next(row for row in rows if row["name"] == "audio")["shape"][1]
    if not 16 <= samples <= 392000:
        raise ValueError("captured acoustic window is outside the original bounds")
    frames = (samples - 16) // 8 + 1
    shapes = dict(audio=[1, samples], encoded=[1, 512, frames], residual=[1, 512, frames], x=[1, frames, 512])
    identities = []
    for row in rows:
        if row["dtype"] != "f32" or row["shape"] != shapes[row["name"]]:
            raise ValueError("captured input type/shape differs")
        _, digest = mapped_float(source_case / row["path"], row["shape"])
        identities.append(dict(name=row["name"], dtype="f32", shape=row["shape"], sha256=digest))
    weights = original_case["model_files"]
    if len(weights) != 1 or list(weights.values()) != [WEIGHTS_SHA256]:
        raise ValueError("external parameter identity differs")
    for name, digest in weights.items():
        if Path(name).name != "mossformer.weights" or sha256(source_case / name) != digest:
            raise ValueError("external parameter bytes differ")
    model = onnx.load(source_graph, load_external_data=False)
    initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
    nodes = {node.name: node.SerializeToString() for node in model.graph.node}
    changes = replace_global_norm(model)
    if (initializers != [tensor.SerializeToString() for tensor in model.graph.initializer[:len(initializers)]] or
            any(node.SerializeToString() != nodes[node.name] for node in model.graph.node
                if node.name in nodes and node.name != changes["removed_node"]["name"])):
        raise ValueError("candidate changed a learned parameter or unrelated node")
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    graph.write_bytes(model.SerializeToString())
    case.mkdir(parents=True, exist_ok=False)
    specification = dict(model=os.path.relpath(graph.resolve(), case.resolve()).replace("\\", "/"), repeat=1,
        inputs=[dict(row, path=os.path.relpath((source_case / row["path"]).resolve(), case.resolve()).replace("\\", "/")) for row in rows],
        model_files={os.path.relpath((source_case / name).resolve(), case.resolve()).replace("\\", "/"): digest for name, digest in weights.items()})
    metadata = dict(candidate_only=True, source_graph_sha256=SOURCE_GRAPH, graph_sha256=sha256(graph),
        source_case_sha256=sha256(source_case / "case.json"), inputs=identities, changes=changes,
        unchanged_source_initializers=len(initializers), script_sha256=sha256(Path(__file__)),
        limitation="Finish-only candidate on captured state; independent full-forward references and their original bounds remain required.")
    json_write(case / "case.json", specification)
    json_write(case / "candidate.json", metadata)
    json_write(descriptor, metadata)
    print(json.dumps(metadata, indent=2))


def compare(case, result, torch_case, original_case):
    case, result, torch_case, original_case = map(Path, (case, result, torch_case, original_case))
    metadata = json.loads((case / "candidate.json").read_text())
    reference = json.loads((torch_case / "finish-taps.json").read_text())
    if (metadata["source_graph_sha256"] != SOURCE_GRAPH or reference["source_graph_sha256"] != SOURCE_GRAPH or
            reference["source_case_sha256"] != metadata["source_case_sha256"] or
            sha256(original_case / "case.json") != metadata["source_case_sha256"]):
        raise ValueError("captured source/Torch finish provenance differs")
    sys.path.insert(0, str(ROOT / "experiments/windows"))
    from verify_probe import verify
    native = verify(case, result, "cpu")
    if native["model_sha256"] != metadata["graph_sha256"] or native["inputs"] != metadata["inputs"]:
        raise ValueError("candidate graph or captured inputs differ")
    if (len(native["outputs"]) != 2 or
            {(row["run"], row["name"]) for row in native["outputs"]} != {(0, name) for name in ["source0", "source1"]}):
        raise ValueError("missing or duplicate candidate channels")
    independent_path = original_case / "independent.json"
    independent = json.loads(independent_path.read_text()) if independent_path.exists() else None
    channels = []
    for name in ["source0", "source1"]:
        row = next(row for row in native["outputs"] if row["name"] == name)
        expected = reference["expected"][name]
        if row["dtype"] != "f32" or row["shape"] != expected["shape"]:
            raise ValueError("candidate channel type/shape differs")
        actual = mapped_float(result / row["file"], row["shape"], row["sha256"])[0]
        wanted = mapped_float(torch_case / expected["file"], expected["shape"], expected["sha256"])[0]
        record = dict(name=name, same_captured_state_torch=waveform_error(actual, wanted))
        if independent:
            target = independent[name]
            values = mapped_float(original_case / target["file"], target["shape"], target["sha256"])[0]
            record["independent_full_forward"] = waveform_error(actual, values)
        channels.append(record)
    report = dict(candidate_only=True, channels=channels,
        finish_only_passed=all(row["same_captured_state_torch"]["within_original_bounds"] for row in channels),
        full_forward_qualification="not performed; same captured state does not qualify the original full pipeline")
    json_write(case / "comparison.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("prepare")
    for name in ["source_graph", "source_case", "graph", "case"]:
        builder.add_argument(name, type=Path)
    comparator = commands.add_parser("compare")
    for name in ["case", "result", "torch_case", "original_case"]:
        comparator.add_argument(name, type=Path)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    (prepare if command == "prepare" else compare)(**arguments)
