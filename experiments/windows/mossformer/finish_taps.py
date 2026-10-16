"""Tap the frozen finish graph and unchanged Torch finish on captured state only.

Preparation runs Torch finish once. Native execution is a separate finish-only
probe invocation. Original full-forward references are never written.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

from diagnose_finish import bind_inputs, mapped_float
from export import ROOT, SOURCE_FILES, SOURCE_REVISION, json_write, load_source, sha256, verify_checkpoint
from qualify import waveform_error


TAPS = {
    "intra_norm": "/intra_norm/LayerNormalization",
    "global_norm": "/global_norm/Add",
    "projection": "/projection/Conv",
    "mask_projection": "/mask_projection/Conv",
    "decoder_input0": "/Gather_2",
    "decoder_input1": "/Gather_3",
    "decoder_output0": "/decoder/Squeeze",
    "decoder_output1": "/decoder_1/Squeeze",
}


def add_outputs(model, samples):
    if [value.name for value in model.graph.output] != ["source0", "source1"]:
        raise ValueError("expected the frozen two-source finish graph")
    frames = (samples - 16) // 8 + 1
    shapes = {"intra_norm": [1, frames, 512], "global_norm": [1, 512, frames],
              "projection": [1, 1024, frames], "mask_projection": [2, 512, frames],
              **{f"decoder_input{i}": [1, 512, frames] for i in range(2)},
              **{f"decoder_output{i}": [1, (frames - 1) * 8 + 16] for i in range(2)}}
    nodes = {node.name: node for node in model.graph.node}
    records = []
    for name, source_node in TAPS.items():
        node = nodes[source_node]
        if len(node.output) != 1:
            raise ValueError(("unexpected multi-output finish boundary", source_node))
        records.append(dict(name=name, output=node.output[0], shape=shapes[name]))
    model.graph.output.extend(helper.make_tensor_value_info(
        row["output"], TensorProto.FLOAT, row["shape"]) for row in records)
    return records


def capture_hooks(finish, save):
    """Observe existing calls without replacing output tensors or repeating math."""
    handles, calls = [], Counter()

    def output(name):
        def hook(module, arguments, result):
            calls[name] += 1
            save(name, result)
        return hook

    for name in ("intra_norm", "global_norm", "projection", "mask_projection"):
        handles.append(getattr(finish, name).register_forward_hook(output(name)))

    def decoder_input(module, arguments):
        index = calls["decoder_input"]
        calls["decoder_input"] += 1
        save(f"decoder_input{index}", arguments[0])

    def decoder_output(module, arguments, result):
        index = calls["decoder_output"]
        calls["decoder_output"] += 1
        save(f"decoder_output{index}", result)

    handles.append(finish.decoder.register_forward_pre_hook(decoder_input))
    handles.append(finish.decoder.register_forward_hook(decoder_output))
    return handles, calls


def prepare(artifact, case, checkpoint, source_parent, references, graph, output, threads=4):
    artifact, case, checkpoint, source_parent, references, graph, output = map(Path,
        (artifact, case, checkpoint, source_parent, references, graph, output))
    if graph.exists() or output.exists() or graph.parent.resolve() != artifact.resolve():
        raise ValueError("use new graph/case paths; graph must remain beside original weights")
    pin = verify_checkpoint(checkpoint)
    program_path = artifact / "program.json"
    program = json.loads(program_path.read_text())
    reference_path = references / "references.json"
    reference = json.loads(reference_path.read_text())
    if (program["checkpoint"] != pin or program["source_revision"] != SOURCE_REVISION or
            program["source_sha256"] != SOURCE_FILES or reference["source_revision"] != SOURCE_REVISION or
            reference["source_sha256"] != SOURCE_FILES or
            reference["checkpoint_sha256"] != pin["files"]["last_best_checkpoint.pt"]["sha256"] or
            program["stages_sha256"] != sha256(Path(__file__).with_name("stages.py"))):
        raise ValueError("source, checkpoint, wrapper or independent provenance differs")
    original = program["graphs"][-1]
    source_case, inputs, identities, original_reference = bind_inputs(case, reference)
    source_graph = artifact / "finish.onnx"
    if (original["name"] != "finish" or original["model"] != "finish.onnx" or
            (case / source_case["model"]).resolve() != source_graph.resolve() or
            sha256(source_graph) != original["sha256"]):
        raise ValueError("captured case does not identify the frozen finish graph")
    for name, digest in program["model_files"].items():
        if sha256(artifact / name) != digest:
            raise ValueError("external model parameters differ")
    samples = inputs["audio"].shape[1]
    model = onnx.load(source_graph, load_external_data=False)
    nodes_before = [node.SerializeToString() for node in model.graph.node]
    tensors_before = [tensor.SerializeToString() for tensor in model.graph.initializer]
    taps = add_outputs(model, samples)
    if (nodes_before != [node.SerializeToString() for node in model.graph.node] or
            tensors_before != [tensor.SerializeToString() for tensor in model.graph.initializer]):
        raise ValueError("tap graph changed source operations or initializers")
    output.mkdir(parents=True, exist_ok=False)
    graph.write_bytes(model.SerializeToString())
    expected = {}

    def save(name, tensor):
        values = tensor.detach().cpu().numpy()
        shape = next((row["shape"] for row in taps if row["name"] == name), [1, samples])
        if values.dtype != np.float32 or list(values.shape) != shape or not np.isfinite(values).all():
            raise ValueError(("Torch boundary type/shape differs", name, list(values.shape), shape))
        path = output / ("torch-" + name + ".f32")
        values.astype("<f4", copy=False).tofile(path)
        expected[name] = dict(file=path.name, shape=shape, sha256=sha256(path))

    import torch
    from stages import Finish
    source, _ = load_source(source_parent)
    torch.set_num_threads(threads)
    raw = source.MossFormer().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    raw.load_state_dict(state, strict=True)
    finish = Finish(raw).eval()
    del raw, state
    handles, calls = capture_hooks(finish, save)
    try:
        with torch.inference_mode():
            outputs = finish(*(torch.from_numpy(inputs[name]) for name in ["audio", "encoded", "residual", "x"]))
        for index, value in enumerate(outputs):
            save(f"source{index}", value)
    finally:
        for handle in handles:
            handle.remove()
    if calls != Counter(intra_norm=1, global_norm=1, projection=1, mask_projection=1,
                        decoder_input=2, decoder_output=2):
        raise ValueError(("unexpected source finish call sequence", dict(calls)))
    tapped_case = dict(model=os.path.relpath(graph.resolve(), output.resolve()).replace("\\", "/"), repeat=1,
        inputs=[dict(row, path=os.path.relpath((case / row["path"]).resolve(), output.resolve()).replace("\\", "/"))
                for row in source_case["inputs"]],
        model_files={os.path.relpath((artifact / name).resolve(), output.resolve()).replace("\\", "/"): digest
                     for name, digest in program["model_files"].items()})
    metadata = dict(diagnostic_only=True, source_graph_sha256=original["sha256"], graph_sha256=sha256(graph),
        source_case_sha256=sha256(case / "case.json"), inputs=identities, taps=taps, expected=expected,
        program_sha256=sha256(program_path), checkpoint_sha256=pin["files"]["last_best_checkpoint.pt"]["sha256"],
        source_revision=SOURCE_REVISION, source_sha256=SOURCE_FILES, weights=program["model_files"],
        stages_sha256=program["stages_sha256"], references_manifest_sha256=sha256(reference_path),
        reference_case=original_reference["id"], torch_threads=threads,
        versions={name: importlib.metadata.version(name) for name in ["torch", "numpy"]},
        script_sha256=sha256(Path(__file__)),
        captured_bytes=sum(int(np.prod(row["shape"])) * 4 for row in expected.values()),
        limitation="Added graph outputs can change optimization. Compare retained source channels with their original native baseline; this is not full-forward qualification.")
    json_write(output / "case.json", tapped_case)
    json_write(output / "finish-taps.json", metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "expected"}, indent=2))


def tensor_error(actual, expected):
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("boundary tensor shape differs or is empty")
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    maximum = squared = reference_squared = actual_squared = dot = 0.0
    for start in range(0, actual.size, 1_000_000):
        a, b = actual[start:start + 1_000_000].astype(np.float64), expected[start:start + 1_000_000].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite finish boundary")
        delta = a - b
        maximum = max(maximum, float(np.max(np.abs(delta))))
        squared += float(np.dot(delta, delta))
        reference_squared += float(np.dot(b, b))
        actual_squared += float(np.dot(a, a))
        dot += float(np.dot(a, b))
    return dict(max_abs_error=maximum, normalized_rms_error=float(np.sqrt(squared / max(reference_squared, 1e-20))),
                rms_error=float(np.sqrt(squared / actual.size)),
                cosine=dot / np.sqrt(actual_squared * reference_squared) if actual_squared * reference_squared else None)


def compare(case, original_case, result, baseline):
    case, original_case, result, baseline = map(Path, (case, original_case, result, baseline))
    sys.path.insert(0, str(ROOT / "experiments/windows"))
    from verify_probe import verify
    metadata = json.loads((case / "finish-taps.json").read_text())
    if sha256(original_case / "case.json") != metadata["source_case_sha256"]:
        raise ValueError("baseline case identity differs")
    native, original = verify(case, result, "cpu"), verify(original_case, baseline, "cpu")
    if (native["model_sha256"] != metadata["graph_sha256"] or
            original["model_sha256"] != metadata["source_graph_sha256"] or native["inputs"] != original["inputs"]):
        raise ValueError("finish graph or captured input identity differs")
    names = ["source0", "source1", *[row["output"] for row in metadata["taps"]]]
    if (len(native["outputs"]) != len(names) or
            {(row["run"], row["name"]) for row in native["outputs"]} != {(0, name) for name in names}):
        raise ValueError("missing or duplicate tapped finish outputs")
    count = json.loads((original_case / "case.json").read_text())["repeat"]
    if (len(original["outputs"]) != 2 * count or
            {(row["run"], row["name"]) for row in original["outputs"]} !=
            {(run, name) for run in range(count) for name in ["source0", "source1"]}):
        raise ValueError("missing or duplicate baseline source channels")

    def read(directory, row):
        if row["dtype"] != "f32":
            raise ValueError("native finish output is not Float32")
        return mapped_float(directory / row["file"], row["shape"], row["sha256"])[0]

    rows = []
    for tap in metadata["taps"]:
        actual = read(result, next(row for row in native["outputs"] if row["name"] == tap["output"]))
        expected = read(case, dict(metadata["expected"][tap["name"]], dtype="f32"))
        rows.append(dict(name=tap["name"], **tensor_error(actual, expected)))
    channels = []
    for name in ["source0", "source1"]:
        actual = read(result, next(row for row in native["outputs"] if row["name"] == name))
        expected = read(case, dict(metadata["expected"][name], dtype="f32"))
        controls = [dict(run=row["run"], **tensor_error(actual, read(baseline, row)))
                    for row in original["outputs"] if row["name"] == name]
        channels.append(dict(name=name, torch_vs_native=waveform_error(actual, expected),
                             instrumentation_vs_original_native=controls))
    report = dict(diagnostic_only=True, taps=rows, channels=channels,
        instrumentation_perturbed=any(control["max_abs_error"] != 0 for channel in channels
                                     for control in channel["instrumentation_vs_original_native"]),
        limitation="Intermediate errors locate divergence on captured state only; original full-forward waveform gates remain unchanged.")
    json_write(case / "finish-tap-comparison.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("prepare")
    for name in ["artifact", "case", "checkpoint", "source_parent", "references", "graph", "output"]:
        builder.add_argument(name, type=Path)
    builder.add_argument("--threads", type=int, default=4)
    comparator = commands.add_parser("compare")
    for name in ["case", "original_case", "result", "baseline"]:
        comparator.add_argument(name, type=Path)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    if arguments.get("threads", 1) < 1:
        parser.error("--threads must be positive")
    (prepare if command == "prepare" else compare)(**arguments)
