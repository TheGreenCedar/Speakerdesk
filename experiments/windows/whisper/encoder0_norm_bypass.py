"""One paired encoder-0 diagnostic; source first norm is the only intervention."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from export import ROOT, sha256, value
from boundary_diagnostic import COMPILED_GELU_SHA256, STAGES, half_distance
from implicit_control import EXTRA_CAPTURES, verify, difference
from materialization_candidate import SourceGraph

FROZEN_OBSERVERS_SHA256 = "9c30bc5f07955ed14132523a5f87373c432bb08ac5422445b5004f29a722e3bb"
PINS = {name: pin for name, _, pin in STAGES}


def prefix_model(source):
    needed, retained = {"Cast_116"}, []
    for node in reversed(source.graph.node):
        if "Cast_42" in node.output:
            continue
        if needed.intersection(node.output):
            retained.append(copy.deepcopy(node))
            needed.difference_update(node.output)
            needed.update(node.input)
    retained.reverse()
    constants = [copy.deepcopy(tensor) for tensor in source.graph.initializer if tensor.name in needed]
    if needed - {tensor.name for tensor in constants} != {"Cast_42"}:
        raise ValueError("encoder-0 has an unexpected external input")
    inferred = onnx.shape_inference.infer_shapes(source)
    info = {item.name: item for item in [*inferred.graph.value_info, *inferred.graph.output]}
    shape = [dim.dim_value for dim in info["Cast_42"].type.tensor_type.shape.dim]
    if any(info[name].type.tensor_type.elem_type != TensorProto.FLOAT16
           for name in ["Cast_42", "LayerNormalization_43", "Cast_116"]):
        raise ValueError("the intervention boundaries must be Half")
    if [dim.dim_value for dim in info["Cast_116"].type.tensor_type.shape.dim] != shape:
        raise ValueError("encoder-0 state shape differs")
    g = SourceGraph("speakerdesk_encoder0_bypass_")
    selected = g.op("Where", g.op("Equal", "bypass", g.const(np.asarray(1, np.int64))),
                    g.op("Cast", "source_first_norm", to=TensorProto.FLOAT16), "LayerNormalization_43")
    nodes = [helper.make_node("Cast", ["positions"], ["Cast_42"],
                              name="speakerdesk_encoder0_positions", to=TensorProto.FLOAT16)]
    changed = []
    for original in retained:
        node = copy.deepcopy(original)
        if "LayerNormalization_43" in node.input:
            if node.output[0] != "Cast_44":
                raise ValueError("first norm has an unexpected consumer")
            node.input[:] = [selected if name == "LayerNormalization_43" else name for name in node.input]
            changed.append(node.output[0])
        nodes.append(node)
        if "LayerNormalization_43" in node.output:
            nodes.extend(g.nodes)
    if changed != ["Cast_44"]:
        raise ValueError("expected one first-norm consumer")
    nodes.extend([helper.make_node("Cast", [selected], ["normalized"], name="observe_normalized", to=TensorProto.FLOAT),
                  helper.make_node("Cast", ["Cast_116"], ["encoder0"], name="observe_encoder0", to=TensorProto.FLOAT)])
    model = helper.make_model(helper.make_graph(nodes, "Whisper-encoder0-first-norm-intervention",
        [value("positions", shape), value("source_first_norm", shape), value("bypass", [], TensorProto.INT64)],
        [value("normalized", shape), value("encoder0", shape)], [*constants, *g.initializers]),
        opset_imports=source.opset_import, ir_version=source.ir_version)
    actual = {node.output[0]: node for node in model.graph.node}
    for original in retained:
        node = copy.deepcopy(actual[original.output[0]])
        if node.output[0] == "Cast_44":
            node.input[:] = original.input
        if node.SerializeToString() != original.SerializeToString():
            raise ValueError("an encoder arithmetic node changed")
    original_constants = {tensor.name: tensor.SerializeToString() for tensor in source.graph.initializer}
    if any(tensor.SerializeToString() != original_constants[tensor.name] for tensor in constants):
        raise ValueError("an encoder parameter changed")
    onnx.checker.check_model(model, full_check=True)
    return model, {"retained_nodes": len(retained), "retained_initializers": len(constants),
                   "changed_consumer": "Cast_44", "input_shape": shape,
                   "arithmetic_unchanged": True, "retained_parameter_bytes_unchanged": True}


def prepare(candidate, retained, extra, output):
    if sha256(candidate / "whisper.onnx") != COMPILED_GELU_SHA256:
        raise ValueError("expected the frozen compiled-GELU candidate")
    arrays = {}
    for name, path, pin in [("positions", retained / "added_positions.npy", PINS["added_positions"]),
                            ("source_first_norm", extra / "first_norm.npy", EXTRA_CAPTURES["first_norm"]),
                            ("source_encoder0", retained / "encoder-0.npy", PINS["encoder-0"])]:
        if sha256(path) != pin:
            raise ValueError(("retained source boundary identity differs", name))
        array = np.load(path, allow_pickle=False)
        if (array.dtype != np.float32 or array.shape != (1, 1500, 384) or not np.isfinite(array).all()
                or np.any(array != array.astype(np.float16).astype(np.float32))):
            raise ValueError("source intervention array is not finite Half-valued state")
        arrays[name] = array
    model, checks = prefix_model(onnx.load(candidate / "whisper.onnx"))
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "encoder0.onnx")
    for name, array in arrays.items():
        array.astype("<f4").tofile(output / (name + ".f32"))
    expected = {"shape": [1, 1500, 384], "positions_sha256": sha256(output / "positions.f32"),
                "normalized": {"file": "source_first_norm.f32", "sha256": sha256(output / "source_first_norm.f32")},
                "encoder0": {"file": "source_encoder0.f32", "sha256": sha256(output / "source_encoder0.f32")}}
    (output / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    cases = []
    for mode, flag in [("baseline", 0), ("bypass", 1)]:
        case = output / mode
        case.mkdir()
        np.asarray(flag, "<i8").tofile(case / "choice.i64")
        metadata = {"model": "../encoder0.onnx", "repeat": 2, "disable_optimizations": True,
            "model_files": {"../encoder0.onnx": sha256(output / "encoder0.onnx"), "../expected.json": sha256(output / "expected.json"),
                            "choice.i64": sha256(case / "choice.i64")},
            "inputs": [{"name": name, "dtype": "f32", "shape": [1, 1500, 384], "path": "../" + name + ".f32"}
                       for name in ["positions", "source_first_norm"]] +
                      [{"name": "bypass", "dtype": "i64", "shape": [], "path": "choice.i64"}]}
        (case / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append({"case": mode + "/case.json", "sha256": sha256(case / "case.json")})
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if total >= 50_000_000:
        raise ValueError(("encoder-0 diagnostic exceeds accepted size cap", total))
    receipt = {"diagnostic_only": True, "source_graph_sha256": COMPILED_GELU_SHA256,
        "graph_sha256": sha256(output / "encoder0.onnx"), "script_sha256": sha256(Path(__file__)),
        "frozen_native_observer_graph_sha256": FROZEN_OBSERVERS_SHA256, "cases": cases,
        "retained_source_pins": {"positions": PINS["added_positions"], "first_norm": EXTRA_CAPTURES["first_norm"], "encoder0": PINS["encoder-0"]},
        "bundle_bytes_before_receipt": total, **checks,
        "rule": "Identical exact source positions and original encoder weights. Baseline uses computed norm; bypass substitutes the retained exact first norm at one consumer only. Original arithmetic remains in both modes.",
        "interpretation": "Require baseline encoder0 bits equal to the existing frozen native observer in both repeats before interpreting source differences under the norm bypass. This is a causal diagnostic, never a model candidate or relaxed gate."}
    (output / "norm-bypass.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def checked_outputs(case, result, provider):
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("encoder norm intervention used CPU model work")
    outputs = {}
    for row in record["outputs"]:
        path = result / row["file"]
        if row["dtype"] != "f32" or row["shape"] != [1, 1500, 384] or sha256(path) != row["sha256"]:
            raise ValueError("encoder norm intervention output identity differs")
        outputs[(row["run"], row["name"])] = np.fromfile(path, "<f4").reshape(1, 1500, 384)
    if set(outputs) != {(run, name) for run in [0, 1] for name in ["normalized", "encoder0"]}:
        raise ValueError("encoder intervention output inventory differs")
    return record, outputs


def compare(artifact, baseline, bypass, provider, frozen_case, frozen_result):
    expected = json.loads((artifact / "expected.json").read_text())
    if sha256(artifact / "positions.f32") != expected["positions_sha256"]:
        raise ValueError("exact source positions differ")
    references = {}
    for name in ["normalized", "encoder0"]:
        spec = expected[name]
        if sha256(artifact / spec["file"]) != spec["sha256"]:
            raise ValueError("retained intervention reference identity differs")
        references[name] = np.fromfile(artifact / spec["file"], "<f4").reshape(expected["shape"])
    frozen = verify(frozen_case, frozen_result, provider)
    if frozen["model_sha256"] != FROZEN_OBSERVERS_SHA256:
        raise ValueError("baseline observer is not the exact frozen compiled-GELU graph")
    if provider == "dml" and set(frozen["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("frozen observer used CPU model work")
    frozen_outputs = {}
    for row in frozen["outputs"]:
        if row["name"] == "observe_encoder_0":
            path = frozen_result / row["file"]
            if row["dtype"] != "f32" or row["shape"] != expected["shape"] or sha256(path) != row["sha256"]:
                raise ValueError("frozen encoder-0 output identity differs")
            frozen_outputs[row["run"]] = np.fromfile(path, "<f4").reshape(expected["shape"])
    if set(frozen_outputs) != {0, 1}:
        raise ValueError("frozen encoder-0 repeats missing")
    records, reports, passed = {}, [], True
    for mode, result in [("baseline", baseline), ("bypass", bypass)]:
        if (artifact / mode / "choice.i64").read_bytes() != np.asarray(mode == "bypass", "<i8").tobytes():
            raise ValueError("intervention mode input differs")
        record, outputs = checked_outputs(artifact / mode, result, provider)
        if record["model_sha256"] != sha256(artifact / "encoder0.onnx"):
            raise ValueError("paired intervention graph differs")
        records[mode] = outputs
        for (run, name), actual in outputs.items():
            repeat = int(np.count_nonzero(actual.view(np.uint32) != outputs[(0, name)].view(np.uint32)))
            report = {"mode": mode, "run": run, "name": name, **difference(actual, references[name]),
                      **half_distance(actual, references[name]), "repeat_bit_differences": repeat}
            passed &= repeat == 0
            if mode == "baseline" and name == "encoder0":
                bits = int(np.count_nonzero(actual.view(np.uint32) != frozen_outputs[run].view(np.uint32)))
                report["versus_frozen_native_bit_differences"] = bits
                passed &= bits == 0
            if mode == "bypass" and name == "normalized":
                bits = int(np.count_nonzero(actual.view(np.uint32) != references[name].view(np.uint32)))
                passed &= bits == 0
            reports.append(report)
    print(json.dumps({"diagnostic_only": True, "causal_intervention_admitted": bool(passed),
        "norm_bypass_encoder0_source_exact": bool(np.array_equal(records["bypass"][(0, "encoder0")], references["encoder0"])),
        "product_gate_unchanged": "Original strict max probability error <1e-4", "reports": reports}, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["candidate", "retained", "extra", "output"]:
        build.add_argument(name, type=Path)
    comparison = commands.add_parser("compare")
    for name in ["artifact", "baseline", "bypass"]:
        comparison.add_argument(name, type=Path)
    comparison.add_argument("--provider", choices=["cpu", "dml"], required=True)
    comparison.add_argument("--frozen-case", type=Path, required=True)
    comparison.add_argument("--frozen-result", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    else:
        raise SystemExit(0 if compare(**args) else 1)
