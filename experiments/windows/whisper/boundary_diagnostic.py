"""Observe retained source boundaries in one exact frozen candidate graph."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper

from export import ROOT, sha256, value
from implicit_control import verify, difference
from qualify import score_report
from rounding_control import RUNTIME_SHA256, METALLIB_SHA256

CANDIDATE_SHA256 = "b117f7591ffb61f5c6781bf2655c8fc49cb0f2d0cfa50903c98447c58412905a"
COMPILED_GELU_SHA256 = "1c47965522f7495fc85814c70c6b17dc4952436774c43225d55665e4ecbc7f14"
FROZEN_CANDIDATES = [CANDIDATE_SHA256, COMPILED_GELU_SHA256]
RECEIPT_SHA256 = "ab5d7c4ca59a4f20a2174a16661312c1f945b41c2b8ec45ad34e15b67010c443"
STAGES = [
    ("conv1-pre", "Cast_10", "b98f3d64abc45644cff8136ea509a8883adf7c355f25a3a85e17b0886de58332"),
    ("conv1", "Cast_20", "33315b619e59ac149c6fd251c61af111d426c023e86f25dee96297e9be8e37c7"),
    ("conv2", "Cast_38", "2ab248309743f4596dc1c281322d8e122f966f963a7ec17084311c7afdc06a32"),
    ("added_positions", "Cast_42", "9a7309b2c76dff6a713659129a9c00a3dfdc400e326f04b7d0cd80f943a2cfc5"),
    ("encoder-0", "Cast_116", "233d15731020ee8fbc66a764334570a13369e07fccd1b73c8eefdb470495502f"),
    ("encoder-1", "Cast_190", "1273030123b6f392830e86db92426e5bd5858dabfdfe4ea2d2d79751f5d81a1a"),
    ("encoder-2", "Cast_264", "a2531829404315d77376c1a79e2c5fa1c5bab9fd43e5e269c7e1938a97ba34d3"),
    ("encoder-3", "Cast_338", "7064c5655b4edf9bf1d83e265d4a9faf21020502fbe9f54e414bf56486320345"),
    ("encoded", "Cast_341", "1789dcac04a422aaee7280786aa15fd444294673b80448338a44ecc7b23c5860"),
    ("decoder-0", "Cast_466", "f50178a14eb716b750fc0d71813af4c4f6df6ccc12c6bcef97b5f637a72767f4"),
    ("decoder-1", "Cast_590", "c7a7f0879af3b1153d7b5447b3ba58104f1c9595d700c3fb5ad0da29916e3fe7"),
    ("decoder-2", "Cast_714", "7c2cc6f44ebbde48c2fb739e4deeaa0801fe1538fa9f1d5b428345d8190b5931"),
    ("decoder-3", "Cast_838", "3ecb7eec7effb638000be7b955197bdb03892654b9dbbd10ad017fb49f1c410b"),
    ("decoded", "Cast_841", "70a34a6bc89468d5b1213ea52c53ac885f8211ba9b0f8686d01bfcc2c60066b9"),
    ("logits", "Cast_845", "ccf9757617e2f17da50a4cbb2a16f7b544dbc08ae0985f740505aed2d6e76ba6"),
    ("probabilities", "probabilities", "9017a34360c5519ca81e70a4b687a3005d007add3651021661dbcc8b7644e5d7"),
]


def prepare(candidate, capture, output, candidate_sha256=CANDIDATE_SHA256):
    if (candidate_sha256 not in FROZEN_CANDIDATES or sha256(candidate / "whisper.onnx") != candidate_sha256
            or sha256(capture / "receipt.json") != RECEIPT_SHA256):
        raise ValueError("frozen candidate or source runtime receipt differs")
    receipt = json.loads((capture / "receipt.json").read_text())
    if (receipt["runtime_sha256"] != RUNTIME_SHA256 or receipt["colocated_metallib_sha256"] != METALLIB_SHA256
            or receipt["independent_max_error"] != 0 or any(row["nonzero"] for row in receipt["comparisons"])):
        raise ValueError("source captures do not reproduce the pinned independent silence baseline")
    original_case = candidate / "cases/neural-silence"
    if sha256(original_case / "input.f32") != receipt["mel_sha256"]:
        raise ValueError("frozen silence input differs from source capture")
    model = onnx.load(candidate / "whisper.onnx")
    original_nodes = [node.SerializeToString() for node in model.graph.node]
    original_initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
    inferred = onnx.shape_inference.infer_shapes(model)
    info = {item.name: item for item in [*inferred.graph.value_info, *inferred.graph.output]}
    output.mkdir(parents=True, exist_ok=False)
    directory = output / "silence"
    directory.mkdir()
    stages, declarations, payload = {}, [], 0
    for stage, tensor, pin in STAGES:
        path = capture / (stage + ".npy")
        if sha256(path) != pin:
            raise ValueError(("retained source tensor identity differs", stage))
        array = np.load(path, allow_pickle=False)
        shape = [dimension.dim_value for dimension in info[tensor].type.tensor_type.shape.dim]
        if stage == "probabilities":
            array = array.reshape(99)
            name = stage
            declarations.append(copy.deepcopy(model.graph.output[0]))
        else:
            name = "observe_" + stage.replace("-", "_")
            if info[tensor].type.tensor_type.elem_type != TensorProto.FLOAT16:
                raise ValueError(("source boundary is not Half", stage))
            model.graph.node.append(helper.make_node("Cast", [tensor], [name], name=name, to=TensorProto.FLOAT))
            declarations.append(value(name, shape))
        if array.dtype != np.float32 or list(array.shape) != shape or not np.isfinite(array).all():
            raise ValueError(("retained boundary shape/type differs", stage, array.shape, shape))
        filename = stage + ".expected.f32"
        array.astype("<f4").tofile(directory / filename)
        payload += array.nbytes
        stages[name] = {"stage": stage, "tensor": tensor, "shape": shape, "file": filename,
                        "sha256": sha256(directory / filename), "source_npy_sha256": pin}
    if [node.SerializeToString() for node in model.graph.node[:len(original_nodes)]] != original_nodes:
        raise ValueError("the diagnostic changed original arithmetic")
    if [tensor.SerializeToString() for tensor in model.graph.initializer] != original_initializers:
        raise ValueError("the diagnostic changed an original parameter")
    del model.graph.output[:]
    model.graph.output.extend(declarations)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, directory / "boundaries.onnx")
    shutil.copyfile(original_case / "input.f32", directory / "input.f32")
    shutil.copyfile(original_case / "expected.json", directory / "original-gate.json")
    shutil.copyfile(original_case / "expected.f32", directory / "expected.f32")
    (directory / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")
    metadata = json.loads((original_case / "case.json").read_text())
    if not metadata.get("disable_optimizations") or metadata["repeat"] != 2:
        raise ValueError("source case lost its original execution settings")
    metadata["model"] = "boundaries.onnx"
    metadata["model_files"] = {name: sha256(directory / name) for name in
                               ["boundaries.onnx", "stages.json", "original-gate.json", "expected.f32"]}
    (directory / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if total >= 150_000_000:
        raise ValueError(("diagnostic exceeds the accepted size cap", total))
    provenance = {"diagnostic_only": True, "source_graph_sha256": candidate_sha256,
        "graph_sha256": sha256(directory / "boundaries.onnx"), "case_sha256": sha256(directory / "case.json"),
        "capture_receipt_sha256": RECEIPT_SHA256, "runtime_sha256": RUNTIME_SHA256, "metallib_sha256": METALLIB_SHA256,
        "independent_reference_sha256": sha256(ROOT / "crates/models/tests/data/whisper-reference.json"),
        "script_sha256": sha256(Path(__file__)), "original_node_count": len(original_nodes),
        "observers_added": len(STAGES) - 1, "original_nodes_initializers_unchanged": True,
        "expectation_payload_bytes": payload, "bundle_bytes_before_provenance": total,
        "original_gate_metadata_sha256": sha256(directory / "original-gate.json"),
        "case": "silence/case.json", "stage_order": [row[0] for row in STAGES],
        "observer_limit": "Adding outputs can change provider scheduling/fusion. Compare final probabilities with the existing frozen single-output native run to detect an observer effect. No arithmetic or source parameters are changed.",
        "gate_unchanged": "Original independent max probability error strictly <1e-4; stage exactness is diagnostic."}
    (output / "boundaries.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def half_distance(actual, expected):
    ah, eh = actual.astype(np.float16), expected.astype(np.float16)
    if np.any(ah.astype(np.float32) != actual) or np.any(eh.astype(np.float32) != expected):
        return {"non_half_values": int(np.count_nonzero(ah.astype(np.float32) != actual)), "max_half_ulp_distance": None}
    def order(x):
        bits = x.view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)
    return {"non_half_values": 0, "max_half_ulp_distance": int(np.max(abs(order(ah) - order(eh)), initial=0))}


def baseline_probabilities(case, result, provider, candidate_sha256=CANDIDATE_SHA256):
    record = verify(case, result, provider)
    if record["model_sha256"] != candidate_sha256:
        raise ValueError("baseline is not the frozen single-output candidate")
    rows = record["outputs"]
    if len(rows) != 2 or {(row["run"], row["name"]) for row in rows} != {(run, "probabilities") for run in [0, 1]}:
        raise ValueError("single-output baseline inventory differs")
    outputs = {}
    for row in rows:
        path = result / row["file"]
        if row["dtype"] != "f32" or row["shape"] != [99] or sha256(path) != row["sha256"]:
            raise ValueError("single-output baseline tensor differs")
        outputs[row["run"]] = np.fromfile(path, "<f4")
    return record, outputs


def compare(case, result, provider, baseline_case=None, baseline_result=None, candidate_sha256=CANDIDATE_SHA256):
    if candidate_sha256 not in FROZEN_CANDIDATES:
        raise ValueError("unknown frozen candidate")
    if (baseline_case is None) != (baseline_result is None):
        raise ValueError("baseline case and result must be supplied together")
    if candidate_sha256 == COMPILED_GELU_SHA256 and baseline_case is None:
        raise ValueError("compiled-GELU diagnosis requires its frozen native baseline")
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("boundary diagnostic used CPU model work")
    stages = json.loads((case / "stages.json").read_text())
    if len(record["outputs"]) != len(stages) * 2 or {(row["run"], row["name"]) for row in record["outputs"]} != {
            (run, name) for run in [0, 1] for name in stages}:
        raise ValueError("boundary output inventory differs")
    baseline = None
    if baseline_case is not None:
        baseline_record, baseline = baseline_probabilities(baseline_case, baseline_result, provider, candidate_sha256)
        if baseline_record["inputs"] != record["inputs"]:
            raise ValueError("baseline and diagnostic inputs differ")
    original = json.loads((case / "original-gate.json").read_text())["independent"]
    first, reports, deterministic, observer_equivalent = {}, [], True, True
    for row in sorted(record["outputs"], key=lambda item: (item["run"], list(stages).index(item["name"]))):
        spec = stages[row["name"]]
        if sha256(case / spec["file"]) != spec["sha256"]:
            raise ValueError("retained source boundary identity differs")
        if row["dtype"] != "f32" or row["shape"] != spec["shape"] or sha256(result / row["file"]) != row["sha256"]:
            raise ValueError("native boundary identity differs")
        actual = np.fromfile(result / row["file"], "<f4").reshape(spec["shape"])
        expected = np.fromfile(case / spec["file"], "<f4").reshape(spec["shape"])
        first.setdefault(row["name"], actual)
        repeat = difference(actual, first[row["name"]])["nonzero"]
        deterministic &= repeat == 0
        report = {"run": row["run"], "stage": spec["stage"], **difference(actual, expected), "repeat_nonzero": repeat}
        if spec["stage"] != "probabilities":
            report.update(half_distance(actual, expected))
        else:
            score = score_report(actual, original["probabilities"], original["codes"])
            route_equal = all(score["actual_route"][key] == score["expected_route"][key] for key in ["code", "confident"])
            report["original_language_gate"] = {**score, "passed": score["max_probability_error"] < 1e-4 and route_equal}
            report["observer_vs_frozen_native"] = None if baseline is None else difference(actual, baseline[row["run"]])
            if baseline is not None:
                bits = int(np.count_nonzero(actual.view(np.uint32) != baseline[row["run"]].view(np.uint32)))
                report["observer_vs_frozen_native"]["bit_differences"] = bits
                observer_equivalent &= bits == 0
        reports.append(report)
    earliest = next((row["stage"] for row in reports if row["run"] == 0 and row["nonzero"]), None)
    print(json.dumps({"diagnostic_only": True, "deterministic": bool(deterministic),
                      "observer_equivalent": None if baseline is None else bool(observer_equivalent),
                      "earliest_observed_source_difference": earliest, "reports": reports}, indent=2))
    return deterministic and observer_equivalent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["candidate", "capture", "output"]:
        build.add_argument(name, type=Path)
    build.add_argument("--candidate-sha256", choices=FROZEN_CANDIDATES, default=CANDIDATE_SHA256)
    comparison = commands.add_parser("compare")
    for name in ["case", "result"]:
        comparison.add_argument(name, type=Path)
    comparison.add_argument("--provider", choices=["cpu", "dml"], required=True)
    comparison.add_argument("--baseline-case", type=Path)
    comparison.add_argument("--baseline-result", type=Path)
    comparison.add_argument("--candidate-sha256", choices=FROZEN_CANDIDATES, default=CANDIDATE_SHA256)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    else:
        raise SystemExit(0 if compare(**args) else 1)
