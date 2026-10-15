"""One exact-affine UINT4 feasibility family; never reads or converts a checkpoint."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("verify_probe", ROOT / "experiments/windows/verify_probe.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
LIMIT = 128 * 1024 * 1024
ATOL, RTOL = 0.002, 0.002


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reconstruct(packed, scales, biases, rows, columns):
    """Independent scalar control: low nibble first, affine groups of 64."""
    if columns % 64 or len(packed) != rows * columns // 2:
        raise ValueError("invalid affine matrix")
    if len(scales) != rows * columns // 64 or len(biases) != len(scales):
        raise ValueError("invalid affine metadata")
    return [f32(f32(((packed[i // 2] >> (4 * (i % 2))) & 15) * scales[i // 64])
                + biases[i // 64]) for i in range(rows * columns)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def float_bytes(values):
    return struct.pack(f"<{len(values)}f", *values)


def read_floats(path):
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError("truncated Float32 output")
    return struct.unpack(f"<{len(data) // 4}f", data)


def build_case(destination, rows, columns, frames, count, expose_weights=False,
               dependent_chain=False, external_model=None):
    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as h

    destination.mkdir()
    inputs = ((np.arange(frames * columns, dtype=np.int32).reshape(frames, columns) % 31) - 15)
    inputs = (inputs / 64).astype("<f4")
    (destination / "input.f32").write_bytes(inputs.tobytes())
    tensors, nodes, outputs, references, matrices = [], [], [], [], []

    def tensor(name, dtype, shape, data):
        tensors.append(T(name=name, data_type=dtype, dims=shape, raw_data=data))

    tensor("unit_scale", T.FLOAT, [], float_bytes([1]))
    tensor("zero", T.UINT4, [], b"\0")
    groups = columns // 64
    tensor("group_shape", T.INT64, [3], struct.pack("<3q", rows, groups, 64))
    tensor("matrix_shape", T.INT64, [2], struct.pack("<2q", rows, columns))
    if dependent_chain:
        if rows != columns:
            raise ValueError("dependent products require square matrices")
        tensor("chain_normalizer", T.FLOAT, [], float_bytes([1 / 256]))
    active_inputs = inputs
    for index in range(count):
        rng = random.Random(71937 + index)
        packed = rng.randbytes(rows * columns // 2)
        # These dyadic values are exactly representable BF16 values, including
        # offsets that cannot be encoded as integer quantization zero points.
        scales = [(-1 if i % 2 else 1) * rng.randrange(1, 256) / 4096
                  for i in range(rows * groups)]
        biases = [-rng.randrange(1, 256) / 2048 for _ in scales]
        prefix = f"matrix{index}"
        tensor(prefix + "_q4", T.UINT4, [rows, columns], packed)
        tensor(prefix + "_scale", T.FLOAT, [rows, groups, 1], float_bytes(scales))
        tensor(prefix + "_offset", T.FLOAT, [rows, groups, 1], float_bytes(biases))
        for op, args, name, attrs in [
            ("DequantizeLinear", [prefix + "_q4", "unit_scale", "zero"], "nibbles", {}),
            ("Reshape", [prefix + "_nibbles", "group_shape"], "groups", {}),
            ("Mul", [prefix + "_groups", prefix + "_scale"], "scaled", {}),
            ("Add", [prefix + "_scaled", prefix + "_offset"], "affine", {}),
            ("Reshape", [prefix + "_affine", "matrix_shape"], "weights", {}),
            ("Transpose", [prefix + "_weights"], "transposed", {"perm": [1, 0]}),
            ("MatMul", ["input" if not dependent_chain or index == 0 else f"matrix{index - 1}_next",
                        prefix + "_transposed"], "product", {}),
        ]:
            nodes.append(h.make_node(op, args, [prefix + "_" + name], name=prefix + "_" + name, **attrs))
        if dependent_chain and index + 1 < count:
            nodes.append(h.make_node("Mul", [prefix + "_product", "chain_normalizer"],
                                     [prefix + "_next"], name=prefix + "_next"))

        # Decode in bounded row blocks; no expanded matrix is saved or retained.
        reference = np.empty((frames, rows), dtype="<f4")
        for start in range(0, rows, 128):
            stop = min(start + 128, rows)
            raw = np.frombuffer(packed[start * columns // 2:stop * columns // 2], dtype=np.uint8)
            nibbles = np.stack((raw & 15, raw >> 4), axis=1).reshape(stop - start, groups, 64)
            scale = np.asarray(scales[start * groups:stop * groups], dtype=np.float32).reshape(-1, groups, 1)
            bias = np.asarray(biases[start * groups:stop * groups], dtype=np.float32).reshape(-1, groups, 1)
            decoded = (nibbles.astype(np.float32) * scale + bias).reshape(stop - start, columns)
            reference[:, start:stop] = active_inputs @ decoded.T
        if dependent_chain:
            active_inputs = reference * np.float32(1 / 256)
        name = prefix + "_product"
        filename = name + ".f32"
        (destination / filename).write_bytes(reference.tobytes())
        outputs.append(h.make_tensor_value_info(name, T.FLOAT,
                                               ["frames" if dependent_chain else frames, rows]))
        references.append(dict(name=name, shape=[frames, rows], file=filename, exact=False))
        if expose_weights:
            name = prefix + "_weights"
            filename = name + ".f32"
            (destination / filename).write_bytes(float_bytes(reconstruct(packed, scales, biases, rows, columns)))
            outputs.append(h.make_tensor_value_info(name, T.FLOAT, [rows, columns]))
            references.append(dict(name=name, shape=[rows, columns], file=filename, exact=True))
        matrices.append(dict(q4_sha256=hashlib.sha256(packed).hexdigest(), packed_bytes=len(packed),
                             affine_metadata_bytes=len(scales) * 8, expanded_bytes=rows * columns * 4))
    model = h.make_model(h.make_graph(nodes, "exact_affine_uint4", [
        h.make_tensor_value_info("input", T.FLOAT,
                                 ["frames" if dependent_chain else frames, columns])], outputs, tensors),
        opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(model)
    if external_model is not None:
        onnx.external_data_helper.convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location="weights.bin", size_threshold=1024)
    model_path = external_model or destination / "model.onnx"
    onnx.save(model, model_path)
    write_json(destination / "case.json", dict(model="../" + model_path.name if external_model else "model.onnx",
        model_files={"../weights.bin": verifier.sha256(model_path.parent / "weights.bin")} if external_model else {},
        inputs=[dict(name="input", dtype="f32", shape=[frames, columns], path="input.f32")],
        repeat=2, disable_optimizations=True, require_dml_only=True, disable_memory_arena=True, idle_ms=2000))
    write_json(destination / "reference.json", dict(outputs=references, atol=ATOL, rtol=RTOL, matrices=matrices))


def independent_partition(nodes, initializer_names):
    """Preserve every node byte; hoist its complete input-independent closure."""
    known = set(initializer_names)
    independent, dependent = [], []
    for node in nodes:
        if all(name in known for name in node.input if name):
            independent.append(node)
            known.update(node.output)
        else:
            dependent.append(node)
    return independent + dependent


def weight_liveness(nodes, matrix_bytes):
    """Replay ORT's default reverse-DFS node-index ordering, before inference."""
    producers = {name: i for i, node in enumerate(nodes) for name in node.output}
    users = {name: set() for name in producers}
    for i, node in enumerate(nodes):
        for name in node.input:
            if name in users:
                users[name].add(i)
    parents = [{producers[name] for name in node.input if name in producers} for node in nodes]
    leaves = [i for i, node in enumerate(nodes) if not any(users[name] for name in node.output)]
    stack, visited, order = [(i, False) for i in leaves], set(), []
    while stack:
        index, leaving = stack.pop()
        if leaving:
            order.append(index)
        elif index not in visited:
            visited.add(index)
            stack.append((index, True))
            stack.extend((parent, False) for parent in sorted(parents[index]) if parent not in visited)
    if len(order) != len(nodes):
        raise ValueError("graph order is incomplete")
    remaining = {name: len(users[name]) for name in matrix_bytes}
    live, peak = {}, 0
    for index in order:
        node = nodes[index]
        for name in node.output:
            if name in matrix_bytes:
                live[name] = matrix_bytes[name]
        peak = max(peak, sum(live.values()))
        for name in set(node.input):
            if name in remaining:
                remaining[name] -= 1
                if remaining[name] == 0:
                    live.pop(name, None)
    return peak


def prepare_scheduling(destination):
    import onnx

    destination = Path(destination)
    destination.mkdir()
    build_case(destination / "interleaved", 3072, 3072, 114, 8,
               dependent_chain=True, external_model=destination / "interleaved.onnx")
    model = onnx.load(destination / "interleaved.onnx", load_external_data=False)
    original_nodes = list(model.graph.node)
    ordered_nodes = independent_partition(original_nodes, [tensor.name for tensor in model.graph.initializer])
    if sorted(node.SerializeToString() for node in original_nodes) != sorted(node.SerializeToString() for node in ordered_nodes):
        raise ValueError("ordering changed a node")
    matrix_bytes = {f"matrix{i}_transposed": 3072 * 3072 * 4 for i in range(8)}
    liveness = {"interleaved": weight_liveness(original_nodes, matrix_bytes),
                "hoisted": weight_liveness(ordered_nodes, matrix_bytes)}
    if liveness != {"interleaved": 8 * 36 * 1024 * 1024, "hoisted": 36 * 1024 * 1024}:
        raise ValueError("pair does not express the declared weight-lifetime contrast")
    del model.graph.node[:]
    model.graph.node.extend(ordered_nodes)
    onnx.save(model, destination / "hoisted.onnx")
    (destination / "hoisted").mkdir()
    (destination / "reference").mkdir()
    reference = json.loads((destination / "interleaved/reference.json").read_text())
    for output in reference["outputs"]:
        filename = output["file"]
        (destination / "interleaved" / filename).rename(destination / "reference" / filename)
        output["file"] = "../reference/" + filename
    (destination / "interleaved/input.f32").rename(destination / "input.f32")
    case = json.loads((destination / "interleaved/case.json").read_text())
    case.update(disable_graph_fusion=True, idle_ms=1500)
    case["inputs"][0]["path"] = "../input.f32"
    for name in ("interleaved", "hoisted"):
        case["model"] = f"../{name}.onnx"
        write_json(destination / name / "case.json", case)
        write_json(destination / name / "reference.json", reference)
    files = {str(p.relative_to(destination)).replace("\\", "/"):
             dict(size=p.stat().st_size, sha256=verifier.sha256(p))
             for p in destination.rglob("*") if p.is_file()}
    total = sum(v["size"] for v in files.values())
    if total > LIMIT:
        raise ValueError("scheduling pair exceeds 128 MiB")
    write_json(destination / "family.json", dict(files=files, bytes=total,
        kind="dependent-order-pair-v1", script_sha256=verifier.sha256(__file__),
        expanded_matrix_bytes=36 * 1024 * 1024, static_weight_liveness_bytes=liveness,
        minimum_peak_commit_reduction_bytes=108 * 1024 * 1024,
        maximum_loaded_commit_difference_bytes=36 * 1024 * 1024,
        private_commit_limit_bytes=2 * 1024 ** 3,
        purpose="ordering feasibility only; no GPU peak or full-model qualification"))


def prepare(destination):
    destination = Path(destination)
    destination.mkdir()
    build_case(destination / "correctness", 4, 128, 3, 1, expose_weights=True)
    build_case(destination / "one", 4096, 4096, 114, 1)
    build_case(destination / "eight", 4096, 4096, 114, 8)
    files = {str(p.relative_to(destination)).replace("\\", "/"): dict(size=p.stat().st_size, sha256=verifier.sha256(p))
             for p in destination.rglob("*") if p.is_file()}
    total = sum(v["size"] for v in files.values())
    if total > LIMIT:
        raise ValueError("prepared family exceeds the fixed 128 MiB budget")
    write_json(destination / "family.json", dict(files=files, bytes=total, matrix_counts=[1, 8],
        purpose="feasibility only; memory receipts required separately", script_sha256=verifier.sha256(__file__)))


def compare_outputs(case_directory, result_directory, record):
    import math

    reference = json.loads((case_directory / "reference.json").read_text())
    expected = reference["outputs"]
    records = record["outputs"]
    if len(records) != 2 * len(expected):
        raise ValueError("missing, extra, or duplicate native outputs")
    seen, first, errors = set(), {}, {}
    for output in records:
        key = (output["run"], output["name"])
        match = next((v for v in expected if v["name"] == output["name"]), None)
        if key in seen or output["run"] not in (0, 1) or match is None:
            raise ValueError("unexpected or duplicate output")
        seen.add(key)
        if output["shape"] != match["shape"] or output["dtype"] != "f32":
            raise ValueError("native output shape or dtype differs")
        path = result_directory / output["file"]
        if path.parent != result_directory or verifier.sha256(path) != output["sha256"]:
            raise ValueError("native output path or hash differs")
        data = path.read_bytes()
        control = (case_directory / match["file"]).read_bytes()
        if len(data) != len(control):
            raise ValueError("native output length differs")
        values, references = read_floats(path), read_floats(case_directory / match["file"])
        if not all(math.isfinite(v) for v in values + references):
            raise ValueError("nonfinite native output or reference")
        if match["exact"]:
            if data != control:
                raise ValueError("affine reconstructed Float32 bits differ")
        elif any(abs(a - b) > reference["atol"] + reference["rtol"] * abs(b)
                 for a, b in zip(values, references)):
            raise ValueError("matrix product exceeds existing 0.002 absolute/relative bounds")
        errors[output["name"]] = max(errors.get(output["name"], 0),
                                     max((abs(a - b) for a, b in zip(values, references)), default=0))
        if output["name"] in first and first[output["name"]] != data:
            raise ValueError("native repeats differ")
        first[output["name"]] = data
    return errors


def check_private_commit_observations(memory, limit):
    for value in memory:
        if (value["source"] != "GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX" or
                not 0 < value["current_bytes"] <= value["lifetime_peak_bytes"]):
            raise ValueError("missing or invalid native private-commit observation")
        if value["lifetime_peak_bytes"] > limit:
            raise ValueError("native lifetime private commit exceeded the per-case 2 GiB guard")
    if any(a["lifetime_peak_bytes"] > b["lifetime_peak_bytes"] for a, b in zip(memory, memory[1:])):
        raise ValueError("native lifetime private-commit peak went backwards")


def compare(family_directory, result_directory, selected=None, scheduling=False):
    family_directory, result_directory = Path(family_directory), Path(result_directory)
    family = json.loads((family_directory / "family.json").read_text())
    for name, pin in family["files"].items():
        path = family_directory / name
        if path.stat().st_size != pin["size"] or verifier.sha256(path) != pin["sha256"]:
            raise ValueError("prepared family identity differs")
    reports = {}
    names = ("interleaved", "hoisted") if scheduling else ("correctness", "one", "eight")
    idle_ms = 1500 if scheduling else 2000
    for name in ((selected,) if selected else names):
        case, result = family_directory / name, result_directory / name
        record = verifier.verify(case, result, "dml")
        if (record.get("require_dml_only") is not True or
                record.get("memory_arena_disabled") is not True or
                record.get("graph_optimizations_disabled") is not True or
                record.get("idle_ms") != idle_ms or
                (scheduling and record.get("dml_graph_fusion_disabled") is not True) or
                set(record["executed_provider_events"]) != {"DmlExecutionProvider"} or
                len(record["run_ms"]) != 2):
            raise ValueError("strict GPU execution or run policy differs")
        reports[name] = dict(maximum_errors=compare_outputs(case, result, record),
                             load_ms=record["load_ms"], cold_ms=record["run_ms"][0], warm_ms=record["run_ms"][1])
        previous = None
        for phase in ("baseline", "loaded", "warm"):
            receipt = json.loads((result / f"phase-{phase}.json").read_text())
            if (receipt["phase"] != phase or receipt["idle_ms"] != idle_ms or
                    receipt["pid"] != record["pid"] or record["pid"] <= 0 or
                    (previous is not None and receipt["unix_ms"] < previous + idle_ms)):
                raise ValueError("missing native idle phase receipt")
            previous = receipt["unix_ms"]
        if scheduling:
            phases = []
            for run in (0, 1):
                for stage in ("start", "complete"):
                    phase = json.loads((result / f"phase-run-{run}-{stage}.json").read_text())
                    if (phase["phase"] != f"run-{stage}" or phase["run"] != run or
                            phase["pid"] != record["pid"] or
                            phase["model_run_returned"] is not (stage == "complete")):
                        raise ValueError("native run phase identity differs")
                    phases.append(phase)
            if (phases[0]["unix_ms"] < json.loads((result / "phase-loaded.json").read_text())["unix_ms"] + idle_ms or
                    any(a["unix_ms"] > b["unix_ms"] for a, b in zip(phases, phases[1:])) or
                    phases[-1]["unix_ms"] > json.loads((result / "phase-warm.json").read_text())["unix_ms"]):
                raise ValueError("native run phase order differs")
            memory = [phase["private_commit"] for phase in phases] + [record["private_commit"]]
            check_private_commit_observations(memory, family["private_commit_limit_bytes"])
            reports[name].update(private_commit=record["private_commit"],
                before_run=memory[0], run_complete_peak_bytes=max(memory[i]["lifetime_peak_bytes"] for i in (1, 3)))
    report = dict(numerical_and_provider_gate="passed", cases=reports,
                residency_gate="UNDECIDED: separately inspect owned-process phase memory slope and peak",
                product_qualification="not run")
    if scheduling and not selected:
        report["scheduling_private_commit_gate"] = scheduling_commit_gate(family, reports)
        report["gpu_peak"] = "unmeasured; process private commit is not GPU peak"
    return report


def scheduling_commit_gate(family, reports):
    first, second = reports["interleaved"], reports["hoisted"]
    reduction = first["private_commit"]["lifetime_peak_bytes"] - second["private_commit"]["lifetime_peak_bytes"]
    reasons = []
    if any(value["private_commit"]["lifetime_peak_bytes"] > family["private_commit_limit_bytes"]
           for value in (first, second)):
        reasons.append("lifetime process private commit exceeded the 2 GiB guard")
    if any(abs(first["before_run"][key] - second["before_run"][key]) >
           family["maximum_loaded_commit_difference_bytes"] for key in ("current_bytes", "lifetime_peak_bytes")):
        reasons.append("loaded private-commit baselines differ by more than one expanded matrix")
    if any(value["run_complete_peak_bytes"] <= value["before_run"]["lifetime_peak_bytes"]
           for value in (first, second)):
        reasons.append("loader peak dominates; no distinct inference peak observed")
    if reduction < family["minimum_peak_commit_reduction_bytes"]:
        reasons.append("peak reduction is less than the predeclared three-matrix (108 MiB) contrast")
    return dict(status="passed" if not reasons else "not_demonstrated", reasons=reasons,
                peak_private_commit_reduction_bytes=reduction,
                claim="process private-commit scheduling contrast only; no GPU peak/residency attribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("destination", type=Path)
    command = sub.add_parser("prepare-scheduling")
    command.add_argument("destination", type=Path)
    command = sub.add_parser("compare")
    command.add_argument("family", type=Path)
    command.add_argument("results", type=Path)
    command.add_argument("--case", choices=("correctness", "one", "eight"))
    command = sub.add_parser("compare-scheduling")
    command.add_argument("family", type=Path)
    command.add_argument("results", type=Path)
    command.add_argument("--case", choices=("interleaved", "hoisted"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.destination)
    elif args.command == "prepare-scheduling":
        prepare_scheduling(args.destination)
    else:
        report = compare(args.family, args.results, args.case, args.command == "compare-scheduling")
        print(json.dumps(report, indent=2))
        if report.get("scheduling_private_commit_gate", {}).get("status", "passed") != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
