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


def build_case(destination, rows, columns, frames, count, expose_weights=False):
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
            ("MatMul", ["input", prefix + "_transposed"], "product", {}),
        ]:
            nodes.append(h.make_node(op, args, [prefix + "_" + name], name=prefix + "_" + name, **attrs))

        # Decode in bounded row blocks; no expanded matrix is saved or retained.
        reference = np.empty((frames, rows), dtype="<f4")
        for start in range(0, rows, 128):
            stop = min(start + 128, rows)
            raw = np.frombuffer(packed[start * columns // 2:stop * columns // 2], dtype=np.uint8)
            nibbles = np.stack((raw & 15, raw >> 4), axis=1).reshape(stop - start, groups, 64)
            scale = np.asarray(scales[start * groups:stop * groups], dtype=np.float32).reshape(-1, groups, 1)
            bias = np.asarray(biases[start * groups:stop * groups], dtype=np.float32).reshape(-1, groups, 1)
            decoded = (nibbles.astype(np.float32) * scale + bias).reshape(stop - start, columns)
            reference[:, start:stop] = inputs @ decoded.T
        name = prefix + "_product"
        filename = name + ".f32"
        (destination / filename).write_bytes(reference.tobytes())
        outputs.append(h.make_tensor_value_info(name, T.FLOAT, [frames, rows]))
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
        h.make_tensor_value_info("input", T.FLOAT, [frames, columns])], outputs, tensors),
        opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, destination / "model.onnx")
    write_json(destination / "case.json", dict(model="model.onnx", model_files={},
        inputs=[dict(name="input", dtype="f32", shape=[frames, columns], path="input.f32")],
        repeat=2, disable_optimizations=True, require_dml_only=True, disable_memory_arena=True, idle_ms=2000))
    write_json(destination / "reference.json", dict(outputs=references, atol=ATOL, rtol=RTOL, matrices=matrices))


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


def compare(family_directory, result_directory, selected=None):
    family_directory, result_directory = Path(family_directory), Path(result_directory)
    family = json.loads((family_directory / "family.json").read_text())
    for name, pin in family["files"].items():
        path = family_directory / name
        if path.stat().st_size != pin["size"] or verifier.sha256(path) != pin["sha256"]:
            raise ValueError("prepared family identity differs")
    reports = {}
    for name in ((selected,) if selected else ("correctness", "one", "eight")):
        case, result = family_directory / name, result_directory / name
        record = verifier.verify(case, result, "dml")
        if (record.get("require_dml_only") is not True or
                record.get("memory_arena_disabled") is not True or
                record.get("graph_optimizations_disabled") is not True or
                record.get("idle_ms") != 2000 or
                set(record["executed_provider_events"]) != {"DmlExecutionProvider"} or
                len(record["run_ms"]) != 2):
            raise ValueError("strict GPU execution or run policy differs")
        reports[name] = dict(maximum_errors=compare_outputs(case, result, record),
                             load_ms=record["load_ms"], cold_ms=record["run_ms"][0], warm_ms=record["run_ms"][1])
        previous = None
        for phase in ("baseline", "loaded", "warm"):
            receipt = json.loads((result / f"phase-{phase}.json").read_text())
            if (receipt["phase"] != phase or receipt["idle_ms"] != 2000 or
                    receipt["pid"] != record["pid"] or record["pid"] <= 0 or
                    (previous is not None and receipt["unix_ms"] < previous + 2000)):
                raise ValueError("missing native idle phase receipt")
            previous = receipt["unix_ms"]
    return dict(numerical_and_provider_gate="passed", cases=reports,
                residency_gate="UNDECIDED: separately inspect owned-process phase memory slope and peak",
                product_qualification="not run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("destination", type=Path)
    command = sub.add_parser("compare")
    command.add_argument("family", type=Path)
    command.add_argument("results", type=Path)
    command.add_argument("--case", choices=("correctness", "one", "eight"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.destination)
    else:
        print(json.dumps(compare(args.family, args.results, args.case), indent=2))


if __name__ == "__main__":
    main()
