"""Rebind frozen Cohere graphs to their original affine UINT4 checkpoint weights."""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import struct

COPY_BYTES = 1 << 20
SHARD_BYTES = 1 << 30
ROOT = Path(__file__).resolve().parents[3]
# Immutable unsharded cohere-fp32-v2 diagnostics, never rewritten by this tool.
GRAPH_PINS = {
    "encoder.onnx": (2022217, "0c243acb5f78842b7573d9a1c46227b9436b7fb21986d03cc45a84bc9d57d9ad"),
    "encoder.weights": (7581848576, "6e8d004d81a1d870de6725665ce381ff4b36df207909ed228bddbd8bb0246100"),
    "decoder-init.onnx": (235472, "dfa9b651acd135de51cd6653b43646325d8a3819212ffee02ed49d4711a959b3"),
    "decoder-step.onnx": (194001, "04f35e5eb203917056598df0c5844922bc0ac808d10d441ea1d9e837dde323c5"),
    "decoder.weights": (614125568, "376f7fc1ca000955e0330e54a1407c0d7fad13b80cc3be7f61b80190452905bf"),
}
CHECKPOINT_PINS = {
    "model.safetensors": (1505759837, "ae947c13ba1cb8ce24c8bf72ded2520dcda1894ebf5a54fbb803c03bc28ef7fb"),
    "config.json": (1218, "c9f2934c33c22014a339d2451493c72f2fddff289ca83c62cd07c6502fcc092d"),
    "tokenizer.json": (1816694, "780ccca2de2ccd289971b1fb7d4f0b5ec2dc908872f6e350181c7eba9db3fa9f"),
}
REFERENCE_PIN = (1214, "93a871a53ebcb7a372a34f1496dca082a6c1b9e9b4518488f4da1ba838860c90")
# Canonical LF bytes; the original Windows receipt used CRLF (03672c204f198738...).
RETAINED_PINS_SHA256 = "fe7882c8eed28d87c40b348f16ecab42bba0335edc52d593356ac56aa22433c7"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def retained_pins(path=None):
    raw = (Path(path) if path else Path(__file__).with_name("compact-retained-pins.json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != RETAINED_PINS_SHA256:
        raise ValueError("frozen retained-case inventory hash differs")
    pins = json.loads(raw, object_pairs_hook=unique_object)
    if len(pins) != 119:
        raise ValueError("frozen retained-case inventory is incomplete")
    return pins


def admit_retained(handles, source, pins):
    opened = {}
    for name, pin in pins.items():
        if (name.startswith("/") or any(char in name for char in ("\\", ":"))
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("unsafe retained-case inventory path")
        opened[name] = pin_open(handles, source / name, pin)
    return opened


def safetensors_header(stream, file_bytes):
    stream.seek(0)
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated checkpoint header")
    length, = struct.unpack("<Q", prefix)
    if not 0 < length <= 16 * COPY_BYTES or 8 + length > file_bytes:
        raise ValueError("invalid checkpoint header size")
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("truncated checkpoint header")
    header = json.loads(raw, object_pairs_hook=unique_object)
    entries, intervals = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if set(entry) != {"dtype", "shape", "data_offsets"} or entry["dtype"] not in {"U32", "BF16", "F32"}:
            raise ValueError(f"unknown checkpoint tensor metadata: {name}")
        shape, offsets = entry["shape"], entry["data_offsets"]
        if (not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape)
                or len(offsets) != 2 or any(type(d) is not int or d < 0 for d in offsets)):
            raise ValueError(f"invalid checkpoint tensor range: {name}")
        start, stop = offsets
        width = 2 if entry["dtype"] == "BF16" else 4
        if stop - start != math.prod(shape) * width or stop + 8 + length > file_bytes:
            raise ValueError(f"checkpoint tensor length differs: {name}")
        entries[name] = dict(entry, offset=8 + length + start, length=stop - start)
        intervals.append((start, stop))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("checkpoint ranges overlap or contain a gap")
        end = stop
    if end + 8 + length != file_bytes:
        raise ValueError("checkpoint has unaccounted bytes")
    return entries


def affine_group(entries, name, shape):
    packed = entries[name]
    scales, biases = name.removesuffix(".weight") + ".scales", name.removesuffix(".weight") + ".biases"
    if (not name.endswith(".weight") or packed["dtype"] != "U32" or len(packed["shape"]) != 2
            or list(shape) != [packed["shape"][0], packed["shape"][1] * 8]
            or shape[1] % 64 or any(key not in entries for key in (scales, biases))):
        raise ValueError(f"unexpected affine parameter: {name}")
    for key in (scales, biases):
        if entries[key]["dtype"] != "BF16" or entries[key]["shape"] != [shape[0], shape[1] // 64]:
            raise ValueError(f"unexpected affine metadata: {key}")
    return scales, biases


def read_range(stream, offset, length):
    stream.seek(offset)
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("source tensor truncated during read")
    return data


def chunks(stream, entry):
    for offset in range(0, entry["length"], COPY_BYTES):
        yield read_range(stream, entry["offset"] + offset, min(COPY_BYTES, entry["length"] - offset))


def bf16_chunks(stream, entry):
    import numpy as np
    for data in chunks(stream, entry):
        # Bit widening is exact; no new quantization of scales or offsets.
        bits = np.frombuffer(data, dtype="<u2").astype("<u4") << 16
        values = bits.view("<f4")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite affine metadata")
        yield values.tobytes()


def source_aliases(stream, entries, used):
    """Admit only the known embedding/output tie, and only with identical bytes."""
    aliases = {}
    for alias, owner in [("proj_out", "decoder.embed_tokens"), ("decoder.embed_tokens", "proj_out")]:
        names = [prefix + suffix for prefix in (alias, owner) for suffix in (".weight", ".scales", ".biases")]
        missing, present = names[:3], names[3:]
        if set(missing).issubset(set(entries) - used) and set(present).issubset(used):
            for name, original in zip(missing, present):
                a, b = entries[name], entries[original]
                if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                    raise ValueError("tied source parameter shape/type differs")
                digest_a, digest_b = hashlib.sha256(), hashlib.sha256()
                for block in chunks(stream, a):
                    digest_a.update(block)
                for block in chunks(stream, b):
                    digest_b.update(block)
                if digest_a.digest() != digest_b.digest():
                    raise ValueError("tied source parameter bytes differ")
                aliases[name] = dict(owner=original, sha256=digest_a.hexdigest())
                used.add(name)
    if used != set(entries):
        raise ValueError(f"incomplete checkpoint coverage: {sorted(set(entries) - used)}")
    return aliases


def verify_affine(stream, entries, name, shape, original, original_range):
    """Compare every reconstructed FP32 byte in bounded row blocks; save none."""
    import numpy as np
    scale_name, bias_name = affine_group(entries, name, shape)
    rows, columns = shape
    groups = columns // 64
    if columns * 4 > COPY_BYTES:
        raise ValueError("one matrix row exceeds the bounded reconstruction buffer")
    for row in range(0, rows, max(1, COPY_BYTES // (columns * 4))):
        count = min(rows - row, max(1, COPY_BYTES // (columns * 4)))
        packed = read_range(stream, entries[name]["offset"] + row * columns // 2, count * columns // 2)
        raw = np.frombuffer(packed, dtype=np.uint8)
        values = np.stack((raw & 15, raw >> 4), axis=1).reshape(count, groups, 64).astype(np.float32)
        metadata = []
        for key in (scale_name, bias_name):
            data = read_range(stream, entries[key]["offset"] + row * groups * 2, count * groups * 2)
            metadata.append((np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4").reshape(count, groups, 1))
        decoded = values * metadata[0] + metadata[1]
        if not np.isfinite(decoded).all():
            raise ValueError("nonfinite reconstructed parameter")
        expected = read_range(original, original_range["offset"] + row * columns * 4, count * columns * 4)
        if decoded.tobytes() != expected:
            raise ValueError(f"affine reconstruction differs from frozen FP32 tensor: {name}")


def external_range(tensor, filename, file_bytes):
    import onnx
    data = unique_object((v.key, v.value) for v in tensor.external_data)
    if (tensor.data_location != onnx.TensorProto.EXTERNAL or set(data) != {"location", "offset", "length"}
            or data["location"] != filename or tensor.data_type != onnx.TensorProto.FLOAT
            or tensor.HasField("raw_data") or any(d <= 0 for d in tensor.dims)
            or any(len(getattr(tensor, key)) for key in
                   ("float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data"))):
        raise ValueError(f"unsupported external initializer: {tensor.name}")
    if any(not data[k].isascii() or not data[k].isdecimal() for k in ("offset", "length")):
        raise ValueError("external offsets must be unsigned decimal integers")
    offset, length = int(data["offset"]), int(data["length"])
    if length != math.prod(tensor.dims) * 4 or offset + length > file_bytes:
        raise ValueError("external tensor range differs")
    return dict(offset=offset, length=length)


class Store:
    """Shared decoder ranges and ordinary 1 GiB external-data files."""
    def __init__(self, directory, prefix, handles):
        self.directory, self.prefix, self.handles = directory, prefix, handles
        self.ranges, self.files = {}, []
        self.stream, self.used = None, 0

    def add(self, key, size, blocks):
        if key in self.ranges:
            if self.ranges[key]["length"] != size:
                raise ValueError("aliased tensor length differs")
            return self.ranges[key]
        if not 0 < size <= SHARD_BYTES:
            raise ValueError("tensor exceeds the shard bound")
        if self.stream is None or self.used + size > SHARD_BYTES:
            name = f"{self.prefix}.weights-{len(self.files):03d}"
            self.files.append(name)
            self.stream = self.handles.enter_context((self.directory / name).open("xb"))
            self.used = 0
        row = dict(location=self.files[-1], offset=self.used, length=size)
        digest, copied = hashlib.sha256(), 0
        for block in blocks:
            if not 0 < len(block) <= COPY_BYTES * 2 or copied + len(block) > size:
                raise ValueError("invalid bounded tensor copy")
            if self.stream.write(block) != len(block):
                raise OSError("incomplete compact weight write")
            digest.update(block)
            copied += len(block)
        if copied != size:
            raise ValueError("truncated compact tensor copy")
        row["sha256"] = digest.hexdigest()
        self.used += copied
        self.ranges[key] = row
        return row


def bind(tensor, row):
    import onnx
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    for key in ("location", "offset", "length"):
        tensor.external_data.add(key=key, value=str(row[key]))
    tensor.data_location = onnx.TensorProto.EXTERNAL


def rewrite(model, checkpoint, entries, original, original_bytes, filename, store, mapping):
    import onnx
    from onnx import helper as h, TensorProto as T, version_converter

    if [(v.domain, v.version) for v in model.opset_import] != [("", 17)]:
        raise ValueError("expected the frozen standard opset-17 graph")
    if model.ir_version != 8:
        raise ValueError("expected the frozen IR-8 graph")
    metadata = copy.deepcopy(model)
    metadata.ClearField("ir_version")
    metadata.ClearField("opset_import")
    for field in ("node", "initializer", "value_info", "input", "output"):
        metadata.graph.ClearField(field)
    before = {v.name: v.SerializeToString() for v in model.graph.initializer}
    io_before = [v.SerializeToString() for v in [*model.graph.input, *model.graph.output]]
    model = version_converter.convert_version(model, 21)
    # UINT4 is an IR-10 type; opset conversion alone does not change the IR.
    model.ir_version = 10
    after_metadata = copy.deepcopy(model)
    after_metadata.ClearField("ir_version")
    after_metadata.ClearField("opset_import")
    for field in ("node", "initializer", "value_info", "input", "output"):
        after_metadata.graph.ClearField(field)
    if after_metadata.SerializeToString() != metadata.SerializeToString():
        raise ValueError("opset conversion changed unrelated model/graph metadata")
    if ({v.name: v.SerializeToString() for v in model.graph.initializer} != before
            or [v.SerializeToString() for v in [*model.graph.input, *model.graph.output]] != io_before):
        raise ValueError("opset conversion changed original initializer or I/O fields")
    # Decoder softmax/reduction adapters may add Constant nodes, not model weights.
    owners = {v.SerializeToString() for v in model.graph.initializer}
    if any(v.external_data and v.SerializeToString() not in owners for v in _all_tensors(model)):
        raise ValueError("external data outside top-level initializers")
    saved, added, reconstructions, coverage = [], [], {}, []
    occupied = {name for v in model.graph.node for name in [*v.input, *v.output]}
    occupied.update(v.name for v in model.graph.initializer)
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.data_location != T.EXTERNAL:
            saved.append(tensor)
            continue
        interval = external_range(tensor, filename, original_bytes)
        source = mapping(tensor.name)
        if source not in entries:
            raise ValueError(f"initializer has no exact source mapping: {tensor.name}")
        row = dict(initializer=tensor.name, source=source, shape=list(tensor.dims), original_range=interval)
        if entries[source]["dtype"] != "U32":
            shape = entries[source]["shape"]
            if tensor.name.endswith("weight") and len(shape) > 2:
                shape = [shape[0], shape[-1], *shape[1:-1]]
            if list(tensor.dims) != shape:
                raise ValueError(f"ordinary source tensor shape differs: {source}")
            binding = store.add((filename, interval["offset"], interval["length"]), interval["length"], chunks(original, interval))
            bind(tensor, binding)
            saved.append(tensor)
            row.update(kind="unchanged_fp32", ranges=[binding])
        else:
            scales, biases = affine_group(entries, source, tensor.dims)
            verify_affine(checkpoint, entries, source, tensor.dims, original, interval)
            prefix = f"__speakerdesk_q4_{index}"
            names = [prefix + suffix for suffix in ("_q", "_scale", "_offset", "_unit", "_zero", "_group_shape", "_shape",
                                                    "_decoded", "_grouped", "_scaled", "_affine")]
            if occupied.intersection(names):
                raise ValueError("quantized adapter name collision")
            occupied.update(names)
            rows, columns = tensor.dims
            quantized = []
            ranges = []
            for name, dtype, shape, key, blocks, size in [
                (prefix + "_q", T.UINT4, [rows, columns], source, chunks(checkpoint, entries[source]), entries[source]["length"]),
                (prefix + "_scale", T.FLOAT, [rows, columns // 64, 1], scales, bf16_chunks(checkpoint, entries[scales]), entries[scales]["length"] * 2),
                (prefix + "_offset", T.FLOAT, [rows, columns // 64, 1], biases, bf16_chunks(checkpoint, entries[biases]), entries[biases]["length"] * 2),
            ]:
                value = T(name=name, data_type=dtype, dims=shape)
                binding = store.add(("checkpoint", key), size, blocks)
                bind(value, binding)
                quantized.append(value)
                ranges.append(binding)
            added.extend(quantized)
            added.extend([h.make_tensor(prefix + "_unit", T.FLOAT, [], [1]),
                          T(name=prefix + "_zero", data_type=T.UINT4, raw_data=b"\0"),
                          h.make_tensor(prefix + "_group_shape", T.INT64, [3], [rows, columns // 64, 64]),
                          h.make_tensor(prefix + "_shape", T.INT64, [2], [rows, columns])])
            nodes = []
            for op, inputs, output in [
                ("DequantizeLinear", [prefix + "_q", prefix + "_unit", prefix + "_zero"], prefix + "_decoded"),
                ("Reshape", [prefix + "_decoded", prefix + "_group_shape"], prefix + "_grouped"),
                ("Mul", [prefix + "_grouped", prefix + "_scale"], prefix + "_scaled"),
                ("Add", [prefix + "_scaled", prefix + "_offset"], prefix + "_affine"),
                ("Reshape", [prefix + "_affine", prefix + "_shape"], tensor.name),
            ]:
                nodes.append(h.make_node(op, inputs, [output], name=output + "_reconstruct"))
            reconstructions[tensor.name] = nodes
            row.update(kind="original_affine_uint4", source_metadata=[scales, biases], ranges=ranges)
        coverage.append(row)
    nodes = []
    for node in model.graph.node:
        for name in node.input:
            nodes.extend(reconstructions.pop(name, []))
        nodes.append(node)
    if reconstructions:
        raise ValueError("quantized initializer has no top-level consumer")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(saved + added)
    return model, coverage


def _all_tensors(model):
    import shard_encoder
    return shard_encoder.tensors(model)


def pin_open(handles, path, pin):
    import shard_encoder
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source is not a regular file: {path.name}")
    stream = handles.enter_context(shard_encoder.open_source_weights(path))
    if os.fstat(stream.fileno()).st_size != pin[0] or hashlib.file_digest(stream, "sha256").hexdigest() != pin[1]:
        raise ValueError(f"frozen source size/hash differs: {path.name}")
    stream.seek(0)
    return stream


def convert(source, checkpoint, destination):
    import onnx
    from export import source_name
    import shard_encoder

    source, checkpoint, destination = Path(source).resolve(), Path(checkpoint).resolve(), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("compact conversion requires a fresh destination")
    destination = destination.resolve()
    if destination.is_relative_to(source) or destination.is_relative_to(checkpoint):
        raise ValueError("destination must be outside both source trees")
    with ExitStack() as handles:
        retained_inventory = retained_pins()
        retained = admit_retained(handles, source, retained_inventory)
        opened = {name: pin_open(handles, source / name, pin) for name, pin in GRAPH_PINS.items()}
        originals = {name: pin_open(handles, checkpoint / name, pin) for name, pin in CHECKPOINT_PINS.items()}
        reference = pin_open(handles, source / "independent-reference.json", REFERENCE_PIN)
        entries = safetensors_header(originals["model.safetensors"], CHECKPOINT_PINS["model.safetensors"][0])
        config = json.loads(originals["config.json"].read())
        if config["quantization"] != {"bits": 4, "group_size": 64, "mode": "affine"}:
            raise ValueError("checkpoint quantization differs")
        destination.mkdir()
        coverage = {}
        with ExitStack() as output_handles:
            stores = {name: Store(destination, name, output_handles) for name in ("encoder", "decoder")}
            for graph in ("encoder", "decoder-init", "decoder-step"):
                filename = "encoder.weights" if graph == "encoder" else "decoder.weights"
                stream = opened[graph + ".onnx"]
                model = onnx.load_model_from_string(stream.read())
                model, coverage[graph] = rewrite(model, originals["model.safetensors"], entries, opened[filename],
                    GRAPH_PINS[filename][0], filename, stores["encoder" if graph == "encoder" else "decoder"], source_name)
                onnx.save(model, destination / (graph + ".onnx"))
        used = {row["source"] for rows in coverage.values() for row in rows}
        used.update(key for rows in coverage.values() for row in rows for key in row.get("source_metadata", []))
        aliases = source_aliases(originals["model.safetensors"], entries, used)
        for name in ("encoder", "decoder-init", "decoder-step"):
            onnx.checker.check_model(str(destination / (name + ".onnx")))
        # Copy retained inputs and expected arrays verbatim, with a complete receipt.
        case_pins = {}
        for graph in ("encoder", "decoder-init", "decoder-step"):
            outgoing = destination / (graph + "-case")
            outgoing.mkdir()
            case_stream, expected_stream = retained[graph + "-case/case.json"], retained[graph + "-case/expected.json"]
            case_stream.seek(0)
            expected_stream.seek(0)
            case = json.loads(case_stream.read())
            expected = json.loads(expected_stream.read())
            if case["model"] != "../" + graph + ".onnx" or case["repeat"] != 2:
                raise ValueError("retained case graph/repetition differs")
            names = {"expected.json", *[v["path"] for v in [*case["inputs"], *expected]]}
            for name in names:
                if (Path(name).name != name or any(char in name for char in ("\\", "/", ":"))
                        or Path(name).suffix not in {".json", ".f32", ".i64"}):
                    raise ValueError("unsafe retained case file")
                key = f"{graph}-case/{name}"
                if key not in retained:
                    raise ValueError("case refers to a file outside its frozen inventory")
                pin, handle = retained_inventory[key], retained[key]
                with (outgoing / name).open("xb") as output:
                    for block in chunks(handle, dict(offset=0, length=pin[0])):
                        output.write(block)
                case_pins[f"{graph}-case/{name}"] = dict(size=pin[0], sha256=pin[1])
            store = stores["encoder" if graph == "encoder" else "decoder"]
            case["model_files"] = {"../" + name: shard_encoder.sha256(destination / name) for name in store.files}
            case.update(disable_optimizations=True, disable_memory_arena=True, require_dml_only=True)
            (outgoing / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        for name, handle in [("tokenizer.json", originals["tokenizer.json"]), ("independent-reference.json", reference)]:
            handle.seek(0)
            with (destination / name).open("xb") as output:
                while block := handle.read(COPY_BYTES):
                    output.write(block)
        with (destination / "prompts.json").open("xb") as output:
            for block in chunks(retained["prompts.json"], dict(offset=0, length=retained_inventory["prompts.json"][0])):
                output.write(block)
        if os.name != "nt":
            for name, stream in opened.items():
                stream.seek(0)
                if hashlib.file_digest(stream, "sha256").hexdigest() != GRAPH_PINS[name][1]:
                    raise ValueError("source changed during conversion")
            for name, stream in originals.items():
                stream.seek(0)
                if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_PINS[name][1]:
                    raise ValueError("checkpoint changed during conversion")
            for name, stream in retained.items():
                stream.seek(0)
                if hashlib.file_digest(stream, "sha256").hexdigest() != retained_inventory[name][1]:
                    raise ValueError("retained cases changed during conversion")
        files = {str(p.relative_to(destination)).replace("\\", "/"): dict(size=p.stat().st_size, sha256=shard_encoder.sha256(p))
                 for p in destination.rglob("*") if p.is_file()}
        manifest = dict(tool_sha256=shard_encoder.sha256(__file__), source_graph_pins=GRAPH_PINS,
            checkpoint_pins=CHECKPOINT_PINS, source_provenance_sha256=retained_inventory["provenance.json"][1],
            retained_inventory_sha256=RETAINED_PINS_SHA256, retained_source_files=retained_inventory,
            onnx_version=onnx.__version__, mapping_source_sha256=shard_encoder.sha256(Path(__file__).with_name("export.py")),
            precision="original UINT4 affine weights, exact BF16 metadata widened to FP32; existing FP32 graph arithmetic",
            opset_conversion="onnx.version_converter 17 to 21; explicit IR 8 to 10 for UINT4; other metadata, original initializers and I/O fields verified unchanged",
            coverage=coverage, source_aliases=aliases, coverage_counts=dict(checkpoint_tensors=len(entries),
                quantized_source_tensors=sum(value["dtype"] == "U32" for value in entries.values()),
                graph_parameters=sum(len(rows) for rows in coverage.values()),
                quantized_graph_parameters=sum(row["kind"] == "original_affine_uint4" for rows in coverage.values() for row in rows),
                unique_output_ranges=sum(len(store.ranges) for store in stores.values()),
                shared_range_references=sum(len(row["ranges"]) for rows in coverage.values() for row in rows)
                    - sum(len(store.ranges) for store in stores.values())),
            copied_case_files=case_pins, files=files,
            qualification="UNQUALIFIED: native outputs, full-model residency, temporary memory and latency required")
        (destination / "compact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("full frozen conversion requires Windows; small synthetic controls are portable")
    result = convert(args.source, args.checkpoint, args.destination)
    print(json.dumps(dict(files=result["files"], qualification=result["qualification"]), indent=2))
