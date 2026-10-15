"""Build-only repack of the admitted encoder into ordinary ONNX external-data shards."""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path

import onnx

MAX_SHARD_BYTES = 1 << 30
COPY_BYTES = 1 << 20
# Immutable cohere-fp32-v2 inputs; runtime admission will later pin the sharded outputs.
SOURCE_PINS = {
    "encoder.onnx": {"size": 2022217, "sha256": "0c243acb5f78842b7573d9a1c46227b9436b7fb21986d03cc45a84bc9d57d9ad"},
    "encoder.weights": {"size": 7581848576, "sha256": "6e8d004d81a1d870de6725665ce381ff4b36df207909ed228bddbd8bb0246100"},
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_source_weights(path):
    if os.name != "nt":
        return path.open("rb", buffering=0)
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # FILE_SHARE_READ excludes writers and deletion until the returned file closes.
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def tensors(message):
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            for child in value if field.is_repeated else [value]:
                yield from tensors(child)


def non_external_graph(model):
    model = copy.deepcopy(model)
    for tensor in tensors(model):
        tensor.ClearField("external_data")
    return model.SerializeToString()


def repack(source, destination, pins, shard_bytes=MAX_SHARD_BYTES):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("sharding requires a fresh output directory")
    destination = destination.resolve()
    if destination.is_relative_to(source) or not 0 < shard_bytes <= MAX_SHARD_BYTES:
        raise ValueError("output must be outside the source; shards must be at most 1 GiB")
    for name in ("encoder.onnx", "encoder.weights"):
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != pins[name]["size"]:
            raise ValueError(f"source file type/size differs: {name}")
    with (source / "encoder.onnx").open("rb") as graph:
        graph_bytes = graph.read(pins["encoder.onnx"]["size"] + 1)
    if (len(graph_bytes) != pins["encoder.onnx"]["size"]
            or hashlib.sha256(graph_bytes).hexdigest() != pins["encoder.onnx"]["sha256"]):
        raise ValueError("source encoder graph hash differs")
    model = onnx.load_model_from_string(graph_bytes)
    unchanged = non_external_graph(model)
    ranges, references = {}, []
    for tensor in tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            if tensor.external_data:
                raise ValueError("external metadata on a non-external tensor")
            continue
        data = {item.key: item.value for item in tensor.external_data}
        if (len(data) != len(tensor.external_data) or set(data) != {"location", "offset", "length"}
                or data["location"] != "encoder.weights"):
            raise ValueError("unsupported or unsafe external-data metadata/path")
        if any(not data[key].isascii() or not data[key].isdecimal() for key in ("offset", "length")):
            raise ValueError("external offsets and lengths must be unsigned decimal integers")
        offset, length = int(data["offset"]), int(data["length"])
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
        if dtype.kind not in "biufc" or any(dim < 0 for dim in tensor.dims):
            raise ValueError("unsupported external tensor type/shape")
        if length <= 0 or length != math.prod(tensor.dims) * dtype.itemsize:
            raise ValueError("external length differs from the tensor byte count")
        if offset + length > pins["encoder.weights"]["size"] or length > shard_bytes:
            raise ValueError("external range is truncated or one tensor exceeds the shard limit")
        if any(tensor.HasField(key) if key == "raw_data" else len(getattr(tensor, key))
               for key in ("raw_data", "float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data")):
            raise ValueError("external tensor also has embedded data")
        key = (offset, length)
        ranges.setdefault(key, {"source_offset": offset, "length": length, "tensors": []})["tensors"].append(tensor.name)
        references.append((tensor, key))
    if not ranges:
        raise ValueError("source encoder has no external tensors")
    shards = []
    with open_source_weights(source / "encoder.weights") as incoming, ExitStack() as handles:
        if os.fstat(incoming.fileno()).st_size != pins["encoder.weights"]["size"]:
            raise ValueError("opened source encoder weights size differs")
        if hashlib.file_digest(incoming, "sha256").hexdigest() != pins["encoder.weights"]["sha256"]:
            raise ValueError("source encoder weights hash differs")
        destination.mkdir(parents=True, exist_ok=False)
        stream, used = None, 0
        for (offset, length), row in ranges.items():
            if stream is None or used + length > shard_bytes:
                name = f"encoder.weights-{len(shards):03d}"
                shards.append(name)
                stream = handles.enter_context((destination / name).open("xb"))
                used = 0
            row.update(location=shards[-1], offset=used)
            incoming.seek(offset)
            remaining, digest = length, hashlib.sha256()
            while remaining:
                block = incoming.read(min(COPY_BYTES, remaining))
                if not block:
                    raise ValueError("source weights truncated during copy")
                if stream.write(block) != len(block):
                    raise OSError("incomplete shard write")
                digest.update(block)
                remaining -= len(block)
            row["sha256"] = digest.hexdigest()
            used += length
        if os.name != "nt":
            incoming.seek(0)
            if hashlib.file_digest(incoming, "sha256").hexdigest() != pins["encoder.weights"]["sha256"]:
                raise ValueError("source weights changed during copy")
    for tensor, key in references:
        for item in tensor.external_data:
            item.value = str(ranges[key][item.key])
    if non_external_graph(model) != unchanged:
        raise ValueError("sharding changed non-external graph fields")
    onnx.save_model(model, destination / "encoder.onnx")
    if non_external_graph(onnx.load(destination / "encoder.onnx", load_external_data=False)) != unchanged:
        raise ValueError("saved graph changed non-external fields")
    files = {name: {"size": (destination / name).stat().st_size, "sha256": sha256(destination / name)}
             for name in ["encoder.onnx", *shards]}
    manifest = dict(source_files={name: pins[name] for name in ("encoder.onnx", "encoder.weights")},
                    tool_sha256=sha256(__file__), onnx_version=onnx.__version__, shard_bytes=shard_bytes,
                    non_external_graph_sha256=hashlib.sha256(unchanged).hexdigest(),
                    files=files, tensors=len(references), unique_ranges=len(ranges), ranges=list(ranges.values()))
    (destination / "encoder-shards.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--shard-bytes", type=int, default=MAX_SHARD_BYTES)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("pinned artifact repacking requires Windows; synthetic tests are portable")
    result = repack(args.source, args.destination, SOURCE_PINS, args.shard_bytes)
    print(json.dumps({"files": result["files"], "tensors": result["tensors"], "unique_ranges": result["unique_ranges"]}, indent=2))
