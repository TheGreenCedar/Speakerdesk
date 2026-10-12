"""Validation only: run pinned upstream Nemotron in a disposable Python environment."""
import argparse
import ast
import ctypes
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import time
import types
import typing
import wave
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("model", type=Path)
parser.add_argument("wav", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--backend", required=True)
parser.add_argument("--preset", choices=["offline", "low"], default="offline")
args = parser.parse_args()
hashes = {
    "nemotron_diarization.py": "e56ab019f0d3400cb19ebc2deefcd7575b418dec60a6ab4bdafe322579636683",
    "sortformer.py": "a013b8dc813698c715c7430a15390e455d39977d7f3b77106aab08d61cdc2378",
    "dsp.py": "c47da00eea77f3bce396d4eb36c694211f40d9e5571393b685eda77f8978b98c",
}
for name, digest in hashes.items():
    assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest() == "21e8427d1795c9c46c5800f56b16061734ffd0dcadd71d9bcf0b4d6ef7261da5"
assert hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest() == "f215091252bc54f8dcc7d682962a659e03a6bcb26d1c86e4728b8c6730130eb4"
# Execute the unchanged upstream graph without importing unrelated audio I/O packages.
module = types.ModuleType("native_nemotron_reference")
sys.modules[module.__name__] = module
namespace = module.__dict__
namespace.update(vars(typing))
namespace.update(mx=mx, nn=nn, np=np, math=math, time=time, Path=Path,
                 dataclass=dataclasses.dataclass, field=dataclasses.field, replace=dataclasses.replace)
spec = importlib.util.spec_from_file_location("reference_dsp", args.source / "dsp.py")
dsp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsp)
namespace.update(hanning=dsp.hanning, mel_filters=dsp.mel_filters)


def classes(file, names=None, scope=None):
    nodes = [node for node in ast.parse((args.source / file).read_text()).body
             if isinstance(node, ast.ClassDef) and (names is None or node.name in names)]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(args.source / file), "exec"), namespace if scope is None else scope)


# Preserve each upstream module's globals: AOSC calls Sortformer's own Model helpers.
sortformer = types.ModuleType("native_sortformer_reference")
sys.modules[sortformer.__name__] = sortformer
sortformer.__dict__.update({k: v for k, v in namespace.items() if not k.startswith("__")})
classes("sortformer.py", {"SortformerModules", "DiarizationSegment", "DiarizationOutput", "Model"}, sortformer.__dict__)
namespace.update({name: getattr(sortformer, name) for name in ["SortformerModules", "DiarizationSegment", "DiarizationOutput"]})
namespace["Sortformer"] = sortformer.Model
classes("nemotron_diarization.py")
def checkpoint_config(value):
    # Upstream preset selection uses dataclasses.replace on these pinned JSON fields.
    cls = dataclasses.make_dataclass("CheckpointConfig", [(key, typing.Any) for key in value])
    return cls(**value)


config = json.loads((args.model / "config.json").read_text(), object_hook=checkpoint_config)
mx.set_default_device(mx.gpu)
started = time.perf_counter()
model = namespace["Model"](config)
model.set_streaming_config(args.preset)
model.load_weights(list(mx.load(str(args.model / "model.safetensors")).items()), strict=True)
model.eval()
mx.eval(model.parameters())
load_ms = (time.perf_counter() - started) * 1000
with wave.open(str(args.wav)) as wav:
    assert wav.getparams()[:3] == (1, 2, 16000)
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
runs = []
for _ in range(2):
    started = time.perf_counter()
    result = model.generate(pcm, sample_rate=16000, threshold=.5, min_duration=0, merge_gap=0)
    mx.eval(result.speaker_probs)
    runs.append((time.perf_counter() - started) * 1000)
record = dict(source_revision="feb25a37b07923bae556e59111995071d66afa0d", source_sha256=hashes,
              preset=args.preset,
              mlx_version=importlib.metadata.version("mlx"), backend=args.backend,
              wav_sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest(), samples=len(pcm),
              valid_frames=len(pcm) // 160, model_load_ms=load_ms, runs_ms=runs,
              turns=[dict(start=x.start, end=x.end, speaker=f"speaker_{x.speaker}") for x in result.segments],
              probabilities=result.speaker_probs.tolist())
# The macOS 15 and 26 wheels share a version but contain different Metal kernels.
dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
dyld._dyld_get_image_name.restype = ctypes.c_char_p
runtime = next(Path(dyld._dyld_get_image_name(i).decode())
               for i in range(dyld._dyld_image_count())
               if dyld._dyld_get_image_name(i).decode().endswith("/libmlx.dylib"))
metal = Path(os.environ.get("MLX_METAL_PATH", runtime.parent / "mlx.metallib"))
record["runtime_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [runtime, metal]}
args.output.write_text(json.dumps(record) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "probabilities"}))
