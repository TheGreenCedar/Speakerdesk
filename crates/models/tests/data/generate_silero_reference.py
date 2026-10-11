"""Validation only: run the pinned upstream graph in a disposable Python environment."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
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
parser.add_argument("--backend", required=True, help="native wheel variant actually loaded")
args = parser.parse_args()
source = args.source.read_bytes()
assert hashlib.sha256(source).hexdigest() == "7838b90f10678d873dbf35b9b3947beed99e7b503cd5f64605c525f056c8270b"
assert hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest() == "65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff"
# Execute the unmodified upstream classes without importing unused audio I/O packages.
nodes = [node for node in ast.parse(source).body
         if isinstance(node, (ast.ClassDef, ast.FunctionDef))
         and node.name in ("_reflect_pad_right", "SileroVADBranch")]
namespace = dict(mx=mx, nn=nn, BranchConfig=object, Optional=typing.Optional,
                 Union=typing.Union, Tuple=typing.Tuple)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source), "exec"), namespace)
with wave.open(str(args.wav)) as wav:
    assert wav.getparams()[:3] == (1, 2, 16000)
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
mx.set_default_device(mx.gpu)
started = time.perf_counter()
config = json.loads((args.model / "config.json").read_text())["branch_16k"]
model = namespace["SileroVADBranch"](types.SimpleNamespace(**config))
model.load_weights([(key.removeprefix("vad_16k."), value)
                    for key, value in mx.load(str(args.model / "model.safetensors")).items()], strict=True)
mx.eval(model.parameters())
load_ms = (time.perf_counter() - started) * 1000


def run():
    states = [None, None]
    contexts = [mx.zeros((1, 64)), mx.zeros((1, 64))]
    frames = []
    started = time.perf_counter()
    for offset in range(0, len(pcm), 512):
        raw = pcm[offset:offset + 512]
        frame = np.pad(raw, (0, 512 - len(raw)))
        peak = float(np.max(np.abs(raw)))
        gain = min(256, max(1, .25 / peak)) if peak else 1
        probabilities = []
        for view, factor in enumerate((1, gain)):
            chunk = mx.array(frame * np.float32(factor))[None, :]
            probability, states[view] = model(mx.concatenate([contexts[view], chunk], axis=-1), state=states[view])
            mx.eval(probability, states[view])
            contexts[view] = chunk[:, -64:]
            probabilities.append(float(probability.item()))
        frames.append(dict(start=offset, end=offset + len(raw), raw_p=probabilities[0], norm_p=probabilities[1], gain=gain))
    return frames, (time.perf_counter() - started) * 1000


frames, cold_ms = run()
_, warm_ms = run()
result = dict(source_revision="feb25a37b07923bae556e59111995071d66afa0d",
              source_sha256=hashlib.sha256(source).hexdigest(),
              mlx_version=importlib.metadata.version("mlx"), backend=args.backend,
              wav_sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest(),
              samples=len(pcm), model_load_ms=load_ms, cold_ms=cold_ms, warm_ms=warm_ms, frames=frames)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "frames"}))
