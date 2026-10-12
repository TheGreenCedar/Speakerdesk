"""Validation only: run pinned mlx-speech Cohere in a disposable Python environment."""
import argparse
import ast
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import sys
import time
import types
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
args = parser.parse_args()
hashes = {
    "config.py": "bd140cad3cbf72185188ffeca37413708ba30202a8247f19fc8b243c997b80e1",
    "encoder.py": "18597f8b13ea63b696c2b2efb3273263213d9cf829782d1c1a9de67482cc8b19",
    "decoder.py": "bc84334a520821a770866f8e7f384cf6dfb400897009b2c931073e47b687132d",
    "feature_extraction.py": "0993162fc6de36870b417795aceefee3d7402a7a232b31c6d765af0c45bc8e2a",
    "tokenizer.py": "a4f5e5d28f792540adbea9d1cf912259a31e75926e0e7e372d0a7cf84492ef71",
    "generation.py": "231e9a0d294e7c4b85b211211a1c2b5c1588c67c0afd20e632505313ce9f1fc7",
}
for name, digest in hashes.items():
    assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest
with (args.model / "model.safetensors").open("rb") as artifact:
    assert hashlib.file_digest(artifact, "sha256").hexdigest() == "ae947c13ba1cb8ce24c8bf72ded2520dcda1894ebf5a54fbb803c03bc28ef7fb"
# Load the unmodified graph without importing unrelated checkpoint/download packages.
package = types.ModuleType("cohere_reference")
package.__path__ = [str(args.source)]
sys.modules[package.__name__] = package
config_module = importlib.import_module("cohere_reference.config")
decoder = importlib.import_module("cohere_reference.decoder")
features = importlib.import_module("cohere_reference.feature_extraction")
tokenizer = importlib.import_module("cohere_reference.tokenizer")
module = types.ModuleType("cohere_reference.generation")
sys.modules[module.__name__] = module
namespace = module.__dict__
namespace.update(mx=mx, np=np, dataclass=dataclasses.dataclass, Path=Path, Any=object, _NO_SPACE_LANGS=frozenset({"ja", "zh"}))
nodes = [node for node in ast.parse((args.source / "generation.py").read_text()).body if isinstance(node, ast.ClassDef)]
tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
exec(compile(ast.fix_missing_locations(tree), str(args.source / "generation.py"), "exec"), namespace)
mx.set_default_device(mx.gpu)
started = time.perf_counter()
config = config_module.CohereAsrConfig.from_path(args.model)
model = decoder.CohereAsrForConditionalGeneration(config)
weights = mx.load(str(args.model / "model.safetensors"))
nn.quantize(model, group_size=64, bits=4, mode="affine", class_predicate=lambda path, module: f"{path}.scales" in weights)
model.load_weights(list(weights.items()), strict=True)
del weights
model.set_dtype(mx.bfloat16)
model.eval()
mx.eval(model.parameters())
load_ms = (time.perf_counter() - started) * 1000
extractor = features.CohereAsrFeatureExtractor.from_dir(args.model)
tokens = tokenizer.CohereAsrTokenizer.from_dir(args.model)
wrapper = namespace["CohereAsrModel"](model, extractor, tokens, config)
with wave.open(str(args.wav)) as wav:
    assert wav.getparams()[:3] == (1, 2, 16000)
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
runs = []
for _ in range(2):
    started = time.perf_counter()
    result = wrapper.transcribe(pcm, language="en")
    runs.append((time.perf_counter() - started) * 1000)
record = dict(source_revision="4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc", source_sha256=hashes,
              mlx_version=importlib.metadata.version("mlx"), backend=args.backend,
              wav_sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest(), samples=len(pcm),
              valid_frames=len(pcm) // 160, model_load_ms=load_ms, runs_ms=runs,
              text=result.text, tokens=result.tokens, language=result.language)
args.output.write_text(json.dumps(record) + "\n")
print(json.dumps(record))
