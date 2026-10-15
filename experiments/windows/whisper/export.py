"""Build-only export of the pinned Whisper-tiny 99-language scoring path."""
import argparse
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.windows.nemotron.export import Graph, sha256, value
from gelu_boundary import HALF_RECIPROCAL, preserve_gelu_boundaries
from position_constant import PROVENANCE as POSITION_PROVENANCE, load_positions


def language_tokens(path):
    tokens = json.loads(path.read_text())
    languages = {name[2:-2]: token for name, token in tokens.items()
                 if re.fullmatch(r"<\|[a-z]{2,3}\|>", name)}
    ids = list(languages.values())
    ordered = sorted(ids)
    sot = tokens.get("<|startoftranscript|>")
    if (len(ids) != 99 or "en" not in languages or "fr" not in languages
            or ordered != list(range(50259, 50358)) or sot != ordered[0] - 1):
        raise ValueError("incompatible Whisper language tokens")
    return list(languages), ids, sot


def load_weights(directory):
    pin = next(p for p in json.loads((ROOT / "crates/host/src/model-artifacts.json").read_text())
               if p["id"] == "whisper-language")
    for name in ["config.json", "added_tokens.json", "model.safetensors"]:
        path, expected = directory / name, pin["files"][name]
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"unpinned Whisper {name}")
    data = (directory / "model.safetensors").read_bytes()
    size = struct.unpack_from("<Q", data)[0]
    weights = {}
    for name, entry in json.loads(data[8:8 + size]).items():
        if name == "__metadata__":
            continue
        if entry["dtype"] != "F16":
            raise ValueError((name, entry["dtype"]))
        start, end = entry["data_offsets"]
        weights[name] = np.frombuffer(data[8 + size + start:8 + size + end], "<f2").astype(np.float32).reshape(entry["shape"])
    return pin, weights


def frontend_buffers():
    window = (.5 * (1 - np.cos(2 * np.pi * np.arange(400) / 400))).astype(np.float32)
    frequencies = np.linspace(0, 8000, 201, dtype=np.float32)
    step = np.log(6.4) / 27
    mel = np.linspace(0, 15 + np.log(8) / step, 82, dtype=np.float32)
    edges = np.where(mel >= 15, np.exp((mel - np.float32(15)) * np.float32(step)) * np.float32(1000),
                     mel * np.float32(200 / 3))
    differences = np.diff(edges)
    slopes = edges[None] - frequencies[:, None]
    filters = np.maximum(np.minimum(-slopes[:, :80] / differences[:80],
                                    slopes[:, 2:] / differences[1:]), np.float32(0))
    filters *= np.float32(2) / (edges[2:] - edges[:80])
    return window, filters.astype(np.float32)


def round_half(g, x):
    return g.op("Cast", g.op("Cast", x, to=TensorProto.FLOAT16), to=TensorProto.FLOAT)


def export_frontend(output):
    g = Graph({})
    window, filters = frontend_buffers()
    angles = np.arange(400)[:, None] * np.arange(201)[None] * (2 * np.pi / 400)
    frames = g.op("Mul", "audio_windows", g.const(window))
    real = g.op("MatMul", frames, g.const(np.cos(angles).astype(np.float32)))
    imaginary = g.op("MatMul", frames, g.const(-np.sin(angles).astype(np.float32)))
    power = g.op("Add", g.op("Mul", real, real), g.op("Mul", imaginary, imaginary))
    mel = g.op("MatMul", power, g.const(filters))
    log = g.op("Div", g.op("Log", g.op("Max", mel, g.const(np.array(1e-10, np.float32)))),
               g.const(np.array(np.log(10), np.float32)))
    maximum = g.op("ReduceMax", log, keepdims=0)
    clipped = g.op("Max", log, g.op("Sub", maximum, g.const(np.array(8, np.float32))))
    normalized = g.op("Div", g.op("Add", clipped, g.const(np.array(4, np.float32))),
                      g.const(np.array(4, np.float32)))
    g.op("Identity", round_half(g, normalized), name="mel")
    model = g.model("Speakerdesk-pinned-Whisper-logmel", [value("audio_windows", [1, 3000, 400])],
                    [value("mel", [1, 3000, 80])])
    onnx.save(model, output / "frontend.onnx")


class WhisperGraph(Graph):
    def linear(self, x, name, source=384, target=384, bias=True):
        return round_half(self, super().linear(x, name, source, target, bias))

    def norm(self, x, name, width=384):
        return round_half(self, super().norm(x, name, width))

    def gelu(self, x):
        # The retained compiled MLX operator rounds the reciprocal itself to Half.
        divided = self.op("Mul", x, self.const(np.array(HALF_RECIPROCAL, np.float32)))
        gated = self.op("Add", self.const(np.array(1, np.float32)), self.op("Erf", divided))
        return round_half(self, self.op("Div", self.op("Mul", x, gated), self.const(np.array(2, np.float32))))

    def attention(self, x, source, name, length):
        heads = lambda y: self.op("Transpose", self.reshape(y, [1, -1, 6, 64]), perm=[0, 2, 1, 3])
        scale = self.const(np.array(np.float16(64**-.25), np.float32))
        q = round_half(self, self.op("Mul", heads(self.linear(x, name + ".query")), scale))
        k = round_half(self, self.op("Mul", heads(self.linear(source, name + ".key", bias=False)), scale))
        v = heads(self.linear(source, name + ".value"))
        scores = round_half(self, self.op("MatMul", q, self.op("Transpose", k, perm=[0, 1, 3, 2])))
        probabilities = round_half(self, self.op("Softmax", scores, axis=-1))
        attended = round_half(self, self.op("MatMul", probabilities, v))
        attended = self.reshape(self.op("Transpose", attended, perm=[0, 2, 1, 3]), [1, length, 384])
        return self.linear(attended, name + ".out")

    def block(self, x, name, encoded=None):
        normalized = self.norm(x, name + ".attn_ln")
        x = round_half(self, self.op("Add", x, self.attention(normalized, normalized, name + ".attn",
                                                             1500 if encoded is None else 1)))
        if encoded is not None:
            x = round_half(self, self.op("Add", x, self.attention(self.norm(x, name + ".cross_attn_ln"),
                                                                 encoded, name + ".cross_attn", 1)))
        hidden = self.gelu(self.linear(self.norm(x, name + ".mlp_ln"), name + ".mlp1", target=1536))
        return round_half(self, self.op("Add", x, self.linear(hidden, name + ".mlp2", source=1536)))


def native_half(model):
    # Keep probe I/O Float32; neural operators and their constants use checkpoint Float16.
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.data_type == TensorProto.FLOAT:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(tensor).astype(np.float16), tensor.name))
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                    attribute.i = TensorProto.FLOAT16
    softmax = model.graph.node[-1]
    source = softmax.input[0]
    softmax.input[0] = "float_logits"
    model.graph.node.insert(len(model.graph.node) - 1, helper.make_node(
        "Cast", [source], ["float_logits"], name="float_logits", to=TensorProto.FLOAT))
    onnx.checker.check_model(model, full_check=True)
    return model


def export(directory, output, positions_path):
    pin, weights = load_weights(directory)
    positions = load_positions(positions_path).astype(np.float32)
    codes, token_ids, sot = language_tokens(directory / "added_tokens.json")
    output.mkdir(parents=True, exist_ok=True)
    export_frontend(output)
    embedding = weights.pop("decoder.token_embedding.weight")
    learned = weights.pop("decoder.positional_embedding")
    if embedding.shape != (51865, 384) or learned.shape != (448, 384):
        raise ValueError("unexpected Whisper embedding shape")
    sot_embedding = (embedding[sot] + learned[0]).astype(np.float16).astype(np.float32)[None, None]
    language_embeddings = embedding[token_ids].T.copy()
    g = WhisperGraph(weights)
    x = round_half(g, "mel")
    for name, channels, stride in [("encoder.conv1", 80, 1), ("encoder.conv2", 384, 2)]:
        x = g.op("Transpose", x, perm=[0, 2, 1])
        x = round_half(g, g.op("Conv", x, g.weight(name + ".weight", [384, 3, channels], (0, 2, 1)),
                              pads=[1, 1], strides=[stride]))
        x = g.op("Transpose", x, perm=[0, 2, 1])
        # MLX Conv1d adds bias after the convolution has rounded to Float16.
        x = round_half(g, g.op("Add", x, g.weight(name + ".bias", [384])))
        x = g.gelu(x)
    x = round_half(g, g.op("Add", x, g.const(positions)))
    for index in range(4):
        x = g.block(x, f"encoder.blocks.{index}")
    encoded = g.norm(x, "encoder.ln_post")
    x = g.const(sot_embedding)
    for index in range(4):
        x = g.block(x, f"decoder.blocks.{index}", encoded)
    logits = round_half(g, g.op("MatMul", g.norm(x, "decoder.ln"), g.const(language_embeddings)))
    g.op("Softmax", g.reshape(logits, [99]), name="probabilities", axis=-1)
    if weights:
        raise ValueError(("unexpected Whisper weight inventory", list(weights)))
    model = g.model("Speakerdesk-pinned-Whisper-language-only", [value("mel", [1, 3000, 80])],
                    [value("probabilities", [99])])
    model = native_half(model)
    model, _ = preserve_gelu_boundaries(model)
    onnx.save(model, output / "whisper.onnx")
    (output / "languages.json").write_text(json.dumps({"codes": codes, "token_ids": token_ids, "sot": sot}, indent=2) + "\n")
    provenance = {"weights": {"repo": pin["repo"], "revision": pin["revision"], "files": pin["files"]},
                  "precision": "Float16 neural operators; Float32 I/O and final language softmax",
                  "onnx_version": onnx.__version__, "numpy_version": np.__version__,
                  "exporter_sha256": sha256(Path(__file__)),
                  "position_constant": POSITION_PROVENANCE,
                  "position_input_sha256": sha256(positions_path),
                  "graph_helper_sha256": sha256(ROOT / "experiments/windows/nemotron/export.py"),
                  "outputs": {name: sha256(output / name) for name in ["whisper.onnx", "frontend.onnx", "languages.json"]},
                  "operators": sorted({node.op_type for node in model.graph.node})}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--positions", type=Path, required=True,
                        help="complete pinned MLX positional parameter (.npy); see position-constant.json")
    args = parser.parse_args()
    export(args.model, args.output, args.positions)
