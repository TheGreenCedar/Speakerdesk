"""Build-only ONNX conversion of Speakerdesk's pinned Nemotron checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_weights(directory):
    pin = next(p for p in json.loads((ROOT / "crates/host/src/model-artifacts.json").read_text())
               if p["id"] == "nemotron")
    for name in ["config.json", "model.safetensors"]:
        path = directory / name
        expected = pin["files"][name]
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"unpinned Nemotron {name}")
    data = (directory / "model.safetensors").read_bytes()
    size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + size])
    weights = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        raw = data[8 + size + start:8 + size + end]
        if entry["dtype"] == "BF16":
            value = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif entry["dtype"] == "F32":
            value = np.frombuffer(raw, dtype="<f4")
        else:
            raise ValueError((name, entry["dtype"]))
        weights[name] = value.reshape(entry["shape"])
    return pin, weights


class Graph:
    def __init__(self, weights):
        self.weights = weights
        self.nodes, self.initializers = [], []
        self.serial = 0

    def const(self, value, name=None):
        if name is None:
            self.serial += 1
            name = f"constant_{self.serial}"
        self.initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, kind, *inputs, name=None, **attrs):
        self.serial += 1
        name = name or f"{kind}_{self.serial}"
        self.nodes.append(helper.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def weight(self, name, shape, transpose=None):
        array = self.weights.pop(name)
        if list(array.shape) != shape:
            raise ValueError((name, array.shape, shape))
        if transpose is not None:
            array = array.transpose(transpose).copy()
        return self.const(array, name)

    def reshape(self, x, shape, name=None):
        return self.op("Reshape", x, self.const(np.asarray(shape, np.int64)), name=name)

    def linear(self, x, name, source, target, bias=True):
        x = self.op("MatMul", x, self.weight(name + ".weight", [target, source], (1, 0)))
        if bias:
            x = self.op("Add", x, self.weight(name + ".bias", [target]))
        return x

    def norm(self, x, name, width=512):
        return self.op("LayerNormalization", x, self.weight(name + ".weight", [width]),
                       self.weight(name + ".bias", [width]), epsilon=1e-5)

    def slice(self, x, start, end, axis):
        return self.op("Slice", x, self.const(np.array([start], np.int64)),
                       self.const(np.array([end], np.int64)),
                       self.const(np.array([axis], np.int64)))

    def model(self, name, inputs, outputs):
        model = helper.make_model(helper.make_graph(self.nodes, name, inputs, outputs, self.initializers),
                                  opset_imports=[helper.make_opsetid("", 17)], ir_version=9,
                                  producer_name="speakerdesk-pinned-nemotron-export")
        onnx.checker.check_model(model, full_check=True)
        return model


def value(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def export_frontend(weights, output):
    g = Graph({})
    window = np.pad(weights["preprocessor.window"], (56, 56))
    framed = g.op("Mul", "audio_windows", g.const(window))
    angles = np.arange(512)[:, None] * np.arange(257)[None] * (2 * np.pi / 512)
    real = g.op("MatMul", framed, g.const(np.cos(angles).astype(np.float32)))
    imaginary = g.op("MatMul", framed, g.const(-np.sin(angles).astype(np.float32)))
    power = g.op("Add", g.op("Mul", real, real), g.op("Mul", imaginary, imaginary))
    mel = g.op("MatMul", power, g.const(weights["preprocessor.fb"][0].T.copy()))
    mel = g.op("Log", g.op("Add", mel, g.const(np.array(2**-24, np.float32))))
    length = g.op("Gather", g.op("Shape", "audio_windows"), g.const(np.array(1, np.int64)), axis=0)
    positions = g.op("Range", g.const(np.array(0, np.int64)), length, g.const(np.array(1, np.int64)))
    mask = g.reshape(g.op("Less", positions, "valid_frames"), [1, -1, 1])
    g.op("Where", mask, mel, g.const(np.array(0., np.float32)), name="features")
    model = g.model("Speakerdesk-pinned-Nemotron3-logmel", [
        value("audio_windows", [1, "frames", 512]), value("valid_frames", [1], TensorProto.INT64)
    ], [value("features", [1, "frames", 128])])
    onnx.save(model, output / "frontend.onnx")


def export(directory, output):
    pin, weights = load_weights(directory)
    output.mkdir(parents=True, exist_ok=True)
    for key in ["preprocessor.window", "preprocessor.fb", "sortformer_modules.learnable_sil_emb"]:
        np.save(output / (key.rsplit(".", 1)[1] + ".npy"), weights[key])
    weights["sortformer_modules.learnable_sil_emb"].astype("<f4").tofile(output / "silence.f32")
    export_frontend(weights, output)
    g = Graph(weights)
    x = g.reshape("features", [1, -1, 1024])
    chunk = g.linear(x, "encoder.pre_encode.proj", 1024, 512, False)
    g.op("Identity", chunk, name="chunk")
    combined = g.op("Concat", "cache", "fifo", chunk, axis=1)
    length = g.op("Gather", g.op("Shape", combined), g.const(np.array(1, np.int64)), axis=0)
    prefix = g.op("Add", g.op("Gather", g.op("Shape", "cache"), g.const(np.array([1], np.int64)), axis=0),
                  g.op("Gather", g.op("Shape", "fifo"), g.const(np.array([1], np.int64)), axis=0))
    valid = g.op("Add", prefix, g.op("Div", g.op("Add", "valid_frames", g.const(np.array([7], np.int64))),
                                   g.const(np.array([8], np.int64))))
    positions = g.op("Range", g.const(np.array(0, np.int64)), length, g.const(np.array(1, np.int64)))
    mask = g.reshape(g.op("Less", positions, valid), [1, 1, 1, -1])
    positions_float = g.reshape(g.op("Cast", positions, to=TensorProto.FLOAT), [-1, 1])
    angles = g.op("Mul", positions_float, g.const(np.power(10000., -np.arange(32) / 32).astype(np.float32)))
    cosine, sine = g.op("Cos", angles), g.op("Sin", angles)

    def rope(x):
        left, right = g.slice(x, 0, 32, 3), g.slice(x, 32, 64, 3)
        a = g.op("Sub", g.op("Mul", left, cosine), g.op("Mul", right, sine))
        b = g.op("Add", g.op("Mul", left, sine), g.op("Mul", right, cosine))
        return g.op("Concat", a, b, axis=3)

    x = g.norm(combined, "encoder.embed_norm")
    for index in range(31):
        prefix_name = f"encoder.layers.{index}"
        qkv = g.reshape(g.linear(g.norm(x, prefix_name + ".norm1"), prefix_name + ".attn.w_qkv",
                                 512, 1536, False), [1, -1, 3, 8, 64])
        q, k, v = [g.op("Transpose", g.op("Gather", qkv, g.const(np.array(i, np.int64)), axis=2),
                        perm=[0, 2, 1, 3]) for i in range(3)]
        q, k = rope(q), rope(k)
        logits = g.op("Mul", g.op("MatMul", q, g.op("Transpose", k, perm=[0, 1, 3, 2])),
                      g.const(np.array(.125, np.float32)))
        logits = g.op("Where", mask, logits, g.const(np.array(-np.inf, np.float32)))
        attention = g.op("MatMul", g.op("Softmax", logits, axis=-1), v)
        attention = g.reshape(g.op("Transpose", attention, perm=[0, 2, 1, 3]), [1, -1, 512])
        x = g.op("Add", x, g.linear(attention, prefix_name + ".attn.out_proj", 512, 512))
        expanded = g.linear(g.norm(x, prefix_name + ".norm2"), prefix_name + ".ffn.linear1", 512, 2048)
        sqrt_two = np.float32(np.sqrt(2.))
        activated = g.op("Div", g.op("Mul", expanded,
                                      g.op("Add", g.const(np.array(1., np.float32)),
                                           g.op("Erf", g.op("Div", expanded, g.const(sqrt_two))))),
                                 g.const(np.array(2., np.float32)))
        x = g.op("Add", x, g.linear(activated, prefix_name + ".ffn.linear2", 2048, 512))
    x = g.norm(x, "encoder.final_norm")
    x = g.linear(x, "sortformer_modules.encoder_proj", 512, 192)
    x = g.op("Transpose", x, perm=[0, 2, 1])
    x = g.op("Conv", x,
                      g.weight("sortformer_modules.subpixel_upsample.weight", [1536, 3, 192], (0, 2, 1)),
                      g.weight("sortformer_modules.subpixel_upsample.bias", [1536]), pads=[1, 1])
    x = g.reshape(g.op("Transpose", x, perm=[0, 2, 1]), [1, -1, 192])
    x = g.linear(g.op("Relu", x), "sortformer_modules.first_hidden_to_hidden", 192, 192)
    x = g.linear(g.op("Relu", x), "sortformer_modules.single_hidden_to_spks", 192, 8)
    high = g.op("Sigmoid", x)
    high_positions = g.op("Range", g.const(np.array(0, np.int64)),
                          g.op("Mul", length, g.const(np.array(8, np.int64))),
                          g.const(np.array(1, np.int64)))
    high_mask = g.reshape(g.op("Less", high_positions, g.op("Mul", valid, g.const(np.array([8], np.int64)))),
                          [1, -1, 1])
    high = g.op("Where", high_mask, high, g.const(np.array(0., np.float32)), name="high")
    g.op("ReduceMean", g.reshape(high, [1, -1, 8, 8]), name="low", axes=[2], keepdims=0)
    for key in ["preprocessor.window", "preprocessor.fb", "sortformer_modules.learnable_sil_emb",
                "sortformer_modules.activity_head.layers.0.weight", "sortformer_modules.activity_head.layers.0.bias",
                "sortformer_modules.activity_head.layers.1.weight", "sortformer_modules.activity_head.layers.1.bias",
                "sortformer_modules.hidden_to_spks.weight", "sortformer_modules.hidden_to_spks.bias"]:
        del weights[key]
    if weights:
        raise ValueError(("unexpected weight inventory", list(weights)))
    model = g.model("Speakerdesk-pinned-Nemotron3-explicit-state", [
        value("features", [1, "frames", 128]), value("cache", [1, "cache_frames", 512]),
        value("fifo", [1, "fifo_frames", 512]), value("valid_frames", [1], TensorProto.INT64),
    ], [value("chunk", [1, "chunk_frames", 512]), value("high", [1, "high_frames", 8]),
        value("low", [1, "low_frames", 8])])
    path = output / "nemotron.onnx"
    onnx.save(model, path)
    provenance = {"weights": {"repo": pin["repo"], "revision": pin["revision"], "files": pin["files"]},
                  "precision": "float32", "onnx_version": onnx.__version__, "numpy_version": np.__version__,
                  "exporter_sha256": sha256(Path(__file__)), "model_sha256": sha256(path),
                  "frontend_sha256": sha256(output / "frontend.onnx"),
                  "silence_sha256": sha256(output / "silence.f32"),
                  "operators": sorted({n.op_type for n in model.graph.node})}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.model, args.output)
