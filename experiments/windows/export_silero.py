"""Build-only export of the pinned Silero graph. Never used by the application."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import wave

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(weights_path, output):
    root = Path(__file__).resolve().parents[2]
    pin = json.loads((root / "crates/host/src/model-artifacts.json").read_text())[0]
    expected = pin["files"]["model.safetensors"]
    assert weights_path.stat().st_size == expected["size"]
    assert sha256(weights_path) == expected["sha256"], "unpinned Silero weights"
    data = weights_path.read_bytes()
    header_size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + header_size])
    weights = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        assert item["dtype"] == "F32"
        start, end = item["data_offsets"]
        weights[name.removeprefix("vad_16k.")] = np.frombuffer(
            data[8 + header_size + start:8 + header_size + end], dtype="<f4"
        ).reshape(item["shape"])

    nodes, initializers = [], []

    def constant(name, values):
        initializers.append(numpy_helper.from_array(np.asarray(values), name))
        return name

    def op(kind, inputs, outputs, **attrs):
        if isinstance(outputs, str):
            outputs = [outputs]
        nodes.append(helper.make_node(kind, inputs, outputs, name=outputs[0], **attrs))

    def weight(name, shape, transpose=None):
        array = weights.pop(name)
        assert list(array.shape) == shape, (name, array.shape)
        if transpose is not None:
            array = array.transpose(transpose).copy()
        return constant(name, array)

    def conv(source, name, shape, stride, padding, bias=True):
        inputs = [source, weight(name + ".weight", shape, (0, 2, 1))]
        if bias:
            inputs.append(weight(name + ".bias", [shape[0]]))
        op("Conv", inputs, name, strides=[stride], pads=[padding, padding])
        return name

    spectrum = conv("audio", "stft_conv", [258, 256, 1], 128, 0, False)
    halves = constant("spectrum_halves", np.array([129, 129], dtype=np.int64))
    op("Split", [spectrum, halves], ["real", "imaginary"], axis=1)
    op("Mul", ["real", "real"], "real_squared")
    op("Mul", ["imaginary", "imaginary"], "imaginary_squared")
    op("Add", ["real_squared", "imaginary_squared"], "power")
    op("Sqrt", ["power"], "magnitude")
    x = "magnitude"
    for name, shape, stride in [
        ("conv1", [128, 3, 129], 1), ("conv2", [64, 3, 128], 2),
        ("conv3", [64, 3, 64], 2), ("conv4", [128, 3, 64], 1),
    ]:
        x = conv(x, name, shape, stride, 1)
        op("Relu", [x], name + "_relu")
        x = name + "_relu"
    flat_shape = constant("flat_shape", np.array([1, 128], dtype=np.int64))
    op("Reshape", [x, flat_shape], "encoded")
    op("Gemm", ["encoded", weight("lstm.Wx", [512, 128]),
                weight("lstm.bias", [512])], "input_gates", transB=1)
    op("Gemm", ["hidden", weight("lstm.Wh", [512, 128]), "input_gates"],
       "all_gates", transB=1)
    quarters = constant("gate_quarters", np.full(4, 128, dtype=np.int64))
    op("Split", ["all_gates", quarters], ["i0", "f0", "g0", "o0"], axis=1)
    for gate, kind in [("i", "Sigmoid"), ("f", "Sigmoid"),
                       ("g", "Tanh"), ("o", "Sigmoid")]:
        op(kind, [gate + "0"], gate)
    op("Mul", ["f", "cell"], "retained_cell")
    op("Mul", ["i", "g"], "new_cell")
    op("Add", ["retained_cell", "new_cell"], "cell_out")
    op("Tanh", ["cell_out"], "cell_activation")
    op("Mul", ["o", "cell_activation"], "hidden_out")
    op("Relu", ["hidden_out"], "hidden_relu")
    conv_shape = constant("conv_shape", np.array([1, 128, 1], dtype=np.int64))
    op("Reshape", ["hidden_relu", conv_shape], "output_input")
    logits = conv("output_input", "final_conv", [1, 1, 128], 1, 0)
    op("Sigmoid", [logits], "probability")
    assert not weights, list(weights)
    value = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    graph = helper.make_graph(nodes, "Speakerdesk-pinned-Silero-v6-16k", [
        value("audio", [1, 1, 640]), value("hidden", [1, 128]), value("cell", [1, 128])
    ], [value("probability", [1, 1, 1]), value("hidden_out", [1, 128]),
        value("cell_out", [1, 128])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="speakerdesk-experimental-export", ir_version=9)
    onnx.checker.check_model(model, full_check=True)
    output.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output / "silero.onnx")

    fixture = root / "crates/models/tests/data/two-voice.wav"
    reference_path = root / "crates/models/tests/data/silero-reference.json"
    reference = json.loads(reference_path.read_text())
    assert sha256(fixture) == reference["wav_sha256"]
    with wave.open(str(fixture), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    assert len(pcm) == reference["samples"]
    states = [np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32)]
    frames = []
    for index, expected_frame in enumerate(reference["frames"]):
        frame = np.zeros(512, dtype=np.float32)
        chunk = pcm[index * 512:(index + 1) * 512]
        frame[:len(chunk)] = chunk
        peak = float(np.max(np.abs(chunk)))
        gain = min(256.0, max(1.0, 0.25 / peak)) if peak else 1.0
        assert gain == expected_frame["gain"]
        views = []
        for view, scale in enumerate([1.0, gain]):
            scaled = frame * np.float32(scale)
            joined = np.concatenate([states[view], scaled])
            views.append(np.concatenate([joined, joined[-2:-66:-1]]))
            states[view] = scaled[-64:].copy()
        frames.append((views, [expected_frame["raw_p"], expected_frame["norm_p"]]))
    with (output / "frames.bin").open("wb") as stream:
        stream.write(b"SDVAD001" + struct.pack("<I", len(frames)))
        for views, expected_frame in frames:
            for view in views:
                stream.write(view.astype("<f4").tobytes())
            stream.write(struct.pack("<ff", *expected_frame))

    provenance = {
        "purpose": "Disposable Windows feasibility probe; not an application backend",
        "weights": {"repo": pin["repo"], "revision": pin["revision"], **expected},
        "reference_sha256": sha256(reference_path), "wav_sha256": sha256(fixture),
        "exporter_sha256": sha256(Path(__file__)), "onnx_version": onnx.__version__,
        "numpy_version": np.__version__, "frames": len(frames), "samples": len(pcm),
        "outputs": {name: sha256(output / name) for name in ["silero.onnx", "frames.bin"]},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.weights, args.output)
