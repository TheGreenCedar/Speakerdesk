"""Isolate native Half Conv1 from F32 convolution and explicit source rounding."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

from export import Graph, sha256, value
from implicit_control import CANDIDATE_SHA256, difference, write_case, verify
from rounding_control import half_grid, RUNTIME_SHA256, METALLIB_SHA256

CONVOLUTION_SHA256 = "d708d8321531e3ac20730fd174010118e8d8489498c602b36a9ab0af70900021"
BIAS_SHA256 = "b98f3d64abc45644cff8136ea509a8883adf7c355f25a3a85e17b0886de58332"


def operator_model(weight, bias, frames=3000):
    g = Graph({})
    half_weight = g.const(np.asarray(weight, np.float16), "half_weight")
    half_bias = g.const(np.asarray(bias, np.float16), "half_bias")
    float_weight = g.const(np.asarray(weight, np.float32), "float_weight")
    float_bias = g.const(np.asarray(bias, np.float32), "float_bias")
    # Native source path; observers keep native probe I/O Float32.
    native_input = g.op("Cast", "mel", to=TensorProto.FLOAT16)
    native = g.op("Conv", g.op("Transpose", native_input, perm=[0, 2, 1]), half_weight,
                  pads=[1, 1], strides=[1])
    native = g.op("Transpose", native, perm=[0, 2, 1])
    native_bias = g.op("Add", native, half_bias)
    g.op("Cast", native, name="native_convolution", to=TensorProto.FLOAT)
    g.op("Cast", native_bias, name="native_bias", to=TensorProto.FLOAT)
    # Inputs and weights are exactly Half-valued, represented in Float32.
    # Preserve convolution rounding before adding the bias, then bias rounding.
    floated = g.op("Conv", g.op("Transpose", "mel", perm=[0, 2, 1]), float_weight,
                   pads=[1, 1], strides=[1])
    floated = g.op("Transpose", floated, perm=[0, 2, 1])
    g.op("Identity", floated, name="f32_convolution")
    rounded = g.op("Cast", half_grid(g, floated), to=TensorProto.FLOAT16)
    rounded = g.op("Cast", rounded, name="corrected_convolution", to=TensorProto.FLOAT)
    biased = g.op("Add", rounded, float_bias)
    g.op("Cast", g.op("Cast", half_grid(g, biased), to=TensorProto.FLOAT16),
         name="corrected_bias", to=TensorProto.FLOAT)
    width, channels = len(bias), weight.shape[1]
    return g.model("Whisper-conv1-rounding-control", [value("mel", [1, frames, channels])],
        [value(name, [1, frames, width]) for name in ["native_convolution", "native_bias",
             "f32_convolution", "corrected_convolution", "corrected_bias"]])


def prepare(candidate, capture, output):
    if sha256(candidate / "whisper.onnx") != CANDIDATE_SHA256:
        raise ValueError("not the frozen nearest-half-v1 graph")
    receipt = json.loads((capture / "receipt.json").read_text())
    if (receipt["runtime_sha256"] != RUNTIME_SHA256 or
            receipt["colocated_metallib_sha256"] != METALLIB_SHA256 or
            receipt["with_bias_vs_pinned_capture_nonzero"] != 0):
        raise ValueError("convolution source does not reproduce the production-runtime capture")
    for filename, pin in [("convolution.npy", CONVOLUTION_SHA256), ("with-bias.npy", BIAS_SHA256)]:
        if sha256(capture / filename) != pin:
            raise ValueError(("source tensor differs", filename))
    mel_path = candidate / "cases/neural-silence/input.f32"
    if sha256(mel_path) != receipt["input_sha256"]:
        raise ValueError("source and native input identity differs")
    mel = np.fromfile(mel_path, "<f4").reshape(1, 3000, 80)
    if not np.all(mel == np.float32(-1.5)):
        raise ValueError("this control requires the frozen silence input")
    source = onnx.load(candidate / "whisper.onnx")
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in source.graph.initializer}
    weight = constants["encoder.conv1.weight"]
    bias = constants["encoder.conv1.bias"]
    if weight.dtype != np.float16 or bias.dtype != np.float16:
        raise ValueError("source weights are not Half")
    # The three source windows have exact binary16 input. Float64 computes the
    # finite contraction independently; no ONNX output supplies expectations.
    raw = []
    for frame in [0, 1, 2999]:
        window = np.zeros((80, 3), np.float64)
        for kernel in range(3):
            location = frame + kernel - 1
            if 0 <= location < 3000:
                window[:, kernel] = mel[0, location]
        raw.append(np.einsum("ock,ck->o", weight.astype(np.float64), window))
    classes = np.ones(3000, np.int64)
    classes[0], classes[-1] = 0, 2
    exact = np.asarray(raw)[classes][None]
    exact_half = exact.astype(np.float16).astype(np.float32)
    expected_convolution = np.load(capture / "convolution.npy", allow_pickle=False)
    expected_bias = np.load(capture / "with-bias.npy", allow_pickle=False)
    if np.any((exact_half.astype(np.float64) + bias).astype(np.float16).astype(np.float32) != expected_bias):
        raise ValueError("exact contraction and separate bias rounding differ from maintained MLX")
    model = operator_model(weight, bias)
    output.mkdir(parents=True, exist_ok=False)
    local = write_case(output, "conv1", model, {"mel": mel}, {
        "native_convolution": expected_convolution, "native_bias": expected_bias,
        "f32_convolution": exact.astype(np.float32),
        "corrected_convolution": expected_convolution, "corrected_bias": expected_bias})
    provenance = {"diagnostic_only": True, "source_graph_sha256": CANDIDATE_SHA256,
        "source_capture_receipt_sha256": sha256(capture / "receipt.json"),
        "script_sha256": sha256(Path(__file__)),
        "graph_sha256": sha256(output / "conv1/control.onnx"),
        "input_sha256": receipt["input_sha256"], "source_convolution_sha256": CONVOLUTION_SHA256,
        "source_bias_sha256": BIAS_SHA256, "runtime_sha256": RUNTIME_SHA256,
        "metallib_sha256": METALLIB_SHA256, "local_ort": ort.__version__, "local": local,
        "source_raw_vs_exact_half": difference(expected_convolution, exact_half),
        "cases": ["conv1/case.json"],
        "rule": "Exact same Half-valued input/weights; native Half Conv versus explicit F32 Conv with qualified RNE grid before Half, then separate bias and Half rounding. No whole graph changed.",
        "limits": "Source silence convolution only; F32 accumulator error is reported separately. Native Half F32 observers may change CPU promotion; native DML execution is required.",
        "operator_pass": "corrected_bias bitexact to the actual pinned production-MLX conv1-pre capture, deterministic repeats, requested provider exclusively executed. Raw contraction is diagnostic: source accumulation itself differs from exact F64 at one masked edge value."}
    (output / "convolution.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def compare(case, result, provider):
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("convolution control used CPU model work")
    expected = json.loads((case / "expected.json").read_text())
    if len(record["outputs"]) != len(expected) * 2 or {(row["run"], row["name"]) for row in record["outputs"]} != {
            (run, name) for run in [0, 1] for name in expected}:
        raise ValueError("convolution control output inventory differs")
    first, reports, passed = {}, [], True
    for row in record["outputs"]:
        spec = expected[row["name"]]
        if sha256(case / spec["file"]) != spec["sha256"]:
            raise ValueError("source convolution expectation identity differs")
        if row["dtype"] != "f32" or row["shape"] != spec["shape"] or sha256(result / row["file"]) != row["sha256"]:
            raise ValueError("native convolution output identity differs")
        actual = np.fromfile(result / row["file"], "<f4").reshape(spec["shape"])
        wanted = np.fromfile(case / spec["file"], "<f4").reshape(spec["shape"])
        first.setdefault(row["name"], actual)
        repeat = difference(actual, first[row["name"]])["nonzero"]
        report = {"run": row["run"], "name": row["name"], **difference(actual, wanted), "repeat_nonzero": repeat}
        reports.append(report)
        passed &= repeat == 0
        if row["name"] == "corrected_bias":
            passed &= report["nonzero"] == 0
    print(json.dumps({"diagnostic_only": True, "corrected_operator_bitexact": bool(passed), "reports": reports}, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["candidate", "capture", "output"]:
        build.add_argument(name, type=Path)
    comparison = commands.add_parser("compare")
    for name in ["case", "result"]:
        comparison.add_argument(name, type=Path)
    comparison.add_argument("--provider", choices=["cpu", "dml"], required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    else:
        raise SystemExit(0 if compare(**args) else 1)
