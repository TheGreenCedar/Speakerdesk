"""One same-input frontend LayerNorm control; no full-model candidate."""
import argparse
from collections import Counter
import json
from pathlib import Path, PureWindowsPath
import struct
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from gelu_control import CAPTURE_SHA256, ROOT, SHAPE, SOURCE_GRAPH_SHA256, digest, metrics

PINS = {
    "input.f32": ("conv0_raw.f32", "cfb50f7c4a1132568e5fa7dbbbfba0337d9c1a012daf7aa1804a30e22ff9435d"),
    "source.f32": ("conv0_norm.f32", "740869f4dd5e02ea8a727ffaf1b2a7bb26019d19c788b975c577494afb1aacc1"),
    "native-cpu.f32": ("native-cpu-conv0_norm.f32", "455d09b3704388c95deec70904101ff662ed041ce15181031f9d61b36b493e89"),
    "native-dml.f32": ("native-dml-conv0_norm.f32", "654b7bbb780fc6a062d3370dc665130a2a79f46099b9587656f057190df1f7d9"),
}
PARAMETERS = {
    "gamma": ("weight", "56febfd09142ffd702c40e46f7693d6c7f8f657079cd38c924ea27a7bb801479"),
    "beta": ("bias", "28491358be534cf642940d283abfac5222fb3d0749b039a7c53dc36e1dcf2781"),
}
SMALL = [1, SHAPE[1], 1]
OUTPUT_SHAPES = {name: SMALL if name in ("mean", "variance", "variance_eps", "denominator") else SHAPE
                 for name in ("mean", "centered", "squared", "variance", "variance_eps", "denominator",
                              "unit", "affine_scaled", "explicit_norm", "fused_norm")}


def source_parameters(path):
    result = {}
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > 1 << 20:
            raise ValueError("unexpected safetensors header size")
        header = json.loads(stream.read(header_size))
        for name, (suffix, pin) in PARAMETERS.items():
            row = header["wav2vec2.feature_extractor.conv_layers.0.layer_norm." + suffix]
            if row["dtype"] != "F32" or row["shape"] != [512]:
                raise ValueError("prepared norm parameter geometry differs")
            begin, end = row["data_offsets"]
            if end - begin != 2048:
                raise ValueError("prepared norm parameter byte range differs")
            stream.seek(8 + header_size + begin)
            data = stream.read(2048)
            if digest(data) != pin:
                raise ValueError("prepared norm parameter identity differs")
            result[name] = np.frombuffer(data, "<f4").copy()
    return result


def make_graph(parameters):
    for name, (_, pin) in PARAMETERS.items():
        if parameters[name].shape != (512,) or digest(parameters[name].astype("<f4").tobytes()) != pin:
            raise ValueError("norm control requires the retained original parameter bytes")
    nodes = [
        helper.make_node("ReduceMean", ["x"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["x", "mean"], ["centered"]),
        helper.make_node("Pow", ["centered", "two"], ["squared"]),
        helper.make_node("ReduceMean", ["squared"], ["variance"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["variance", "epsilon"], ["variance_eps"]),
        helper.make_node("Sqrt", ["variance_eps"], ["denominator"]),
        helper.make_node("Div", ["centered", "denominator"], ["unit"]),
        helper.make_node("Mul", ["unit", "gamma"], ["affine_scaled"]),
        helper.make_node("Add", ["affine_scaled", "beta"], ["explicit_norm"]),
        helper.make_node("LayerNormalization", ["x", "gamma", "beta"], ["fused_norm"],
                         axis=-1, epsilon=float(np.float32(1e-5)), stash_type=TensorProto.FLOAT),
    ]
    initializers = [numpy_helper.from_array(value, name) for name, value in parameters.items()]
    initializers += [numpy_helper.from_array(np.asarray(value, np.float32), name)
                     for name, value in (("two", 2), ("epsilon", 1e-5))]
    graph = helper.make_graph(nodes, "first-frontend-layernorm-same-input",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)],
                              [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                               for name, shape in OUTPUT_SHAPES.items()], initializers)
    model = helper.make_model(graph, ir_version=7, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def prepare(source_graph, safetensors, captures, output):
    if digest(source_graph.read_bytes()) != SOURCE_GRAPH_SHA256:
        raise ValueError("original CTC graph identity differs")
    if digest((captures / "capture.json").read_bytes()) != CAPTURE_SHA256:
        raise ValueError("unchanged source capture identity differs")
    data = {name: (captures / original).read_bytes() for name, (original, _) in PINS.items()}
    if any(digest(data[name]) != pin for name, (_, pin) in PINS.items()):
        raise ValueError("retained input/source/native tensor identity differs")
    parameters = source_parameters(safetensors)
    model = make_graph(parameters).SerializeToString()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in data.items():
        (output / name).write_bytes(value)
    (output / "norm.onnx").write_bytes(model)
    case = dict(model="norm.onnx", model_files={"norm.onnx": digest(model)}, repeat=2,
                disable_optimizations=True,
                inputs=[dict(name="x", dtype="f32", shape=SHAPE, path="input.f32")])
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    provenance = dict(source_graph_sha256=SOURCE_GRAPH_SHA256, capture_sha256=CAPTURE_SHA256,
                      parameters={name: pin for name, (_, pin) in PARAMETERS.items()},
                      tensors={name: pin for name, (_, pin) in PINS.items()}, graph_sha256=digest(model),
                      tool_sha256=digest(Path(__file__).read_bytes()),
                      distinct_captured_rows=1,
                      limit="One distinct captured vector repeated 96 times; no whole-model qualification or new numerical gate")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def read_outputs(record, directory):
    runs = [{}, {}]
    for row in record["outputs"]:
        index, name = row["run"], row["name"]
        if (index not in (0, 1) or name not in OUTPUT_SHAPES or name in runs[index]
                or row["dtype"] != "f32" or row["shape"] != OUTPUT_SHAPES[name]):
            raise ValueError("unexpected or duplicate norm control output")
        data = (directory / row["file"]).read_bytes()
        if digest(data) != row["sha256"]:
            raise ValueError("norm output hash differs")
        value = np.frombuffer(data, "<f4").reshape(row["shape"])
        if not np.isfinite(value).all():
            raise ValueError("nonfinite norm control output")
        runs[index][name] = value
    if any(set(run) != set(OUTPUT_SHAPES) for run in runs):
        raise ValueError("both complete norm output sets are required")
    for name in OUTPUT_SHAPES:
        if metrics(runs[0][name], runs[1][name])["unequal_bits"]:
            raise ValueError("norm output did not repeat bit-exactly")
    return runs


def compare(case, result, provider):
    sys.path.insert(0, str(ROOT / "experiments/windows"))
    from verify_probe import verify
    record = verify(case, result, provider)
    graph_data = (case / "norm.onnx").read_bytes()
    graph = onnx.load_model_from_string(graph_data)
    parameters = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.graph.initializer
                  if tensor.name in PARAMETERS}
    if digest(graph_data) != digest(make_graph(parameters).SerializeToString()):
        raise ValueError("norm control graph differs")
    if any(digest((case / name).read_bytes()) != pin for name, (_, pin) in PINS.items()):
        raise ValueError("retained norm control tensor differs")
    if record.get("graph_optimizations_disabled") is not True:
        raise ValueError("this control requires disabled ORT graph optimizations")
    events = json.loads((result / PureWindowsPath(record["profile"]).name).read_text())
    placements = Counter((event.get("args", {}).get("op_name"), event.get("args", {}).get("provider"))
                         for event in events if event.get("args", {}).get("provider"))
    if provider == "dml" and any(ep == "CPUExecutionProvider" for _, ep in placements):
        raise ValueError("a norm control branch fell back to CPU")
    runs = read_outputs(record, result)
    wanted = np.fromfile(case / "source.f32", "<f4").reshape(SHAPE)
    retained_native = np.fromfile(case / f"native-{provider}.f32", "<f4").reshape(SHAPE)
    reports = []
    for index, run in enumerate(runs):
        # These are primitive Float32 reassemblies, not a source-kernel oracle.
        reassembled = run["unit"] * parameters["gamma"] + parameters["beta"]
        reports.append(dict(run=index, versus_source={name: metrics(run[name], wanted)
                            for name in ("explicit_norm", "fused_norm")},
                            isolated_explicit_vs_original_native=metrics(run["explicit_norm"], retained_native),
                            explicit_vs_affine_reassembly=metrics(run["explicit_norm"], reassembled),
                            variance_min=float(run["variance"].min()), variance_max=float(run["variance"].max())))
    report = dict(provider=provider, reports=reports,
                  executed_operators=[dict(operator=op, provider=ep, events=count)
                                      for (op, ep), count in placements.items()],
                  limit="One unique input vector. No captured source mean/inverse-std/affine intermediate exists, and the standard operator does not prescribe MLX's SIMD reduction tree. Full score gate remains unchanged and unqualified.")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("prepare")
    for name in ("source_graph", "safetensors", "captures", "output"):
        build.add_argument(name, type=Path)
    check = modes.add_parser("compare")
    check.add_argument("case", type=Path)
    check.add_argument("result", type=Path)
    check.add_argument("--provider", choices=("cpu", "dml"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.source_graph, args.safetensors, args.captures, args.output)
    else:
        compare(args.case, args.result, args.provider)
