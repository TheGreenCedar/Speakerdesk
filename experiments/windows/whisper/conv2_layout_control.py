"""Isolate Conv2 contraction/layout from GELU on one retained source input."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from export import Graph, sha256, value
from boundary_diagnostic import CANDIDATE_SHA256
from implicit_control import CAPTURES, EXTRA_CAPTURES, compare
from materialization_control import pinned_runtime
from rounding_control import half_grid


def gelu(g, x):
    scaled = half_grid(g, g.op("Mul", x, g.const(np.asarray(.70703125, np.float32))))
    erf = half_grid(g, g.op("Erf", scaled))
    gated = half_grid(g, g.op("Add", erf, g.const(np.asarray(1, np.float32))))
    product = half_grid(g, g.op("Mul", x, gated))
    result = half_grid(g, g.op("Div", product, g.const(np.asarray(2, np.float32))))
    return scaled, erf, result


def operator_model(weight, bias):
    g = Graph({})
    convolution = g.op("Conv", g.op("Transpose", "x", perm=[0, 2, 1]),
                       g.const(weight.astype(np.float32), "weight_ncl"), pads=[1, 1], strides=[2])
    convolution = g.op("Transpose", convolution, perm=[0, 2, 1])
    padded = g.op("Pad", "x", g.const(np.asarray([0, 1, 0, 0, 1, 0], np.int64)),
                  g.const(np.asarray(0, np.float32)), mode="constant")
    # NLC windows are [kernel position, input channel], unlike NCL Conv weights.
    indices = 2 * np.arange(1500, dtype=np.int64)[:, None] + np.arange(3, dtype=np.int64)[None]
    unfolded = g.op("Gather", padded, g.const(indices), axis=1)
    unfolded = g.reshape(unfolded, [1, 1500, 1152])
    nlc_weight = weight.transpose(0, 2, 1).reshape(384, 1152).T.copy()
    matrix = g.op("MatMul", unfolded, g.const(nlc_weight.astype(np.float32), "weight_nlc"))
    bias_name = g.const(bias.astype(np.float32), "bias")
    outputs = []
    for name, raw in [("conv", convolution), ("nlc", matrix)]:
        pre = half_grid(g, g.op("Add", half_grid(g, raw), bias_name))
        scaled, erf, post = gelu(g, pre)
        observations = [(name + "_bias", pre), (name + "_gelu", post)] if name == "conv" else [
            ("nlc_bias", pre), ("nlc_scaled", scaled), ("nlc_erf", erf), ("nlc_gelu", post)]
        for output, tensor in observations:
            g.op("Identity", tensor, name=output)
            outputs.append(value(output, [1, 1500, 384]))
    return g.model("Whisper-Conv2-layout-and-GELU-control", [value("x", [1, 3000, 384])], outputs)


def prepare(candidate, capture, extra, output):
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.gpu)
    runtime = pinned_runtime()
    if sha256(candidate / "whisper.onnx") != CANDIDATE_SHA256:
        raise ValueError("expected exact frozen source-rules-v1 graph")
    for directory, name, pin in [(capture, "conv1", CAPTURES["conv1"]),
                                  (extra, "pre", EXTRA_CAPTURES["pre"]), (extra, "post", EXTRA_CAPTURES["post"])]:
        if sha256(directory / (name + ".npy")) != pin:
            raise ValueError(("source operator capture differs", name))
    model = onnx.load(candidate / "whisper.onnx")
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    weight, bias = constants["encoder.conv2.weight"], constants["encoder.conv2.bias"]
    if weight.dtype != np.float16 or weight.shape != (384, 384, 3) or bias.dtype != np.float16:
        raise ValueError("unexpected pinned Conv2 parameter layout")
    source_pre = np.load(extra / "pre.npy", allow_pickle=False)
    source_post = np.load(extra / "post.npy", allow_pickle=False)
    half = mx.array(source_pre).astype(mx.float16)
    actual = np.asarray(nn.gelu(half).astype(mx.float32))
    if np.any(actual != source_post):
        raise ValueError("pinned compiled GELU does not reproduce the retained Conv2 output")
    scaled = half * mx.array(np.float16(.70703125))
    mx.eval(scaled)
    erf = mx.erf(scaled)
    mx.eval(erf)
    gated = erf + mx.array(np.float16(1))
    mx.eval(gated)
    product = half * gated
    mx.eval(product)
    recipe = product / mx.array(np.float16(2))
    mx.eval(recipe)
    if np.any(np.asarray(recipe.astype(mx.float32)) != source_post):
        raise ValueError("explicit source Half recipe differs from compiled Conv2 GELU")
    expected = {"source_pre": source_pre, "source_post": source_post,
                "source_scaled": np.asarray(scaled.astype(mx.float32)), "source_erf": np.asarray(erf.astype(mx.float32))}
    x = np.load(capture / "conv1.npy", allow_pickle=False)
    if np.any(x != x.astype(np.float16).astype(np.float32)):
        raise ValueError("source Conv1 output is not exactly Half-valued")
    graph = operator_model(weight, bias)
    output.mkdir(parents=True, exist_ok=False)
    case = output / "conv2"
    case.mkdir()
    onnx.save(graph, case / "control.onnx")
    x.astype("<f4").tofile(case / "x.f32")
    for name, array in expected.items():
        if array.dtype != np.float32 or array.shape != (1, 1500, 384) or not np.isfinite(array).all():
            raise ValueError("invalid source intermediate")
        array.astype("<f4").tofile(case / (name + ".f32"))
    aliases = {"conv_bias": "source_pre", "conv_gelu": "source_post", "nlc_bias": "source_pre",
               "nlc_scaled": "source_scaled", "nlc_erf": "source_erf", "nlc_gelu": "source_post"}
    declarations = {name: {"shape": [1, 1500, 384], "file": target + ".f32", "sha256": sha256(case / (target + ".f32"))}
                    for name, target in aliases.items()}
    (case / "expected.json").write_text(json.dumps(declarations, indent=2) + "\n")
    metadata = {"model": "control.onnx", "model_files": {name: sha256(case / name) for name in ["control.onnx", "expected.json"]},
                "repeat": 2, "disable_optimizations": True,
                "inputs": [{"name": "x", "dtype": "f32", "shape": [1, 3000, 384], "path": "x.f32"}]}
    (case / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
    local = {name: {"nonzero": int(np.count_nonzero(array != expected[aliases[name]])),
                    "max_error": float(np.max(abs(array - expected[aliases[name]])))}
             for name, array in zip(aliases, cpu.run(list(aliases), {"x": x}))}
    provenance = {"diagnostic_only": True, **runtime, "source_graph_sha256": CANDIDATE_SHA256,
        "graph_sha256": sha256(case / "control.onnx"), "case_sha256": sha256(case / "case.json"),
        "script_sha256": sha256(Path(__file__)), "input_capture_sha256": CAPTURES["conv1"],
        "source_pre_sha256": EXTRA_CAPTURES["pre"], "source_post_sha256": EXTRA_CAPTURES["post"],
        "source_half_gelu_recipe_bitexact": True, "local_ort": ort.__version__, "local": local,
        "case": "conv2/case.json", "whole_model_changed": False,
        "rule": "Identical retained Half-valued input/weights. Compare F32 NCL Conv with explicit NLC unfold F32 MatMul, each with contraction Half RNE, separate bias Half RNE, and reviewed Half GELU recipe.",
        "interpretation": "Only a pre-GELU difference supports a contraction hypothesis. If nlc_bias and nlc_scaled match but nlc_erf differs, the first isolated discrepancy is Erf rather than Conv. Numerical results are diagnostic; original product gates remain unchanged."}
    (output / "conv2.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["candidate", "capture", "extra", "output"]:
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
