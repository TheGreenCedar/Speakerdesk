"""One pinned-runtime matrix for source Half materializations, not a full model."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

from export import Graph, ROOT, sha256, value
from implicit_control import CANDIDATE_SHA256, CAPTURES, EXTRA_CAPTURES, difference, write_case, verify
from rounding_control import half_grid, RUNTIME_SHA256, METALLIB_SHA256


SOURCE_PINS = {
    "layer_norm": {"url": "https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/layer_norm.metal", "sha256": "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"},
    "gemv": {"url": "https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/gemv.h", "sha256": "296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65"},
    "gemm": {"url": "https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h", "sha256": "ed49c8b9c73957f0cbc9dea30f484cf84fb821aece68233d6d1967856200d5c7"},
}


def half_fma(g, x, weight, bias):
    """One Half fused affine round, including F32 double-round midpoint repair.

    Half products are exactly representable in F32. TwoSum recovers the low
    part of the exact product+bias sum. A nonzero low part matters only when
    the F32 sum is exactly a Half midpoint. Reflecting across its rounded Half
    value identifies the other Half neighbor without another exponent rule.
    Contract: Half-valued finite operands; abs(product+bias)<65504. This is
    the bounded Whisper affine range, not an overflow conversion routine.
    """
    product = g.op("Mul", x, weight)
    high = g.op("Add", product, bias)
    virtual_bias = g.op("Sub", high, product)
    low = g.op("Add", g.op("Sub", product, g.op("Sub", high, virtual_bias)),
               g.op("Sub", bias, virtual_bias))
    rounded = half_grid(g, high)
    delta = g.op("Sub", high, rounded)
    other = g.op("Add", high, delta)
    zero = g.const(np.asarray(0, np.float32))
    midpoint = g.op("And", g.op("Equal", other, half_grid(g, other)),
                    g.op("Not", g.op("Equal", delta, zero)))
    away = g.op("Greater", g.op("Mul", low, delta), zero)
    return g.op("Where", g.op("And", midpoint, away), other, rounded)


def norm(g, x, weight, bias):
    mean = g.op("ReduceMean", x, axes=[-1], keepdims=1)
    centered = g.op("Sub", x, mean)
    variance = g.op("ReduceMean", g.op("Mul", centered, centered), axes=[-1], keepdims=1)
    scale = g.op("Div", g.const(np.asarray(1, np.float32)),
                 g.op("Sqrt", g.op("Add", variance, g.const(np.asarray(1e-5, np.float32)))))
    normalized = half_grid(g, g.op("Mul", centered, scale))
    return half_fma(g, normalized, weight, bias)


def operator_model(inputs):
    g = Graph({})
    out = []
    def emit(name, tensor, shape):
        g.op("Identity", tensor, name=name)
        out.append(value(name, shape))
    def half(x):
        return g.op("Cast", x, to=TensorProto.FLOAT16)
    def full(x):
        return g.op("Cast", x, to=TensorProto.FLOAT)
    for operation, kind in [("add", "Add"), ("mul", "Mul")]:
        left, right = operation + "_a", operation + "_b"
        emit(operation + "_native", full(g.op(kind, half(left), half(right))), inputs[left].shape)
        emit(operation + "_corrected", half_grid(g, g.op(kind, left, right)), inputs[left].shape)
    # The exact lattice inputs make these contractions independent of F32
    # reduction order while exercising Half ties and cancellation.
    raw = g.op("MatMul", "matrix", "weight")
    native = g.op("MatMul", half("matrix"), half("weight"))
    shape = [inputs["matrix"].shape[0], inputs["weight"].shape[1]]
    emit("matmul_native", full(native), shape)
    emit("matmul_corrected", half_grid(g, raw), shape)
    emit("encoder_native", full(g.op("Add", native, half("bias"))), shape)
    emit("encoder_corrected", half_grid(g, g.op("Add", raw, "bias")), shape)
    raw_decoder = g.op("MatMul", "decoder", "weight")
    native_decoder = g.op("MatMul", half("decoder"), half("weight"))
    emit("decoder_native", full(g.op("Add", native_decoder, half("bias"))), [1, shape[1]])
    emit("decoder_corrected", half_grid(g, g.op("Add", half_grid(g, raw_decoder), "bias")), [1, shape[1]])
    for prefix in ["norm", "captured_norm"]:
        x, weight, bias = prefix + "_x", prefix + "_weight", prefix + "_bias"
        emit(prefix + "_native", full(g.op("LayerNormalization", half(x), half(weight), half(bias),
             axis=-1, epsilon=1e-5)), inputs[x].shape)
        emit(prefix + "_corrected", norm(g, x, weight, bias), inputs[x].shape)
    # Use the source normalized Half values as an independent input to prove
    # fused affine semantics separately from any reduction-order differences.
    emit("affine_ordinary", half_grid(g, g.op("Add", g.op("Mul", "norm_normalized", "norm_weight"),
             "norm_bias")), inputs["norm_x"].shape)
    emit("affine_corrected", half_fma(g, "norm_normalized", "norm_weight", "norm_bias"), inputs["norm_x"].shape)
    return g.model("Whisper-source-materialization-matrix", [value(name, array.shape) for name, array in inputs.items()], out)


def pinned_runtime():
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    loaded = [Path(dyld._dyld_get_image_name(i).decode()) for i in range(dyld._dyld_image_count())
              if dyld._dyld_get_image_name(i).decode().endswith("/libmlx.dylib")]
    if len(loaded) != 1 or sha256(loaded[0]) != RUNTIME_SHA256 or sha256(loaded[0].with_name("mlx.metallib")) != METALLIB_SHA256:
        raise ValueError("source matrix requires the pinned production MLX library and kernels")
    return {"library": str(loaded[0]), "runtime_sha256": RUNTIME_SHA256, "metallib_sha256": METALLIB_SHA256}


def prepare(candidate, retained, extra, output):
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    runtime = pinned_runtime()
    if sha256(candidate / "whisper.onnx") != CANDIDATE_SHA256:
        raise ValueError("source model graph identity differs")
    for directory, name, pin in [(retained, "added_positions", CAPTURES["added_positions"]),
                                  (extra, "first_norm", EXTRA_CAPTURES["first_norm"])]:
        if sha256(directory / (name + ".npy")) != pin:
            raise ValueError("retained source LayerNorm identity differs")
    rng = np.random.default_rng(0x48414C46)
    positive = np.arange(0x4c01, dtype=np.uint16).view(np.float16)
    values = np.concatenate([positive, -positive]).astype(np.float32)
    left = np.concatenate([values, values, values])
    right = np.concatenate([-values, np.roll(values, 751), np.roll(values, 12913)])
    matrix = (rng.integers(-32, 33, (1500, 384)) / 32).astype(np.float32)
    weight = (rng.integers(-32, 33, (384, 384)) / 32).astype(np.float32)
    bias = np.linspace(-.9, .9, 384).astype(np.float16).astype(np.float32)
    normal = (rng.integers(-16, 17, (32, 384)) / 8).astype(np.float32)
    normal[1] = 0
    normal[2] = 4
    normal[3] = np.tile([4 + 2**-7, 4 - 2**-7], 192)
    normal[4] = np.tile([2**-24, -2**-24], 192)
    normalized = np.asarray(mx.fast.layer_norm(mx.array(normal).astype(mx.float16), None, None, 1e-5).astype(mx.float32))
    norm_weight = np.ones(384, np.float32)
    norm_bias = np.full(384, .125, np.float32)
    choices = np.arange(0x3800, 0x4c01, dtype=np.uint16).view(np.float16).astype(np.float64)
    witnesses = 0
    for channel, x in enumerate(normalized[0].astype(np.float64)):
        product = x * choices
        for tiny in [2**-24, -2**-24]:
            exact = (product + tiny).astype(np.float16)
            ordinary = (product.astype(np.float32) + np.float32(tiny)).astype(np.float16)
            found = np.flatnonzero(exact != ordinary)
            if len(found):
                norm_weight[channel], norm_bias[channel] = choices[found[0]], tiny
                witnesses += 1
                break
    if witnesses < 100:
        raise ValueError("affine double-round hostile class was not constructed")
    source = onnx.load(candidate / "whisper.onnx")
    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in source.graph.initializer}
    inputs = {"add_a": left, "add_b": right, "mul_a": left, "mul_b": right,
              "matrix": matrix, "weight": weight, "bias": bias, "decoder": matrix[:1].copy(),
              "norm_x": normal, "norm_weight": norm_weight, "norm_bias": norm_bias,
              "norm_normalized": normalized, "captured_norm_x": np.load(retained / "added_positions.npy"),
              "captured_norm_weight": constants["encoder.blocks.0.attn_ln.weight"].astype(np.float32),
              "captured_norm_bias": constants["encoder.blocks.0.attn_ln.bias"].astype(np.float32)}
    if any(np.any(x != x.astype(np.float16).astype(np.float32)) for x in inputs.values()):
        raise ValueError("operator input is not exactly Half-valued")
    arrays = {name: mx.array(x).astype(mx.float16) for name, x in inputs.items()}
    source_outputs = {
        "add": arrays["add_a"] + arrays["add_b"], "mul": arrays["mul_a"] * arrays["mul_b"],
        "matmul": arrays["matrix"] @ arrays["weight"],
        "encoder": mx.addmm(arrays["bias"], arrays["matrix"], arrays["weight"]),
        "decoder": mx.addmm(arrays["bias"], arrays["decoder"], arrays["weight"]),
        "norm": mx.fast.layer_norm(arrays["norm_x"], arrays["norm_weight"], arrays["norm_bias"], 1e-5),
        "captured_norm": mx.fast.layer_norm(arrays["captured_norm_x"], arrays["captured_norm_weight"], arrays["captured_norm_bias"], 1e-5)}
    mx.eval(*source_outputs.values())
    captured = {name: np.asarray(x.astype(mx.float32)) for name, x in source_outputs.items()}
    if np.any(captured["captured_norm"] != np.load(extra / "first_norm.npy")):
        raise ValueError("actual pinned runtime does not reproduce retained LayerNorm")
    exact_affine = (normalized.astype(np.float64) * norm_weight.astype(np.float64) + norm_bias.astype(np.float64)).astype(np.float16).astype(np.float32)
    if np.any(exact_affine != captured["norm"]):
        raise ValueError("maintained LayerNorm does not use normalized-Half followed by fused Half affine")
    expected = {name + suffix: array for name, array in captured.items() for suffix in ["_native", "_corrected"]}
    expected.update(affine_ordinary=captured["norm"], affine_corrected=captured["norm"])
    output.mkdir(parents=True, exist_ok=False)
    local = write_case(output, "operators", operator_model(inputs), inputs, expected)
    case_path = output / "operators/case.json"
    case_metadata = json.loads(case_path.read_text())
    # Bind source expectation metadata through the existing declared-file
    # inventory as well as the arrays' own recorded hashes.
    case_metadata["model_files"]["expected.json"] = sha256(output / "operators/expected.json")
    case_path.write_text(json.dumps(case_metadata, indent=2) + "\n")
    manifest = {"diagnostic_only": True, **runtime, "source_graph_sha256": CANDIDATE_SHA256, "primary_sources": SOURCE_PINS,
        "independent_product_reference_sha256": sha256(ROOT / "crates/models/tests/data/whisper-reference.json"),
        "script_sha256": sha256(Path(__file__)), "grid_sha256": sha256(Path(__file__).with_name("rounding_control.py")),
        "source_affine_fma_matches_exact": True, "affine_hostile_channels": witnesses,
        "local_ort": ort.__version__, "local": local,
        "case": "operators/case.json", "source_capture_files": {name: sha256(output / "operators" / (name + ".expected.f32")) for name in expected},
        "rules": {"add_mul": "F32 operation then Half RNE", "matmul": "F32 accumulation then Half RNE",
                  "encoder_addmm": "F32 dot plus bias then Half RNE", "decoder_addmm": "F32 dot then Half RNE, followed by bias then Half RNE",
                  "layer_norm": "F32 centered statistics and normalization, Half RNE, then one fused Half affine RNE; TwoSum repairs meaningful double-round ties"},
        "finite_affine_contract": "Half-valued operands, abs(product+bias)<65504; no overflow class is emitted to the native probe.",
        "whole_model_changed": False, "native_qualified": False}
    (output / "materialization.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def compare(case, result, provider):
    record = verify(case, result, provider)
    if provider == "dml" and set(record["executed_provider_events"]) != {"DmlExecutionProvider"}:
        raise ValueError("source materialization matrix used CPU model work")
    expected = json.loads((case / "expected.json").read_text())
    if len(record["outputs"]) != len(expected) * 2 or {(row["run"], row["name"]) for row in record["outputs"]} != {
            (run, name) for run in [0, 1] for name in expected}:
        raise ValueError("materialization output inventory differs")
    first, reports, deterministic, bitexact = {}, [], True, True
    for row in record["outputs"]:
        spec = expected[row["name"]]
        if sha256(case / spec["file"]) != spec["sha256"]:
            raise ValueError("pinned CPU source expectation identity differs")
        if row["dtype"] != "f32" or row["shape"] != spec["shape"] or sha256(result / row["file"]) != row["sha256"]:
            raise ValueError("native matrix output identity differs")
        actual = np.fromfile(result / row["file"], "<f4").reshape(spec["shape"])
        wanted = np.fromfile(case / spec["file"], "<f4").reshape(spec["shape"])
        first.setdefault(row["name"], actual)
        repeat = difference(actual, first[row["name"]])["nonzero"]
        report = {"run": row["run"], "name": row["name"], **difference(actual, wanted), "repeat_nonzero": repeat}
        reports.append(report)
        deterministic &= repeat == 0
        if row["name"].endswith("_corrected"):
            bitexact &= report["nonzero"] == 0
    print(json.dumps({"diagnostic_only": True, "source_materializations_bitexact": bool(bitexact),
                      "deterministic": bool(deterministic), "product_gate_unchanged": "strict max probability error <1e-4",
                      "reports": reports}, indent=2))
    return deterministic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ["candidate", "retained", "extra", "output"]:
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
