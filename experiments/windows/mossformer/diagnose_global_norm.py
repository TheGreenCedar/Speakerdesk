"""Offline F64 GroupNorm oracle on already captured finish taps; no inference.

Each runtime output is checked against its own saved normalization input.
Learned affine values and epsilon come from the exact frozen finish graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from diagnose_finish import mapped_float
from export import ROOT, SOURCE_FILES, SOURCE_REVISION, json_write, sha256


SOURCE_GRAPH = "a37b2265f942ad8247aced7ba48562e7ad0ba3b8f2df836c0768e9b6be531ad4"
ORT_KERNEL = "https://github.com/microsoft/onnxruntime/blob/df2ba1cf81/onnxruntime/core/providers/cpu/nn/instance_norm.cc#L33-L41"
SOURCE_NORM = "https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/clearvoice/clearvoice/models/mossformer2_ss/mossformer2.py#L128-L137"


def moments(values):
    """Two F64 passes, avoiding a full-window F64 temporary."""
    flat = values.reshape(-1)
    total = 0.0
    for start in range(0, flat.size, 1_000_000):
        chunk = flat[start:start + 1_000_000].astype(np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("nonfinite normalization input")
        total += float(chunk.sum(dtype=np.float64))
    mean = total / flat.size
    squares = 0.0
    for start in range(0, flat.size, 1_000_000):
        centered = flat[start:start + 1_000_000].astype(np.float64) - mean
        squares += float(np.dot(centered, centered))
    return mean, squares / flat.size


def own_input_oracle(intra, actual, gamma, beta, epsilon):
    # The captured input is [B,T,C]; source GroupNorm consumes its [B,C,T] transpose.
    if (intra.ndim != 3 or intra.shape[0] != 1 or intra.shape[1] < 1 or
            actual.shape != (1, intra.shape[2], intra.shape[1]) or
            gamma.shape != (intra.shape[2],) or beta.shape != gamma.shape or
            not np.isfinite(gamma).all() or not np.isfinite(beta).all()):
        raise ValueError("incompatible normalization/affine tensor shapes or values")
    mean, variance = moments(intra)
    inverse_std = 1.0 / np.sqrt(variance + epsilon)
    gamma, beta = gamma.astype(np.float64), beta.astype(np.float64)
    normal = np.zeros((2, 2), np.float64)
    rhs = np.zeros(2, np.float64)
    squared = reference_squared = maximum = 0.0
    frames = intra.shape[1]
    for start in range(0, frames, 2048):
        x = intra[0, start:start + 2048, :].astype(np.float64)
        y = actual[0, :, start:start + 2048].T.astype(np.float64)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite normalization output")
        centered_affine = (x - mean) * gamma
        oracle = centered_affine * inverse_std + beta
        delta = y - oracle
        squared += float(np.sum(delta * delta))
        reference_squared += float(np.sum(oracle * oracle))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        # Fit y-beta = alpha * ((x-mean64)*gamma) + shift*gamma.
        # Centering first avoids an ill-conditioned fit for large input offsets.
        target = y - beta
        normal[0, 0] += np.sum(centered_affine * centered_affine)
        normal[0, 1] += np.sum(centered_affine * gamma)
        normal[1, 1] += x.shape[0] * np.sum(gamma * gamma)
        rhs[0] += np.sum(centered_affine * target)
        rhs[1] += np.sum(target * gamma)
    normal[1, 0] = normal[0, 1]
    alpha, shift = np.linalg.solve(normal, rhs)
    fit_squares = 0.0
    for start in range(0, frames, 2048):
        x = intra[0, start:start + 2048, :].astype(np.float64)
        y = actual[0, :, start:start + 2048].T.astype(np.float64)
        delta = y - (((x - mean) * alpha + shift) * gamma + beta)
        fit_squares += float(np.sum(delta * delta))
    return dict(elements=int(intra.size), mean_f64=mean, centered_variance_f64=variance,
        inverse_std_f64=float(inverse_std), epsilon=epsilon,
        output_vs_own_input_f64=dict(max_abs_error=maximum, rms_error=float(np.sqrt(squared / intra.size)),
            normalized_rms_error=float(np.sqrt(squared / max(reference_squared, 1e-20)))),
        fitted_scalar_normalization=dict(inverse_std=float(alpha), scale_ratio_to_f64=float(alpha / inverse_std),
            implied_mean=float(mean - shift / alpha), implied_variance=float(1 / alpha**2 - epsilon),
            rms_residual=float(np.sqrt(fit_squares / intra.size))),
        limitation="Fitted moments describe the saved output only; they do not directly observe a kernel accumulator.")


def diagnose(case, result, source_graph, output):
    case, result, source_graph, output = map(Path, (case, result, source_graph, output))
    if output.exists():
        raise ValueError("diagnosis output must be new")
    metadata = json.loads((case / "finish-taps.json").read_text())
    if (metadata["source_graph_sha256"] != SOURCE_GRAPH or sha256(source_graph) != SOURCE_GRAPH or
            metadata["source_revision"] != SOURCE_REVISION or metadata["source_sha256"] != SOURCE_FILES):
        raise ValueError("finish/source provenance differs")
    sys.path.insert(0, str(ROOT / "experiments/windows"))
    from verify_probe import verify
    native = verify(case, result, "cpu")
    if native["model_sha256"] != metadata["graph_sha256"]:
        raise ValueError("native capture belongs to a different tap graph")
    graph = onnx.load(source_graph, load_external_data=False)
    nodes = {node.name: node for node in graph.graph.node}
    norm = nodes["/global_norm/InstanceNormalization"]
    epsilon = next(helper.get_attribute_value(a) for a in norm.attribute if a.name == "epsilon")
    grouped_shape = numpy_helper.to_array(nodes["/global_norm/Constant"].attribute[0].t)
    if norm.op_type != "InstanceNormalization" or not np.array_equal(grouped_shape, [0, 1, -1]) or not np.isclose(epsilon, 1e-8, rtol=1e-7, atol=0):
        raise ValueError("frozen global GroupNorm semantics differ")
    tensors = {tensor.name: tensor for tensor in graph.graph.initializer}
    parameters = []
    for name in ["onnx::Mul_142", "onnx::Add_143"]:
        tensor = tensors[name]
        if tensor.data_type != TensorProto.FLOAT or list(tensor.dims) != [512, 1] or tensor.external_data:
            raise ValueError("expected the frozen inline affine parameters")
        parameters.append(numpy_helper.to_array(tensor).reshape(-1))
    names = [row["output"] for row in metadata["taps"]]
    expected_outputs = {(0, name) for name in ["source0", "source1", *names]}
    if (len(native["outputs"]) != len(expected_outputs) or
            {(row["run"], row["name"]) for row in native["outputs"]} != expected_outputs):
        raise ValueError("missing or duplicate native boundary captures")
    values = {"torch": {}, "native_cpu": {}}
    for name in ["intra_norm", "global_norm"]:
        expected = metadata["expected"][name]
        values["torch"][name] = mapped_float(case / expected["file"], expected["shape"], expected["sha256"])[0]
        tap = next(row for row in metadata["taps"] if row["name"] == name)
        row = next(row for row in native["outputs"] if row["name"] == tap["output"])
        if row["dtype"] != "f32" or row["shape"] != expected["shape"]:
            raise ValueError("native boundary type or shape differs")
        values["native_cpu"][name] = mapped_float(result / row["file"], row["shape"], row["sha256"])[0]
    report = dict(diagnostic_only=True, source_graph_sha256=SOURCE_GRAPH,
        captured_case_sha256=sha256(case / "case.json"), native_result_sha256=sha256(result / "result.json"),
        runtime_sha256=native["runtime_sha256"], source_norm=SOURCE_NORM, ort_cpu_kernel=ORT_KERNEL,
        affine_sha256=[hashlib.sha256(value.astype("<f4").tobytes()).hexdigest() for value in parameters],
        own_input_controls={name: own_input_oracle(value["intra_norm"], value["global_norm"], *parameters, epsilon)
                            for name, value in values.items()},
        interpretation="If native output departs from its own F64 oracle while Torch agrees, and a scalar fitted variance explains the native output, the reduction is isolated. No graph or reference has changed.")
    json_write(output, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["case", "result", "source_graph", "output"]:
        parser.add_argument(name, type=Path)
    diagnose(**vars(parser.parse_args()))
