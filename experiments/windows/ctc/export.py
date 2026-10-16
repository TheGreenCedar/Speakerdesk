"""Build-only ONNX conversion preserving Speakerdesk's existing dequantized CTC math."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def match_attention_scaling(model):
    nodes = {node.name: node for node in model.graph.node}
    scale_name = "speakerdesk_attention_query_scale"
    assert not any(value.name == scale_name for value in model.graph.initializer)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(.125, np.float32), scale_name))
    for index in range(24):
        prefix = f"/model/encoder/layers.{index}/self_attn/sdpa/"
        query, key, product = [nodes[prefix + name] for name in ["Mul", "Mul_1", "MatMul"]]
        assert list(query.input) == [prefix + "Transpose_output_0", prefix + "Sqrt_1_output_0"]
        assert list(key.input) == [prefix + "Transpose_2_output_0", prefix + "Sqrt_2_output_0"]
        assert list(product.input) == [query.output[0], key.output[0]]
        # Match the shipped model's exactly representable query scale, avoiding two rounded sqrt factors.
        query.input[1] = scale_name
        key.op_type = "Identity"
        del key.input[1:]
    return model


def dequantize_graph(model, manifest):
    producers = {value: node for node in model.graph.node for value in node.output}
    consumers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    weights = {value.name: value for value in model.graph.initializer}
    parameters = {parameter["sources"][0]: parameter for parameter in manifest["parameters"]
                  if len(parameter["sources"]) == 3}
    remove, replace, additions = set(), {}, []
    for node in model.graph.node:
        if node.op_type != "MatMulInteger":
            continue
        quantized = producers[node.input[0]]
        assert quantized.op_type == "DynamicQuantizeLinear"
        assert node.input[2] == quantized.output[2]
        cast, = consumers[node.output[0]]
        assert cast.op_type == "Cast"
        multiply, = consumers[cast.output[0]]
        assert multiply.op_type == "Mul"
        scale_product = producers[next(value for value in multiply.input if value != cast.output[0])]
        assert scale_product.op_type == "Mul"
        assert quantized.output[1] in scale_product.input
        scale_name, = [value for value in scale_product.input if value != quantized.output[1]]
        weight = numpy_helper.to_array(weights[node.input[1]])
        zero = numpy_helper.to_array(weights[node.input[3]])
        scale = numpy_helper.to_array(weights[scale_name])
        assert weight.dtype == np.uint8 and zero.ndim == scale.ndim == 0
        decoded = (weight.astype(np.float32) - zero.astype(np.float32)) * scale
        parameter = parameters[node.input[1]]
        assert set(parameter["sources"]) == {node.input[1], node.input[3], scale_name}
        # The installed MLX artifact transposes these same matrices into Linear weights.
        assert hashlib.sha256(decoded.T.copy().tobytes()).hexdigest() == parameter["converted_tensor_sha256"]
        name = node.input[1] + "_dequantized"
        additions.append(numpy_helper.from_array(decoded, name))
        replace[node.name] = helper.make_node("MatMul", [quantized.input[0], name],
                                              list(multiply.output), name=node.name + "_float")
        remove.update(item.name for item in (quantized, cast, multiply, scale_product))
    assert len(replace) == 146
    nodes = [replace.get(node.name, node) for node in model.graph.node if node.name not in remove]
    # Discard only unreachable graph material; the retained graph keeps its execution order.
    needed = {output.name for output in model.graph.output}
    kept = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [value for value in [*model.graph.initializer, *additions] if value.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    # Gather after normalizing the entire vocabulary, as the native timing contract requires.
    logits, = model.graph.output
    model.graph.node.extend([
        helper.make_node("LogSoftmax", [logits.name], ["log_probabilities"], axis=-1),
        helper.make_node("Gather", ["log_probabilities", "token_ids"], ["scores"], axis=-1),
    ])
    model.graph.input.append(helper.make_tensor_value_info("token_ids", onnx.TensorProto.INT64, ["columns"]))
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT,
                                                           [1, "num_frames", "columns"]))
    return model


def main(args):
    manifest = json.loads((ROOT / "speakerdesk/alignment-mlx/manifest.json").read_text())
    assert digest(args.source) == manifest["source_model_sha256"]
    assert args.source.stat().st_size == manifest["source_model_bytes"]
    args.output.mkdir(parents=True, exist_ok=False)
    model = dequantize_graph(onnx.load(args.source), manifest)
    destination = args.output / "ctc.onnx"
    onnx.save_model(model, destination, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="ctc.weights", size_threshold=1024)
    onnx.checker.check_model(str(destination))
    del model
    from qualify import prepare_cases
    qualification = prepare_cases(args.output)

    provenance = dict(source_sha256=digest(args.source), exporter_sha256=digest(__file__),
                      qualification=qualification,
                      outputs={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                               for p in args.output.iterdir() if p.is_file()})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return qualification["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    raise SystemExit(0 if main(parser.parse_args()) else 1)
