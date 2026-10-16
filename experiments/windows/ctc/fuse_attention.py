"""One diagnostic candidate: preserve the source query scale and use maintained MHA."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper

from export import digest

ORIGINAL_GRAPH_SHA256 = "4e9c26adc08f23712374a98b7a8056fffac4f1646fa6b51ca5ee7b1c0dbb81f1"
ORIGINAL_WEIGHTS_SHA256 = "ebb4aa54f64240fe95c079c0b3ec326498f647ebe8d9bef50d316d47a59a53e5"
SCALE = "speakerdesk_mha_source_query_scale"
CASE_NAMES = ("whole", "400", "719", "720", "partial_tail", "silence")


def fuse_attention(model):
    """Validate all 24 original attention paths before changing the graph in place."""
    # ONNX names are optional; the unchanged exported LogSoftmax/Gather are unnamed.
    named_nodes = [n for n in model.graph.node if n.name]
    nodes = {n.name: n for n in named_nodes}
    initializers = {v.name: v for v in model.graph.initializer}
    if len(nodes) != len(named_nodes) or SCALE in initializers:
        raise ValueError("duplicate nodes or existing MHA candidate material")
    if any(n.op_type == "MultiHeadAttention" or n.name.startswith("speakerdesk_mha_")
           for n in model.graph.node):
        raise ValueError("graph already contains an attention candidate")
    if any(o.domain == "com.microsoft" and o.version != 1 for o in model.opset_import):
        raise ValueError("unexpected Microsoft operator version")
    replacements = {}
    for index in range(24):
        prefix = f"/model/encoder/layers.{index}/self_attn/"

        def require(name, op, inputs=None):
            n = nodes.get(prefix + name)
            if (n is None or n.op_type != op or n.domain
                    or (inputs is not None and list(n.input) != inputs)
                    or len(n.output) != 1):
                raise ValueError(f"unexpected CTC attention topology: layer {index}, {name}")
            return n

        projected = []
        for projection in ("q", "k", "v"):
            bias = f"model.encoder.layers.{index}.self_attn.{projection}_proj.bias"
            parameter = initializers.get(bias)
            if parameter is None or parameter.data_type != onnx.TensorProto.FLOAT or list(parameter.dims) != [1024]:
                raise ValueError(f"missing full Float32 projection bias: {bias}")
            n = require(f"{projection}_proj/Add", "Add", [
                bias, prefix + f"{projection}_proj/MatMul_output_0"])
            projected.append(n.output[0])

        reshaped = [require(name, "Reshape") for name in ("Reshape_2", "Reshape_5", "Reshape_8")]
        if any(len(n.input) != 2 or n.input[0] != value for n, value in zip(reshaped, projected)):
            raise ValueError(f"attention projection bypasses its bias: layer {index}")
        transposed = []
        for name, reshape, permutation in zip(
            ("sdpa/Transpose", "sdpa/Transpose_2", "sdpa/Transpose_1"),
            reshaped, ([0, 2, 1, 3], [0, 2, 3, 1], [0, 2, 1, 3]),
        ):
            n = require(name, "Transpose", [reshape.output[0]])
            if {a.name: helper.get_attribute_value(a) for a in n.attribute} != {"perm": permutation}:
                raise ValueError(f"unexpected attention head order: layer {index}")
            transposed.append(n.output[0])
        query = require("sdpa/Mul", "Mul", [transposed[0], prefix + "sdpa/Sqrt_1_output_0"])
        key = require("sdpa/Mul_1", "Mul", [transposed[1], prefix + "sdpa/Sqrt_2_output_0"])
        product = require("sdpa/MatMul", "MatMul", [query.output[0], key.output[0]])
        softmax = require("sdpa/Softmax", "Softmax", [product.output[0]])
        if {a.name: helper.get_attribute_value(a) for a in softmax.attribute} != {"axis": -1}:
            raise ValueError(f"unexpected attention softmax: layer {index}")
        context = require("sdpa/MatMul_1", "MatMul", [softmax.output[0], transposed[2]])
        joined = require("sdpa/Transpose_3", "Transpose", [context.output[0]])
        if {a.name: helper.get_attribute_value(a) for a in joined.attribute} != {"perm": [0, 2, 1, 3]}:
            raise ValueError(f"unexpected attention output order: layer {index}")
        output = require("Reshape_9", "Reshape")
        if output.input[0] != joined.output[0]:
            raise ValueError(f"unexpected attention output: layer {index}")
        scaled = f"speakerdesk_mha_{index}_query"
        replacements[output.name] = [
            # Scale after the full projection bias, as in the pinned maintained source.
            helper.make_node("Mul", [projected[0], SCALE], [scaled], name=scaled),
            helper.make_node("MultiHeadAttention", [scaled, projected[1], projected[2]],
                             list(output.output), name=f"speakerdesk_mha_{index}",
                             domain="com.microsoft", num_heads=16, scale=1.0),
        ]

    rewritten = [item for n in model.graph.node for item in replacements.get(n.name, [n])]
    needed = {v.name for v in model.graph.output}
    kept = []
    for n in reversed(rewritten):
        if needed.intersection(n.output):
            kept.append(n)
            needed.update(n.input)
    kept.reverse()
    if sum(n.domain == "com.microsoft" and n.op_type == "MultiHeadAttention" for n in kept) != 24:
        raise ValueError("candidate did not retain all 24 attention blocks")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    # Existing parameters and their external-data references stay byte-for-byte intact.
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(.125, np.float32), SCALE))
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return model


def prepare_cases(source, graph, destination):
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for name in CASE_NAMES:
        original = source.parent / name
        specification = json.loads((original / "case.json").read_text())
        if (original / specification["model"]).resolve() != source.resolve():
            raise ValueError("qualification case does not belong to the pinned original graph")
        case = destination / name
        case.mkdir()
        for filename in ["expected.f32", *[v["path"] for v in specification["inputs"]]]:
            if Path(filename).name != filename:
                raise ValueError("unexpected CTC qualification tensor path")
            shutil.copyfile(original / filename, case / filename)
        specification["model"] = Path(os.path.relpath(graph, case)).as_posix()
        specification["model_files"] = {
            Path(os.path.relpath(source.parent / "ctc.weights", case)).as_posix(): ORIGINAL_WEIGHTS_SHA256,
            specification["model"]: digest(graph),
        }
        (case / "case.json").write_text(json.dumps(specification, indent=2) + "\n")
        records.append(dict(case=name, original_case_sha256=digest(original / "case.json"),
                            unchanged_expected_sha256=digest(case / "expected.f32")))
    return records


def main(args):
    source, output = args.source.resolve(strict=True), args.output.absolute()
    if output.exists() or output.parent.resolve() != source.parent:
        raise ValueError("write a fresh sibling graph so existing external weights remain unchanged")
    if digest(source) != ORIGINAL_GRAPH_SHA256 or digest(source.parent / "ctc.weights") != ORIGINAL_WEIGHTS_SHA256:
        raise ValueError("candidate requires the exact original Float32 graph and weights")
    model = onnx.load(source, load_external_data=False)
    for parameter in model.graph.initializer:
        if parameter.data_location == onnx.TensorProto.EXTERNAL:
            if {v.key: v.value for v in parameter.external_data}.get("location") != "ctc.weights":
                raise ValueError("unexpected CTC external-weight path")
    model = fuse_attention(model)
    onnx.save_model(model, output)
    onnx.checker.check_model(str(output))
    cases = prepare_cases(source, output, args.cases) if args.cases else []
    provenance = dict(candidate="source-scaled-maintained-mha-v1", status="unqualified",
                      original_graph_sha256=ORIGINAL_GRAPH_SHA256,
                      unchanged_weights_sha256=ORIGINAL_WEIGHTS_SHA256,
                      graph_sha256=digest(output), transform_sha256=digest(__file__), cases=cases,
                      changes="24 attention decompositions replaced; full q projection bias then *.125; MHA scale1; no other arithmetic changes",
                      limit="AMD operator runs; full independent score gate is still 1e-4 and decides this candidate")
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", type=Path, help="copy existing input/reference tensors into fresh candidate cases")
    main(parser.parse_args())
