"""Add diagnostic outputs to the exact ReDimNet graph without loading external weights.

Place output_graph beside the original redimnet.weights. The native probe verifies
the recorded weight digest before execution; this tool also checks it if present.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import sha256


TAPS = {
    "tap_frontend": "/graph/spec/Cast_1",
    "tap_stem_conv": "/graph/backbone/graph/stem/stem.0/Conv",
    "tap_stem_norm": "/graph/backbone/graph/stem/stem.1/Add_1",
    "tap_stem_group_norm": "/graph/backbone/graph/stem_gnorm/Add",
    "tap_backbone": "/graph/backbone/Cast_1",
    "tap_pre_l2": "/graph/linear/Gemm",
}

# Each stage returns its upsampled, group-normalized feature map to the shared arrival list.
STAGE_TAPS = {
    f"tap_stage{index}": f"/graph/backbone/graph/stage{index}/stage{index}.{last}/Add"
    for index, last in enumerate([10, 11, 12, 11, 11, 10])
}
STAGE_TAPS["tap_stage_mix"] = "/graph/backbone/graph/fin_wght1d/ReduceSum"

STAGE0_NORM_TAPS = {
    "tap_stage0_pre_norm": "/graph/backbone/graph/stage0/stage0.9/Resize",
    "tap_stage0_unit_norm": "/graph/backbone/graph/stage0/stage0.10/InstanceNormalization",
    "tap_stage0": "/graph/backbone/graph/stage0/stage0.10/Add",
}


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def add_taps(model, backbone_stages=False, stage0_norm=False):
    if backbone_stages and stage0_norm:
        raise ValueError("choose one diagnostic tap set")
    if [output.name for output in model.graph.output] != ["embedding"]:
        raise ValueError("expected the original embedding-only graph")
    inferred = onnx.shape_inference.infer_shapes(model)
    values = {value.name: value for value in inferred.graph.value_info}
    nodes = {node.name: node for node in model.graph.node}
    names = {name for node in model.graph.node for name in node.output}
    taps = []
    selected = STAGE0_NORM_TAPS if stage0_norm else STAGE_TAPS if backbone_stages else TAPS
    for name, node_name in selected.items():
        node = nodes[node_name]
        if len(node.output) != 1 or name in names:
            raise ValueError(("unexpected tap identity", name))
        source = node.output[0]
        info = values[source]
        tensor = info.type.tensor_type
        if tensor.elem_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            raise ValueError(("unexpected tap type", source, tensor.elem_type))
        shape = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                 for dim in tensor.shape.dim]
        # The shared native probe reads FP32/Int64 outputs, including these FP16 intermediates.
        model.graph.node.append(helper.make_node("Cast", [source], [name],
                                                  name=name, to=TensorProto.FLOAT))
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        taps.append(dict(name=name, source=source, source_dtype=tensor.elem_type, shape=shape))
    return taps


def prepare(graph, source_case, output_graph, output_case, input_root=None, weights_root=None,
            backbone_stages=False, stage0_norm=False):
    graph, source_case = Path(graph).resolve(), Path(source_case).resolve()
    output_graph, output_case = Path(output_graph).resolve(), Path(output_case).resolve()
    input_root = Path(input_root).resolve() if input_root else source_case.parent
    weights_root = Path(weights_root).resolve() if weights_root else output_graph.parent
    if output_graph.exists() or output_case.exists():
        raise ValueError("diagnostic graph and case paths must be new")
    if output_graph.parent != weights_root:
        raise ValueError("output graph must be beside the unchanged external weights")
    specification = json.loads(source_case.read_text())
    if len(specification["inputs"]) != 1:
        raise ValueError("expected the original single-audio case")
    audio = specification["inputs"][0]
    if (audio["name"], audio["dtype"], audio["shape"]) != ("audio", "f32", [1, 96000]):
        raise ValueError("unexpected waveform contract")
    audio_path = input_root / audio["path"]
    if audio_path.stat().st_size != 96000 * 4:
        raise ValueError("unexpected waveform length")
    model = onnx.load(graph, load_external_data=False)
    locations = {item.value for tensor in model.graph.initializer
                 for item in tensor.external_data if item.key == "location"}
    recorded = specification.get("model_files", {})
    if locations != {"redimnet.weights"} or len(recorded) != 1:
        raise ValueError("expected the pinned single external parameter file")
    weight_name, weight_digest = next(iter(recorded.items()))
    if Path(weight_name).name != "redimnet.weights":
        raise ValueError("case does not identify the original external parameters")
    weights = weights_root / "redimnet.weights"
    checked_weights = weights.exists()
    if checked_weights and sha256(weights) != weight_digest:
        raise ValueError("external parameters differ from the original case")
    tensors_before = [tensor.SerializeToString() for tensor in model.graph.initializer]
    taps = add_taps(model, backbone_stages, stage0_norm)
    if tensors_before != [tensor.SerializeToString() for tensor in model.graph.initializer]:
        raise ValueError("diagnostic changed an initializer")
    case = dict(model=os.path.relpath(output_graph, output_case).replace("\\", "/"), repeat=1,
                model_files={os.path.relpath(weights, output_case).replace("\\", "/"): weight_digest},
                inputs=[dict(audio, path="audio.f32")])
    metadata = dict(diagnostic_only=True, source_graph_sha256=sha256(graph),
                    source_case_sha256=sha256(source_case), input_sha256=sha256(audio_path),
                    independent_reference_sha256=sha256(source_case.parent / "expected.json"),
                    weights_sha256=weight_digest, external_weights_checked_locally=checked_weights,
                    graph_sha256=None, taps=taps,
                    tap_set="stage0_norm" if stage0_norm else "backbone_stages" if backbone_stages else "components",
                    limitation="Added outputs can change provider fusion or placement; compare the retained embedding with its original-provider baseline.")
    output_case.mkdir(parents=True, exist_ok=False)
    with output_graph.open("xb") as stream:
        stream.write(model.SerializeToString())
    shutil.copyfile(audio_path, output_case / "audio.f32")
    metadata["graph_sha256"] = sha256(output_graph)
    json_write(output_case / "case.json", case)
    json_write(output_case / "taps.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("source_case", type=Path, help="original case.json")
    parser.add_argument("output_graph", type=Path)
    parser.add_argument("output_case", type=Path)
    parser.add_argument("--input-root", type=Path, help="lookup root for original case input paths")
    parser.add_argument("--weights-root", type=Path, help="runtime weight directory; also the output graph directory")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--backbone-stages", action="store_true",
                           help="tap the six stage returns and final feature-map mixture instead of component boundaries")
    selection.add_argument("--stage0-norm", action="store_true",
                           help="bracket stage 0 GroupNorm with its input, unit-normalized output and affine stage return")
    args = parser.parse_args()
    prepare(args.graph, args.source_case, args.output_graph, args.output_case,
            args.input_root, args.weights_root, args.backbone_stages, args.stage0_norm)
