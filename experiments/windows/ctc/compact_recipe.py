"""Freeze setup-only reconstruction offsets from the unchanged qualified CTC graph."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRAPH_SHA256 = "233ac8980dab9c142ef9fe91ab1d3b5d6309a9549f59a284e50199f7b1081699"
WEIGHTS_SHA256 = "ebb4aa54f64240fe95c079c0b3ec326498f647ebe8d9bef50d316d47a59a53e5"
WEIGHTS_BYTES = 1301979472


def recipe(initializers, manifest, manifest_sha256):
    """Consume metadata only; every external byte must map once to a source parameter."""
    parameters = {}
    for parameter in manifest["parameters"]:
        sources = parameter["sources"]
        name = sources[0] + ("_dequantized" if len(sources) == 3 else "")
        if name in parameters:
            raise ValueError("duplicate parameter source")
        parameters[name] = parameter
    tensors, seen = [], set()
    for value in initializers:
        name, dtype, shape, external = value
        if not external:
            continue
        if name in seen or name not in parameters or dtype != 1:
            raise ValueError("unknown, duplicate or non-FLOAT external initializer")
        seen.add(name)
        parameter = parameters[name]
        transform = parameter["transform"]
        expected_shape = parameter["shape"]
        if transform == "UINT8_dequantize_then_I-O_to_O-I":
            expected_shape = expected_shape[::-1]
        elif transform == "OI-kernel_to_O-kernel-I":
            o, k, i = expected_shape
            expected_shape = [o, i, k]
        elif transform != "identity":
            raise ValueError("unknown transform")
        count = 1
        for dim in expected_shape:
            count *= dim
        if (set(external) != {"location", "offset", "length"}
                or external["location"] != "ctc.weights" or shape != expected_shape
                or int(external["length"]) != count * 4):
            raise ValueError("external layout differs from source parameter")
        tensors.append(dict(initializer=name, parameter=parameter["target"],
                            shape=shape, offset=int(external["offset"]), bytes=count * 4))
    tensors.sort(key=lambda t: t["offset"])
    offset = 0
    for tensor in tensors:
        if tensor["offset"] != offset:
            raise ValueError("external store has an overlap or gap")
        offset += tensor["bytes"]
    if seen != parameters.keys() or offset != WEIGHTS_BYTES or len(tensors) != 422:
        raise ValueError("incomplete qualified weight coverage")
    return dict(source_sha256=manifest["source_model_sha256"],
                source_bytes=manifest["source_model_bytes"], graph_sha256=GRAPH_SHA256,
                manifest_sha256=manifest_sha256, weights_sha256=WEIGHTS_SHA256,
                weights_bytes=WEIGHTS_BYTES, tensors=tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.graph.read_bytes()
    if hashlib.sha256(data).hexdigest() != GRAPH_SHA256:
        raise ValueError("not the qualified CTC graph")
    import onnx
    graph = onnx.load_model_from_string(data).graph  # Never opens external data.
    values = [(v.name, v.data_type, list(v.dims), {x.key: x.value for x in v.external_data})
              for v in graph.initializer]
    manifest_data = (ROOT / "speakerdesk/alignment-mlx/manifest.json").read_bytes()
    result = recipe(values, json.loads(manifest_data), hashlib.sha256(manifest_data).hexdigest())
    with args.output.open("x", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
