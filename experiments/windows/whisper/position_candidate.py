"""Replace only the positional parameter in the frozen compiled-Half-GELU graph."""
import argparse
import json
from pathlib import Path
import shutil

import onnx
from onnx import numpy_helper

from export import sha256
from position_constant import PROVENANCE, load_positions

GELU_SHA256 = "20ee2dd9b6e00a211e54a62a4753c3a1e5b069c265d6b3d23bcb7836c07a0c59"


def prepare(original, positions, output):
    if sha256(original / "whisper.onnx") != GELU_SHA256:
        raise ValueError("expected the frozen compiled-Half-GELU graph")
    table = load_positions(positions)
    model = onnx.load(original / "whisper.onnx")
    candidates = [value for value in model.graph.initializer if list(value.dims) == PROVENANCE["shape"]]
    if len(candidates) != 1 or candidates[0].data_type != onnx.TensorProto.FLOAT16:
        raise ValueError("unexpected Whisper positional parameter inventory")
    target = candidates[0]
    target.CopyFrom(numpy_helper.from_array(table, target.name))
    onnx.checker.check_model(model, full_check=True)
    output.mkdir(parents=True, exist_ok=False)
    onnx.save(model, output / "whisper.onnx")
    shutil.copyfile(original / "languages.json", output / "languages.json")
    cases = []
    for case in sorted((original / "cases").glob("neural-*")):
        destination = output / "cases" / case.name
        destination.mkdir(parents=True)
        for name in ["input.f32", "expected.f32", "expected.json"]:
            shutil.copyfile(case / name, destination / name)
        metadata = json.loads((case / "case.json").read_text())
        if not metadata.get("disable_optimizations"):
            raise ValueError("compiled-Half-GELU case lost disabled graph optimizations")
        metadata["model_files"] = {"../../whisper.onnx": sha256(output / "whisper.onnx")}
        (destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases.append(str(destination.relative_to(output) / "case.json"))
    provenance = {"source_graph_sha256": GELU_SHA256,
                  "model_sha256": sha256(output / "whisper.onnx"),
                  "script_sha256": sha256(Path(__file__)),
                  "parameter_manifest_sha256": sha256(Path(__file__).with_name("position-constant.json")),
                  "position_input_sha256": sha256(positions), "position_constant": PROVENANCE,
                  "changed_initializer": target.name,
                  "rule": "One audio-independent positional parameter replaced; graph nodes, other constants and independent expectations unchanged",
                  "disable_optimizations": True, "cases": cases, "qualified": False}
    (output / "positions.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["original", "positions", "output"]:
        parser.add_argument(name, type=Path)
    prepare(**vars(parser.parse_args()))
