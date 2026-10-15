"""Build-only dynamic, staged ONNX export of the pinned raw MossFormer2 graph."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
SOURCE_FILES = {
    "conv_module.py": "b413aa0cb0df34742460a1f1af2e2cd1d2005900af8ac17e8de8675b8f80de86",
    "fsmn.py": "62c376b2708ac1c59853c13cbe9acb7ea011082c39f1a42e1af0f8213a00bb76",
    "layer_norm.py": "d28fea41b3c5a2dd8bb7a1a8c1c97492fcc7eb88cc7b46283fc21e4dbf19f062",
    "mossformer2.py": "66f14af8efc0758d8c79bd7d1054851558081c5f6f1150cdf133ae4b6ca9c442",
    "mossformer2_block.py": "2760c18c1eb68f8ba2f1160e9a46dc6adfbcb397a693dad27ae6e43e8a02faf0",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_source(parent):
    for name, digest in SOURCE_FILES.items():
        if sha256(parent / "upstream" / name) != digest:
            raise ValueError(f"unpinned ClearerVoice source: {name}")
    initializer = parent / "upstream/__init__.py"
    if initializer.exists() and initializer.read_bytes().strip():
        raise ValueError("upstream package initializer must be empty")
    sys.path.insert(0, str(parent))
    source = importlib.import_module("upstream.mossformer2")
    block = importlib.import_module("upstream.mossformer2_block")
    if Path(source.__file__).resolve() != (parent / "upstream/mossformer2.py").resolve():
        raise ValueError("a different ClearerVoice package was imported")
    return source, block


def verify_checkpoint(path):
    pin = json.loads((ROOT / "crates/host/src/separation-artifact.json").read_text())
    expected = pin["files"]["last_best_checkpoint.pt"]
    if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
        raise ValueError("unpinned MossFormer2 checkpoint")
    return pin


def externalize(model, stream, offsets):
    from onnx import external_data_helper
    for tensor in model.graph.initializer:
        if len(tensor.raw_data) < 4096:
            continue
        digest = hashlib.sha256(tensor.raw_data).hexdigest()
        if digest not in offsets:
            offsets[digest] = stream.tell(), len(tensor.raw_data)
            stream.write(tensor.raw_data)
        offset, length = offsets[digest]
        external_data_helper.set_external_data(tensor, "mossformer.weights", offset=offset, length=length)
        tensor.ClearField("raw_data")


def export(checkpoint, source_parent, output):
    pin = verify_checkpoint(checkpoint)
    import onnx
    import torch
    from attention import lower_attention
    from normalization import apply_stage
    from stages import adapt_for_export, program
    source, block = load_source(source_parent)
    model = source.MossFormer().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    parameters = json.loads((ROOT / "crates/models/src/mossformer2/tensor_manifest.json").read_text())
    if len(state) != len(parameters):
        raise ValueError("unexpected MossFormer2 tensor inventory")
    for parameter in parameters:
        tensor = state[parameter["name"]]
        if tensor.dtype != torch.float32 or list(tensor.shape) != parameter["shape"]:
            raise ValueError(("invalid source tensor", parameter["name"]))
        raw = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        if hashlib.sha256(raw).hexdigest() != parameter["sha256"]:
            raise ValueError(("source tensor digest differs", parameter["name"]))
    model.load_state_dict(state, strict=True)
    del state
    adapt_for_export(model, block)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    audio = torch.zeros((1, 2071), dtype=torch.float32)
    encoded = torch.zeros((1, 512, 257), dtype=torch.float32)
    latent = encoded.transpose(1, 2).contiguous()
    offsets, graphs = {}, []
    temporary = output / ".torch-export.onnx"
    with (output / "mossformer.weights").open("wb") as weights, torch.inference_mode():
        for name, module in program(model):
            started = time.perf_counter()
            if name == "encoder":
                inputs, input_names, output_names = (audio,), ["audio"], ["encoded", "residual", "x"]
                axes = {"audio": {1: "samples"}, "encoded": {2: "frames"},
                        "residual": {2: "frames"}, "x": {1: "frames"}}
            elif name == "finish":
                inputs, input_names, output_names = (audio, encoded, encoded, latent), ["audio", "encoded", "residual", "x"], ["source0", "source1"]
                axes = {"audio": {1: "samples"}, "encoded": {2: "frames"}, "residual": {2: "frames"},
                        "x": {1: "frames"}, "source0": {1: "samples"}, "source1": {1: "samples"}}
            else:
                inputs, input_names, output_names = (latent,), ["x"], ["x_out"]
                axes = {"x": {1: "frames"}, "x_out": {1: "frames"}}
            torch.onnx.export(module.eval(), inputs, str(temporary), input_names=input_names,
                              output_names=output_names, dynamic_axes=axes, opset_version=17,
                              dynamo=False, do_constant_folding=True, external_data=False)
            graph = onnx.load(str(temporary))
            domains = {node.domain for node in graph.graph.node}
            if domains - {"", "ai.onnx"}:
                raise ValueError(("nonstandard model operators", name, domains))
            _, normalization_lowerings = apply_stage(graph, name)
            attention_lowerings = lower_attention(graph) if name.startswith("flash-") else []
            externalize(graph, weights, offsets)
            weights.flush()
            path = output / f"{name}.onnx"
            onnx.save(graph, str(path))
            onnx.checker.check_model(str(path), full_check=True)
            temporary.unlink()
            row = {"name": name, "model": path.name, "sha256": sha256(path),
                   "inputs": input_names, "outputs": output_names,
                   "operators": sorted({node.op_type for node in graph.graph.node}),
                   "normalization_lowerings": normalization_lowerings,
                   "attention_lowerings": attention_lowerings,
                   "export_seconds": time.perf_counter() - started}
            graphs.append(row)
            print(json.dumps(row), flush=True)
    manifest = {"checkpoint": pin, "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_FILES,
                "sample_bounds": [16, 392000], "latent_channels": 512, "dtype": "float32",
                "graphs": graphs, "model_files": {"mossformer.weights": sha256(output / "mossformer.weights")},
                "exporter_sha256": sha256(Path(__file__)), "stages_sha256": sha256(Path(__file__).with_name("stages.py")),
                "normalization_sha256": sha256(Path(__file__).with_name("normalization.py")),
                "attention_sha256": sha256(Path(__file__).with_name("attention.py")),
                "tool_versions": {name: importlib.metadata.version(name) for name in [
                    "torch", "onnx", "numpy", "einops", "rotary-embedding-torch", "torchinfo"]},
                "source_adaptations": ["padding=(-frames)%256", "disable rotary memoization only",
                    "bounded Float32 centered-variance sums for encoder/finish GroupNorm(1,512), original epsilon/affine",
                    "FLASH attention uses bounded MatMul contractions and broadcast Mul outer products; no ellipsis CPU fallback"],
                "semantics": "raw whole-window float32; encoder, 24 FLASH/FSMN pairs, finish; no RMS or sliding windows"}
    json_write(output / "program.json", manifest)
    print(json.dumps({"graphs": len(graphs), "external_weights_bytes": (output / "mossformer.weights").stat().st_size,
                      "program": str(output / "program.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source_parent", type=Path, help="directory containing pinned upstream/ source package")
    parser.add_argument("output", type=Path, help="fresh private artifact directory")
    args = parser.parse_args()
    export(args.checkpoint, args.source_parent, args.output)
