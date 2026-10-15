"""Build-only export of Speakerdesk's pinned MLX Cohere checkpoint. No app Python."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import wave

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, numpy_helper
from safetensors import safe_open
from tokenizers import Tokenizer
import torch
from transformers import CohereAsrConfig, CohereAsrFeatureExtractor, CohereAsrForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

ROOT = Path(__file__).resolve().parents[3]
LANGUAGES = ("ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh")
CACHE_NAMES = [f"{kind}_{layer}" for layer in range(8)
               for kind in ("self_k", "self_v", "cross_k", "cross_v")]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def dequantize(packed, scales, biases):
    """MLX affine: low nibble first, 8 values/U32, one scale and offset/64."""
    assert packed.dtype == torch.uint32 and packed.ndim == 2
    rows, words = packed.shape
    assert scales.shape == biases.shape == (rows, words // 8)
    values = ((packed.to(torch.int64).unsqueeze(-1) >> torch.arange(0, 32, 4)) & 15)
    values = values.reshape(rows, words // 8, 64).float()
    return (values * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).reshape(rows, words * 8)


def source_name(target):
    name = target.removeprefix("model.")
    for index, source in [(0, "conv0"), (2, "dw"), (3, "pw_convs.0"),
                          (5, "dw"), (6, "pw_convs.1")]:
        prefix = f"encoder.subsampling.layers.{index}."
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            if source == "dw":
                return f"encoder.subsampling.dw_{suffix}_{0 if index == 2 else 1}"
            return f"encoder.subsampling.{source}.{suffix}"
    return name.replace(".conv.depthwise_conv.weight", ".conv.dw_weight").replace(
        ".conv.depthwise_conv.bias", ".conv.dw_bias")


def make_config(source):
    config = CohereAsrConfig(encoder_config=source["encoder"], **{
        key: value for key, value in source["decoder"].items() if key != "encoder_hidden_size"
    })
    config._attn_implementation = "eager"
    config.encoder_config._attn_implementation = "eager"
    return config


def load_model(directory):
    source = json.loads((directory / "config.json").read_text())
    assert source["quantization"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    config = make_config(source)
    with torch.device("meta"):
        model = CohereAsrForConditionalGeneration(config)
    converted, consumed = {}, set()
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        for target, expected in model.state_dict().items():
            if target.endswith("num_batches_tracked"):
                value = torch.tensor(0, dtype=torch.int64)
            else:
                name = source_name(target)
                value = weights.get_tensor(name)
                consumed.add(name)
                if value.dtype == torch.uint32:
                    prefix = name.removesuffix(".weight")
                    scale, bias = prefix + ".scales", prefix + ".biases"
                    value = dequantize(value, weights.get_tensor(scale), weights.get_tensor(bias))
                    consumed.update((scale, bias))
                else:
                    value = value.float()
                    if target.endswith("weight") and value.ndim > 2:
                        value = value.permute(0, value.ndim - 1, *range(1, value.ndim - 1)).contiguous()
                assert value.shape == expected.shape, (target, name, value.shape, expected.shape)
            converted[target] = value
        assert consumed == keys, (sorted(keys - consumed), sorted(consumed - keys))
    model.load_state_dict(converted, strict=True, assign=True)
    # The only nonpersistent buffer was constructed on meta along with the graph.
    positions = model.model.encoder.encode_positions
    positions.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, config.encoder_config.hidden_size, 2).float()
                                         / config.encoder_config.hidden_size))
    model.eval()
    return model


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.eval()

    def forward(self, features, attention_mask):
        result = self.encoder(features, attention_mask=attention_mask)
        return result.last_hidden_state, result.attention_mask.to(torch.int64)


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.eval()

    def forward(self, tokens, positions, encoded, self_mask, cross_mask, *past):
        cache = EncoderDecoderCache(DynamicCache(config=self.decoder.config),
                                    DynamicCache(config=self.decoder.config))
        for layer in range(len(past) // 4):
            cache.self_attention_cache.update(past[4 * layer], past[4 * layer + 1], layer)
            cache.cross_attention_cache.update(past[4 * layer + 2], past[4 * layer + 3], layer)
            cache.is_updated[layer] = True
        result = self.decoder(input_ids=tokens, position_ids=positions, encoder_hidden_states=encoded,
                              attention_mask=self_mask, encoder_attention_mask=cross_mask,
                              past_key_values=cache, use_cache=True)
        logits = self.proj_out(result.last_hidden_state[:, -1, :])
        states = []
        for own, cross in zip(cache.self_attention_cache.layers, cache.cross_attention_cache.layers):
            states.extend((own.keys, own.values, cross.keys, cross.values))
        return (logits, *states)


def decoder_inputs(tokens, offset, encoded, encoder_mask, past=()):
    ids = torch.tensor([tokens], dtype=torch.int64)
    positions = torch.arange(offset, offset + len(tokens), dtype=torch.int64).unsqueeze(0)
    allowed = torch.arange(offset + len(tokens))[None, :] <= positions.reshape(-1, 1)
    self_mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None, :, :]
    cross_mask = torch.where(encoder_mask[:, None, None, :].bool(), 0.0,
                             torch.finfo(torch.float32).min).expand(1, 1, len(tokens), -1)
    return (ids, positions, encoded, self_mask, cross_mask, *past)


def prompt(tokenizer, language):
    assert language in LANGUAGES
    values = ["▁", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
              f"<|{language}|>", f"<|{language}|>", "<|pnc|>", "<|noitn|>",
              "<|notimestamp|>", "<|nodiarize|>"]
    ids = [tokenizer.token_to_id(value) for value in values]
    assert None not in ids, (language, values, ids)
    return ids


def transcribe(encoder, decoder, features, mask, tokens):
    encoded, encoder_mask = encoder(features, mask)
    emitted, past, offset = [], (), 0
    for _ in range(448):
        inputs = decoder_inputs(tokens, offset, encoded, encoder_mask, past)
        result = decoder(*inputs)
        token = result[0].argmax(-1).item()
        if token == 3:
            break
        emitted.append(token)
        offset += len(tokens)
        tokens, past = [token], result[1:]
    else:
        raise AssertionError("Cohere reached the 448-token bound")
    return emitted, encoded, encoder_mask


def export_graph(module, args, path, input_names, output_names, dynamic_axes, weight_store):
    """Trace maintained PyTorch layers; bind weights externally to avoid protobuf's 2GB limit."""
    torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False, export_params=False,
                      do_constant_folding=False, training=torch.onnx.TrainingMode.EVAL)
    graph = onnx.load(path, load_external_data=False)
    state = module.state_dict()
    weight_path, offsets = weight_store
    with weight_path.open("ab") as stream:
        for value in list(graph.graph.input):
            if value.name not in state:
                continue
            array = state[value.name].detach().cpu().numpy()
            if value.name not in offsets:
                start = stream.tell()
                stream.write(array.tobytes())
                offsets[value.name] = (start, array.nbytes)
            tensor = numpy_helper.from_array(array, name=value.name)
            start, length = offsets[value.name]
            external_data_helper.set_external_data(tensor, location=weight_path.name,
                                                   offset=start, length=length)
            tensor.ClearField("raw_data")
            tensor.data_location = TensorProto.EXTERNAL
            graph.graph.initializer.append(tensor)
            graph.graph.input.remove(value)
    onnx.save(graph, path)
    onnx.checker.check_model(str(path))
    return [value.name for value in graph.graph.input]


def write_case(directory, model, names, inputs, outputs, output_names):
    directory.mkdir(parents=True, exist_ok=False)
    case = {"model": "../" + model, "repeat": 2, "inputs": []}
    expected = []
    for destination, values in [(case["inputs"], zip(names, inputs)),
                                (expected, zip(output_names, outputs))]:
        for name, value in values:
            array = value.detach().cpu().numpy()
            dtype = {np.dtype("float32"): "f32", np.dtype("int64"): "i64"}[array.dtype]
            prefix = "expected-" if destination is expected else ""
            path = f"{prefix}{name}.{dtype}"
            array.tofile(directory / path)
            destination.append(dict(name=name, dtype=dtype, shape=list(array.shape), path=path))
    json_write(directory / "case.json", case)
    json_write(directory / "expected.json", expected)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    pin = next(item for item in json.loads((ROOT / "crates/host/src/model-artifacts.json").read_text())
               if item["id"] == "cohere-speech")
    for name in ("model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json"):
        path = args.model / name
        assert path.stat().st_size == pin["files"][name]["size"], name
        assert sha256(path) == pin["files"][name]["sha256"], name
    reference_path = ROOT / "crates/models/tests/data/cohere-reference.json"
    reference = json.loads(reference_path.read_text())
    wav_path = ROOT / "crates/models/tests/data/two-voice.wav"
    assert sha256(wav_path) == reference["wav_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    prompts = {language: prompt(tokenizer, language) for language in LANGUAGES}
    with wave.open(str(wav_path)) as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    extractor = CohereAsrFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    feature_data = extractor(pcm, sampling_rate=16000, return_tensors="pt")
    features, mask = feature_data.input_features.float(), feature_data.attention_mask.to(torch.int64)
    assert mask.sum().item() == reference["valid_frames"]
    print("Loading exact packed checkpoint into maintained FP32 graph", flush=True)
    model = load_model(args.model)
    encoder, decoder = Encoder(model), Decoder(model)
    started = time.perf_counter()
    emitted, encoded, encoder_mask = transcribe(encoder, decoder, features, mask, prompts["en"])
    eager = dict(tokens=emitted, text=tokenizer.decode(emitted, skip_special_tokens=True),
                 duration_s=time.perf_counter() - started, valid_frames=int(mask.sum()),
                 encoded_valid_frames=int(encoder_mask.sum()))
    json_write(args.output / "eager-reference-comparison.json", eager)
    assert emitted == reference["tokens"], ("maintained graph does not match independent tokens", eager)
    assert eager["text"] == reference["text"]
    print("Independent text and all 31 tokens match", flush=True)
    json_write(args.output / "prompts.json", prompts)
    shutil.copyfile(args.model / "tokenizer.json", args.output / "tokenizer.json")
    shutil.copyfile(reference_path, args.output / "independent-reference.json")
    encoder_names = ["features", "attention_mask"]
    encoder_outputs = ["encoded", "encoder_mask"]
    encoder_args = (features, mask)
    kept = export_graph(encoder, encoder_args, args.output / "encoder.onnx", encoder_names,
                        encoder_outputs, {"features": {1: "frames"}, "attention_mask": {1: "frames"},
                                          "encoded": {1: "encoded_frames"},
                                          "encoder_mask": {1: "encoded_frames"}},
                        (args.output / "encoder.weights", {}))
    assert kept == encoder_names
    write_case(args.output / "encoder-case", "encoder.onnx", kept, encoder_args,
               (encoded, encoder_mask), encoder_outputs)
    names = ["tokens", "positions", "encoded", "self_mask", "cross_mask"]
    output_names = ["logits", *["present_" + name for name in CACHE_NAMES]]
    init_args = decoder_inputs(prompts["en"], 0, encoded, encoder_mask)
    init_result = decoder(*init_args)
    step_args = decoder_inputs([int(init_result[0].argmax(-1))], len(prompts["en"]),
                               encoded, encoder_mask, init_result[1:])
    step_result = decoder(*step_args)
    store = (args.output / "decoder.weights", {})
    for label, values in [("init", init_args), ("step", step_args)]:
        all_names = names + ([] if label == "init" else ["past_" + name for name in CACHE_NAMES])
        dynamic = {"tokens": {1: "tokens"}, "positions": {1: "tokens"},
                   "encoded": {1: "encoded_frames"},
                   "self_mask": {2: "tokens", 3: "total_tokens"},
                   "cross_mask": {2: "tokens", 3: "encoded_frames"}}
        for name in all_names[5:] + output_names[1:]:
            dynamic[name] = {2: "encoded_frames" if "cross_" in name else
                             ("past_tokens" if name.startswith("past_") else "total_tokens")}
        kept = export_graph(decoder, values, args.output / f"decoder-{label}.onnx", all_names,
                            output_names, dynamic, store)
        selected = [values[all_names.index(name)] for name in kept]
        write_case(args.output / f"decoder-{label}-case", f"decoder-{label}.onnx", kept, selected,
                   init_result if label == "init" else step_result, output_names)
    files = [path for path in args.output.iterdir() if path.is_file()]
    artifacts = {path.name: dict(size=path.stat().st_size, sha256=sha256(path)) for path in files}
    for label in ("encoder", "decoder-init", "decoder-step"):
        case_path = args.output / f"{label}-case" / "case.json"
        case = json.loads(case_path.read_text())
        weights = "encoder.weights" if label == "encoder" else "decoder.weights"
        case["model_files"] = {"../" + weights: artifacts[weights]["sha256"]}
        json_write(case_path, case)
    provenance = dict(source=pin, precision="FP32 activations and exact affine-dequantized FP32 weights; not MLX BF16 kernels",
                      quantization="w[row,k] = float(q_u4) * float(scale_bf16[row,k//64]) + float(bias_bf16[row,k//64])",
                      independent_reference_sha256=sha256(reference_path), wav_sha256=sha256(wav_path),
                      exporter_sha256=sha256(__file__),
                      packages={name: importlib.metadata.version(name) for name in
                                ("torch", "transformers", "onnx", "numpy", "safetensors", "tokenizers")},
                      upstream_graph_sources={name: sha256(Path(__import__("transformers").__file__).parent / name)
                                              for name in ("models/cohere_asr/modeling_cohere_asr.py",
                                                           "models/parakeet/modeling_parakeet.py")},
                      outputs=artifacts)
    json_write(args.output / "provenance.json", provenance)
    print(json.dumps({"output": str(args.output), "eager": eager, "files": provenance["outputs"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
