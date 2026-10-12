"""Dev-only raw forward reference from the unmodified pinned ClearerVoice source.

Dependencies: torch==2.8.0 numpy==2.3.3 einops==0.8.1
rotary-embedding-torch==0.8.9 torchinfo==1.8.0.
No decode.py stitching, RMS normalization, quantization, or app cache writes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
CHECKPOINT_SHA256 = "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
SOURCE_FILES = {
    "conv_module.py": "b413aa0cb0df34742460a1f1af2e2cd1d2005900af8ac17e8de8675b8f80de86",
    "fsmn.py": "62c376b2708ac1c59853c13cbe9acb7ea011082c39f1a42e1af0f8213a00bb76",
    "layer_norm.py": "d28fea41b3c5a2dd8bb7a1a8c1c97492fcc7eb88cc7b46283fc21e4dbf19f062",
    "mossformer2.py": "66f14af8efc0758d8c79bd7d1054851558081c5f6f1150cdf133ae4b6ca9c442",
    "mossformer2_block.py": "2760c18c1eb68f8ba2f1160e9a46dc6adfbcb397a693dad27ae6e43e8a02faf0",
}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path, help="parent of an upstream/ package with the pinned model files")
    parser.add_argument("wav", type=Path, help="public mono 16 kHz WAV")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", default="mix_2s,partial_tail,silence,tail_impulse,mix_whole")
    args = parser.parse_args()
    assert digest(args.checkpoint) == CHECKPOINT_SHA256
    for name, expected in SOURCE_FILES.items():
        assert digest(args.source / "upstream" / name) == expected, name
    sys.path.insert(0, str(args.source))
    from upstream.mossformer2 import MossFormer
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    model = MossFormer().eval()
    model.load_state_dict(state, strict=True)
    assert len(state) == 1099
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
    with wave.open(str(args.wav)) as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    assert len(pcm) >= 32007 and np.isfinite(pcm).all()
    tail = np.zeros(32007, dtype=np.float32)
    tail[-9] = 1
    cases = {"mix_2s": pcm[:32000], "partial_tail": pcm[:32007], "silence": np.zeros(32000, dtype=np.float32), "tail_impulse": tail, "mix_whole": pcm}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"checkpoint_sha256": CHECKPOINT_SHA256, "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_FILES,
              "torch_version": importlib.metadata.version("torch"), "rotary_embedding_version": importlib.metadata.version("rotary-embedding-torch"),
              "wav_sha256": digest(args.wav), "sample_rate": 16000, "forward": "raw full-window float32 CPU; no wrapper", "cases": []}
    torch.set_num_threads(4)
    with torch.inference_mode():
        for name in args.cases.split(","):
            audio = cases[name]
            started = time.perf_counter()
            estimates = model(torch.from_numpy(audio.copy()).unsqueeze(0))
            assert len(estimates) == 2
            input_name = f"{name}-input.f32le"
            audio.astype("<f4").tofile(args.output / input_name)
            output_names = []
            for channel, estimate in enumerate(estimates):
                estimate = estimate.squeeze(0).numpy()
                assert estimate.shape == audio.shape and np.isfinite(estimate).all()
                output_name = f"{name}-channel{channel}.f32le"
                estimate.astype("<f4").tofile(args.output / output_name)
                output_names.append(output_name)
            row = {"name": name, "samples": len(audio), "input": input_name, "output": output_names, "seconds": time.perf_counter() - started}
            result["cases"].append(row)
            (args.output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
