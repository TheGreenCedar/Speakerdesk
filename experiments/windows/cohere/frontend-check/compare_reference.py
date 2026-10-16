"""Diagnose native frontend drift from maintained Transformers; decoding remains the acceptance gate."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
from transformers import CohereAsrFeatureExtractor


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    pcm = np.fromfile(args.pcm, dtype="<f4")
    assert pcm.size >= 320, "valid=1 follows the finite Mac guard; maintained Torch divides by zero"
    native = np.fromfile(args.features, dtype="<f4").reshape(1, pcm.size // 160 + 1, 128)
    extractor = CohereAsrFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    reference = extractor(pcm, sampling_rate=16000, return_tensors="pt")
    expected = reference.input_features.numpy()
    assert native.shape == expected.shape and int(reference.attention_mask.sum()) == pcm.size // 160
    assert np.isfinite(native).all() and np.isfinite(expected).all()
    left, right = native.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    difference = np.abs(left - right)
    report = dict(samples=pcm.size, valid_frames=pcm.size // 160, shape=list(native.shape),
        max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        rmse=float(np.sqrt(np.mean(difference * difference))),
        cosine=float(np.dot(left, right) / np.linalg.norm(left) / np.linalg.norm(right)),
        input_sha256=sha256(args.pcm), native_features_sha256=sha256(args.features),
        reference_features_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        reference_source_sha256=sha256(inspect.getfile(CohereAsrFeatureExtractor)),
        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "librosa")},
        limitation="Native StdRng/StandardNormal and Torch use different per-crop normal samples and backend math. This diagnostic does not qualify transcription.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("pcm", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    run(parser.parse_args())
