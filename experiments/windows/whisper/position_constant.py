"""The complete pinned MLX encoder positional parameter, independent of PCM."""
import hashlib
import json
from pathlib import Path

import numpy as np


PROVENANCE = json.loads(Path(__file__).with_name("position-constant.json").read_text())


def load_positions(path):
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != PROVENANCE["shape"] or values.dtype not in (np.float16, np.float32):
        raise ValueError("unexpected Whisper positional parameter shape or dtype")
    half = values.astype("<f2")
    if not np.isfinite(values).all() or not np.array_equal(values, half.astype(values.dtype)):
        raise ValueError("Whisper positional parameter must contain exact finite Half values")
    if hashlib.sha256(half.tobytes()).hexdigest() != PROVENANCE["raw_little_endian_sha256"]:
        raise ValueError("Whisper positional parameter does not match the pinned MLX source table")
    return half
