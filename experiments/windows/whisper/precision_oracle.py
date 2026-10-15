"""Small MLX operator observation; never writes the independent model references."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observe(output):
    output.mkdir(parents=True, exist_ok=False)
    # Fixed synthetic logits at Whisper's attention width; no audio or golden fitting.
    inputs = np.random.default_rng(20261015).normal(0, 2, (1, 6, 3, 1500)).astype(np.float16)
    x = mx.array(inputs)
    outputs = {"precise": mx.softmax(x, axis=-1, precise=True),
               "default": mx.softmax(x, axis=-1, precise=False),
               "float_island": mx.softmax(x.astype(mx.float32), axis=-1).astype(mx.float16)}
    mx.eval(*outputs.values())
    arrays = {name: np.asarray(value.astype(mx.float32)) for name, value in outputs.items()}
    if not np.array_equal(arrays["precise"], arrays["float_island"]):
        raise ValueError("explicit Float32 boundary did not reproduce MLX precise softmax")
    if np.array_equal(arrays["precise"], arrays["default"]):
        raise ValueError("operator input did not distinguish the precision boundary")
    inputs.astype("<f4").tofile(output / "input.f32")
    for name, array in arrays.items():
        array.astype("<f4").tofile(output / (name + ".f32"))
    package = Path(mx.__file__).parent
    result = {"shape": list(inputs.shape), "mlx_version": mx.__version__,
              "device": str(mx.default_device()), "numpy_version": np.__version__,
              "script_sha256": digest(Path(__file__)),
              "runtime_sha256": digest(package / "lib/libmlx.dylib"),
              "metallib_sha256": digest(package / "lib/mlx.metallib"),
              "files": {p.name: digest(p) for p in sorted(output.glob("*.f32"))},
              "default_vs_precise_max_error": float(np.max(np.abs(arrays["default"] - arrays["precise"]))),
              "default_vs_precise_different": int(np.count_nonzero(arrays["default"] != arrays["precise"])),
              "float_island_vs_precise_max_error": 0.0}
    (output / "oracle.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    observe(parser.parse_args().output)
