"""Compare admitted native ReDimNet outputs with the independent saved-profile vectors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify

IDENTITY_REUSE_COSINE = 0.9999


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    assert vector.shape == (192,) and np.isfinite(vector).all()
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-5, ("graph output is not L2 normalized", norm)
    return vector / norm


def compare(case, result, provider):
    record = verify(case, result, provider)
    expected = json.loads((case / "expected.json").read_text())
    repetitions = json.loads((case / "case.json").read_text())["repeat"]
    seen, vectors, rows = set(), [], []
    for value in record["outputs"]:
        assert value["name"] == "embedding" and value["dtype"] == "f32" and value["shape"] == [1, 192]
        assert value["run"] not in seen
        vector = normalized(np.fromfile(result / value["file"], dtype="<f4"))
        cosine = float(np.dot(vector, expected["vector"]))
        assert cosine >= IDENTITY_REUSE_COSINE, ("independent saved-profile gate failed", cosine)
        repeat_cosine = float(np.dot(vector, vectors[0])) if vectors else 1.0
        assert repeat_cosine >= IDENTITY_REUSE_COSINE, ("resident output changed", repeat_cosine)
        vectors.append(vector)
        seen.add(value["run"])
        rows.append(dict(run=value["run"], legacy_cosine=cosine, repeat_cosine=repeat_cosine))
    assert seen == set(range(repetitions)), seen
    print(json.dumps(dict(passed=True, speaker=expected["speaker"], runs=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    args = parser.parse_args()
    compare(args.case, args.result, args.provider)
