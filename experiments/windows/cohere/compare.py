"""Compare the native Rust ORT probe's named outputs against retained eager case tensors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


def compare(case, result, provider, atol, rtol):
    expected = {value["name"]: value for value in json.loads((case / "expected.json").read_text())}
    records = verify(case, result, provider)["outputs"]
    assert records, "empty native probe result"
    seen = {}
    rows = []
    for record in records:
        name = record["name"]
        item = expected[name]
        assert record["dtype"] == item["dtype"] and record["shape"] == item["shape"], record
        dtype = {"f32": "<f4", "i64": "<i8"}[item["dtype"]]
        wanted = np.fromfile(case / item["path"], dtype=dtype).reshape(item["shape"])
        actual = np.fromfile(result / record["file"], dtype=dtype).reshape(item["shape"])
        if item["dtype"] == "i64":
            np.testing.assert_array_equal(actual, wanted, err_msg=name)
        else:
            assert np.isfinite(actual).all(), name
            np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol, err_msg=name)
        if name == "logits":
            assert actual.argmax(-1).tolist() == wanted.argmax(-1).tolist()
        seen.setdefault(record["run"], set()).add(name)
        rows.append(dict(run=record["run"], name=name,
                         max_abs=float(np.max(np.abs(actual - wanted), initial=0))))
    repetitions = json.loads((case / "case.json").read_text())["repeat"]
    assert set(seen) == set(range(repetitions)), seen
    assert all(names == set(expected) for names in seen.values()), seen
    print(json.dumps(dict(passed=True, runs=len(seen), outputs=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.002)
    args = parser.parse_args()
    compare(args.case, args.result, args.provider, args.atol, args.rtol)
