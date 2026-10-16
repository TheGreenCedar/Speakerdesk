"""Check shared native ORT probe results against prepared graph-call tensors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from export import sha256
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


def compare(case, result, tolerance, provider):
    expected = json.loads((case / "expected.json").read_text())
    record = verify(case, result, provider)
    reports = []
    rows = record["outputs"]
    if {row["name"] for row in rows} != set(expected):
        raise ValueError("missing or unexpected outputs")
    runs = sorted({row["run"] for row in rows})
    identities = {(row["run"], row["name"]) for row in rows}
    if runs != [0, 1] or len(rows) != len(expected) * 2 or identities != {
            (run, name) for run in [0, 1] for name in expected}:
        raise ValueError("expected exactly two complete graph runs")
    for row in rows:
        reference = expected[row["name"]]
        path = case / reference["file"]
        if sha256(path) != reference["sha256"] or row["dtype"] != "f32" or row["shape"] != reference["shape"]:
            raise ValueError(("output identity mismatch", row["name"]))
        output = result / row["file"]
        if sha256(output) != row["sha256"]:
            raise ValueError("output hash mismatch")
        actual = np.fromfile(output, "<f4")
        wanted = np.fromfile(path, "<f4")
        if actual.shape != wanted.shape or not np.isfinite(actual).all():
            raise ValueError(("invalid result", row["name"]))
        error = float(np.max(np.abs(actual - wanted), initial=0))
        decisions = int(np.count_nonzero((actual > .5) != (wanted > .5))) if row["name"] in ["high", "low"] else None
        reports.append({"run": row["run"], "name": row["name"], "max_error": error,
                        "changed_activity_decisions": decisions})
        if error > tolerance or decisions:
            raise ValueError(reports[-1])
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--tolerance", type=float, default=.001)
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    args = parser.parse_args()
    compare(args.case_directory, args.result_directory, args.tolerance, args.provider)
