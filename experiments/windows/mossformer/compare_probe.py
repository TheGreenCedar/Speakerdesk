"""Admit native MossFormer graph calls and compare their numerical outputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from export import sha256
from qualify import waveform_error
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


def expected_values(directory, record):
    path = directory / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError("comparison tensor hash changed")
    data = np.fromfile(path, "<f4")
    if data.size != np.prod(record["shape"]) or not np.isfinite(data).all():
        raise ValueError("invalid comparison tensor")
    return data.reshape(record["shape"])


def compare(case, result, provider):
    record = verify(case, result, provider)
    expected = json.loads((case / "expected.json").read_text())
    independent_path = case / "independent.json"
    independent = json.loads(independent_path.read_text()) if independent_path.exists() else {}
    rows = record["outputs"]
    if len(rows) != 2 * len(expected) or {(row["run"], row["name"]) for row in rows} != {
            (run, name) for run in [0, 1] for name in expected}:
        raise ValueError("expected every output of two native repetitions")
    reports, previous = [], {}
    accepted = True
    for row in rows:
        target = expected[row["name"]]
        if row["dtype"] != "f32" or row["shape"] != target["shape"]:
            raise ValueError("native output type or shape changed")
        actual = expected_values(result, row)
        control = waveform_error(actual, expected_values(case, target))
        report = {"run": row["run"], "name": row["name"], "cpu_control": control}
        accepted &= control["within_original_bounds"]
        if row["name"] in independent:
            original = waveform_error(actual, expected_values(case, independent[row["name"]]))
            report["independent_full_window"] = original
            accepted &= original["within_original_bounds"]
        if row["name"] in previous:
            error = float(np.max(np.abs(actual - previous[row["name"]]), initial=0))
            report["repeat_max_error"] = error
            accepted &= error == 0
        previous[row["name"]] = actual
        reports.append(report)
    response = {"numerical_controls_passed": bool(accepted), "reports": reports,
                "scope": "Per-call GPU/CPU comparison; prepared state is from the complete CPU chain."}
    print(json.dumps(response, indent=2))
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    args = parser.parse_args()
    result = compare(args.case_directory, args.result_directory, args.provider)
    raise SystemExit(0 if result["numerical_controls_passed"] else 1)
