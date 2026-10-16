"""Check native Whisper outputs against original MLX references and macOS ORT CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from export import sha256
from qualify import score_report
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


def compare(case, result, provider):
    expected = json.loads((case / "expected.json").read_text())
    record = verify(case, result, provider)
    reference_path = case / expected["file"]
    if sha256(reference_path) != expected["sha256"]:
        raise ValueError("prepared CPU reference hash mismatch")
    wanted = np.fromfile(reference_path, "<f4").reshape(expected["shape"])
    rows = record["outputs"]
    if len(rows) != 2 or {(row["run"], row["name"]) for row in rows} != {
            (run, expected["name"]) for run in [0, 1]}:
        raise ValueError("expected two complete output repetitions")
    reports, previous = [], None
    qualified = True
    for row in rows:
        path = result / row["file"]
        if row["dtype"] != "f32" or row["shape"] != expected["shape"] or sha256(path) != row["sha256"]:
            raise ValueError("native output identity mismatch")
        actual = np.fromfile(path, "<f4")
        if actual.size != wanted.size or not np.isfinite(actual).all():
            raise ValueError("invalid native output")
        actual = actual.reshape(wanted.shape)
        report = {"run": row["run"], "name": row["name"],
                  "mac_cpu_max_error": float(np.max(np.abs(actual - wanted))),
                  "repeat_max_error": None if previous is None else float(np.max(np.abs(actual - previous)))}
        independent = expected["independent"]
        if row["name"] == "probabilities":
            comparison = score_report(actual, independent["probabilities"], independent["codes"])
            report["independent"] = comparison
            same_decision = all(comparison["actual_route"][key] == comparison["expected_route"][key]
                                for key in ["code", "confident"])
            qualified &= comparison["max_probability_error"] < 1e-4 and same_decision
        else:
            error = max(abs(float(actual[0, point["frame"], point["band"]]) - point["value"])
                        for point in independent["sampled_mel"])
            report["independent_sampled_mel_error"] = error
            qualified &= error < 1e-4
        if previous is not None:
            qualified &= report["repeat_max_error"] == 0
        previous = actual
        reports.append(report)
    response = {"qualified_under_original_1e_4_bound": bool(qualified), "reports": reports}
    print(json.dumps(response, indent=2))
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    args = parser.parse_args()
    result = compare(args.case_directory, args.result_directory, args.provider)
    raise SystemExit(0 if result["qualified_under_original_1e_4_bound"] else 1)
