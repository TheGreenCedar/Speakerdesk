"""Compare one CPU/DML tap pair and detect changes to their original embeddings.

Arguments: diagnostic case, CPU result, DML result, original case, original CPU
result, original DML result. Results include raw profiles from the native probe.
This reports diagnostic differences; it does not qualify the instrumented graph.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import sha256, verify


IDENTITY_REUSE_COSINE = .9999


def read_outputs(record, directory, names, repetitions):
    rows = record["outputs"]
    if (len(rows) != len(names) * repetitions or
            {(row["run"], row["name"]) for row in rows} !=
            {(run, name) for run in range(repetitions) for name in names}):
        raise ValueError("missing or duplicate native outputs")
    outputs = {}
    for row in rows:
        path = directory / row["file"]
        if row["dtype"] != "f32" or sha256(path) != row["sha256"]:
            raise ValueError("native output type or hash differs")
        values = np.fromfile(path, "<f4")
        if (any(dim < 0 for dim in row["shape"]) or
                not values.size or values.size != math.prod(row["shape"]) or not np.isfinite(values).all()):
            raise ValueError("invalid native output")
        if row["name"] == "embedding" and row["shape"] != [1, 192]:
            raise ValueError("unexpected embedding shape")
        outputs[row["run"], row["name"]] = values.reshape(row["shape"])
    return outputs


def difference(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(("tensor shape differs", actual.shape, reference.shape))
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    delta = actual - reference
    norms = np.linalg.norm(actual), np.linalg.norm(reference)
    return dict(shape=list(actual.shape), max_absolute_error=float(np.max(np.abs(delta), initial=0)),
                rms_error=float(np.sqrt(np.mean(delta ** 2))),
                relative_l2_error=float(np.linalg.norm(delta) / norms[1]) if norms[1] else None,
                cosine=float(np.sum(actual * reference) / np.prod(norms)) if all(norms) else None,
                reference_min=float(reference.min()), reference_max=float(reference.max()),
                actual_min=float(actual.min()), actual_max=float(actual.max()))


def embedding_report(actual, baseline, reference):
    if any(not np.isclose(np.linalg.norm(value.astype(np.float64)), 1, rtol=0, atol=1e-5)
           for value in [actual, baseline]):
        raise ValueError("embedding is not normalized")
    comparison = difference(actual, baseline)
    actual_cosine = difference(actual.reshape(-1), reference)["cosine"]
    baseline_cosine = difference(baseline.reshape(-1), reference)["cosine"]
    return dict(baseline_comparison=comparison, legacy_cosine=actual_cosine,
                baseline_legacy_cosine=baseline_cosine,
                original_gate_passed=baseline_cosine >= IDENTITY_REUSE_COSINE,
                diagnostic_gate_passed=actual_cosine >= IDENTITY_REUSE_COSINE,
                numerical_perturbation=comparison["max_absolute_error"] != 0,
                gate_changed=(actual_cosine >= IDENTITY_REUSE_COSINE) !=
                             (baseline_cosine >= IDENTITY_REUSE_COSINE))


def compare(case, cpu, dml, original_case, original_cpu, original_dml):
    metadata = json.loads((case / "taps.json").read_text())
    original_specification = json.loads((original_case / "case.json").read_text())
    specification = json.loads((case / "case.json").read_text())
    if (sha256(original_case / "case.json") != metadata["source_case_sha256"] or
            sha256(original_case / original_specification["model"]) != metadata["source_graph_sha256"] or
            sha256(original_case / "expected.json") != metadata["independent_reference_sha256"] or
            sha256(case / specification["model"]) != metadata["graph_sha256"] or
            sha256(case / specification["inputs"][0]["path"]) != metadata["input_sha256"]):
        raise ValueError("diagnostic and original case identities differ")
    names = ["embedding", *[tap["name"] for tap in metadata["taps"]]]
    if specification["repeat"] != 1:
        raise ValueError("expected one diagnostic run per provider")
    tapped, baselines = {}, {}
    for provider, output, baseline in [("cpu", cpu, original_cpu), ("dml", dml, original_dml)]:
        tapped[provider] = read_outputs(verify(case, output, provider), output, names, 1)
        baseline_record = verify(original_case, baseline, provider)
        if baseline_record["inputs"] != json.loads((output / "result.json").read_text())["inputs"]:
            raise ValueError("diagnostic changed the original inputs")
        baselines[provider] = read_outputs(baseline_record, baseline, ["embedding"],
                                         original_specification["repeat"])
    reference = np.asarray(json.loads((original_case / "expected.json").read_text())["vector"], np.float64)
    if (reference.shape != (192,) or not np.isfinite(reference).all() or
            np.linalg.norm(reference) == 0):
        raise ValueError("invalid independent embedding reference")
    taps = []
    for tap in metadata["taps"]:
        name = tap["name"]
        actual, expected = tapped["dml"][0, name], tapped["cpu"][0, name]
        if (len(expected.shape) != len(tap["shape"]) or
                any(isinstance(dim, int) and dim != actual_dim
                    for dim, actual_dim in zip(tap["shape"], expected.shape))):
            raise ValueError(("tap shape differs from graph", name))
        taps.append(dict(name=name, source=tap["source"], **difference(actual, expected)))
    embeddings = {provider: [dict(baseline_run=run, **embedding_report(
                                tapped[provider][0, "embedding"], baseline, reference))
                             for (run, name), baseline in baselines[provider].items()]
                  for provider in ["cpu", "dml"]}
    result = dict(diagnostic_only=True, taps=taps, embeddings=embeddings,
                  instrumentation_changed_embedding=any(row["numerical_perturbation"]
                        for rows in embeddings.values() for row in rows),
                  instrumentation_changed_legacy_gate=any(row["gate_changed"]
                        for rows in embeddings.values() for row in rows),
                  limitation="Taps can change graph fusion/placement. Interpret tap differences together with original-provider embedding comparisons; no qualification pass is emitted.")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["case", "cpu", "dml", "original_case", "original_cpu", "original_dml"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    compare(args.case, args.cpu, args.dml, args.original_case, args.original_cpu, args.original_dml)
