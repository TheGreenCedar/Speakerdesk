"""Drive native graph calls with their own recurrent outputs, including cache compression."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from export import ROOT, sha256
from qualify import Session, compare, read_pcm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_probe import verify


class NativeGraphs:
    def __init__(self, args, provider, directory):
        self.args, self.provider, self.directory = args, provider, directory
        directory.mkdir()
        self.calls = 0
        self.nonempty_cache_calls = 0

    def run(self, model, inputs):
        case = self.directory / f"{self.calls:03}-{model}"
        case.mkdir()
        values = []
        for name, array in inputs.items():
            dtype = "i64" if array.dtype == np.int64 else "f32"
            array.astype("<i8" if dtype == "i64" else "<f4").tofile(case / f"{name}.{dtype}")
            values.append(dict(name=name, dtype=dtype, shape=list(array.shape), path=f"{name}.{dtype}"))
        (case / "case.json").write_text(json.dumps(dict(
            model=str((self.args.artifacts / f"{model}.onnx").resolve()), repeat=1, inputs=values), indent=2))
        result = case / "result"
        with (case / "process.log").open("w") as stream:
            subprocess.run([str(self.args.probe), str(self.args.runtime), self.provider,
                            str(case / "case.json"), str(result)], check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
        record = verify(case, result, self.provider)
        outputs = {}
        for row in record["outputs"]:
            path = result / row["file"]
            if row["run"] != 0 or row["dtype"] != "f32" or sha256(path) != row["sha256"]:
                raise ValueError("native output identity differs")
            value = np.fromfile(path, "<f4").reshape(row["shape"])
            if not np.isfinite(value).all():
                raise ValueError("nonfinite native output")
            outputs[row["name"]] = value
        self.calls += 1
        if model == "nemotron":
            self.nonempty_cache_calls += inputs["cache"].shape[1] > 0
            if set(outputs) != {"chunk", "high", "low"}:
                raise ValueError("incomplete native neural output")
            return [outputs[name] for name in ["chunk", "high", "low"]]
        if set(outputs) != {"features"}:
            raise ValueError("incomplete native frontend output")
        return outputs["features"]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "crates/models/tests/data/two-voice.wav"
    # Six identical source windows exercise spill and compression without inventing an acoustic oracle.
    pcm = np.tile(read_pcm(source), 6)
    reports = []
    for preset in ["offline", "low"]:
        predictions = {}
        for provider in ["cpu", "dml"]:
            graphs = NativeGraphs(args, provider, args.output / f"{preset}-{provider}")
            session = Session(lambda inputs: graphs.run("nemotron", inputs), args.artifacts,
                              preset, frontend=lambda inputs: graphs.run("frontend", inputs))
            predictions[provider] = session.feed(pcm, True)
            predictions[provider].astype("<f4").tofile(graphs.directory / "probabilities.f32")
            report = dict(preset=preset, provider=provider, calls=graphs.calls,
                          frames=len(predictions[provider]), compressed=session.compressed,
                          nonempty_cache_calls=int(graphs.nonempty_cache_calls))
            print(json.dumps(report), flush=True)
            if not session.compressed or not graphs.nonempty_cache_calls:
                raise ValueError("fixture failed to exercise cache compression and continuation")
            reports.append(report)
        comparison = compare(predictions["dml"], predictions["cpu"])
        reports.append(dict(preset=preset, comparison=comparison))
        (args.output / "progress.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps(comparison), flush=True)
        if comparison["max_probability_error"] > .001 or comparison["changed_activity_decisions"]:
            raise ValueError("native recurrent CPU/DirectML behavior differs")
    result = dict(passed=True, source_sha256=sha256(source), repetitions=6, reports=reports,
                  limit="CPU/DirectML recurrent parity; independent long acoustic reference remains separate.")
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    main(parser.parse_args())
