"""Seal and copy retained raw forward references without running a model."""
import argparse
import array
import json
import math
from pathlib import Path
import shutil

from export import ROOT, SOURCE_FILES, SOURCE_REVISION, json_write, sha256


def prepare(references, output):
    pin = json.loads((ROOT / "crates/host/src/separation-artifact.json").read_text())
    checkpoint_hash = pin["files"]["last_best_checkpoint.pt"]["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, directory in enumerate(references):
        path = directory / "reference.json"
        reference = json.loads(path.read_text())
        if (reference["checkpoint_sha256"] != checkpoint_hash or reference["source_revision"] != SOURCE_REVISION
                or reference["source_sha256"] != SOURCE_FILES or reference["sample_rate"] != 16000
                or reference["forward"] != "raw full-window float32 CPU; no wrapper"):
            raise ValueError("incompatible independent MossFormer2 reference")
        for case in reference["cases"]:
            samples = case["samples"]
            if not 16 <= samples <= 392000 or len(case["output"]) != 2:
                raise ValueError("invalid independent waveform shape")
            files = []
            for name in [case["input"], *case["output"]]:
                if Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError("reference paths must be filenames")
                source = directory / name
                data = source.read_bytes()
                values = array.array("f")
                values.frombytes(data)
                if len(data) != samples * 4 or not all(math.isfinite(value) for value in values):
                    raise ValueError("invalid independent raw Float32 waveform")
                destination = output / f"ref-{index}" / name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, destination)
                files.append({"path": str(destination.relative_to(output)), "sha256": sha256(destination)})
            cases.append({"id": f"ref-{index}-{case['name']}", "samples": samples,
                          "original_record_sha256": sha256(path), "input": files[0], "outputs": files[1:]})
    manifest = {"checkpoint_sha256": checkpoint_hash, "source_revision": SOURCE_REVISION,
                "source_sha256": SOURCE_FILES, "cases": cases}
    json_write(output / "references.json", manifest)
    print(json.dumps({"cases": len(cases), "manifest": str(output / "references.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="fresh private directory")
    parser.add_argument("references", nargs="+", type=Path, help="existing independent reference directories")
    args = parser.parse_args()
    prepare(args.references, args.output)
