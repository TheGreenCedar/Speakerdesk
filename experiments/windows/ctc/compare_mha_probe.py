"""Original CTC numerical gate plus rejection of CPU attention in a DML candidate run."""
import argparse
import json
from pathlib import Path, PureWindowsPath

import onnx

from qualify import compare_native


def check_attention_placement(events, provider):
    if provider == "dml" and any(
        event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event["args"].get("op_name") in ("Attention", "MultiHeadAttention")
        for event in events
    ):
        raise ValueError("CTC attention fell back to CPU despite other DirectML model work")


def main(args):
    specification = json.loads((args.case / "case.json").read_text())
    graph = onnx.load(args.case / specification["model"], load_external_data=False)
    if sum(n.domain == "com.microsoft" and n.op_type == "MultiHeadAttention" for n in graph.graph.node) != 24:
        raise ValueError("native case is not the 24-block MHA candidate")
    record = json.loads((args.result / "result.json").read_text())
    profile = args.result / PureWindowsPath(record["profile"]).name
    check_attention_placement(json.loads(profile.read_text()), args.provider)
    # Reuse the complete shared runtime/input/provider admission and unchanged 1e-4 score gate.
    return compare_native(args.case, args.result, args.provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"), required=True)
    raise SystemExit(0 if main(parser.parse_args())["passed"] else 1)
