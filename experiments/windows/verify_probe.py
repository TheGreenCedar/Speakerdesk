"""Admit a native graph run before comparing its numerical outputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath

# Microsoft.Windows.AI.MachineLearning 2.4.89, x64; API 27.
WINDOWS_ML_SHA256 = "6c916621c1807eba630388868bff8eb4cc9718b9fa17464fc9368eb14ad0f33e"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(case_directory, result_directory, provider):
    case_directory, result_directory = Path(case_directory), Path(result_directory)
    case_path = case_directory / "case.json"
    case = json.loads(case_path.read_text())
    record = json.loads((result_directory / "result.json").read_text())
    if provider not in ("cpu", "dml") or record["provider_requested"] != provider:
        raise ValueError("native result belongs to a different provider")
    if record["runtime_sha256"] != WINDOWS_ML_SHA256 or record.get("execution_verified") is False:
        raise ValueError("native runtime or execution qualification differs")
    runtime_pin = json.loads((Path(__file__).resolve().parents[2] /
                             "crates/models/src/windows-runtime.json").read_text())["files"]
    libraries = {item["name"]: item for item in record["loaded_runtime_libraries"]}
    required = {"onnxruntime.dll", "DirectML.dll"} if provider == "dml" else {"onnxruntime.dll"}
    if not required.issubset(libraries) or any(
            name not in runtime_pin or item["sha256"] != runtime_pin[name]["sha256"]
            for name, item in libraries.items()):
        raise ValueError("actual loaded runtime libraries differ from their pins")
    if record["case_sha256"] != sha256(case_path):
        raise ValueError("native result belongs to a different case")
    if record["model_sha256"] != sha256(case_directory / case["model"]):
        raise ValueError("native result belongs to a different graph")
    for name, expected in case.get("model_files", {}).items():
        if sha256(case_directory / name) != expected or record["model_files"].get(name) != expected:
            raise ValueError("native external model parameters differ")
    expected_inputs = [{"name": value["name"], "dtype": value["dtype"], "shape": value["shape"],
                        "sha256": sha256(case_directory / value["path"])} for value in case["inputs"]]
    if record["inputs"] != expected_inputs:
        raise ValueError("native input identity differs")
    # Returned profiles may have been copied from Windows to a different evidence directory.
    profile = result_directory / PureWindowsPath(record["profile"]).name
    events = json.loads(profile.read_text())
    placements = Counter(event["args"]["provider"] for event in events
                         if event.get("args", {}).get("provider"))
    if dict(placements) != record["executed_provider_events"]:
        raise ValueError("native profile disagrees with the execution record")
    if provider == "dml" and not placements["DmlExecutionProvider"]:
        raise ValueError("DirectML registration without executed model work")
    if provider == "cpu" and (not placements["CPUExecutionProvider"] or
                               set(placements) != {"CPUExecutionProvider"}):
        raise ValueError("CPU control executed on a different provider")
    return record
