"""Prepare frozen PCM or compare native captures; this tool never runs a model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import urllib.request
import wave

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = HERE / "windows-qualification-v2.json"
RETAINED_HASHES = {
    "fr-1-noise10db": "8442387b816edc0cc41ef5a3171ecb3901a9ca73f18a6ff1edeb62f68a778e11",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_pcm16(path):
    with wave.open(str(path)) as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
            raise ValueError("expected 16 kHz mono PCM16: " + str(path))
        return np.frombuffer(stream.readframes(stream.getnframes()), "<i2").astype(np.float32) / np.float32(32768)


def stress_cases(en, fr, noisy):
    rows = []
    for language, audio in [("en", en), ("fr", fr)]:
        rows.append(("quiet-" + language, audio * np.float32(.01), {"source": language, "gain": .01}))
        rows.append(("clipped-" + language, np.clip(audio * np.float32(4), -1, 1), {"source": language, "gain": 4}))
    gap = np.zeros(12800, np.float32)
    rows.extend([
        ("switch-en-fr", np.concatenate([en, gap, fr]), {"order": ["en", "fr"], "gap_samples": 12800}),
        ("switch-fr-en", np.concatenate([fr, gap, en]), {"order": ["fr", "en"], "gap_samples": 12800}),
    ])
    rms = [float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) for audio in [en, fr]]
    if min(rms) <= 0:
        raise ValueError("overlap needs two nonzero retained sources")
    scaled = [(audio.astype(np.float64) * (min(rms) / level)).astype(np.float32)
              for audio, level in zip([en, fr], rms)]
    size = max(map(len, scaled))
    overlap = sum((np.pad(audio, (0, size - len(audio))) for audio in scaled),
                  np.zeros(size, np.float32)) * np.float32(.5)
    rows.append(("overlap-en-fr", overlap, {"rms": rms, "target_rms": min(rms), "gain": .5}))
    rows.append(("noise-fr-10db", noisy, {"retained_id": "fr-1-noise10db"}))
    origin = int(np.flatnonzero(en)[0])
    for size in [159, 160, 161, 319, 320, 321, 399, 400, 401]:
        rows.append((f"short-{size}", en[origin:origin + size], {"source": "en_us-0", "origin": origin, "numerical_only": True}))
    for size in [479999, 480000, 480001]:
        audio = np.zeros(size, np.float32)
        audio[:len(en)] = en
        if size > 480000:
            audio[480000] = 1
        rows.append((f"context-{size}", audio, {"bitexact_to": "natural-en_us-0", "ignored_tail": size > 480000}))
    return rows


class Downloads:
    def __init__(self, budget):
        self.limit, self.bytes = budget, 0

    def get(self, url, limit):
        if not url.startswith("https://"):
            raise ValueError("corpus transport must use HTTPS")
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(min(limit, self.limit - self.bytes) + 1)
        self.bytes += len(data)
        if len(data) > limit or self.bytes > self.limit:
            raise ValueError("frozen corpus download budget exceeded")
        return data

    def stream(self, url):
        if not url.startswith("https://"):
            raise ValueError("corpus transport must use HTTPS")
        return BodyStream(urllib.request.urlopen(url, timeout=30), self)


class BodyStream:
    def __init__(self, response, downloads):
        self.response, self.downloads = response, downloads

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.response.close()

    def read(self, size):
        if size < 0 or self.downloads.bytes + size > self.downloads.limit:
            raise ValueError("archive read would exceed frozen cumulative corpus budget")
        data = self.response.read(size)
        self.downloads.bytes += len(data)
        return data


def prepare(destination, retained, prior_downloaded_bytes=0):
    contract = json.loads(CONTRACT.read_text())
    budget = contract["budget"]["maximum_new_download_bytes"]
    if not 0 <= prior_downloaded_bytes < budget:
        raise ValueError("prior corpus bytes exhaust the frozen budget")
    destination.mkdir(parents=True, exist_ok=False)
    downloads = Downloads(budget - prior_downloaded_bytes)
    manifest = {"schema_version": 1, "contract_id": contract["id"], "contract_sha256": sha256(CONTRACT),
                "sample_rate": 16000, "dtype": "f32le", "cases": [], "assets": [],
                "preparation_tool_sha256": sha256(Path(__file__)), "numpy": np.__version__,
                "transport_tool_sha256": sha256(HERE / "fleurs_archive.py"),
                "prior_downloaded_bytes": prior_downloaded_bytes}
    try:
        metadata = ROOT / "tests/metadata/whisper-tiny/added_tokens.json"
        if sha256(metadata) != contract["mac_models"]["added_tokens.json"]:
            raise ValueError("language metadata pin differs")
        manifest["codes"] = [token[2:-2] for token in json.loads(metadata.read_text())
                             if re.fullmatch(r"<\|[a-z]{2,3}\|>", token)]
        if len(manifest["codes"]) != 99:
            raise ValueError("expected exactly 99 language columns")

        def add(identifier, audio, group, recipe, human_language=None, original=None):
            audio = np.asarray(audio, dtype="<f4")
            if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError("invalid prepared audio: " + identifier)
            path = destination / (identifier + ".f32")
            with path.open("xb") as stream:
                stream.write(audio.tobytes())
            row = {"id": identifier, "file": path.name, "sha256": sha256(path), "samples": len(audio),
                   "group": group, "recipe": recipe, "human_language": human_language}
            if original is not None:
                row["legacy_reference"] = original
            manifest["cases"].append(row)

        legacy_path = ROOT / contract["legacy"]["reference"]
        wav = ROOT / "crates/models/tests/data/two-voice.wav"
        if sha256(legacy_path) != contract["legacy"]["reference_sha256"] or sha256(wav) != contract["legacy"]["wav_sha256"]:
            raise ValueError("retained independent fixture differs")
        legacy = json.loads(legacy_path.read_text())
        pcm = read_pcm16(wav)
        for audio, expected in zip([pcm, pcm[:48000], pcm[16000:16549], np.zeros(512, np.float32)], legacy["cases"]):
            add("legacy-" + expected["name"], audio, "legacy", {"source": wav.name, "case": expected["name"]},
                original=expected)
        raw = {}
        for config, language in contract["natural"]["configs"].items():
            from fleurs_archive import selected_audio
            selected, transport = selected_audio(config, contract["natural"]["revision"], downloads,
                contract["budget"]["maximum_metadata_bytes_per_config"], contract["budget"]["maximum_audio_asset_bytes"])
            write_json(destination / (config + "-members.json"), transport)
            for index, row in enumerate(selected):
                identifier = f"{config}-{index}"
                source = destination / (identifier + "-source.wav")
                source.write_bytes(row["data"])
                import soundfile as sf
                decoded, rate = sf.read(source, dtype="float32")
                if rate != 16000 or decoded.ndim != 1 or len(decoded) != row["num_samples"] or not np.isfinite(decoded).all():
                    raise ValueError("natural asset format/sample count differs")
                # Match the retained pilot's PCM16 preparation, not a new frontend.
                normalized = destination / (identifier + ".wav")
                sf.write(normalized, decoded, rate, subtype="PCM_16")
                audio = read_pcm16(normalized)
                asset = {"id": identifier, "archive_member": row["member"], "fleurs_id": row["id"],
                    "source_sha256": sha256(source), "wav_sha256": sha256(normalized),
                    "samples": len(audio), "words": row["raw_transcription"], "language": language}
                manifest["soundfile"] = importlib.metadata.version("soundfile")
                manifest["libsndfile"] = sf.__libsndfile_version__
                manifest["assets"].append(asset)
                raw[identifier] = audio
                add("natural-" + identifier, audio, "natural", {"asset": identifier}, language)
        noisy_path = retained / "fr-1-noise10db.wav"
        if sha256(noisy_path) != RETAINED_HASHES["fr-1-noise10db"]:
            raise ValueError("retained noise stress differs")
        for identifier, audio, recipe in stress_cases(raw["en_us-0"], raw["fr_fr-1"], read_pcm16(noisy_path)):
            add(identifier, audio, "stress", recipe)
        if len(manifest["cases"]) != contract["budget"]["cases"]:
            raise ValueError("frozen case count differs")
        manifest["downloaded_bytes"] = downloads.bytes
        manifest["cumulative_downloaded_bytes"] = prior_downloaded_bytes + downloads.bytes
        manifest["prepared_bytes"] = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
        if manifest["prepared_bytes"] > contract["budget"]["maximum_prepared_bytes"]:
            raise ValueError("frozen prepared storage budget exceeded")
        write_json(destination / "cases.json", manifest)
        print(json.dumps({"cases_sha256": sha256(destination / "cases.json"), "cases": len(manifest["cases"]),
                          "downloaded_bytes": downloads.bytes, "cumulative_downloaded_bytes": manifest["cumulative_downloaded_bytes"],
                          "prepared_bytes": manifest["prepared_bytes"]}, indent=2))
    except Exception as error:
        write_json(destination / "preparation-failed.json", {"passed": False, "error": str(error),
            "downloaded_bytes": downloads.bytes, "cumulative_downloaded_bytes": prior_downloaded_bytes + downloads.bytes,
            "partial_manifest": manifest})
        raise


def vector(row):
    values = np.asarray(row["probabilities"], dtype="<f4")
    if values.shape != (99,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("invalid native language probability vector")
    if abs(float(values.astype(np.float64).sum()) - 1) > 1e-5:
        raise ValueError("native language probabilities are not normalized")
    if hashlib.sha256(values.tobytes()).hexdigest() != row["scores_sha256"]:
        raise ValueError("native score bytes differ from their receipt")
    if row["probability_bits"] != values.view("<u4").tolist():
        raise ValueError("native score bit-pattern receipt differs")
    return values


def expected_ids(contract):
    result = [row["id"] for row in contract["legacy"]["cases"]]
    result += [f"natural-{config}-{row}" for config in contract["natural"]["configs"] for row in [0, 1]]
    result += ["quiet-en", "clipped-en", "quiet-fr", "clipped-fr", "switch-en-fr", "switch-fr-en", "overlap-en-fr", "noise-fr-10db"]
    result += [f"short-{size}" for size in contract["stress"]["short"]["lengths"]]
    result += [f"context-{size}" for size in contract["stress"]["context"]["lengths"]]
    return result


def compare(cases_path, mac_path, windows_path, output):
    contract = json.loads(CONTRACT.read_text())
    report = {"contract_id": contract["id"], "contract_sha256": sha256(CONTRACT), "passed": False,
              "activation_recommended": False, "cases": [], "errors": [],
              "captures": {"mac_file": str(mac_path), "windows_file": str(windows_path)}}
    try:
        report["captures"].update(mac_sha256=sha256(mac_path), windows_sha256=sha256(windows_path))
        cases = json.loads(cases_path.read_text())
        if cases["contract_sha256"] != sha256(CONTRACT) or [case["id"] for case in cases["cases"]] != expected_ids(contract):
            raise ValueError("case manifest contract/count differs")
        metadata = ROOT / "tests/metadata/whisper-tiny/added_tokens.json"
        if sha256(metadata) != contract["mac_models"]["added_tokens.json"]:
            raise ValueError("independent language metadata identity differs")
        codes = [token[2:-2] for token in json.loads(metadata.read_text()) if re.fullmatch(r"<\|[a-z]{2,3}\|>", token)]
        if cases["codes"] != codes:
            raise ValueError("case language columns differ from the pinned independent metadata")
        legacy_path = ROOT / contract["legacy"]["reference"]
        if sha256(legacy_path) != contract["legacy"]["reference_sha256"]:
            raise ValueError("independent legacy reference identity differs")
        legacy = {"legacy-" + row["name"]: row for row in json.loads(legacy_path.read_text())["cases"]}
        for case in cases["cases"]:
            if sha256(cases_path.parent / case["file"]) != case["sha256"]:
                raise ValueError("prepared PCM identity differs: " + case["id"])
            if case["id"] in legacy and case.get("legacy_reference") != legacy[case["id"]]:
                raise ValueError("embedded independent legacy expectation differs: " + case["id"])
        captures = [json.loads(path.read_text()) for path in [mac_path, windows_path]]
        if sha256(ROOT / "crates/models/src/windows-runtime.json") != contract["windows_runtime"]["pins_sha256"]:
            raise ValueError("frozen Windows runtime manifest differs")
        indexed = []
        for platform, capture in zip(["macos", "windows"], captures):
            if (capture["platform"] != platform or not capture["passed_capture"]
                    or capture["cases_sha256"] != sha256(cases_path) or capture["contract_sha256"] != sha256(CONTRACT)
                    or not 113 <= capture["successful_probes"] <= 128 or len(capture["rows"]) != 112):
                raise ValueError("native capture identity, completion or budget differs: " + platform)
            if any(capture["isolation"].get(key) is not True for key in ["pre_cancelled", "invalid_pcm", "after_failure_bitexact"]):
                raise ValueError("native request-isolation controls did not complete")
            for name, digest in {
                "test_source_sha256": ROOT / "crates/models/tests/whisper_windows_qualification.rs",
                "policy_source_sha256": ROOT / "crates/core/src/language.rs",
                "adapter_source_sha256": ROOT / ("crates/models/src/whisper_lid/network.rs" if platform == "macos"
                                                else "crates/models/src/whisper_lid/network_windows.rs"),
            }.items():
                if capture[name] != sha256(digest):
                    raise ValueError("native compiled source identity differs: " + name)
            expected_models = contract["mac_models"] if platform == "macos" else {
                name: pin["sha256"] for name, pin in contract["windows_models"].items()}
            if capture["models"] != expected_models:
                raise ValueError("native checkpoint/graph identity differs")
            expected_runtime = contract["mac_runtime"] if platform == "macos" else {
                name: pin["sha256"] for name, pin in json.loads((ROOT / "crates/models/src/windows-runtime.json").read_text())["files"].items()}
            required = ["libmlx.dylib", "mlx.metallib"] if platform == "macos" else ["onnxruntime.dll", "DirectML.dll"]
            if any(capture["runtime"].get(name) != expected_runtime[name] for name in required):
                raise ValueError("actual loaded native runtime identity differs")
            index = {(row["id"], row["run"]): row for row in capture["rows"]}
            if len(index) != 112:
                raise ValueError("duplicate native capture rows")
            indexed.append(index)
            if [row["name"] for row in capture["controls"]] != ["pre_cancelled_before_graph_entry", "empty", "nan", "infinity", "after_failure"]:
                raise ValueError("native control receipts differ")
            if any("error" not in row["result"] for row in capture["controls"][:4]):
                raise ValueError("native invalid/cancelled request unexpectedly returned a vector")
            control = capture["controls"][-1]["result"]
            vector(control)
            if control["scores_sha256"] != index[("natural-en_us-0", 0)]["scores_sha256"]:
                raise ValueError("post-failure native request changed")
        if captures[1]["mac_reference_sha256"] != sha256(mac_path):
            raise ValueError("Windows did not bind the frozen Mac capture before inference")
        if captures[0]["policy_sequences"] != captures[1]["policy_sequences"]:
            raise ValueError("actual-core policy sequence decisions differ")
        fields = contract["gates"]["core_exact_fields"]
        for case in cases["cases"]:
            rows = [[index[(case["id"], run)] for run in [0, 1]] for index in indexed]
            values = [[vector(row) for row in backend] for backend in rows]
            for backend, scores in zip(rows, values):
                if any(row["codes"] != cases["codes"] for row in backend):
                    raise ValueError("native code order differs")
                if scores[0].tobytes() != scores[1].tobytes() or backend[0]["core"] != backend[1]["core"]:
                    raise ValueError("same-backend repeat changed")
            left, right = rows[0][0], rows[1][0]
            delta = values[1][0].astype(np.float64) - values[0][0].astype(np.float64)
            record = {"id": case["id"], "max_absolute_probability_drift": float(np.max(np.abs(delta))),
                      "l1_probability_drift": float(np.abs(delta).sum()), "total_variation": float(np.abs(delta).sum() / 2),
                      "mac": left["core"], "windows": right["core"], "top3_mac": left["top3"], "top3_windows": right["top3"]}
            errors = []
            if any(left["core"][field] != right["core"][field] for field in fields):
                errors.append("actual-core categorical decision changed")
            human = case.get("human_language")
            if human:
                record["human_language"] = human
                record["mac_top1_correct"] = left["core"]["winner"] == human
                record["windows_top1_correct"] = right["core"]["winner"] == human
                for platform, row in [("mac", left), ("windows", right)]:
                    if row["confident"] and row["core"]["winner"] != human:
                        errors.append(platform + " confidently wrong human label (activation blocked)")
                    if human in contract["natural"]["unsupported"] and row["core"]["language"] is not None:
                        errors.append(platform + " unsupported speech routed to supported language")
            if "legacy_reference" in case:
                wanted_core = next(item["expected_core"] for item in contract["legacy"]["cases"] if item["id"] == case["id"])
                if any(row["core"] != wanted_core for backend in rows for row in backend):
                    errors.append("original independent route contract changed")
                original = np.asarray([case["legacy_reference"]["probabilities"][code] for code in cases["codes"]], np.float32)
                error = float(np.max(np.abs(values[1][0].astype(np.float64) - original.astype(np.float64))))
                record["legacy_strict_1e_4"] = {"max_error": error, "passed": error < 1e-4}
            record["errors"] = errors
            report["cases"].append(record)
            report["errors"].extend(case["id"] + ": " + error for error in errors)
        report["passed"] = not report["errors"]
    except Exception as error:
        report["errors"].append(str(error))
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--retained", type=Path, required=True)
    prep.add_argument("--prior-downloaded-bytes", type=int, default=0,
                      help="bytes spent by retained earlier transport attempts; deducted from the same60MiB corpus budget")
    comparison = commands.add_parser("compare")
    for name in ["cases", "mac", "windows", "output"]:
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.retained, args.prior_downloaded_bytes)
    else:
        result = compare(args.cases, args.mac, args.windows, args.output)
        print(json.dumps({"passed": result["passed"], "errors": result["errors"]}, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
