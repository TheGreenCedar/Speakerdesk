"""Receipt/selection regressions only; no acoustic inference or downloads."""
import copy
import hashlib
import importlib.util
import json
import io
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("windows_qualification", Path(__file__).with_name("windows_qualification.py"))
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def scores_row(codes, values, core):
    values = np.asarray(values, "<f4")
    order = np.argsort(-values, kind="stable")
    probability, margin = float(values[order[0]]), float(values[order[0]] - values[order[1]])
    return {"codes": codes, "probabilities": values.tolist(), "probability_bits": values.view("<u4").tolist(),
            "scores_sha256": hashlib.sha256(values.tobytes()).hexdigest(), "core": core,
            "confident": probability >= .90 and margin >= .20,
            "top3": [{"code": codes[i], "probability": float(values[i])} for i in order[:3]]}


class WindowsQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = json.loads(Q.CONTRACT.read_text())
        metadata = json.loads((Q.ROOT / "tests/metadata/whisper-tiny/added_tokens.json").read_text())
        self.codes = [key[2:-2] for key in metadata if Q.re.fullmatch(r"<\|[a-z]{2,3}\|>", key)]
        independent = {row["name"]: row for row in json.loads((Q.ROOT / self.contract["legacy"]["reference"]).read_text())["cases"]}
        self.cases = {"contract_sha256": Q.sha256(Q.CONTRACT), "codes": self.codes, "cases": []}
        self.outputs = {}
        for identifier in Q.expected_ids(self.contract):
            file = self.root / (identifier + ".f32")
            file.write_bytes(np.zeros(1, "<f4").tobytes())
            row = {"id": identifier, "file": file.name, "samples": 1, "sha256": Q.sha256(file), "human_language": None}
            if identifier.startswith("legacy-"):
                expected = next(c for c in self.contract["legacy"]["cases"] if c["id"] == identifier)
                original = independent[expected["reference_name"]]
                row["legacy_reference"] = original
                values = [original["probabilities"][code] for code in self.codes]
                core = expected["expected_core"]
            else:
                human = self.contract["natural"]["configs"][identifier[len("natural-"):].rsplit("-", 1)[0]] if identifier.startswith("natural-") else "en"
                row["human_language"] = human if identifier.startswith("natural-") else None
                values = np.zeros(99, np.float32)
                values[self.codes.index(human)] = .99
                values[self.codes.index("fr" if human == "en" else "en")] = .01
                unsupported = human in ["ru", "hi"]
                core = {"winner": human, "language": None if unsupported else human, "review": unsupported,
                        "reason": "unsupported" if unsupported else "detected", "remembered": None if unsupported else human}
            self.outputs[identifier] = scores_row(self.codes, values, core)
            self.cases["cases"].append(row)
        self.case_path = self.root / "cases.json"
        self.case_path.write_text(json.dumps(self.cases))
        self.mac_path, self.win_path = self.root / "mac.json", self.root / "windows.json"
        self.mac, self.windows = [self.capture(platform) for platform in ["macos", "windows"]]
        self.serial = 0
        self.save()

    def capture(self, platform):
        models = self.contract["mac_models"] if platform == "macos" else {k: v["sha256"] for k, v in self.contract["windows_models"].items()}
        runtime = self.contract["mac_runtime"] if platform == "macos" else {k: v["sha256"] for k, v in json.loads((Q.ROOT / "crates/models/src/windows-runtime.json").read_text())["files"].items()}
        return {"platform": platform, "passed_capture": True, "cases_sha256": Q.sha256(self.case_path),
                "contract_sha256": Q.sha256(Q.CONTRACT), "successful_probes": 113, "models": models, "runtime": runtime,
                "test_source_sha256": Q.sha256(Q.ROOT / "crates/models/tests/whisper_windows_qualification.rs"),
                "policy_source_sha256": Q.sha256(Q.ROOT / "crates/core/src/language.rs"),
                "adapter_source_sha256": Q.sha256(Q.ROOT / ("crates/models/src/whisper_lid/network.rs" if platform == "macos" else "crates/models/src/whisper_lid/network_windows.rs")),
                "policy_sequences": ["synthetic receipt fixture, actual Rust policy owns execution"],
                "isolation": {key: True for key in ["pre_cancelled", "invalid_pcm", "after_failure_bitexact"]},
                "controls": [{"name": name, "result": {"error": "rejected"}} for name in ["pre_cancelled_before_graph_entry", "empty", "nan", "infinity"]]
                    + [{"name": "after_failure", "result": copy.deepcopy(self.outputs["natural-en_us-0"])}],
                "rows": [dict(copy.deepcopy(self.outputs[case["id"]]), id=case["id"], run=run) for run in [0, 1] for case in self.cases["cases"]]}

    def save(self):
        self.mac_path.write_text(json.dumps(self.mac))
        self.windows["mac_reference_sha256"] = Q.sha256(self.mac_path)
        self.win_path.write_text(json.dumps(self.windows))

    def compare(self):
        self.save()
        self.serial += 1
        path = self.root / f"comparison-{self.serial}.json"
        result = Q.compare(self.case_path, self.mac_path, self.win_path, path)
        self.assertEqual(json.loads(path.read_text()), result)
        return result

    def test_original_scalar_failure_is_preserved_separately_from_windows_contract(self):
        for row in self.windows["rows"]:
            if row["id"] == "legacy-partial_frame":
                values = np.asarray(row["probabilities"], np.float32)
                values[self.codes.index("en")] += np.float32(.003)
                values[self.codes.index("ko")] -= np.float32(.003)
                row.update(scores_row(self.codes, values, row["core"]))
        report = self.compare()
        self.assertTrue(report["passed"], report["errors"])
        self.assertFalse(report["activation_recommended"])
        legacy = next(row for row in report["cases"] if row["id"] == "legacy-partial_frame")
        self.assertFalse(legacy["legacy_strict_1e_4"]["passed"])
        self.assertGreater(legacy["legacy_strict_1e_4"]["max_error"], .002)

    def test_changed_actual_core_decision_fails_without_a_scalar_threshold(self):
        for field, changed in [("winner", "fr"), ("language", None), ("review", True), ("reason", "best_effort"), ("remembered", None)]:
            with self.subTest(field=field):
                original = copy.deepcopy(self.windows)
                for row in self.windows["rows"]:
                    if row["id"] == "quiet-en":
                        row["core"][field] = changed
                report = self.compare()
                self.assertFalse(report["passed"])
                self.assertTrue(any("categorical" in error for error in report["errors"]))
                self.windows = original

    def test_runtime_output_repetition_and_frozen_reference_receipts_fail_closed(self):
        changes = [lambda c: c["runtime"].update({"onnxruntime.dll": "0" * 64}),
                   lambda c: c["models"].update({"whisper.onnx": "0" * 64}),
                   lambda c: c.update(passed_capture=False),
                   lambda c: c["rows"][0].update(codes=list(reversed(self.codes))),
                   lambda c: c["rows"][0]["probabilities"].__setitem__(0, 2),
                   lambda c: c["rows"][56].update(scores_sha256="0" * 64)]
        for change in changes:
            with self.subTest(change=change):
                original = copy.deepcopy(self.windows)
                change(self.windows)
                self.assertFalse(self.compare()["passed"])
                self.windows = original
        self.save()
        self.windows["mac_reference_sha256"] = "0" * 64
        self.win_path.write_text(json.dumps(self.windows))
        result = Q.compare(self.case_path, self.mac_path, self.win_path, self.root / "wrong-mac.json")
        self.assertFalse(result["passed"])
        self.assertIn("frozen Mac capture", result["errors"][0])

    def test_valid_but_changed_repeat_fails_and_retains_the_returned_vector(self):
        row = next(row for row in self.windows["rows"] if row["id"] == "quiet-en" and row["run"] == 1)
        values = np.asarray(row["probabilities"], np.float32)
        values[self.codes.index("en")] -= np.float32(.001)
        values[self.codes.index("fr")] += np.float32(.001)
        row.update(scores_row(self.codes, values, row["core"]))
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertIn("same-backend repeat changed", report["errors"])
        saved = json.loads(Path(report["captures"]["windows_file"]).read_text())
        captured = next(r for r in saved["rows"] if r["id"] == "quiet-en" and r["run"] == 1)
        self.assertEqual(captured["probability_bits"], values.view("<u4").tolist())

    def test_changed_embedded_original_reference_cannot_hide_old_scalar_failure(self):
        self.cases["cases"][2]["legacy_reference"]["probabilities"]["en"] += .003
        self.case_path.write_text(json.dumps(self.cases))
        for capture in [self.mac, self.windows]:
            capture["cases_sha256"] = Q.sha256(self.case_path)
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertIn("embedded independent legacy expectation differs", report["errors"][0])

    def test_source_common_confident_label_error_and_unsupported_leak_remain_failures(self):
        for identifier in ["natural-de_de-0", "natural-ru_ru-0"]:
            with self.subTest(identifier=identifier):
                original = copy.deepcopy([self.mac, self.windows])
                for capture in [self.mac, self.windows]:
                    for row in capture["rows"]:
                        if row["id"] == identifier:
                            row.update(copy.deepcopy(self.outputs["natural-en_us-0"]))
                report = self.compare()
                self.assertFalse(report["passed"])
                self.assertTrue(any("mac confidently wrong" in error for error in report["errors"]))
                if "ru_ru" in identifier:
                    self.assertTrue(any("unsupported speech" in error for error in report["errors"]))
                self.mac, self.windows = original

    def test_frozen_stress_pcm_covers_context_equivalence_and_nonzero_crop_origins(self):
        en = np.linspace(-.5, .5, 1024, dtype=np.float32)
        en[:7] = 0
        fr = np.full(900, .25, np.float32)
        rows = Q.stress_cases(en, fr, fr)
        self.assertEqual(len(rows), 20)
        by_id = {identifier: (audio, recipe) for identifier, audio, recipe in rows}
        for size in self.contract["stress"]["short"]["lengths"]:
            audio, recipe = by_id[f"short-{size}"]
            self.assertEqual(recipe["origin"], 7)
            np.testing.assert_array_equal(audio, en[7:7 + size])
        contexts = [by_id[f"context-{size}"][0] for size in [479999, 480000, 480001]]
        np.testing.assert_array_equal(np.pad(contexts[0], (0, 1)), contexts[1])
        np.testing.assert_array_equal(contexts[1], contexts[2][:480000])
        self.assertEqual(contexts[2][480000], 1)
        self.assertTrue(all(audio.dtype == np.float32 and np.isfinite(audio).all() for audio, _ in by_id.values()))

    def test_archive_transport_keeps_member_order_labels_and_stops_before_bulk_download(self):
        import tarfile
        from fleurs_archive import selected_audio
        stream = io.BytesIO()
        contents = {"200.wav": b"complete first wave", "100.wav": b"complete second wave", "300.wav": os.urandom(131072)}
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, content in contents.items():
                member = tarfile.TarInfo("dev/" + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        payload = stream.getvalue()
        labels = "12\t100.wav\tsecond human transcript\tnormalized\tphonic\t16001\tFEMALE\n11\t200.wav\tfirst human transcript\tnormalized\tphonic\t16000\tMALE\n"
        class Transport(Q.Downloads):
            def get(self, url, limit):
                data = labels.encode() if url.endswith("dev.tsv") else json.dumps([{
                    "path": "data/it_it/audio/dev.tar.gz", "size": len(payload),
                    "lfs": {"size": len(payload), "oid": hashlib.sha256(payload).hexdigest()}}]).encode()
                self.bytes += len(data)
                return data
            def stream(self, url):
                self.response = io.BytesIO(payload)
                return Q.BodyStream(self.response, self)
        transport = Transport(65536)
        selected, receipt = selected_audio("it_it", "a" * 40, transport, 1048576, 2097152)
        self.assertEqual([row["filename"] for row in selected], ["200.wav", "100.wav"])
        self.assertEqual([row["id"] for row in selected], [11, 12])
        self.assertEqual([row["raw_transcription"] for row in selected], ["first human transcript", "second human transcript"])
        self.assertEqual([row["data"] for row in selected], list(contents.values())[:2])
        self.assertTrue(transport.response.closed)
        self.assertFalse(receipt["whole_archive_downloaded"])
        self.assertLess(receipt["archive_prefix_bytes_read"], len(payload) / 4)
        exhausted = Q.Downloads(1)
        response = io.BytesIO(b"must not be read")
        with self.assertRaisesRegex(ValueError, "cumulative corpus budget"):
            Q.BodyStream(response, exhausted).read(2)
        self.assertEqual(response.tell(), 0)


if __name__ == "__main__":
    unittest.main()
