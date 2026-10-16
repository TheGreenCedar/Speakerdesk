"""Unsigned NSIS payload provenance follows the locked bundler's byte patch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/windows_payload.py"
SPEC = importlib.util.spec_from_file_location("windows_payload", SCRIPT)
PAYLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAYLOAD)


class WindowsPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = self.root / "speakerdesk.exe"
        self.lock = self.root / "package-lock.json"
        self.installed = self.root / "cli-package.json"
        self.lock.write_text(json.dumps({"packages": {"node_modules/@tauri-apps/cli": {"version": "2.12.1"}}}))
        self.installed.write_text(json.dumps({"version": "2.12.1"}))
        # A synthetic unsigned PE32+ header with one runtime bundle marker.
        data = bytearray(512)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 60, 64)
        data[64:68] = b"PE\0\0"
        struct.pack_into("<H", data, 68, 0x8664)
        struct.pack_into("<H", data, 84, 240)
        struct.pack_into("<H", data, 88, 0x20B)
        struct.pack_into("<I", data, 196, 16)
        self.before = bytes(data) + b"unrelated-prefix\0__TAURI_BUNDLE_TYPE_VAR_UNK\0unrelated-tail"
        self.after = bytes(data) + b"unrelated-prefix\0__TAURI_BUNDLE_TYPE_VAR_NSS\0unrelated-tail"
        self.executable.write_bytes(self.before)

    def derive(self):
        return PAYLOAD.expected_nsis_executable(self.executable, self.lock, self.installed)

    def test_input_and_installed_payload_are_distinct_and_only_marker_bytes_change(self):
        receipt = self.derive()
        self.assertEqual(receipt["input_sha256"], hashlib.sha256(self.before).hexdigest())
        self.assertEqual(receipt["expected_payload_sha256"], hashlib.sha256(self.after).hexdigest())
        self.assertNotEqual(receipt["input_sha256"], receipt["expected_payload_sha256"])
        self.assertEqual(receipt["bytes"], len(self.after))
        self.assertEqual(receipt["marker_offset"], 529)
        self.assertFalse(receipt["observed"])
        self.assertEqual(self.executable.read_bytes(), self.before)
        # The regression: hashing Tauri's restored target cannot identify the
        # installed executable, even though its length and all other bytes match.
        self.assertNotEqual(hashlib.sha256(self.executable.read_bytes()).hexdigest(),
                            hashlib.sha256(self.after).hexdigest())

    def test_unrelated_payload_bytes_are_not_normalized_away(self):
        self.executable.write_bytes(self.before + b"changed")
        receipt = self.derive()
        self.assertEqual(receipt["expected_payload_sha256"], hashlib.sha256(self.after + b"changed").hexdigest())

    def test_ambiguous_missing_or_already_patched_marker_fails(self):
        for data in (self.before.replace(b"__TAURI_BUNDLE_TYPE_VAR_UNK", b"missing"),
                     self.before + b"__TAURI_BUNDLE_TYPE_VAR_UNK", self.after,
                     self.before + b"__TAURI_BUNDLE_TYPE_VAR_NSS"):
            with self.subTest(data=data[-40:]):
                self.executable.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "exactly one UNK"):
                    self.derive()

    def test_lock_and_installed_cli_must_both_match_reviewed_version(self):
        for path, value in ((self.lock, {"packages": {"node_modules/@tauri-apps/cli": {"version": "2.12.2"}}}),
                            (self.installed, {"version": "2.12.0"})):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "locked and installed"):
                    self.derive()
                path.write_bytes(original)

    def test_signed_malformed_or_non_x64_pe_is_rejected(self):
        signed = bytearray(self.before)
        struct.pack_into("<II", signed, 232, len(signed), 128)
        non_x64 = bytearray(self.before)
        struct.pack_into("<H", non_x64, 68, 0xAA64)
        for data in (b"", b"not a PE", self.before[:100], bytes(signed), bytes(non_x64)):
            with self.subTest(length=len(data)):
                self.executable.write_bytes(data)
                with self.assertRaises(ValueError):
                    self.derive()

    def test_build_tool_emits_the_expected_payload_receipt(self):
        completed = subprocess.run([sys.executable, str(SCRIPT), "--executable", str(self.executable),
                                    "--package-lock", str(self.lock), "--installed-cli", str(self.installed)],
                                   text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(completed.stdout), self.derive())


if __name__ == "__main__":
    unittest.main()
