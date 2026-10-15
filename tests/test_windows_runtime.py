"""A cached native dependency must be as qualified as a newly fetched one."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "fetch_windows_runtime", Path(__file__).resolve().parents[1] / "scripts/fetch_windows_runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


class WindowsRuntimeTests(unittest.TestCase):
    def test_archive_cache_and_notices_are_checked_before_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "runtime.zip", root / "native"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("native/runtime.dll", b"runtime")
                package.writestr("license.txt", b"license")
                package.writestr("../escape", b"not an allowed artifact")
            files = {name: dict(source=source, sha256=hashlib.sha256(data).hexdigest()) for name, source, data in [
                ("runtime.dll", "native/runtime.dll", b"runtime"), ("LICENSE.txt", "license.txt", b"license")]}
            pin = dict(archive_sha256=RUNTIME.digest(archive), files=files)
            self.assertEqual(RUNTIME.fetch(destination, pin, archive), destination)
            self.assertEqual({p.name for p in destination.iterdir()}, set(files))
            self.assertFalse((root / "escape").exists())
            RUNTIME.fetch(destination, pin, archive)
            (destination / "LICENSE.txt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from its pin"):
                RUNTIME.fetch(destination, pin, archive)
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "archive differs"):
                RUNTIME.fetch(root / "other", pin, archive)
            self.assertFalse((root / "other").exists())


if __name__ == "__main__":
    unittest.main()
