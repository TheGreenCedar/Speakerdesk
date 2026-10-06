"""Regression protection for cache compatibility, exact reuse and PR routing."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ci_cache import keys, source_changed
from component_cache import restore, run


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = ['requirements-packaging.lock.txt', 'requirements-voice.lock.txt',
                      'speakerdesk/app.py', 'speakerdesk/static/app.js',
                      'packaging/runtime.spec', 'packaging/overrides/models.py',
                      'scripts/setup.sh', 'scripts/build_macos.sh', 'scripts/ci_cache.py',
                      'scripts/component_cache.py', 'desktop/src-tauri/Cargo.lock',
                      'desktop/src-tauri/Cargo.toml', 'desktop/src-tauri/src/main.rs',
                      'desktop/src-tauri/Entitlements.plist', 'desktop/capture/MeetingCapture.swift',
                      'desktop/capture/Info.plist', 'desktop/package-lock.json']
        for name in self.paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original')
        self.tools = {'os': '15.7', 'arch': 'arm64', 'rust': '1.97.1',
                      'python': '3.13.15', 'swift': '6.2', 'sdk': '15.5', 'uv': '0.12.10'}

    def test_changed_input_invalidates_owning_component(self):
        cases = [('speakerdesk/app.py', 'runtime-key'),
                 ('speakerdesk/static/app.js', 'runtime-key'),
                 ('packaging/runtime.spec', 'runtime-key'),
                 ('packaging/overrides/models.py', 'runtime-key'),
                 ('requirements-voice.lock.txt', 'runtime-key'),
                 ('desktop/capture/Info.plist', 'capture-key'),
                 ('desktop/capture/MeetingCapture.swift', 'capture-key'),
                 ('desktop/src-tauri/Cargo.lock', 'cargo-key'),
                 ('desktop/package-lock.json', 'npm-key')]
        baseline = keys(self.root, self.paths, self.tools)
        for path, key in cases:
            with self.subTest(path=path):
                (self.root / path).write_text('changed')
                self.assertNotEqual(baseline[key], keys(self.root, self.paths, self.tools)[key])
                (self.root / path).write_text('original')

    def test_rust_edit_preserves_python_and_swift_and_compatible_objects(self):
        baseline = keys(self.root, self.paths, self.tools)
        (self.root / 'desktop/src-tauri/src/main.rs').write_text('changed')
        current = keys(self.root, self.paths, self.tools)
        self.assertNotEqual(baseline['compiler-key'], current['compiler-key'])
        for key in ['compiler-prefix', 'runtime-key', 'capture-key', 'uv-key']:
            self.assertEqual(baseline[key], current[key])

    def test_toolchain_or_platform_change_blocks_all_reuse(self):
        baseline = keys(self.root, self.paths, self.tools)
        for name in self.tools:
            with self.subTest(tool=name):
                current = keys(self.root, self.paths, dict(self.tools, **{name: 'different'}))
                for key in baseline:
                    self.assertNotEqual(baseline[key], current[key])

    def test_symlink_input_is_rejected(self):
        path = self.root / 'speakerdesk/app.py'
        path.unlink()
        path.symlink_to(self.root / 'desktop/package-lock.json')
        with self.assertRaises(ValueError):
            keys(self.root, self.paths, self.tools)

    def test_docs_filter_preserves_all_unknown_source_changes(self):
        self.assertFalse(source_changed(['README.md', 'docs/guide.md']))
        for path in ['.github/workflows/ci.yml', 'requirements-ci.lock.txt',
                     'packaging/licenses/NOTICE', 'new-subsystem/file', 'docs.py']:
            self.assertTrue(source_changed(['docs/guide.md', path]))


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / 'cache'
        self.output = self.root / 'binary'
        self.command = [sys.executable, '-c',
                        'from pathlib import Path; Path(' + repr(str(self.output)) + ').write_bytes(b"built")']

    def test_exact_hit_skips_build_and_restores_verified_bytes(self):
        with patch.dict(os.environ, {'APPLE_SIGNING_IDENTITY': '-'}):
            run(self.cache, 'exact-key', self.output, self.command)
            self.output.unlink()
            # A cache hit must avoid this failing child process.
            run(self.cache, 'exact-key', self.output, [sys.executable, '-c', 'raise SystemExit(42)'])
        self.assertEqual(self.output.read_bytes(), b'built')

    def test_tampered_or_wrong_identity_output_is_rejected(self):
        with patch.dict(os.environ, {'APPLE_SIGNING_IDENTITY': '-'}):
            run(self.cache, 'exact-key', self.output, self.command)
        with self.assertRaisesRegex(ValueError, 'identity or digest'):
            restore(self.cache, 'wrong-key', self.output)
        (self.cache / 'payload').write_bytes(b'tampered')
        self.output.unlink()
        with self.assertRaisesRegex(ValueError, 'identity or digest'):
            restore(self.cache, 'exact-key', self.output)
        self.assertFalse(self.output.exists())

    def test_developer_id_build_never_reads_or_writes_component_cache(self):
        with patch.dict(os.environ, {'APPLE_SIGNING_IDENTITY': '-'}):
            run(self.cache, 'exact-key', self.output, self.command)
        (self.cache / 'payload').write_bytes(b'tampered')
        with patch.dict(os.environ, {'APPLE_SIGNING_IDENTITY': 'Developer ID'}):
            run(self.cache, 'exact-key', self.output, self.command)
        self.assertEqual(self.output.read_bytes(), b'built')
        self.assertEqual((self.cache / 'payload').read_bytes(), b'tampered')


if __name__ == '__main__':
    unittest.main()
