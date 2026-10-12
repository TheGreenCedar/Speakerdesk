"""Regression protection for cache compatibility, exact reuse and PR routing."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import ci_cache
from ci_cache import keys, source_changed
from component_cache import restore, run


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = ['Cargo.toml', 'crates/core/src/live.rs', 'scripts/fetch_mlx_runtime.sh', 'scripts/native_licenses.py',
                      'speakerdesk/static/app.js', 'speakerdesk/static/style.css',
                      'speakerdesk/templates/index.html', 'speakerdesk/voice_calibration.json',
                      'speakerdesk/alignment-mlx/manifest.json',
                      'scripts/build_macos.sh', 'scripts/ci_cache.py',
                      'scripts/component_cache.py', 'Cargo.lock',
                      'desktop/src-tauri/Cargo.toml', 'desktop/src-tauri/src/main.rs',
                      'desktop/src-tauri/Entitlements.plist', 'desktop/capture/MeetingCapture.swift',
                      'desktop/capture/Info.plist', 'desktop/package-lock.json']
        self.paths+=['desktop/capture/echo/source-lock.json','desktop/capture/echo/speakerdesk_echo.cc',
                     'scripts/setup_echo.py','scripts/echo_compile_guard.py',
                     '.cargo/config.toml','rust-toolchain.toml','vendor/mlx-sys/build.rs']
        for name in self.paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original')
        self.tools = {'os': '15.7', 'arch': 'arm64', 'rust': '1.99.0',
                      'python': '3.13.15', 'swift': '6.2', 'sdk': '15.5',
                      'target': 'aarch64-apple-darwin', 'deployment': '15.0',
                      'cc': 'clang', 'cmake': '4.2', 'ninja': '1.13',
                      'image': 'fixture image', 'sccache': '0.16.0', 'rustflags': ''}

    def test_changed_input_invalidates_owning_component(self):
        cases = [('crates/core/src/live.rs', 'compiler-key'),
                 ('scripts/fetch_mlx_runtime.sh', 'compiler-key'),
                 ('scripts/native_licenses.py', 'compiler-key'),
                 ('desktop/capture/Info.plist', 'capture-key'),
                 ('desktop/capture/MeetingCapture.swift', 'capture-key'),
                 ('Cargo.lock', 'cargo-key'),
                 ('desktop/package-lock.json', 'npm-key')]
        baseline = keys(self.root, self.paths, self.tools)
        for path, key in cases:
            with self.subTest(path=path):
                (self.root / path).write_text('changed')
                self.assertNotEqual(baseline[key], keys(self.root, self.paths, self.tools)[key])
                (self.root / path).write_text('original')

    def test_rust_edit_preserves_swift_and_compatible_objects(self):
        baseline = keys(self.root, self.paths, self.tools)
        (self.root / 'desktop/src-tauri/src/main.rs').write_text('changed')
        current = keys(self.root, self.paths, self.tools)
        self.assertNotEqual(baseline['compiler-key'], current['compiler-key'])
        for key in ['compiler-prefix', 'capture-key']:
            self.assertEqual(baseline[key], current[key])

    def test_toolchain_or_platform_change_blocks_all_reuse(self):
        baseline = keys(self.root, self.paths, self.tools)
        for name in self.tools.keys() - {'python'}:
            with self.subTest(tool=name):
                current = keys(self.root, self.paths, dict(self.tools, **{name: 'different'}))
                for key in baseline:
                    self.assertNotEqual(baseline[key], current[key])

    def test_manifest_version_preserves_compiler_compatibility(self):
        manifest = self.root / 'desktop/src-tauri/Cargo.toml'
        manifest.write_text('[package]\nname = "speakerdesk"\nversion = "0.3.0"\n')
        baseline = keys(self.root, self.paths, self.tools)
        manifest.write_text('[package]\nname = "speakerdesk"\nversion = "0.4.0"\n')
        current = keys(self.root, self.paths, self.tools)
        self.assertEqual(baseline['compiler-prefix'], current['compiler-prefix'])
        self.assertNotEqual(baseline['compiler-key'], current['compiler-key'])

    def test_embedded_resources_change_exact_key_without_discarding_compatible_objects(self):
        baseline = keys(self.root, self.paths, self.tools)
        for name in ['speakerdesk/static/app.js', 'speakerdesk/static/style.css',
                     'speakerdesk/templates/index.html', 'speakerdesk/voice_calibration.json',
                     'speakerdesk/alignment-mlx/manifest.json']:
            with self.subTest(input=name):
                (self.root / name).write_text('changed')
                try:
                    current = keys(self.root, self.paths, self.tools)
                    self.assertEqual(baseline['compiler-prefix'], current['compiler-prefix'])
                    self.assertNotEqual(baseline['compiler-key'], current['compiler-key'])
                finally:
                    (self.root / name).write_text('original')

    def test_native_contract_blocks_compiler_fallback(self):
        baseline = keys(self.root, self.paths, self.tools)
        for name in ['.cargo/config.toml', 'rust-toolchain.toml', 'vendor/mlx-sys/build.rs',
                     'scripts/fetch_mlx_runtime.sh', 'scripts/native_licenses.py',
                     'scripts/setup_echo.py', 'scripts/echo_compile_guard.py',
                     'desktop/capture/echo/source-lock.json', 'desktop/capture/echo/speakerdesk_echo.cc']:
            with self.subTest(input=name):
                (self.root / name).write_text('changed')
                current = keys(self.root, self.paths, self.tools)
                self.assertNotEqual(baseline['compiler-prefix'], current['compiler-prefix'])
                (self.root / name).write_text('original')

    def test_profile_changes_exact_key_and_non_rust_tools_do_not_split_object_store(self):
        baseline = keys(self.root, self.paths, dict(self.tools, profile='debug-fixtures', node='24'))
        release = keys(self.root, self.paths, dict(self.tools, profile='release', node='24'))
        self.assertNotEqual(baseline['compiler-key'], release['compiler-key'])
        self.assertEqual(baseline['compiler-prefix'], release['compiler-prefix'])
        for name in ('python', 'node'):
            with self.subTest(tool=name):
                tools = dict(self.tools, profile='debug-fixtures', node='24')
                tools[name] = 'different'
                changed = keys(self.root, self.paths, tools)
                self.assertEqual(baseline['compiler-key'], changed['compiler-key'])

    def test_linux_identity_uses_host_target_without_apple_tools(self):
        def command(*args):
            if args[0] in ('sw_vers', 'xcodebuild', 'xcrun'):
                raise AssertionError('Linux must not invoke Apple tools')
            if args == ('rustc', '-Vv'):
                return 'release: 1.99.0\nhost: x86_64-unknown-linux-gnu'
            return 'fixture tool identity'
        with patch.object(ci_cache.platform, 'system', return_value='Linux'), patch.object(ci_cache, 'command', side_effect=command):
            tools = ci_cache.tool_identity('debug-fixtures')
        self.assertEqual(tools['target'], 'x86_64-unknown-linux-gnu')
        self.assertEqual(tools['profile'], 'debug-fixtures')

    def test_macos_deployment_identity_matches_existing_jobs(self):
        def command(*args):
            return 'host: aarch64-apple-darwin' if args == ('rustc', '-Vv') else 'fixture tool identity'
        with patch.object(ci_cache.platform, 'system', return_value='Darwin'), \
                patch.object(ci_cache, 'command', side_effect=command), \
                patch.dict(os.environ, {'MACOSX_DEPLOYMENT_TARGET': '14.0'}):
            native = ci_cache.tool_identity('debug-fixtures-release-default')
            producer = ci_cache.tool_identity('release-default-features-adhoc')
        self.assertEqual(native['deployment'], '14.0')
        self.assertEqual(producer['deployment'], '15.0')

    def test_lock_changes_preserve_download_prefixes_but_change_exact_keys(self):
        baseline = keys(self.root, self.paths, self.tools)
        for path, name in [('Cargo.lock', 'cargo'),
                           ('desktop/package-lock.json', 'npm')]:
            with self.subTest(cache=name):
                (self.root / path).write_text('changed lock')
                current = keys(self.root, self.paths, self.tools)
                self.assertEqual(baseline[name + '-prefix'], current[name + '-prefix'])
                self.assertNotEqual(baseline[name + '-key'], current[name + '-key'])
                (self.root / path).write_text('original')

    def test_symlink_input_is_rejected(self):
        path = self.root / 'crates/core/src/live.rs'
        path.unlink()
        path.symlink_to(self.root / 'desktop/package-lock.json')
        with self.assertRaises(ValueError):
            keys(self.root, self.paths, self.tools)

    def test_echo_component_key_binds_source_guard_and_toolchain_without_rust_edits(self):
        baseline = keys(self.root, self.paths, self.tools)['echo-key']
        for name in ['desktop/capture/echo/source-lock.json',
                     'desktop/capture/echo/speakerdesk_echo.cc',
                     'scripts/setup_echo.py', 'scripts/echo_compile_guard.py']:
            with self.subTest(input=name):
                path = self.root / name
                path.write_text('changed')
                self.assertNotEqual(baseline, keys(self.root, self.paths, self.tools)['echo-key'])
                path.write_text('original')
        (self.root / 'crates/core/src/live.rs').write_text('changed')
        self.assertEqual(baseline, keys(self.root, self.paths, self.tools)['echo-key'])
        self.assertNotEqual(baseline, keys(self.root, self.paths, dict(self.tools, cc='other clang'))['echo-key'])

    def test_dependency_outputs_survive_application_ci_and_docs_edits(self):
        baseline = keys(self.root, self.paths, self.tools)['dependency-key']
        for name in ['crates/core/src/live.rs', 'speakerdesk/static/app.js',
                     'desktop/src-tauri/src/main.rs', 'scripts/build_macos.sh',
                     'scripts/ci_cache.py', '.github/workflows/ci.yml', 'docs/ci-build-reuse.md']:
            with self.subTest(input=name):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('changed')
                self.assertEqual(baseline, keys(self.root, self.paths + [name], self.tools)['dependency-key'])
                path.write_text('original')
        for name in ['vendor/mlx-sys/build.rs', 'scripts/fetch_mlx_runtime.sh',
                     'desktop/capture/echo/source-lock.json', 'scripts/echo_compile_guard.py']:
            with self.subTest(native=name):
                path = self.root / name
                path.write_text('changed')
                self.assertNotEqual(baseline, keys(self.root, self.paths, self.tools)['dependency-key'])
                path.write_text('original')
        for tool in ['cc', 'sdk', 'target', 'rust', 'image', 'deployment']:
            with self.subTest(tool=tool):
                self.assertNotEqual(baseline, keys(self.root, self.paths, dict(self.tools, **{tool: 'changed'}))['dependency-key'])

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
