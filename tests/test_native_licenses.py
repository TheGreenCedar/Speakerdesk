"""The generated app notice preserves dependency text and refuses unpinned text."""
import hashlib
import base64
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_licenses import collect
import native_licenses
import setup_echo
from component_cache import run as run_component


class NativeLicenseTests(unittest.TestCase):
    def test_echo_cache_hit_without_sources_still_generates_native_notices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / 'scripts/native_licenses.py'
            script.parent.mkdir()
            script.write_bytes((Path(__file__).resolve().parents[1] / 'scripts/native_licenses.py').read_bytes())
            for relative in ['LICENSE', 'packaging/THIRD_PARTY_NOTICES.md',
                             '.cache/native/mlx-0.32.2/LICENSE', 'vendor/mlx-sys/src/mlx-c/LICENSE']:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Independent native license text')
            (root / 'packaging/native-license-sources.json').write_text('{"sources": {}, "packages": {}}')
            source = root / '.cache/echo-source/src'
            notices = {name: ('/* Original ' + name + ' notice */').encode() for name in [
                'LICENSE', 'PATENTS', 'AUTHORS', 'third_party/abseil-cpp/LICENSE',
                'common_audio/third_party/ooura/fft_size_128/ooura_fft.cc']}
            lock = root / 'lock.json'
            lock.write_text(json.dumps({'archives': {}, 'files': {name: {
                'url': 'https://example.org/' + name, 'sha256': hashlib.sha256(data).hexdigest()
            } for name, data in notices.items()}}))
            output = root / 'echo.dylib'
            with patch.dict(os.environ, {'APPLE_SIGNING_IDENTITY': '-'}):
                run_component(root / 'echo-cache', 'exact', output,
                              [sys.executable, '-c', 'from pathlib import Path; Path(' + repr(str(output)) + ').write_bytes(b"library")'])
                output.unlink()
                run_component(root / 'echo-cache', 'exact', output, [sys.executable, '-c', 'raise SystemExit(42)'])
            self.assertFalse(source.exists())
            metadata = {'packages': [{'id': 'app', 'name': 'speakerdesk', 'source': None}],
                        'workspace_members': ['app'], 'resolve': {'nodes': [{'id': 'app', 'deps': []}]}}
            with patch.object(setup_echo, 'SOURCE', source), patch.object(setup_echo, 'LOCK', lock), \
                    patch.object(setup_echo.urllib.request, 'build_opener') as factory, \
                    patch.object(setup_echo, 'build', side_effect=AssertionError('Echo cache hit must not compile')), \
                    patch('subprocess.check_output', return_value=json.dumps(metadata).encode()), \
                    patch.object(sys, 'argv', [str(script)]):
                factory.return_value.open.side_effect = lambda url, **kwargs: io.BytesIO(base64.b64encode(notices[url.removeprefix('https://example.org/')]))
                runpy.run_path(str(script), run_name='__main__')
            text = (root / 'target/native/THIRD_PARTY_NOTICES.txt').read_text()
            for data in notices.values():
                self.assertIn(data.decode(), text)
            self.assertTrue((source.parent / 'VERIFIED-LOCK.json').is_file())

    def test_notice_contains_normal_build_and_declared_licenses_not_dev_only_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = []
            for name in ('speakerdesk', 'runtime', 'binding', 'dev-only'):
                source = root / name
                source.mkdir()
                filename = 'TERMS' if name == 'binding' else 'license'
                (source / filename).write_text(f'Original {name} license.\n')
                packages.append(dict(id=name, name=name, version='1', source=None if name == 'speakerdesk' else 'registry',
                                     manifest_path=str(source / 'Cargo.toml'), license='MIT',
                                     license_file='TERMS' if name == 'binding' else None))
            metadata = dict(packages=packages, workspace_members=['speakerdesk'], resolve=dict(nodes=[
                dict(id='speakerdesk', deps=[dict(pkg='runtime', dep_kinds=[dict(kind=None)]), dict(pkg='dev-only', dep_kinds=[dict(kind='dev')])]),
                dict(id='runtime', deps=[dict(pkg='binding', dep_kinds=[dict(kind='build')])]),
                dict(id='binding', deps=[]), dict(id='dev-only', deps=[])]))
            destination = root / 'THIRD_PARTY_NOTICES.txt'
            collect(metadata, destination)
            text = destination.read_text()
            self.assertIn('runtime 1 (MIT)', text)
            self.assertIn('Original runtime license.', text)
            self.assertIn('Original binding license.', text)
            self.assertNotIn('dev-only', text)
            self.assertNotIn('Original speakerdesk license.', text)
            (root / 'binding/TERMS').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing declared license'):
                collect(metadata, destination)

    def test_pinned_fallback_is_exact_verified_cached_and_preserves_extra_notices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'; source.mkdir()
            packages = [dict(id=name, name=name, version='1', source=None if name == 'speakerdesk' else 'registry',
                             manifest_path=str(source / 'Cargo.toml'), license='MIT') for name in ('speakerdesk', 'runtime')]
            metadata = dict(packages=packages, workspace_members=['speakerdesk'], resolve=dict(nodes=[
                dict(id='speakerdesk', deps=[dict(pkg='runtime', dep_kinds=[dict(kind=None)])]), dict(id='runtime', deps=[])]))
            data = b'Original upstream license.\n'
            digest = hashlib.sha256(data).hexdigest()
            pins = {'runtime-1': {'url': 'https://example.com/pinned/LICENSE', 'sha256': digest}}
            destination = root / 'THIRD_PARTY_NOTICES.txt'
            with patch('native_licenses.urlopen', return_value=io.BytesIO(data)) as request:
                collect(metadata, destination, pins, {'Shipped icons': 'Original icon license.'})
                collect(metadata, destination, pins, {'Shipped icons': 'Original icon license.'})
            self.assertEqual(request.call_count, 1)
            self.assertIn('Original upstream license.', destination.read_text())
            self.assertIn('Original icon license.', destination.read_text())
            packages[1]['version'] = '2'
            with self.assertRaisesRegex(ValueError, 'Missing license text: runtime 2'):
                collect(metadata, destination, pins)
            packages[1]['version'] = '1'
            (root / 'license-sources' / digest).write_text('Tampered text.')
            with self.assertRaisesRegex(ValueError, 'License hash mismatch'):
                collect(metadata, destination, pins)

    def test_components_keep_feature_specific_dependency_edges_without_duplicate_notices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = {}
            for name in ('speakerdesk', 'speakerdesk-capture-windows', 'shared', 'capture-only', 'dev-only'):
                source = root / name
                source.mkdir()
                (source / 'LICENSE').write_text(f'Original {name} notice.\n')
                packages[name] = dict(id=name, name=name, version='1',
                                      source=None if name.startswith('speakerdesk') else 'registry',
                                      manifest_path=str(source / 'Cargo.toml'), license='MIT')
            def component(name, shared_dependencies):
                return dict(packages=list(packages.values()), workspace_members=[name], resolve=dict(nodes=[
                    dict(id=name, deps=[dict(pkg='shared', dep_kinds=[dict(kind=None)])]),
                    dict(id='shared', deps=[dict(pkg=dependency, dep_kinds=[dict(kind=kind)])
                                           for dependency, kind in shared_dependencies]),
                    dict(id='capture-only', deps=[]), dict(id='dev-only', deps=[])]))
            desktop = component('speakerdesk', [])
            capture = component('speakerdesk-capture-windows', [('capture-only', None), ('dev-only', 'dev')])
            destination = root / 'notice.txt'
            collect(desktop, destination, components=[(capture, 'speakerdesk-capture-windows')])
            text = destination.read_text()
            self.assertEqual(text.count('Original shared notice.'), 1)
            self.assertIn('Original capture-only notice.', text)
            self.assertNotIn('Original dev-only notice.', text)

    def test_windows_entrypoint_uses_pinned_runtime_notices_and_both_locked_component_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in ['LICENSE', 'packaging/THIRD_PARTY_NOTICES.md',
                             '.cache/echo-source/src/LICENSE', '.cache/echo-source/src/PATENTS',
                             '.cache/echo-source/src/AUTHORS',
                             '.cache/echo-source/src/third_party/abseil-cpp/LICENSE']:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Original Windows build notice.\n', encoding='utf-8')
            ooura = root / '.cache/echo-source/src/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc'
            ooura.parent.mkdir(parents=True)
            ooura.write_text('/* Original FFT notice */\ncode', encoding='utf-8')
            (root / 'packaging/native-license-sources.json').write_text('{"sources": {}, "packages": {}}')
            runtime = root / 'runtime'
            runtime.mkdir()
            runtime_files = {}
            for name in ('onnxruntime.dll', 'DirectML.dll', 'Microsoft.Windows.AI.MachineLearning.dll',
                         'LICENSE.txt', 'ThirdPartyNotices.txt'):
                data = f'Pinned Microsoft text for {name}.'.encode()
                (runtime / name).write_bytes(data)
                runtime_files[name] = dict(sha256=hashlib.sha256(data).hexdigest())
            pin = root / 'crates/models/src/windows-runtime.json'
            pin.parent.mkdir(parents=True)
            pin.write_text(json.dumps(dict(version='2.4.89', files=runtime_files)))
            def metadata(name):
                return dict(packages=[dict(id=name, name=name, source=None)],
                            workspace_members=[name], resolve=dict(nodes=[dict(id=name, deps=[])]))
            manifest = root / 'capture/Cargo.toml'
            destination = root / 'staged/THIRD_PARTY_NOTICES.txt'
            arguments = ['--platform', 'windows', '--capture-manifest', str(manifest),
                         '--runtime-directory', str(runtime), '--destination', str(destination)]
            with patch.object(native_licenses, 'ROOT', root), \
                    patch.object(setup_echo, 'sources'), \
                    patch('subprocess.check_output', side_effect=[
                        json.dumps(metadata('speakerdesk')).encode(),
                        json.dumps(metadata('speakerdesk-capture-windows')).encode()]) as cargo:
                native_licenses.main(arguments)
            for command in cargo.call_args_list:
                self.assertIn('--locked', command.args[0])
                self.assertIn('x86_64-pc-windows-msvc', command.args[0])
            self.assertIn(str(manifest), cargo.call_args_list[1].args[0])
            text = destination.read_text(encoding='utf-8')
            self.assertIn('Pinned Microsoft text for LICENSE.txt.', text)
            self.assertIn('Pinned Microsoft text for ThirdPartyNotices.txt.', text)
            self.assertNotIn('Apple MLX 0.32.2', text)
            (runtime / 'LICENSE.txt').write_text('Tampered notice')
            with patch.object(native_licenses, 'ROOT', root), patch.object(setup_echo, 'sources'), \
                    patch('subprocess.check_output', return_value=json.dumps(metadata('speakerdesk')).encode()):
                with self.assertRaisesRegex(ValueError, 'differs from its pin'):
                    native_licenses.main(arguments)
            self.assertEqual(destination.read_text(encoding='utf-8'), text)
