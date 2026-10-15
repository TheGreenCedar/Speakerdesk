"""Portable boundary tests; native certificate trust is checked by SignTool on Windows."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('windows_signing', ROOT / 'scripts/windows_signing.py')
SIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIGN)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


class WindowsSigningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stage, self.target, self.tmp, self.receipts = [self.root / p for p in ('stage', 'release', 'temp', 'receipts')]
        self.version = '0.7.5'
        self.runtime = {'onnxruntime.dll': {'sha256': sha(b'microsoft-runtime')}, 'LICENSE.txt': {'sha256': sha(b'license')}}
        put(self.root / 'crates/models/src/windows-runtime.json', {'files': self.runtime})
        for name in self.runtime:
            put(self.stage / 'runtime' / name, b'microsoft-runtime' if name.endswith('.dll') else b'license')
        self.contract = copy.deepcopy(SIGN.CONTRACT)
        self.contract['nsis_plugins'] = {'System.dll': sha(b'pinned-plugin')}
        self.addCleanup(patch.stopall)
        patch.object(SIGN, 'ROOT', self.root).start()
        patch.object(SIGN, 'CONTRACT', self.contract).start()
        data = bytearray(128)
        data[:2] = b'MZ'
        struct.pack_into('<I', data, 60, 64)
        data[64:68] = b'PE\0\0'
        self.pe = bytes(data)
        self.app = self.pe + b'__TAURI_BUNDLE_TYPE_VAR_NSS'

    def classify(self, path):
        return SIGN.classify(path, self.stage, self.target, self.tmp, self.version)

    def test_metadata_cannot_select_other_credentials_or_endpoints(self):
        value = SIGN.metadata('https://eus.codesigning.azure.net/', 'shared-account', 'public-release', 'speakerdesk-abc')
        SIGN.validate_metadata(value)
        for change in ({'ExcludeCredentials': []}, {'ClientSecret': 'not-allowed'},
                       {'Endpoint': 'https://eus.codesigning.azure.net.evil.test'}, {'CertificateProfileName': '../other'}):
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                SIGN.validate_metadata(dict(value, **change))

    def test_exact_paths_post_patch_and_preserved_bytes_are_required(self):
        app = put(self.target / 'speakerdesk.exe', self.app)
        self.assertEqual(self.classify(app)['role'], 'app')
        runtime = self.stage / 'runtime/onnxruntime.dll'
        self.assertEqual(self.classify(runtime)['disposition'], 'microsoft_preserved')
        plugin = put(self.target / 'nsis/x64/Plugins/x86-unicode/System.dll', b'pinned-plugin')
        self.assertEqual(self.classify(plugin)['disposition'], 'nsis_preserved')
        bad_inputs = [put(self.root / 'elsewhere/speakerdesk.exe', self.app),
                      put(self.stage / 'unknown.dll', self.pe),
                      put(self.tmp / 'nested/random.tmp', self.pe)]
        for path in bad_inputs:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.classify(path)
        for path, changed in ((app, self.app.replace(b'NSS', b'UNK')), (runtime, b'changed'), (plugin, b'changed')):
            put(path, changed)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.classify(path)
        generated = put(self.tmp / 'nsA12.tmp', self.pe)
        self.assertEqual(self.classify(generated)['role'], 'uninstaller')
        put(generated, b'not-pe')
        with self.assertRaises(ValueError):
            self.classify(generated)

    def test_reparse_or_symlink_ancestors_are_rejected(self):
        actual = put(self.root / 'actual/speakerdesk-capture.exe', self.pe)
        link = self.root / 'linked'
        try:
            link.symlink_to(actual.parent, target_is_directory=True)
        except OSError:
            self.skipTest('Creating links requires privileges on this host')
        with self.assertRaisesRegex(ValueError, 'links/reparse'):
            SIGN.regular(link / actual.name)

    def test_tools_are_verified_against_pinned_archive_not_local_receipt(self):
        tools = self.root / 'tools'
        archive = tools / 'Signer.nupkg'
        tools.mkdir()
        contents = {'bin/signtool.exe': b'signtool', 'bin/dlib.dll': b'dlib', 'support.json': b'{}'}
        with zipfile.ZipFile(archive, 'w') as output:
            for name, value in contents.items():
                output.writestr(name, value)
                put(tools / 'Signer' / name, value)
        self.contract.update(client_packages=[{'id': 'Signer', 'sha256': sha(archive.read_bytes())}],
                             signtool='Signer/bin/signtool.exe', dlib='Signer/bin/dlib.dll')
        SIGN.verify_tools(tools)
        for path in (tools / 'Signer/bin/dlib.dll', tools / 'Signer/injected.dll', archive):
            original = path.read_bytes() if path.exists() else None
            put(path, b'changed')
            with self.subTest(path=path), self.assertRaises(ValueError):
                SIGN.verify_tools(tools)
            path.unlink() if original is None else path.write_bytes(original)

    def fixture_inventory(self):
        for role, name in {'capture': 'speakerdesk-capture.exe', 'echo': 'speakerdesk_echo.dll',
                           'app': 'speakerdesk.exe', 'uninstaller': 'uninstall.exe',
                           'installer': 'Speakerdesk_0.7.5_x64-setup.exe'}.items():
            data = self.app if role == 'app' else self.pe + role.encode()
            put(self.receipts / 'payloads' / name, data)
            put(self.receipts / (role + '.json'), dict(role=role, disposition='owned_signed',
                input_sha256=sha(self.app) if role == 'app' else sha(data),
                sha256=sha(data), authenticode_status='Valid', timestamped=True, publisher_subject='CN=Albert Najjar'))
            if role in ('capture', 'echo'):
                put(self.stage / name, data)
            elif role == 'installer':
                put(self.target / 'bundle/nsis' / name, data)
        # Tauri restores the loose app; it is deliberately not the signed payload oracle.
        put(self.target / 'speakerdesk.exe', self.app.replace(b'NSS', b'UNK'))
        put(self.stage / 'speakerdesk-capture.exe.capability.json',
            {'helper_sha256': SIGN.digest(self.stage / 'speakerdesk-capture.exe')})
        put(self.stage / 'THIRD_PARTY_NOTICES.txt', b'notices')
        bootstrap = put(self.root / 'tauri/MicrosoftEdgeWebview2Setup.exe', b'microsoft-webview')
        crt = put(self.root / 'VC/14.50/x64/Microsoft.VC143.CRT/vcruntime140.dll', b'microsoft-vc')
        lines = ['!define INSTALLWEBVIEW2MODE "embedBootstrapper"', '!define WEBVIEW2INSTALLERPATH ""',
                 f'!define WEBVIEW2BOOTSTRAPPERPATH "{bootstrap}"',
                 f'!define MAINBINARYSRCPATH "{self.target / "speakerdesk.exe"}"',
                 'File "${MAINBINARYSRCPATH}"', 'File "/oname=$TEMP\\MicrosoftEdgeWebview2Setup.exe" "${WEBVIEW2BOOTSTRAPPERPATH}"',
                 'File "/oname=$TEMP\\MicrosoftEdgeWebView2RuntimeInstaller.exe" "${WEBVIEW2INSTALLERPATH}"']
        for name in ('speakerdesk-capture.exe', 'speakerdesk-capture.exe.capability.json', 'speakerdesk_echo.dll', 'THIRD_PARTY_NOTICES.txt',
                     'runtime/onnxruntime.dll', 'runtime/LICENSE.txt'):
            lines.append(f'File /a "/oname={name}" "{self.stage / name}"')
        lines.append(f'File /a "/oname={crt.name}" "{crt}"')
        self.script = put(self.target / 'nsis/x64/installer.nsi', '\n'.join(lines).encode())
        package = SIGN.package_inputs(self.script, self.stage, self.target)
        for entry in package:
            if entry['kind'] != 'hook_or_pinned':
                entry.update(authenticode_status='Valid', publisher_subject='CN=Microsoft Corporation', verified=True)
        put(self.receipts / 'package.json', {'script_sha256': SIGN.digest(self.script), 'post_build_inputs': package,
            'app_packaged_sha256': sha(self.app)})

    def inventory(self):
        return SIGN.observed_payloads(self.receipts, self.stage, self.target, self.tmp, self.version)

    def test_actual_signed_snapshots_bind_inventory_and_capability_not_loose_app(self):
        self.fixture_inventory()
        result = self.inventory()
        self.assertEqual(result['speakerdesk.exe'], sha(self.app))
        self.assertNotEqual(result['speakerdesk.exe'], SIGN.digest(self.target / 'speakerdesk.exe'))
        self.assertEqual(result['uninstall.exe'], SIGN.digest(self.receipts / 'payloads/uninstall.exe'))
        self.assertEqual(result['vcruntime140.dll'], sha(b'microsoft-vc'))
        mutations = [(self.receipts / 'app.json', {}),
            (self.receipts / 'payloads/uninstall.exe', b'changed'),
            (self.stage / 'speakerdesk-capture.exe', b'changed'),
            (self.stage / 'speakerdesk-capture.exe.capability.json', {'helper_sha256': '0' * 64}),
            (self.target / 'speakerdesk.exe', self.app.replace(b'NSS', b'UNK') + b'unrelated mutation'),
            (self.receipts / 'package.json', {'script_sha256': SIGN.digest(self.script), 'post_build_inputs': []})]
        for path, value in mutations:
            original = path.read_bytes()
            put(path, value)
            with self.subTest(path=path), self.assertRaises((ValueError, KeyError)):
                self.inventory()
            path.write_bytes(original)

    def test_package_rejects_new_pe_classes_sources_destinations_and_incomplete_inputs(self):
        self.fixture_inventory()
        original = self.script.read_text()
        unknown = put(self.root / 'unknown.dll', b'unknown')
        for changed in (original + f'\nFile "{unknown}"',
                        original.replace('/oname=speakerdesk_echo.dll', '/oname=other.dll'),
                        original.replace(str(self.stage / 'speakerdesk_echo.dll'), str(unknown)),
                        original.replace('File "${MAINBINARYSRCPATH}"', ''),
                        original + '\nFile "${UNKNOWN}"'):
            self.script.write_text(changed)
            with self.subTest(changed=changed[-80:]), self.assertRaises(ValueError):
                SIGN.package_inputs(self.script, self.stage, self.target)
        self.script.write_text(original)


if __name__ == '__main__':
    unittest.main()
