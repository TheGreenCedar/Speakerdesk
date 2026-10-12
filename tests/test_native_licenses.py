"""The generated app notice preserves dependency text and refuses unpinned text."""
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_licenses import collect


class NativeLicenseTests(unittest.TestCase):
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
