"""Build-only native license collection: exact reachable dependency text, no models."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_licenses import collect


class NativeLicenseTests(unittest.TestCase):
    def test_native_graph_collects_normal_build_and_declared_licenses_not_dev_only_packages(self):
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
            destination = root / 'output'
            collect(metadata, destination)
            self.assertEqual((destination / 'runtime-1/license').read_text(), 'Original runtime license.\n')
            self.assertEqual((destination / 'binding-1/TERMS').read_text(), 'Original binding license.\n')
            self.assertFalse((destination / 'dev-only-1').exists())
            self.assertEqual([p['name'] for p in json.loads((destination / 'packages.json').read_text())], ['binding', 'runtime'])
            (root / 'binding/TERMS').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing declared license'):
                collect(metadata, destination)

    def test_missing_registry_text_uses_only_exact_retained_package_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'; source.mkdir()
            retained = root / 'retained/runtime-1'; retained.mkdir(parents=True)
            (retained / 'LICENSE').write_text('Original upstream license.\n')
            (retained / 'NOTICE').write_text('Pinned upstream source and hash.\n')
            packages = [dict(id=name, name=name, version='1', source=None if name == 'speakerdesk' else 'registry',
                             manifest_path=str(source / 'Cargo.toml'), license='MIT') for name in ('speakerdesk', 'runtime')]
            metadata = dict(packages=packages, workspace_members=['speakerdesk'], resolve=dict(nodes=[
                dict(id='speakerdesk', deps=[dict(pkg='runtime', dep_kinds=[dict(kind=None)])]), dict(id='runtime', deps=[])]))
            destination = root / 'output'
            collect(metadata, destination, root / 'retained')
            self.assertEqual((destination / 'runtime-1/LICENSE').read_text(), 'Original upstream license.\n')
            self.assertEqual((destination / 'runtime-1/NOTICE').read_text(), 'Pinned upstream source and hash.\n')
            packages[1]['version'] = '2'
            with self.assertRaisesRegex(ValueError, 'Missing license text: runtime 2'):
                collect(metadata, destination, root / 'retained')
