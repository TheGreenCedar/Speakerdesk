"""Pinned model delivery and immutable source admission, without network or compilers."""
import hashlib
import io
import json
import runpy
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_localvqe as learned
from component_cache import run


class LocalVQESourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.lock = self.root / 'lock.json'
        self.pins = {}
        for name, value in [('ROOT', self.root), ('CACHE', self.cache), ('LOCK', self.lock)]:
            p = patch.object(learned, name, value)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(learned.urllib.request, 'urlopen', side_effect=AssertionError('No network in regression'))
        p.start(); self.addCleanup(p.stop)

    def pin(self, name, data):
        (self.cache / name).write_bytes(data)
        self.pins[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), url='https://example.invalid/' + name)
        self.lock.write_text(json.dumps(dict(artifacts=self.pins)))

    def archive(self, name, member='LICENSE', data=b'Original public source license'):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            entry = tarfile.TarInfo('pinned-root/' + member)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        self.pin(name, stream.getvalue())

    def test_required_model_is_prepared_even_when_component_cache_restores_only_library(self):
        data = b'Independent model data pin'
        self.pin(learned.MODEL, data)
        library = self.root / 'echo.dylib'
        library.write_bytes(b'cached library')
        component = self.root / 'component'
        command = [sys.executable, '-c', 'raise SystemExit(42)']
        # Publish one exact component, then prove its hit does not run a compiler.
        with patch.dict('os.environ', {'APPLE_SIGNING_IDENTITY': '-'}):
            run(component, 'key', library, [sys.executable, '-c', 'pass'])
            library.unlink()
            run(component, 'key', library, command)
        with patch.object(sys, 'argv', ['setup_echo.py', '--prepare-model']):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/setup_echo.py'), run_name='__main__')
        self.assertEqual(library.read_bytes(), b'cached library')
        self.assertEqual((self.root / 'desktop/capture/echo' / learned.MODEL).read_bytes(), data)
        (self.cache / learned.MODEL).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'differs from its pin'):
            learned.model()

    def test_source_cache_rejects_modified_extra_and_traversing_files(self):
        self.archive('LocalVQE.tar.gz')
        self.archive('ggml.tar.gz')
        source = learned.sources()
        self.assertEqual((source / 'LICENSE').read_bytes(), b'Original public source license')
        (source / 'LICENSE').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Modified pinned'):
            learned.sources()
        (source / 'LICENSE').write_bytes(b'Original public source license')
        (source / 'extra.cpp').write_bytes(b'unreviewed')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            learned.sources()
        (source / 'extra.cpp').unlink()
        self.archive('LocalVQE.tar.gz', '../escape')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            learned.sources()
        self.assertFalse((self.cache / 'escape').exists())

    def test_prepared_cache_is_reconstructed_and_verified_without_changing_good_mtimes(self):
        self.archive('LocalVQE.tar.gz', 'ggml/part.cpp', b'old\n')
        self.archive('ggml.tar.gz')
        self.pin(learned.MODEL,b'model')
        patch_file=self.root / 'desktop/capture/echo/localvqe-portability.patch'
        patch_file.parent.mkdir(parents=True)
        patch_file.write_text('--- a/part.cpp\n+++ b/part.cpp\n@@ -1 +1 @@\n-old\n+new\n')
        prepared=learned.prepare()
        part=prepared / 'part.cpp'
        before=part.stat().st_mtime_ns
        self.assertEqual(learned.prepare(),prepared)
        self.assertEqual(part.stat().st_mtime_ns,before)
        self.assertEqual((self.cache/'source/ggml/part.cpp').read_bytes(),b'old\n')
        for mutation in ('changed','missing','extra','symlink'):
            with self.subTest(mutation=mutation):
                if mutation=='changed':part.write_bytes(b'changed')
                elif mutation=='missing':part.unlink()
                elif mutation=='extra':(prepared/'unreviewed.cpp').write_bytes(b'extra')
                else:
                    part.unlink();part.symlink_to(self.cache/'source/ggml/part.cpp')
                with self.assertRaisesRegex(ValueError,'prepared LocalVQE'):
                    learned.prepare()
                if part.is_symlink():part.unlink()
                part.write_bytes(b'new\n')
                (prepared/'unreviewed.cpp').unlink(missing_ok=True)
