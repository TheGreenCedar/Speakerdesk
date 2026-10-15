"""Pinned source fetching tolerates transient outages without weakening integrity."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_echo


class EchoDownloadTests(unittest.TestCase):
    def test_windows_git_tree_verification_preserves_archive_modes_and_rejects_modified_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / 'expected'
            expected.mkdir()
            (expected / 'build.sh').write_bytes(b'build source')
            (expected / 'build.sh').chmod(0o755)
            (expected / 'LICENSE').write_bytes(b'license')
            tree = setup_echo.tree_hash(expected).hex()
            payload = BytesIO()
            with tarfile.open(fileobj=payload, mode='w:gz') as archive:
                for path in expected.iterdir():
                    archive.add(path, arcname=path.name)
            source, lock = root / 'src', root / 'lock.json'
            lock.write_text(json.dumps({'archives': {'component': {'url': 'https://example.org/source',
                                                                   'git_tree': tree}}, 'files': {}}))
            with (patch.object(setup_echo, 'SOURCE', source), patch.object(setup_echo, 'LOCK', lock),
                  patch.object(setup_echo.platform, 'system', return_value='Windows'),
                  patch.object(setup_echo.urllib.request, 'build_opener') as factory):
                factory.return_value.open.return_value = BytesIO(payload.getvalue())
                setup_echo.sources()
                # NTFS cannot expose the original executable bit through stat.
                (source / 'component/build.sh').chmod(0o644)
                setup_echo.sources()
                factory.return_value.open.assert_called_once()
                (source / 'component/build.sh').write_bytes(b'modified source')
                with self.assertRaisesRegex(RuntimeError, 'Modified pinned echo source'):
                    setup_echo.sources()

    def test_transient_retries_are_bounded_and_do_not_retry_invalid_sources(self):
        url = 'https://example.org/pinned/LICENSE?format=TEXT'
        expected = b'Independent fixture license'
        for failures, data, success, attempts in [
            ([503], expected, True, 2),
            ([503, 503, 503], expected, False, 3),
            ([404], expected, False, 1),
            ([], b'wrong source bytes', False, 1),
        ]:
            with self.subTest(failures=failures, data=data), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'src'
                lock = root / 'source-lock.json'
                lock.write_text(json.dumps({'archives': {}, 'files': {'LICENSE': {
                    'url': url, 'sha256': hashlib.sha256(expected).hexdigest()}}}))
                responses = [HTTPError(url, code, 'upstream failure', {}, None) for code in failures]
                responses.append(BytesIO(base64.b64encode(data)))
                with (patch.object(setup_echo, 'SOURCE', source),
                      patch.object(setup_echo, 'LOCK', lock),
                      patch.object(setup_echo.urllib.request, 'build_opener') as factory,
                      patch.object(setup_echo.time, 'sleep')):
                    factory.return_value.open.side_effect = responses
                    if success:
                        setup_echo.sources()
                        self.assertEqual((source / 'LICENSE').read_bytes(), expected)
                        self.assertTrue((root / 'VERIFIED-LOCK.json').is_file())
                    else:
                        with self.assertRaises((HTTPError, RuntimeError)):
                            setup_echo.sources()
                        self.assertFalse((source / 'LICENSE').exists())
                        self.assertFalse((root / 'VERIFIED-LOCK.json').exists())
                    self.assertEqual(factory.return_value.open.call_count, attempts)


if __name__ == '__main__':
    unittest.main()
