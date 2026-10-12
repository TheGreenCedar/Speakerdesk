"""Pinned source fetching tolerates transient outages without weakening integrity."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_echo


class EchoDownloadTests(unittest.TestCase):
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
