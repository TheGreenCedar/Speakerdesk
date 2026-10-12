"""Native build telemetry tolerates archive renames without weakening its guard."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_echo


class EchoBuildTests(unittest.TestCase):
    def test_disappearing_archive_does_not_abort_build_or_hide_output_budget(self):
        for oversized, error in ((False, FileNotFoundError), (True, FileNotFoundError),
                                 (False, PermissionError)):
            with self.subTest(oversized=oversized, error=error), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                build = root / 'build'
                build.mkdir()
                archive = build / 'libabsl_int128.a.incoming'
                archive.write_bytes(b'temporary archive')
                stable = build / 'stable.o'
                with stable.open('wb') as output:
                    output.truncate(512 * 1024**2 + 1 if oversized else 12)
                (build / 'libspeakerdesk_echo.dylib').write_bytes(b'finished library')
                target = root / 'desktop/capture/echo/libspeakerdesk_echo.dylib'
                target.parent.mkdir(parents=True)
                child = Mock(pid=123, returncode=0)
                child.poll.side_effect = [None, 0, 0]
                real_stat, real_is_file = Path.stat, Path.is_file

                def stat(path, *args, **kwargs):
                    if path == archive:
                        raise error(str(path))
                    return real_stat(path, *args, **kwargs)

                def is_file(path):
                    # Ninja renames the archive after the old guard's is_file check.
                    return True if path == archive else real_is_file(path)

                with (patch.object(setup_echo, 'ROOT', root),
                      patch.object(setup_echo, 'BUILD', build),
                      patch.object(setup_echo.platform, 'system', return_value='Darwin'),
                      patch.object(setup_echo.platform, 'machine', return_value='arm64'),
                      patch.object(setup_echo, 'sources', return_value={}),
                      patch.object(setup_echo.subprocess, 'check_output', return_value='clang'),
                      patch.object(setup_echo.subprocess, 'run'),
                      patch.object(setup_echo.subprocess, 'Popen', return_value=child),
                      patch.object(setup_echo.time, 'sleep'),
                      patch.object(Path, 'stat', stat),
                      patch.object(Path, 'is_file', is_file)):
                    if error is PermissionError:
                        with self.assertRaises(PermissionError):
                            setup_echo.build()
                        self.assertFalse(target.exists())
                    elif oversized:
                        with self.assertRaisesRegex(RuntimeError, 'Native build output budget'):
                            setup_echo.build()
                        self.assertFalse(target.exists())
                    else:
                        setup_echo.build()
                        self.assertEqual(target.read_bytes(), b'finished library')
                        receipt = json.loads((build / 'BUILD-RECEIPT.json').read_text())
                        self.assertIsNone(receipt['stop_reason'])
                        self.assertEqual(receipt['exit_code'], 0)
                        self.assertGreaterEqual(receipt['output_bytes'], stable.stat().st_size)
                child.wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
