"""Installer integrity/rollback and measurement report ownership, without network or inference."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import download_voice_model as installer
import measure_voice as measurement


class VoiceToolTests(unittest.TestCase):
    def test_pinned_download_is_bounded_validated_and_failed_staging_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pin = root/'pin.json'; body = b'small test model'
            spec = {'model_id': 'test/model', 'revision': 'fixed-revision', 'total_bytes': len(body),
                    'files': {'model/weights.bin': {'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}}}
            pin.write_text(json.dumps(spec))
            with patch.object(installer, 'PIN_PATH', pin), patch.object(installer.shutil, 'disk_usage', return_value=SimpleNamespace(free=50_000_000_000)), patch.object(installer.urllib.request, 'urlopen', side_effect=lambda *a, **kw: BytesIO(body)) as fetch:
                self.assertEqual(installer.download(root/'installed'), len(body))
                self.assertEqual((root/'installed/model/weights.bin').read_bytes(), body)
                self.assertEqual(fetch.call_args.args[0], 'https://huggingface.co/test/model/resolve/fixed-revision/model/weights.bin')
                with self.assertRaisesRegex(ValueError, 'existing data'): installer.download(root/'installed')
                for bad in [body+b'extra', body[:-1], b'Small test model']:
                    fetch.side_effect = lambda *a, **kw: BytesIO(bad)
                    with self.subTest(body=bad), self.assertRaises(ValueError): installer.download(root/'failed')
                    self.assertFalse((root/'failed').exists()); self.assertEqual(list(root.glob('.voice-download-*')), [])
                with patch.object(installer.shutil, 'disk_usage', return_value=SimpleNamespace(free=40_000_000_000)):
                    fetch.reset_mock()
                    with self.assertRaisesRegex(ValueError, '40 GB'): installer.download(root/'no-space')
                    fetch.assert_not_called()

    def test_existing_report_is_preserved_on_cli_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); report = root/'report.json'; report.write_text('existing report')
            script = Path(__file__).resolve().parents[1]/'scripts/measure_voice.py'
            result = subprocess.run([sys.executable, str(script), '--model-dir', str(root/'missing'),
                                     '--fixtures', str(root/'missing.json'), '--output', str(report), '--smoke'],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1); self.assertIn('new measurement report', result.stderr)
            self.assertEqual(report.read_text(), 'existing report')

    def test_native_supervisor_terminates_worker_on_wall_time_or_memory_breach(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary)/'new.json', timeout_seconds=60, maximum_rss_mib=1024)
            for times, rss, message in [([0, 61], '1', 'wall-time'), ([0, 1], str(1024*1024+1), 'resident-memory'),
                                        ([0, 1], '', 'monitoring is unavailable')]:
                child = SimpleNamespace(pid=123, poll=unittest.mock.Mock(return_value=None),
                                        kill=unittest.mock.Mock(), wait=unittest.mock.Mock())
                with (self.subTest(message=message),
                      patch.object(measurement.subprocess, 'Popen', return_value=child),
                      patch.object(measurement.time, 'monotonic', side_effect=times),
                      patch.object(measurement.shutil, 'disk_usage', return_value=SimpleNamespace(free=50_000_000_000)),
                      patch.object(measurement.subprocess, 'run', return_value=SimpleNamespace(stdout=rss, returncode=0))):
                    with self.assertRaisesRegex(ValueError, message): measurement.bounded_worker(args)
                    child.kill.assert_called_once(); child.wait.assert_called_once()


if __name__ == '__main__': unittest.main()
