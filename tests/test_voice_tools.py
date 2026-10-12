"""Installer integrity and rollback without network or inference."""
import hashlib
from contextlib import redirect_stdout
from io import BytesIO
from io import StringIO
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import download_voice_model as installer
import generate_voice_fixtures as generator


class VoiceToolTests(unittest.TestCase):
    def test_generated_corpus_keeps_native_manifest_bounds_without_the_python_app(self):
        voices = [dict(key=k, say_name=n, voice_id=k, role=r) for k, n, _, _, r in generator.VOICES]
        count = 0
        def audio_utility(budget, command):
            nonlocal count
            if command[0] == '/usr/bin/say':
                Path(command[command.index('-o') + 1]).write_bytes(b'fabricated AIFF')
            else:
                count += 1
                pcm = [round((4000 + count) * math.sin(2 * math.pi * (300 + count) * i / 16000)) for i in range(40000)]
                generator.write_pcm(Path(command[2]), pcm)
        is_file = Path.is_file
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'corpus'
            with (patch.object(generator.platform, 'system', return_value='Darwin'),
                  patch.object(Path, 'is_file', lambda p: str(p) in ('/usr/bin/say', '/usr/bin/afconvert', '/usr/bin/osascript') or is_file(p)),
                  patch.object(generator, 'installed_voices', return_value=('fabricated inventory', voices)),
                  patch.object(generator.Budget, 'run', audio_utility),
                  patch.object(generator.shutil, 'disk_usage', return_value=SimpleNamespace(free=50_000_000_000)),
                  redirect_stdout(StringIO())):
                generator.generate(output)
            report = json.loads((output / 'generation-report.json').read_text())
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(report['session_groups'], 36)
            self.assertEqual(report['embedding_clips'], 72)
            self.assertEqual(report['unique_audio_sha256'], 36)
            self.assertEqual(sum(len(g['clips']) for g in manifest['groups']), 72)
            self.assertTrue(all(c['clean'] is True for g in manifest['groups'] for c in g['clips']))

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



if __name__ == '__main__': unittest.main()
