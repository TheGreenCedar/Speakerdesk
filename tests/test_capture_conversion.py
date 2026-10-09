"""Exercise the production macOS SRC with generated buffers, without devices."""
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(platform.system() == 'Darwin' and shutil.which('xcrun'), 'Requires macOS AVAudioConverter')
class CaptureConversionTests(unittest.TestCase):
    def test_buffered_output_keeps_samples_timestamps_and_eof_tail(self):
        with tempfile.TemporaryDirectory(prefix='speakerdesk-conversion-') as folder:
            binary = Path(folder)/'conversion-check'
            sources = [ROOT/'desktop/capture/CaptureAudioConverter.swift', ROOT/'tests/support/capture_conversion.swift']
            contents = {source: source.read_bytes() for source in sources}
            # Compile an immutable local snapshot. Cloud filesystem metadata
            # changes must not invalidate a build of unchanged source bytes.
            for source, content in contents.items():
                (Path(folder)/source.name).write_bytes(content)
            result = subprocess.run([
                'xcrun', 'swiftc', '-O', '-assert-config', 'Debug', '-j', '1', '-num-threads', '1',
                '-module-cache-path', str(Path(folder)/'modules'),
                *(str(Path(folder)/source.name) for source in sources), '-o', str(binary),
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            for source, content in contents.items():
                self.assertEqual(source.read_bytes(), content, 'Source changed during conversion test')
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(result.stdout.count('frames=32000 max_waveform_error=0.0 contiguous=true EOF_idempotent=true'), 4)
