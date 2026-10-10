"""Actual async Apple startup gate and converter controls; no devices or NN."""
import json
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(platform.system() == 'Darwin' and shutil.which('xcrun'), 'Requires macOS AVAudioConverter')
class AppleCaptureStartupTests(unittest.TestCase):
    def test_late_missing_invalid_cancelled_and_failed_startup(self):
        source = (ROOT / 'desktop/capture/MeetingCapture.swift').read_text()
        peer = (ROOT / 'tests/support/apple_startup_orchestration.swift').read_text()
        emission = source[source.index('    private func emitConverted('):source.index('    private func finishConversion()')]
        self.assertIn(emission, peer, 'Peer must execute the unchanged production hold/packet method')
        # The legacy route still starts SCStream before microphone setup; the
        # processed route alone uses the real async first-tap startup gate.
        self.assertIn('if processingJobID == nil, let stream { try await stream.startCapture() }\n            if microphone {', source)
        self.assertIn('if processingJobID != nil {stateLock.withLock {startHost=hostSeconds()}}\n                try engine.start()', source)
        self.assertNotIn('emit(["type":"recording"', source[source.index('    private func packet('):source.index('    private func failCapture(')])
        self.assertIn('if processingJobID != nil {stateLock.withLock {failed=true}}\n            if audioSuspended {audioQueue.resume();audioSuspended=false}', source)
        self.assertIn('await stopOwner.run { [self] in', source)
        self.assertIn('guard active && !paused && !failed && processingAnnounced else', source)
        with tempfile.TemporaryDirectory(prefix='speakerdesk-apple-startup-') as folder:
            folder = Path(folder)
            inputs = [ROOT/'desktop/capture/CaptureAudioConverter.swift', ROOT/'desktop/capture/AppleCaptureRoute.swift',
                      ROOT/'desktop/capture/LosslessRedundantChannels.swift', ROOT/'tests/support/apple_startup_orchestration.swift']
            for path in inputs:
                (folder/path.name).write_bytes(path.read_bytes())
            binary = folder/'startup-check'
            compile_result = subprocess.run(['xcrun', 'swiftc', '-O', '-assert-config', 'Debug', '-j', '1', '-num-threads', '1',
                '-module-cache-path', str(folder/'modules'), *(str(folder/path.name) for path in inputs), '-o', str(binary)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt['count'], 13)
            self.assertTrue(all(receipt['checks'].values()))
            self.assertEqual(receipt['old_error_code'], 6)
            self.assertEqual(receipt['old_emitted_packets'], 0)
            self.assertFalse(receipt['device_opened'])
            self.assertFalse(receipt['capture_started'])
