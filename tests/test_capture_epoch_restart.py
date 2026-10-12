"""Pure callback-generation/reset controls; no AV engine, SDK devices or NN."""
import json
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(platform.system()=='Darwin' and shutil.which('xcrun'), 'Requires Foundation-only Swift')
class CaptureEpochRestartTests(unittest.TestCase):
    def test_same_helper_restarts_clear_state_and_refuse_old_callbacks(self):
        source=(ROOT/'desktop/capture/MeetingCapture.swift').read_text()
        peer=(ROOT/'tests/support/capture_epoch_restart.swift').read_text()
        reset=source[source.index('    private func resetCaptureEpoch()'):source.index('    private func packet(')]
        prefix=source[source.index('    private func packet('):source.index('        let targetFormat =')]
        stop_error=source[source.index('    func stream(_ stream: SCStream, didStopWithError error: Error) {'):source.index('\n}\n',source.index('    func stream(_ stream: SCStream, didStopWithError error: Error) {'))]
        self.assertIn(stop_error,peer)
        self.assertIn('guard acceptedSystemGeneration(stream) != nil else {return}',stop_error)
        self.assertIn(reset,peer)
        self.assertIn(prefix,peer)
        self.assertIn('await stopOwner.reset()\n        resetCaptureEpoch()',source)
        self.assertIn('let generation=stateLock.withLock {captureGeneration}\n                node.installTap',source)
        self.assertIn('generation:generation,observation:observation)',source)
        self.assertIn('guard let generation=acceptedSystemGeneration(stream)',source)
        with tempfile.TemporaryDirectory(prefix='speakerdesk-epoch-peer-') as folder:
            folder=Path(folder)
            route=(ROOT/'desktop/capture/AppleCaptureRoute.swift').read_text()
            resume_boundary=route[route.index('struct AppleMicrophoneResumeBoundary {'):route.index('\nfunc requireSameAppleInputDevice')]
            snapshot=folder/'restart.swift';snapshot.write_text(resume_boundary+'\n'+peer)
            binary=folder/'restart'
            built=subprocess.run(['xcrun','swiftc','-parse-as-library','-O','-j','1','-num-threads','1','-module-cache-path',str(folder/'modules'),str(snapshot),'-o',str(binary)],capture_output=True,text=True,timeout=45)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            receipt=json.loads(result.stdout)
            self.assertEqual(receipt['count'],9)
            self.assertTrue(all(receipt['checks'].values()))
            self.assertFalse(receipt['device_opened'])
