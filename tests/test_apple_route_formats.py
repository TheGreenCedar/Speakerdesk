"""Execute production Apple route predicates on generated formats, no devices."""
import json
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(platform.system()=='Darwin' and shutil.which('xcrun'), 'Requires macOS AVAudioFormat')
class AppleRouteFormatTests(unittest.TestCase):
    def test_microphone_and_render_roles_with_actual_native_predicates(self):
        with tempfile.TemporaryDirectory(prefix='speakerdesk-route-formats-') as directory:
            directory=Path(directory)
            sources=[ROOT/'desktop/capture/AppleCaptureRoute.swift',
                     ROOT/'desktop/capture/LosslessRedundantChannels.swift',
                     ROOT/'tests/support/apple_route_formats.swift']
            binary=directory/'formats'
            built=subprocess.run(['xcrun','swiftc','-O','-assert-config','Debug','-j','1','-num-threads','1',
                '-module-cache-path',str(directory/'modules'),*(str(p) for p in sources),'-o',str(binary)],
                capture_output=True,text=True,timeout=120)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            receipt=json.loads(result.stdout)
            self.assertEqual(receipt['count'],19)
            self.assertTrue(all(receipt['checks'].values()))
            self.assertFalse(receipt['device_opened'])
            self.assertFalse(receipt['capture_started'])
            self.assertEqual(receipt['neural_calls'],0)
