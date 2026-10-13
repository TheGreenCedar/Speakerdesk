"""Production fence and callback admission state, without opening audio devices."""
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''
@main struct CheckFlow {
    static func main() {
        let flow = CaptureFlowControl(byteLimit: 16)
        precondition(flow.reserveCopy(12))
        precondition(!flow.reserveCopy(5))
        flow.releaseCopy(12)
        precondition(flow.reserveCopy(16))
        precondition(!flow.reserveCopy(1))
        flow.releaseCopy(16)
        precondition(!flow.reserveCopy(-1))
        precondition(flow.requestFence(at: 1) == nil)
        flow.enableFences()
        let queue = DispatchQueue(label: "capture.production.fence.test")
        let held = DispatchSemaphore(value: 0)
        let entered = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        var events: [String] = []
        queue.async { entered.signal(); held.wait(); events.append("converted PCM") }
        entered.wait()
        let fence = flow.requestFence(at: 10)!
        precondition(flow.requestFence(at: 11) == nil, "fences must coalesce behind conversion")
        queue.async {
            precondition(flow.completeFence(fence) == 1)
            events.append("fence \(fence.time)")
            precondition(events == ["converted PCM", "fence 10.0"])
            finished.signal()
        }
        held.signal()
        precondition(finished.wait(timeout: .now() + 5) == .success)
        let old = flow.requestFence(at: 12)!
        flow.invalidateFences()
        precondition(flow.requestFence(at: 13) == nil, "paused/stopped producers cannot queue fences")
        flow.enableFences()
        let current = flow.requestFence(at: 14)!
        precondition(flow.completeFence(old) == nil, "old epoch cannot publish")
        precondition(flow.requestFence(at: 15) == nil, "old completion cannot release new pending fence")
        precondition(flow.completeFence(current) == 2)
        print("capture flow passed; no audio device opened")
    }
}
'''


@unittest.skipUnless(platform.system() == 'Darwin' and shutil.which('xcrun'), 'Requires Foundation Swift')
class CaptureFlowTests(unittest.TestCase):
    def test_delayed_conversion_coalesces_fences_and_bounds_copied_pcm(self):
        source = (ROOT / 'desktop/capture/MeetingCapture.swift').read_text()
        production = source[source.index('final class CaptureFlowControl {'):source.index('final class MeetingCapture:')]
        with tempfile.TemporaryDirectory(prefix='speakerdesk-flow-') as folder:
            folder = Path(folder)
            for name, text, success in [
                ('production', production, True),
                ('unbounded-negative', production.replace('bytes <= byteLimit - copiedBytes', 'true'), False),
                ('stale-negative', production.replace('guard fence.epoch == epoch else { return nil }', ''), False),
            ]:
                path = folder / f'{name}.swift'
                path.write_text('import Foundation\n' + text + DRIVER)
                binary = folder / name
                built = subprocess.run(['xcrun', 'swiftc', '-parse-as-library', '-assert-config', 'Debug', '-j', '1',
                    '-num-threads', '1', '-module-cache-path', str(folder / 'modules'), str(path), '-o', str(binary)],
                    capture_output=True, text=True, timeout=45)
                self.assertEqual(built.returncode, 0, built.stderr)
                result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10,
                                        cwd=folder)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
