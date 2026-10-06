"""Loopback browser fixture: real API/assets/SQLite, synthetic PCM peers only.

No device API, model download or model inference is invoked. Stop with Ctrl-C.
"""
import json
import signal
import sys
import unittest
from pathlib import Path

from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support.meeting_harness import MeetingHarness


class Fixture(MeetingHarness, unittest.TestCase):
    pass


if __name__ == '__main__':
    fixture = Fixture()
    fixture.setUp()
    server = None
    try:
        fixture.scenario = 'language_browser'
        jid = fixture.start(['microphone'])
        fixture.wait_for(lambda: fixture.job(jid)['status'] == 'recording')
        fixture.control(jid, 'pause')
        fixture.wait_for(lambda: len(fixture.job(jid)['document']['segments']) == 1)
        server = make_server('127.0.0.1', 0, fixture.app, threaded=True)
        print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}',
                          'id': jid, 'folder': str(fixture.root/jid)}), flush=True)
        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
        server.serve_forever()
    finally:
        if server:
            server.server_close()
        # Release the intentionally delayed configuration acknowledgement.
        if fixture.manager.jid:
            (fixture.root/fixture.manager.jid/'language-release').touch()
        fixture.tearDown()
