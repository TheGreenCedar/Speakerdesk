"""Real pipes and SQLite/WAV persistence with synthetic peers; no devices or AI."""
import json
import io
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speakerdesk'))
from app import create_app
from live_meeting import RATE, SourceMixer

PROTOCOL = Path(__file__).resolve().parents[1] / 'support' / 'meeting_protocol.py'


class SyntheticEchoPeer:
    """Protocol-only Linux peer; passes mic through without claiming AEC."""
    def __init__(self, sink):
        self.sink = sink

    def add(self, tracks, final=False):
        if not len(tracks['microphone']):
            return
        originals = {name: tracks[name] for name in ('microphone', 'system')}
        originals['microphone_clean'] = tracks['microphone'].copy()
        mixed = np.clip(originals['system'] + originals['microphone_clean'], -1, 1).astype('<f4')
        self.sink(mixed, originals)

    def close(self):
        pass


class MeetingHarness:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(self.root)
        self.client = self.app.test_client()
        token = re.search(r'name="speakerdesk-token" content="([^"]+)"',
                          self.client.get('/').get_data(as_text=True)).group(1)
        self.headers = {'X-Speakerdesk-Token': token}
        self.manager = self.app.extensions['speakerdesk']['meetings']
        self.manager.helper_path = lambda: PROTOCOL
        self.scenario = 'normal'
        self.children = []
        real_popen = subprocess.Popen

        def synthetic_peer(command, **kwargs):
            role = 'worker' if len(command) > 1 else 'capture'
            folder = Path(kwargs['stderr'].name).parent
            language = json.loads(command[-1])['language'] if role == 'worker' else 'en'
            process = real_popen([sys.executable, str(PROTOCOL), role, str(folder), self.scenario, language], **kwargs)
            self.children.append(process)
            return process

        self.peers = patch('live_meeting.subprocess.Popen', side_effect=synthetic_peer)
        self.preflight = patch('live_meeting.preflight', return_value=[])
        self.peers.start()
        self.preflight.start()
        self.platform_peers=[]
        if platform.system() != 'Darwin':
            # Linux Source checks exercise protocol/persistence with a native
            # platform peer. macOS package checks use the actual linked DSP.
            self.platform_peers=[patch('live_meeting.EchoMixer', SyntheticEchoPeer),
                patch('live_meeting.SourceMixer',side_effect=lambda sources,sink,**kw:
                      SourceMixer(sources,sink,echo_factory=SyntheticEchoPeer,**kw))]
            for peer in self.platform_peers:peer.start()

    def tearDown(self):
        self.manager.close()
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        self.wait_for(lambda: self.manager.jid is None)
        self.peers.stop()
        self.preflight.stop()
        for peer in reversed(self.platform_peers):peer.stop()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.temp.cleanup()

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(.01)
        self.fail(f'Synthetic lifecycle did not reach the expected state within {timeout} seconds.')

    def start(self, sources):
        response = self.client.post('/api/meetings', headers=self.headers,
                                    json={'name': 'Synthetic', 'language': 'en', 'sources': sources})
        self.assertEqual(response.status_code, 201, response.json)
        jid = response.json['id']
        self.wait_for(lambda: (self.root / jid / 'worker-started').exists())
        return jid

    def job(self, jid):
        return self.client.get(f'/api/jobs/{jid}').json

    def control(self, jid, action):
        response = self.client.post(f'/api/meetings/{jid}/{action}', headers=self.headers)
        self.assertEqual(response.status_code, 202, response.json)

    def samples(self, jid, filename):
        with wave.open(str(self.root / jid / filename), 'rb') as recording:
            self.assertEqual((recording.getnchannels(), recording.getsampwidth(), recording.getframerate()), (1, 2, RATE))
            return np.frombuffer(recording.readframes(recording.getnframes()), dtype='<i2')
