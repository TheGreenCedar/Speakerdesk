"""Real pipes and SQLite/WAV persistence with synthetic peers; no devices or AI."""
import json
import io
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from app import create_app
from live_meeting import RATE

PROTOCOL = Path(__file__).parent / 'support' / 'meeting_protocol.py'


class MeetingLifecycleTests(unittest.TestCase):
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
            process = real_popen([sys.executable, str(PROTOCOL), role, str(folder), self.scenario], **kwargs)
            self.children.append(process)
            return process

        self.peers = patch('live_meeting.subprocess.Popen', side_effect=synthetic_peer)
        self.preflight = patch('live_meeting.preflight', return_value=[])
        self.peers.start()
        self.preflight.start()

    def tearDown(self):
        self.manager.close()
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        self.wait_for(lambda: self.manager.jid is None)
        self.peers.stop()
        self.preflight.stop()
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

    def test_selected_sources_pause_resume_stop_and_reopen_preserve_audio_and_transcript(self):
        for sources in (['microphone'], ['system'], ['microphone', 'system']):
            with self.subTest(sources=sources):
                jid = self.start(sources)
                self.wait_for(lambda: self.job(jid)['status'] == 'recording')
                self.control(jid, 'pause')
                self.wait_for(lambda: self.job(jid)['status'] == 'paused')
                self.assertAlmostEqual(self.client.get('/api/meeting').json['duration'], .1)
                self.control(jid, 'resume')
                self.wait_for(lambda: self.job(jid)['status'] == 'recording')
                self.control(jid, 'stop')
                self.wait_for(lambda: self.manager.jid is None)
                saved = self.job(jid)
                self.assertEqual(saved['status'], 'ready')
                self.assertEqual(saved['duration'], .35)
                self.assertEqual(saved['document']['segments'][0]['text'], 'Synthetic transport result.')
                self.assertEqual(saved['document']['segments'][0]['end'], .35)
                values = {'microphone': (.2, .4), 'system': (.3, .1)}
                for source, pair in values.items():
                    if source in sources:
                        expected = np.concatenate([np.full(1600, pair[0]), np.full(4000, pair[1])])
                        np.testing.assert_array_equal(self.samples(jid, f'{source}.wav'), (expected * 32767).astype('<i2'))
                    else:
                        self.assertFalse((self.root / jid / f'{source}.wav').exists())
                expected_mix = sum(np.concatenate([np.full(1600, values[source][0]), np.full(4000, values[source][1])]) for source in sources)
                np.testing.assert_array_equal(self.samples(jid, 'audio.wav'), (expected_mix * 32767).astype('<i2'))
                with self.client.get(f'/api/jobs/{jid}/audio') as response:
                    self.assertEqual(response.status_code, 200)
                self.assertIn('Synthetic transport result.', self.client.get(f'/api/jobs/{jid}/export/txt').get_data(as_text=True))
                reopened = create_app(self.root)
                try:
                    self.assertEqual(reopened.test_client().get(f'/api/jobs/{jid}').json['document'], saved['document'])
                finally:
                    reopened.extensions['speakerdesk']['meetings'].close()
                    reopened.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    def test_stop_during_model_startup_cancels_without_starting_capture(self):
        self.scenario = 'startup'
        jid = self.start(['microphone'])
        self.control(jid, 'stop')
        self.wait_for(lambda: self.manager.jid is None, timeout=2)
        self.assertEqual(self.job(jid)['status'], 'failed')
        self.assertFalse((self.root / jid / 'capture-commands.jsonl').exists())
        self.assertFalse((self.root / jid / 'audio.wav').exists())

    def test_live_result_hook_names_the_track_and_finalization_keeps_latest_identity(self):
        recognizer=self.app.extensions['speakerdesk']['recognition']
        calls=[]
        def recognize(jid,track):
            calls.append((jid,track))
            job=self.manager.get(jid)
            self.assertEqual(job['document']['provenance']['kind'],'local_inference')
            job['document']['speakers'][track]='Priya'
            self.manager.put(job)
        with patch.object(recognizer,'observe',side_effect=recognize):
            jid=self.start(['microphone'])
            self.wait_for(lambda:self.job(jid)['status']=='recording')
            self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        self.assertEqual(calls,[(jid,'speaker_0')])
        self.assertEqual(self.job(jid)['document']['speakers']['speaker_0'],'Priya')
        self.assertEqual(self.job(jid)['status'],'ready')

    def test_failed_recording_directory_creation_does_not_strand_the_manager(self):
        try:
            with patch.object(Path, 'mkdir', side_effect=OSError('Synthetic directory failure.')), self.assertLogs(self.app.logger, level='ERROR'):
                response = self.client.post('/api/meetings', headers=self.headers,
                                            json={'name': 'Unavailable storage', 'language': 'en', 'sources': ['microphone']})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(self.client.get('/api/jobs').json, [])
            status = self.client.get('/api/meeting')
            self.assertEqual(status.status_code, 200, status.json)
            self.assertEqual(status.json['status'], 'idle')
            self.assertEqual(self.children, [])
        finally:
            # The baseline has no worker to clean up but leaves an invalid owner ID.
            if getattr(self.manager, 'thread', None) is None:self.manager.jid = None
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.job(jid)['status'] == 'recording')
        self.control(jid, 'stop')
        self.wait_for(lambda: self.manager.jid is None)
        self.assertEqual(self.job(jid)['status'], 'ready')

    def test_capture_error_preserves_unflushed_audio_tail(self):
        self.scenario = 'capture_error'
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.manager.jid is None)
        saved = self.job(jid)
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('Synthetic capture failure', saved['message'])
        self.assertEqual(len(self.samples(jid, 'audio.wav')), 17600)
        self.assertEqual(saved['duration'], 1.1)

    def test_capture_error_after_stopped_acknowledgement_is_not_saved_as_success(self):
        self.scenario = 'capture_error_after_stop'
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.manager.jid is None)
        saved = self.job(jid)
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('Synthetic capture failure', saved['message'])
        self.assertEqual(len(self.samples(jid, 'audio.wav')), 17600)

    def test_backlog_failure_preserves_all_audio_and_exact_saved_duration(self):
        self.scenario = 'backlog'
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.manager.jid is None)
        saved = self.job(jid)
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('behind', saved['message'])
        audio = self.samples(jid, 'audio.wav')
        self.assertEqual(len(audio), 545600)
        self.assertEqual(saved['duration'], len(audio) / RATE)
        np.testing.assert_array_equal(audio, np.full(545600, int(.2 * 32767), dtype='<i2'))

    def test_worker_error_stops_capture_and_preserves_recording(self):
        self.scenario = 'worker_error'
        jid = self.start(['system'])
        self.wait_for(lambda: self.manager.jid is None)
        saved = self.job(jid)
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('Synthetic worker failure', saved['message'])
        self.assertEqual(len(self.samples(jid, 'audio.wav')), 17600)
        self.assertEqual(saved['duration'], 1.1)

    def test_worker_error_terminates_a_capture_helper_that_ignores_stop(self):
        self.scenario = 'worker_error_stalled'
        jid = self.start(['system'])
        self.wait_for(lambda: (self.root / jid / 'capture-ignored-stop').exists())
        self.wait_for(lambda: self.manager.jid is None, timeout=18)
        saved = self.job(jid)
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('Synthetic worker failure', saved['message'])
        self.assertEqual(len(self.samples(jid, 'audio.wav')), 17600)
        self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_late_recording_acknowledgement_cannot_undo_stop(self):
        self.scenario = 'delayed_stop'
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.job(jid)['status'] == 'recording')
        self.control(jid, 'stop')
        # This clock follows the late recording event on the same output pipe.
        self.wait_for(lambda: self.job(jid)['duration'] == .25)
        self.assertEqual(self.job(jid)['status'], 'finishing')
        (self.root / jid / 'capture-release').touch()
        self.wait_for(lambda: self.manager.jid is None)
        self.assertEqual(self.job(jid)['status'], 'ready')
        self.assertEqual(len(self.samples(jid, 'audio.wav')), 12000)
        self.assertEqual(self.job(jid)['document']['segments'][0]['end'], .75)

    def test_close_waits_for_saved_audio_and_terminal_job_state(self):
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.job(jid)['status'] == 'recording')
        # Pause acknowledges that the first packet has reached the manager.
        self.control(jid, 'pause')
        self.wait_for(lambda: self.job(jid)['status'] == 'paused')
        self.manager.close()
        self.assertIsNone(self.manager.jid)
        self.assertEqual(self.job(jid)['status'], 'failed')
        self.assertEqual(self.job(jid)['duration'], .1)
        self.assertEqual(len(self.samples(jid, 'audio.wav')), 1600)

    def test_file_or_pipe_close_error_releases_meeting_and_preserves_transcript(self):
        class FailingClose:
            def __init__(self, handle):self.handle=handle
            def __getattr__(self, name):return getattr(self.handle,name)
            def close(self):
                self.handle.close()
                raise OSError('Synthetic buffered close failure.')

        original_open=Path.open
        for target in ('file','pipe'):
            with self.subTest(target=target):
                def open_with_failure(path, *args, **kwargs):
                    handle=original_open(path,*args,**kwargs)
                    return FailingClose(handle) if target=='file' and path.name=='capture.log' else handle
                with patch.object(Path,'open',open_with_failure):
                    jid=self.start(['microphone'])
                    self.wait_for(lambda:self.job(jid)['status']=='recording')
                    self.control(jid,'pause')
                    self.wait_for(lambda:self.job(jid)['status']=='paused')
                    if target=='pipe':self.manager.capture.stdin=FailingClose(self.manager.capture.stdin)
                    self.control(jid,'stop')
                    self.wait_for(lambda:self.manager.jid is None)
                saved=self.job(jid)
                self.assertEqual(saved['status'],'failed')
                self.assertIn('cleanup failed',saved['message'])
                self.assertEqual(saved['document']['segments'][0]['text'],'Synthetic transport result.')
                self.assertEqual(len(self.samples(jid,'audio.wav')),1600)
                self.assertIsNone(self.manager.capture)
                self.assertIsNone(self.manager.worker)
                self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_source_track_saturates_without_wrapping_pcm_samples(self):
        self.scenario = 'source_clip'
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.job(jid)['status'] == 'recording')
        self.control(jid, 'stop')
        self.wait_for(lambda: self.manager.jid is None)
        self.assertEqual(self.job(jid)['status'], 'ready')
        expected = np.full(1600, 32767, dtype='<i2')
        np.testing.assert_array_equal(self.samples(jid, 'audio.wav'), expected)
        np.testing.assert_array_equal(self.samples(jid, 'microphone.wav'), expected)

    def test_import_inference_and_live_worker_cannot_run_at_the_same_time(self):
        recording = io.BytesIO()
        with wave.open(recording, 'wb') as audio:
            audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(RATE)
            audio.writeframes(b'\x00\x00' * RATE)
        recording.seek(0)
        upload = self.client.post('/api/jobs', headers=self.headers,
                                  data={'files': [(recording, 'Import.wav')], 'language': 'en'})
        jid = upload.json[0]['id']
        self.wait_for(lambda: self.job(jid)['status'] == 'uploaded')
        started, release = threading.Event(), threading.Event()

        def infer_later(*args):
            started.set()
            release.wait(timeout=5)
            raise RuntimeError('Synthetic import worker failure.')

        try:
            with patch('app.preflight', return_value=[]), patch('app.infer', side_effect=infer_later):
                meeting = self.start(['microphone'])
                self.wait_for(lambda: self.job(meeting)['status'] == 'recording')
                response = self.client.post(f'/api/jobs/{jid}/run', headers=self.headers)
                self.assertEqual(response.status_code, 409, response.json)
                self.assertFalse(started.is_set())
                self.control(meeting, 'stop')
                self.wait_for(lambda: self.manager.jid is None)
                self.assertEqual(self.client.post(f'/api/jobs/{jid}/run', headers=self.headers).status_code, 202)
                self.assertTrue(started.wait(timeout=2))
                response = self.client.post('/api/meetings', headers=self.headers,
                                            json={'name': 'Blocked', 'language': 'en', 'sources': ['system']})
                self.assertEqual(response.status_code, 409, response.json)
                self.assertEqual(len(self.client.get('/api/jobs').json), 2)
                self.assertEqual(len(self.children), 2)
                release.set()
                self.wait_for(lambda: self.job(jid)['status'] == 'failed')
        finally:
            release.set()
