"""CPU import/editor/export regressions through real Flask, WAV and SQLite boundaries."""
import io
import re
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from werkzeug.datastructures import FileStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from app import create_app


def pcm_file():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\x01\x00' * 16000)
    output.seek(0)
    return output


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(self.root)
        self.client = self.app.test_client()
        token = re.search(r'name="speakerdesk-token" content="([^"]+)"',
                          self.client.get('/').get_data(as_text=True)).group(1)
        self.headers = {'X-Speakerdesk-Token': token}

    def tearDown(self):
        self.app.extensions['speakerdesk']['meetings'].close()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.temp.cleanup()

    def job(self, jid):
        return self.client.get(f'/api/jobs/{jid}').json

    def wait_prepared(self, jid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.job(jid)
            if job['status'] != 'preparing':
                return job
            time.sleep(.01)
        self.fail('Import preparation did not finish within five seconds.')

    def upload(self, files):
        response = self.client.post('/api/jobs', headers=self.headers,
                                    data={'files': files, 'language': 'en'})
        response.request.input_stream.close()
        return response

    def test_pcm_batch_import_edit_export_and_restart_preserve_user_data(self):
        originals = [pcm_file(), pcm_file()]
        original_bytes = [audio.getvalue() for audio in originals]
        response = self.upload([(audio, name) for audio, name in zip(originals, ['First.wav', 'Second.wav'])])
        self.assertEqual(response.status_code, 201)
        jobs = [self.wait_prepared(job['id']) for job in response.json]
        self.assertEqual([job['status'] for job in jobs], ['uploaded', 'uploaded'])
        for job, expected in zip(jobs, original_bytes):
            with self.client.get(f'/api/jobs/{job["id"]}/audio') as audio:
                self.assertEqual(audio.get_data(), expected)
            self.assertEqual(job['duration'], 1)
        jid = jobs[0]['id']
        document = {'speakers': {'speaker_0': ' Alice '}, 'segments': [
            {'id': 'one', 'start': .125, 'end': .875, 'speaker': 'speaker_0', 'text': 'Keep my edits.'}],
            'provenance': {'kind': 'local_inference'}, 'warnings': []}
        saved = self.client.put(f'/api/jobs/{jid}/transcript', headers=self.headers,
                                json={'revision': 0, 'document': document, 'imported': True})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json['document']['provenance']['kind'], 'imported')
        self.assertEqual(saved.json['document']['speakers']['speaker_0'], 'Alice')
        before = saved.json['document']
        bad = {**before, 'segments': [{**before['segments'][0], 'end': 2}]}
        self.assertEqual(self.client.put(f'/api/jobs/{jid}/transcript', headers=self.headers,
                                        json={'revision': 1, 'document': bad}).status_code, 400)
        self.assertEqual(self.job(jid)['document'], before)
        self.assertEqual(self.client.put(f'/api/jobs/{jid}/transcript', headers=self.headers,
                                        json={'revision': 0, 'document': before}).status_code, 409)
        for kind, marker in [('txt', 'Alice: Keep my edits.'), ('srt', '00:00:00,125 --> 00:00:00,875'),
                             ('vtt', '00:00:00.125 --> 00:00:00.875'), ('json', 'Keep my edits.')]:
            exported = self.client.get(f'/api/jobs/{jid}/export/{kind}')
            self.assertEqual(exported.status_code, 200)
            self.assertIn(marker, exported.get_data(as_text=True))
        with patch('app.preflight', return_value=[]), patch('app.infer') as infer:
            self.assertEqual(self.client.post(f'/api/jobs/{jid}/run', headers=self.headers).status_code, 409)
            infer.assert_not_called()
        reopened = create_app(self.root)
        try:
            self.assertEqual(reopened.test_client().get(f'/api/jobs/{jid}').json['document'], before)
        finally:
            reopened.extensions['speakerdesk']['meetings'].close()
            reopened.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.assertEqual(self.client.delete(f'/api/jobs/{jid}', headers=self.headers).status_code, 200)
        self.assertFalse((self.root / jid).exists())
        self.assertEqual(self.client.get(f'/api/jobs/{jid}').status_code, 404)
        self.assertEqual(self.job(jobs[1]['id'])['status'], 'uploaded')

    def test_failed_batch_save_does_not_leave_hidden_jobs_or_orphaned_audio(self):
        original_save = FileStorage.save
        count = 0

        def disk_failure(item, destination, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('Synthetic disk write failure.')
            return original_save(item, destination, *args, **kwargs)

        with patch.object(FileStorage, 'save', autospec=True, side_effect=disk_failure), self.assertLogs(self.app.logger, level='ERROR'):
            response = self.upload([(pcm_file(), 'First.wav'), (pcm_file(), 'Second.wav')])
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.client.get('/api/jobs').json, [])
        self.assertEqual([path for path in self.root.iterdir() if path.is_dir()], [])

    def test_invalid_and_full_queue_batches_do_not_partially_import(self):
        response = self.upload([(pcm_file(), 'First.wav'), (io.BytesIO(b'no'), 'Bad.txt')])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get('/api/jobs').json, [])
        import threading
        release = threading.Event()
        started = threading.Event()

        def prepare_later(source, destination):
            started.set()
            release.wait(timeout=5)
            raise ValueError('Synthetic decode failure.')

        try:
            with patch('app.normalize', side_effect=prepare_later):
                batch = self.upload([(pcm_file(), f'{index}.wav') for index in range(16)])
                self.assertEqual(batch.status_code, 201)
                self.assertTrue(started.wait(timeout=2))
                jid = batch.json[0]['id']
                self.assertEqual(self.upload([(pcm_file(), 'Overflow.wav')]).status_code, 429)
                self.assertEqual(self.client.delete(f'/api/jobs/{jid}', headers=self.headers).status_code, 409)
                self.assertEqual(self.client.post(f'/api/jobs/{jid}/run', headers=self.headers).status_code, 409)
                self.assertEqual(self.client.put(f'/api/jobs/{jid}/transcript', headers=self.headers,
                                                json={'revision': 0, 'document': {}}).status_code, 409)
                release.set()
                self.app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        finally:
            release.set()
        failed = self.job(jid)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('Synthetic decode failure', failed['message'])
        self.assertTrue((self.root / jid / 'source.wav').exists())
