"""Bound downloads before attachment while preserving canonical meeting data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from app import create_app
from transcript import ExportTooLarge, export


class ExportBoundsTests(unittest.TestCase):
    def document(self):
        return {'speakers': {'speaker_0': 'Zoë'}, 'segments': [
            {'id': 'words', 'start': 0., 'end': 1., 'speaker': 'speaker_0',
             'text': 'Checked café 日本語', 'edited': True},
            {'id': 'blank', 'start': 1., 'end': 2., 'speaker': 'speaker_0',
             'text': '', 'audio_state': 'digital_silence'}]}

    def test_utf8_limit_is_inclusive_and_all_formats_preserve_words(self):
        doc = self.document()
        before = copy.deepcopy(doc)
        for kind in ('txt', 'srt', 'vtt', 'json'):
            with self.subTest(kind=kind):
                content = export(doc, kind)[0]
                size = len(content.encode('utf-8'))
                self.assertGreater(size, len(content))
                with patch('transcript.MAX_EXPORT_BYTES', size):
                    self.assertEqual(export(doc, kind)[0], content)
                with patch('transcript.MAX_EXPORT_BYTES', size - 1):
                    with self.assertRaises(ExportTooLarge):
                        export(doc, kind)
                self.assertIn('Checked café 日本語', content)
        self.assertEqual(doc, before)
        self.assertEqual(json.loads(export(doc, 'json')[0]), doc)

    def test_empty_plain_exports_and_vtt_keep_existing_format(self):
        doc = {'speakers': {'speaker_0': 'Voice'}, 'segments': []}
        self.assertEqual(export(doc, 'txt')[0], '\n')
        self.assertEqual(export(doc, 'srt')[0], '\n')
        self.assertEqual(export(doc, 'vtt')[0], 'WEBVTT\n\n')

    def test_get_and_head_refuse_large_attachment_without_changing_saved_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(root)
            ext = app.extensions['speakerdesk']
            try:
                jid = 'a' * 32
                folder = root / jid
                folder.mkdir()
                audio = folder / 'audio.wav'
                audio.write_bytes(b'original recording fixture')
                job = {'id': jid, 'created': 1., 'name': 'Export fixture',
                       'status': 'ready', 'duration': 2., 'revision': 7,
                       'document': self.document()}
                ext['recognition'].put(job)
                client = app.test_client()
                saved = client.get('/api/jobs/' + jid).json
                for kind in ('txt', 'srt', 'vtt', 'json'):
                    url = f'/api/jobs/{jid}/export/{kind}'
                    with patch('transcript.MAX_EXPORT_BYTES', 8):
                        for method in ('GET', 'HEAD'):
                            response = client.open(url, method=method)
                            self.assertEqual(response.status_code, 413)
                            self.assertNotIn('Content-Disposition', response.headers)
                    response = client.get(url)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('attachment;', response.headers['Content-Disposition'])
                    self.assertIn('Checked café 日本語', response.text)
                self.assertEqual(client.get('/api/jobs/' + jid).json, saved)
                self.assertEqual(audio.read_bytes(), b'original recording fixture')
            finally:
                ext['meetings'].close()
                ext['executor'].shutdown(wait=True, cancel_futures=True)
