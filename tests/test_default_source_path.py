"""Ordinary app factory chooses one source path; no device or neural startup."""
import tempfile
from pathlib import Path
import unittest
from app import create_app


class DefaultSourcePathTests(unittest.TestCase):
    def test_ordinary_factory_enables_one_path_without_starting_recording_or_models(self):
        with tempfile.TemporaryDirectory() as folder:
            app=create_app(Path(folder)/'recordings')
            manager=app.extensions['speakerdesk']['meetings']
            try:
                self.assertTrue(manager.channel_transcription)
                self.assertTrue(manager.source_innovation)
                self.assertTrue(manager.source_startup_hold)
                self.assertIsNone(manager.capture);self.assertIsNone(manager.worker);self.assertIsNone(manager.jid)
                self.assertEqual(app.test_client().get('/api/meeting').status_code,200)
            finally:
                manager.close();app.extensions['speakerdesk']['close_voice']()
                app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
