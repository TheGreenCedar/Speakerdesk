"""Ordinary startup selects the measured GPU policy without an injected backend."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
import voice_gpu_runtime as gpu


class GPUStartupTests(unittest.TestCase):
    def close_app(self, app):
        app.extensions['speakerdesk']['meetings'].close()
        close = app.extensions['speakerdesk'].get('close_voice')
        if close:close()
        app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    def test_environment_config_uses_gpu_without_starting_a_neural_child(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict(os.environ, {'SPEAKERDESK_VOICE_CONFIG':str(gpu.RELEASE_CALIBRATION),
                                     'SPEAKERDESK_MODELS':str(Path(temporary)/'models')}), \
             patch.object(gpu, 'require_runtime'), patch.object(gpu, 'verify_artifact'), \
             patch('voice_gpu_process.launch_owned_worker') as launch:
            app = create_app(Path(temporary)/'data')
            try:
                self.assertTrue(app.test_client().get('/api/people').json['voice_available'])
                backend, calibration = app.extensions['speakerdesk']['voice_runtime']
                self.assertIn(':mlx-0.32.2:GPU:', backend.model.revision)
                self.assertEqual(calibration.model, backend.model)
                self.assertEqual(calibration.threshold, .5827445050278138)
                launch.assert_not_called()
            finally:self.close_app(app)

    def test_changed_policy_fails_closed_before_child_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root/'changed.json'
            policy.write_text(gpu.RELEASE_CALIBRATION.read_text().replace('0.5827445050278138','0.3'))
            with (patch.dict(os.environ, {'SPEAKERDESK_VOICE_CONFIG':str(policy),
                                         'SPEAKERDESK_MODELS':str(root/'models')}),
                  patch.object(gpu, 'require_runtime'), patch.object(gpu, 'verify_artifact'),
                  patch('voice_setup.VoiceSetup.installed', return_value=True),
                  patch('voice_setup.VoiceSetup.supported', return_value=True),
                  patch('voice_gpu_process.launch_owned_worker') as launch):
                app = create_app(root/'data')
                try:
                    self.assertFalse(app.test_client().get('/api/people').json['voice_available'])
                    launch.assert_not_called()
                finally:self.close_app(app)


if __name__ == '__main__':unittest.main()
