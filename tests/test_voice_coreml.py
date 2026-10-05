"""CPU boundary regressions; no Core ML, model bytes, inference or real voices."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import voice_coreml as core
from voice_profiles import VoiceClip
from app import create_app


class CoreMLBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.audio = self.root/'speech.wav'
        self.samples = (np.sin(np.arange(16000*12)*.031)*6000).astype('<i2')
        self.write_audio(self.samples)
        self.clip = VoiceClip('meeting', 'track', 'segment', 1., 4.)

    def tearDown(self):
        self.temp.cleanup()

    def write_audio(self, samples, *, rate=16000):
        with wave.open(str(self.audio), 'wb') as out:
            out.setparams((1, 2, rate, 0, 'NONE', 'not compressed')); out.writeframes(samples.tobytes())

    def test_pcm_crop_repeat_and_center_crop_preserve_gain_and_fixed_shape(self):
        samples = core.read_clip(self.audio, self.clip)
        np.testing.assert_array_equal(samples, self.samples[16000:64000].astype(np.float32)/32768)
        fixed = core.prepared_audio(samples)
        self.assertEqual((fixed.shape, fixed.dtype), ((96000,), np.dtype('float32')))
        np.testing.assert_array_equal(fixed[:48000], samples); np.testing.assert_array_equal(fixed[48000:], samples)
        long = np.arange(16000*8, dtype=np.float32)
        np.testing.assert_array_equal(core.prepared_audio(long), long[16000:112000])
        for bad in [np.zeros(31999), np.zeros(160001), np.full(32000, np.nan), np.zeros((1, 32000))]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError): core.prepared_audio(bad)

    def test_wave_format_truncation_and_bad_bounds_fail(self):
        self.write_audio(self.samples, rate=8000)
        with self.assertRaisesRegex(ValueError, '16 kHz'): core.read_clip(self.audio, self.clip)
        self.write_audio(self.samples)
        for bounds in [(11, 14), (-1, 2), (0, 1), (0, float('nan'))]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                core.read_clip(self.audio, VoiceClip('m', 't', 's', *bounds))
        self.audio.write_bytes(b'not a wave file')
        with self.assertRaisesRegex(ValueError, 'could not read'): core.read_clip(self.audio, self.clip)

    def test_silence_and_saturation_reject_before_native_load(self):
        backend = core.ReDimNet2CoreML(self.root/'missing')
        with patch.object(backend, 'validate') as validate:
            for samples in [np.zeros(16000*12, dtype='<i2'), np.full(16000*12, 32767, dtype='<i2')]:
                self.write_audio(samples); self.assertFalse(backend.embed(self.audio, self.clip).clean)
            validate.assert_not_called()
        self.assertIsNone(backend._predictor)

    def test_lazy_load_prediction_contract_and_invalid_output(self):
        predictor = Mock(); predictor.predict.return_value = {'embedding': np.ones((1, 192), dtype=np.float32)}
        factory = Mock(return_value=predictor)
        fake = SimpleNamespace(models=SimpleNamespace(CompiledMLModel=factory), ComputeUnit={'CPU_ONLY': 'cpu'})
        backend = core.ReDimNet2CoreML(self.root, 'CPU_ONLY'); self.assertIsNone(backend._predictor)
        with patch.object(backend, 'validate') as validate, patch.dict(sys.modules, {'coremltools': fake}):
            result = backend.embed(self.audio, self.clip); second = backend.embed(self.audio, self.clip)
            validate.assert_called_once(); factory.assert_called_once()
        call = predictor.predict.call_args.args[0]['audio']
        self.assertEqual((call.shape, call.dtype), ((1, 96000), np.dtype('float32')))
        self.assertAlmostEqual(sum(v*v for v in result.vector), 1.)
        self.assertAlmostEqual(result.metrics['raw_output_norm'], np.sqrt(192)); self.assertEqual(second.metrics['load_ms'], 0.)
        for bad in [None, {}, {'embedding': np.zeros(192)}, {'embedding': np.ones(191)},
                    {'embedding': np.ones(192, dtype=np.int32)}, {'embedding': np.full(192, np.inf)}]:
            predictor.predict.return_value = bad
            with self.subTest(output=type(bad).__name__), self.assertRaises(ValueError): backend.embed(self.audio, self.clip)
        predictor.predict.side_effect = RuntimeError('native failure')
        with self.assertRaises(ValueError) as failure: backend.embed(self.audio, self.clip)
        self.assertIsInstance(failure.exception.__cause__,RuntimeError)
        self.assertNotIn('native failure',str(failure.exception))

    def test_integrity_detects_same_size_tampering_extra_files_and_symlinks(self):
        config = {'model_type': 'redimnet2-b6-speaker-coreml', 'sample_rate': 16000, 'input_samples': 96000,
                  'embedding_dimension': 192, 'input_name': 'audio', 'output_name': 'embedding',
                  'output_normalized': True, 'compiled_model': 'test.mlmodelc', 'artifact_revision': 'test'}
        files = {'config.json': json.dumps(config).encode(), 'test.mlmodelc/weights/weight.bin': b'small fixture'}
        pin = {'compiled_model': 'test.mlmodelc', 'artifact_revision': 'test', 'files': {}}
        for name, body in files.items():
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
            pin['files'][name] = {'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
        weight = self.root/'test.mlmodelc/weights/weight.bin'
        with patch.object(core, 'PIN', pin):
            core.verify_artifact(self.root); weight.write_bytes(b'Small fixture')
            with self.assertRaisesRegex(ValueError, 'checksum'): core.verify_artifact(self.root)
            weight.write_bytes(files['test.mlmodelc/weights/weight.bin'])
            extra = self.root/'test.mlmodelc/extra'; extra.write_text('unexpected')
            with self.assertRaisesRegex(ValueError, 'unexpected compiled'): core.verify_artifact(self.root)
            extra.unlink(); extra.symlink_to(self.root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symbolic'): core.verify_artifact(self.root)
            extra.unlink(); weight.unlink(); weight.symlink_to(self.audio)
            with self.assertRaisesRegex(ValueError, 'symbolic'): core.verify_artifact(self.root)

    def test_policy_model_pin_and_runtime_requirements(self):
        self.assertNotEqual(core.model_identity('ALL'), core.model_identity('CPU_ONLY'))
        with self.assertRaises(ValueError): core.model_identity('invented')
        with patch.object(core.platform, 'system', return_value='Darwin'), patch.object(core.platform, 'machine', return_value='arm64'), patch.object(core.platform, 'mac_ver', return_value=('15.0', (), '')), patch.object(core.importlib.metadata, 'version', return_value='9.0'):
            core.require_runtime()
            with patch.object(core.platform, 'mac_ver', return_value=('14.7', (), '')):
                with self.assertRaisesRegex(ValueError, 'macOS 15'): core.require_runtime()
            with patch.object(core.importlib.metadata, 'version', return_value='8.0'):
                with self.assertRaisesRegex(ValueError, '9.0'): core.require_runtime()

    def measured_config(self):
        return {'schema_version': 1, 'approved_for_recognition': True, 'fixture_source': 'synthetic', 'compute_units': 'ALL',
                'model_dir': 'model', 'model': core.model_identity('ALL').payload(),
                'calibration': {'dataset_id': 'cpu:calibration', 'threshold': .8, 'margin': .1, 'genuine_trials': 2,
                                'impostor_trials': 2, 'false_accept_rate': 0., 'false_reject_rate': 0.},
                'held_out': {'dataset_id': 'cpu:held-out', 'genuine_trials': 2, 'impostor_trials': 2,
                             'false_accept_rate': 0., 'false_reject_rate': 0., 'meets_error_limits': True}}

    def test_reviewed_independent_compatible_config_is_required(self):
        path = self.root/'voice.json'; config = self.measured_config(); path.write_text(json.dumps(config))
        with patch.object(core.ReDimNet2CoreML, 'validate') as validate:
            backend, policy = core.load_approved_runtime(path); validate.assert_called_once()
            self.assertIsNone(backend._predictor); self.assertEqual(policy.model, backend.model)
        changes = [{'approved_for_recognition': False}, {'held_out': {}}, {'model': {}},
                   {'held_out': {**config['held_out'], 'dataset_id': 'cpu:calibration'}},
                   {'held_out': {**config['held_out'], 'meets_error_limits': False}}, {'fixture_source': 'unconsented'}]
        with patch.object(core.ReDimNet2CoreML, 'validate') as validate:
            for change in changes:
                path.write_text(json.dumps({**config, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError): core.load_approved_runtime(path)
            validate.assert_not_called()

    def test_bad_voice_config_preserves_names_and_app(self):
        path = self.root/'bad.json'; path.write_text('[]')
        with patch.dict(os.environ, {'SPEAKERDESK_VOICE_CONFIG': str(path)}): app = create_app(self.root/'data')
        try:
            response = app.test_client().get('/api/people'); self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json['voice_available']); self.assertIn('local setup', response.json['voice_message'])
            self.assertEqual(app.test_client().get('/').status_code, 200)
        finally:
            app.extensions['speakerdesk']['meetings'].close()
            app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__': unittest.main()
