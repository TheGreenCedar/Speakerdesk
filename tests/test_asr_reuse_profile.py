"""Model-free portable activation contracts; no neural imports or sockets."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import asr_reuse_profile as subject
from final_asr_reuse import digest


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.package=self.root/'relocated-frozen-runtime';self.package.mkdir()
        self.models=self.root/'user-selected-model';self.models.mkdir()
        self.distribution=self.root/'installed-packages';self.distribution.mkdir()
        self.state={'config':{'extra':{'quantization':{'bits':4,'group_size':64,'mode':'affine'}}},
                    'feature_config':{'dither':1e-5,'sr':16000},'training':False,
                    'projection_dtype':'mlx.core.uint32'}
        self.provider=SimpleNamespace(config=copy.deepcopy(self.state['config']),
            feature_extractor=SimpleNamespace(**self.state['feature_config']),
            model=SimpleNamespace(training=False,proj_out=SimpleNamespace(
                weight=SimpleNamespace(dtype='mlx.core.uint32'))),tokenizer=object())
        self.mx=SimpleNamespace(gpu='gpu',metal=SimpleNamespace(is_available=lambda:True),
            default_device=lambda:SimpleNamespace(type='gpu'),
            default_stream=lambda device:SimpleNamespace(device=SimpleNamespace(type='gpu')))
        self.profile={'schema_version':1,'identity':{'device':'gpu',
            'model_file_sha256':{'model.safetensors':self.write(self.models/'model.safetensors',b'weight'),
                                  'tokenizer.json':self.write(self.models/'tokenizer.json',b'tokens')},
            'runtime_versions':{'mlx':'0.32.2','mlx-speech':'0.5.3','numpy':'2.5.3'},
            'decoder_source_sha256':{'mlx_speech/generation/cohere_asr.py':
                self.write(self.distribution/'mlx_speech/generation/cohere_asr.py',b'forward-source')},
            'policy_source_sha256':{'canonical_runtime.py':self.write(self.package/'canonical_runtime.py',b'policy')},
            'loaded_provider_state':copy.deepcopy(self.state)}}
        sha=self.write(self.package/'asr-reuse-profile.json',json.dumps(self.profile).encode())
        self.addCleanup(patch.stopall)
        patch.object(subject,'DATA',self.package).start()
        patch.object(subject,'PROFILE_SHA256',sha).start()
        patch.object(subject.importlib.metadata,'version',side_effect=lambda name:self.profile['identity']['runtime_versions'].get(name,'0.5.7')).start()
        patch.object(subject.importlib.metadata,'distribution',return_value=SimpleNamespace(
            locate_file=lambda name:self.distribution/name)).start()
        self.config={'cohere_path':str(self.models)}

    @staticmethod
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def activate(self,**changes):
        return subject.initialize_reuse({**self.config,**changes},self.provider,self.mx)

    def test_default_qualified_reuse_is_empty_and_process_local(self):
        cache,status=self.activate()
        self.assertTrue(status['enabled']);self.assertTrue(cache.qualified)
        self.assertEqual(cache.mode,'reuse');self.assertEqual(len(cache.entries),0)
        self.assertIsNone(cache.provider)
        self.assertEqual(cache.identity_sha256,digest(self.profile['identity']))
        self.assertNotIn('model_instance',cache.identity['loaded_provider_state'])
        self.assertNotIn('tokenizer_instance',cache.identity['loaded_provider_state'])
        self.assertFalse(status['final_namespace_execution_qualified'])
        second,_=self.activate();self.assertIsNot(second,cache)

    def test_observe_and_explicit_disabled(self):
        cache,status=self.activate(asr_final_reuse_mode='observe')
        self.assertEqual(cache.mode,'observe');self.assertTrue(status['enabled'])
        for mode in (None,'off'):
            cache,status=self.activate(asr_final_reuse_mode=mode)
            self.assertIsNone(cache);self.assertEqual(status['reason'],'explicitly_disabled')

    def test_relocated_and_frozen_style_raw_files_work_without_task_paths(self):
        self.assertEqual(subject.qualified_identity(self.models,self.provider,self.mx),self.profile['identity'])
        encoded=json.dumps(self.profile['identity'])
        self.assertNotIn(str(self.root),encoded)
        self.assertNotIn('/Users/',encoded)

    def test_model_token_missing_or_changed_fails_to_fresh(self):
        for name in ('model.safetensors','tokenizer.json'):
            with self.subTest(name=name):
                path=self.models/name;original=path.read_bytes();path.write_bytes(b'changed')
                cache,status=self.activate();self.assertIsNone(cache);self.assertFalse(status['enabled'])
                path.unlink();cache,status=self.activate();self.assertIsNone(cache)
                path.write_bytes(original)

    def test_runtime_or_installed_decoder_or_policy_mismatch_fails_to_fresh(self):
        with patch.object(subject.importlib.metadata,'version',return_value='other'):
            self.assertIsNone(self.activate()[0])
        for path in (self.distribution/'mlx_speech/generation/cohere_asr.py',self.package/'canonical_runtime.py'):
            original=path.read_bytes();path.write_bytes(b'other')
            self.assertIsNone(self.activate()[0]);path.unlink();self.assertIsNone(self.activate()[0])
            path.write_bytes(original)

    def test_profile_corruption_and_missing_fail_to_fresh(self):
        path=self.package/'asr-reuse-profile.json';path.write_bytes(b'{}')
        self.assertIsNone(self.activate()[0]);path.unlink();self.assertIsNone(self.activate()[0])

    def test_CPU_device_CPU_stream_and_missing_metal_fail_to_fresh(self):
        for attribute,value in (
            ('default_device',lambda:SimpleNamespace(type='cpu')),
            ('default_stream',lambda device:SimpleNamespace(device=SimpleNamespace(type='cpu')))):
            with patch.object(self.mx,attribute,value):self.assertIsNone(self.activate()[0])
        with patch.object(self.mx.metal,'is_available',return_value=False):self.assertIsNone(self.activate()[0])
        with patch.object(self.mx.metal,'is_available',side_effect=RuntimeError('backend unavailable')):
            self.assertIsNone(self.activate()[0])

    def test_changed_loaded_config_feature_precision_or_training_fails_to_fresh(self):
        variants=[lambda:setattr(self.provider.model,'training',True),
                  lambda:setattr(self.provider.model.proj_out.weight,'dtype','float32'),
                  lambda:setattr(self.provider.feature_extractor,'dither',0),
                  lambda:self.provider.config['extra']['quantization'].update(bits=8)]
        for change in variants:
            original=copy.deepcopy(self.provider);change();self.assertIsNone(self.activate()[0]);self.provider=original

    def test_arbitrary_experimental_identity_cannot_override_profile(self):
        cache,_=self.activate(asr_reuse_identity={'device':'gpu'},asr_reuse_qualification='f'*64)
        self.assertEqual(cache.identity,self.profile['identity'])
        (self.models/'model.safetensors').write_bytes(b'wrong')
        self.assertIsNone(self.activate(asr_reuse_identity=self.profile['identity'],
            asr_reuse_qualification=digest(self.profile['identity']))[0])

    def test_real_Models_constructor_uses_portable_profile_and_refuses_later_CPU_stream(self):
        import sys
        from live_refinement import Models
        import speech_admission
        import inference_worker
        mx=self.mx
        mx.set_memory_limit=lambda value:None;mx.set_cache_limit=lambda value:None
        diar=SimpleNamespace(set_streaming_config=lambda value:None,init_streaming_state=lambda:object())
        speech=SimpleNamespace(session=lambda **kwargs:object())
        modules={'mlx':SimpleNamespace(core=mx),'mlx.core':mx,
            'mlx_audio.vad':SimpleNamespace(load=lambda *args,**kwargs:diar),
            'mlx_speech.generation.cohere_asr':SimpleNamespace(
                CohereAsrModel=SimpleNamespace(from_path=lambda path:self.provider))}
        with patch.dict(sys.modules,modules),patch.object(inference_worker,'check_memory'),\
                patch.object(speech_admission,'SileroModel',return_value=speech),\
                patch.object(speech_admission,'FrameArchive',return_value=object()):
            models=Models({**self.config,'diar_path':'unused','speech_path':'unused',
                           'audio_path':str(self.root/'unused.wav')})
        self.assertTrue(models.final_asr_reuse.qualified)
        self.assertTrue(models.asr_reuse_profile['enabled'])
        self.assertEqual(len(models.final_asr_reuse.entries),0)
        with patch.object(mx,'default_stream',return_value=SimpleNamespace(device=SimpleNamespace(type='cpu'))):
            with self.assertRaisesRegex(RuntimeError,'GPU'):
                models.begin_asr_request({'stage':'refined'})
            with self.assertRaisesRegex(RuntimeError,'GPU'):
                models.transcribe([], 'en', [])

    def test_invalid_mode_is_fresh_not_an_unvalidated_cache(self):
        self.assertIsNone(self.activate(asr_final_reuse_mode='unsafe')[0])


class ProductionProfileTests(unittest.TestCase):
    def test_committed_legacy_profile_preserves_actual_inherited_qualification(self):
        p=subject.profile()
        self.assertEqual(p['qualification']['source_commit'],'3a08593e391d527bc15b6f03064aaafcc04f13cb')
        self.assertEqual(p['qualification']['outcome_sha256'],'8c3b34eda40d096b944a4f206316f8d590698eb209235ce1bfe0906c5c0febed')
        self.assertFalse(p['qualification']['final_namespace_execution_qualified'])
        self.assertEqual(len(p['identity']['model_file_sha256']),10)
        self.assertEqual(len(p['identity']['decoder_source_sha256']),8)
        changed={name for name,expected in p['identity']['policy_source_sha256'].items()
                 if subject.file_digest(subject.DATA/name)!=expected}
        self.assertEqual(changed,{'canonical_runtime.py','live_refinement.py','meeting_refinement.py'})
        current=subject.current_profile()
        self.assertEqual(current['lineage']['legacy_profile_sha256'],subject.PROFILE_SHA256)
        for name,expected in current['identity']['policy_source_sha256'].items():
            self.assertEqual(subject.file_digest(subject.DATA/name),expected)
        self.assertNotIn('/Users/',json.dumps(p))

    def test_package_ships_profile_own_policy_raw_files_and_decoder_raw_sources(self):
        root=Path(__file__).resolve().parents[1];spec=(root/'packaging/runtime.spec').read_text()
        self.assertIn("'asr_reuse_profile'",spec)
        self.assertIn("'speakerdesk/asr-reuse-profile.json'",spec)
        self.assertIn("'mlx_speech':'pyz+py'",spec)
        for name in subject.profile()['identity']['policy_source_sha256']:
            self.assertIn("'"+name[:-3]+"'",spec)


if __name__=='__main__':unittest.main()
