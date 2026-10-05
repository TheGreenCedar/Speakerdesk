"""CPU language contracts, with synthetic PCM and model-boundary substitutes.

These protect routing, timing, persistence and the pinned decoder API. They do
not measure acoustic language accuracy or execute a trained model.
"""
import ast
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import re
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'speakerdesk'))
from app import create_app
from language_detection import LanguagePolicy, SpeechTranscriber, WhisperLanguageDetector, language_tokens, detector_issues
from pipeline import infer, preflight
from transcript import validate, export
from voice_profiles import automatic_clips

METADATA = ROOT/'tests/metadata/whisper-tiny'


def scores(language, probability=.97, runner='en'):
    runner = 'fr' if runner == language else runner
    return {language: probability, runner: 1-probability}


def cohere_model():
    model = Mock()
    model.transcribe.side_effect = lambda audio, **kw: types.SimpleNamespace(
        text={'en': 'Original English', 'fr': 'Français original'}[kw['language']], tokens=[1])
    return model


def model_modules(asr, diar=None):
    """Substitute only model/GPU boundaries; run real worker routing code."""
    mx=types.ModuleType('mlx.core')
    for method in ('set_memory_limit','set_cache_limit','clear_cache','synchronize'):
        setattr(mx,method,Mock())
    mx.get_peak_memory=lambda:0
    mlx=types.ModuleType('mlx');mlx.core=mx
    cohere=types.ModuleType('mlx_speech.generation.cohere_asr')
    cohere.CohereAsrModel=types.SimpleNamespace(from_path=lambda path:asr)
    vad=types.ModuleType('mlx_audio.vad');vad.load=lambda path,strict:diar
    return {'mlx':mlx,'mlx.core':mx,cohere.__name__:cohere,vad.__name__:vad}


class LanguageRoutingTests(unittest.TestCase):
    def transcriber(self, distributions):
        detector = Mock()
        detector.detect.side_effect = distributions
        with patch('language_detection.WhisperLanguageDetector', return_value=detector):
            result = SpeechTranscriber(cohere_model(), 'auto', '/local/approved')
        return result

    def test_same_speaker_can_change_english_to_french_inside_one_phrase(self):
        transcriber = self.transcriber([scores('en'), scores('fr')])
        pcm = np.full(6*16000, .1, dtype=np.float32)
        passages = transcriber.transcribe(pcm, 16000, ('speaker_0',))
        self.assertEqual([(p['start'], p['end'], p['language'], p['text']) for p in passages],
                         [(0, 3, 'en', 'Original English'), (3, 6, 'fr', 'Français original')])
        self.assertEqual([call.kwargs['language'] for call in transcriber.asr.transcribe.call_args_list], ['en', 'fr'])
        for call in transcriber.asr.transcribe.call_args_list:
            np.testing.assert_array_equal(call.args[0], pcm[:48000])

    def test_hysteresis_abstains_on_first_disputed_window_and_does_not_force_previous_language(self):
        transcriber = self.transcriber([scores('en'), scores('fr', .8), scores('fr', .8)])
        pcm = np.full(9*16000, .1, dtype=np.float32)
        passages = transcriber.transcribe(pcm, 16000, ('speaker_0',))
        self.assertEqual([p['language'] for p in passages], ['en', None, 'fr'])
        self.assertEqual(passages[1]['text'], '')
        self.assertTrue(passages[1]['review'])
        self.assertEqual(passages[1]['language_detection']['reason'], 'change_pending')
        self.assertEqual([c.kwargs['language'] for c in transcriber.asr.transcribe.call_args_list], ['en', 'fr'])

    def test_unsupported_and_ambiguous_speech_never_call_cohere_or_become_english(self):
        for probabilities, reason in [(scores('ru'), 'unsupported'), ({'en': .51, 'fr': .49}, 'uncertain')]:
            with self.subTest(reason=reason):
                transcriber = self.transcriber([probabilities])
                passage = transcriber.transcribe(np.full(16000, .1, dtype=np.float32), 16000, ('speaker_0',))[0]
                self.assertIsNone(passage['language'])
                self.assertEqual(passage['text'], '')
                self.assertEqual(passage['language_detection']['reason'], reason)
                self.assertEqual(passage['language_detection']['candidates'][0]['language'], max(probabilities, key=probabilities.get))
                transcriber.asr.transcribe.assert_not_called()

    def test_short_silent_and_overlapping_speech_abstain_before_detection(self):
        for pcm, speaker in [(np.full(8000, .1, dtype=np.float32), ('speaker_0',)),
                             (np.zeros(16000, dtype=np.float32), ('speaker_0',)),
                             (np.full(16000, .1, dtype=np.float32), ('speaker_0', 'speaker_1'))]:
            transcriber = self.transcriber([])
            passage = transcriber.transcribe(pcm, 16000, speaker)[0]
            self.assertIsNone(passage['language'])
            self.assertEqual(passage['end'], len(pcm)/16000)
            transcriber.detector.detect.assert_not_called()
            transcriber.asr.transcribe.assert_not_called()

    def test_confident_continuous_language_joins_probes_without_dropping_tail(self):
        transcriber = self.transcriber([scores('fr'), scores('fr'), scores('fr')])
        pcm = np.arange(100001, dtype=np.float32)/1000000
        passages = transcriber.transcribe(pcm, 16000, ('speaker_0',))
        self.assertEqual(passages[0]['start'], 0)
        self.assertEqual(passages[-1]['end'], len(pcm)/16000)
        self.assertTrue(all(p['language']=='fr' for p in passages))
        reconstructed = np.concatenate([call.args[0] for call in transcriber.asr.transcribe.call_args_list])
        np.testing.assert_array_equal(reconstructed, pcm)
        self.assertTrue(all(len(call.args[0]) <= 96000 for call in transcriber.asr.transcribe.call_args_list))

    def test_language_state_is_separate_for_speakers_and_weak_continuity_is_reviewed(self):
        policy = LanguagePolicy()
        self.assertEqual(policy.decide(scores('en'), ('speaker_0',))['language'], 'en')
        self.assertEqual(policy.decide(scores('fr', .8), ('speaker_1',))['language'], 'fr')
        continuation = policy.decide(scores('en', .65), ('speaker_0',))
        self.assertEqual(continuation['language'], 'en')
        self.assertTrue(continuation['review'])
        self.assertEqual(policy.decide(scores('fr', .8), ('speaker_0',))['language'], None)

    def test_manual_override_bypasses_detector_and_keeps_full_crop(self):
        with patch('language_detection.WhisperLanguageDetector') as detector:
            transcriber = SpeechTranscriber(cohere_model(), 'fr')
        pcm = np.full(24*16000, .1, dtype=np.float32)
        passages = transcriber.transcribe(pcm, 16000, ('speaker_0',))
        detector.assert_not_called()
        self.assertEqual([(p['start'],p['end'],p['language']) for p in passages], [(0,24,'fr')])
        np.testing.assert_array_equal(transcriber.asr.transcribe.call_args.args[0], pcm)

    def test_abstention_invalidates_prior_continuity_before_new_weak_evidence(self):
        for contrary in [scores('ru'), {'fr':.55,'en':.45}]:
            policy=LanguagePolicy();policy.decide(scores('en'),('speaker_0',))
            self.assertIsNone(policy.decide(contrary,('speaker_0',))['language'])
            self.assertIsNone(policy.decide(scores('en',.65),('speaker_0',))['language'])
        policy=LanguagePolicy();policy.decide(scores('en'),('speaker_0',))
        policy.decide(scores('fr',.8),('speaker_0',))
        self.assertIsNone(policy.decide(scores('en',.65),('speaker_0',))['language'])

    def test_invalid_audio_and_bad_distribution_fail_explicitly(self):
        transcriber = self.transcriber([])
        for pcm, rate in [(np.zeros(100), 48000), (np.zeros((100,2)),16000), (np.array([np.nan]),16000)]:
            with self.assertRaises(ValueError):transcriber.transcribe(pcm,rate,('speaker_0',))
        for probabilities in [{'en': float('nan')}, {'en': .8, 'fr': .8}, {}]:
            with self.assertRaises(ValueError):LanguagePolicy().decide(probabilities, ('speaker_0',))


class DecoderContractTests(unittest.TestCase):
    def test_short_probe_is_padded_as_waveform_silence_before_mel_features(self):
        mx=types.ModuleType('mlx.core');mx.array=np.array
        mlx=types.ModuleType('mlx');mlx.core=mx
        audio_module=types.ModuleType('mlx_audio.stt.models.whisper.audio')
        audio_module.N_SAMPLES=480000
        audio_module.pad_or_trim=lambda audio,length:np.pad(audio[:length],(0,max(0,length-len(audio))))
        def frontend(audio,n_mels):
            self.assertEqual(len(audio),480000)
            np.testing.assert_array_equal(audio[16000:],np.zeros(464000))
            # A silence feature value different from numeric zero exposes a
            # regression that pads Mel frames instead of the waveform.
            return np.full((3000,n_mels),-1.5,dtype=np.float32)
        audio_module.log_mel_spectrogram=frontend
        decoder=types.ModuleType('mlx_audio.stt.models.whisper.decoding')
        decoder.detect_language=Mock(return_value=(None,scores('fr')))
        detector=WhisperLanguageDetector.__new__(WhisperLanguageDetector)
        detector.model=types.SimpleNamespace(dims=types.SimpleNamespace(n_mels=80),dtype=np.float16)
        detector.tokens=object()
        with patch.dict(sys.modules,{'mlx':mlx,'mlx.core':mx,audio_module.__name__:audio_module,decoder.__name__:decoder}):
            self.assertEqual(detector.detect(np.full(16000,.1,dtype=np.float32)),scores('fr'))
        mel=decoder.detect_language.call_args.args[1]
        self.assertEqual(mel.shape,(3000,80))
        np.testing.assert_array_equal(mel[-1],np.full(80,-1.5))

    def test_real_pinned_decoder_scores_all_languages_with_sot_only_on_cpu(self):
        # Execute the actual dependency's detection function with NumPy operations
        # and deterministic logits, without importing MLX or running a model.
        source = METADATA/'detect_language.py'
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='detect_language')
        # Lightweight CI has no MLX package. Keep an exact pinned excerpt and
        # compare it with the actual dependency when that distribution exists.
        try:
            installed = importlib.metadata.distribution('mlx-audio').locate_file('mlx_audio/stt/models/whisper/decoding.py')
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is not None:
            actual = next(n for n in ast.parse(installed.read_text()).body
                          if isinstance(n,ast.FunctionDef) and n.name=='detect_language')
            self.assertEqual(ast.dump(function,include_attributes=False),
                             ast.dump(actual,include_attributes=False))
        program = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
        mx = types.SimpleNamespace(array=np.array, full=np.full, inf=np.inf, float32=np.float32,
            argmax=np.argmax, softmax=lambda x,axis:np.exp(x-np.max(x,axis=axis,keepdims=True))/np.sum(np.exp(x-np.max(x,axis=axis,keepdims=True)),axis=axis,keepdims=True))
        scope={'mx':mx,'np':np};exec(compile(ast.fix_missing_locations(program),str(source),'exec'),scope)
        config=json.loads((METADATA/'config.json').read_text())
        tokens=language_tokens(json.loads((METADATA/'added_tokens.json').read_text()),config['vocab_size'])
        self.assertEqual(len(tokens.all_language_codes),99)
        self.assertIn('ru',tokens.all_language_codes)
        logits=np.zeros((1,1,config['vocab_size']),dtype=np.float32)
        logits[0,0,123]=1000 # A non-language token must never win.
        logits[0,0,tokens.all_language_tokens[tokens.all_language_codes.index('fr')]]=12
        model=Mock();model.dims=types.SimpleNamespace(n_audio_ctx=config['max_source_positions'],n_audio_state=config['d_model'])
        model.encoder.return_value=np.zeros((1,config['max_source_positions'],config['d_model']),dtype=np.float32)
        model.logits.return_value=logits
        _, probabilities=scope['detect_language'](model,np.zeros((3000,config['num_mel_bins']),dtype=np.float32),tokens)
        self.assertEqual(max(probabilities,key=probabilities.get),'fr')
        self.assertEqual(len(probabilities),99)
        self.assertAlmostEqual(sum(probabilities.values()),1,places=5)
        np.testing.assert_array_equal(model.logits.call_args.args[0],[[50258]])
        model.get_tokenizer.assert_not_called()
        model.decode.assert_not_called()

    def test_missing_or_corrupt_metadata_disables_auto_but_preserves_manual_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name, kind in [('diar','nemotron_diarization'),('asr','cohere_asr')]:
                folder=root/name;folder.mkdir()
                (folder/'config.json').write_text(json.dumps({'model_type':kind}))
                (folder/'model.safetensors').touch()
            cfg={'diar_path':str(root/'diar'),'cohere_path':str(root/'asr'), 'lid_path':str(root/'missing'),
                 'diar_python':sys.executable,'asr_python':sys.executable,'device':'mlx'}
            self.assertEqual(preflight(cfg,'fr'),[])
            self.assertTrue(preflight(cfg,'auto'))
            (root/'missing').mkdir();(root/'missing/config.json').write_text('{}')
            self.assertTrue(detector_issues(root/'missing'))


class WorkerLanguageTests(unittest.TestCase):
    def test_import_pipeline_routes_mixed_language_with_correct_offsets_and_cleans_only_crops(self):
        import inference_worker
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            audio=folder/'audio.wav'
            with wave.open(str(audio),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                wav.writeframes(np.full(6*16000,3276,dtype='<i2').tobytes())
            original=audio.read_bytes()
            model=folder/'cohere';model.mkdir();(model/'model.safetensors').touch()
            asr=cohere_model();detector=Mock();detector.detect.side_effect=[scores('en'),scores('fr')]
            def worker(python,task,request,destination):
                if task=='diarize':return {'turns':['0 6 speaker_0']}
                return inference_worker.run(task,request)
            cfg={'diar_path':'unused','cohere_path':str(model),'lid_path':'local-approved',
                 'diar_python':sys.executable,'asr_python':sys.executable,'diar_kind':'nemotron','device':'mlx'}
            with patch.dict(sys.modules,model_modules(asr)), patch('inference_worker.check_memory'), \
                 patch('language_detection.WhisperLanguageDetector',return_value=detector), \
                 patch('pipeline.preflight',return_value=[]),patch('pipeline.run_worker',side_effect=worker):
                document=validate(infer(audio,6,'auto',folder,lambda message:None,cfg),6)
            self.assertEqual([(s['start'],s['end'],s['language'],s['text']) for s in document['segments']],
                             [(0,3,'en','Original English'),(3,6,'fr','Français original')])
            self.assertTrue(all(segment['voice_eligible'] for segment in document['segments']))
            clips=automatic_clips({'id':'auto-import','document':document},'speaker_0')
            self.assertEqual([(c.start,c.end) for c in clips],[(0,3),(3,6)])
            self.assertEqual(document['provenance']['language'],'auto')
            self.assertIn('whisper-tiny',document['provenance']['language_detector'])
            self.assertEqual(audio.read_bytes(),original)
            self.assertFalse((folder/'crops').exists())
            self.assertEqual([c.kwargs['language'] for c in asr.transcribe.call_args_list],['en','fr'])

    def test_live_worker_emits_auto_subpassages_and_blank_uncertainty_with_original_timeline(self):
        import live_worker
        for probabilities, languages, texts in [([scores('en'),scores('fr')],['en','fr'],['Original English','Français original']),
                                                ([scores('en'),scores('ru')],['en',None],['Original English',''])]:
            with self.subTest(languages=languages):
                class Diarizer:
                    def set_streaming_config(self,preset):pass
                    def init_streaming_state(self):return types.SimpleNamespace(frames_processed=0)
                    def feed(self,pcm,state,**kwargs):
                        start=state.frames_processed*.01
                        state.frames_processed+=len(pcm)//160
                        segments=([types.SimpleNamespace(start=start,end=state.frames_processed*.01,speaker=0)] if len(pcm) else [])
                        return types.SimpleNamespace(segments=segments),state
                pcm=np.full(16000,.1,dtype='<f4').tobytes()
                lines=[json.dumps({'type':'audio','pcm':base64.b64encode(pcm).decode()}) for _ in range(6)]
                lines.append(json.dumps({'type':'stop'}))
                detector=Mock();detector.detect.side_effect=probabilities
                asr=cohere_model();output=io.StringIO()
                with patch.dict(sys.modules,model_modules(asr,Diarizer())),patch('inference_worker.check_memory'), \
                     patch('language_detection.WhisperLanguageDetector',return_value=detector), \
                     patch('sys.stdin',io.StringIO('\n'.join(lines)+'\n')),contextlib.redirect_stdout(output):
                    live_worker.run({'diar_path':'unused','cohere_path':'unused','language':'auto','lid_path':'local-approved'})
                events=[json.loads(line) for line in output.getvalue().splitlines()]
                segments=[e['segment'] for e in events if e['type']=='segment']
                self.assertEqual([(s['start'],s['end']) for s in segments],[(0,3),(3,6)])
                self.assertEqual([s['language'] for s in segments],languages)
                self.assertEqual([s['text'] for s in segments],texts)
                self.assertTrue(all(s['speaker_candidates']==['speaker_0'] for s in segments))
                self.assertTrue(all(s['voice_eligible'] for s in segments))
                job={'id':'auto-live','document':{'provenance':{'kind':'local_inference'},'segments':segments}}
                clips=automatic_clips(job,'speaker_0')
                self.assertEqual([(c.start,c.end) for c in clips],[(0,3),(3,6)])
                self.assertEqual(events[-1]['type'],'finished')
                self.assertEqual(events[-1]['duration'],6)
                if languages[-1] is None:
                    self.assertTrue(segments[-1]['review'])
                    self.assertEqual(segments[-1]['language_detection']['reason'],'unsupported')
                    self.assertEqual(asr.transcribe.call_count,1)


    def test_auto_overlap_stays_blank_and_cannot_supply_voice_enrollment_clips(self):
        import inference_worker
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);audio=folder/'audio.wav'
            with wave.open(str(audio),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                wav.writeframes(np.full(6*16000,3276,dtype='<i2').tobytes())
            original=audio.read_bytes()
            model=folder/'cohere';model.mkdir();(model/'model.safetensors').touch()
            asr=cohere_model();detector=Mock()
            def worker(python,task,request,destination):
                if task=='diarize':return {'turns':['0 6 speaker_0','0 6 speaker_1']}
                return inference_worker.run(task,request)
            config={'diar_path':'unused','cohere_path':str(model),'lid_path':'local-approved',
                    'diar_python':sys.executable,'asr_python':sys.executable,'diar_kind':'nemotron','device':'mlx'}
            with patch.dict(sys.modules,model_modules(asr)),patch('inference_worker.check_memory'), \
                 patch('language_detection.WhisperLanguageDetector',return_value=detector), \
                 patch('pipeline.preflight',return_value=[]),patch('pipeline.run_worker',side_effect=worker):
                document=validate(infer(audio,6,'auto',folder,lambda message:None,config),6)
            self.assertEqual([(s['start'],s['end']) for s in document['segments']],[(0,3),(3,6)])
            self.assertTrue(all(s['text']=='' and s['review'] and not s['voice_eligible'] for s in document['segments']))
            self.assertTrue(all(s['language_detection']['reason']=='overlapping_speech' for s in document['segments']))
            self.assertEqual(automatic_clips({'id':'overlap','document':document},'speaker_0'),[])
            self.assertEqual(automatic_clips({'id':'overlap','document':document},'overlap'),[])
            detector.detect.assert_not_called();asr.transcribe.assert_not_called()
            self.assertEqual(audio.read_bytes(),original)


class LanguageAppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.app=create_app(self.root);self.client=self.app.test_client()
        self.html=self.client.get('/').get_data(as_text=True)
        token=re.search(r'name="speakerdesk-token" content="([^"]+)"',self.html)[1]
        self.headers={'X-Speakerdesk-Token':token}

    def tearDown(self):
        self.app.extensions['speakerdesk']['meetings'].close()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
        self.temp.cleanup()

    def test_auto_defaults_for_new_imports_and_meetings_and_manual_override_persists(self):
        for language in [None,'fr']:
            output=io.BytesIO()
            with wave.open(output,'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\x01\x00'*16000)
            output.seek(0)
            data={'files':(output,'Meeting.wav')}
            if language:data['language']=language
            response=self.client.post('/api/jobs',headers=self.headers,data=data)
            self.assertEqual(response.status_code,201)
            response.request.input_stream.close()
            jid=response.json[0]['id']
            self.app.extensions['speakerdesk']['executor'].submit(lambda:None).result(timeout=5)
            self.assertEqual(self.client.get(f'/api/jobs/{jid}').json['language'],language or 'auto')
        manager=self.app.extensions['speakerdesk']['meetings']
        with patch.object(manager,'start',return_value={'id':'synthetic'}) as start:
            self.assertEqual(self.client.post('/api/meetings',headers=self.headers,json={'sources':['microphone']}).status_code,201)
            self.assertEqual(start.call_args.args[1],'auto')
        config=self.client.get('/api/config').json
        self.assertEqual(config['default_language'],'auto')
        self.assertEqual(config['languages']['auto'],'Automatic')
        # The only language selector is optional Settings UI, outside start form.
        form=self.html.split('<form id="meeting-form">')[1].split('</form>')[0]
        self.assertNotIn('id="language"',form)
        self.assertEqual(self.html.count('id="language"'),1)
        self.assertIn('id="language"',self.html.split('<div id="setup"')[1])

    def test_setup_detects_same_size_corrupt_language_weights_and_offers_repair(self):
        # Tiny bytes substitute the checkpoint artifact; no real model is used.
        model_root=self.root/'models';checkpoint=model_root/'language';checkpoint.mkdir(parents=True)
        weights=checkpoint/'model.safetensors';weights.write_bytes(b'bad')
        spec={'name':'Language detector','directory':'language','bytes':3,
              'files':['model.safetensors'],
              'file_sha256':{'model.safetensors':hashlib.sha256(b'yes').hexdigest()}}
        with patch('model_setup.SPECS',[spec]),patch.dict('os.environ',{'SPEAKERDESK_MODELS':str(model_root)}):
            app=create_app(self.root/'repair-test')
            try:
                client=app.test_client()
                self.assertFalse(client.get('/api/setup').json['core_ready'])
                weights.write_bytes(b'yes')
                self.assertTrue(client.get('/api/setup').json['core_ready'])
                weights.write_bytes(b'bad')
                self.assertFalse(client.get('/api/setup').json['models'][0]['installed'])
            finally:
                app.extensions['speakerdesk']['meetings'].close()
                app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_language_metadata_and_blank_review_passage_survive_editor_export_and_restart(self):
        jid='f'*32
        folder=self.root/jid;folder.mkdir();(folder/'audio.wav').write_bytes(b'retained audio')
        document=validate({'speakers':{'speaker_0':'Speaker 1'},'segments':[
            {'id':'en','start':0,'end':3,'speaker':'speaker_0','text':'Original words','language':'en'},
            {'id':'uncertain','start':3,'end':6,'speaker':'speaker_0','text':'','language':None,
             'review':True,'language_detection':{'mode':'auto','reason':'uncertain','candidates':[{'language':'fr','probability':.5}]}}]},6)
        import sqlite3
        with contextlib.closing(sqlite3.connect(self.root/'jobs.sqlite')) as database:
            database.execute('INSERT INTO jobs VALUES (?,?)',(jid,json.dumps({'id':jid,'status':'ready','created':1,
                'name':'Meeting','duration':6,'revision':1,'language':'auto','document':document})));database.commit()
        document['segments'][0]['text']='Edited original words'
        saved=self.client.put(f'/api/jobs/{jid}/transcript',headers=self.headers,json={'revision':1,'document':document})
        self.assertEqual(saved.status_code,200)
        expected={**document['segments'][1],'voice_eligible':False,'speaker_candidates':['speaker_0']}
        self.assertEqual(saved.json['document']['segments'][1],expected)
        exported=self.client.get(f'/api/jobs/{jid}/export/json').json
        self.assertEqual(exported['segments'][1]['language_detection']['reason'],'uncertain')
        self.assertEqual((folder/'audio.wav').read_bytes(),b'retained audio')
        reopened=create_app(self.root)
        try:self.assertEqual(reopened.test_client().get(f'/api/jobs/{jid}').json['document'],saved.json['document'])
        finally:
            reopened.extensions['speakerdesk']['meetings'].close()
            reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':unittest.main()
