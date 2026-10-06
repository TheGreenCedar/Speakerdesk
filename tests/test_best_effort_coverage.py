"""Real routing/engine contracts with explicit CPU-only acoustic peers."""
import copy
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock,patch
import wave
import numpy as np
from pcm_peer import varying_pcm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from language_detection import SpeechTranscriber
from live_refinement import Engine, Inbox, context_regions
from meeting_refinement import RefinementController
from pipeline import speech_crops
from pipeline import infer
from rolling_refinement import RATE, reconcile_window, segment_version
from transcript import export
from voice_profiles import speaker_audio_eligible


class RoutingModels:
    """No trained model is executed: detector/ASR are controlled CPU peers."""
    def __init__(self):
        self.detector=Mock();self.detector.detect.return_value={'en':.99,'fr':.01}
        self.asr=Mock();self.asr.transcribe.side_effect=lambda audio,**kw:types.SimpleNamespace(text=kw['language']+' retained words',tokens=[1])
        self.context=None;self.start=0;self.seeds=[]
    def set_language_context(self,context,start):
        self.context=context;self.start=start;self.seeds.append(copy.deepcopy(context))
    def transcribe(self,audio,language,names,overlap=False):
        return SpeechTranscriber(self.asr,language,detector=self.detector,context=self.context).transcribe(
            audio,RATE,tuple(names),start_sample=self.start,max_asr_seconds=24.5,allow_overlap=overlap)
    def batch_turns(self,audio):return []


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'audio.wav'
        with wave.open(str(self.path),'wb') as f:
            f.setparams((1,2,RATE,0,'NONE','not compressed'));f.writeframes(b'\x01\x00\xff\xff'*(90*RATE//2))
        self.models=RoutingModels();self.events=[];self.engine=Engine({'audio_path':str(self.path),'language':'auto'},self.models,self.events.append,Inbox())
    def tearDown(self):self.temp.cleanup()
    def decode(self,start=0,epoch=0,language='auto',names=()):
        return self.engine.decode(varying_pcm(RATE,1/32767,dtype=np.float32),language,names,start,epoch,overlap=len(names)>1)
    def test_coverage_includes_pauses_and_missing_speakers_without_assigning_their_words(self):
        turns=[{'start':1,'end':2,'speaker':'speaker_0'},{'start':3,'end':4,'speaker':'speaker_1'}]
        self.assertEqual(speech_crops(turns,coverage=(0,5)),[
            {'start':0,'end':1,'speakers':[]},{'start':1,'end':2,'speakers':['speaker_0']},
            {'start':2,'end':3,'speakers':[]},{'start':3,'end':4,'speakers':['speaker_1']},
            {'start':4,'end':5,'speakers':[]}])
        self.assertEqual(context_regions(turns,coverage=(0,5)),speech_crops(turns,coverage=(0,5)))
        self.engine.received=round(.1*RATE);self.engine.processed=.1
        self.engine.commit(final=True)
        row=self.engine.document['segments'][0]
        self.assertEqual((row['text'],row['speaker'],row['speaker_candidates']),('en retained words','unassigned',[]))
        self.assertEqual(row['transcription_review']['reason'],'short_acoustic_context')
        self.assertFalse(row['voice_eligible']);self.assertEqual(self.engine.cursor,.1)
    def test_successful_prior_language_reaches_abstaining_asr_across_calls_and_unknown_overlap(self):
        self.decode(0)
        self.models.detector.detect.side_effect=RuntimeError('CPU detector abstention')
        for start,names in [(1,()),(2,('speaker_0','speaker_1'))]:
            row=self.decode(start,names=names)[0]
            self.assertEqual((row['text'],row['language_detection']['reason']),('en retained words','recent_context'))
        self.assertEqual(self.models.seeds[-1],{'language':'en','end_sample':RATE})
    def test_import_worker_keeps_absolute_context_between_crops_without_nvidia_turns(self):
        import inference_worker
        from test_language_detection import model_modules
        detector=Mock();detector.detect.side_effect=[{'en':.99,'fr':.01},{'en':.99,'fr':.01},
            RuntimeError('CPU detector abstention'),RuntimeError('CPU detector abstention')]
        cfg={'diar_path':'cpu','cohere_path':'cpu','lid_path':'cpu','diar_python':sys.executable,
            'asr_python':sys.executable,'diar_kind':'nemotron','device':'mlx'}
        checkpoint=Path(self.temp.name)/'cpu-checkpoint';checkpoint.mkdir()
        (checkpoint/'model.safetensors').touch();cfg['cohere_path']=str(checkpoint)
        def worker(python,task,request,folder):
            return {'turns':[]} if task=='diarize' else inference_worker.run(task,request)
        original=self.path.read_bytes()
        with patch.dict(sys.modules,model_modules(self.models.asr)),patch('inference_worker.check_memory'), \
                patch('inference_worker.importlib.metadata.version',return_value='CPU peer'), \
                patch('language_detection.WhisperLanguageDetector',return_value=detector), \
                patch('pipeline.preflight',return_value=[]),patch('pipeline.run_worker',side_effect=worker):
            doc=infer(self.path,12,'auto',Path(self.temp.name),lambda _:None,cfg)
        self.assertEqual([(s['start'],s['end']) for s in doc['segments']],[(0,6),(6,12)])
        self.assertTrue(all(s['text']=='en retained words' and not s['voice_eligible'] for s in doc['segments']))
        self.assertEqual(doc['segments'][1]['language_detection']['context_end_sample'],6*RATE)
        self.assertEqual(self.path.read_bytes(),original);self.assertFalse((Path(self.temp.name)/'crops').exists())
    def test_epoch_expiry_future_and_failed_switch_do_not_reuse_old_language(self):
        self.decode(0)
        self.models.detector.detect.side_effect=RuntimeError('CPU detector abstention')
        for start,epoch in [(0,0),(62,0),(1,1)]:
            row=self.decode(start,epoch)[0];self.assertEqual(row['text'],'')
            self.assertEqual(row['language_detection']['reason'],'needs_language')
        self.models.detector.detect.side_effect=None;self.models.detector.detect.return_value={'fr':.99,'en':.01}
        self.models.asr.transcribe.side_effect=RuntimeError('CPU ASR failure')
        row=self.decode(1)[0];self.assertEqual(row['transcription_review']['reason'],'transcription_failed')
        self.models.detector.detect.side_effect=RuntimeError('CPU detector abstention')
        row=self.decode(2)[0];self.assertEqual(row['text'],'');self.assertIsNone(self.models.seeds[-1])
    def test_manual_mode_overrides_detector_even_without_speaker_or_with_overlap(self):
        self.models.detector.detect.side_effect=RuntimeError('must not run')
        for names in [(),('speaker_0','speaker_1')]:
            row=self.decode(language='fr',names=names)[0];self.assertEqual(row['text'],'fr retained words')
        self.models.detector.detect.assert_not_called()
    def test_dispatched_context_retains_same_language_failure_but_clears_contradictions(self):
        good={'id':'good','start':0,'end':1,'text':'Established English','language':'en',
            'language_epoch':0,'language_detection':{'mode':'auto','reason':'detected'},'speaker':'speaker_0'}
        for language,reason,expected in [('en','detected',{'language':'en','end_sample':RATE}),
                ('fr','detected',None),(None,'unsupported',None),('stale-fr','detected',{'language':'en','end_sample':2*RATE})]:
            with self.subTest(language=language,reason=reason):
                failed={**good,'id':'failed','start':1,'end':2,'text':'','language':language,
                    'language_detection':{'mode':'auto','reason':reason},'transcription_review':{'reason':'transcription_failed'}}
                current=[good,failed];stored={'good':good,'failed':failed}
                if language=='stale-fr':
                    merged={**good,'end':2,'text':'Merged complete English'}
                    stale={**good,'id':'failed','start':1,'end':2,'language':'fr','text':'Superseded French split'}
                    current=[merged];stored={'good':merged,'failed':stale}
                class Ledger:
                    def __init__(self):
                        self.lock=threading.RLock();self.processed=self.duration=3;self.refining_saved=True;self.requests=[]
                        self.job={'id':'fixture','language':'auto','duration':3,'language_epoch':0,'document':{'speakers':{'speaker_0':'Speaker 1'},'segments':current},
                            'rolling_sources':stored,'rolling_refinement':{'version':1,'completed_sample':2*RATE,'pending':[],
                                'last_scheduled_at':0,'cancelled':False}}
                    def get(self,jid):return copy.deepcopy(self.job)
                    def put(self,job):self.job=copy.deepcopy(job)
                    def queue_worker(self,request):self.requests.append(request);return True
                ledger=Ledger();controller=RefinementController(ledger);controller.enabled=True
                controller.schedule('fixture',force=True)
                self.assertEqual(ledger.requests[0]['language_context'],expected)
    def test_historical_refinement_uses_explicit_preceding_snapshot_not_future_resident_context(self):
        self.decode(9)
        self.decode(70)
        self.models.detector.detect.side_effect=RuntimeError('CPU detector abstention')
        request={'operation_id':'historical','language':'auto','language_epoch':0,'references':[],
            'language_context':{'language':'fr','end_sample':10*RATE},
            'window':{'id':'historical','start_sample':11*RATE,'end_sample':12*RATE,
                'context_start_sample':11*RATE,'context_end_sample':12*RATE}}
        row=self.engine.refine(request)['candidates'][0]
        self.assertEqual((row['text'],row['speaker']),('fr retained words','unassigned'))
        self.assertFalse(row['voice_eligible']);self.assertEqual(row['language_detection']['context_end_sample'],10*RATE)
        request['language_context']=None
        row=self.engine.refine(request)['candidates'][0]
        self.assertEqual(row['text'],'');self.assertEqual(row['language_detection']['reason'],'needs_language')
    def test_digital_silence_is_internal_coverage_without_enrollment_or_export_cues(self):
        with wave.open(str(self.path),'wb') as f:
            f.setparams((1,2,RATE,0,'NONE','not compressed'));f.writeframes(b'\x00\x00'*3*RATE)
        self.engine.received=3*RATE;self.engine.processed=3;self.engine.turns=[{'start':0,'end':3,'speaker':'speaker_0'}]
        self.engine.commit(final=True);row=self.engine.document['segments'][0]
        self.assertEqual(row['audio_state'],'digital_silence');self.assertFalse(row['voice_eligible'])
        self.assertFalse(speaker_audio_eligible({**row,'voice_eligible':True},'speaker_0'))
        self.models.asr.transcribe.assert_not_called();self.models.detector.detect.assert_not_called()
        for kind in ('txt','srt','vtt'):
            self.assertNotIn('00:00:',export(self.engine.document,kind)[0])
        self.assertIn('digital_silence',export(self.engine.document,'json')[0])
    def test_long_silence_crosses_context_caps_without_fake_boundary_failures(self):
        with wave.open(str(self.path),'wb') as f:
            f.setparams((1,2,RATE,0,'NONE','not compressed'));f.writeframes(b'\x00\x00'*40*RATE)
        self.engine.received=40*RATE;self.engine.processed=40
        self.engine.turns=[{'start':0,'end':40,'speaker':'speaker_0'}];self.engine.commit(final=True)
        self.assertEqual(self.engine.cursor,40)
        self.assertTrue(all(s.get('audio_state')=='digital_silence' and not s.get('transcription_review')
            and not s['voice_eligible'] for s in self.engine.document['segments']))
        self.assertFalse(any(e['type']=='boundary_candidate' for e in self.events))
        self.models.asr.transcribe.assert_not_called();self.models.detector.detect.assert_not_called()
    def test_blank_async_revision_retains_protected_edit_and_silence_finishes_without_unresolved_error(self):
        old={'id':'edited','start':0,'end':1,'speaker':'speaker_0','text':'User words','protected_fields':['text']}
        blank={'id':'blank','start':1,'end':2,'speaker':'speaker_0','text':''}
        document={'speakers':{'speaker_0':'Speaker 1'},'segments':[old,blank]}
        rows=[{'start':0,'end':1,'speaker':'speaker_0','text':'','audio_state':'digital_silence'},
            {'start':1,'end':2,'speaker':'speaker_0','text':'','audio_state':'digital_silence'}]
        result=reconcile_window(document,{'id':'w','start_sample':0,'end_sample':2*RATE},
            {s['id']:segment_version(s) for s in document['segments']},rows)
        self.assertEqual(result['document']['segments'][0],old)
        self.assertEqual(result['document']['segments'][1]['refinement_state'],'refined')
        text=export(result['document'],'txt')[0]
        self.assertIn('User words',text);self.assertEqual(len(text.strip().splitlines()),1)


if __name__=='__main__':unittest.main()
