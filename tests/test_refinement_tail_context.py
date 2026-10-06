"""CPU ownership/context regressions; injected text is not acoustic AI evidence."""
import copy
from pathlib import Path
import sys,tempfile,types,unittest,wave
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from live_refinement import Engine,Inbox,coalesce_blanks
from language_detection import SpeechTranscriber
from rolling_refinement import RATE,reconcile_window,segment_version,successful_coverage,successful_non_speech

class RefinementTailContextTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'audio.wav';self.end=4.981125
        pcm=np.tile(np.array([1,-1],dtype='<i2'),round(self.end*RATE)//2)
        with wave.open(str(self.path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(RATE);f.writeframes(pcm.tobytes())
        self.models=Mock();self.models.batch_turns.return_value=[{'start':0,'end':4.74,'speaker':'speaker_0'}]
        self.events=[];self.engine=Engine({'audio_path':str(self.path),'language':'en'},self.models,self.events.append,Inbox())
    def tearDown(self):self.temp.cleanup()
    def request(self,start=0,epoch=0,language_start=0):
        return {'operation_id':'r','language_epoch':epoch,'language':'en','language_start_sample':round(language_start*RATE),
                'window':{'id':'w','start_sample':round(start*RATE),'end_sample':round(self.end*RATE),
                          'context_start_sample':round(max(0,start-3)*RATE),'context_end_sample':round(self.end*RATE)},
                'references':[{'start':0,'end':4.74,'speaker_candidates':['speaker_0']}]}
    def response(self,texts,partial=False):
        sequence=iter(texts)
        def decode(pcm,*args,**kwargs):
            return [{'start':0,'end':len(pcm)/RATE,'language':'en','text':next(sequence),'review':partial,
                     **({'transcription_review':{'reason':'token_limit','partial_text':True}} if partial else {})}]
        self.models.transcribe.side_effect=decode
    def test_actual_failure_shape_refines_tail_with_same_origin_context(self):
        self.response(['The blue folder contains seven pages.','library before six oclock.','library before six oclock.'])
        result=self.engine.refine(self.request());tail=result['candidates'][-1]
        self.assertEqual(tail['text'],'');self.assertEqual(tail['audio_state'],'context_no_new_words')
        self.assertEqual((tail['start'],tail['end']),(4.74,self.end));self.assertTrue(successful_coverage(tail))
        self.assertFalse(successful_non_speech(tail));self.assertFalse(tail['voice_eligible'])
        self.assertEqual([len(c.args[0]) for c in self.models.transcribe.call_args_list],[round(4.74*RATE),3*RATE,round(3.241125*RATE)])
    def test_tail_only_owned_window_uses_left_context_without_publishing_prefix(self):
        self.models.batch_turns.return_value=[{'start':0,'end':3,'speaker':'speaker_0'}]
        self.response(['library before six oclock.','library before six oclock.'])
        result=self.engine.refine(self.request(4.74))
        self.assertEqual(len(result['candidates']),1);tail=result['candidates'][0]
        self.assertEqual((tail['start'],tail['end'],tail['text']),(4.74,self.end,''))
        self.assertEqual(tail['context_evidence']['start_sample'],round(1.74*RATE))
        old=dict(tail,id='old',text='you',machine_revision=1);old.pop('context_evidence');old.pop('audio_state')
        doc={'speakers':{},'segments':[old]};win=self.request(4.74)['window']
        merged=reconcile_window(doc,win,{'old':segment_version(old)},[tail])
        self.assertEqual(merged['document']['segments'][0]['text'],'');self.assertEqual(merged['previous_revision']['segments'],[old])
        protected={**old,'protected_fields':['text']};doc['segments']=[protected]
        self.assertEqual(reconcile_window(doc,win,{'old':segment_version(protected)},[tail])['document'],doc)
    def test_new_genuine_suffix_remains_owned_and_visible(self):
        self.models.batch_turns.return_value=[{'start':0,'end':3,'speaker':'speaker_0'}]
        self.response(['Earlier words.','Earlier words. Blue.'])
        tail=self.engine.refine(self.request(4.74))['candidates'][0]
        self.assertEqual(tail['text'],'Blue.');self.assertEqual((tail['start'],tail['end']),(4.74,self.end))
    def test_language_epoch_prevents_context_crossing(self):
        self.engine.inbox.latest_epoch=1
        self.models.batch_turns.return_value=[{'start':0,'end':3,'speaker':'speaker_0'}]
        self.response(['Blue.'])
        tail=self.engine.refine(self.request(4.74,epoch=1,language_start=4.74))['candidates'][0]
        self.assertEqual(tail['text'],'Blue.');self.assertEqual(self.models.transcribe.call_count,1)
        self.assertEqual(len(self.models.transcribe.call_args.args[0]),round(.241125*RATE))
    def test_partial_decoder_does_not_prove_an_empty_suffix(self):
        self.models.batch_turns.return_value=[{'start':0,'end':3,'speaker':'speaker_0'}]
        self.response(['Earlier words.','Earlier words.'],partial=True)
        tail=self.engine.refine(self.request(4.74))['candidates'][0]
        self.assertEqual(tail['transcription_review']['reason'],'refinement_incomplete');self.assertFalse(successful_coverage(tail))
    def test_successful_prefix_context_applies_only_after_original_probe_endpoint(self):
        class Detector:
            def detect(self,pcm):return {'en':.99,'fr':.01} if len(pcm)>=RATE else {'en':.4,'fr':.6}
            def no_speech_probability(self,pcm):return .1
        class Models:
            def __init__(self):
                self.asr=Mock();self.asr.transcribe.return_value=types.SimpleNamespace(text='Earlier words.',tokens=[1])
            def set_language_context(self,context,start):self.context=context;self.start=start
            def transcribe(self,pcm,language,names,overlap=False):
                return SpeechTranscriber(self.asr,language,detector=Detector(),context=self.context).transcribe(
                    pcm,RATE,tuple(names),start_sample=self.start,max_asr_seconds=24.5)
        models=Models();engine=Engine({'audio_path':str(self.path),'language':'auto'},models,self.events.append,Inbox())
        row=engine.transcribe_owned(4,4.2,{'epoch':0,'language':'auto'},['speaker_0'])[0]
        self.assertEqual(row['text'],'');self.assertTrue(successful_coverage(row))
        self.assertEqual(models.asr.transcribe.call_count,2)
        self.assertEqual([len(c.args[0]) for c in models.asr.transcribe.call_args_list],[3*RATE,round(3.2*RATE)])
        self.assertEqual(row['context_evidence']['prefix_end_sample'],4*RATE)
    def test_real_speaker_switch_does_not_borrow_previous_speaker_words(self):
        self.models.batch_turns.return_value=[{'start':0,'end':4.74,'speaker':'speaker_0'},
            {'start':4.74,'end':self.end,'speaker':'speaker_1'}]
        request=self.request();request['references'].append({'start':4.74,'end':self.end,'speaker_candidates':['speaker_1']})
        self.response(['Previous speaker.','Blue.'])
        tail=self.engine.refine(request)['candidates'][-1]
        self.assertEqual(tail['text'],'Blue.');self.assertEqual(tail['speaker_candidates'],['speaker_1'])
        self.assertEqual(self.models.transcribe.call_count,2)
    def test_successful_empty_asr_finishes_blank_inspection_without_erasing_words(self):
        candidate={'start':0,'end':1,'speaker':'unassigned','text':'','review':True,
                   'acoustic_evidence':{'source':'whisper_sot','no_speech_probability':.88},
                   'transcription_review':{'reason':'empty_result','partial_text':False}}
        win={'id':'w','start_sample':0,'end_sample':RATE}
        for text in ('','Previous usable words.'):
            old={**candidate,'id':'old','text':text,'machine_revision':1};doc={'speakers':{},'segments':[old]}
            result=reconcile_window(doc,win,{'old':segment_version(old)},[candidate])
            if text:self.assertEqual(result['document'],doc)
            else:
                row=result['document']['segments'][0];self.assertEqual(row['refinement_state'],'refined')
                self.assertEqual(row['transcription_review']['reason'],'empty_result')
            self.assertFalse(successful_coverage(candidate));self.assertFalse(successful_non_speech(candidate))
        for reason in ('transcription_failed','insufficient_acoustic_context','token_limit'):
            failed={**candidate,'transcription_review':{'reason':reason,'partial_text':False}}
            result=reconcile_window({'speakers':{},'segments':[]},win,{},[failed])
            self.assertEqual(result['document']['segments'][0]['refinement_state'],'unresolved')
    def test_context_observations_not_coalesced_or_accepted_with_wrong_bounds(self):
        self.models.batch_turns.return_value=[{'start':0,'end':3,'speaker':'speaker_0'}]
        self.response(['Earlier words.','Earlier words.']);tail=self.engine.refine(self.request(4.74))['candidates'][0]
        corrupted=copy.deepcopy(tail);corrupted['context_evidence']['end_sample']+=1
        self.assertFalse(successful_coverage(corrupted));self.assertTrue(successful_coverage(tail))
        self.assertEqual(coalesce_blanks([tail,copy.deepcopy(tail)]),[tail,tail])

if __name__=='__main__':unittest.main()
