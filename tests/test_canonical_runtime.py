"""Activated resident/host path with explicit CPU peers, not acoustic proof."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from live_refinement import Engine,Inbox
from speech_admission import SpeechFrames, INPUT_POLICY
from app import create_app
from meeting_refinement import activity_references
from live_refinement import align_tracks
RATE=16000

class Peer:
    def __init__(self):
        self.received=0;self.calls=[];self.speech_live=SimpleNamespace(evidence=SpeechFrames())
        self.mixed=False;self.silent_after=None;self.text='  Please review the secs  '
    def feed(self,audio,final=False):
        start=self.received;self.received+=len(audio)
        ledger=self.speech_live.evidence
        end=self.received if final else self.received//512*512
        while ledger.end_sample<end:
            a=ledger.end_sample;b=min(a+512,end)
            ledger.append(a,b,.9 if self.silent_after is None or a<self.silent_after else .01)
        turns=[]
        if start<self.received:
            middle=(start+self.received)//2
            turns=[{'start':start/RATE,'end':middle/RATE,'speaker':'speaker_0'},
                   {'start':middle/RATE,'end':self.received/RATE,'speaker':'speaker_1' if self.mixed else 'speaker_0'}]
        return turns,end/RATE
    def transcribe(self,audio,language,names,overlap=False):
        self.calls.append(len(audio));return [{'start':0,'end':len(audio)/RATE,'text':self.text.strip(),
             'cohere_raw_text':self.text,'language':language if language!='auto' else 'en'}]
    def batch_turns(self,audio):return [{'start':0,'end':len(audio)/RATE,'speaker':'speaker_7'}]
    def metrics(self):return {}

class CanonicalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.path=self.root/'audio.wav'
        with wave.open(str(self.path),'wb') as wav:
            wav.setparams((1,2,RATE,0,'NONE','none'));wav.writeframes(b'\x01\x00'*60*RATE)
        self.peer=Peer();self.events=[];self.inbox=Inbox();self.jid='a'*32
        self.engine=Engine({'audio_path':str(self.path),'language':'en','job_id':self.jid,'canonical_utterances':True},self.peer,self.events.append,self.inbox)
    def tearDown(self):self.temp.cleanup()
    def feed(self,start,end,mode='en',epoch=0):
        for n in range(start,end):self.engine.handle({'type':'audio','start_sample':n*RATE,'end_sample':(n+1)*RATE,'language':mode,'language_epoch':epoch})
    def rows(self):return [event['candidate'] for event in self.events if event['type']=='canonical_revision']
    def test_uncertain_only_audio_survives_final_receipt_without_asr_or_complete_claim(self):
        def negative(audio,final=False):
            self.peer.received+=len(audio);ledger=self.peer.speech_live.evidence
            end=self.peer.received if final else self.peer.received//512*512
            while ledger.end_sample<end:
                a=ledger.end_sample;b=min(a+512,end)
                ledger.append(a,b,.01,observation={'input_policy':INPUT_POLICY,'constant_value':None})
            return [],end/RATE
        self.peer.feed=negative;self.feed(0,1);self.engine.handle({'type':'stop'})
        final=self.events[-1]
        self.assertEqual(final['canonical_uncertain_samples'],RATE)
        self.assertEqual(self.peer.calls,[]);self.assertEqual(self.rows(),[])
        journal=[json.loads(line) for line in self.engine.canonical.archive.path.read_text().splitlines()]
        self.assertEqual(sum(e['end_sample']-e['start_sample'] for e in journal
            if e['type']=='speech_admission_uncertain'),RATE)
        app=create_app(self.root/'uncertain-home');manager=app.extensions['speakerdesk']['meetings']
        manager.duration=1
        job={'id':self.jid,'created':1,'status':'refining','kind':'meeting','name':'CPU uncertain',
             'language':'en','duration':1,'revision':0,'canonical_utterances':True,
             'document':{'speakers':{},'segments':[],'provenance':{},'warnings':[]}}
        try:
            manager.refinement.initialize(job);manager.put(job);manager.refinement.ready(self.jid,True)
            manager.refinement.capture_done(self.jid,observed_sample=RATE,uncertain_samples=RATE)
            actual=manager.get(self.jid)
            self.assertEqual(actual['canonical_uncertain_samples'],RATE)
            self.assertEqual(actual['refinement_status'],'unresolved')
            self.assertNotEqual(actual['rolling_refinement']['completed_sample'],RATE)
        finally:
            manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
    def test_same_identity_raw_unicode_replacement_and_pause_remains_open(self):
        self.feed(0,7);first=self.rows()[-1];self.assertEqual(first['text'],self.peer.text)
        self.peer.text='  Please review the section together. 12.5 go go café 👩🏽‍💻  '
        self.feed(7,10);latest=self.rows()[-1]
        self.assertEqual(latest['id'],first['id']);self.assertEqual(latest['text'],self.peer.text)
        self.assertEqual(latest['text_audio_anchor'],{'start_sample':0,'end_sample':latest['end_sample']})
        self.engine.handle({'type':'flush','request_id':'pause','through_sample':10*RATE})
        self.assertEqual(self.rows()[-1]['canonical_state'],'open');self.assertFalse(self.engine.canonical.book.closed)
        self.feed(10,11);self.engine.handle({'type':'stop'})
        self.assertEqual(self.rows()[-1]['id'],first['id']);self.assertEqual(self.rows()[-1]['canonical_state'],'sealed')
        self.assertEqual(self.rows()[-1]['end_sample'],11*RATE)
    def test_nvidia_turns_never_split_words_or_authorize_mixed_voice(self):
        self.peer.mixed=True;self.feed(0,7);self.engine.handle({'type':'stop'})
        rows=self.engine.canonical.book.snapshot();self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['speaker_candidates'],['speaker_0','speaker_1']);self.assertFalse(rows[0]['voice_eligible'])
        self.assertEqual(rows[0]['text'],self.peer.text)
    def test_nvidia_lookahead_cannot_truncate_retained_utterance_audio(self):
        original=self.peer.feed
        def lagged(audio,final=False):
            turns,end=original(audio,final);return turns,max(0,end-.2)
        self.peer.feed=lagged;self.feed(0,7);self.engine.handle({'type':'stop'})
        row=self.rows()[-1]
        self.assertEqual((row['start_sample'],row['end_sample']),(0,7*RATE))
        self.assertEqual(row['canonical_state'],'sealed');self.assertFalse(row['voice_eligible'])
        self.assertEqual(row['speaker_activity']['regions'][-1]['speakers'],[])
        self.assertEqual(self.events[-1]['canonical_observed_sample'],7*RATE)
    def test_silence_seals_vad_object_without_nvidia_word_partition(self):
        self.peer.silent_after=6*RATE;self.feed(0,8)
        rows=self.engine.canonical.book.snapshot();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['state'],'sealed')
        self.assertLess(rows[0]['end_sample'],7*RATE);self.assertGreaterEqual(rows[0]['end_sample'],6*RATE)
    def test_ordered_language_boundary_seals_old_object_at_retained_audio(self):
        self.feed(0,2);self.engine.handle({'type':'language','generation':1,'language':'fr','start_sample':2*RATE})
        self.feed(2,6,'fr',1);self.engine.handle({'type':'stop'})
        rows=self.engine.canonical.book.snapshot();self.assertEqual(len(rows),2)
        self.assertEqual([(row['start_sample'],row['end_sample'],row['language_epoch']) for row in rows],[(0,2*RATE,0),(2*RATE,6*RATE,1)])
        self.assertEqual([self.engine.canonical.project(row)['language'] for row in rows],['en','fr'])
    def test_long_audio_keeps_prior_text_and_raw_parts_without_lexical_stitching(self):
        with patch('live_refinement.split_same_origin',side_effect=AssertionError('lexical stitching')):
            self.feed(0,50);self.engine.handle({'type':'stop'})
        self.assertEqual(len(self.engine.canonical.book.rows),1)
        row=self.rows()[-1];self.assertEqual(row['end_sample'],50*RATE);self.assertEqual(row['text'],self.peer.text)
        self.assertEqual(row['canonical_unresolved'],'unresolved_alignment');self.assertFalse(row['voice_eligible'])
        self.assertLessEqual(max(self.peer.calls),392000)
        journal=[json.loads(line) for line in self.engine.canonical.archive.path.read_text().splitlines()]
        retained=next(event for event in journal if event['type']=='bounded_machine_version')
        self.assertEqual(len(retained['parts']),3);self.assertTrue(all(part['text']==self.peer.text for part in retained['parts']))
    def test_whole_utterance_refinement_replaces_same_id_and_preserves_anchor(self):
        self.feed(0,7);self.engine.handle({'type':'stop'});row=self.rows()[-1]
        request={'type':'refine','canonical':row,'operation_id':'one','language_epoch':0,
            'language':'en','window':{'id':row['id'],'start_sample':row['start_sample'],'end_sample':row['end_sample']},
            'references':[{'start':0,'end':7,'speaker_candidates':['speaker_0']}]}
        self.peer.text='The entire canonical phrase is refined.'
        result=self.engine.refine(request);candidate=result['canonical_candidate']
        self.assertEqual(candidate['id'],row['id']);self.assertEqual(candidate['audio_revision'],row['audio_revision'])
        self.assertEqual(candidate['refinement_state'],'refined');self.assertEqual(candidate['text'],self.peer.text)
        self.assertEqual(candidate['speaker_candidates'],['speaker_0']);self.assertGreater(result['fast_sequence'],self.events[-2].get('fast_sequence',0))
    def test_host_preserves_correction_and_rejects_stale_anchor(self):
        self.feed(0,6);candidate=self.rows()[-1];app=create_app(self.root/'home');manager=app.extensions['speakerdesk']['meetings']
        job={'id':self.jid,'created':1,'status':'paused','kind':'meeting','name':'CPU','language':'en','duration':6,'revision':0,
             'document':{'speakers':{'speaker_0':'Speaker1'},'segments':[],'provenance':{},'warnings':[]},'canonical_utterances':True}
        manager.refinement.initialize(job);manager.put(job);manager.duration=6
        try:
            manager.refinement.canonical(self.jid,{'candidate':candidate,'fast_sequence':1})
            current=manager.get(self.jid);row=current['document']['segments'][0]
            prior_anchor={'start_sample':candidate['start_sample'],'end_sample':candidate['end_sample']}
            row.update(text='My correction',protected_fields=['text'],machine_revision=5,text_audio_anchor=prior_anchor);manager.put(current)
            changed=copy.deepcopy(candidate);changed.update(text='Machine correction',canonical_machine_revision=candidate['canonical_machine_revision']+1,
                end_sample=candidate['end_sample']+RATE,end=candidate['end']+1,audio_revision=candidate['audio_revision']+1,
                text_audio_anchor={'start_sample':candidate['start_sample'],'end_sample':candidate['end_sample']+RATE})
            manager.duration=7
            manager.refinement.canonical(self.jid,{'candidate':changed,'fast_sequence':2})
            actual=manager.get(self.jid)['document']['segments'][0];self.assertEqual(actual['text'],'My correction');self.assertEqual(actual['refinement_state'],'edited')
            self.assertEqual(actual['text_audio_anchor'],prior_anchor)
            stale=copy.deepcopy(changed);stale['end_sample']-=1;stale['end']=stale['end_sample']/RATE
            with self.assertRaisesRegex(ValueError,'Stale canonical'):manager.refinement.canonical(self.jid,{'candidate':stale,'fast_sequence':3})
            self.assertEqual(manager.get(self.jid)['document']['segments'][0],actual)
        finally:manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
    def test_cancelled_decode_preserves_book_revision_and_raw_candidate_history(self):
        self.feed(0,7);self.engine.handle({'type':'stop'});row=self.rows()[-1]
        original=copy.deepcopy(self.engine.canonical.book.rows[row['id']])
        request={'type':'refine','canonical':row,'operation_id':'cancel-me','language_epoch':0,
            'language':'en','window':{'id':row['id'],'start_sample':0,'end_sample':7*RATE},
            'references':[{'start':0,'end':7,'speaker_candidates':['speaker_0']}]}
        self.peer.text='Candidate arriving after cancellation'
        real=self.peer.transcribe
        def cancelled(*args,**kwargs):
            result=real(*args,**kwargs);self.inbox.cancelled.add('cancel-me');return result
        self.peer.transcribe=cancelled
        result=self.engine.refine(request);self.assertTrue(result['cancelled'])
        self.assertEqual(self.engine.canonical.book.rows[row['id']],original)
        self.assertIn('Candidate arriving after cancellation',self.engine.canonical.archive.path.read_text())
    def test_batch_speaker_mapping_uses_independent_sample_ledger_and_gaps(self):
        row={'canonical_utterance_id':'one','start_sample':0,'end_sample':32000,'audio_revision':2,
            'start':0.,'end':2.,'speaker_candidates':['speaker_0','speaker_1'],
            'speaker_activity':{'audio_revision':2,'regions':[
                {'start_sample':0,'end_sample':12800,'speakers':['speaker_0']},
                {'start_sample':12800,'end_sample':16000,'speakers':[]},
                {'start_sample':16000,'end_sample':32000,'speakers':['speaker_1']}]}}
        refs=activity_references([row],4000,30000)
        self.assertEqual(refs,[{'start':.25,'end':.8,'speaker_candidates':['speaker_0']},
            {'start':.8,'end':1.,'speaker_candidates':[]},{'start':1.,'end':1.875,'speaker_candidates':['speaker_1']}])
        batch=[{'start':.25,'end':.8,'speaker':'speaker_7'},{'start':1.,'end':1.875,'speaker':'speaker_8'}]
        self.assertEqual(align_tracks(batch,refs),{'speaker_7':'speaker_0','speaker_8':'speaker_1'})
        row['speaker_activity']['audio_revision']=1
        self.assertEqual(activity_references([row],0,32000),[])

if __name__=='__main__':unittest.main()
