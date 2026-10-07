"""Real canonical/host recovery with fabricated CPU acoustic peers, not ASR proof."""
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from language_detection import SpeechTranscriber
from live_refinement import Engine, Inbox
from meeting_refinement import activity_references
from speech_admission import FRAME, INPUT_POLICY, SpeechFrames
from utterances import complete_non_speech

RATE=16000


class EvidencePeer:
    def __init__(self,frames,mutate=None):self.frames,self.mutate=frames,mutate
    def admission(self,start,end):
        receipt=self.frames.admission(start,end)
        if self.mutate:self.mutate(receipt)
        return receipt


class ModelsPeer:
    def __init__(self,raw,kind='negative'):
        self.received=0;self.start=0;self.kind=kind
        self.speech_live=SimpleNamespace(evidence=SpeechFrames());self.speech_historical=None
        self.asr=Mock();self.asr.transcribe.return_value=SimpleNamespace(text=raw,tokens=[1,2])
        self.detector=Mock();self.detector.detect.return_value={'en':.99,'fr':.01}
    def feed(self,audio,final=False):
        first=self.received;self.received+=len(audio)
        ledger=self.speech_live.evidence;end=self.received if final else self.received//FRAME*FRAME
        while ledger.end_sample<end:
            a=ledger.end_sample
            ledger.append(a,min(a+FRAME,end),.8,observation={'input_policy':INPUT_POLICY})
        turns=[dict(start=first/RATE,end=self.received/RATE,speaker='speaker_0')] if len(audio) else []
        return turns,end/RATE
    def set_language_context(self,context,start):self.start=start
    def transcribe(self,audio,language,names,overlap=False):
        evidence=self.speech_historical if self.speech_historical is not None else self.speech_live.evidence
        return SpeechTranscriber(self.asr,language,detector=self.detector,speech_evidence=evidence).transcribe(
            audio,RATE,tuple(names),max_asr_seconds=24.5,start_sample=self.start)
    def begin_refinement(self,audio,start):
        frames=SpeechFrames(start)
        for a in range(start,start+len(audio),FRAME):
            frames.append(a,min(a+FRAME,start+len(audio)),.8 if self.kind in ('positive','empty_asr','failed_asr') else .01,
                          observation={'input_policy':INPUT_POLICY})
        mutations={
            'partial':lambda e:e.update(end_sample=e['end_sample']-1),
            'pending':lambda e:e.update(complete=False,decision='pending'),
            'uncertain':lambda e:e.update(decision='uncertain',uncertain_regions=[dict(start_sample=start,end_sample=start+len(audio))]),
            'old_policy':lambda e:e.pop('input_policy',None),
            'wrong_model':lambda e:e.update(model_revision='wrong'),
            'superset':lambda e:e.update(start_sample=max(0,start-1),end_sample=e['end_sample']+1),
        }
        self.speech_historical=EvidencePeer(frames,mutations.get(self.kind))
        if self.kind=='missing':self.speech_historical=None
        if self.kind=='failed_evidence':
            self.speech_historical=Mock();self.speech_historical.admission.side_effect=RuntimeError('CPU evidence failure')
        if self.kind=='empty_asr':self.asr.transcribe.return_value=SimpleNamespace(text='',tokens=[])
        if self.kind=='failed_asr':self.asr.transcribe.side_effect=RuntimeError('CPU ASR failure')
    def end_refinement(self):self.speech_historical=None
    def batch_turns(self,audio):return []
    def metrics(self):return {}


class CanonicalNegativeRecoveryTests(unittest.TestCase):
    def run_path(self,raw,*,mode='en',kind='negative',protect_worker=False,protect_host=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'audio.wav'
            pcm=(np.sin(np.arange(6*RATE)*.017)*12).astype('<i2')
            with wave.open(str(path),'wb') as output:
                output.setparams((1,2,RATE,0,'NONE','none'));output.writeframes(pcm.tobytes())
            peer=ModelsPeer(raw,kind);events=[];jid='a'*32
            engine=Engine(dict(audio_path=str(path),language=mode,job_id=jid,canonical_utterances=True),peer,events.append,Inbox())
            for second in range(6):
                engine.handle(dict(type='audio',start_sample=second*RATE,end_sample=(second+1)*RATE,language=mode,language_epoch=0))
            engine.handle(dict(type='stop'))
            row=[e['candidate'] for e in events if e['type']=='canonical_revision'][-1]
            if protect_worker:
                engine.canonical.book.edit(row['id'],row['canonical_machine_revision'],'Human correction retained.')
                row=engine.canonical.project(engine.canonical.book.rows[row['id']])
            # Obsolete timing must be invalidated by the actual machine revision.
            book_row=engine.canonical.book.rows[row['id']]
            for key in ('alignment','reading_word_evidence','assembly_provenance','bounded_decode_provenance'):
                book_row[key]={'obsolete':True}
            calls=peer.asr.transcribe.call_count
            request=dict(type='refine',canonical=row,operation_id='negative-recheck',language_epoch=0,language=mode,
                window=dict(id=row['id'],start_sample=row['start_sample'],end_sample=row['end_sample']),
                references=activity_references([row],row['start_sample'],row['end_sample']))
            result=engine.refine(request);candidate=result['canonical_candidate']
            journal=[json.loads(line) for line in engine.canonical.archive.path.read_text().splitlines()]
            current_book=copy.deepcopy(engine.canonical.book.rows[row['id']])
            app=create_app(root/'host');manager=app.extensions['speakerdesk']['meetings'];manager.duration=6
            job=dict(id=jid,created=1,status='ready',kind='meeting',name='CPU negative recovery',language=mode,
                duration=6,revision=0,canonical_utterances=True,document=dict(speakers={},segments=[],provenance={},warnings=[]))
            try:
                manager.refinement.initialize(job);manager.put(job)
                manager.refinement.canonical(jid,dict(candidate=row,fast_sequence=1))
                if protect_host:
                    saved=manager.get(jid);saved_row=saved['document']['segments'][0]
                    saved_row.update(text='Human correction retained.',protected_fields=['text']);manager.put(saved)
                manager.refinement.canonical(jid,dict(candidate=candidate,fast_sequence=2))
                client=app.test_client()
                headers={'X-Speakerdesk-Token':re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').text)[1]}
                response=client.get('/api/jobs/'+jid,headers=headers)
                self.assertEqual(response.status_code,200,response.json)
                actual=response.json['document']['segments'][0]
                saved=manager.get(jid)
                self.assertEqual(saved['document']['segments'][0],actual)
                reopened=create_app(root/'host')
                try:
                    fresh=reopened.test_client()
                    token=re.search(r'name="speakerdesk-token" content="([^"]+)"',fresh.get('/').text)[1]
                    persisted=fresh.get('/api/jobs/'+jid,headers={'X-Speakerdesk-Token':token})
                    self.assertEqual(persisted.status_code,200,persisted.json)
                    self.assertEqual(persisted.json['document']['segments'][0],actual)
                finally:
                    reopened.extensions['speakerdesk']['meetings'].close()
                    reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
                return row,candidate,current_book,actual,journal,saved,peer.asr.transcribe.call_count-calls
            finally:
                manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_exact_negative_clears_machine_words_through_engine_stop_host_api_and_storage(self):
        for raw in ('Thank you. Thank you.','Unrelated fabricated machine words.'):
            for mode in ('en','auto'):
                with self.subTest(raw=raw,mode=mode):
                    before,candidate,book,actual,journal,saved,calls=self.run_path(raw,mode=mode)
                    self.assertEqual(candidate['text'],'');self.assertEqual(actual['text'],'')
                    self.assertEqual(candidate['audio_state'],'model_non_speech');self.assertEqual(calls,0)
                    self.assertEqual(candidate['refinement_state'],'refined')
                    self.assertNotIn('canonical_unresolved',candidate)
                    self.assertEqual(candidate['audio_revision'],before['audio_revision'])
                    self.assertGreater(candidate['canonical_machine_revision'],before['canonical_machine_revision'])
                    self.assertEqual(candidate['text_audio_anchor'],before['text_audio_anchor'])
                    self.assertFalse(book['voice_eligible'])
                    for key in ('alignment','reading_word_evidence','assembly_provenance','bounded_decode_provenance'):
                        self.assertNotIn(key,book)
                    versions=[e['version'] for e in journal if e['type']=='machine_version' and e['version'].get('non_speech_evidence')]
                    self.assertEqual(versions[-1]['previous_text'],raw)
                    history=saved['refinement_history'][before['id']]
                    self.assertEqual(history['segments'][0]['text'],raw)

    def test_human_corrections_survive_negative_worker_and_host_candidates(self):
        for location in ('worker','host'):
            with self.subTest(location=location):
                _,_,_,actual,_,_,calls=self.run_path('Old machine words',protect_worker=location=='worker',protect_host=location=='host')
                self.assertEqual(actual['text'],'Human correction retained.');self.assertEqual(calls,0)

    def test_missing_failed_partial_or_empty_recognition_cannot_clear_prior_words(self):
        for kind in ('partial','pending','uncertain','old_policy','wrong_model','superset','missing',
                     'failed_evidence','empty_asr','failed_asr'):
            with self.subTest(kind=kind):
                _,candidate,_,actual,_,_,_=self.run_path('Retained machine words',kind=kind)
                self.assertEqual(candidate['text'],'Retained machine words');self.assertEqual(actual['text'],'Retained machine words')

    def test_positive_quiet_classification_preserves_legitimate_gratitude(self):
        for mode in ('en','auto'):
            with self.subTest(mode=mode):
                _,candidate,_,actual,_,_,calls=self.run_path('Thank you. Thank you.',mode=mode,kind='positive')
                self.assertEqual(candidate['text'],'Thank you. Thank you.');self.assertEqual(actual['text'],candidate['text'])
                self.assertGreater(calls,0)

    def test_negative_qualification_requires_exact_current_observed_policy(self):
        frames=SpeechFrames()
        frames.append(0,FRAME,.01,observation={'input_policy':INPUT_POLICY})
        row=dict(text='',audio_state='model_non_speech',acoustic_evidence=frames.admission(0,FRAME))
        self.assertTrue(complete_non_speech(row,0,FRAME))
        conditioned=copy.deepcopy(row)
        conditioned['acoustic_evidence']['maximum_model_probability']=.999
        self.assertTrue(complete_non_speech(conditioned,0,FRAME))
        with patch('speech_admission.INPUT_POLICY','future_current_policy'):
            self.assertFalse(complete_non_speech(row,0,FRAME))
            conditioned['acoustic_evidence']['input_policy']='future_current_policy'
            self.assertTrue(complete_non_speech(conditioned,0,FRAME))
        for field,value in (('maximum_probability',float('nan')),('maximum_probability',True),
                            ('maximum_probability',.5),('end_sample',FRAME-1),('complete',False),
                            ('uncertain_regions',[dict(start_sample=0,end_sample=FRAME)])):
            wrong=copy.deepcopy(row);wrong['acoustic_evidence'][field]=value
            self.assertFalse(complete_non_speech(wrong,0,FRAME),(field,value))
        wrong=copy.deepcopy(row);wrong['transcription_review']={'reason':'transcription_failed'}
        self.assertFalse(complete_non_speech(wrong,0,FRAME))


if __name__=='__main__':unittest.main()
