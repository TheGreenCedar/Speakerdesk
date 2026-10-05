"""CPU regression boundaries: failed speech stays visible; retry never erases edits."""
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from language_detection import SpeechTranscriber
from pipeline import infer, retry_passage
from review import retain_unassigned_audio, review_reason
from transcript import export, validate


class PassageReviewTests(unittest.TestCase):
    def test_asr_empty_failure_and_token_limit_retain_neighbor_words_and_exact_times(self):
        for second,reason,text in [(types.SimpleNamespace(text='',tokens=[]),'empty_result',''),
                (RuntimeError('Synthetic per-passage failure'),'transcription_failed',''),
                (types.SimpleNamespace(text='Usable partial words',tokens=[1]*448),'token_limit','Usable partial words')]:
            with self.subTest(reason=reason):
                asr=Mock();asr.transcribe.side_effect=[types.SimpleNamespace(text='Reliable first passage',tokens=[1]),second,
                                                    types.SimpleNamespace(text='Reliable last passage',tokens=[1])]
                detector=Mock();detector.detect.side_effect=[{'en':.99,'fr':.01},{'fr':.99,'en':.01},{'en':.99,'fr':.01}]
                with patch('language_detection.WhisperLanguageDetector',return_value=detector):
                    transcriber=SpeechTranscriber(asr,'auto','cpu-fixture')
                rows=transcriber.transcribe(np.full(9*16000,.1,dtype=np.float32),16000,('speaker_0',))
                self.assertEqual([(r['start'],r['end']) for r in rows],[(0,3),(3,6),(6,9)])
                self.assertEqual([r['text'] for r in rows],['Reliable first passage',text,'Reliable last passage'])
                self.assertEqual(rows[1]['transcription_review']['reason'],reason)
                self.assertTrue(rows[1]['review']);self.assertFalse(rows[0]['review']);self.assertFalse(rows[2]['review'])

    def test_unknown_overlap_partial_and_unassigned_are_explicit_in_each_export(self):
        rows=[dict(id='known',start=0.,end=2.,speaker='speaker_0',text='Keep these words',review=False),
              dict(id='unknown',start=2.,end=4.,speaker='speaker_0',text='',language=None,
                   language_detection={'mode':'auto','reason':'uncertain'},review=True),
              dict(id='partial',start=4.,end=6.,speaker='speaker_0',text='Partial words',
                   transcription_review={'reason':'token_limit'},review=True),
              dict(id='mixed',start=7.,end=9.,speaker='overlap',text='',speaker_candidates=['speaker_0','speaker_1'],review=True)]
        doc=retain_unassigned_audio({'speakers':{'speaker_0':'Speaker 1','overlap':'Overlapping speakers'},'segments':rows},10)
        validate(doc,10)
        before=copy.deepcopy(doc);retain_unassigned_audio(doc,10);self.assertEqual(doc,before)
        for kind in ('txt','srt','vtt'):
            text=export(doc,kind)[0]
            for marker in ('Keep these words','Partial words','Language uncertain','Transcript may be incomplete',
                           'voices are not separated','may be silence or missed speech'):
                self.assertIn(marker,text)
            self.assertIn('00:00:02',text)
        self.assertEqual(json.loads(export(doc,'json')[0]),doc)
        doc['segments'][1].update(text='Manually checked words',review_resolution='words_reviewed')
        self.assertEqual(review_reason(doc['segments'][1]),'')
        mixed=next(s for s in doc['segments'] if s['id']=='mixed')
        mixed.update(text='Heard words',review_resolution='words_reviewed')
        self.assertIn('not separated',review_reason(mixed))

    def test_no_nvidia_turns_return_audio_review_without_calling_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);audio=folder/'audio.wav';audio.write_bytes(b'not-read-with-no-crops')
            cfg={'diar_kind':'nemotron','diar_path':'cpu-fixture','diar_python':sys.executable,'device':'mlx'}
            with patch('pipeline.preflight',return_value=[]),patch('pipeline.run_worker',return_value={'turns':[]}) as worker:
                doc=validate(infer(audio,70,'auto',folder,lambda _:None,cfg),70)
            self.assertEqual(worker.call_count,1)
            self.assertEqual([(s['start'],s['end']) for s in doc['segments']],[(0,30),(30,60),(60,70)])
            self.assertTrue(all(not s['voice_eligible'] and s['confidence'] is None for s in doc['segments']))
            self.assertTrue(all('may be silence' in review_reason(s) for s in doc['segments']))
            self.assertFalse((folder/'crops').exists())

    def test_manual_retry_crops_only_requested_audio_and_bypasses_detector_and_diarizer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);audio=folder/'audio.wav'
            with wave.open(str(audio),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\x01\x00'*16000*10)
            model=folder/'cohere';model.mkdir();(model/'config.json').write_text('{"model_type":"cohere_asr"}');(model/'model.safetensors').touch()
            cfg={'cohere_path':str(model),'asr_python':sys.executable,'device':'mlx'}
            def worker(python,task,request,destination):
                self.assertEqual(task,'transcribe');self.assertEqual(request['language'],'fr')
                self.assertNotIn('lid_path',request)
                with wave.open(request['chunks'][0]['audio']) as f:self.assertEqual(f.getnframes(),16000*3)
                return {'regions':[[dict(start=0.,end=3.,text='Original French words',language='fr',review=False)]]}
            original=audio.read_bytes()
            with patch('pipeline.run_worker',side_effect=worker):
                result=retry_passage(audio,4.,7.,'fr',folder,cfg)
            self.assertEqual((result['start'],result['end']),(4.,7.));self.assertEqual(audio.read_bytes(),original)
            self.assertEqual(list(folder.glob('passage-retry-*')),[])
            with self.assertRaises(ValueError):retry_passage(audio,0.,3.,'auto',folder,cfg)


class PassageRetryAPITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.app=create_app(self.root);self.client=self.app.test_client()
        self.headers={'X-Speakerdesk-Token':re.search(r'name="speakerdesk-token" content="([^"]+)"',self.client.get('/').text)[1]}
        self.ext=self.app.extensions['speakerdesk'];self.jid='d'*32
        folder=self.root/self.jid;folder.mkdir();(folder/'audio.wav').write_bytes(b'original audio fixture')
        self.original={'id':self.jid,'created':1.,'name':'CPU retry fixture','status':'ready','duration':10.,'revision':4,
                      'document':{'speakers':{'speaker_0':'Speaker 1'},'segments':[
                          dict(id='partial',start=0.,end=3.,speaker='speaker_0',text='Edited reliable words',review=True),
                          dict(id='neighbor',start=4.,end=7.,speaker='speaker_0',text='Neighbor must remain',review=False)],
                          'provenance':{'kind':'local_inference'}}}
        self.ext['recognition'].put(copy.deepcopy(self.original))
    def tearDown(self):
        self.ext['meetings'].close();self.ext['executor'].shutdown(wait=True,cancel_futures=True);self.temp.cleanup()
    def job(self):return self.client.get('/api/jobs/'+self.jid).json
    def post(self,**changes):
        return self.client.post('/api/jobs/'+self.jid+'/retry',headers=self.headers,
               json=dict(revision=4,segment_id='partial',language='fr',**changes))

    def test_retry_requires_token_explicit_language_matching_revision_and_existing_audio(self):
        self.assertEqual(self.client.post('/api/jobs/'+self.jid+'/retry',json={}).status_code,403)
        for body,code in [({'revision':4,'segment_id':'partial','language':'auto'},400),
                          ({'revision':3,'segment_id':'partial','language':'fr'},409),
                          ({'revision':4,'segment_id':'missing','language':'fr'},404)]:
            with patch('app.retry_passage') as worker:
                self.assertEqual(self.client.post('/api/jobs/'+self.jid+'/retry',headers=self.headers,json=body).status_code,code)
                worker.assert_not_called()
        (self.root/self.jid/'audio.wav').unlink();self.assertEqual(self.post().status_code,409)

    def test_retry_candidate_is_persisted_without_replacing_user_or_neighbor_text(self):
        started=threading.Event();release=threading.Event()
        def compute(*args):
            started.set();self.assertTrue(release.wait(3))
            return dict(text='New candidate words',review=False,language='fr')
        try:
            with patch('app.retry_passage',side_effect=compute):
                self.assertEqual(self.post().status_code,202);self.assertTrue(started.wait(1))
                self.assertEqual(self.post().status_code,409)
                self.assertEqual(self.client.put('/api/jobs/'+self.jid+'/transcript',headers=self.headers,
                      json={'revision':4,'document':self.original['document']}).status_code,409)
                self.assertEqual(self.job()['document'],self.original['document'])
                release.set();self.ext['executor'].submit(lambda:None).result(timeout=3)
        finally:release.set()
        saved=self.job();self.assertEqual(saved['status'],'ready');self.assertEqual(saved['revision'],5)
        self.assertEqual([s['text'] for s in saved['document']['segments']],['Edited reliable words','Neighbor must remain'])
        candidate=saved['document']['segments'][0]['retry_candidate']
        self.assertEqual(candidate['text'],'New candidate words');self.assertEqual(candidate['original_text'],'Edited reliable words')
        text=self.client.get('/api/jobs/'+self.jid+'/export/txt').text
        self.assertIn('Edited reliable words',text);self.assertNotIn('New candidate words',text)
        self.assertEqual((self.root/self.jid/'audio.wav').read_bytes(),b'original audio fixture')
        reopened=create_app(self.root)
        try:self.assertEqual(reopened.test_client().get('/api/jobs/'+self.jid).json['document'],saved['document'])
        finally:reopened.extensions['speakerdesk']['meetings'].close();reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_failed_retry_returns_reviewable_error_and_preserves_all_words(self):
        with patch('app.retry_passage',side_effect=RuntimeError('Synthetic native error')),self.assertLogs(self.app.logger,level='ERROR'):
            self.assertEqual(self.post().status_code,202);self.ext['executor'].submit(lambda:None).result(timeout=3)
        saved=self.job();self.assertEqual(saved['status'],'ready')
        self.assertEqual([s['text'] for s in saved['document']['segments']],['Edited reliable words','Neighbor must remain'])
        self.assertIn('retained',saved['document']['segments'][0]['retry_candidate']['error'])

    def test_candidate_is_removed_on_timing_track_edits_and_cannot_be_forged(self):
        for change in ({'start':.5},{'speaker':'speaker_1'}):
            with self.subTest(change=change):
                job=copy.deepcopy(self.original);job['document']['speakers']['speaker_1']='Speaker 2'
                job['document']['segments'][0]['retry_candidate']={'id':'server-owned','start':0.,'end':3.,
                    'speaker':'speaker_0','text':'Earlier audio words','language':'fr','original_text':'Edited reliable words'}
                self.ext['recognition'].put(job)
                incoming=copy.deepcopy(job['document']);incoming['segments'][0].update(change)
                response=self.client.put('/api/jobs/'+self.jid+'/transcript',headers=self.headers,json={'revision':4,'document':incoming})
                self.assertEqual(response.status_code,200)
                self.assertNotIn('retry_candidate',response.json['document']['segments'][0])
        self.ext['recognition'].put(copy.deepcopy(self.original));incoming=copy.deepcopy(self.original['document'])
        incoming['segments'][0]['retry_candidate']={'text':'Forged model candidate'}
        response=self.client.put('/api/jobs/'+self.jid+'/transcript',headers=self.headers,json={'revision':4,'document':incoming})
        self.assertEqual(response.status_code,200);self.assertNotIn('retry_candidate',response.json['document']['segments'][0])

if __name__=='__main__':unittest.main()
