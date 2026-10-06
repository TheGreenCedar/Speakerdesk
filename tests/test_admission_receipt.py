"""Fabricated CPU neural scores test receipt integrity, never acoustic accuracy."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from admission_receipt import execution,validate_receipt,retained_pcm_digest
from speech_admission import SpeechSession
from app import create_app


class ScorePeer:
    normalized_view=True
    def __init__(self,probability=.01,fail_after=None):
        self.probability=probability;self.calls=0;self.fail_after=fail_after
    def initial_state(self):return None
    def feed(self,chunk,state):
        self.calls+=1
        if self.fail_after is not None and self.calls>self.fail_after:raise RuntimeError('CPU score failure')
        return self.probability,None


class AdmissionReceipts(unittest.TestCase):
    def setUp(self):self.identity=execution('a'*32,'b'*32)
    def receipt(self,session,phase='stop',request_id='stop-one'):
        return {**self.identity,**session.inspection(),'phase':phase,'request_id':request_id}
    def test_original_pcm_hash_and_fixed_counts_survive_packet_grouping(self):
        pcm=np.resize(np.array([.125,-.125],dtype=np.float32),1025)
        hashes=[]
        for size in (33,512,1025):
            session=SpeechSession(ScorePeer())
            for a in range(0,len(pcm),size):session.feed(pcm[a:a+size],a)
            session.feed([],len(pcm),final=True)
            receipt=self.receipt(session)
            self.assertEqual(receipt['uncertain_samples'],0)
            self.assertEqual(receipt['negative_constant_samples'],0)
            self.assertEqual(receipt['model_negative_samples'],len(pcm))
            self.assertEqual(receipt['decision'],'no_speech')
            validate_receipt(receipt,self.identity,phase='stop',request_id='stop-one',
                received_sample=len(pcm),observed_sample=len(pcm),uncertain_samples=0)
            hashes.append(receipt['pcm_sha256'])
        expected=hashlib.sha256(np.rint(pcm*32768).astype('<i2').tobytes()).hexdigest()
        self.assertEqual(hashes,[expected]*3)
    def test_pause_prefix_and_final_partial_frame_exclude_virtual_padding(self):
        session=SpeechSession(ScorePeer());session.feed(np.zeros(513,dtype=np.float32),0)
        paused=self.receipt(session,'pause','pause-one')
        self.assertEqual((paused['received_sample'],paused['end_sample'],paused['closed']),(513,512,False))
        self.assertEqual(paused['pcm_sha256'],hashlib.sha256(b'\0\0'*512).hexdigest())
        session.feed([],513,final=True);final=self.receipt(session)
        self.assertEqual(final['negative_constant_samples'],513)
        self.assertEqual(final['pcm_sha256'],hashlib.sha256(b'\0\0'*513).hexdigest())
        self.assertEqual(final['decision'],'no_speech')
    def test_failed_inspection_and_typed_receipt_mutations_are_rejected(self):
        failed=SpeechSession(ScorePeer(fail_after=2))
        with self.assertRaises(RuntimeError):failed.feed(np.zeros(1024,dtype=np.float32),0,final=True)
        with self.assertRaises(ValueError):failed.inspection()
        session=SpeechSession(ScorePeer());session.feed(np.zeros(512,dtype=np.float32),0,final=True)
        receipt=self.receipt(session)
        for key,value in [('schema_version',1),('schema_version',True),('input_policy','raw_and_peak025_gaincap256_per512_v1'),('received_sample',True),('end_sample',511),
                          ('closed',False),('inspection_state','failed'),('uncertain_samples',True),
                          ('execution_id','c'*32),('job_id','d'*32),('input_policy','unverified'),
                          ('pcm_sha256','0'*64)]:
            broken={**receipt,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_receipt(broken,self.identity,phase='stop',request_id='stop-one',
                    received_sample=512,observed_sample=512,uncertain_samples=0,
                    pcm_sha256=receipt['pcm_sha256'])
        for missing in (None,{}):
            with self.assertRaises(ValueError):validate_receipt(missing,self.identity,phase='stop',
                request_id='stop-one',received_sample=512,observed_sample=512)
    def test_host_rejects_changed_saved_pcm_without_publishing_stop_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);app=create_app(root);manager=app.extensions['speakerdesk']['meetings']
            folder=root/self.identity['job_id'];folder.mkdir();path=folder/'audio.wav'
            def write(value):
                with wave.open(str(path),'wb') as pcm:
                    pcm.setparams((1,2,16000,0,'NONE','none'));pcm.writeframes(value*512)
            write(b'\0\0');session=SpeechSession(ScorePeer());session.feed(np.zeros(512,dtype=np.float32),0,final=True)
            receipt=self.receipt(session)
            job={'id':self.identity['job_id'],'status':'finishing','duration':512/16000,'created':1.,
                'language':'en','revision':0,'canonical_utterances':True,'admission_execution':self.identity,
                'capture_inspection_request':{'request_id':'stop-one','through_sample':512},
                'document':{'speakers':{},'segments':[],'provenance':{},'warnings':[]}}
            try:
                manager.refinement.initialize(job);manager.put(job);manager.duration=512/16000
                manager.refinement.ready(job['id'],True)
                before=copy.deepcopy(manager.get(job['id']));write(b'\x01\0')
                with self.assertRaises(ValueError):manager.refinement.capture_done(job['id'],
                    observed_sample=512,uncertain_samples=0,admission=receipt)
                self.assertEqual(manager.get(job['id']),before)
                write(b'\0\0');manager.refinement.capture_done(job['id'],
                    observed_sample=512,uncertain_samples=0,admission=receipt)
                saved=manager.get(job['id']);self.assertEqual(saved['capture_admission'],receipt)
                self.assertEqual(saved['refinement_status'],'complete')
                self.assertEqual(retained_pcm_digest(path,512),receipt['pcm_sha256'])
            finally:
                manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
