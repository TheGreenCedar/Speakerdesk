"""Device-free transport/gain controls; fake probabilities are not accuracy."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from speech_admission import FRAME, SpeechSession, SpeechFrames, FrameArchive, INPUT_POLICY
from speech_conditioning import resolve, SOURCE_UNITY_POLICY, SourceConditioning
from admission_receipt import execution, validate_receipt
from utterances import complete_non_speech
from live_refinement import Models


def config():
    source={'catalog_sha256':'c'*64,'source_id':'microphone_clean','source_revision':1}
    return {'job_id':'a'*32,'capture_source':source, 'experimental_apple_vad_gain':True,
            'capture_processing_contract':{'processor':'apple_voice_processing_io','mode':'active',
              'output':'observed_microphone_tap','job_id':'a'*32,'capture_source':dict(source),
              'configuration_sha256':'b'*64,'receipt_sha256':'d'*64}}


class ShapeModel:
    normalized_view=True
    def __init__(self):self.inputs=[]
    def initial_state(self):return 0
    def feed(self,chunk,state):
        self.inputs.append((chunk.copy(),state))
        return (.9 if np.max(abs(chunk))>.02 else .02), state+1


class SourceConditioningTests(unittest.TestCase):
    def test_default_raw_quiet_and_source_name_do_not_disable_boost(self):
        for c in ({}, {'capture_source':config()['capture_source']},
                  {**config(), 'experimental_apple_vad_gain':False}):
            self.assertIsNone(resolve(c))
            model=ShapeModel();s=SpeechSession(model,conditioning=resolve(c))
            s.feed(np.linspace(-.001,.001,FRAME,dtype=np.float32),0,final=True)
            self.assertEqual(s.evidence.admission(0,FRAME)['decision'],'speech')
            self.assertAlmostEqual(s.evidence.frames[0][4]['gain'],250.,places=4)

    def test_unity_source_cannot_renormalize_low_signal_to_full_scale(self):
        model=ShapeModel();s=SpeechSession(model,conditioning=resolve(config()))
        x=np.linspace(-.0003,.0003,FRAME,dtype=np.float32);original=x.copy()
        s.feed(x,0,final=True)
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(model.inputs[0][0],model.inputs[1][0])
        self.assertEqual(s.evidence.admission(0,FRAME)['decision'],'no_speech')
        self.assertEqual(s.evidence.frames[0][4]['gain'],1.)

    def test_positive_real_quiet_probability_is_not_amplitude_vetoed(self):
        class Positive(ShapeModel):
            def feed(self,x,state):return .8,state+1
        x=np.linspace(-1e-7,1e-7,FRAME,dtype=np.float32)
        s=SpeechSession(Positive(),conditioning=resolve(config()));s.feed(x,0,final=True)
        self.assertEqual(s.evidence.admission(0,FRAME)['decision'],'speech')
        self.assertEqual(s.inspection()['speech_samples'],FRAME)

    def test_processed_normal_quiet_and_tail_pass_original_pcm(self):
        for scale in (.03,.1):
            model=ShapeModel();s=SpeechSession(model,conditioning=resolve(config()))
            x=np.linspace(-scale,scale,700,dtype=np.float32);s.feed(x,0,final=True)
            self.assertEqual(s.evidence.admission(0,700)['decision'],'speech')
            self.assertEqual(s.evidence.end_sample,700)
            expected=hashlib.sha256(np.clip(np.rint(x*32768),-32768,32767).astype('<i2').tobytes()).hexdigest()
            self.assertEqual(s.inspection()['pcm_sha256'],expected)

    def test_packet_partition_recurrent_states_and_virtual_tail_invariant(self):
        frames=[]
        for chunks in ((700,),(1,311,200,188)):
            model=ShapeModel();s=SpeechSession(model,conditioning=resolve(config()));pos=0
            x=np.linspace(-.04,.04,700,dtype=np.float32)
            for count in chunks:s.feed(x[pos:pos+count],pos);pos+=count
            s.feed(np.zeros(0,dtype=np.float32),pos,final=True)
            self.assertEqual((s.state,s.normalized_state),(2,2));frames.append(s.evidence.frames)
        self.assertEqual(frames[0],frames[1])

    def test_system_view_unchanged_only_same_catalog(self):
        c=config();c['capture_source']={**c['capture_source'],'source_id':'system'}
        self.assertIsNone(resolve(c))
        for key,value in (('catalog_sha256','e'*64),('source_revision',True),('unexpected',1)):
            changed=copy.deepcopy(c);changed['capture_source'][key]=value
            with self.assertRaises(ValueError):resolve(changed)

    def test_wrong_missing_unknown_and_bypass_processing_decline(self):
        mutations=[lambda c:c.pop('capture_processing_contract'),
          lambda c:c.update(experimental_apple_vad_gain=1),
          lambda c:c['capture_processing_contract'].update(mode='bypass'),
          lambda c:c['capture_processing_contract'].update(processor='webrtc'),
          lambda c:c['capture_processing_contract'].update(job_id='f'*32),
          lambda c:c['capture_processing_contract'].update(receipt_sha256=''),
          lambda c:c['capture_source'].update(source_revision=2)]
        for change in mutations:
            c=copy.deepcopy(config());change(c)
            with self.assertRaises(ValueError):resolve(c)

    def test_contract_is_frozen_and_hash_binds_receipt(self):
        c=config();first=resolve(c)
        with self.assertRaises(TypeError):first.capture_source['source_id']='system'
        c['capture_processing_contract']['receipt_sha256']='e'*64
        self.assertNotEqual(first.contract_sha256,resolve(c).contract_sha256)
        with self.assertRaises(ValueError):SourceConditioning('bad',{})

    def test_archive_and_inspection_keep_new_policy_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            profile=resolve(config());s=SpeechSession(ShapeModel(),conditioning=profile,
              max_frames=1,archive=FrameArchive(Path(folder)/'frames'))
            s.feed(np.linspace(-.04,.04,700,dtype=np.float32),0,final=True)
            observed=s.evidence.admission(0,700)
            self.assertEqual(observed['input_policy'],SOURCE_UNITY_POLICY)
            self.assertEqual(observed['conditioning_sha256'],profile.contract_sha256)
            identity=execution('a'*32,'f'*32,conditioning=profile)
            receipt={**identity,**s.inspection(),'phase':'stop','request_id':'stop'}
            validate_receipt(receipt,identity,phase='stop',request_id='stop',received_sample=700,observed_sample=700)
            wrong=execution('a'*32,'f'*32)
            with self.assertRaises(ValueError):validate_receipt(receipt,wrong,phase='stop',request_id='stop',received_sample=700,observed_sample=700)

    def test_unqualified_new_policy_cannot_erase_prior_machine_words(self):
        s=SpeechSession(ShapeModel(),conditioning=resolve(config()))
        s.feed(np.linspace(-.0003,.0003,FRAME,dtype=np.float32),0,final=True)
        row={'text':'','audio_state':'model_non_speech','acoustic_evidence':s.evidence.admission(0,FRAME)}
        self.assertFalse(complete_non_speech(row,0,FRAME))

    def test_invalid_model_source_config_refuses_before_model_imports(self):
        with self.assertRaisesRegex(ValueError,'contract'):
            Models({'experimental_apple_vad_gain':True})

    def test_saved_refinement_uses_same_source_conditioning(self):
        class Speech:
            def inspect_frames(self,*args,**kwargs):self.seen=kwargs;return 'observed'
        with tempfile.TemporaryDirectory() as folder:
            owner=object.__new__(Models);owner.speech=Speech();owner.config={'audio_path':str(Path(folder)/'audio.wav')}
            owner.speech_conditioning=resolve(config())
            owner.begin_refinement(np.zeros(512,dtype=np.float32),0)
            self.assertIs(owner.speech.seen['conditioning'],owner.speech_conditioning)


if __name__=='__main__':unittest.main()
