"""Physical-PCM controls for recurrent VAD carryover; no acoustic AI claims."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from speech_admission import SpeechSession, FrameArchive, FRAME, INPUT_POLICY
from language_detection import SpeechTranscriber

class RecurrentPositive:
    normalized_view=True
    def __init__(self):self.calls=[]
    def initial_state(self):return 0
    def feed(self,chunk,state):
        self.calls.append((chunk.copy(),state))
        return .99,state+1

class ConstantSpeechTailTests(unittest.TestCase):
    def test_neural_hysteresis_retains_quiet_nonconstant_frame_after_zero_frame(self):
        class Model(RecurrentPositive):
            def feed(self,chunk,state):return (.99 if state<2 else .4),state+1
        speech=np.linspace(-1e-9,1e-9,FRAME,dtype=np.float32);speech[-1]=0.
        session=SpeechSession(Model());session.feed(np.concatenate((speech,np.zeros(FRAME),speech)),0,final=True)
        self.assertEqual(session.evidence.admission(FRAME,2*FRAME)['decision'],'no_speech')
        self.assertEqual(session.evidence.admission(2*FRAME,3*FRAME)['decision'],'speech')
        self.assertEqual(session.evidence.frames[-1][2],.4)

    def test_empty_tail_after_speech_never_reaches_asr_but_neural_state_advances(self):
        # Last physical speech sample is zero, matching the real fixture.
        prefix=np.linspace(-.004,.004,FRAME,dtype=np.float32);prefix[-1]=0.
        pcm=np.concatenate((prefix,np.zeros(3*FRAME,dtype=np.float32)))
        original=pcm.copy();model=RecurrentPositive();session=SpeechSession(model)
        session.feed(pcm,0,final=True)
        receipt=session.evidence.admission(FRAME,len(pcm))
        self.assertTrue(receipt['complete']);self.assertEqual(receipt['decision'],'no_speech')
        self.assertEqual(receipt['speech_regions'],[])
        self.assertEqual(receipt['maximum_probability'],0.)
        self.assertEqual(receipt['maximum_model_probability'],.99)
        self.assertEqual(receipt['input_policy'],INPUT_POLICY)
        self.assertEqual((session.state,session.normalized_state),(4,4))
        self.assertEqual(len(model.calls),8)
        for frame in session.evidence.frames[1:]:
            self.assertTrue(frame[4]['constant_pcm_override'])
            self.assertEqual(frame[4]['raw_probability'],.99)
            self.assertEqual(frame[4]['normalized_probability'],.99)
        for language in ('en','auto'):
            asr=Mock();detector=Mock()
            row=SpeechTranscriber(asr,language,detector=detector,speech_evidence=session.evidence).transcribe(
                pcm[FRAME:],16000,(),start_sample=FRAME)[0]
            self.assertEqual(row['text'],'');asr.transcribe.assert_not_called();detector.detect.assert_not_called()
        np.testing.assert_array_equal(pcm,original)

    def test_nonconstant_quiet_single_frame_and_actual_words_are_retained(self):
        pcm=np.linspace(-1e-9,1e-9,FRAME,dtype=np.float32)
        session=SpeechSession(RecurrentPositive());session.feed(pcm,0,final=True)
        self.assertEqual(session.evidence.admission(0,FRAME)['decision'],'speech')
        asr=Mock();asr.transcribe.return_value=types.SimpleNamespace(text='Thank you.',tokens=[1])
        row=SpeechTranscriber(asr,'en',speech_evidence=session.evidence).transcribe(pcm,16000,())[0]
        self.assertEqual(row['text'],'Thank you.');asr.transcribe.assert_called_once()
        np.testing.assert_array_equal(asr.transcribe.call_args.args[0],pcm)

    def test_constant_dc_and_partial_final_frame_survive_archival_packet_changes(self):
        for value in (0.,.04,-.04):
            pcm=np.full(3*FRAME+73,value,dtype=np.float32)
            with tempfile.TemporaryDirectory() as directory:
                session=SpeechSession(RecurrentPositive(),max_frames=1,archive=FrameArchive(Path(directory)/'frames'))
                session.feed(pcm[:123],0);session.feed(pcm[123:],123,final=True)
                receipt=session.evidence.admission(0,len(pcm))
                self.assertEqual(receipt['decision'],'no_speech')
                self.assertEqual(receipt['speech_regions'],[])
                self.assertEqual(receipt['maximum_model_probability'],.99)
                self.assertEqual(session.inspection()['speech_samples'],0)
                self.assertEqual(session.inspection()['negative_constant_samples'],len(pcm))

    def test_constant_step_is_not_discarded_as_a_stable_frame(self):
        pcm=np.concatenate((np.zeros(FRAME,dtype=np.float32),np.full(FRAME,.04,dtype=np.float32)))
        session=SpeechSession(RecurrentPositive());session.feed(pcm,0,final=True)
        self.assertEqual(session.evidence.admission(0,FRAME)['decision'],'no_speech')
        self.assertEqual(session.evidence.admission(FRAME,2*FRAME)['decision'],'speech')
        self.assertIsNone(session.evidence.frames[-1][4]['constant_value'])

if __name__=='__main__':unittest.main()
