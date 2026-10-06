"""Prevent promoted tail hallucinations while retaining real quiet/short speech."""
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from language_detection import SpeechTranscriber
from live_refinement import Engine,Inbox
from transcript import export
from voice_profiles import speaker_audio_eligible

RATE=16000

class Evidence:
    def detect(self,audio):return {'en':.99,'fr':.01}
    def no_speech_probability(self,audio):return .96

class AcousticTests(unittest.TestCase):
    def model(self,text='yes',tokens=1):
        asr=Mock();asr.transcribe.return_value=types.SimpleNamespace(text=text,tokens=[1]*tokens)
        return asr
    def test_frontend_invalid_inputs_are_retained_not_decoded_or_called_silence(self):
        for mode in ('en','auto'):
            for samples in (78,159,160,319):
                with self.subTest(mode=mode,samples=samples):
                    asr=self.model();detector=Mock()
                    row=SpeechTranscriber(asr,mode,detector=detector).transcribe(np.ones(samples),RATE,('speaker_0',))[0]
                    asr.transcribe.assert_not_called();detector.detect.assert_not_called()
                    self.assertEqual(row['text'],'');self.assertEqual(row['audio_state'],'insufficient_acoustic_context')
                    self.assertTrue(row['review']);self.assertEqual(row['end'],samples/RATE)
    def test_valid_short_candidate_is_recoverable_and_truncation_is_preserved(self):
        for tokens in (1,448):
            asr=self.model('Okay.',tokens)
            row=SpeechTranscriber(asr,'en').transcribe(np.ones(658),RATE,('speaker_0',))[0]
            asr.transcribe.assert_called_once()
            self.assertEqual(row['text'],'');self.assertEqual(row['transcription_review']['candidate_text'],'Okay.')
            self.assertEqual(row['transcription_review']['partial_text'],tokens==448)
            self.assertTrue(row['review'])
            row.update(speaker='speaker_0',speaker_candidates=['speaker_0'],finalized=True)
            self.assertFalse(speaker_audio_eligible(row,'speaker_0'))
            self.assertNotIn('Okay.',export({'speakers':{'speaker_0':'One'},'segments':[row]},'txt')[0])
            self.assertIn('Okay.',export({'speakers':{'speaker_0':'One'},'segments':[row]},'json')[0])
    def test_quiet_real_short_control_reaches_asr_without_amplitude_cutoff(self):
        asr=self.model('the blue')
        row=SpeechTranscriber(asr,'en').transcribe(np.full(4000,1/32768),RATE,('speaker_0',))[0]
        self.assertEqual(row['text'],'the blue');asr.transcribe.assert_called_once()
        self.assertNotIn('transcription_review',row)
    def test_no_speech_score_with_missing_speaker_demotes_instead_of_erasing_candidate(self):
        for mode in ('auto','en'):
            asr=self.model('Thank you.')
            row=SpeechTranscriber(asr,mode,detector=Evidence()).transcribe(np.full(RATE,.0001),RATE,())[0]
            asr.transcribe.assert_called_once();self.assertEqual(row['text'],'')
            self.assertEqual(row['transcription_review']['candidate_text'],'Thank you.')
            self.assertEqual(row['acoustic_evidence']['no_speech_probability'],.96)
            self.assertTrue(row['review']);self.assertEqual(row['audio_state'],'possible_non_speech')
    def test_independent_acoustic_warning_does_not_gate_named_speech_or_language(self):
        row=SpeechTranscriber(self.model('Real words'),'auto',detector=Evidence()).transcribe(np.ones(RATE),RATE,('speaker_0',))[0]
        self.assertEqual(row['text'],'Real words');self.assertEqual(row['language'],'en')

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'audio.wav'
        with wave.open(str(self.path),'wb') as f:
            f.setparams((1,2,RATE,0,'NONE','not compressed'));f.writeframes(b'\x01\x00'*5*RATE)
        self.events=[];self.models=Mock()
        self.engine=Engine({'audio_path':str(self.path),'language':'en'},self.models,self.events.append,Inbox())
        self.config=self.engine.language_at(0)
    def tearDown(self):self.temp.cleanup()
    def results(self,texts,partial=False):
        self.models.transcribe.side_effect=[[{'start':0,'end':len(a)/RATE,'text':text,'language':'en','review':partial,
            **({'transcription_review':{'reason':'token_limit','partial_text':True}} if partial else {})}]
            for a,text in [(np.ones(3*RATE),t) for t in texts]]
    def test_same_origin_complete_prefix_proves_no_new_tail_words(self):
        self.results(['Earlier words.','Earlier words.'])
        row=self.engine.transcribe_owned(4,4.041125,self.config,['speaker_0'])[0]
        self.assertEqual(row['text'],'');self.assertEqual(row['audio_state'],'context_no_new_words')
        self.assertFalse(row['voice_eligible'])
        calls=self.models.transcribe.call_args_list
        self.assertEqual(len(calls[0].args[0]),3*RATE)
        self.assertEqual(len(calls[1].args[0]),round(3.041125*RATE))
    def test_genuine_same_origin_repetition_survives(self):
        self.results(['yes.','yes. yes.'])
        row=self.engine.transcribe_owned(4,4.3,self.config,['speaker_0'])[0]
        self.assertEqual(row['text'],'yes.')
    def test_truncated_prefix_cannot_establish_no_new_words(self):
        self.results(['Earlier words.','Earlier words.'],partial=True)
        row=self.engine.transcribe_owned(4,4.041125,self.config,['speaker_0'])[0]
        self.assertEqual(row['transcription_review']['reason'],'refinement_incomplete')
        self.assertTrue(row['review']);self.assertNotEqual(row.get('audio_state'),'context_no_new_words')
        self.assertTrue(any(e['type']=='boundary_candidate' for e in self.events))
    def test_tail_context_cannot_cross_language_epoch(self):
        self.engine.timeline.append({'start_sample':4*RATE,'epoch':1,'language':'fr'})
        self.results(['bonjour'])
        self.engine.transcribe_owned(4,4.25,self.engine.language_at(4*RATE),['speaker_0'])
        self.assertEqual(len(self.models.transcribe.call_args.args[0]),4000)

if __name__=='__main__':unittest.main()
