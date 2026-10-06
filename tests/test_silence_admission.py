"""CPU admission/reconciliation contracts; trained observations live in local evidence."""
import copy
from pathlib import Path
import sys
import types
import unittest
import tempfile
import wave
from unittest.mock import Mock
import numpy as np
from pcm_peer import varying_pcm, SpeechEvidencePeer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from language_detection import SpeechTranscriber
from live_refinement import coalesce_blanks,Engine,Inbox
from rolling_refinement import reconcile_window,segment_version

class Detector:
    def __init__(self,p):self.p=p
    def detect(self,audio):return {'en':.99,'fr':.01}
    def no_speech_probability(self,audio):
        if isinstance(self.p,Exception):raise self.p
        return self.p

class SilenceAdmissionTests(unittest.TestCase):
    def test_quiet_brief_and_genuine_thanks_not_filtered_by_words_or_amplitude(self):
        for text in ('the blue','Thank you. Thank you.','Please keep these words.'):
          for gain in (1,.004,.001):
            model=Mock();model.transcribe.return_value=types.SimpleNamespace(text=text,tokens=[1])
            row=SpeechTranscriber(model,'en',detector=Detector(.8828125),speech_evidence=SpeechEvidencePeer()).transcribe(varying_pcm(4000,1/32768)*gain,16000,('speaker_0',))[0]
            self.assertEqual(row['text'],text);model.transcribe.assert_called_once()
    def test_missing_pending_or_failed_vad_blocks_asr_without_asserting_silence(self):
        class Failed:
            def admission(self,a,b):raise RuntimeError('Fixture VAD failure')
        for evidence in (None,SpeechEvidencePeer('pending'),Failed()):
            model=Mock()
            row=SpeechTranscriber(model,'en',detector=Detector(.99),speech_evidence=evidence).transcribe(varying_pcm(16000),16000,('speaker_0',))[0]
            self.assertEqual(row['text'],'');self.assertEqual(row['audio_state'],'speech_evidence_pending')
            model.transcribe.assert_not_called()
    def test_successful_full_silence_replaces_machine_words_and_retains_prior_revision(self):
        for state,evidence in [('model_non_speech',SpeechEvidencePeer('no_speech').admission(0,16000))]:
            old={'id':'old','start':0,'end':1,'speaker':'unassigned','text':'Open the purple folder.','machine_revision':1}
            doc={'speakers':{},'segments':[old]};candidate=dict(start=0,end=1,speaker='unassigned',text='',audio_state=state,acoustic_evidence=evidence)
            result=reconcile_window(doc,{'id':'w','start_sample':0,'end_sample':16000},{'old':segment_version(old)},[candidate])
            self.assertEqual(result['document']['segments'][0]['text'],'')
            self.assertEqual(result['previous_revision']['segments'],[old]);self.assertEqual(result['document']['segments'][0]['refinement_state'],'refined')
            protected=copy.deepcopy(old);protected['protected_fields']=['text'];doc['segments']=[protected]
            kept=reconcile_window(doc,{'id':'w','start_sample':0,'end_sample':16000},{'old':segment_version(protected)},[candidate])
            self.assertEqual(kept['document']['segments'],[protected])
    def test_blank_failed_uncertain_or_partial_audio_cannot_erase_prior_words(self):
        old={'id':'old','start':0,'end':1,'speaker':'unassigned','text':'Genuine words','machine_revision':1}
        for candidate in [dict(start=0,end=1,speaker='unassigned',text=''),
                          dict(start=0,end=1,speaker='unassigned',text='',audio_state='model_non_speech',acoustic_evidence=SpeechEvidencePeer('pending').admission(0,16000)),
                          dict(start=0,end=.99,speaker='unassigned',text='',audio_state='model_non_speech',acoustic_evidence=SpeechEvidencePeer('no_speech').admission(0,15840)),
                          dict(start=0,end=1,speaker='unassigned',text='',audio_state='digital_silence',transcription_review={'reason':'transcription_failed'})]:
            result=reconcile_window({'speakers':{},'segments':[old]},{'id':'w','start_sample':0,'end_sample':16000},{'old':segment_version(old)},[candidate])
            self.assertEqual(result['document']['segments'],[old])
    def test_blank_coalescence_preserves_each_acoustic_observation(self):
        rows=[dict(start=i,end=i+1,speaker='unassigned',text='',language=None,language_epoch=0,audio_state='model_non_speech',
                   acoustic_evidence=SpeechEvidencePeer('no_speech').admission(i*16000,(i+1)*16000)) for i in range(2)]
        self.assertEqual(coalesce_blanks(copy.deepcopy(rows)),rows)
    def test_protected_neighbor_conflict_is_covered_not_falsely_missing_audio(self):
        old={'id':'old','start':0,'end':1,'speaker':'unassigned','text':'Machine words','machine_revision':1}
        protected=dict(old,id='edited',start=1,end=2,text='Protected words',protected_fields=['text'])
        candidate=dict(start=0,end=2,speaker='unassigned',text='',audio_state='model_non_speech',acoustic_evidence=SpeechEvidencePeer('no_speech').admission(0,32000))
        result=reconcile_window({'speakers':{},'segments':[old,protected]},{'id':'w','start_sample':0,'end_sample':32000},
                                {r['id']:segment_version(r) for r in (old,protected)},[candidate])
        self.assertEqual(result['document']['segments'],[old,protected])
        self.assertEqual(result['coverage_gaps']['old'],[])
    def test_long_non_speech_owns_each_core_without_straddling_or_missing_samples(self):
        class Models:
            def __init__(self):
                self.asr=Mock();self.asr.transcribe.side_effect=AssertionError('No non-speech decode allowed')
            def set_language_context(self,context,start):self.start=start
            def transcribe(self,audio,language,names,overlap=False):
                return SpeechTranscriber(self.asr,language,detector=Detector(.99),speech_evidence=SpeechEvidencePeer('no_speech')).transcribe(audio,16000,names,max_asr_seconds=24.5,start_sample=self.start)
        for state,pcm in [('digital_silence',np.zeros(45*16000,dtype='<i2')),
                          ('constant_signal',np.full(45*16000,1,dtype='<i2')),
                          ('model_non_speech',varying_pcm(45*16000,2,dtype='<i2'))]:
          with self.subTest(state=state),tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'audio.wav'
            with wave.open(str(path),'wb') as output:
                output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000);output.writeframes(pcm.tobytes())
            events=[];engine=Engine({'audio_path':str(path),'language':'en'},Models(),events.append,Inbox())
            engine.received=45*16000;engine.processed=45;engine.turns=[];engine.commit(final=True)
            rows=sorted(engine.document['segments'],key=lambda r:r['start'])
            self.assertEqual(engine.cursor,45);self.assertEqual(rows[0]['start'],0);self.assertEqual(rows[-1]['end'],45)
            self.assertTrue(all(a['end']==b['start'] for a,b in zip(rows,rows[1:])))
            self.assertTrue(all(r['audio_state']=='model_non_speech' and not r['text'] and not r['voice_eligible'] for r in rows))
            self.assertFalse(any(e['type']=='boundary_candidate' for e in events))
            self.assertEqual(sum(r['end']-r['start'] for r in rows),45)
            for row in rows:
                for part in row.get('activity_regions',[]):self.assertTrue(row['start']<=part['start']<part['end']<=row['end'])

if __name__=='__main__':unittest.main()
