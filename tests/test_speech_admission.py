"""CPU control/transport tests. Synthetic model probabilities are not AI proof."""
import sys
import tempfile
import unittest
import wave
import types
from unittest.mock import Mock
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from audio import pcm16_bytes
from speech_admission import FRAME, SpeechFrames, SpeechSession, FrameArchive
from live_refinement import read_audio
from language_detection import SpeechTranscriber


class SpeechAdmissionTests(unittest.TestCase):
    def test_no_evidence_and_partial_evidence_are_pending(self):
        frames = SpeechFrames()
        self.assertEqual(frames.admission(0, FRAME)['decision'], 'pending')
        frames.append(0, FRAME, .01)
        self.assertEqual(frames.admission(0, FRAME+1)['decision'], 'pending')
        self.assertEqual(frames.admission(0, FRAME)['decision'], 'no_speech')

    def test_one_positive_frame_survives_without_minimum_duration_filter(self):
        frames = SpeechFrames()
        for n, score in enumerate((.01, .8, .4, .1)):
            frames.append(n*FRAME, (n+1)*FRAME, score)
        result = frames.admission(0, 4*FRAME)
        self.assertEqual(result['decision'], 'speech')
        self.assertEqual(result['speech_regions'], [{'start_sample': FRAME, 'end_sample': 3*FRAME}])

    def test_invalid_probability_and_out_of_order_frames_fail(self):
        for score in (float('nan'), float('inf'), -1, 2, True):
            with self.assertRaises(ValueError): SpeechFrames().append(0, FRAME, score)
        with self.assertRaises(ValueError): SpeechFrames().append(1, FRAME, .1)

    def test_capture_packetization_preserves_state_and_original_endpoint(self):
        class Boundary:
            def initial_state(self):return 0
            def feed(self, chunk, state):
                self.chunks.append(chunk.copy())
                return .8 if np.any(chunk) else .01, state+1
        model = Boundary();model.chunks=[]
        session = SpeechSession(model, 900)
        session.feed(np.ones(100), 900)
        self.assertEqual(session.evidence.admission(900, 1000)['decision'], 'pending')
        session.feed(np.ones(600), 1000, final=True)
        self.assertEqual(session.state, 2)
        self.assertEqual(session.evidence.end_sample, 1600)
        self.assertEqual(len(model.chunks), 2)
        self.assertTrue(np.all(model.chunks[1][188:] == 0))
        with self.assertRaises(ValueError):session.feed(np.ones(1), 1600)

    def test_all_pcm16_values_round_trip_without_quiet_attenuation(self):
        samples = np.arange(-32768, 32768, dtype=np.int16)
        decoded = samples.astype(np.float32)/32768
        self.assertEqual(pcm16_bytes(decoded), samples.astype('<i2').tobytes())

    def test_saved_model_audio_matches_capture_and_soundfile_scale(self):
        # Exercise the actual capture writer / resident reader boundary, with
        # every quiet integer and the negative full-scale value included.
        samples = np.arange(-32768, 32768, dtype=np.int16)
        decoded = samples.astype(np.float32)/32768
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'capture.wav'
            with wave.open(str(path), 'wb') as writer:
                writer.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                writer.writeframes(pcm16_bytes(decoded))
            np.testing.assert_array_equal(read_audio(path, 0, len(samples)), decoded)

    def test_failed_neural_frame_cannot_be_retried_as_complete_silence(self):
        for outcome in (RuntimeError('Neural failure'), float('nan')):
            class Boundary:
                def initial_state(self):return 0
                def feed(self, chunk, state):
                    self.calls += 1
                    if isinstance(outcome, Exception):raise outcome
                    return outcome, state+1
            model = Boundary();model.calls = 0
            session = SpeechSession(model)
            with self.assertRaises((RuntimeError, ValueError)):
                session.feed(np.ones(FRAME), 0)
            self.assertTrue(session.failed)
            self.assertEqual(session.evidence.admission(0, FRAME)['decision'], 'pending')
            with self.assertRaises(ValueError):session.feed(np.zeros(FRAME), FRAME, final=True)
            self.assertEqual(model.calls, 1)

    def test_float_capture_saturates_and_rejects_nonfinite_audio(self):
        result = np.frombuffer(pcm16_bytes(np.array([-2., -1., 0., 1., 2.])), dtype='<i2')
        self.assertEqual(result.tolist(), [-32768, -32768, 0, 32767, 32767])
        with self.assertRaises(ValueError):pcm16_bytes(np.array([np.nan]))

    def test_bounded_hot_frames_preserve_queryable_durable_raw_history(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=FrameArchive(Path(folder)/'evidence.jsonl')
            frames=SpeechFrames(max_frames=32,archive=archive)
            for n in range(1000):frames.append(n*FRAME,(n+1)*FRAME,.8 if 10<=n<12 else .01)
            self.assertLessEqual(len(frames.frames),32)
            past=frames.admission(0,20*FRAME)
            self.assertTrue(past['complete'])
            self.assertEqual(past['speech_regions'],[{'start_sample':10*FRAME,'end_sample':12*FRAME}])
            self.assertEqual(frames.admission(900*FRAME,1000*FRAME)['decision'],'no_speech')
            self.assertTrue(archive.path.stat().st_size>0)
            frames.persist()
            self.assertEqual(frames.frames,[])
            self.assertEqual(frames.admission(0,20*FRAME)['decision'],'speech')

    def test_archive_failure_cannot_drop_evidence_or_retry_neural_state(self):
        class Boundary:
            def initial_state(self):return 0
            def feed(self,chunk,state):return .01,state+1
        class FailedArchive:
            def append(self,frames):raise OSError('Fixture write failed')
            def query(self,a,b):return []
        session=SpeechSession(Boundary(),max_frames=1,archive=FailedArchive())
        session.feed(np.zeros(FRAME),0)
        with self.assertRaises(OSError):session.feed(np.zeros(FRAME),FRAME)
        self.assertTrue(session.failed)
        self.assertEqual(session.evidence.end_sample,FRAME)
        self.assertEqual(session.evidence.admission(0,2*FRAME)['decision'],'pending')
        with self.assertRaises(ValueError):SpeechFrames(max_frames=2)

    def test_language_context_names_and_overlap_never_bypass_missing_speech(self):
        for language in ('en','fr','auto'):
            for speakers in (('speaker_0',),(),('speaker_0','speaker_1')):
                for decision in ('pending','no_speech'):
                    frames=SpeechFrames()
                    if decision=='no_speech':
                        for a in range(0,16000,FRAME):frames.append(a,min(a+FRAME,16000),.01)
                    asr=Mock();detector=Mock();detector.detect.return_value={'en':.99,'fr':.01}
                    transcriber=SpeechTranscriber(asr,language,detector=detector,
                        context={'language':'en','end_sample':0},speech_evidence=frames)
                    row=transcriber.transcribe(np.ones(16000,dtype=np.float32)*.000001,16000,speakers)[0]
                    self.assertEqual(row['text'],'')
                    self.assertEqual(row['audio_state'],'model_non_speech' if decision=='no_speech' else 'speech_evidence_pending')
                    asr.transcribe.assert_not_called();detector.detect.assert_not_called()

    def test_positive_short_evidence_is_not_vetoed_by_whisper_or_amplitude(self):
        frames=SpeechFrames();frames.append(0,512,.8);frames.append(512,1000,.4)
        detector=Mock();detector.no_speech_probability.side_effect=AssertionError('Whisper cannot gate speech')
        asr=Mock();asr.transcribe.return_value=types.SimpleNamespace(text='Blue',tokens=[1])
        pcm=np.ones(1000,dtype=np.float32)*.000001
        row=SpeechTranscriber(asr,'en',detector=detector,speech_evidence=frames).transcribe(pcm,16000,('speaker_0',))[0]
        self.assertEqual(row['text'],'Blue')
        np.testing.assert_array_equal(asr.transcribe.call_args.args[0],pcm)
        detector.no_speech_probability.assert_not_called()


if __name__ == '__main__':unittest.main()
