"""CPU control/transport tests. Synthetic model probabilities are not AI proof."""
import sys
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from audio import pcm16_bytes
from speech_admission import FRAME, SpeechFrames, SpeechSession
from live_refinement import read_audio


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


if __name__ == '__main__':unittest.main()
