"""Actual retained VAD receipts through the CPU import worker/routing boundary.

PCM and ASR here are substitutes. The retained GPU continuous-decode outcome
supports the context hypothesis separately; these tests verify source behavior.
"""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
from pcm_peer import varying_pcm
from test_language_detection import model_modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import inference_worker
from pipeline import infer
from speech_admission import SpeechFrames
from transcript import export, validate

RECEIPTS = json.loads((Path(__file__).parent/'metadata/import_tail_receipts.json').read_text())
RATE = 16000


class ReceiptSpeech:
    """Read-only observations from actual synthetic model runs, not PCM scoring."""
    def __init__(self, records):
        self.records = records

    def session(self, start_sample=0, *, archive=None):
        ledger = SpeechFrames(start_sample)
        for a, b, _, _, observation in self.records:
            ledger.append(a, b, max(observation['raw_probability'], observation['normalized_probability']),
                          observation=observation)
        assert [list(frame) for frame in ledger.frames] == self.records
        session = types.SimpleNamespace(received=start_sample, evidence=ledger)

        def feed(audio, start, *, final=False):
            assert start == session.received
            session.received += len(audio)
            if final:
                assert session.received == ledger.end_sample

        session.feed = feed
        return session


class ImportTailContext(unittest.TestCase):
    def run_receipt(self, name, *, turns=None, language='auto', detector_scores=None):
        case = RECEIPTS[name]
        count = case['frames'][-1][1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root/'audio.wav'
            pcm = varying_pcm(count, 3, dtype='<i2')
            # Fabricated low nonconstant PCM continues to the actual receipt's
            # final admitted frame; the rest is exact zero, without reading WAV.
            last = max(frame[1] for frame in case['frames'] if frame[3])
            pcm[last:] = 0
            with wave.open(str(audio), 'wb') as wav:
                wav.setparams((1, 2, RATE, 0, 'NONE', 'none'))
                wav.writeframes(pcm.tobytes())
            original = audio.read_bytes()
            checkpoint = root/'checkpoint';checkpoint.mkdir()
            (checkpoint/'model.safetensors').touch()
            config = dict(diar_path='cpu', cohere_path=str(checkpoint), speech_path='cpu',
                lid_path='cpu', diar_python=sys.executable, asr_python=sys.executable,
                diar_kind='nemotron', device='mlx')
            asr = Mock()
            continuous = RECEIPTS['gpu_continuous_probe']['raw_text']

            def decode(samples, **kwargs):
                # An isolated actual tail anchor gets the retained bad output;
                # a full continuous anchor gets the retained successful output.
                text = (continuous if len(samples) >= 44800 else 'you') if name == 'auto-real-thanks' else case['result']['words'][0]['text']
                return types.SimpleNamespace(text='  '+text+'  ', tokens=[1])

            asr.transcribe.side_effect = decode
            detector = Mock()
            detector.detect.side_effect = detector_scores
            detector.detect.return_value = dict(en=.99, fr=.01)

            def worker(python, task, request, folder):
                if task == 'diarize':
                    return dict(turns=case['diarization'] if turns is None else turns)
                return inference_worker.run(task, request)

            with patch.dict(sys.modules, model_modules(asr)), patch('inference_worker.check_memory'), \
                 patch('inference_worker.importlib.metadata.version', return_value='CPU receipt replay'), \
                 patch('speech_admission.SileroModel', return_value=ReceiptSpeech(case['frames'])), \
                 patch('language_detection.WhisperLanguageDetector', return_value=detector), \
                 patch('pipeline.preflight', return_value=[]), patch('pipeline.run_worker', side_effect=worker):
                document = validate(infer(audio, count/RATE, language, root, lambda _:None, config), count/RATE)
            self.assertEqual(audio.read_bytes(), original)
            self.assertFalse((root/'crops').exists())
        return document, asr, pcm

    def test_faint_tail_crossing_utterance_is_decoded_once_with_continuous_context(self):
        document, asr, pcm = self.run_receipt('auto-real-thanks')
        asr.transcribe.assert_called_once()
        np.testing.assert_array_equal(asr.transcribe.call_args.args[0], pcm[:50816].astype(np.float32)/32768)
        words = [row for row in document['segments'] if row['text']]
        self.assertEqual([row['cohere_raw_text'] for row in words],
                         ['  '+RECEIPTS['gpu_continuous_probe']['raw_text']+'  '])
        self.assertEqual((words[0]['start'], words[0]['end']), (0, 3.176))
        self.assertEqual(words[0]['speaker'], 'unassigned')
        self.assertEqual(words[0]['speaker_candidates'], ['speaker_0'])
        self.assertEqual(words[0]['activity_regions'], [
            {'start':0., 'end':2.8000000000000003, 'speakers':['speaker_0']},
            {'start':2.8000000000000003, 'end':3.176, 'speakers':[]}])
        self.assertTrue(words[0]['review']);self.assertFalse(words[0]['voice_eligible'])
        self.assertEqual(document['diarization'],
                         [{'start':0., 'end':2.8000000000000003, 'speaker':'speaker_0'}])
        text, _ = export(document, 'txt')
        self.assertEqual(text.count('Thank you, thank you, the folder is ready.'), 1)
        self.assertNotIn('Overlapping', text)
        self.assertAlmostEqual(sum(row['end']-row['start'] for row in document['segments']), len(pcm)/RATE)
        self.assertTrue(all(not row['text'] and row['audio_state']=='model_non_speech'
                            for row in document['segments'] if row['start'] >= 3.176))

    def test_quiet_control_exact_zero_gap_keeps_silence_guard_and_raw_sentence(self):
        document, asr, pcm = self.run_receipt('auto-quiet')
        asr.transcribe.assert_called_once()
        self.assertEqual(len(asr.transcribe.call_args.args[0]), 56960)
        words = [row for row in document['segments'] if row['text']]
        self.assertEqual(words[0]['cohere_raw_text'],
                         '  '+RECEIPTS['auto-quiet']['result']['words'][0]['text']+'  ')
        self.assertEqual(words[0]['speaker'], 'speaker_0')
        self.assertTrue(all(not row['text'] for row in document['segments'] if row['start'] >= 3.56))

    def test_manual_language_keeps_the_same_continuous_original_pcm(self):
        document, asr, pcm = self.run_receipt('auto-real-thanks', language='en')
        asr.transcribe.assert_called_once()
        np.testing.assert_array_equal(asr.transcribe.call_args.args[0], pcm[:50816].astype(np.float32)/32768)
        self.assertEqual(asr.transcribe.call_args.kwargs['language'], 'en')
        self.assertEqual(document['segments'][0]['speaker'], 'unassigned')

    def test_compatible_weak_language_probe_keeps_one_raw_continuous_decode(self):
        document, asr, _ = self.run_receipt('auto-real-thanks',
            detector_scores=[dict(en=.99,fr=.01), dict(en=.61,fr=.39)])
        asr.transcribe.assert_called_once()
        words = [row for row in document['segments'] if row['text']]
        self.assertEqual(len(words), 1)
        self.assertEqual(len(words[0]['language_detection']['probes']), 2)
        self.assertTrue(words[0]['language_review'])

    def test_real_speaker_change_keeps_separate_raw_decodes(self):
        document, asr, _ = self.run_receipt('auto-real-thanks',
            turns=['0 1.4 speaker_0', '1.4 3.9698125 speaker_1'])
        self.assertEqual(asr.transcribe.call_count, 2)
        words = [row for row in document['segments'] if row['text']]
        self.assertEqual([row['speaker'] for row in words], ['speaker_0', 'speaker_1'])
        self.assertEqual(words[0]['end'], 1.4)
        self.assertFalse(any(row.get('decode_context') for row in words))

    def test_speaker_return_keeps_three_separate_raw_decodes(self):
        document, asr, _ = self.run_receipt('auto-real-thanks',
            turns=['0 1 speaker_0', '1 2 speaker_1', '2 3.9698125 speaker_0'])
        self.assertEqual(asr.transcribe.call_count, 3)
        self.assertEqual([row['speaker'] for row in document['segments'] if row['text']],
                         ['speaker_0', 'speaker_1', 'speaker_0'])
        self.assertFalse(any(row.get('decode_context') for row in document['segments']))

    def test_different_speaker_after_unknown_gap_keeps_all_boundaries(self):
        document, asr, _ = self.run_receipt('auto-real-thanks',
            turns=['0 2.8 speaker_0', '2.96 3.9698125 speaker_1'])
        self.assertEqual(asr.transcribe.call_count, 3)
        self.assertEqual([row['speaker'] for row in document['segments'] if row['text']],
                         ['speaker_0', 'unassigned', 'speaker_1'])

    def test_true_overlap_keeps_explicit_overlap_and_separate_tail(self):
        document, asr, _ = self.run_receipt('auto-real-thanks',
            turns=['0 2.8 speaker_0', '0 2.8 speaker_1'])
        self.assertEqual(asr.transcribe.call_count, 2)
        self.assertEqual([row['speaker'] for row in document['segments'] if row['text']],
                         ['overlap', 'unassigned'])


if __name__ == '__main__':
    unittest.main()
