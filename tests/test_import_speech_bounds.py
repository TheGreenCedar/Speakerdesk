"""Import/retry PCM admission boundaries with scripted CPU model observations.

These exercise the production worker and routing code, not acoustic accuracy.
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
from pipeline import infer, retry_passage
from speech_admission import FRAME, SpeechSession
from transcript import export, validate

RATE = 16000


class ScriptedSpeech:
    """Fixed frame scores independent of amplitude; never a trained VAD claim."""
    normalized_view = True
    def __init__(self, spans):
        self.spans = spans
    def initial_state(self):
        return 0
    def feed(self, chunk, state):
        a, b = state*FRAME, (state+1)*FRAME
        return (.8 if any(first < b and last > a for first, last in self.spans) else .01), state+1
    def session(self, start_sample=0, *, archive=None):
        return SpeechSession(self, start_sample, archive=archive)


class ImportSpeechBounds(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = Path(self.temp.name)
        self.audio = self.root/'audio.wav';self.pcm = varying_pcm(12*RATE, 3, dtype='<i2')
        with wave.open(str(self.audio), 'wb') as wav:
            wav.setparams((1, 2, RATE, 0, 'NONE', 'none'));wav.writeframes(self.pcm.tobytes())
        self.original = self.audio.read_bytes()
        checkpoint = self.root/'checkpoint';checkpoint.mkdir()
        (checkpoint/'model.safetensors').touch()
        (checkpoint/'config.json').write_text(json.dumps(dict(model_type='cohere_asr')))
        self.config = dict(diar_path='cpu', cohere_path=str(checkpoint), speech_path='cpu',
            lid_path='cpu', diar_python=sys.executable, asr_python=sys.executable,
            diar_kind='nemotron', device='mlx')
        self.asr = Mock();self.asr.transcribe.return_value = types.SimpleNamespace(text='Thank you.', tokens=[1])
        self.detector = Mock();self.detector.detect.return_value = dict(en=.99, fr=.01)

    def tearDown(self):
        self.temp.cleanup()

    def replace_pcm(self, pcm):
        self.pcm = pcm
        with wave.open(str(self.audio), 'wb') as wav:
            wav.setparams((1,2,RATE,0,'NONE','none'));wav.writeframes(pcm.tobytes())
        self.original = self.audio.read_bytes()

    def run_import(self, spans, *, language='en', turns=()):
        def worker(python, task, request, folder):
            return dict(turns=list(turns)) if task == 'diarize' else inference_worker.run(task, request)
        with patch.dict(sys.modules, model_modules(self.asr)), patch('inference_worker.check_memory'), \
             patch('inference_worker.importlib.metadata.version', return_value='CPU boundary'), \
             patch('speech_admission.SileroModel', return_value=ScriptedSpeech(spans)), \
             patch('language_detection.WhisperLanguageDetector', return_value=self.detector), \
             patch('pipeline.preflight', return_value=[]), patch('pipeline.run_worker', side_effect=worker):
            document = validate(infer(self.audio, 12, language, self.root, lambda _:None, self.config), 12)
        self.assertEqual(self.audio.read_bytes(), self.original)
        self.assertFalse((self.root/'crops').exists())
        return document

    def assert_bounded_import(self, language):
        document = self.run_import([(0, 32*FRAME)], language=language)
        self.asr.transcribe.assert_called_once()
        decoded = self.asr.transcribe.call_args.args[0]
        # 1.024 seconds of admitted speech plus the canonical 200ms context.
        self.assertEqual(len(decoded), 32*FRAME + 3200)
        np.testing.assert_array_equal(decoded, self.pcm[:32*FRAME + 3200].astype(np.float32)/32768)
        words = [row for row in document['segments'] if row['text']]
        self.assertEqual(len(words), 1)
        self.assertEqual((words[0]['start'], words[0]['end']), (0, (32*FRAME + 3200)/RATE))
        self.assertEqual(words[0]['speaker'], 'unassigned')
        self.assertEqual(words[0]['text'], 'Thank you.')
        self.assertAlmostEqual(sum(row['end']-row['start'] for row in document['segments']), 12)
        negative = [row for row in document['segments'] if not row['text']]
        self.assertTrue(negative)
        self.assertTrue(all(row['audio_state']=='model_non_speech' for row in negative))
        text, _ = export(document, 'txt');self.assertEqual(text.count('Thank you.'), 1)

    def test_manual_import_unknown_gap_does_not_admit_long_negative_tail(self):
        self.assert_bounded_import('en')

    def test_automatic_import_unknown_gap_does_not_admit_long_negative_tail(self):
        self.assert_bounded_import('auto')

    def test_entire_model_negative_recording_never_calls_decoder_or_detector(self):
        document = self.run_import([], language='auto')
        self.asr.transcribe.assert_not_called();self.detector.detect.assert_not_called()
        self.assertTrue(all(not row['text'] for row in document['segments']))
        self.assertEqual(export(document, 'txt')[0].strip(), '')

    def test_zero_unknown_gap_inside_positive_neighbor_frames_never_decodes(self):
        pcm = self.pcm.copy();pcm[320:720] = 0;self.replace_pcm(pcm)
        document = self.run_import([(0,2*FRAME)],
            turns=('0 .02 speaker_0','.045 12 speaker_1'))
        gap = next(row for row in document['segments'] if row['speaker']=='unassigned'
                   and row['start']==.02 and row['end']==.045)
        self.assertEqual(gap['text'], '')
        self.assertEqual(gap['audio_state'], 'digital_silence')
        # Preserve the positive clipped neural receipt instead of rewriting it
        # as model-negative. The independent physical crop is exactly zero.
        self.assertEqual(gap['acoustic_evidence']['decision'], 'speech')
        self.assertEqual(gap['pcm_evidence'], dict(kind='exact_digital_silence',
            start_sample=320,end_sample=720))
        self.assertEqual(self.asr.transcribe.call_count, 2)
        self.assertTrue(all(np.any(call.args[0]) for call in self.asr.transcribe.call_args_list))

    def test_allzero_import_with_positive_model_observations_never_routes_or_decodes(self):
        self.replace_pcm(np.zeros(12*RATE,dtype='<i2'))
        for language in ('en','auto'):
            with self.subTest(language=language):
                self.asr.reset_mock();self.detector.reset_mock()
                document = self.run_import([(0,12*RATE)],language=language)
                self.asr.transcribe.assert_not_called();self.detector.detect.assert_not_called()
                self.assertTrue(all(not row['text'] and row['audio_state']=='digital_silence'
                                    for row in document['segments']))
                self.assertEqual(export(document,'txt')[0].strip(), '')

    def test_exact_nonzero_dc_with_positive_observations_never_routes_or_decodes(self):
        for value in (1,1311):
            with self.subTest(sample=value):
                self.asr.reset_mock();self.detector.reset_mock()
                self.replace_pcm(np.full(12*RATE,value,dtype='<i2'))
                document = self.run_import([(0,12*RATE)],language='auto')
                self.asr.transcribe.assert_not_called();self.detector.detect.assert_not_called()
                self.assertTrue(all(row['audio_state']=='constant_signal' and not row['text']
                                    for row in document['segments']))
                self.assertTrue(all(row['pcm_evidence']['kind']=='exact_constant_signal'
                                    and row['pcm_evidence']['sample_value']==value/32768
                                    for row in document['segments']))

    def test_one_quantum_variation_on_dc_offset_keeps_quiet_raw_speech(self):
        self.replace_pcm((varying_pcm(12*RATE,1,dtype='<i2')+1000).astype('<i2'))
        document = self.run_import([(0,12*RATE)])
        self.asr.transcribe.assert_called_once()
        np.testing.assert_array_equal(self.asr.transcribe.call_args.args[0],
                                     self.pcm.astype(np.float32)/32768)
        self.assertEqual([row['text'] for row in document['segments'] if row['text']], ['Thank you.'])

    def test_single_quiet_positive_frame_and_genuine_thanks_are_retained(self):
        document = self.run_import([(FRAME, 2*FRAME)])
        self.asr.transcribe.assert_called_once()
        decoded = self.asr.transcribe.call_args.args[0]
        self.assertEqual(len(decoded), 2*FRAME + 3200)
        self.assertGreater(float(np.max(np.abs(decoded))), 0)
        self.assertLess(float(np.max(np.abs(decoded))), .001)
        self.assertEqual([row['text'] for row in document['segments'] if row['text']], ['Thank you.'])

    def test_later_speaker_crop_uses_original_sample_anchors_and_preserves_repeated_words(self):
        document = self.run_import([(0,32*FRAME),(224*FRAME,256*FRAME)],
            turns=('0 6 speaker_0','6 12 speaker_1'))
        words = [row for row in document['segments'] if row['text']]
        self.assertEqual([(row['speaker'],row['text']) for row in words],
            [('speaker_0','Thank you.'),('speaker_1','Thank you.')])
        self.assertEqual((words[1]['start'],words[1]['end']),
            ((224*FRAME-3200)/RATE,(256*FRAME+3200)/RATE))
        self.assertEqual(words[1]['acoustic_evidence']['speech_regions'],
            [{'start_sample':224*FRAME,'end_sample':256*FRAME}])
        np.testing.assert_array_equal(self.asr.transcribe.call_args_list[1].args[0],
            self.pcm[224*FRAME-3200:256*FRAME+3200].astype(np.float32)/32768)

    def run_retry(self):
        def worker(python, task, request, folder):
            return inference_worker.run(task, request)
        with patch.dict(sys.modules, model_modules(self.asr)), patch('inference_worker.check_memory'), \
             patch('inference_worker.importlib.metadata.version', return_value='CPU boundary'), \
             patch('speech_admission.SileroModel', return_value=ScriptedSpeech([(0,32*FRAME),(96*FRAME,128*FRAME)])), \
             patch('pipeline.speech_issues', return_value=[]), patch('pipeline.run_worker', side_effect=worker):
            result = retry_passage(self.audio, 0, 6, 'en', self.root, self.config)
        self.assertEqual(self.audio.read_bytes(), self.original)
        self.assertEqual(list(self.root.glob('passage-retry-*')), [])
        return result

    def test_retry_preserves_both_speech_parts_without_decoding_the_long_internal_gap(self):
        self.asr.transcribe.side_effect = [types.SimpleNamespace(text='  Thank you.  ', tokens=[1]),
                                          types.SimpleNamespace(text='  Quiet speech.  ', tokens=[1])]
        result = self.run_retry()
        self.assertEqual(self.asr.transcribe.call_count, 2)
        self.assertEqual(result['text'], '  Thank you.     Quiet speech.  ')
        self.assertEqual((result['start'], result['end']), (0, 6))
        self.assertEqual([len(call.args[0]) for call in self.asr.transcribe.call_args_list],
                         [32*FRAME + 3200, 32*FRAME + 6400])
        self.assertEqual([part['cohere_raw_text'] for part in result['retry_parts'] if part['text']],
                         ['  Thank you.  ','  Quiet speech.  '])

    def test_failed_later_retry_decode_retains_first_raw_words_and_partial_review(self):
        self.asr.transcribe.side_effect = [types.SimpleNamespace(text='  Thank you.  ', tokens=[1]),
                                          RuntimeError('CPU decoder boundary')]
        result = self.run_retry()
        self.assertEqual(self.asr.transcribe.call_count, 2)
        self.assertEqual(result['text'], '  Thank you.  ')
        self.assertTrue(result['review'])
        self.assertEqual(result['transcription_review']['reason'], 'transcription_failed')
        self.assertTrue(result['transcription_review']['partial_text'])
        self.assertTrue(any(part.get('transcription_review',{}).get('reason')=='transcription_failed'
                            for part in result['retry_parts']))

    def test_allzero_retry_with_positive_model_observations_never_decodes(self):
        self.replace_pcm(np.zeros(12*RATE,dtype='<i2'))
        result = self.run_retry()
        self.asr.transcribe.assert_not_called()
        self.assertEqual(result['text'], '')
        self.assertTrue(any(part['audio_state']=='digital_silence' for part in result.get('retry_parts',[result])))


if __name__ == '__main__':
    unittest.main()
