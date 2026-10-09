"""Model-free exact PCM/text anchor producer controls; no neural execution."""
import hashlib
import re
import unittest
from unittest.mock import Mock
import numpy as np
from inference_worker import attach_piece_alignment
from reading_turns import MODEL, CALIBRATION

class ContractAligner:
    """Fabricated envelopes wholly inside known synthetic activity."""
    def align(self, pcm, text, *, start_sample, language):
        return dict(raw_text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            audio_anchor=dict(start_sample=start_sample,end_sample=start_sample+len(pcm)),
            model_sha256=MODEL,timing_kind='ctc_emission_cell_envelope',
            frame_calibration_id=CALIBRATION,score_calibration_id=CALIBRATION,
            audio_float32_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            words=[dict(text=u.group(),start_char=u.start(),end_char=u.end(),status='aligned',
                start_sample=start_sample+1280+i*4400,end_sample=start_sample+2560+i*4400)
                for i,u in enumerate(re.finditer(r'\S+',text))])

class ImportReadingWorker(unittest.TestCase):
    def test_subpiece_gets_exact_physical_slice_and_absolute_anchor(self):
        audio=np.linspace(-.01,.01,32000,dtype=np.float32)
        piece=dict(start=.25,end=.75,language='en',text='Exact repeated repeated 11.',cohere_raw_text='  Exact repeated repeated 11.  ')
        aligner=Mock();aligner.align.return_value={'receipt':'provider'}
        attach_piece_alignment(piece,audio,16000,64000,aligner)
        args,kw=aligner.align.call_args
        np.testing.assert_array_equal(args[0],audio[4000:12000])
        self.assertEqual(args[1],piece['text']);self.assertEqual(kw,dict(start_sample=68000,language='en'))
        self.assertEqual(piece['audio_float32_sha256'],hashlib.sha256(audio[4000:12000].astype('<f4').tobytes()).hexdigest())
        self.assertEqual(piece['cohere_raw_text'],'  Exact repeated repeated 11.  ')
    def test_unsupported_blank_and_wrong_rate_do_not_call_provider(self):
        aligner=Mock()
        for lang,text,rate in [('xx','Bonjour',16000),('en','',16000),('en','Words',8000)]:
            attach_piece_alignment(dict(start=0,end=1,language=lang,text=text),np.ones(16000),rate,0,aligner)
        aligner.align.assert_not_called()
    def test_out_of_bounds_cannot_attach(self):
        for a,b in [(-.1,.5),(0,1.1),(.5,.5)]:
            with self.assertRaises(ValueError):attach_piece_alignment(dict(start=a,end=b,language='en',text='Words'),np.ones(16000),16000,0,Mock())
    def test_required_failure_preserves_raw_words_without_fallback(self):
        piece=dict(start=0,end=1,language='en',text='Words',cohere_raw_text=' Words ')
        aligner=Mock();aligner.align.side_effect=RuntimeError('GPU required')
        with self.assertRaisesRegex(RuntimeError,'Required transcript timing failed') as failure:
            attach_piece_alignment(piece,np.ones(16000),16000,0,aligner)
        self.assertEqual(str(failure.exception.__cause__),'GPU required')
        self.assertNotIn('reading_alignment',piece);self.assertEqual(piece['text'],'Words')
        self.assertEqual(piece['cohere_raw_text'],' Words ');self.assertNotIn('reading_alignment_error',piece)
        self.assertEqual(aligner.align.call_count,1)
