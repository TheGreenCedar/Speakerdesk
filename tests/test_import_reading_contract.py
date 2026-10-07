"""Qualified provider-receipt contract through saved API, renderer and export.

All word times in this file are explicitly fabricated contract-test values.
No acoustic accuracy or GPU alignment result is claimed by these controls.
"""
import copy
import hashlib
import re
import unittest
from unittest.mock import patch
import soundfile as sf

import test_import_tail_product_path as product_path
import inference_worker
from reading_turns import CALIBRATION, MODEL, validated_turns


class ImportReadingContract(unittest.TestCase):
    def imported(self, name, *, modify=lambda result:None):
        run=inference_worker.run

        def worker(task,request):
            result=run(task,request)
            for chunk,pieces in zip(result.get('chunks',request['chunks']),result['regions']):
                if not chunk.get('decode_context'):continue
                for piece in pieces:
                    if not piece['text']:continue
                    a=round((chunk['audio_start']+piece['start'])*16000)
                    b=round((chunk['audio_start']+piece['end'])*16000)
                    raw=piece['text']
                    # Test-only distinct envelopes stay inside original A
                    # activity and leave an observed unknown tail in place.
                    words=[dict(text=unit.group(),start_char=unit.start(),end_char=unit.end(),
                        status='aligned',start_sample=4800+i*4400,end_sample=6400+i*4400)
                        for i,unit in enumerate(re.finditer(r'\S+',raw))]
                    pcm,rate=sf.read(request['audio'],start=a,frames=b-a,dtype='float32')
                    assert rate==16000 and len(pcm)==b-a
                    audio_hash=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
                    alignment=dict(raw_text=raw,text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        audio_anchor=dict(start_sample=a,end_sample=b),model_sha256=MODEL,
                        timing_kind='ctc_emission_cell_envelope',frame_calibration_id=CALIBRATION,
                        score_calibration_id=CALIBRATION,audio_float32_sha256=audio_hash,words=words)
                    modify(alignment)
                    piece.update(reading_alignment=alignment,audio_float32_sha256=audio_hash)
            return result

        with patch('inference_worker.run',side_effect=worker):
            return product_path.ImportTailProductPath.imported(self,name)

    def test_qualified_timing_preserves_known_words_in_saved_API_renderer_and_TXT(self):
        rows,evidence=self.imported('qualified-contract')
        self.assertEqual(evidence['ASR_calls'],1)
        self.assertEqual(rows[0]['speaker'],'unassigned')
        turns=validated_turns(rows[0])
        self.assertEqual([turn['speaker'] for turn in turns],['speaker_0'])
        self.assertEqual(''.join(turn['text'] for turn in turns),rows[0]['text'])
        self.assertEqual(evidence['rendered'][0]['readingLabels'],['Speaker 1'])
        self.assertIn('Speaker 1: '+rows[0]['text'],evidence['exported_txt'])
        self.assertNotIn('Unknown speaker: '+rows[0]['text'],evidence['exported_txt'])
        self.assertTrue(rows[0]['cohere_raw_text'].startswith('  Thank you'))

    def test_word_in_unknown_activity_remains_locally_unassigned(self):
        def change(result):
            result['words'][-1].update(start_sample=45200,end_sample=46000)
        rows,evidence=self.imported('unknown-word-contract',modify=change)
        turns=validated_turns(rows[0])
        self.assertEqual([turn['speaker'] for turn in turns],['speaker_0','unassigned'])
        self.assertEqual([turn['attribution'] for turn in turns],['single','unknown'])
        self.assertEqual(''.join(turn['text'] for turn in turns),rows[0]['text'])
        self.assertIn('Speaker 1: ',evidence['exported_txt'])
        self.assertIn('Unknown speaker: ready.',evidence['exported_txt'])

    def test_partial_timing_does_not_erase_unresolved_raw_words(self):
        def change(result):
            result['words'][-1].update(status='unresolved',start_sample=None,end_sample=None)
        rows,evidence=self.imported('partial-contract',modify=change)
        turns=validated_turns(rows[0])
        self.assertEqual([turn['attribution'] for turn in turns],['single','unknown'])
        self.assertEqual(''.join(turn['text'] for turn in turns),rows[0]['text'])
        self.assertEqual(evidence['ASR_calls'],1)

    def test_unqualified_stale_or_mismatched_receipts_cannot_assign_any_words(self):
        mutations=[lambda result:result.update(model_sha256='0'*64),
                   lambda result:result.update(frame_calibration_id='unqualified'),
                   lambda result:result.update(score_calibration_id='unqualified'),
                   lambda result:result.update(text_sha256='0'*64),
                   lambda result:result.update(audio_float32_sha256='0'*64),
                   lambda result:result['audio_anchor'].update(end_sample=44800),
                   lambda result:result.update(words=[None]*len(result['words']))]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                rows,evidence=self.imported('rejected-contract-'+str(index),modify=mutation)
                self.assertEqual(validated_turns(rows[0]),[])
                self.assertEqual(rows[0]['speaker'],'unassigned')
                self.assertIn('Unknown speaker: '+rows[0]['text'],evidence['exported_txt'])
                self.assertEqual(evidence['ASR_calls'],1)

    def test_human_text_edit_invalidates_saved_projection(self):
        rows,_=self.imported('edit-contract')
        edited=copy.deepcopy(rows[0]);edited['text']='Human correction.'
        self.assertEqual(validated_turns(edited),[])


if __name__=='__main__':unittest.main()
