"""Synthetic data contracts only; invented envelopes are not acoustic proof."""
import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from canonical_assembly import assemble_parts,part_utterance
from utterances import UtteranceBook,RevisionArchive
from word_alignment import MODEL_SHA256


def book_for(seconds=36,**kwargs):
    book=UtteranceBook('synthetic-only',**kwargs)
    samples=seconds*16000
    book.observe({'start_sample':0,'end_sample':samples,'complete':True,'decision':'speech',
        'speech_regions':[{'start_sample':0,'end_sample':samples}]})
    return book,book.finish()[0]


def part(request,text,spans):
    current=part_utterance(request,text)
    words=[{'text':match.group(),'start_char':match.start(),'end_char':match.end(),
        'status':'aligned','start_sample':start,'end_sample':end}
        for match,(start,end) in zip(re.finditer(r'\S+',text),spans)]
    return {'request':copy.deepcopy(request),'text':text,'complete':True,
        'alignment':{'complete':True,'status':'aligned','raw_text':text,
            'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'words':words,
            'audio_anchor':current['text_audio_anchor'],'model_sha256':MODEL_SHA256,
            'timing_kind':'ctc_emission_cell_envelope','frame_calibration_id':'invented-fixture-grid',
            'score_calibration_id':'invented-fixture-policy'}}


def repeated_parts(book,row):
    requests=book.decode_requests(row['id'])
    return [part(requests[0],'go  12.5 go',[(1000,5000),(260000,270000),(300000,310000)]),
            part(requests[1],'12.5 go\t12.5',[(260000,270000),(300000,310000),(400000,410000)])]


class AssemblyTests(unittest.TestCase):
    def test_coarse_envelope_near_seam_cannot_claim_exact_core_ownership(self):
        # Documented AMI 'designing' counterexample translated to the 18s
        # fixture seam: emission end is 560 samples before the seam, while
        # the external coarse reference crosses it. No acoustic claim here.
        book,row=book_for();requests=book.decode_requests(row['id'])
        parts=[part(requests[0],'before designing',[(250000,255000),(280000,287440)]),
               part(requests[1],'before designing after',[(250000,255000),(280000,287440),(400000,410000)])]
        result=assemble_parts(row,requests,parts)
        self.assertFalse(result['complete'])
        self.assertEqual(result['reason'],'boundary_uncertainty_unresolved')
        self.assertIsNone(result['text']);self.assertEqual(result['words'],[])

    def test_opposing_context_ownership_cannot_silently_omit_or_duplicate_a_word(self):
        for opposing in ('omit','duplicate'):
            book,row=book_for();requests=book.decode_requests(row['id'])
            first,last=((289000,290000),(286000,287000)) if opposing=='omit' else ((286000,287000),(289000,290000))
            parts=[part(requests[0],'before final',[(250000,255000),first]),
                   part(requests[1],'before final after',[(250000,255000),last,(400000,410000)])]
            # Both independent alignments are complete and in range, but they
            # disagree about which disjoint core owns the shared final word.
            result=assemble_parts(row,requests,parts)
            self.assertFalse(result['complete'],opposing)
            self.assertIn(result['reason'],('context_ownership_unresolved','boundary_uncertainty_unresolved'))
            self.assertIsNone(result['text'])
    def test_shared_context_missing_unit_or_changed_raw_unit_remains_unresolved(self):
        for defect in ('missing','different'):
            book,row=book_for();parts=repeated_parts(book,row)
            if defect=='missing':parts[1]=part(parts[1]['request'],'go 12.5',[(300000,310000),(400000,410000)])
            else:parts[1]=part(parts[1]['request'],'13.5 go 12.5',[(260000,270000),(300000,310000),(400000,410000)])
            result=assemble_parts(row,book.decode_requests(row['id']),parts)
            self.assertFalse(result['complete']);self.assertEqual(result['reason'],'context_ownership_unresolved')
    def test_unknown_timing_kind_and_missing_calibration_metadata_cannot_assemble(self):
        for key,value in [('timing_kind','invented_exact_word_edges'),('frame_calibration_id',None),
                          ('score_calibration_id',False),('frame_calibration_id',' '),('score_calibration_id',7)]:
            book,row=book_for(seconds=1);request=book.decode_requests(row['id'])[0]
            supplied=part(request,'go',[(1000,2000)]);supplied['alignment'][key]=value
            result=assemble_parts(row,[request],[supplied])
            self.assertFalse(result['complete']);self.assertEqual(result['reason'],'unresolved_alignment')
    def test_core_ownership_preserves_real_repeats_numbers_and_raw_spacing(self):
        book,row=book_for();parts=repeated_parts(book,row)
        assembled=assemble_parts(row,book.decode_requests(row['id']),parts)
        self.assertTrue(assembled['complete']);self.assertEqual(assembled['text'],'go  12.5 go\t12.5')
        self.assertEqual([w['text'] for w in assembled['words']],['go','12.5','go','12.5'])
        self.assertEqual([(w['start_sample'],w['end_sample']) for w in assembled['words']],
            [(1000,5000),(260000,270000),(300000,310000),(400000,410000)])
        self.assertEqual([assembled['text'][w['start_char']:w['end_char']] for w in assembled['words']],
            ['go','12.5','go','12.5'])
        row=book.apply_decode_parts(row['id'],parts,stage='refined')
        self.assertEqual(row['text'],assembled['text']);self.assertEqual(row['refinement_state'],'refined')
        self.assertEqual(row['alignment']['machine_revision'],row['machine_revision'])
        self.assertEqual(len(book.snapshot()),1)

    def test_unresolved_boundary_missing_part_or_weak_alignment_keeps_prior_text(self):
        for alteration in ('boundary','missing','weak','empty_core','calibration'):
            book,row=book_for();row=book.apply_model(row['id'],0,'Prior retained words',
                start_sample=0,end_sample=36*16000,stage='live',complete=True)
            parts=repeated_parts(book,row)
            if alteration=='boundary':parts[0]['alignment']['words'][-1].update(start_sample=287000,end_sample=289000)
            elif alteration=='missing':parts.pop()
            elif alteration=='weak':parts[0]['alignment']['words'][0]['status']='unresolved'
            elif alteration=='calibration':parts[1]['alignment']['frame_calibration_id']='other-grid'
            else:
                parts[1]=part(parts[1]['request'],'context',[(260000,270000)])
            result=book.apply_decode_parts(row['id'],parts,stage='refined')
            self.assertEqual(result['text'],row['text'],alteration)
            self.assertEqual(result['machine_revision'],row['machine_revision'])
            self.assertFalse(result['machine_versions'][-1]['complete'])
            self.assertNotIn('alignment',result)

    def test_stale_audio_revision_and_context_bounds_reject_before_any_mutation(self):
        for field in ('audio_revision','machine_revision','language_epoch','end_sample'):
            book,row=book_for();parts=repeated_parts(book,row);before=book.snapshot()
            parts[0]['request'][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):book.apply_decode_parts(row['id'],parts,stage='live')
            self.assertEqual(book.snapshot(),before)

    def test_human_correction_stays_primary_and_late_alignment_is_not_attached(self):
        book,row=book_for();row=book.edit(row['id'],0,'My own words')
        row=book.apply_decode_parts(row['id'],repeated_parts(book,row),stage='refined')
        self.assertEqual(row['text'],'My own words');self.assertNotIn('alignment',row)
        self.assertTrue(row['machine_versions'][-1]['complete'])
        self.assertEqual(row['machine_versions'][-1]['text'],'go  12.5 go\t12.5')

    def test_utf16_offsets_keep_original_unicode(self):
        book,row=book_for(seconds=1);request=book.decode_requests(row['id'])[0]
        raw='🙂 e\u0301 12.5'
        row=book.apply_decode_parts(row['id'],[part(request,raw,[(1000,2000),(3000,4000),(5000,6000)])],stage='refined')
        self.assertEqual(row['text'],raw)
        words=row['alignment']['words']
        self.assertEqual((words[1]['start_char'],words[1]['start_utf16']),(2,3))
        self.assertEqual((words[1]['end_char'],words[1]['end_utf16']),(4,5))

    def test_raw_contexts_archive_before_bounded_hot_history_and_failed_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=RevisionArchive(Path(folder)/'raw.jsonl')
            book,row=book_for(history_limit=2,archive=archive)
            for _ in range(5):row=book.apply_decode_parts(row['id'],repeated_parts(book,row),stage='refined')
            events=[json.loads(value) for value in archive.path.read_text().splitlines()]
            self.assertEqual(len(events),5);self.assertEqual(len(row['machine_versions']),2)
            self.assertEqual(events[0]['parts'][0]['text'],'go  12.5 go')
            self.assertNotIn('parts',row['machine_versions'][0])
            before=book.snapshot();book.archive=Mock();book.archive.append.side_effect=OSError('synthetic full disk')
            with self.assertRaises(OSError):book.apply_decode_parts(row['id'],repeated_parts(book,row),stage='refined')
            self.assertEqual(book.snapshot(),before)

    def test_direct_word_attachment_cannot_skip_or_change_raw_display_units(self):
        book,row=book_for(seconds=1);row=book.apply_model(row['id'],0,'go go',
            start_sample=0,end_sample=16000,stage='refined',complete=True)
        digest=hashlib.sha256(row['text'].encode()).hexdigest()
        words=[{'text':'go','start_char':0,'end_char':2,'start_sample':1000,'end_sample':2000}]
        with self.assertRaises(ValueError):book.attach_alignment(row['id'],row['machine_revision'],digest,words,audio_revision=row['audio_revision'])
        words.append({'text':'12','start_char':3,'end_char':5,'start_sample':3000,'end_sample':4000})
        with self.assertRaises(ValueError):book.attach_alignment(row['id'],row['machine_revision'],digest,words,audio_revision=row['audio_revision'])


if __name__=='__main__':unittest.main()
