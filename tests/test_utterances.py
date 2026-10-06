"""CPU lifecycle contract tests, not acoustic or alignment accuracy evidence."""
import hashlib
import sys
import json
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from utterances import UtteranceBook, RevisionArchive, MAX_DECODE_SAMPLES


def observed(start, end, regions):
    return {'start_sample': start, 'end_sample': end, 'complete': True,
            'decision': 'speech' if regions else 'no_speech',
            'speech_regions': [{'start_sample': a, 'end_sample': b} for a, b in regions]}


class UtteranceTests(unittest.TestCase):
    def test_stable_identity_across_growing_live_and_full_refined_replacement(self):
        book = UtteranceBook('job');row = book.observe(observed(0, 8000, [(1000, 5000)]))[0]
        identity = row['id']
        row = book.apply_model(identity, 0, 'Hello', start_sample=0, end_sample=8000, stage='live', complete=True)
        row = book.observe(observed(8000, 16000, [(9000, 15000)]))[0]
        self.assertEqual(row['id'], identity)
        row = book.apply_model(identity, row['machine_revision'], 'Hello there', start_sample=0, end_sample=16000, stage='live', complete=True)
        book.observe(observed(16000, 24000, []));row = book.snapshot()[0]
        row = book.apply_model(identity, row['machine_revision'], 'Hello there.', start_sample=0, end_sample=18200, stage='refined', complete=True)
        self.assertEqual(row['text'], 'Hello there.')
        self.assertEqual(len(book.snapshot()), 1)
        self.assertEqual(row['refinement_state'], 'refined')

    def test_human_edit_survives_model_candidate_and_old_alignment_cannot_attach(self):
        book = UtteranceBook('job');row = book.observe(observed(0, 8000, [(0, 5000)]))[0]
        row = book.edit(row['id'], 0, 'My correction')
        row = book.apply_model(row['id'], row['machine_revision'], 'Machine version', start_sample=0, end_sample=8000, stage='live', complete=True)
        self.assertEqual(row['text'], 'My correction')
        self.assertEqual(row['machine_versions'][-1]['text'], 'Machine version')
        with self.assertRaises(ValueError):book.attach_alignment(row['id'], 0, hashlib.sha256(b'Machine version').hexdigest(), [], audio_revision=row['audio_revision'])

    def test_language_boundary_seals_and_never_borrows_previous_epoch(self):
        book = UtteranceBook('job');book.observe(observed(0, 8000, [(0, 7000)]));book.boundary(8000, 1)
        rows = book.observe(observed(8000, 16000, [(8100, 14000)]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['state'], 'sealed')
        self.assertEqual(rows[1]['start_sample'], 8000)
        self.assertEqual(rows[1]['language_epoch'], 1)

    def test_pending_evidence_cannot_close_or_invent_an_utterance(self):
        book = UtteranceBook('job');e = observed(0, 8000, []);e['complete'] = False
        with self.assertRaises(ValueError):book.observe(e)
        self.assertEqual(book.snapshot(), [])

    def test_empty_decoder_result_keeps_prior_words_and_cannot_clear_them(self):
        book = UtteranceBook('job');row = book.observe(observed(0, 8000, [(0, 5000)]))[0]
        row = book.apply_model(row['id'], 0, 'Retained words', start_sample=0, end_sample=8000, stage='live', complete=True)
        revision = row['machine_revision']
        row = book.apply_model(row['id'], revision, ' ', start_sample=0, end_sample=8000, stage='refined', complete=True)
        self.assertEqual(row['text'], 'Retained words')
        self.assertEqual(row['machine_revision'], revision)
        self.assertFalse(row['machine_versions'][-1]['complete'])

    def test_audio_extension_rejects_old_alignment_even_with_unchanged_text(self):
        book = UtteranceBook('job');row = book.observe(observed(0, 8000, [(0, 5000)]))[0]
        row = book.apply_model(row['id'], 0, 'Hello', start_sample=0, end_sample=8000, stage='live', complete=True)
        revision, audio_revision = row['machine_revision'], row['audio_revision']
        digest = hashlib.sha256(b'Hello').hexdigest()
        words = [{'text': 'Hello', 'start_char':0,'end_char':5,'start_sample': 1000, 'end_sample': 4500}]
        book.attach_alignment(row['id'], revision, digest, words, audio_revision=audio_revision)
        row = book.observe(observed(8000, 16000, [(9000, 15000)]))[0]
        self.assertNotIn('alignment', row)
        self.assertEqual(row['machine_revision'], revision)
        with self.assertRaises(ValueError):book.attach_alignment(row['id'], revision, digest, words, audio_revision=audio_revision)
        with self.assertRaises(ValueError):book.apply_model(row['id'], revision, 'Old decode', start_sample=0, end_sample=8000, stage='live', complete=True)

    def test_finish_rejects_new_audio_or_language_boundary(self):
        book = UtteranceBook('job');book.observe(observed(0, 8000, [(0, 5000)]));before = book.finish()
        with self.assertRaises(ValueError):book.observe(observed(8000, 16000, [(9000, 15000)]))
        with self.assertRaises(ValueError):book.boundary(8000, 1)
        self.assertEqual(book.finish(), before)

    def test_bad_region_does_not_partially_modify_state(self):
        book = UtteranceBook('job')
        with self.assertRaises(ValueError):book.observe(observed(0, 8000, [(0, 5000), (4000, 6000)]))
        self.assertEqual(book.snapshot(), [])
        self.assertEqual(book.cursor, 0)

    def test_boolean_revisions_and_missing_word_positions_fail_closed(self):
        book = UtteranceBook('job');row = book.observe(observed(0, 8000, [(0, 5000)]))[0]
        with self.assertRaises(ValueError):book.edit(row['id'], False, 'Wrong revision')
        row = book.edit(row['id'], 0, 'Human words')
        digest = hashlib.sha256(b'Human words').hexdigest()
        with self.assertRaises(ValueError):book.attach_alignment(row['id'], row['machine_revision'], digest, [], audio_revision=row['audio_revision'])

    def test_continuous_vad_never_assigns_two_voices_to_one_clean_voice(self):
        book = UtteranceBook('job');book.observe(observed(0, 8000, [(0, 8000)]));row = book.finish()[0]
        regions = [{'start_sample': 0, 'end_sample': 4000, 'speakers': ['speaker_0']},
                   {'start_sample': 4000, 'end_sample': 8000, 'speakers': ['speaker_1']}]
        row = book.attach_activity(row['id'], row['audio_revision'], regions)
        self.assertFalse(row['voice_eligible'])
        self.assertEqual(row['speaker_candidates'], ['speaker_0', 'speaker_1'])
        self.assertEqual(len(book.snapshot()), 1)
        self.assertEqual(row['speaker_activity']['regions'], regions)

    def test_missing_overlap_and_stale_speaker_evidence_are_not_clean_voice(self):
        for owners in ([], ['speaker_0', 'speaker_1'], ['unknown_mixed'], ['overlap_unknown'], ['unassigned']):
            book = UtteranceBook('job');book.observe(observed(0, 8000, [(0, 8000)]));row = book.finish()[0]
            row = book.attach_activity(row['id'], row['audio_revision'],
                [{'start_sample': 0, 'end_sample': 8000, 'speakers': owners}])
            self.assertFalse(row['voice_eligible'])
        book = UtteranceBook('job');row = book.observe(observed(0, 8000, [(0, 5000)]))[0]
        old = row['audio_revision'];row = book.observe(observed(8000, 16000, [(9000, 15000)]))[0]
        with self.assertRaises(ValueError):book.attach_activity(row['id'], old,
            [{'start_sample': 0, 'end_sample': 8000, 'speakers': ['speaker_0']}])
        self.assertFalse(book.snapshot()[0]['voice_eligible'])

    def test_vad_negative_context_does_not_erase_a_second_nvidia_voice(self):
        book = UtteranceBook('job');book.observe(observed(0, 8000, [(0, 5000)]));row = book.finish()[0]
        regions = [{'start_sample': 0, 'end_sample': 5000, 'speakers': ['speaker_0']},
                   {'start_sample': 5000, 'end_sample': 8000, 'speakers': ['speaker_1']}]
        row = book.attach_activity(row['id'], row['audio_revision'], regions)
        self.assertFalse(row['voice_eligible'])
        self.assertEqual(row['speaker_candidates'], ['speaker_0', 'speaker_1'])

    def test_clean_single_nvidia_voice_with_empty_context_can_be_eligible(self):
        book = UtteranceBook('job');book.observe(observed(0, 8000, [(0, 5000)]));row = book.finish()[0]
        regions = [{'start_sample': 0, 'end_sample': 5000, 'speakers': ['speaker_0']},
                   {'start_sample': 5000, 'end_sample': 8000, 'speakers': []}]
        row = book.attach_activity(row['id'], row['audio_revision'], regions)
        self.assertTrue(row['voice_eligible'])

    def test_hot_versions_are_bounded_only_after_raw_versions_are_archived(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=RevisionArchive(Path(folder)/'versions.jsonl')
            book=UtteranceBook('job',history_limit=2,archive=archive)
            row=book.observe(observed(0,8000,[(0,5000)]))[0]
            for n in range(12):
                row=book.apply_model(row['id'],row['machine_revision'],f'Raw {n}',
                    start_sample=0,end_sample=8000,stage='live',complete=True)
            self.assertEqual(len(row['machine_versions']),2)
            events=[json.loads(line) for line in archive.path.read_text().splitlines()]
            self.assertEqual(len(events),12)
            self.assertEqual([e['version']['text'] for e in events],[f'Raw {n}' for n in range(12)])
        with self.assertRaises(ValueError):UtteranceBook('job',history_limit=2)

    def test_continuous_utterance_has_bounded_overlapping_context_requests(self):
        book=UtteranceBook('job');row=book.observe(observed(0,90*16000,[(0,90*16000)]))[0]
        requests=book.decode_requests(row['id'])
        self.assertEqual(len(requests),5)
        self.assertEqual(requests[0]['core_start_sample'],0)
        self.assertEqual(requests[-1]['core_end_sample'],90*16000)
        self.assertTrue(all(b['core_start_sample']==a['core_end_sample'] for a,b in zip(requests,requests[1:])))
        self.assertTrue(all(r['end_sample']-r['start_sample']<=MAX_DECODE_SAMPLES for r in requests))
        self.assertEqual(len({r['utterance_id'] for r in requests}),1)


if __name__ == '__main__':unittest.main()
