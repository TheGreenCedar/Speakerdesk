"""Real canonical publication/reconciliation with fabricated CPU acoustic inputs.

These regressions prove temporal attribution behavior, not model accuracy. The
coordinated acoustic replay recipe is documented separately.
"""
import copy
import hashlib
import re
import unittest

import test_canonical_runtime as canonical_fixture
from meeting_refinement import activity_references
from reading_turns import CALIBRATION, MODEL, validated_turns
from transcript import export
from app import create_app
RATE = canonical_fixture.RATE


class CanonicalAttributionTests(unittest.TestCase):
    setUp = canonical_fixture.CanonicalTests.setUp
    tearDown = canonical_fixture.CanonicalTests.tearDown
    feed = canonical_fixture.CanonicalTests.feed
    rows = canonical_fixture.CanonicalTests.rows

    def activity(self, turns, text, times, *, alignment='complete'):
        self.peer.text = text
        original = self.peer.feed
        def feed(audio, final=False):
            start = self.peer.received / RATE
            _, observed = original(audio, final)
            end = self.peer.received / RATE
            return [dict(t, start=max(start, t['start']), end=min(end, t['end']))
                    for t in turns if t['start'] < end and t['end'] > start], observed
        self.peer.feed = feed
        self.peer.alignment_supported = lambda language: language == 'en'
        self.alignment_calls = []
        def align(request, raw, **kwargs):
            self.alignment_calls.append(copy.deepcopy(request))
            if alignment == 'absent':
                return None
            words = []
            for match, sample in zip(re.finditer(r'\S+', raw), times):
                timed = sample is not None and sample + 3200 <= request['end_sample']
                words.append(dict(text=match.group(), start_char=match.start(), end_char=match.end(),
                    start_sample=sample if timed else None, end_sample=sample + 3200 if timed else None,
                    status='aligned' if timed else 'unresolved'))
            return dict(raw_text=raw, text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                audio_anchor=dict(request), words=words, model_sha256=MODEL,
                timing_kind='ctc_emission_cell_envelope', frame_calibration_id=CALIBRATION,
                score_calibration_id=CALIBRATION)
        self.peer.align_canonical = align

    def assert_turns(self, row, owners):
        turns = row.get('reading_turns', [])
        self.assertEqual([t['speaker'] for t in turns], owners)
        self.assertEqual(''.join(t['text'] for t in turns), self.peer.text)
        self.assertFalse(row['voice_eligible'])
        return turns

    def test_open_and_sealed_five_second_a_then_b_survive_host_and_export(self):
        self.activity([dict(start=0, end=5, speaker='speaker_0'),
                       dict(start=5, end=10, speaker='speaker_1')],
                      '  Alpha café  Bravo 👩🏽‍💻  ', [RATE, 3*RATE, 6*RATE, 8*RATE])
        self.feed(0, 10)
        live = self.rows()[-1]
        self.assertEqual(live['canonical_state'], 'open')
        self.assert_turns(live, ['speaker_0', 'speaker_1'])
        self.engine.handle({'type': 'stop'})
        sealed = self.rows()[-1]
        self.assertEqual(sealed['canonical_state'], 'sealed')
        self.assertEqual(sealed['id'], live['id'])
        self.assert_turns(sealed, ['speaker_0', 'speaker_1'])
        app = create_app(self.root/'host');manager = app.extensions['speakerdesk']['meetings']
        manager.duration = 10
        job = dict(id=self.jid, created=1, status='ready', kind='meeting', name='CPU attribution',
            language='en', duration=10, revision=0, canonical_utterances=True,
            document=dict(speakers={}, segments=[], provenance={}, warnings=[]))
        try:
            manager.refinement.initialize(job);manager.put(job)
            manager.refinement.canonical(self.jid, dict(candidate=live, fast_sequence=1))
            manager.refinement.canonical(self.jid, dict(candidate=sealed, fast_sequence=2))
            document = manager.get(self.jid)['document'];saved = document['segments'][0]
            self.assertEqual(validated_turns(saved), sealed['reading_turns'])
            text, _ = export(document, 'txt')
            self.assertIn('Speaker 1:', text);self.assertIn('Speaker 2:', text)
            self.assertNotIn('Multiple speakers:', text);self.assertNotIn('Overlapping speakers:', text)
            for word in ('Alpha', 'café', 'Bravo', '👩🏽‍💻'):
                self.assertEqual(text.count(word), 1)
        finally:
            manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    def test_returning_a_keeps_three_turns_and_refinement_maps_local_tracks(self):
        turns = [dict(start=0, end=3, speaker='speaker_0'),
                 dict(start=3, end=6, speaker='speaker_1'),
                 dict(start=6, end=10, speaker='speaker_0')]
        self.activity(turns, '  First second third  ', [RATE, 4*RATE, 8*RATE])
        self.feed(0, 10)
        self.assert_turns(self.rows()[-1], ['speaker_0', 'speaker_1', 'speaker_0'])
        self.engine.handle({'type': 'stop'});row = self.rows()[-1]
        self.peer.batch_turns = lambda audio: [dict(t, speaker={
            'speaker_0': 'speaker_7', 'speaker_1': 'speaker_2'}[t['speaker']]) for t in turns]
        request = dict(type='refine', canonical=row, operation_id='attribution', language_epoch=0,
            language='en', window=dict(id=row['id'], start_sample=0, end_sample=10*RATE),
            references=activity_references([row], 0, 10*RATE))
        refined = self.engine.refine(request)['canonical_candidate']
        self.assertEqual(refined['refinement_state'], 'refined')
        self.assertEqual((refined['id'], refined['audio_revision']), (row['id'], row['audio_revision']))
        self.assert_turns(refined, ['speaker_0', 'speaker_1', 'speaker_0'])

    def test_only_simultaneous_activity_produces_an_overlap_turn(self):
        self.activity([dict(start=0, end=6, speaker='speaker_0'),
                       dict(start=4, end=10, speaker='speaker_1')],
                      'left together right', [RATE, 5*RATE, 8*RATE])
        self.feed(0, 10)
        self.assert_turns(self.rows()[-1], ['speaker_0', 'overlap_speaker_0_speaker_1', 'speaker_1'])

    def test_activity_gap_and_null_alignment_words_stay_unknown(self):
        self.activity([dict(start=0, end=3, speaker='speaker_0'),
                       dict(start=5, end=10, speaker='speaker_1')],
                      'left gap missing right', [RATE, 4*RATE, None, 8*RATE])
        self.feed(0, 10)
        turns = self.assert_turns(self.rows()[-1], ['speaker_0', 'unassigned', 'speaker_1'])
        self.assertEqual(turns[1]['text'], 'gap missing ')
        self.assertIsNone(turns[1]['start']);self.assertIsNone(turns[1]['end'])

    def test_absent_alignment_preserves_raw_text_and_activity_without_union_attribution(self):
        self.activity([dict(start=0, end=5, speaker='speaker_0'),
                       dict(start=5, end=10, speaker='speaker_1')],
                      '  Full raw text unchanged.  ', [None]*4, alignment='absent')
        self.feed(0, 9);self.engine.handle({'type': 'stop'})
        row = self.rows()[-1]
        self.assertEqual(row['text'], self.peer.text)
        self.assertEqual(row['source_speaker_candidates'], ['speaker_0', 'speaker_1'])
        self.assertEqual(row['speaker'], 'unassigned')
        self.assertTrue(row['review']);self.assertFalse(row['voice_eligible'])
        self.assertNotIn('reading_turns', row)
        refs = activity_references([row], row['start_sample'], row['end_sample'])
        self.assertTrue(all(len(r['speaker_candidates']) <= 1 for r in refs))

    def test_appended_audio_retains_text_prefix_turns_until_next_decode_and_shrink_invalidates(self):
        self.activity([dict(start=0, end=5, speaker='speaker_0'),
                       dict(start=5, end=12, speaker='speaker_1')],
                      'Alpha café Bravo next', [RATE, 3*RATE, 6*RATE, 8*RATE])
        self.feed(0, 10);decoded = self.rows()[-1]
        self.assert_turns(decoded, ['speaker_0', 'speaker_1'])
        calls = len(self.alignment_calls)
        self.feed(10, 12);extended = self.rows()[-1]
        self.assert_turns(extended, ['speaker_0', 'speaker_1'])
        self.assertEqual(extended['text_audio_anchor'], decoded['text_audio_anchor'])
        self.assertGreater(extended['audio_revision'], decoded['audio_revision'])
        self.assertEqual(len(self.alignment_calls), calls)
        book = self.engine.canonical.book;row = book.rows[extended['id']]
        book._set_end(row, row['end_sample'] - 1)
        self.assertNotIn('reading_word_evidence', row)

    def test_ambiguous_refinement_track_cannot_claim_a_single_or_overlapping_speaker(self):
        self.activity([dict(start=0, end=5, speaker='speaker_0'),
                       dict(start=5, end=10, speaker='speaker_1')],
                      'Alpha Bravo', [RATE, 8*RATE])
        self.feed(0, 10);self.engine.handle({'type': 'stop'});row = self.rows()[-1]
        self.peer.batch_turns = lambda audio: [dict(start=0, end=10, speaker='speaker_7')]
        request = dict(type='refine', canonical=row, operation_id='ambiguous', language_epoch=0,
            language='en', window=dict(id=row['id'], start_sample=0, end_sample=10*RATE),
            references=activity_references([row], 0, 10*RATE))
        refined = self.engine.refine(request)['canonical_candidate']
        self.assertEqual(refined['speaker'], 'unassigned')
        self.assertEqual(refined['source_speaker_candidates'], ['unknown_mixed'])
        self.assertEqual(refined['text'], self.peer.text)
        self.assertNotIn('reading_turns', refined)
        self.assertTrue(refined['review']);self.assertFalse(refined['voice_eligible'])


if __name__ == '__main__':
    unittest.main()
