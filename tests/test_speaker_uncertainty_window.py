"""Ownership contracts; retained actual model evidence, no new model execution."""
import copy
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get('SPEAKER_UNCERTAINTY_SOURCE_ROOT',
    Path(__file__).resolve().parents[1])) / 'speakerdesk'))
from reading_turns import alignment_words, bind_words, project_turns
from test_reading_turns import fixture


def row_for(word_start, word_end, regions):
    row = fixture('word')
    row['speaker_activity']['regions'] = regions
    row['reading_word_evidence']['words'][0].update(start_sample=word_start, end_sample=word_end)
    return row


def region(a, b, *owners):
    return {'start_sample': a, 'end_sample': b, 'speakers': list(owners)}


class SpeakerUncertaintyWindowTests(unittest.TestCase):
    def test_actual_two_voice_raw_evidence_recovers_only_existing_owners(self):
        data = json.loads((Path(__file__).parent / 'acceptance/actual-two-voice-boundary-evidence.json').read_text())
        row = data['segment']
        words = alignment_words(data['alignment'], row['text'], row['start_sample'], row['end_sample'])
        bind_words(row, words)
        turns = project_turns(row)
        self.assertEqual(''.join(t['text'] for t in turns), row['text'])
        self.assertEqual([t['speaker'] for t in turns], ['speaker_0', 'speaker_1'])
        self.assertTrue(all(t['attribution'] == 'single' for t in turns))
        self.assertEqual(sum(len(t['text']) for t in turns), 145)
        # The CTC envelope still remains an emission cell, not a phonetic edge.
        self.assertEqual(turns[0]['start'], 640 / 16000)

    def test_gap_without_competing_owner_does_not_cancel_inside_word(self):
        row = row_for(74000, 78720, [region(0, 79280, 'speaker_0'),
            region(79280, 85000), region(85000, 160000, 'speaker_1')])
        self.assertEqual(project_turns(row)[0]['speaker'], 'speaker_0')

    def test_calibrated_designing_counterexample_with_competitor_stays_unknown(self):
        # Frozen AMI counterexample: emission ends before seam, reference crosses.
        row = row_for(74000, 78720, [region(0, 79280, 'speaker_0'),
            region(79280, 160000, 'speaker_1')])
        self.assertEqual(project_turns(row)[0]['attribution'], 'unknown')

    def test_word_entirely_in_gap_never_borrows_neighbour(self):
        row = row_for(80500, 81000, [region(0, 80000, 'speaker_0'), region(80000, 160000)])
        self.assertEqual(project_turns(row)[0]['attribution'], 'unknown')

    def test_competitor_after_short_gap_and_actual_overlap_remain_uncertain(self):
        for names in [('speaker_1',), ('speaker_0', 'speaker_1')]:
            with self.subTest(names=names):
                row = row_for(75000, 79000, [region(0, 80000, 'speaker_0'),
                    region(80000, 81000), region(81000, 160000, *names)])
                self.assertEqual(project_turns(row)[0]['attribution'], 'unknown')
        row = row_for(90000, 93000, [region(0, 80000, 'speaker_0'),
            region(80000, 160000, 'speaker_0', 'speaker_1')])
        self.assertEqual(project_turns(row)[0]['speaker'], 'overlap_speaker_0_speaker_1')

    def test_retained_text_prefix_checks_competitor_in_new_audio(self):
        row = row_for(75000, 79000, [region(0, 80000, 'speaker_0'),
            region(80000, 160000, 'speaker_1')])
        row['text_audio_anchor']['end_sample'] = 80000
        row['reading_word_evidence']['audio_anchor'] = copy.deepcopy(row['text_audio_anchor'])
        self.assertEqual(project_turns(row)[0]['attribution'], 'unknown')


if __name__ == '__main__':
    unittest.main()
