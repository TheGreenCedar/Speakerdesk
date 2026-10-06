"""Pure-data alignment contracts; invented anchors are not acoustic evidence.

No audio, acoustic model, ONNX Runtime, CTC extension, download or native build
is used here. These checks own raw-text preservation, null timing and CAS guards.
"""
import copy
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from word_alignment import (AcceptancePolicy, AlignmentRequest, FrameClock,
    TokenAnchor, align_ctc_scores, annotate_speakers, attachment_for, materialize,
    parse_vocabulary, prepare_text)

VOCABULARY = ('<s>', '<pad>', '</s>', '<unk>', ' ', 'g', 'o', '1', '2', '5',
              'é', 'a', 'أ', 'ه', 'ل', 'ا', 'ε', '🙂')
CLOCK = FrameClock(320, 0, 'invented-fixture-grid')
POLICY = AcceptancePolicy(-1., 'invented-fixture-policy')


def mapped(text, anchors, *, clock=CLOCK, policy=POLICY, samples=8000):
    return materialize(prepare_text(text, VOCABULARY), anchors, clock=clock,
                       policy=policy, audio_start_sample=16000, audio_num_samples=samples)


def utterance(text='go go'):
    return {'id': 'utterance-fixture', 'machine_revision': 3, 'audio_revision': 4,
            'text': text, 'start_sample': 16000, 'end_sample': 24000,
            'text_audio_anchor': {'start_sample': 16000, 'end_sample': 24000}}


REPEATED_WORDS = (TokenAnchor(0, 5, 1, -.1), TokenAnchor(1, 6, 2, -.1),
    TokenAnchor(2, 4, 3, -.1), TokenAnchor(3, 5, 4, -.1), TokenAnchor(4, 6, 5, -.1))


class TextMappingTests(unittest.TestCase):
    def test_missing_capital_tokens_map_without_changing_cohere_or_attachment_offsets(self):
        raw = 'Go GO'
        prepared = prepare_text(raw,VOCABULARY)
        self.assertFalse(prepared.unsupported)
        self.assertEqual([t.text for t in prepared.targets],['g','o',' ','g','o'])
        result = mapped(raw,REPEATED_WORDS)
        self.assertTrue(result['complete'])
        self.assertEqual(result['raw_text'],raw)
        self.assertEqual(result['text_sha256'],hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual([c['raw_text'] for c in result['characters']],list(raw))
        self.assertEqual([(w['text'],w['start_char'],w['end_char']) for w in result['words']],
                         [('Go',0,2),('GO',3,5)])
        self.assertEqual([(a['raw_text'],a['acoustic_token'],a['start_char'])
                          for a in result['acoustic_aliases']],[('G','g',0),('G','g',3),('O','o',4)])
        row = utterance(raw)
        attachment = attachment_for(result,AlignmentRequest.from_utterance(row),row)
        self.assertEqual([w['text'] for w in attachment['words']],['Go','GO'])
        self.assertTrue(attachment['words'][0]['acoustic_aliases'])

    def test_supported_case_is_preserved_and_expanding_lowercase_stays_unresolved(self):
        exact = prepare_text('G',VOCABULARY+('G',))
        self.assertEqual(exact.targets[0].text,'G')
        self.assertFalse(exact.acoustic_aliases)
        expanding = prepare_text('İ',VOCABULARY+('i','\u0307'))
        self.assertEqual(expanding.unsupported,({'start_char':0,'end_char':1,'text':'İ'},))
        result = materialize(expanding,(),clock=CLOCK,policy=POLICY,
                             audio_start_sample=16000,audio_num_samples=8000)
        self.assertEqual((result['reason'],result['raw_text']),('unsupported_text','İ'))
        self.assertIsNone(result['words'][0]['start_sample'])

    def test_unicode_composition_and_both_offset_units_preserve_raw_cohere(self):
        raw = '🙂 e\u0301 أَهلا'
        prepared = prepare_text(raw, VOCABULARY)
        self.assertEqual(prepared.raw_text, raw)
        self.assertEqual(prepared.text_sha256, hashlib.sha256(raw.encode()).hexdigest())
        accent = next(t for t in prepared.targets if t.text == 'é')
        self.assertEqual((accent.start_char, accent.end_char), (2, 4))
        self.assertEqual(raw[accent.start_char:accent.end_char], 'e\u0301')
        # Unsupported Arabic vowel mark must keep all text and prevent timing,
        # rather than silently aligning a text with the mark removed.
        result = mapped(raw, ())
        self.assertEqual(result['reason'], 'unsupported_text')
        self.assertEqual(result['words'][1]['text'], 'e\u0301')
        self.assertEqual((result['words'][1]['start_char'], result['words'][1]['start_utf16']), (2, 3))
        self.assertTrue(all(w['start_sample'] is None for w in result['words']))

    def test_digits_decimal_and_repeated_words_are_never_normalized_or_deduplicated(self):
        prepared = prepare_text('12.5 12.5', VOCABULARY)
        self.assertEqual([t.text for t in prepared.targets], ['1','2','5',' ','1','2','5'])
        self.assertEqual([(t.start_char,t.end_char) for t in prepared.targets[:3]], [(0,1),(1,2),(3,4)])
        result = mapped('go go', REPEATED_WORDS)
        self.assertEqual([(w['text'],w['start_char'],w['end_char']) for w in result['words']],
                         [('go',0,2),('go',3,5)])
        self.assertEqual([(w['start_sample'],w['end_sample']) for w in result['words']],
                         [(16320,16960),(17280,17920)])
        self.assertTrue(result['complete'])

    def test_space_token_and_greek_epsilon_are_data_not_library_markers(self):
        vocabulary = parse_vocabulary('<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n  4\nε 5\n')
        self.assertEqual(vocabulary[4], ' ')
        self.assertEqual([t.token_id for t in prepare_text(' ε\tε ', vocabulary).targets], [5,4,5])
        with self.assertRaises(ValueError):parse_vocabulary('<s> 0\n<pad> 0\n')


class TimingEvidenceTests(unittest.TestCase):
    def test_unverified_clock_does_not_consume_scores_or_optional_runtime(self):
        class UnreadScores:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Unverified alignment must not consume scores.')
        result = align_ctc_scores('go', UnreadScores(), VOCABULARY,
            clock=FrameClock(320), policy=None, audio_start_sample=16000, audio_num_samples=8000)
        self.assertEqual(result['reason'], 'frame_clock_unverified')
        self.assertEqual(result['raw_text'], 'go')

    def test_unverified_clock_or_score_policy_never_materializes_timestamps(self):
        for clock, policy, reason in [(FrameClock(320), POLICY, 'frame_clock_unverified'),
                                     (CLOCK, None, 'acceptance_policy_unverified')]:
            with self.subTest(reason=reason):
                result = mapped('go go', REPEATED_WORDS, clock=clock, policy=policy)
                self.assertEqual(result['reason'], reason)
                self.assertFalse(result['complete'])
                self.assertTrue(all(w['start_sample'] is None for w in result['words']))

    def test_repeated_character_requires_observed_blank_separator(self):
        unsupported = mapped('oo', (TokenAnchor(0,6,1,-.1),TokenAnchor(1,6,3,-.1)))
        self.assertFalse(unsupported['complete'])
        self.assertIsNone(unsupported['words'][0]['start_sample'])
        aligned = mapped('oo', (TokenAnchor(0,6,1,-.1),TokenAnchor(1,6,3,-.1,2,-.1)))
        self.assertTrue(aligned['complete'])
        self.assertEqual((aligned['words'][0]['start_sample'],aligned['words'][0]['end_sample']), (16320,17280))

    def test_weak_outside_reordered_or_nonfinite_anchors_do_not_extrapolate(self):
        cases = [(TokenAnchor(0,5,1,-5.),TokenAnchor(1,6,2,-.1)),
                 (TokenAnchor(0,5,24,-.1),TokenAnchor(1,6,25,-.1)),
                 (TokenAnchor(0,5,2,-.1),TokenAnchor(1,6,1,-.1)),
                 (TokenAnchor(0,5,1,float('nan')),TokenAnchor(1,6,2,-.1))]
        for anchors in cases:
            with self.subTest(anchors=anchors):
                result = mapped('go', anchors)
                self.assertEqual(result['raw_text'], 'go')
                self.assertFalse(result['complete'])
                self.assertIsNone(result['words'][0]['start_sample'])

    def test_blank_alignment_never_becomes_non_speech_proof(self):
        for raw in ('', '   ', '...'):
            result = mapped(raw, ())
            self.assertEqual(result['reason'], 'no_acoustic_target')
            self.assertFalse(result['non_speech_proof'])
            self.assertFalse(result['complete'])

    def test_overlap_and_activity_gaps_preserve_unknown_owner(self):
        result = mapped('go', REPEATED_WORDS[:2])
        overlap = annotate_speakers(result,[{'start_sample':16000,'end_sample':18000,'speaker':'s0'},
                                           {'start_sample':16500,'end_sample':18000,'speaker':'s1'}])[0]
        self.assertEqual((overlap['speaker'],overlap['speaker_candidates']), ('unknown',['s0','s1']))
        self.assertEqual(overlap['speaker_state'], 'multiple_speakers')
        gap = annotate_speakers(result,[{'start_sample':16320,'end_sample':16600,'speaker':'s0'}])[0]
        self.assertEqual((gap['speaker'],gap['speaker_candidates']), ('unknown',['s0']))
        clean = annotate_speakers(result,[{'start_sample':16000,'end_sample':18000,'speaker':'s0'}])[0]
        self.assertEqual(clean['speaker'], 's0')


class AttachmentTests(unittest.TestCase):
    def test_attachment_rejects_unknown_timing_kind_or_untyped_calibration_ids(self):
        row=utterance();request=AlignmentRequest.from_utterance(row)
        for key,value in [('timing_kind','anything'),('frame_calibration_id',None),
                          ('score_calibration_id',None),('frame_calibration_id',''),('score_calibration_id',True)]:
            result=mapped('go go',REPEATED_WORDS);result[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):attachment_for(result,request,row)
    def test_audio_growth_text_edits_and_revision_changes_reject_late_attachment(self):
        row = utterance();request = AlignmentRequest.from_utterance(row)
        result = mapped('go go', REPEATED_WORDS)
        attachment = attachment_for(result, request, row)
        self.assertEqual((attachment['machine_revision'],attachment['audio_revision']), (3,4))
        self.assertEqual([w['text'] for w in attachment['words']], ['go','go'])
        for key, value in [('text','go 12'),('machine_revision',4),('audio_revision',5),('end_sample',25000)]:
            changed=copy.deepcopy(row);changed[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):attachment_for(result, request, changed)

    def test_changed_numbers_or_missing_word_coverage_cannot_attach(self):
        row=utterance();request=AlignmentRequest.from_utterance(row)
        result=mapped('go go', REPEATED_WORDS)
        for altered in [dict(result, raw_text='go 12'),dict(result, words=result['words'][:1])]:
            with self.assertRaises(ValueError):attachment_for(altered, request, row)
        old_anchor=utterance();old_anchor['text_audio_anchor']['end_sample']=20000
        with self.assertRaises(ValueError):AlignmentRequest.from_utterance(old_anchor)


if __name__ == '__main__':
    unittest.main()
