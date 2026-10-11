"""Generate transcript goldens: validate() accept/reject and export() outputs.

Run with the oracle venv against the detached v0.6.6 worktree:
    PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_transcript.py
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_lib as oracle  # noqa: E402

oracle.activate_oracle()
import reading_turns  # noqa: E402
import transcript  # noqa: E402

oracle.imported_from_worktree(reading_turns, transcript)


def valid_doc():
    """One document reused by the export cases; covers the projection cases."""
    return {
        'schema_version': 1,
        'speakers': {'speaker_0': 'Ada Lovelace',
                     'speaker_1': 'Ben Gunn',
                     'unassigned': 'Unknown speaker'},
        'segments': [
            {'id': 's-hello', 'speaker': 'speaker_0',
             'start': 0.5, 'end': 4.25, 'text': 'Hello, everyone.'},
            {'id': 's-unassigned', 'speaker': 'unassigned',
             'start': 4.75, 'end': 7.0, 'text': 'A stray voice joins.'},
            {'id': 's-symbols', 'speaker': 'speaker_1', 'start': 7.5, 'end': 11.0,
             'text': 'Arrows --> and <tags> and café 👩🏽‍💻 unicode.'},
            {'id': 's-blank', 'speaker': 'speaker_1',
             'start': 12.0, 'end': 13.5, 'text': '   '},
            {'id': 's-overlap', 'speaker': 'overlap_speaker_0_speaker_1',
             'start': 14.0, 'end': 16.0, 'text': 'Both speak at once.'},
            {'id': 's-coarse', 'speaker': 'speaker_0', 'start': 17.0, 'end': 21.0,
             'text': 'Only coarse timing survived.', 'coarse_timing': True},
            {'id': 's-parent-slice', 'speaker': 'speaker_1',
             'start': 22.0, 'end': 25.5, 'text': 'Reading slice parent timing.',
             'reading_slice': True},
            sliced_segment('slice-turns', 26.0, 32.0,
                           text='one two three four five six',
                           region_words=[(2, ['speaker_0']),
                                         (2, ['speaker_0', 'speaker_1']),
                                         (2, ['speaker_1'])],
                           unknown_word=None),
            sliced_segment('slice-mixed', 33.0, 38.0,
                           text='alpha beta gamma delta',
                           region_words=[(2, ['speaker_0']),
                                         (2, ['speaker_1'])],
                           unknown_word=1),
        ],
    }


def sliced_segment(segment_id, start, end, *, text, region_words, unknown_word):
    """Fabricate a stored segment whose reading_turns pass validated_turns().

    region_words partitions the words of text into speaker-activity regions:
    each entry is (word_count, [owner tracks]) and determines whether the
    projected turns read as single, overlap or unknown attribution.
    """
    rate = 16000
    row = {'id': segment_id, 'canonical_utterance_id': segment_id,
           'machine_revision': 0, 'canonical_machine_revision': 0,
           'audio_revision': 0, 'language_epoch': 0,
           'start_sample': round(start * rate), 'end_sample': round(end * rate),
           'start': start, 'end': end, 'text': text,
           'text_audio_anchor': {'start_sample': round(start * rate),
                                 'end_sample': round(end * rate)},
           'protected_fields': [], 'speaker': 'speaker_0',
           'speaker_candidates': ['speaker_0']}
    matches = list(re.finditer(r'\S+', text))
    assert sum(count for count, _ in region_words) == len(matches)
    width = (end - start) * rate / len(matches)
    regions, words, index = [], [], 0
    for count, owners in region_words:
        region_start = round(start * rate + index * width)
        region_end = round(start * rate + (index + count) * width)
        regions.append({'start_sample': region_start,
                        'end_sample': region_end, 'speakers': owners})
        for _ in range(count):
            match = matches[index]
            # Words sit at slot centers so project_turns' MARGIN (0.25 s)
            # uncertainty window never crosses a region boundary.
            a = round(start * rate + (index + 0.3) * width)
            b = round(start * rate + (index + 0.7) * width)
            if index == unknown_word:
                a = b = None
            words.append({'text': match.group(), 'start_char': match.start(),
                          'end_char': match.end(), 'start_sample': a,
                          'end_sample': b})
            index += 1
    row['speaker_activity'] = {'audio_revision': 0,
                               'observed_end_sample': round(end * rate),
                               'regions': regions}
    reading_turns.bind_words(row, words)
    row['reading_turns'] = reading_turns.project_turns(row)
    row['reading_turn_provenance'] = {
        'utterance_id': row['id'], 'machine_revision': row['machine_revision'],
        'canonical_machine_revision': row['canonical_machine_revision'],
        'audio_revision': row['audio_revision'],
        'language_epoch': row['language_epoch'],
        'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
        'audio_anchor': row['text_audio_anchor'],
        'activity_sha256': hashlib.sha256(json.dumps(
            row['speaker_activity'], sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        'turns_sha256': hashlib.sha256(json.dumps(
            row['reading_turns'], sort_keys=True, separators=(',', ':'),
            ensure_ascii=False).encode()).hexdigest(),
        'calibration_id': reading_turns.CALIBRATION,
        'model_sha256': reading_turns.MODEL}
    assert reading_turns.validated_turns(row), 'fabricated projection rejected'
    return row


def base_document():
    """Small valid document used by the validate cases."""
    return {'speakers': {'speaker_0': 'Ada Lovelace',
                         'speaker_1': 'Ben Gunn'},
            'segments': [
                {'id': 'a', 'speaker': 'speaker_1', 'start': 6.0, 'end': 9.5,
                 'text': 'Later passage.'},
                {'id': 'b', 'speaker': 'speaker_0', 'start': 0.5, 'end': 4.0,
                 'text': 'Earlier passage.'}]}


def run():
    duration = 30.0
    valid = base_document()

    cases = [
        # validate(): ordering, id defaulting, schema_version normalization.
        ('validate-accept-sorted', valid, duration),
        ('validate-accept-default-ids',
         {'speakers': {'speaker_0': '  Padded Name  '},
          'segments': [{'speaker': 'speaker_0', 'start': 1, 'end': 2,
                        'text': 'No id supplied.'}]}, 30.0),
        # Rejections, one per rule.
        ('validate-reject-not-object', ['not', 'a', 'dict'], duration),
        ('validate-reject-no-speakers',
         {'speakers': {}, 'segments': []}, duration),
        ('validate-reject-speaker-name-blank',
         {'speakers': {'speaker_0': '   '}, 'segments': []}, duration),
        ('validate-reject-segment-not-object',
         {'speakers': {'speaker_0': 'A'}, 'segments': ['x']}, duration),
        ('validate-reject-non-numeric-times',
         {'speakers': {'speaker_0': 'A'},
          'segments': [{'speaker': 'speaker_0', 'start': 'late', 'end': 9,
                        'text': 'x'}]}, duration),
        ('validate-reject-times-out-of-range',
         {'speakers': {'speaker_0': 'A'},
          'segments': [{'speaker': 'speaker_0', 'start': 29.0, 'end': 30.5,
                        'text': 'x'}]}, duration),
        ('validate-reject-nonfinite-times',
         {'speakers': {'speaker_0': 'A'},
          'segments': [{'speaker': 'speaker_0', 'start': 0,
                        'end': {'$float': 'inf'}, 'text': 'x'}]}, duration),
        ('validate-reject-unknown-speaker',
         {'speakers': {'speaker_0': 'A'},
          'segments': [{'speaker': 'speaker_9', 'start': 0, 'end': 1,
                        'text': 'x'}]}, duration),
        ('validate-reject-duplicate-ids',
         {'speakers': {'speaker_0': 'A'},
          'segments': [{'id': 'dup', 'speaker': 'speaker_0', 'start': 0,
                        'end': 1, 'text': 'one'},
                       {'id': 'dup', 'speaker': 'speaker_0', 'start': 2,
                        'end': 3, 'text': 'two'}]}, duration),
        # Bounds described parametrically so the golden files stay small.
        ('validate-reject-too-many-segments',
         {'speakers': {'speaker_0': 'A'},
          'segments': {'$repeat_segments': {'count': 20001, 'segment': {
              'speaker': 'speaker_0', 'start': 0, 'end': 1, 'text': 'x'}}}},
         duration),
        ('validate-reject-too-many-speakers',
         {'speakers': {f'speaker_{i}': f'Name {i}' for i in range(65)},
          'segments': []}, duration),
        ('validate-reject-text-too-long',
         {'speakers': {'speaker_0': 'A'},
          'segments': [{'speaker': 'speaker_0', 'start': 0, 'end': 1,
                        'text': {'$repeat_text': {'count': 20001,
                                                  'text': 'x'}}}]}, duration),
    ]
    for name, document, doc_duration in cases:
        spec = copy.deepcopy(document)
        oracle.dump_golden(
            'transcript', name,
            {'document': spec, 'duration': doc_duration},
            oracle.outcome(transcript.validate, oracle.expand(document),
                           doc_duration))

    for kind in ('txt', 'srt', 'vtt', 'json'):
        document = valid_doc()
        result = oracle.outcome(transcript.export, document, kind)
        if 'value' in result:
            content, mimetype = result['value']
            result = {'content': content, 'mimetype': mimetype}
        oracle.dump_golden('transcript', f'export-{kind}',
                           {'document': document, 'kind': kind}, result)

    empty = {'speakers': {'speaker_0': 'Ada'}, 'segments': []}
    for kind in ('txt', 'vtt'):
        result = oracle.outcome(transcript.export, empty, kind)
        if 'value' in result:
            content, mimetype = result['value']
            result = {'content': content, 'mimetype': mimetype}
        oracle.dump_golden('transcript', f'export-{kind}-empty',
                           {'document': empty, 'kind': kind}, result)

    oversized = {'speakers': {'speaker_0': 'Ada'},
                 'segments': {'$repeat_segments': {'count': 20000, 'segment': {
                     'speaker': 'speaker_0', 'start': 0, 'end': 100,
                     'text': {'$repeat_text': {'count': 900,
                                               'text': 'word '}}}}}}
    oracle.dump_golden(
        'transcript', 'export-too-large',
        {'document': oversized, 'kind': 'txt',
         'note': 'input described parametrically: 20,000 segments x 900-word '
                 'text exceeds the 16 MiB export limit without materializing '
                 'the document in this file'},
        oracle.outcome(transcript.export, oracle.expand(oversized), 'txt'))


if __name__ == '__main__':
    run()
