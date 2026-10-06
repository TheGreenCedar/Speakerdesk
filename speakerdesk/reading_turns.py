"""Lossless reading projection; never replace Cohere text or split its anchor.

English CTC emission envelopes are coarse evidence, not phonetic edges. A
speaker transition, missing activity, or weak timing remains unassigned.
"""
import copy
import hashlib
import json
import re

RATE = 16000
MARGIN = 4000
CALIBRATION = 'ami-english-coarse-v1-5ab4e661e62f'
MODEL = 'e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c'


def alignment_words(result, text, start, end):
    """Partial results keep null words; stale/unqualified results provide none."""
    if not isinstance(result, dict):
        return []
    if (result.get('raw_text') != text
            or result.get('text_sha256') != hashlib.sha256(text.encode()).hexdigest()
            or result.get('audio_anchor') != {'start_sample': start, 'end_sample': end}
            or result.get('model_sha256') != MODEL
            or result.get('timing_kind') != 'ctc_emission_cell_envelope'
            or result.get('frame_calibration_id') != CALIBRATION
            or result.get('score_calibration_id') != CALIBRATION):
        return []
    units = list(re.finditer(r'\S+', text))
    words = result.get('words')
    if not isinstance(words, list) or len(words) != len(units):
        return []
    output = []
    previous = start
    for word, unit in zip(words, units):
        if (word.get('text') != unit.group()
                or (word.get('start_char'), word.get('end_char')) != unit.span()):
            return []
        a, b = word.get('start_sample'), word.get('end_sample')
        timed = (word.get('status') == 'aligned' and type(a) is int and type(b) is int
                 and previous <= a < b <= end
                 and start + MARGIN <= a and b <= end - MARGIN)
        output.append({'text': unit.group(), 'start_char': unit.start(), 'end_char': unit.end(),
                       'start_sample': a if timed else None, 'end_sample': b if timed else None})
        if timed:
            previous = b
    return output


def bind_words(row, words):
    if row.get('protected_fields') or row.get('text_audio_anchor') != {
            'start_sample': row['start_sample'], 'end_sample': row['end_sample']}:
        return
    row['reading_word_evidence'] = {
        'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
        'machine_revision': row['machine_revision'], 'audio_revision': row['audio_revision'],
        'audio_anchor': copy.deepcopy(row['text_audio_anchor']), 'words': copy.deepcopy(words),
        'calibration_id': CALIBRATION, 'model_sha256': MODEL,
        'timing_kind': 'ctc_emission_cell_envelope'}


def project_turns(row):
    evidence = row.get('reading_word_evidence')
    activity = row.get('speaker_activity') or {}
    if (not isinstance(evidence, dict) or row.get('protected_fields')
            or evidence.get('text_sha256') != hashlib.sha256(row['text'].encode()).hexdigest()
            or evidence.get('machine_revision') != row['machine_revision']
            or evidence.get('audio_revision') != row['audio_revision']
            or evidence.get('audio_anchor') != row.get('text_audio_anchor')
            or evidence.get('calibration_id') != CALIBRATION or evidence.get('model_sha256') != MODEL
            or activity.get('audio_revision') != row['audio_revision']):
        return []
    units = list(re.finditer(r'\S+', row['text']))
    words = evidence.get('words')
    if not units or not isinstance(words, list) or len(units) != len(words):
        return []
    # Merge identical adjacent activity; model feed boundaries are not turns.
    regions = []
    cursor = row['start_sample']
    for region in activity.get('regions', []):
        a, b, names = region['start_sample'], region['end_sample'], sorted(region['speakers'])
        if not cursor == a < b <= row['end_sample']:
            return []
        if regions and regions[-1]['speakers'] == names:
            regions[-1]['end_sample'] = b
        else:
            regions.append(dict(start_sample=a, end_sample=b, speakers=names))
        cursor = b
    if cursor != row['end_sample']:
        return []
    turns = []
    position = 0
    previous_sample = row['start_sample']
    for index, (word, unit) in enumerate(zip(words, units)):
        if (word.get('text') != unit.group()
                or (word.get('start_char'), word.get('end_char')) != unit.span()):
            return []
        a, b = word.get('start_sample'), word.get('end_sample')
        speaker, state = 'unassigned', 'unknown'
        timed = (type(a) is int and type(b) is int
                 and previous_sample <= a < b <= row['end_sample'])
        if timed:
            previous_sample = b
            for i, region in enumerate(regions):
                # The frozen 250ms guard applies at every activity transition,
                # including gaps and overlap onset, not artificial feed cells.
                left = region['start_sample'] + (MARGIN if i else 0)
                right = region['end_sample'] - (MARGIN if i + 1 < len(regions) else 0)
                names = region['speakers']
                if left <= a < b <= right and names and all(re.fullmatch(r'speaker_\d+', n) for n in names):
                    speaker = names[0] if len(names) == 1 else 'overlap_' + '_'.join(names)
                    state = 'single' if len(names) == 1 else 'overlap'
                    break
        # Every raw character occurs once. Whitespace belongs to the preceding
        # word, and initial whitespace belongs to the first display slice.
        stop = units[index + 1].start() if index + 1 < len(units) else len(row['text'])
        if turns and (turns[-1]['speaker'], turns[-1]['attribution']) == (speaker, state):
            turn = turns[-1]
            turn['end_offset'] = stop
            turn['text'] += row['text'][position:stop]
            if state != 'unknown':
                turn['end'] = b / RATE
        else:
            turns.append({'text': row['text'][position:stop], 'start_offset': position,
                          'end_offset': stop, 'speaker': speaker, 'attribution': state,
                          'start': a / RATE if state != 'unknown' else None,
                          'end': b / RATE if state != 'unknown' else None})
        position = stop
    return turns


def validated_turns(segment):
    """Validate a saved server projection before export; edits fall back intact."""
    receipt = segment.get('reading_turn_provenance') or {}
    turns = segment.get('reading_turns')
    if (not isinstance(turns, list) or not turns or segment.get('protected_fields')
            or receipt.get('utterance_id') != segment.get('canonical_utterance_id')
            or receipt.get('machine_revision') != segment.get('machine_revision')
            or receipt.get('canonical_machine_revision') != segment.get('canonical_machine_revision')
            or receipt.get('audio_revision') != segment.get('audio_revision')
            or receipt.get('language_epoch') != segment.get('language_epoch')
            or receipt.get('text_sha256') != hashlib.sha256(segment['text'].encode()).hexdigest()
            or receipt.get('audio_anchor') != segment.get('text_audio_anchor')
            or receipt.get('activity_sha256') != hashlib.sha256(json.dumps(segment.get('speaker_activity'),
                sort_keys=True,separators=(',',':')).encode()).hexdigest()
            or receipt.get('turns_sha256') != hashlib.sha256(json.dumps(turns,
                sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            or receipt.get('calibration_id') != CALIBRATION or receipt.get('model_sha256') != MODEL):
        return []
    cursor = 0
    for turn in turns:
        if (not isinstance(turn, dict) or type(turn.get('start_offset')) is not int
                or type(turn.get('end_offset')) is not int
                or not cursor == turn['start_offset'] < turn['end_offset'] <= len(segment['text'])
                or turn.get('text') != segment['text'][cursor:turn['end_offset']]
                or turn.get('attribution') not in ('single','overlap','unknown')):
            return []
        state, speaker = turn['attribution'], turn.get('speaker')
        if state == 'unknown':
            if speaker != 'unassigned' or turn.get('start') is not None or turn.get('end') is not None:
                return []
        else:
            pattern = r'speaker_\d+' if state == 'single' else r'overlap_speaker_\d+(?:_speaker_\d+)+'
            if (not isinstance(speaker,str) or not re.fullmatch(pattern,speaker)
                    or type(turn.get('start')) not in (int,float) or type(turn.get('end')) not in (int,float)
                    or not segment['start'] <= turn['start'] < turn['end'] <= segment['end']):
                return []
        cursor = turn['end_offset']
    return turns if cursor == len(segment['text']) else []
