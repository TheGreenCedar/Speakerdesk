"""Explain unresolved speech without inventing ASR confidence or transcript words."""
REASONS = {
    'uncertain': 'Language uncertain; audio retained',
    'needs_language': 'Choose the meeting language; audio retained',
    'recent_context': 'Language inferred from recent speech',
    'best_effort': 'Language uncertain; best-effort transcript',
    'change_pending': 'Language change uncertain; audio retained',
    'unsupported': 'Detected language unsupported; audio retained',
    'insufficient_speech': 'Too little usable speech; audio retained',
    'insufficient_acoustic_context': 'Too little acoustic context to decode; audio retained',
    'short_acoustic_context': 'Unconfirmed words from a short audio fragment; review the recording',
    'possible_non_speech': 'Audio may contain no speech; model words need review',
    'overlapping_speech': 'Overlapping speakers; voices are not separated',
    'empty_result': 'No transcript text returned; audio retained',
    'transcription_failed': 'Transcription failed for this passage; audio retained',
    'token_limit': 'Transcript may be incomplete; audio retained',
    'unassigned_audio': 'Audio outside detected speech; may be silence or missed speech',
    'refinement_incomplete': 'Refinement incomplete; previous words and audio retained',
}


def review_reason(segment):
    mixed=len(segment.get('speaker_candidates', [])) > 1 or str(segment.get('speaker', '')).startswith('overlap')
    if segment.get('review_resolution') == 'words_reviewed' and segment.get('text', '').strip():
        return REASONS['overlapping_speech'] if mixed else 'Speaker uncertain' if segment.get('speaker')=='unassigned' else ''
    transcription = segment.get('transcription_review') or {}
    if transcription.get('reason') in REASONS:
        return REASONS[transcription['reason']]
    if mixed:
        return REASONS['overlapping_speech']
    detection = segment.get('language_detection') or {}
    if detection.get('mode') == 'auto' and detection.get('reason') in ('recent_context','best_effort'):
        return REASONS[detection['reason']]
    if detection.get('mode') == 'auto' and not segment.get('language'):
        return REASONS.get(detection.get('reason'), REASONS['uncertain'])
    if not segment.get('text', '').strip():
        return REASONS['empty_result']
    if segment.get('speaker') == 'unassigned':
        return 'Speaker uncertain'
    return ''


def export_text(segment):
    text = segment['text']
    reason = review_reason(segment)
    return text+(' ' if text.strip() else '')+f'[Needs review: {reason}]' if reason else text


def retain_unassigned_audio(document, duration):
    """Account for audio outside returned passages without calling it speech."""
    if duration <= 0:return document
    segments = document['segments']
    cursor, gaps = 0., []
    for segment in sorted(segments, key=lambda s:s['start']):
        if segment['start']-cursor > .001:gaps.append((cursor, segment['start']))
        cursor = max(cursor, segment['end'])
    if duration-cursor > .001:gaps.append((cursor, duration))
    if not gaps:return document
    document['speakers']['unassigned'] = 'Unassigned audio'
    identifiers = {s['id'] for s in segments}
    for start,end in gaps:
        while end-start > .001:
            sid = 'unassigned-'+str(len(identifiers))
            while sid in identifiers:sid += '-gap'
            identifiers.add(sid)
            stop = min(end, start+30)
            segments.append({'id':sid, 'start':start, 'end':stop, 'speaker':'unassigned',
                             'text':'', 'language':None, 'review':True, 'voice_eligible':False,
                             'speaker_candidates':[], 'confidence':None, 'timing':'audio_coverage',
                             'transcription_review':{'reason':'unassigned_audio','partial_text':False}})
            start = stop
    segments.sort(key=lambda s:(s['start'],s['end']))
    return document
