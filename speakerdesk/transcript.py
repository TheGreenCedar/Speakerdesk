"""Canonical transcript validation and interoperable exports; no word-time inference."""
import copy
import json
import math
from review import export_text

LANGUAGES = {'en':'English','de':'German','fr':'French','it':'Italian','es':'Spanish',
             'pt':'Portuguese','el':'Greek','nl':'Dutch','pl':'Polish','vi':'Vietnamese',
             'zh':'Chinese','ar':'Arabic','ja':'Japanese','ko':'Korean'}


def validate(document, duration):
    if not isinstance(document, dict):
        raise ValueError('Transcript must be a JSON object.')
    speakers = document.get('speakers')
    segments = document.get('segments')
    if not isinstance(speakers, dict) or not 1 <= len(speakers) <= 64:
        raise ValueError('Supply 1–64 speaker names.')
    if any(not isinstance(k, str) or len(k) > 80 or not isinstance(v, str)
           or not v.strip() or len(v) > 100 for k, v in speakers.items()):
        raise ValueError('Speaker names must be nonempty text, up to 100 characters.')
    if not isinstance(segments, list) or len(segments) > 20000:
        raise ValueError('Supply a segment list with at most 20,000 entries.')
    result = copy.deepcopy(document)
    identifiers = set()
    for i, s in enumerate(result['segments']):
        if not isinstance(s, dict):
            raise ValueError('Each segment must be an object.')
        try:
            start, end = float(s['start']), float(s['end'])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'Segment {i+1}: supply numeric start and end times.') from None
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 0.001:
            raise ValueError(f'Segment {i+1}: times must fall within the recording and end after start.')
        if s.get('speaker') not in speakers or not isinstance(s.get('text'), str) or len(s['text']) > 20000:
            raise ValueError(f'Segment {i+1}: invalid speaker or text.')
        sid = s.get('id', f'seg-{i}')
        if not isinstance(sid, str) or len(sid) > 100 or sid in identifiers:
            raise ValueError('Segment IDs must be unique strings.')
        identifiers.add(sid)
        s.update(id=sid, start=start, end=end)
    result['segments'].sort(key=lambda s: (s['start'], s['end']))
    result['speakers'] = {k:v.strip() for k,v in speakers.items()}
    result['schema_version'] = 1
    return result


def timestamp(seconds, separator='.'):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{secs:02}{separator}{ms:03}'


def export(document, kind):
    if kind == 'json':
        return json.dumps(document, ensure_ascii=False, indent=2) + '\n', 'application/json'
    lines = ['WEBVTT', ''] if kind == 'vtt' else []
    for n, s in enumerate(document['segments'], 1):
        speaker = document['speakers'][s['speaker']]
        # Plain subtitle text avoids VTT/HTML interpretation and cue injection.
        passage = export_text(s)
        text = ' '.join(passage.split()).replace('-->', '→').replace('<', '‹').replace('>', '›')
        name = ' '.join(speaker.split()).replace('-->', '→').replace('<', '‹').replace('>', '›')
        if kind == 'txt':
            lines.append(f'[{timestamp(s["start"])} – {timestamp(s["end"])}] {speaker}: {passage}')
        else:
            sep = ',' if kind == 'srt' else '.'
            lines.extend([str(n), f'{timestamp(s["start"],sep)} --> {timestamp(s["end"],sep)}', f'{name}: {text}', ''])
    return '\n'.join(lines) + '\n', 'text/plain; charset=utf-8' if kind != 'vtt' else 'text/vtt; charset=utf-8'
