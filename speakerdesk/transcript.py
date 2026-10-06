"""Canonical transcript validation and interoperable exports; no word-time inference."""
import copy
import json
import math
from itertools import chain
from review import export_text
from reading_turns import validated_turns

LANGUAGES = {'en':'English','de':'German','fr':'French','it':'Italian','es':'Spanish',
             'pt':'Portuguese','el':'Greek','nl':'Dutch','pl':'Polish','vi':'Vietnamese',
             'zh':'Chinese','ar':'Arabic','ja':'Japanese','ko':'Korean'}

MAX_EXPORT_BYTES = 16 * 1024 * 1024


class ExportTooLarge(ValueError):
    """The serialized download exceeds the local export limit."""


def bounded_text(parts):
    chunks = []
    size = 0
    for part in parts:
        size += len(part.encode('utf-8'))
        if size > MAX_EXPORT_BYTES:
            raise ExportTooLarge('Export exceeds the 16 MiB download limit. The saved meeting is unchanged.')
        chunks.append(part)
    return ''.join(chunks)


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
        parts = json.JSONEncoder(ensure_ascii=False, indent=2).iterencode(document)
        return bounded_text(chain(parts, ('\n',))), 'application/json'

    def lines():
        if kind == 'vtt':
            yield 'WEBVTT'
            yield ''
        def projected_passages():
            for s in document['segments']:
                if not s.get('text','').strip():continue
                turns=validated_turns(s)
                # Subtitle cues require actual timing for every slice. Preserve
                # the complete parent cue if even one word has unknown timing.
                if turns and (kind=='txt' or all(t['attribution']!='unknown' for t in turns)):
                    for turn in turns:
                        yield dict(s,text=turn['text'],speaker=turn['speaker'],reading_slice=True,
                            start=turn['start'] if turn['start'] is not None else s['start'],
                            end=turn['end'] if turn['end'] is not None else s['end'],
                            coarse_timing=turn['attribution']!='unknown',review=False,
                            speaker_candidates=[])
                else:yield s
        passages = projected_passages()
        emitted = kind == 'vtt'
        for n, s in enumerate(passages, 1):
            emitted = True
            speaker = document['speakers'].get(s['speaker'],
                'Unknown speaker' if s['speaker']=='unassigned' else 'Overlapping speakers')
            # Plain subtitle text avoids VTT/HTML interpretation and cue injection.
            passage = export_text(s)
            text = ' '.join(passage.split()).replace('-->', '→').replace('<', '‹').replace('>', '›')
            name = ' '.join(speaker.split()).replace('-->', '→').replace('<', '‹').replace('>', '›')
            if kind == 'txt':
                timing=('Coarse timing ' if s.get('coarse_timing') else 'Timing unknown; parent audio ' if s.get('reading_slice') else '')
                yield f'[{timing}{timestamp(s["start"])} – {timestamp(s["end"])}] {speaker}: {passage}'
            else:
                sep = ',' if kind == 'srt' else '.'
                yield str(n)
                yield f'{timestamp(s["start"],sep)} --> {timestamp(s["end"],sep)}'
                yield f'{name}: {text}'
                yield ''
        if not emitted:
            yield ''

    return bounded_text(line + '\n' for line in lines()), 'text/plain; charset=utf-8' if kind != 'vtt' else 'text/vtt; charset=utf-8'
