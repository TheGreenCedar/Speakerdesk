"""Strict retained-source audio routing outside the qualified neural adapter.

The worker still accepts only app-owned recording files. Source routes require
the recording's immutable catalog, exact source binding and sample revision.
Legacy clips continue to require audio.wav.
"""
import hashlib
import json
import math
import wave
from pathlib import Path
from capture_sources import RATE, source_path, validate_catalog, validate_binding,interval_pcm_digest


def clip_audio(canonical_audio, clip):
    path = Path(canonical_audio)
    if getattr(clip,'capture_source',None) is None:
        return path
    actual = clip.capture_source
    return source_path(path, actual.get('source_id'))


def validate_voice_audio(audio_root, audio, clip):
    root = Path(audio_root).resolve()
    if (not isinstance(clip.meeting_id, str) or not clip.meeting_id
            or Path(clip.meeting_id).name != clip.meeting_id or clip.meeting_id in ('.', '..')):
        raise ValueError('Voice clip is outside this app meeting.')
    folder = root/clip.meeting_id
    if folder.is_symlink():raise ValueError('Voice recording cannot be a symbolic link.')
    expected = clip_audio(folder/'audio.wav', clip)
    candidate = Path(audio)
    if candidate.is_symlink() or candidate.resolve() != expected:
        raise ValueError('Voice clip is outside this app meeting.')
    if getattr(clip,'capture_source',None) is not None:
        manifest = folder/'capture-sources.json'
        if manifest.is_symlink():raise ValueError('Voice source catalog cannot be a symbolic link.')
        try:value = validate_catalog(json.loads(manifest.read_text()), clip.meeting_id)
        except OSError as error:raise ValueError('Retained voice source catalog is unavailable.') from error
        validate_binding(value, {'capture_source': clip.capture_source})
        if type(clip.audio_revision) is not int or clip.audio_revision < 0:
            raise ValueError('Voice clip needs its retained source revision.')
        if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (clip.start, clip.end))
                or (clip.start, clip.end) != (round(clip.start*RATE)/RATE, round(clip.end*RATE)/RATE)):
            raise ValueError('Voice clip differs from its retained sample bounds.')
        evidence=clip.capture_audio
        if (not isinstance(evidence,dict) or evidence.get('capture_source')!=clip.capture_source
                or evidence.get('audio_revision')!=clip.audio_revision or evidence.get('encoding')!='pcm_s16le'
                or evidence.get('sample_rate')!=RATE or type(evidence.get('start_sample')) is not int
                or type(evidence.get('end_sample')) is not int
                or not evidence['start_sample']<=round(clip.start*RATE)<round(clip.end*RATE)<=evidence['end_sample']
                or interval_pcm_digest(expected,evidence['start_sample'],evidence['end_sample'])!=evidence.get('pcm_sha256')):
            raise ValueError('Voice source PCM differs from its published revision.')
    elif getattr(clip,'audio_revision',None) is not None:
        raise ValueError('Unbound voice source revision.')
    return expected


def clip_pcm_digest(audio, clip):
    """Hash the exact retained PCM16 crop, without preparing or running a model."""
    from voice_waveform import read_clip
    # Preserve the adapter's shape/format/bounds refusals before a worker launch.
    read_clip(audio, clip)
    first, last = round(clip.start*RATE), round(clip.end*RATE)
    with wave.open(str(audio), 'rb') as source:
        source.setpos(first); raw = source.readframes(last-first)
    if len(raw) != (last-first)*2:raise ValueError('Incomplete retained voice crop.')
    return hashlib.sha256(raw).hexdigest()
