"""Local voice identity contracts. No model loading, downloading, or network access.

The approved local adapter owns waveform loading, quality checks and embedding.
Nemotron's meeting-local slots/cache are deliberately not an embedding adapter.
"""
from dataclasses import asdict, dataclass
import math
import re
from pathlib import Path
from typing import Protocol


def runtime_available(backend, calibration):
    """Apply the adapter's execution gate as well as profile/calibration identity."""
    if backend is None or calibration is None or backend.model != calibration.model:
        return False
    check = getattr(backend, 'require_execution_policy', None)
    if check is not None:
        try:
            check()
        except ValueError:
            return False
    return True


@dataclass(frozen=True)
class VoiceModel:
    model_id: str
    revision: str
    artifact_sha256: str
    dimension: int
    sample_rate: int = 16000

    def __post_init__(self):
        if (not self.model_id or not self.revision or len(self.artifact_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.artifact_sha256)
                or not 1 <= self.dimension <= 4096 or self.sample_rate != 16000):
            raise ValueError('Supply a pinned, compatible voice model identity.')

    def payload(self):
        return asdict(self)


@dataclass(frozen=True)
class Calibration:
    """Measured policy for one exact model; no guessed production defaults."""
    model: VoiceModel
    dataset_id: str
    threshold: float
    margin: float
    genuine_trials: int
    impostor_trials: int
    false_accept_rate: float
    false_reject_rate: float
    minimum_clips: int = 2

    def __post_init__(self):
        values = (self.threshold, self.margin, self.false_accept_rate, self.false_reject_rate)
        if (not self.dataset_id or not all(math.isfinite(v) for v in values)
                or not 0 < self.threshold <= 1 or not 0 < self.margin <= 1
                or not 0 <= self.false_accept_rate <= 1 or not 0 <= self.false_reject_rate <= 1
                or self.genuine_trials < 1 or self.impostor_trials < 1 or not 2 <= self.minimum_clips <= 12):
            raise ValueError('Voice matching needs measured threshold and margin calibration.')


@dataclass(frozen=True)
class VoiceClip:
    meeting_id: str
    track_id: str
    segment_id: str
    start: float
    end: float

    def payload(self):
        return asdict(self)


@dataclass(frozen=True)
class ClipEmbedding:
    vector: tuple[float, ...]
    clean: bool
    metrics: dict | None = None


class LocalVoiceBackend(Protocol):
    """Implement only with an approved local model; never upload audio."""
    model: VoiceModel

    def embed(self, audio: Path, clip: VoiceClip) -> ClipEmbedding: ...


def normalized(vector, dimension):
    if (not isinstance(vector, (list, tuple)) or len(vector) != dimension
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)):
        raise ValueError('Invalid voice embedding.')
    norm = math.sqrt(sum(v*v for v in vector))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid voice embedding.')
    return [v/norm for v in vector]


def eligible_segment(segment, track_id):
    # A six-second ASR phrase can need text review while its diarized audio is single-speaker.
    return speaker_audio_eligible(segment, track_id) and 2 <= segment['end']-segment['start'] <= 10


def speaker_audio_eligible(segment, track_id):
    return speaker_audio_reason(segment, track_id) is None


def speaker_audio_reason(segment, track_id):
    """Why saved source audio cannot establish this track's identity."""
    if segment.get('speaker') != track_id:
        return 'different_speaker'
    if segment.get('audio_state'):
        return 'unverified_audio'
    if segment.get('finalized') is False:
        return 'unfinished'
    candidates = segment.get('speaker_candidates', [track_id])
    if track_id.startswith('overlap') or len(candidates) > 1:
        return 'overlap'
    if len(candidates) != 1 or segment.get('voice_eligible', not segment.get('review')) is not True:
        return 'unverified_audio'
    return None


def speaker_audio_ranges(segment, track_id):
    """Keep whole-anchor eligibility strict; crop only current owned activity.

    Canonical VAD anchors can include speech without a diarized owner at their
    edges. That does not invalidate a separate, fully observed solo interval.
    Names and word edits do not change its audio; ownership/timing edits do.
    Reading slices and their coarse word timestamps are not voice evidence.
    """
    if speaker_audio_eligible(segment, track_id):
        return [(segment['start'], segment['end'])]
    if (segment.get('speaker') != track_id or segment.get('audio_state')
            or not re.fullmatch(r'speaker_\d+', track_id)
            or segment.get('canonical_unresolved')
            or segment.get('finalized') is not True
            or segment.get('canonical_state') != 'sealed'
            or segment.get('canonical_utterance_id') != segment.get('id')
            or segment.get('speaker_candidates') != [track_id]
            or segment.get('source_speaker_candidates') != [track_id]
            or set(segment.get('protected_fields', [])) & {'speaker', 'start', 'end'}):
        return []
    a, b, revision = (segment.get(k) for k in ('start_sample', 'end_sample', 'audio_revision'))
    activity = segment.get('speaker_activity')
    if (type(a) is not int or type(b) is not int or not 0 <= a < b
            or type(revision) is not int or revision < 0
            or (segment['start'], segment['end']) != (a/16000, b/16000)
            or not isinstance(activity, dict) or activity.get('audio_revision') != revision
            or activity.get('observed_end_sample') != b
            or not isinstance(activity.get('regions'), list)):
        return []
    ranges, cursor = [], a
    for region in activity['regions']:
        if not isinstance(region, dict):
            return []
        left, right, owners = (region.get(k) for k in ('start_sample', 'end_sample', 'speakers'))
        if (type(left) is not int or type(right) is not int or not cursor == left < right <= b
                or not isinstance(owners, list) or owners not in ([], [track_id])):
            return []  # Never salvage a mixed/unknown ownership anchor here.
        if owners:
            if ranges and ranges[-1][1] == left:
                ranges[-1] = (ranges[-1][0], right)
            else:
                ranges.append((left, right))
        cursor = right
    return [(left/16000, right/16000) for left, right in ranges] if cursor == b else []


def voice_clip_choices(job, track_id):
    """The exact selection IDs accepted for saved, source-owned audio windows."""
    choices = {}
    for segment in (job.get('document') or {}).get('segments', []):
        for left, right in speaker_audio_ranges(segment, track_id):
            start = left
            while right-start >= 2:
                end = min(start+6, right) if right-left > 10 else right
                whole = (start, end) == (segment['start'], segment['end'])
                sid = segment['id'] if whole else f"{segment['id']}@{round(start*16000)}:{round(end*16000)}"
                choices[sid] = VoiceClip(job['id'], track_id, segment['id'], start, end)
                start = end
    return choices


def automatic_clips(job, track_id, minimum=2):
    """Separate 2–10s clips; long import crops contribute nonoverlapping 6s windows."""
    if (job.get('document') or {}).get('provenance', {}).get('kind') != 'local_inference':
        return []
    clips, last_end = [], -1.
    for clip in sorted(voice_clip_choices(job, track_id).values(), key=lambda c: (c.end, c.start)):
        if clip.start >= last_end:
            clips.append(clip)
            last_end = clip.end
            if len(clips) == minimum:
                return clips
    return []


def clips_current(job, clips):
    document = job.get('document') or {}
    if document.get('provenance', {}).get('kind') != 'local_inference':
        return False
    choices = {track: tuple(voice_clip_choices(job, track).values()) for track in {c.track_id for c in clips}}
    return all(c in choices[c.track_id] for c in clips)


def clean_clips(job, track_id, segment_ids, minimum=2):
    document = job.get('document') or {}
    if document.get('provenance', {}).get('kind') != 'local_inference':
        raise ValueError('Use finalized local transcription clips for voice recognition.')
    if (not isinstance(segment_ids, list) or not minimum <= len(segment_ids) <= 12
            or any(not isinstance(s, str) for s in segment_ids) or len(set(segment_ids)) != len(segment_ids)):
        raise ValueError(f'Choose {minimum}–12 separate clean passages.')
    choices = voice_clip_choices(job, track_id)
    clips = []
    for sid in segment_ids:
        if sid not in choices:
            raise ValueError('Choose clean, finalized passages from one speaker, each 2–10 seconds long.')
        clips.append(choices[sid])
    clips.sort(key=lambda c: c.start)
    if any(a.end > b.start for a, b in zip(clips, clips[1:])):
        raise ValueError('Choose separate, nonoverlapping passages.')
    return clips


def extract(backend, audio, clips):
    vectors = []
    for clip in clips:
        result = backend.embed(audio, clip)
        if result.clean is not True:
            raise ValueError('A selected passage is unusable for voice recognition. Choose another clean passage.')
        vectors.append(normalized(result.vector, backend.model.dimension))
    return vectors


def make_profile(model, vectors, clips):
    if len(vectors) < 2 or len(vectors) != len(clips):
        raise ValueError('A voice profile requires multiple clean passages.')
    vectors = [normalized(v, model.dimension) for v in vectors]
    centroid = normalized([sum(v[i] for v in vectors)/len(vectors) for i in range(model.dimension)], model.dimension)
    return {'model': model.payload(), 'centroid': centroid,
            'clips': [c.payload() for c in clips], 'consent': 'explicit_remember_voice'}


def rank_matches(model, vectors, profiles):
    """Shared scoring for runtime decisions and offline calibration."""
    vectors = [normalized(v, model.dimension) for v in vectors]
    ranked = []
    for profile in profiles:
        if profile['model'] != model.payload():
            continue  # Versions/conversions are separate embedding spaces.
        centroid = normalized(profile['centroid'], model.dimension)
        scores = [max(-1., min(1., sum(a*b for a, b in zip(v, centroid)))) for v in vectors]
        ranked.append((sum(scores)/len(scores), min(scores), profile))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


def propose_match(model, calibration, vectors, profiles):
    """Unknown unless every clip passes and the winner is separated from runner-up."""
    if calibration.model != model or len(vectors) < calibration.minimum_clips:
        return None
    ranked = rank_matches(model, vectors, profiles)
    return match_from_ranked(model, calibration, ranked)


def match_from_ranked(model, calibration, ranked):
    """Apply the shared measured acceptance rule to already scored candidates."""
    if not ranked:
        return None
    score, lowest, best = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if lowest < calibration.threshold or score-runner_up < calibration.margin:
        return None
    return {'person_id': best['person_id'], 'profile_version': best['version'],
            'score': score, 'margin': score-runner_up, 'model': model.payload(),
            'calibration': {'dataset_id': calibration.dataset_id, 'threshold': calibration.threshold,
                            'margin': calibration.margin, 'minimum_clips': calibration.minimum_clips}}
