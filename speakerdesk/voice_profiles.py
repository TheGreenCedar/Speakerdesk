"""Local voice identity contracts. No model loading, downloading, or network access.

A future approved adapter owns waveform loading, quality checks and embedding.
Nemotron's meeting-local slots/cache are deliberately not an embedding adapter.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Protocol


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


def clean_clips(job, track_id, segment_ids, minimum=2):
    document = job.get('document') or {}
    if document.get('provenance', {}).get('kind') != 'local_inference':
        raise ValueError('Use finalized local transcription clips for voice recognition.')
    if (not isinstance(segment_ids, list) or not minimum <= len(segment_ids) <= 12
            or any(not isinstance(s, str) for s in segment_ids) or len(set(segment_ids)) != len(segment_ids)):
        raise ValueError(f'Choose {minimum}–12 separate clean passages.')
    segments = {s['id']: s for s in document['segments']}
    clips = []
    for sid in segment_ids:
        segment = segments.get(sid)
        if (not segment or segment['speaker'] != track_id or segment.get('finalized') is False
                or segment.get('review') or track_id.startswith('overlap')
                or len(segment.get('speaker_candidates', [track_id])) != 1
                or not 2 <= segment['end']-segment['start'] <= 10):
            raise ValueError('Choose clean, finalized passages from one speaker, each 2–10 seconds long.')
        clips.append(VoiceClip(job['id'], track_id, sid, segment['start'], segment['end']))
    clips.sort(key=lambda c: c.start)
    if any(a.end > b.start for a, b in zip(clips, clips[1:])):
        raise ValueError('Choose separate, nonoverlapping passages.')
    return clips


def extract(backend, audio, clips):
    vectors = []
    for clip in clips:
        result = backend.embed(audio, clip)
        if result.clean is not True:
            raise ValueError('A selected passage has noise or overlapping voices. Choose another passage.')
        vectors.append(normalized(result.vector, backend.model.dimension))
    return vectors


def make_profile(model, vectors, clips):
    if len(vectors) < 2 or len(vectors) != len(clips):
        raise ValueError('A voice profile requires multiple clean passages.')
    vectors = [normalized(v, model.dimension) for v in vectors]
    centroid = normalized([sum(v[i] for v in vectors)/len(vectors) for i in range(model.dimension)], model.dimension)
    return {'model': model.payload(), 'centroid': centroid,
            'clips': [c.payload() for c in clips], 'consent': 'explicit_remember_voice'}


def propose_match(model, calibration, vectors, profiles):
    """Unknown unless every clip passes and the winner is separated from runner-up."""
    if calibration.model != model or len(vectors) < calibration.minimum_clips:
        return None
    vectors = [normalized(v, model.dimension) for v in vectors]
    ranked = []
    for profile in profiles:
        if profile['model'] != model.payload():
            continue  # Versions/conversions are separate embedding spaces.
        centroid = normalized(profile['centroid'], model.dimension)
        scores = [sum(a*b for a, b in zip(v, centroid)) for v in vectors]
        ranked.append((sum(scores)/len(scores), min(scores), profile))
    ranked.sort(key=lambda row: row[0], reverse=True)
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
