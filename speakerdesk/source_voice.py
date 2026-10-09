"""Source provenance adapter around the unchanged qualified voice policy.

Eligibility, consent clip limits, waveform preparation, scoring and calibration
remain in voice_profiles. This layer changes only which retained input a clip
belongs to and which revision authorizes that selection.
"""
import copy
from dataclasses import asdict,dataclass
from pathlib import Path
import voice_profiles as policy
from capture_sources import validate_binding,utterance_id


@dataclass(frozen=True)
class SourceVoiceClip(policy.VoiceClip):
    capture_source: dict
    audio_revision: int
    capture_audio: dict

    def payload(self):
        return asdict(self)


def clip_from_payload(value):
    if 'capture_source' in value:return SourceVoiceClip(**value)
    return policy.VoiceClip(**value)


def qualified_job(job):
    result={**job,'document':copy.deepcopy(job.get('document') or {})}
    kept=[]
    for segment in result['document'].get('segments',[]):
        try:source=validate_binding(job.get('capture_source_catalog'),segment)
        except ValueError:continue
        if source and (type(segment.get('audio_revision')) is not int or segment['audio_revision']<0):continue
        if source and segment.get('id')!=utterance_id(job['id'],segment.get('language_epoch'),segment.get('start_sample'),source):continue
        if source:
            evidence=segment.get('capture_audio')
            if (not isinstance(evidence,dict) or evidence.get('capture_source')!=source
                    or evidence.get('audio_revision')!=segment['audio_revision']
                    or evidence.get('start_sample')!=segment.get('start_sample')
                    or evidence.get('end_sample')!=segment.get('end_sample')):continue
        kept.append(segment)
    result['document']['segments']=kept
    return result


def bind_clips(job,clips):
    segments={s['id']:s for s in job['document']['segments']}
    result=[]
    for clip in clips:
        row=segments[clip.segment_id];source=row.get('capture_source')
        result.append(SourceVoiceClip(**clip.payload(),capture_source=copy.deepcopy(source),
            audio_revision=row['audio_revision'],capture_audio=copy.deepcopy(row['capture_audio'])) if source else clip)
    return result


def voice_clip_choices(job,track_id):
    job=qualified_job(job);choices=policy.voice_clip_choices(job,track_id)
    return dict(zip(choices,bind_clips(job,list(choices.values()))))


def automatic_clips(job,track_id,minimum=2):
    job=qualified_job(job)
    return bind_clips(job,policy.automatic_clips(job,track_id,minimum))


def clean_clips(job,track_id,segment_ids,minimum=2):
    job=qualified_job(job)
    return bind_clips(job,policy.clean_clips(job,track_id,segment_ids,minimum))


def clips_current(job,clips):
    if (job.get('document') or {}).get('provenance',{}).get('kind')!='local_inference':return False
    choices={track:tuple(voice_clip_choices(job,track).values()) for track in {c.track_id for c in clips}}
    return all(c in choices[c.track_id] for c in clips)


def extract(backend,audio,clips):
    from voice_source_audio import clip_audio,validate_voice_audio
    vectors=[]
    for clip in clips:
        selected=clip_audio(audio,clip)
        if isinstance(clip,SourceVoiceClip):
            selected=validate_voice_audio(Path(audio).parent.parent,selected,clip)
        vectors.extend(policy.extract(backend,selected,[clip]))
    return vectors
