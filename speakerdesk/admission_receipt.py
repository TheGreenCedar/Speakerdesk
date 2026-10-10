"""Typed neural-inspection receipts bound to a job, operation and retained PCM.

An observed prefix is not a no-speech verdict or whole-recording completion.
Missing/failed receipts never satisfy this contract. No model is run here.
"""
import copy
import hashlib
import re
import wave
from speech_admission import SILERO_SPEC, INPUT_POLICY, RATE


def execution(job_id, execution_id, *, conditioning=None):
    result={'schema_version':2,'component':'silero_v6',
            'model_revision':SILERO_SPEC['revision'],'model_sha256':SILERO_SPEC['sha256'],
            'input_policy':INPUT_POLICY,'job_id':job_id,'execution_id':execution_id}
    if conditioning is not None:
        from speech_conditioning import SourceConditioning
        if type(conditioning) is not SourceConditioning:raise ValueError('Unverified conditioning identity.')
        result.update(input_policy=conditioning.input_policy,
                      conditioning_sha256=conditioning.contract_sha256,capture_source=dict(conditioning.capture_source))
    return validate_execution(result,job_id)


def validate_execution(value,job_id):
    expected={'schema_version':2,'component':'silero_v6',
              'model_revision':SILERO_SPEC['revision'],'model_sha256':SILERO_SPEC['sha256'],
              'input_policy':INPUT_POLICY,'job_id':job_id}
    from speech_conditioning import SOURCE_UNITY_POLICY, PRODUCTION_UNITY_POLICY
    if isinstance(value,dict) and value.get('input_policy') in (SOURCE_UNITY_POLICY, PRODUCTION_UNITY_POLICY):
        source=value.get('capture_source')
        if (re.fullmatch(r'[0-9a-f]{64}',str(value.get('conditioning_sha256',''))) is None
                or not isinstance(source,dict) or set(source)!={'catalog_sha256','source_id','source_revision'}
                or source.get('source_id')!='microphone_clean' or type(source.get('source_revision')) is not int
                or source['source_revision']!=1
                or re.fullmatch(r'[0-9a-f]{64}',str(source.get('catalog_sha256',''))) is None):
            raise ValueError('Invalid source-conditioned inspection identity.')
        expected.update(input_policy=value['input_policy'],conditioning_sha256=value['conditioning_sha256'],
                        capture_source=copy.deepcopy(source))
    if (not isinstance(value,dict) or type(value.get('schema_version')) is not int
            or any(value.get(key)!=wanted for key,wanted in expected.items())
            or not isinstance(job_id,str) or re.fullmatch(r'[0-9a-f]{32}',job_id) is None
            or re.fullmatch(r'[0-9a-f]{32}',str(value.get('execution_id',''))) is None):
        raise ValueError('Invalid speech-inspection execution identity.')
    return {**expected,'execution_id':value['execution_id']}


def validate_receipt(value,identity,*,phase,request_id,received_sample,observed_sample,
                     uncertain_samples=None,pcm_sha256=None):
    if not isinstance(identity,dict):raise ValueError('Missing speech-inspection execution identity.')
    identity=validate_execution(identity,identity.get('job_id'))
    if isinstance(value,dict):validate_execution(value,identity['job_id'])
    if (not isinstance(value,dict) or any(value.get(key)!=wanted for key,wanted in identity.items())
            or phase not in ('pause','stop') or value.get('phase')!=phase
            or not isinstance(request_id,str) or not request_id or value.get('request_id')!=request_id
            or type(received_sample) is not int or type(observed_sample) is not int
            or not 0<=observed_sample<=received_sample
            or value.get('inspection_state')!='observed_prefix'
            or type(value.get('start_sample')) is not int or value['start_sample']!=0
            or type(value.get('received_sample')) is not int or value['received_sample']!=received_sample
            or type(value.get('end_sample')) is not int or value['end_sample']!=observed_sample
            or type(value.get('closed')) is not bool or value['closed']!=(phase=='stop')
            or (phase=='stop' and observed_sample!=received_sample)
            or value.get('audio_encoding')!='pcm_s16le'
            or re.fullmatch(r'[0-9a-f]{64}',str(value.get('pcm_sha256',''))) is None):
        raise ValueError('Speech-inspection receipt differs from its operation or audio horizon.')
    counts=[value.get(key) for key in ('speech_samples','uncertain_samples','negative_constant_samples','model_negative_samples')]
    if (any(type(n) is not int or n<0 for n in counts) or sum(counts)!=observed_sample
            or (uncertain_samples is not None and
                (type(uncertain_samples) is not int or uncertain_samples!=counts[1]))
            or (pcm_sha256 is not None and pcm_sha256!=value['pcm_sha256'])):
        raise ValueError('Speech-inspection counts or retained PCM differ.')
    decision='speech' if counts[0] else 'uncertain' if counts[1] else 'no_speech'
    if value.get('decision')!=decision:
        raise ValueError('Speech-inspection decision differs from observed evidence.')
    return copy.deepcopy(value)


def retained_pcm_digest(path,end_sample):
    if type(end_sample) is not int or end_sample<0:
        raise ValueError('Invalid retained speech-inspection horizon.')
    digest=hashlib.sha256()
    with wave.open(str(path),'rb') as source:
        if (source.getnchannels(),source.getsampwidth(),source.getframerate())!=(1,2,RATE):
            raise ValueError('Speech inspection requires retained mono16k PCM.')
        if source.getnframes()<end_sample:
            raise ValueError('Retained audio is shorter than its inspection receipt.')
        remaining=end_sample
        while remaining:
            count=min(65536,remaining);data=source.readframes(count)
            if len(data)!=count*2:raise ValueError('Retained inspection audio is incomplete.')
            digest.update(data);remaining-=count
    return digest.hexdigest()
