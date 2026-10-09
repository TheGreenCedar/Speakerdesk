"""Fixed capture inputs on the recording clock; no client-supplied audio paths.

This optional prototype catalog never changes old/import recording identities.
A source is an audio provenance boundary, not a human identity.
"""
import hashlib
import json
import re
from pathlib import Path

RATE = 16000
SOURCE_IDS = ('microphone_clean', 'system')
POLICY = 'aligned_aec_microphone_and_original_system_v1'


def catalog(job_id, *, innovation=False, startup=False):
    if not isinstance(job_id, str) or not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('Invalid capture recording identity.')
    if startup and not innovation:raise ValueError('Startup source requires innovation evidence.')
    value = {'schema_version': 1, 'job_id': job_id, 'sample_rate': RATE,
            'clock': 'recording_samples', 'policy': POLICY,
            'sources': [{'source_id': source, 'basename': source+'.wav',
                         'revision': 1} for source in SOURCE_IDS]}
    if innovation:
        from render_innovation import POLICY_SHA256
        value.update(schema_version=2,policy='causal_innovation_with_native_fallback_development_v1',
            derivation={'parameters_sha256':POLICY_SHA256,'native':'microphone_native.wav','native_mix':'audio_native.wav',
                'raw':'microphone.wav','render':'system_reference.wav',
                'receipts':'innovation-receipts.jsonl','summary':'innovation-evidence.json',
                'production_admissible':False})
    if startup:
        from render_startup import POLICY_SHA256
        value.update(schema_version=3,policy='bounded_startup_innovation_development_v1')
        value['derivation']['parameters_sha256']=POLICY_SHA256
    return value


def validate_catalog(value, job_id):
    version=value.get('schema_version') if isinstance(value,dict) else None
    expected = catalog(job_id,innovation=version in (2,3),startup=version==3)
    if not isinstance(value,dict) or json.dumps(value,sort_keys=True)!=json.dumps(expected,sort_keys=True):
        raise ValueError('Unsupported capture source catalog.')
    return expected


def binding(value, source_id):
    value = validate_catalog(value, value.get('job_id') if isinstance(value, dict) else None)
    if source_id not in SOURCE_IDS:
        raise ValueError('Unknown capture input.')
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'catalog_sha256': digest, 'source_id': source_id, 'source_revision': 1}


def validate_binding(value, source):
    actual = source.get('capture_source')
    if not value:
        if actual is not None:
            raise ValueError('Source-qualified output has no recording catalog.')
        return None
    validate_catalog(value, value['job_id'])
    if (not isinstance(actual, dict) or json.dumps(actual,sort_keys=True)!=
            json.dumps(binding(value, actual.get('source_id')),sort_keys=True)):
        raise ValueError('Output differs from its capture source.')
    return actual


def source_path(audio_path, source_id):
    path = Path(audio_path)
    if path.name != 'audio.wav' or path.is_symlink() or source_id not in SOURCE_IDS:
        raise ValueError('Capture input requires the canonical recording directory.')
    result = path.with_name(source_id+'.wav')
    if result.is_symlink():
        raise ValueError('Capture input cannot follow a symbolic link.')
    return result


def utterance_id(job_id, epoch, start_sample, capture_source=None):
    seed = f'{job_id}:{epoch}:{start_sample}'
    if capture_source is not None:
        seed += ':'+json.dumps(capture_source, sort_keys=True, separators=(',', ':'))
    return 'utterance-'+hashlib.sha256(seed.encode()).hexdigest()[:24]


def speaker_id(source_id, local_id):
    """Disjoint stable namespaces for measured source-local acoustic tracks."""
    match = re.fullmatch(r'speaker_(\d+)', local_id)
    if source_id not in SOURCE_IDS or not match or int(match[1]) >= 32:
        raise ValueError('Source speaker track exceeds the bounded namespace.')
    return f'speaker_{32*SOURCE_IDS.index(source_id)+int(match[1])}'


def interval_pcm_digest(path,start_sample,end_sample):
    """Retained physical PCM16 identity, including anchors longer than ASR caps."""
    import wave
    if type(start_sample) is not int or type(end_sample) is not int or not 0<=start_sample<end_sample:
        raise ValueError('Invalid capture PCM anchor.')
    digest=hashlib.sha256()
    with wave.open(str(path),'rb') as source:
        if (source.getnchannels(),source.getsampwidth(),source.getframerate())!=(1,2,RATE) or source.getnframes()<end_sample:
            raise ValueError('Retained capture PCM does not cover its anchor.')
        source.setpos(start_sample);remaining=end_sample-start_sample
        while remaining:
            count=min(65536,remaining);data=source.readframes(count)
            if len(data)!=count*2:raise ValueError('Incomplete retained capture PCM.')
            digest.update(data);remaining-=count
    return digest.hexdigest()


def validate_inspections(job, folder, value, *, phase, request_id, received_sample,
                         observed_sample, uncertain_samples=None):
    """Validate each actual PCM receipt; never disguise two ledgers as mono."""
    from admission_receipt import validate_receipt, retained_pcm_digest
    validate_catalog(job['capture_source_catalog'], job['id'])
    identities = job.get('source_admission_executions')
    if (not isinstance(value, dict) or set(value) != set(SOURCE_IDS)
            or not isinstance(identities, dict) or set(identities) != set(SOURCE_IDS)):
        raise ValueError('Incomplete capture-source inspection.')
    if len({r.get('execution_id') for r in identities.values()})!=len(SOURCE_IDS):
        raise ValueError('Capture sources share one inspection execution.')
    result = {}
    for source_id in SOURCE_IDS:
        receipt = value[source_id]
        endpoint = receipt.get('end_sample') if isinstance(receipt, dict) else None
        result[source_id] = validate_receipt(receipt, identities[source_id],
            phase=phase, request_id=request_id, received_sample=received_sample,
            observed_sample=endpoint,
            pcm_sha256=retained_pcm_digest(source_path(Path(folder)/'audio.wav', source_id), endpoint))
    if observed_sample != min(r['end_sample'] for r in result.values()):
        raise ValueError('Capture-source summary horizon differs.')
    # Summary is a maximum of independent counts, not a claimed union of audio.
    if uncertain_samples is not None and uncertain_samples != max(r['uncertain_samples'] for r in result.values()):
        raise ValueError('Capture-source summary uncertainty differs.')
    return result
