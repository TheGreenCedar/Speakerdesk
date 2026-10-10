"""Server-owned capture processing provenance; source names confer no authority.

The reviewed helper capability and post-start node/state/format packet bind
Apple input-node output to retained PCM. Unknown routes retain legacy gain.
Build capabilities are server-owned, never supplied through HTTP.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from capture_sources import binding, validate_catalog

BASENAME = 'capture-processing.json'
EXPORT_CONTRACT = 'apple_processed_mono16k_v1'
CAPABILITY_VERSION = 'apple_input_node_output_route_v1'
LEGACY = 'legacy_unknown_or_raw'
PROCESSED = 'qualified_apple_active_processed_microphone'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def hash_value(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None

class ProducerRegistry:
    def __init__(self, qualifications=()):
        self._qualifications = {}
        for value in qualifications:
            if (not isinstance(value,dict) or set(value) != {'helper_sha256','export_contract','capability_version'}
                    or not hash_value(value['helper_sha256']) or value['capability_version'] != CAPABILITY_VERSION
                    or value['export_contract'] != EXPORT_CONTRACT):
                raise ValueError('Invalid trusted capture producer qualification.')
            key = value['helper_sha256']
            if key in self._qualifications:
                raise ValueError('Duplicate capture producer qualification.')
            self._qualifications[key] = copy.deepcopy(value)

    @classmethod
    def for_helper(cls, helper_path):
        helper = Path(helper_path)
        manifest = Path(str(helper)+'.capability.json')
        if not manifest.exists() and not manifest.is_symlink():
            return cls()
        if (helper.is_symlink() or not helper.is_file() or manifest.is_symlink()
                or not manifest.is_file() or manifest.stat().st_uid != os.getuid()
                or manifest.stat().st_size > 4096):
            raise ValueError('Unsafe capture helper capability manifest.')
        entry = json.loads(manifest.read_bytes())
        registry = cls([entry])
        if entry['helper_sha256'] != file_digest(helper):
            raise ValueError('Capture helper capability differs from its built artifact.')
        return registry

    def qualification(self, helper_sha256):
        return copy.deepcopy(self._qualifications.get(helper_sha256))

def validate_metadata(value, job_id, catalog, registry=None):
    validate_catalog(catalog,job_id)
    required = {'schema_version','job_id','capture_source','producer','classification','qualification'}
    if (not isinstance(value,dict) or set(value) != required or value['schema_version'] != 1
            or type(value['schema_version']) is not int or value['job_id'] != job_id
            or value['capture_source'] != binding(catalog,'microphone_clean')):
        raise ValueError('Capture processing crossed its recording/source.')
    packet = value['producer']
    packet_keys = {'type','schema_version','job_id','helper_sha256','processor','export_contract',
                   'output','stage','postprocessing','stream_epoch','format','readbacks',
                   'capability_version','route','state_phase','channel_mapping','callback_format','io_formats'}
    if (not isinstance(packet,dict) or set(packet) != packet_keys
            or packet['type'] != 'capture_processing' or type(packet['schema_version']) is not int
            or packet['schema_version'] != 1 or packet['job_id'] != job_id
            or not hash_value(packet['helper_sha256']) or type(packet['stream_epoch']) is not int
            or packet['stream_epoch'] != 0):
        raise ValueError('Unsupported capture processing epoch/producer.')
    if not isinstance(packet['route'],dict):
        raise ValueError('Capture processing route is malformed.')
    readbacks = packet['readbacks']
    if (not isinstance(readbacks,dict) or set(readbacks) != {'input_enabled','output_enabled','engine_running','bypassed','manual_rendering','input_muted','agc_enabled'}
            or any(type(x) is not bool for x in readbacks.values())):
        raise ValueError('Capture processing requires typed runtime readbacks.')
    audio_format = packet['format']
    if (not isinstance(audio_format,dict) or set(audio_format) != {'sample_rate','channels','layout'}
            or type(audio_format['sample_rate']) is not int or type(audio_format['channels']) is not int
            or audio_format['sample_rate'] <= 0 or audio_format['channels'] <= 0
            or not isinstance(audio_format['layout'],str)):
        raise ValueError('Capture processing format is absent or malformed.')
    callback_format = packet['callback_format']
    io_formats = packet['io_formats']
    def valid_format(fmt):
        return (isinstance(fmt,dict) and set(fmt) == {'sample_rate','channels','layout'}
                and type(fmt['sample_rate']) is int and fmt['sample_rate'] > 0
                and type(fmt['channels']) is int and fmt['channels'] > 0
                and isinstance(fmt['layout'],str))
    if (not valid_format(callback_format) or not isinstance(io_formats,dict)
            or set(io_formats) != {'input_output','output_input'}
            or not all(valid_format(fmt) for fmt in io_formats.values())):
        raise ValueError('Capture processing requires post-start I/O and callback formats.')
    mapping_supported = ((callback_format['channels'] == 1 and callback_format['layout'] == 'mono'
                          and packet['channel_mapping'] == 'mono_direct')
                         or (callback_format['channels'] == 9
                             and callback_format['layout'] == 'discrete_float32_noninterleaved'
                             and packet['channel_mapping'] == 'nine_discrete_exact_replicas'))
    qualification = value['qualification']
    supported = (packet['processor'] == 'apple_voice_processing_io'
        and packet['capability_version'] == CAPABILITY_VERSION
        and packet['export_contract'] == EXPORT_CONTRACT and packet['output'] == 'processed_microphone'
        and packet['route'] == {'node':'inputNode','bus':0,'tap_scope':'output'}
        and type(packet['route'].get('bus')) is int and packet['state_phase'] == 'after_start'
        and packet['stage'] == 'retained_source_pcm' and packet['postprocessing'] == 'identity'
        and audio_format == {'sample_rate':16000,'channels':1,'layout':'mono'}
        and mapping_supported and callback_format == io_formats['input_output'] == io_formats['output_input']
        and readbacks['input_enabled'] and readbacks['output_enabled'] and readbacks['engine_running']
        and not readbacks['bypassed'] and not readbacks['manual_rendering'] and not readbacks['input_muted'])
    if qualification is not None:
        if (not isinstance(qualification,dict) or set(qualification) != {'helper_sha256','export_contract','capability_version'}
                or qualification['helper_sha256'] != packet['helper_sha256']
                or qualification['export_contract'] != EXPORT_CONTRACT or qualification['capability_version'] != CAPABILITY_VERSION):
            raise ValueError('Capture output qualification differs from its producer.')
        if registry is not None and registry.qualification(packet['helper_sha256']) != qualification:
            raise ValueError('Capture producer is not in the application qualification registry.')
    # A downstream innovation/WebRTC stage cannot be renamed an identity
    # Apple output. A qualified exporter must retain its own mono PCM directly.
    supported = supported and catalog['schema_version'] == 1
    expected = PROCESSED if supported and qualification is not None else LEGACY
    if value['classification'] != expected:
        raise ValueError('Capture processing classification is not supported by evidence.')
    return copy.deepcopy(value)

def install(job, folder, packet, registry, helper_path):
    """Called on an owned native stdout packet before any source PCM advances."""
    if job.get('kind') != 'meeting':
        raise ValueError('Imports cannot claim native capture processing.')
    if job.get('capture_processing') is not None:
        raise ValueError('Capture processing cannot change within the recording epoch.')
    if not job.get('capture_source_catalog'):
        raise ValueError('Capture processing requires a fixed retained-source catalog.')
    helper = Path(helper_path)
    if helper.is_symlink() or not helper.is_file() or packet.get('helper_sha256') != file_digest(helper):
        raise ValueError('Capture processing did not come from the owned helper identity.')
    qualification = registry.qualification(packet['helper_sha256'])
    value = {'schema_version':1,'job_id':job['id'],
        'capture_source':binding(job['capture_source_catalog'],'microphone_clean'),
        'producer':copy.deepcopy(packet),'classification':LEGACY,'qualification':qualification}
    # Derive eligibility, never accept a client-supplied classification.
    provisional = copy.deepcopy(value)
    provisional['classification'] = PROCESSED
    try:
        value = validate_metadata(provisional,job['id'],job['capture_source_catalog'],registry)
    except ValueError:
        value = validate_metadata(value,job['id'],job['capture_source_catalog'],registry)
    folder = Path(folder)
    path = folder/BASENAME
    if path.exists() or path.is_symlink():
        raise ValueError('Capture processing receipt already exists.')
    encoded = json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(encoded) > 8192:
        raise ValueError('Capture processing receipt is too large.')
    with path.open('xb') as stream:
        os.chmod(path,0o600)
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    reference = {'basename':BASENAME,'sha256':hashlib.sha256(encoded).hexdigest()}
    job['capture_processing'] = reference
    return value

def configure(config, job, folder, registry):
    """The immutable server receipt is identical for live and saved workers."""
    result = dict(config)
    for key in ('source_processing_metadata','capture_processing_reference','experimental_apple_vad_gain','capture_processing_contract'):
        result.pop(key,None)
    reference = job.get('capture_processing')
    if reference is None:
        return result
    if job.get('kind') != 'meeting':
        raise ValueError('Imports cannot inherit native capture processing.')
    if (not isinstance(reference,dict) or set(reference) != {'basename','sha256'}
            or reference['basename'] != BASENAME or not hash_value(reference['sha256'])):
        raise ValueError('Capture processing reference is not server owned.')
    folder = Path(folder).resolve()
    path = folder/BASENAME
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_size > 8192:
        raise ValueError('Capture processing receipt is missing or unsafe.')
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != reference['sha256']:
        raise ValueError('Capture processing receipt changed after publication.')
    value = validate_metadata(json.loads(encoded),job['id'],job.get('capture_source_catalog'),registry)
    result['source_processing_metadata'] = value
    result['capture_processing_reference'] = copy.deepcopy(reference)
    return result

def worker_metadata(config):
    """Recheck the server's fixed receipt before any worker/model state exists.

    Worker config is private parent IPC. HTTP clients cannot construct this
    envelope or choose the trusted registry used to install its backing file.
    """
    value = config.get('source_processing_metadata')
    reference = config.get('capture_processing_reference')
    if value is None and reference is None:
        return None
    if (not isinstance(reference,dict) or set(reference) != {'basename','sha256'}
            or reference['basename'] != BASENAME or not hash_value(reference['sha256'])):
        raise ValueError('Worker processing metadata has no server receipt binding.')
    audio_path = Path(config['audio_path'])
    if audio_path.name not in ('audio.wav','microphone_clean.wav','system.wav') or audio_path.is_symlink():
        raise ValueError('Worker processing source path is not fixed.')
    path = audio_path.with_name(BASENAME)
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_size > 8192:
        raise ValueError('Worker processing receipt is absent or unsafe.')
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != reference['sha256'] or json.loads(encoded) != value:
        raise ValueError('Worker processing receipt/config changed after publication.')
    return validate_metadata(value,config['job_id'],config.get('capture_source_catalog'))
