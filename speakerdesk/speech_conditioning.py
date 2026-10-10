"""Source-bound VAD gain, with explicit qualified and experimental contracts.

Qualified app inputs require a server-owned producer receipt. Observed Apple
probe taps use only the separate isolated validation opt-in; raw, unknown,
import and system inputs retain the established quiet-speech gain behavior.
"""
import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType

SOURCE_UNITY_POLICY = 'apple_active_microphone_tap_raw_and_unity_view_v1'
PRODUCTION_UNITY_POLICY = 'qualified_apple_processed_microphone_raw_and_unity_view_v1'


@dataclass(frozen=True)
class SourceConditioning:
    contract_sha256: str
    capture_source: dict
    input_policy: str = SOURCE_UNITY_POLICY

    def __post_init__(self):
        source = self.capture_source
        if (not re.fullmatch(r'[0-9a-f]{64}', str(self.contract_sha256))
                or self.input_policy not in (SOURCE_UNITY_POLICY, PRODUCTION_UNITY_POLICY)
                or not isinstance(source,dict) or set(source) != {'catalog_sha256','source_id','source_revision'}
                or source['source_id'] != 'microphone_clean' or type(source['source_revision']) is not int
                or source['source_revision'] != 1
                or not re.fullmatch(r'[0-9a-f]{64}', str(source['catalog_sha256']))):
            raise ValueError('Invalid source conditioning identity.')
        object.__setattr__(self, 'capture_source', MappingProxyType(dict(source)))

    def gain(self, peak):
        if type(peak) not in (int, float) or not math.isfinite(peak) or peak < 0:
            raise ValueError('Invalid physical frame peak.')
        return 1.


def resolve(config):
    """Resolve qualified server provenance or the separate validation opt-in.

    Source names and enabled graph flags alone never select app unity gain.
    Unknown/raw/import/system peers keep the existing quiet-speech behavior.
    """
    if config.get('source_processing_metadata') is not None or config.get('capture_processing_reference') is not None:
        from capture_processing import worker_metadata, PROCESSED, digest
        metadata = worker_metadata(config)
        # The root shared-weight owner has no per-source input. Validate first,
        # then choose conditioning only when the source router creates peers.
        actual = config.get('capture_source')
        if actual is None and config.get('capture_source_catalog'):
            return None
        source = metadata['capture_source']
        if (not isinstance(actual,dict) or set(actual) != {'catalog_sha256','source_id','source_revision'}
                or type(actual.get('source_revision')) is not int
                or actual.get('catalog_sha256') != source['catalog_sha256']
                or actual.get('source_revision') != source['source_revision']):
            raise ValueError('Processing policy crossed its retained source identity.')
        if actual.get('source_id') == 'system':
            return None
        if actual != source:
            raise ValueError('Processing policy is not its microphone input.')
        if metadata['classification'] != PROCESSED:
            return None
        return SourceConditioning(digest(metadata),dict(source),PRODUCTION_UNITY_POLICY)
    enabled = config.get('experimental_apple_vad_gain', False)
    if type(enabled) is not bool:
        raise ValueError('Invalid VAD conditioning experiment flag.')
    if not enabled:
        return None
    contract = config.get('capture_processing_contract')
    required = {'processor', 'mode', 'output', 'job_id', 'capture_source',
                'configuration_sha256', 'receipt_sha256'}
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError('Missing bound capture graph contract.')
    source = contract['capture_source']
    if (contract['processor'] != 'apple_voice_processing_io'
            or contract['mode'] != 'active' or contract['output'] != 'observed_microphone_tap'
            or contract['job_id'] != config.get('job_id')
            or not isinstance(source, dict) or set(source) != {'catalog_sha256','source_id','source_revision'}
            or source['source_id'] != 'microphone_clean' or type(source['source_revision']) is not int
            or source['source_revision'] != 1
            or not re.fullmatch(r'[0-9a-f]{64}', str(source['catalog_sha256']))
            or any(not re.fullmatch(r'[0-9a-f]{64}', str(contract[k]))
                   for k in ('configuration_sha256', 'receipt_sha256'))):
        raise ValueError('Unsupported capture graph provenance.')
    actual = config.get('capture_source')
    # A source router may carry the recording's mic descriptor to system; that
    # peer keeps legacy gain only when its recording catalog identity matches.
    if isinstance(actual,dict) and actual.get('source_id') == 'system':
        if (set(actual) != {'catalog_sha256','source_id','source_revision'}
                or type(actual.get('source_revision')) is not int
                or actual.get('catalog_sha256') != source['catalog_sha256']
                or actual.get('source_revision') != source['source_revision']):
            raise ValueError('System conditioning crossed its recording.')
        return None
    if actual != source:
        raise ValueError('VAD conditioning crossed its capture source.')
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    return SourceConditioning(digest, dict(source))
