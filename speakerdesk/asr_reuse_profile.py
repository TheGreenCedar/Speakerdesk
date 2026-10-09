"""Portable qualification namespace for selected-live decoder result reuse.

Validation reads pinned files and metadata only. Any mismatch disables reuse;
current admission, routing, decoding, timing and authority checks remain active.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

PROFILE_SHA256 = 'c61ad59ca8cdcf90e47b332fef192ad279f685d1aa85d6a8c467cd27e14ed777'
SOURCE_PROFILE_SHA256 = '3d3ac2dfaaca83de54ac9bcafa9fec14c4f3da32444e7284ede224493d5b6e62'
DATA = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
PROCESS_FIELDS = ('model_instance', 'tokenizer_instance')


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def profile():
    path = DATA / 'asr-reuse-profile.json'
    if file_digest(path) != PROFILE_SHA256:
        raise ValueError('ASR reuse profile differs.')
    result = json.loads(path.read_text())
    if result['schema_version'] != 1 or result['identity']['device'] != 'gpu':
        raise ValueError('ASR reuse profile is unsupported.')
    return result


def current_profile():
    """An independently sealed current namespace preserves its legacy lineage."""
    legacy = profile()
    path = DATA / 'asr-source-reuse-profile.json'
    if not path.exists():
        return legacy
    if file_digest(path) != SOURCE_PROFILE_SHA256:
        raise ValueError('Current ASR reuse profile differs.')
    result = json.loads(path.read_text())
    lineage = result.get('lineage', {})
    if (result.get('schema_version') != 1 or lineage.get('legacy_profile_sha256') != PROFILE_SHA256
            or lineage.get('reuse_algorithm_sha256') != legacy['identity']['policy_source_sha256']['final_asr_reuse.py']
            or result.get('namespace') != 'source_bound_canonical_v1'):
        raise ValueError('Current ASR reuse lineage differs.')
    # Only policy identity is extended. Decoder/model/runtime/loaded precision
    # must remain the exact independently qualified inherited backend.
    old = {k:v for k,v in legacy['identity'].items() if k != 'policy_source_sha256'}
    new = {k:v for k,v in result['identity'].items() if k != 'policy_source_sha256'}
    if old != new:
        raise ValueError('Current ASR reuse backend lineage differs.')
    old_policy=legacy['identity']['policy_source_sha256']
    new_policy=result['identity']['policy_source_sha256']
    if (not set(old_policy)<=set(new_policy) or any(new_policy[name]!=old_policy[name]
            for name in ('final_asr_reuse.py','language_detection.py','language_probe_cache.py') if name in old_policy)):
        raise ValueError('Current ASR reuse algorithm or routing lineage differs.')
    return result


def qualified_identity(directory, provider, mx):
    """Return the verified portable namespace, or raise before enabling reuse."""
    from final_asr_reuse import provider_state
    if (not mx.metal.is_available() or mx.default_device().type != mx.gpu
            or mx.default_stream(mx.gpu).device.type != mx.gpu):
        raise ValueError('ASR reuse requires the qualified GPU device and stream.')
    approved = current_profile()
    identity = approved['identity']
    for name, expected in identity['model_file_sha256'].items():
        if file_digest(Path(directory) / name) != expected:
            raise ValueError('ASR checkpoint differs: ' + name)
    for name, expected in identity['runtime_versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('ASR runtime differs: ' + name)
    distribution = importlib.metadata.distribution('mlx-speech')
    for name, expected in identity['decoder_source_sha256'].items():
        if file_digest(distribution.locate_file(name)) != expected:
            raise ValueError('ASR decoder differs: ' + name)
    for name, expected in identity['policy_source_sha256'].items():
        if file_digest(DATA / name) != expected:
            raise ValueError('ASR policy differs: ' + name)
    state = provider_state(provider)
    for name in PROCESS_FIELDS:
        state.pop(name, None)
    if state != identity['loaded_provider_state']:
        raise ValueError('Loaded ASR configuration or precision differs.')
    # Process-local provider/tokenizer identities are freshly read by each
    # decoder signature; they are never copied from the qualification report.
    return identity


def initialize_reuse(config, provider, mx):
    """Published Models default: qualified reuse, explicit observe/off, else fresh."""
    mode = config.get('asr_final_reuse_mode', 'reuse')
    if mode in (None, 'off'):
        return None, {'enabled': False, 'reason': 'explicitly_disabled'}
    try:
        if mode not in ('reuse', 'observe'):
            raise ValueError('Unsupported ASR reuse mode.')
        identity = qualified_identity(config['cohere_path'], provider, mx)
        approved = current_profile()
        current = approved.get('namespace') == 'source_bound_canonical_v1'
        execution_qualified = approved.get('qualification', {}).get('current_namespace_execution_qualified') is True
        if current and mode == 'reuse' and not execution_qualified:
            raise ValueError('Current ASR reuse namespace needs its observer qualification.')
        from final_asr_reuse import FinalAsrReuse, digest
        cache = FinalAsrReuse(identity, mode=mode,
            qualification=digest(identity) if not current or execution_qualified else None)
        return cache, {'enabled': True, 'mode': mode,
                       'identity_sha256': cache.identity_sha256,
                       'qualification': 'source_bound_observer' if current else 'inherited_combined_engine_policy',
                       'final_namespace_execution_qualified': execution_qualified if current else False}
    except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        # This optimization is optional. Failure never substitutes cached text
        # or changes the fresh recognizer's normal behavior.
        return None, {'enabled': False, 'reason': type(exc).__name__ + ': ' + str(exc)}
