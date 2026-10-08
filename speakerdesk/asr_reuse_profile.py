"""Portable qualification namespace for selected-live decoder result reuse.

Validation reads pinned files and metadata only. Any mismatch disables reuse;
current admission, routing, decoding, timing and authority checks remain active.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

PROFILE_SHA256 = 'a0f67fd3a6e994fe567d381cda916a091c958c65a495f047482a0108d284b4a7'
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


def qualified_identity(directory, provider, mx):
    """Return the verified portable namespace, or raise before enabling reuse."""
    from final_asr_reuse import provider_state
    if (not mx.metal.is_available() or mx.default_device().type != mx.gpu
            or mx.default_stream(mx.gpu).device.type != mx.gpu):
        raise ValueError('ASR reuse requires the qualified GPU device and stream.')
    approved = profile()
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
        from final_asr_reuse import FinalAsrReuse, digest
        cache = FinalAsrReuse(identity, mode=mode, qualification=digest(identity))
        return cache, {'enabled': True, 'mode': mode,
                       'identity_sha256': cache.identity_sha256,
                       'qualification': 'inherited_combined_engine_policy',
                       'final_namespace_execution_qualified': False}
    except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        # This optimization is optional. Failure never substitutes cached text
        # or changes the fresh recognizer's normal behavior.
        return None, {'enabled': False, 'reason': type(exc).__name__ + ': ' + str(exc)}
