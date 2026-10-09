"""Pinned timing-provider hooks; replace together when multilingual qualification is accepted.

No download or preference ownership. verify() performs no neural initialization;
gpu_check() evaluates bounded supplied audio on the required GPU in a child.
"""
import importlib.machinery
import importlib.util
import hashlib
import json
import resource
from pathlib import Path
import sys
import threading
from alignment_artifact import ALIGNMENT_SPEC
from alignment_model import (DERIVED_DIRECTORY, MODEL_SHA256, MANIFEST_SHA256,
                             digest_file, manifest as _tensor_manifest, prepare_model, require_runtime as _require_runtime)
from alignment_text import (PROVIDER_ID, SUPPORTED_LANGUAGES, CALIBRATED_LANGUAGES,
                            NORMALIZATION_POLICY_ID, TIMING_UNIT_POLICY_ID)

_cache = {}
_cache_lock = threading.RLock()

def manifest():
    """Bind reusable tensor data to the exact installed behavior, without neural initialization."""
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
    identity=json.loads((root/'alignment-mlx'/'provider-identity.json').read_text())
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    digest=hashlib.sha256(encoded).hexdigest()
    if (digest!=ALIGNMENT_SPEC['provider_identity_sha256'] or identity['schema_version']!=1
            or identity['provider_id']!=PROVIDER_ID or identity['supported_languages']!=list(SUPPORTED_LANGUAGES)
            or identity['timing_accuracy_calibrated_languages']!=list(CALIBRATED_LANGUAGES)
            or identity['normalization_policy_id']!=NORMALIZATION_POLICY_ID
            or identity['timing_unit_policy_id']!=TIMING_UNIT_POLICY_ID):
        raise ValueError('Transcript timing behavior identity differs. Update Speakerdesk.')
    source_root=root
    for relative,expected in identity['implementation_source_sha256'].items():
        if digest_file(source_root/relative)!=expected:
            raise ValueError('Transcript timing implementation differs: '+relative)
    return {**_tensor_manifest(),'provider_identity':identity,'provider_identity_sha256':digest}

def require_native_runtime():
    """Load only the non-neural native timing extension, detecting ABI mismatch."""
    package = importlib.util.find_spec('ctc_segmentation')
    native = package and importlib.machinery.PathFinder.find_spec(
        'ctc_segmentation.ctc_segmentation_dyn', package.submodule_search_locations)
    if native is None or not isinstance(native.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError('Transcript timing needs the packaged native timing runtime. Update Speakerdesk.')
    if native.name not in sys.modules:
        extension = importlib.util.module_from_spec(native)
        native.loader.exec_module(extension)
        sys.modules[native.name] = extension


def require_runtime(metadata):
    _require_runtime(metadata)
    require_native_runtime()


def verify(directory):
    """Verify every artifact and the exact runtime without initializing neural inference."""
    folder = Path(directory)
    metadata = manifest()
    require_runtime(metadata)
    files = {**ALIGNMENT_SPEC['file_sha256'], DERIVED_DIRECTORY+'/weights.npz': MODEL_SHA256}
    signature = tuple((name, (folder/name).stat().st_size, (folder/name).stat().st_mtime_ns,
                       (folder/name).stat().st_ctime_ns, (folder/name).stat().st_ino)
                      for name in files)
    with _cache_lock:
        if _cache.get(folder) == signature:
            return metadata
    for name, digest in files.items():
        if digest_file(folder/name) != digest:
            raise ValueError('Transcript timing model verification failed. Retry setup.')
    if (folder/DERIVED_DIRECTORY/'weights.npz').stat().st_size != metadata['converted_weights_bytes']:
        raise ValueError('Prepared transcript timing model is incomplete. Retry setup.')
    with _cache_lock:
        _cache[folder] = signature
    return metadata


def gpu_check(directory):
    """Disposable worker: evaluate real GPU scores; never use a CPU neural fallback."""
    require_native_runtime()
    import numpy as np
    from coarse_alignment import CoarseAlignment
    from mlx_ctc_forward import forward_scores
    provider = CoarseAlignment(directory)
    execution = {}
    scores = forward_scores(provider.model, np.zeros(8000, dtype=np.float32), execution=execution)
    if (execution.get('backend') != 'mlx_metal_gpu'
            or execution.get('evaluated_and_GPU_synchronized') is not True
            or not np.isfinite(np.asarray(scores)).all()):
        raise RuntimeError('Transcript timing GPU compatibility check failed.')
    import mlx.core as mx
    metadata=manifest()
    return {'model_sha256': MODEL_SHA256, 'manifest_sha256': MANIFEST_SHA256,
            'provider_id':metadata['provider_identity']['provider_id'],
            'provider_identity_sha256':metadata['provider_identity_sha256'],
            'runtime': provider.conversion['runtime'], 'execution': execution,
            'metrics': {'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'peak_mlx_bytes': mx.get_peak_memory()}}


def validate_receipt(receipt, metadata):
    execution = receipt['execution']
    if (receipt['model_sha256'] != MODEL_SHA256 or receipt['manifest_sha256'] != MANIFEST_SHA256
            or receipt.get('provider_id') != metadata['provider_identity']['provider_id']
            or receipt.get('provider_identity_sha256') != metadata['provider_identity_sha256']
            or receipt['runtime'] != metadata['runtime'] or execution['backend'] != 'mlx_metal_gpu'
            or execution['evaluated_and_GPU_synchronized'] is not True):
        raise ValueError('Transcript timing needs a new GPU compatibility check. Retry setup.')
