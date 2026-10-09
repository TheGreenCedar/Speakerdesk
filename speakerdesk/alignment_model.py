"""Portable, deterministic data preparation for the pinned GPU CTC model.

NumPy/stdlib bookkeeping only; no ONNX session, neural inference or download.
The committed byte map is usable only after the entire source digest matches.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

MANIFEST_SHA256 = '6b729731170447b9680faf87287cfac235500770b5135729e970fafacf390f9b'
MODEL_SHA256 = '38ff6a225e75caa6e19c35a9b5823015e2550451bd0a394a10f6cda87f42058d'
CALIBRATION_ID = 'ami-english-coarse-mlx-f32-v1-5ab4e661e62f-38ff6a225e75'
TOKEN_SHA256 = 'a7a044c52cb29cbe8b0dc1953e92cefd4ca16b0ed968177b6beab21f9a7d0b31'
DERIVED_DIRECTORY = 'mlx-f32-v1-38ff6a225e75'
DATA = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)) / 'alignment-mlx'

def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def manifest():
    path = DATA / 'manifest.json'
    if digest_file(path) != MANIFEST_SHA256:
        raise ValueError('GPU alignment tensor manifest differs.')
    result = json.loads(path.read_text())
    if (result['schema_version'] != 1 or result['converted_weights_sha256'] != MODEL_SHA256
            or result['tokens_sha256'] != TOKEN_SHA256 or result['calibration_id'] != CALIBRATION_ID):
        raise ValueError('GPU alignment model/calibration identity differs.')
    return result

def require_runtime(metadata):
    """Exact tested runtime and forward components; no alternate backend."""
    for name, version in metadata['runtime'].items():
        if name == 'component_source_sha256':
            continue
        if importlib.metadata.version(name) != version:
            raise RuntimeError('GPU alignment requires the packaged runtime: ' + name)
    # The private copies are byte-identical to the calibrated upstream files,
    # but bypass its eager public STT-family initializer. Verify the sources
    # actually used by our adapter, in both ordinary and frozen runtimes.
    from mlx_ctc_components import source_files, BASE_SOURCE_SHA256
    files, base = source_files()
    expected_files = metadata['runtime']['component_source_sha256']
    if set(files) != set(expected_files):
        raise RuntimeError('GPU alignment forward component inventory differs.')
    for name, expected in expected_files.items():
        if digest_file(files[name]) != expected:
            raise RuntimeError('GPU alignment forward component differs: ' + name)
    if digest_file(base) != BASE_SOURCE_SHA256:
        raise RuntimeError('GPU alignment base component differs.')

def prepared(directory):
    """Presence/size check for optional setup status; loader verifies digests."""
    try:
        metadata = manifest()
        path = Path(directory) / DERIVED_DIRECTORY / 'weights.npz'
        return path.is_file() and path.stat().st_size == metadata['converted_weights_bytes']
    except (OSError, ValueError, KeyError):
        return False

def _tensor(stream, info):
    import numpy as np
    dtype = np.dtype(info['dtype'])
    size = math.prod(info['shape']) * dtype.itemsize
    if 'scalar_hex' in info:
        raw = bytes.fromhex(info['scalar_hex'])
    else:
        if info['bytes'] != size:
            raise ValueError('Source tensor byte length differs.')
        stream.seek(info['offset'])
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('Source tensor data is truncated.')
    return np.frombuffer(raw, dtype=dtype).reshape(info['shape'])

def _convert(source, destination, metadata):
    import numpy as np
    used = set()
    with source.open('rb') as stream, zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in metadata['parameters']:
            inputs = []
            for name in item['sources']:
                used.add(name)
                inputs.append(_tensor(stream, metadata['initializers'][name]))
            transform = item['transform']
            if transform == 'identity':
                value = inputs[0]
            elif transform == 'OI-kernel_to_O-kernel-I':
                value = inputs[0].transpose(0, 2, 1)
            elif transform == 'UINT8_dequantize_then_I-O_to_O-I':
                q, scale, zero = inputs
                value = ((q.astype(np.float32) - np.float32(zero.item())) * np.float32(scale.item())).T
            else:
                raise ValueError('Unknown pinned tensor transform.')
            value = np.ascontiguousarray(value)
            if (value.dtype != np.float32 or list(value.shape) != item['shape']
                    or not np.isfinite(value).all()
                    or hashlib.sha256(value.tobytes()).hexdigest() != item['converted_tensor_sha256']):
                raise ValueError('Converted tensor identity differs: ' + item['target'])
            member = zipfile.ZipInfo(item['target'] + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            with archive.open(member, 'w', force_zip64=True) as output:
                np.lib.format.write_array(output, value, allow_pickle=False)
            del inputs, value
    if used != set(metadata['initializers']):
        raise ValueError('Unmapped source initializer.')

def prepare_model(directory):
    """Verify source and atomically prepare/reuse exact converted GPU weights."""
    from filelock import FileLock
    metadata = manifest()
    folder = Path(directory)
    source = folder / 'model.int8.onnx'
    if (source.stat().st_size != metadata['source_model_bytes']
            or digest_file(source) != metadata['source_model_sha256']
            or digest_file(folder / 'tokens.txt') != TOKEN_SHA256):
        raise ValueError('Pinned alignment source model or tokens differ.')
    target = folder / DERIVED_DIRECTORY
    target.mkdir(exist_ok=True)
    with FileLock(str(target / 'prepare.lock'), timeout=60):
        weights = target / 'weights.npz'
        if weights.exists():
            if (weights.stat().st_size != metadata['converted_weights_bytes']
                    or digest_file(weights) != MODEL_SHA256):
                raise ValueError('Prepared GPU alignment weights differ; refusing to load.')
            return target, metadata
        # Actual output requirement plus bounded write reserve, not a fixed host floor.
        if shutil.disk_usage(target).free < metadata['converted_weights_bytes'] + 64 * 1024**2:
            raise OSError('Insufficient disk space to prepare optional English timing.')
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=target) as temporary:
            staged = Path(temporary) / 'weights.npz'
            _convert(source, staged, metadata)
            if staged.stat().st_size != metadata['converted_weights_bytes'] or digest_file(staged) != MODEL_SHA256:
                raise ValueError('Prepared GPU alignment digest differs.')
            staged.replace(weights)
    return target, metadata
