"""Pinned local ReDimNet2 extraction; no downloads, audio uploads or startup prediction.

Waveform checks screen silence/clipping, not acoustic overlap or noise identity.
Callers must also supply reviewed single-speaker finalized passages.
"""
import hashlib
import importlib.metadata
import json
import math
import platform
import threading
import time
import wave
from pathlib import Path
import numpy as np
from voice_profiles import Calibration, ClipEmbedding, VoiceModel, normalized

PIN = json.loads(Path(__file__).with_name('redimnet2_artifact.json').read_text())
COMPUTE_UNITS = ('ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE')
RATE, INPUT_SAMPLES, DIMENSION = 16000, 96000, 192


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(512*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(compute_units):
    if compute_units not in COMPUTE_UNITS:
        raise ValueError('Choose a supported Core ML compute policy.')
    digest = hashlib.sha256(json.dumps(PIN['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return VoiceModel(PIN['model_id'], f"{PIN['revision']}:sd-pcm-v1:{compute_units}", digest, DIMENSION)


def verify_artifact(root):
    """Validate local bytes against the source-owned pin before native loading."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Voice model directory is missing or is a symbolic link.')
    for relative, expected in PIN['files'].items():
        path = root/relative
        if any((root/Path(*Path(relative).parts[:i])).is_symlink() for i in range(1, len(Path(relative).parts)+1)):
            raise ValueError('Voice model files must not be symbolic links.')
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError(f'Voice model file has an unexpected size: {relative}')
        if file_digest(path) != expected['sha256']:
            raise ValueError(f'Voice model checksum does not match the approved artifact: {relative}')
    compiled = root/PIN['compiled_model']
    if any(p.is_symlink() for p in compiled.rglob('*')):
        raise ValueError('Voice model files must not be symbolic links.')
    actual = {str(p.relative_to(root)) for p in compiled.rglob('*') if not p.is_dir()}
    expected = {p for p in PIN['files'] if p.startswith(PIN['compiled_model']+'/')}
    if actual != expected:
        raise ValueError('Voice model contains unexpected compiled files.')
    config = json.loads((root/'config.json').read_text())
    contract = {'model_type': 'redimnet2-b6-speaker-coreml', 'sample_rate': RATE,
                'input_samples': INPUT_SAMPLES, 'embedding_dimension': DIMENSION,
                'input_name': 'audio', 'output_name': 'embedding', 'output_normalized': True,
                'compiled_model': PIN['compiled_model'], 'artifact_revision': PIN['artifact_revision']}
    if any(config.get(k) != v for k, v in contract.items()):
        raise ValueError('Voice model configuration is incompatible with this adapter.')


def require_runtime():
    version = platform.mac_ver()[0]
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not version or int(version.split('.')[0]) < 15:
        raise ValueError('Local voice recognition requires Apple Silicon and macOS 15 or later.')
    try:
        version = importlib.metadata.version('coremltools')
    except importlib.metadata.PackageNotFoundError:
        raise ValueError('The local voice runtime is not installed.') from None
    if version != '9.0':
        raise ValueError('This voice adapter requires the pinned Core ML Tools 9.0 runtime.')


def read_clip(audio, clip):
    if (not math.isfinite(clip.start) or not math.isfinite(clip.end)
            or not 0 <= clip.start < clip.end or not 2 <= clip.end-clip.start <= 10):
        raise ValueError('Voice passages must be 2–10 seconds long.')
    try:
        with wave.open(str(audio), 'rb') as recording:
            if (recording.getnchannels(), recording.getsampwidth(), recording.getframerate(), recording.getcomptype()) != (1, 2, RATE, 'NONE'):
                raise ValueError('Voice recognition needs normalized mono 16 kHz PCM16 meeting audio.')
            start, end = round(clip.start*RATE), round(clip.end*RATE)
            if end > recording.getnframes():
                raise ValueError('A voice passage extends beyond the saved audio.')
            recording.setpos(start)
            raw = recording.readframes(end-start)
            if len(raw) != (end-start)*2:
                raise ValueError('A voice passage is incomplete.')
    except (wave.Error, EOFError):
        raise ValueError('Voice recognition could not read the meeting audio.') from None
    return np.frombuffer(raw, dtype='<i2').astype(np.float32)/32768.


def prepared_audio(samples):
    """Fixed shape: repeat 2–6 second clips and center-crop 6–10 second clips."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or not RATE*2 <= len(samples) <= RATE*10 or not np.isfinite(samples).all():
        raise ValueError('Supply 2–10 seconds of finite mono audio.')
    if len(samples) >= INPUT_SAMPLES:
        start = (len(samples)-INPUT_SAMPLES)//2
        return np.ascontiguousarray(samples[start:start+INPUT_SAMPLES], dtype=np.float32)
    return np.ascontiguousarray(np.tile(samples, math.ceil(INPUT_SAMPLES/len(samples)))[:INPUT_SAMPLES], dtype=np.float32)


class ReDimNet2CoreML:
    def __init__(self, model_dir, compute_units='ALL'):
        self.root = Path(model_dir)
        self.compute_units = compute_units
        self.model = model_identity(compute_units)
        self._predictor = None
        self._lock = threading.RLock()

    def validate(self):
        verify_artifact(self.root)
        require_runtime()

    def embed(self, audio, clip):
        samples = read_clip(audio, clip)
        if float(np.sqrt(np.mean(samples.astype(np.float64)**2))) < 1/32768 or np.any(np.abs(samples) >= 32767/32768):
            return ClipEmbedding((), False)
        prepared = prepared_audio(samples)
        with self._lock:
            load_ms = 0.
            if self._predictor is None:
                self.validate()
                started = time.perf_counter()
                try:
                    import coremltools as ct
                    self._predictor = ct.models.CompiledMLModel(str(self.root/PIN['compiled_model']),
                                                               compute_units=ct.ComputeUnit[self.compute_units])
                except Exception as exc:
                    raise ValueError('Voice recognition could not start. Restart Speakerdesk or retry model setup.') from exc
                load_ms = (time.perf_counter()-started)*1000
            try:
                started = time.perf_counter()
                output = self._predictor.predict({'audio': prepared[None, :]})
                predict_ms = (time.perf_counter()-started)*1000
            except Exception as exc:
                raise ValueError('Voice recognition could not use this audio. Try other clean passages or restart Speakerdesk.') from exc
        if not isinstance(output, dict):
            raise ValueError('The voice model returned an invalid prediction.')
        vector = np.asarray(output.get('embedding'))
        if vector.shape not in ((DIMENSION,), (1, DIMENSION)) or vector.dtype.kind != 'f':
            raise ValueError('The voice model did not return a 192-dimensional floating-point embedding.')
        values = normalized(vector.reshape(-1).tolist(), DIMENSION)
        return ClipEmbedding(tuple(values), True, {'load_ms': load_ms, 'predict_ms': predict_ms,
                             'raw_output_norm': float(np.linalg.norm(vector.astype(np.float64))),
                             'input_samples': INPUT_SAMPLES})


def read_voice_config(config_path):
    path = Path(config_path)
    if path.stat().st_size > 65536:
        raise ValueError('Voice configuration is too large.')
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Voice configuration must be a measured JSON object.')
    return config


def approved_calibration(config, model):
    """Validate frozen policy and independent held-out evidence before recognition."""
    if config.get('approved_for_recognition') is not True:
        raise ValueError('Voice recognition needs a passing measured calibration.')
    if config.get('schema_version') != 1 or config.get('fixture_source') not in ('synthetic', 'consented', 'licensed'):
        raise ValueError('Voice configuration needs labelled fixture provenance.')
    if config.get('model') != model.payload():
        raise ValueError('Voice calibration belongs to a different model or compute policy.')
    held_out = config.get('held_out', {})
    if not isinstance(held_out, dict) or not isinstance(config.get('calibration'), dict):
        raise ValueError('Voice configuration is missing its measured evaluation results.')
    if held_out.get('meets_error_limits') is not True:
        raise ValueError('Frozen calibration did not pass the held-out error limits.')
    for key in ('genuine_trials', 'impostor_trials'):
        if not isinstance(held_out.get(key), int) or held_out[key] < 1:
            raise ValueError('Voice calibration requires independent held-out genuine and impostor trials.')
    for key in ('false_accept_rate', 'false_reject_rate'):
        value = held_out.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Voice calibration requires measured held-out error rates.')
    if not held_out.get('dataset_id') or held_out['dataset_id'] == config['calibration'].get('dataset_id'):
        raise ValueError('Use separate calibration and held-out meeting sets.')
    return Calibration(model, **config['calibration'])


def load_approved_runtime(config_path, *, model_dir=None):
    """A passing shipped calibration enables recognition; smoke results cannot."""
    config = read_voice_config(config_path)
    backend = ReDimNet2CoreML(model_dir or Path(config_path).parent/config.get('model_dir', ''), config.get('compute_units', 'ALL'))
    calibration = approved_calibration(config, backend.model)
    backend.validate()  # Hash/platform/dependency checks; no Core ML import or prediction.
    return backend, calibration
