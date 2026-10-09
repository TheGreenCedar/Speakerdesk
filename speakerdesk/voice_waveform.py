"""Exact source-extracted waveform/artifact helpers; no CoreML import or model prediction."""
import hashlib
import json
import math
import wave
from pathlib import Path
import numpy as np
PIN = json.loads(Path(__file__).with_name("redimnet2_artifact.json").read_text())
RATE, INPUT_SAMPLES, DIMENSION = 16000, 96000, 192


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(512*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
