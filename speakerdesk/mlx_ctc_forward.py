"""Unqualified cached-weight CTC forward prototype, GPU only.

Uses the pinned MIT MLX Audio Wav2Vec2 implementation and Apache-2.0 model
weights. No alternate transcription, downloader, ORT session or CPU inference.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def load_gpu_model(*, cache_limit_bytes=None, directory=None):
    # Keep import/init behind the caller's coordinated model lease.
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError('GPU required; Metal unavailable. No CPU fallback.')
    mx.set_default_device(mx.gpu)
    if mx.default_device().type != mx.gpu or mx.default_stream(mx.gpu).device.type != mx.gpu:
        raise RuntimeError('GPU device/stream policy could not be established.')
    allocator_policy = {}
    if cache_limit_bytes is not None:
        assert cache_limit_bytes == 512 * 1024**2
        allocator_policy = dict(GPU_allocator_cache_limit_bytes=cache_limit_bytes,
            previous_GPU_allocator_cache_limit_bytes=mx.set_cache_limit(cache_limit_bytes))
    import mlx.nn as nn
    from mlx_audio.stt.models.mms.mms import Model
    from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig
    from mlx.utils import tree_flatten
    folder = Path(directory) if directory is not None else ROOT / 'mlx-ctc-prototype'
    conversion = json.loads((folder / 'conversion.json').read_text())
    conversion.update(allocator_policy)
    with (folder / 'weights.npz').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != conversion['converted_weights_sha256']:
        raise ValueError('Converted weight digest changed.')
    config = ModelConfig.from_dict(json.loads((folder / 'config.json').read_text()))
    with mx.stream(mx.gpu):
        model = Model(config)
        # ONNX stores the already-folded positional kernel. Direct convolution
        # avoids an invented/rounded reconstruction of weight-normalization g/v.
        model.wav2vec2.encoder.pos_conv_embed.conv = nn.Conv1d(
            1024, 1024, 128, padding=64, groups=16, bias=True)
        weights = mx.load(str(folder / 'weights.npz'))
        wanted = {item['target']: tuple(item['shape']) for item in conversion['parameters']}
        actual = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
        if actual != wanted or set(weights) != set(wanted):
            raise ValueError('Model/config and converted tensor inventory differ.')
        if any(tuple(weights[name].shape) != shape or weights[name].dtype != mx.float32
               for name, shape in wanted.items()):
            raise ValueError('Converted tensor shape/dtype mismatch.')
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        mx.synchronize(mx.gpu)
    return model, conversion

def forward_scores(model, pcm):
    """Float32 normalized waveform -> float32 scores; never decode CTC text."""
    import mlx.core as mx
    if (not mx.metal.is_available() or mx.default_device().type != mx.gpu
            or mx.default_stream(mx.gpu).device.type != mx.gpu):
        raise RuntimeError('GPU policy lost; refusing neural inference.')
    with mx.stream(mx.gpu):
        audio = mx.array(pcm, dtype=mx.float32)
        # Preserve the normalization formula; GPU reduction order can differ.
        x = (audio - mx.mean(audio)) / mx.sqrt(mx.var(audio) + mx.array(1e-5, dtype=mx.float32))
        logits = model(x[None, :])[0]
        scores = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(scores)
        mx.synchronize(mx.gpu)
    return scores
