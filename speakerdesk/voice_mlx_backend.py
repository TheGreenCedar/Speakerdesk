"""Isolated strict GPU LocalVoiceBackend; no activation, downloads or People writes."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import threading
import time

import numpy as np
from mlx_voice_mil import Graph, GPUProgram
from voice_profiles import ClipEmbedding, VoiceModel, normalized
from voice_waveform import PIN, DIMENSION, read_clip, prepared_audio, verify_artifact


def implementation_identity():
    root = Path(__file__).parent
    identity = dict(schema_version=1, backend='speakerdesk-redimnet2-mlx-gpu-v1', runtime='mlx==0.32.2',
                    waveform_policy='sd-pcm-v1', source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ('voice_mlx_backend.py', 'mlx_voice_mil.py', 'voice_waveform.py')})
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def model_identity():
    _, implementation_sha = implementation_identity()
    artifact_sha = hashlib.sha256(json.dumps(PIN['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return VoiceModel(PIN['model_id'], f"{PIN['revision']}:sd-pcm-v1:mlx-0.32.2:GPU:{implementation_sha}", artifact_sha, DIMENSION)


class ReDimNet2MLX:
    def __init__(self, model_dir, *, require_admission, on_entered, on_declaration):
        self.root = Path(model_dir)
        self.model = model_identity()
        self.require_admission = require_admission
        self.on_entered, self.on_declaration = on_entered, on_declaration
        self._program = self._mx = None
        self._lock = threading.RLock()

    def require_execution_policy(self):
        self.require_admission()

    def validate(self):
        self.require_admission()
        if platform.system() != 'Darwin' or platform.machine() != 'arm64' or importlib.metadata.version('mlx') != '0.32.2':
            raise ValueError('Pinned Apple Silicon MLX GPU runtime required')
        verify_artifact(self.root)
        graph = Graph(self.root / PIN['compiled_model'])
        self.require_admission()
        import mlx.core as mx
        if not mx.metal.is_available():
            raise ValueError('Metal required; no CPU neural fallback')
        grant = self.require_admission()
        mx.set_memory_limit(grant['maximum_MLX_bytes'])
        mx.set_cache_limit(64 * 1024**2)
        program = GPUProgram(graph, mx)
        if program.stream.device != mx.Device(mx.gpu):
            raise ValueError('Explicit GPU stream required')
        self._mx, self._program = mx, program

    def embed(self, audio, clip):
        # Refuse before reading even an already-loaded model's next input.
        self.require_admission()
        samples = read_clip(audio, clip)
        if float(np.sqrt(np.mean(samples.astype(np.float64)**2))) < 1/32768 or np.any(np.abs(samples) >= 32767/32768):
            return ClipEmbedding((), False)
        prepared = prepared_audio(samples)
        with self._lock:
            load_ms = 0.
            if self._program is None:
                started = time.perf_counter()
                self.validate()  # Model remains resident for this worker's later clips.
                load_ms = (time.perf_counter() - started) * 1000
            self.require_admission()
            self.on_entered(clip)
            count = 0
            started = time.perf_counter()

            def declaration(node):
                nonlocal count
                grant = self.require_admission()
                peak = self._mx.get_peak_memory()
                if peak > grant['maximum_MLX_bytes']:
                    raise ValueError('MLX peak exceeded grant after synchronized declaration')
                count += 1
                self.on_declaration(node, count, peak)

            output = self._program.forward(prepared[None, :], declaration)
            self._mx.eval(output)
            self._mx.synchronize(self._program.stream)
            raw = np.asarray(output)
            if raw.shape != (1, DIMENSION) or raw.dtype != np.float32 or not np.isfinite(raw).all() or count != 809:
                raise ValueError('Complete finite192 GPU result required')
            self.require_admission()
            values = normalized(raw.reshape(-1).tolist(), DIMENSION)
            metrics = dict(load_ms=load_ms, predict_ms=(time.perf_counter() - started) * 1000,
                           raw_output_norm=float(np.linalg.norm(raw.astype(np.float64))), input_samples=96000,
                           raw_embedding=raw.reshape(-1).tolist(),
                           completed_GPU_declarations=count, stream_device=str(self._program.stream.device),
                           MLX_peak_bytes=self._mx.get_peak_memory(), CPU_neural_retry=False)
            return ClipEmbedding(tuple(values), True, metrics)
