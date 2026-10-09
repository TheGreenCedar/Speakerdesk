"""One bounded, offline resident GPU voice owner; no network or automatic retry."""
import json
from pathlib import Path
import resource
import sys
import time

from source_voice import clip_from_payload
from voice_gpu_runtime import release_policy, require_runtime, verify_artifact
from voice_gpu_process import MAX_FRAME_BYTES, MAX_MLX_BYTES


def emit(message):
    encoded = json.dumps(message, allow_nan=False, separators=(',', ':'))
    if len(encoded.encode('utf-8')) > MAX_FRAME_BYTES:
        raise ValueError('Voice worker response is too large.')
    print(encoded, flush=True)


def run(init):
    model, _ = release_policy(init['config_path'])
    require_runtime()
    verify_artifact(init['model_dir'])
    if init.get('maximum_MLX_bytes') != MAX_MLX_BYTES:
        raise ValueError('Voice GPU memory admission differs from the qualified bound.')
    audio_root = Path(init['audio_root']).resolve()
    deadline = time.monotonic()+90

    def admission():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == 'darwin' else 1024
        if time.monotonic() >= deadline or peak > 2*1024**3:
            raise ValueError('Voice inference exceeded its admitted resources.')
        return {'maximum_MLX_bytes':MAX_MLX_BYTES}

    from thread_bound_backend import ThreadBoundVoiceBackend
    from voice_mlx_backend import ReDimNet2MLX
    backend = ThreadBoundVoiceBackend(lambda:ReDimNet2MLX(init['model_dir'], require_admission=admission,
        on_entered=lambda _clip:None, on_declaration=lambda *_args:None), model, require_admission=admission)
    try:
        emit({'type':'ready', 'model':model.payload()})
        last_id = 0
        while line := sys.stdin.readline(MAX_FRAME_BYTES+1):
            if len(line.encode('utf-8')) > MAX_FRAME_BYTES or not line.endswith('\n'):
                raise ValueError('Voice worker request is too large.')
            request = json.loads(line)
            if request.get('op') == 'shutdown':
                return
            if request.get('op') != 'embed' or type(request.get('id')) is not int or request['id'] != last_id+1:
                raise ValueError('Invalid voice request sequence.')
            clip = clip_from_payload(request['clip'])
            from voice_source_audio import validate_voice_audio,clip_pcm_digest
            audio = validate_voice_audio(audio_root,request['audio'],clip)
            pcm_digest=clip_pcm_digest(audio,clip) if getattr(clip,'capture_source',None) is not None else None
            if request.get('source_pcm_sha256')!=pcm_digest:
                raise ValueError('Voice source input differs from its retained crop.')
            # Use the existing Mac capacity check, without reserving an arbitrary GB floor.
            from inference_worker import check_memory
            check_memory()
            deadline = time.monotonic()+90
            result = backend.embed(audio, clip)
            if pcm_digest is not None and clip_pcm_digest(audio,clip)!=pcm_digest:
                raise ValueError('Voice source PCM changed during inference.')
            emit({'type':'embedding', 'id':request['id'], 'model':model.payload(),
                  'vector':result.vector, 'clean':result.clean, 'metrics':result.metrics,
                  'owner_thread_id':backend.owner_thread_id,
                  **({'source_pcm_sha256':pcm_digest} if pcm_digest is not None else {})})
            last_id = request['id']
    finally:
        backend.close()


if __name__ == '__main__':
    try:run(json.loads(sys.argv[1]))
    except Exception:
        # Keep private paths and embeddings out of error frames and UI diagnostics.
        emit({'error':'Voice GPU worker could not complete this request.'})
        sys.exit(1)
