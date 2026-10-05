"""Offline model workers. Accept audio only; never read reference transcripts."""
import importlib.metadata
import json
import resource
import re
import subprocess
import sys
import time
from pathlib import Path


def check_memory():
    status=subprocess.run(['memory_pressure','-Q'],capture_output=True,text=True,timeout=10)
    match=re.search(r'memory free percentage:\s*(\d+)%',status.stdout)
    if status.returncode or not match:
        raise RuntimeError('Unable to check Mac memory pressure before inference.')
    if int(match[1]) < 15:
        raise RuntimeError('Mac memory pressure is high. Close memory-heavy apps and retry.')


def run(task, request):
    check_memory()
    import mlx.core as mx

    mx.set_memory_limit(5 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    start = time.perf_counter()
    model_path = Path(request['model_path']).resolve()
    if not (model_path / 'model.safetensors').is_file():
        raise RuntimeError('Missing local model checkpoint.')
    if task == 'diarize':
        from mlx_audio.vad import load

        model = load(model_path, strict=True)
        loaded = time.perf_counter()
        result = model.generate(request['audio'])
        output = {'turns': [f'{s.start} {s.end} speaker_{s.speaker}' for s in result.segments]}
    elif task == 'transcribe':
        from mlx_speech.generation.cohere_asr import CohereAsrModel
        import soundfile as sf

        model = CohereAsrModel.from_path(model_path)
        loaded = time.perf_counter()
        texts = []
        for chunk in request['chunks']:
            check_memory()
            audio, rate = sf.read(chunk['audio'], dtype='float32')
            result = model.transcribe(audio, sample_rate=rate, language=request['language'],
                                      max_new_tokens=448)
            if len(result.tokens) >= 448:
                raise RuntimeError('Cohere reached token limit; shorten this speech region.')
            texts.append(result.text.strip())
            print(f'Transcribed region {len(texts)}/{len(request["chunks"])}', flush=True)
            mx.clear_cache()
        output = {'texts': texts}
    else:
        raise ValueError('Unknown task.')
    mx.synchronize()
    output['metrics'] = {'load_seconds': loaded-start,
                         'inference_seconds': time.perf_counter()-loaded,
                         'worker_seconds': time.perf_counter()-start,
                         'peak_mlx_bytes': mx.get_peak_memory(),
                         'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    packages = ['mlx', 'mlx-audio' if task == 'diarize' else 'mlx-speech']
    output['versions'] = {p: importlib.metadata.version(p) for p in packages}
    return output


if __name__ == '__main__':
    task, request_file, output_file = sys.argv[1:]
    try:
        result = run(task, json.loads(Path(request_file).read_text()))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {'error': f'{type(exc).__name__}: {exc}'}
    Path(output_file).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(1 if 'error' in result else 0)
