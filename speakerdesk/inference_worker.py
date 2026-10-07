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
        from language_detection import SpeechTranscriber
        from speech_admission import SileroModel, FrameArchive
        from utterances import UtteranceBook
        import uuid
        import soundfile as sf

        speech=SileroModel(request['speech_path'])
        speech_session=speech.session(archive=FrameArchive(
            Path(request['audio']).parent/f'speech-import-{uuid.uuid4().hex}.jsonl'))
        with sf.SoundFile(request['audio']) as recording:
            if (recording.samplerate != 16000 or recording.channels != 1
                    or not 0 < recording.frames <= 7200*16000):
                raise ValueError('Invalid speech-admission recording.')
            while len(pcm:=recording.read(16000,dtype='float32')):
                check_memory()
                speech_session.feed(pcm,speech_session.received)
        import numpy as np
        speech_session.feed(np.empty(0,dtype=np.float32),speech_session.received,final=True)
        # A positive frame admits its canonical speech object, not an entire
        # diarization/gap crop. Use the same settling/context policy as live.
        speech_book=UtteranceBook('import-'+uuid.uuid4().hex)
        for begin in range(0,speech_session.received,60*16000):
            speech_book.observe(speech_session.evidence.admission(
                begin,min(begin+60*16000,speech_session.received)))
        utterances=speech_book.finish()
        model = CohereAsrModel.from_path(model_path)
        transcriber = SpeechTranscriber(model, request['language'], request.get('lid_path'),
            speech_evidence=speech_session.evidence)
        loaded = time.perf_counter()
        regions = []
        for chunk in request['chunks']:
            check_memory()
            audio, rate = sf.read(chunk['audio'], dtype='float32')
            origin=round(chunk.get('audio_start',chunk.get('start',0))*rate)
            edges={0,len(audio)}
            for utterance in utterances:
                begin=max(0,utterance['start_sample']-origin)
                end=min(len(audio),utterance['end_sample']-origin)
                if begin<end:edges.update((begin,end))
            edges=sorted(edges);pieces=[]
            for begin,end in zip(edges,edges[1:]):
                # Negative gaps still return reviewed original-audio coverage;
                # SpeechTranscriber independently checks every resulting slice.
                for piece in transcriber.transcribe(audio[begin:end],rate,tuple(chunk['speakers']),
                        start_sample=origin+begin,max_asr_seconds=18):
                    pieces.append({**piece,'start':piece['start']+begin/rate,
                                   'end':piece['end']+begin/rate})
            regions.append(pieces)
            print(f'Transcribed region {len(regions)}/{len(request["chunks"])}', flush=True)
            mx.clear_cache()
        output = {'regions': regions}
    else:
        raise ValueError('Unknown task.')
    mx.synchronize()
    output['metrics'] = {'load_seconds': loaded-start,
                         'inference_seconds': time.perf_counter()-loaded,
                         'worker_seconds': time.perf_counter()-start,
                         'peak_mlx_bytes': mx.get_peak_memory(),
                         'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    packages = ['mlx', 'mlx-audio' if task == 'diarize' else 'mlx-speech']
    if task == 'transcribe' and request['language'] == 'auto':packages.append('mlx-audio')
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
