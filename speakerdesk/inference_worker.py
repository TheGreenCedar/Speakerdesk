"""Offline model workers. Accept audio only; never read reference transcripts."""
import importlib.metadata
import json
import resource
import re
import subprocess
import sys
import time
from pathlib import Path


def utterance_tail_chunks(chunks, utterances, *, rate=16000, max_seconds=18):
    """Keep an admitted continuation with its speech object, not a gap decode.

    Only a clean single-speaker crop followed by an unassigned gap qualifies.
    A speech region must cross that boundary and its utterance must finish in
    the gap. Real switches/overlap and the existing decode size limit remain.
    The activity ledger is retained; context does not establish word ownership.
    """
    import copy
    planned = copy.deepcopy(chunks)
    index = 0
    while index+1 < len(planned):
        current, following = planned[index:index+2]
        boundary = round(current['end']*rate)
        if (len(current['speakers']) != 1 or current.get('activity_regions')
                or following['speakers'] or abs(current['end']-following['start']) > 1e-6):
            index += 1;continue
        utterance = next((row for row in utterances
            if round(current['start']*rate) <= row['start_sample'] < boundary < row['end_sample']
            and row['end_sample'] <= round(following['end']*rate)
            and row['end_sample']-round(current.get('audio_start',current['start'])*rate) <= max_seconds*rate
            and any(part['start_sample'] < boundary < part['end_sample'] for part in row['speech_regions'])), None)
        if utterance is None:
            index += 1;continue
        end = utterance['end_sample']/rate
        current['activity_regions'] = [
            {'start':current['start'], 'end':current['end'], 'speakers':list(current['speakers'])},
            {'start':following['start'], 'end':end, 'speakers':[]}]
        current['decode_context'] = {'policy':'speech_utterance_tail_context_v1',
            'start_sample':utterance['start_sample'], 'end_sample':utterance['end_sample'],
            'boundary_sample':boundary}
        current['end'] = current['audio_end'] = end
        current.pop('audio', None)
        if end < following['end']:
            following['start'] = following['audio_start'] = end
            following.pop('audio', None)
        else:
            planned.pop(index+1)
        index += 1
    return planned


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
        chunks=utterance_tail_chunks(request['chunks'],utterances)
        model = CohereAsrModel.from_path(model_path)
        transcriber = SpeechTranscriber(model, request['language'], request.get('lid_path'),
            speech_evidence=speech_session.evidence)
        loaded = time.perf_counter()
        regions = []
        aligner = None
        alignment_error = None
        for chunk in chunks:
            check_memory()
            if 'audio' in chunk:
                audio, rate = sf.read(chunk['audio'], dtype='float32')
            else:
                # Context is unchanged original PCM, read in a bounded crop.
                first=round(chunk['audio_start']*16000);last=round(chunk['audio_end']*16000)
                audio, rate = sf.read(request['audio'],start=first,frames=last-first,dtype='float32')
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
                    if (chunk.get('decode_context') and request.get('alignment_path')
                            and piece.get('language') == 'en' and piece.get('text')):
                        if aligner is None and alignment_error is None:
                            try:
                                from coarse_alignment import CoarseAlignment
                                aligner = CoarseAlignment(request['alignment_path'])
                            except Exception as exc:
                                alignment_error = f'{type(exc).__name__}: {exc}'
                        if alignment_error is not None:
                            piece['reading_alignment_error'] = alignment_error
                        else:
                            attach_piece_alignment(piece, audio[begin:end], rate,
                                origin+begin, aligner)
                    pieces.append({**piece,'start':piece['start']+begin/rate,
                                   'end':piece['end']+begin/rate})
            regions.append(pieces)
            print(f'Transcribed region {len(regions)}/{len(chunks)}', flush=True)
            mx.clear_cache()
        output = {'regions': regions}
        if chunks != request['chunks']:
            output['chunks']=[{key:value for key,value in chunk.items() if key!='audio'} for chunk in chunks]
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


def attach_piece_alignment(piece, audio, rate, origin, aligner):
    """Bind supplied English words to their exact original physical PCM slice.

    This optional reading evidence never replaces Cohere text. The consumer
    validates the provider identity, text, sample anchor and PCM hash again.
    Provider failures leave words unassigned; no alternate neural provider runs.
    """
    import hashlib
    import numpy as np
    if rate != 16000 or piece.get('language') != 'en' or not piece.get('text'):
        return
    a, b = round(piece['start']*rate), round(piece['end']*rate)
    if not 0 <= a < b <= len(audio):
        raise ValueError('Reading evidence is outside the original decode PCM.')
    pcm = np.asarray(audio[a:b], dtype=np.float32)
    digest = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
    try:
        result = aligner.align(pcm, piece['text'], start_sample=origin+a, language='en')
    except Exception as exc:
        piece['reading_alignment_error'] = f'{type(exc).__name__}: {exc}'
        return
    if isinstance(result, dict):
        piece.update(reading_alignment=result, audio_float32_sha256=digest)


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
