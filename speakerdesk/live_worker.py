"""Resident, incremental NVIDIA diarization plus phrase-window Cohere ASR.

JSONL stdin is local 16kHz mono float32 PCM; stdout contains real model results.
No reference transcript, external service, or invented word alignment is used.
"""
import base64
import contextlib
import json
import resource
import sys
import time
import uuid
from pathlib import Path


def emit(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def run(config):
    from inference_worker import check_memory
    check_memory()
    import numpy as np
    import mlx.core as mx
    from mlx_audio.vad import load
    from mlx_speech.generation.cohere_asr import CohereAsrModel
    from pipeline import speech_crops
    from language_detection import SpeechTranscriber

    mx.set_memory_limit(5 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    started = time.perf_counter()
    # Redirect library diagnostics away from the framed result stream.
    with contextlib.redirect_stdout(sys.stderr):
        diar = load(Path(config['diar_path']), strict=True)
        diar.set_streaming_config('low')
        state = diar.init_streaming_state()
        asr = CohereAsrModel.from_path(Path(config['cohere_path']))
        transcriber = SpeechTranscriber(asr, config['language'], config.get('lid_path'))
    emit({'type': 'ready', 'load_seconds': time.perf_counter()-started,
          'diarization_preset': 'low', 'asr': 'phrase_windows',
          'peak_mlx_bytes':mx.get_peak_memory(),
          'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    audio = np.empty(0, dtype=np.float32)
    base = 0
    received = 0
    turns = []
    committed = 0.0

    def commit(final=False):
        nonlocal audio, base, turns, committed
        available = min(received / 16000, state.frames_processed * .01)
        regions = speech_crops(turns, max_seconds=6)
        for region in regions:
            start = max(committed, region['start'])
            end = min(region['end'], available)
            if end <= start:
                continue
            # Wait for a settled end, or bound the phrase to six seconds.
            if not final and end > available-.35 and end-start < 5.99:
                break
            if end-start < .15:
                committed = end
                continue
            check_memory()
            begin_sample = max(0, round(start*16000)-base)
            end_sample = min(len(audio), round(end*16000)-base)
            if end_sample <= begin_sample:
                continue
            tick = time.perf_counter()
            names = region['speakers']
            speaker = names[0] if len(names)==1 else 'overlap_'+'_'.join(s.split('_')[-1] for s in names)
            with contextlib.redirect_stdout(sys.stderr):
                passages = transcriber.transcribe(audio[begin_sample:end_sample], 16000, tuple(names))
            for passage in passages:
                emit({'type':'segment', 'segment':{**passage,'id':uuid.uuid4().hex,
                    'start':start+passage['start'], 'end':min(end,start+passage['end']), 'speaker':speaker,
                    'speaker_candidates':names,'voice_eligible':len(names)==1,'timing':'diarized_phrase',
                    'confidence':None, 'review':passage['review'] or len(names)>1 or end-start>=5.99},
                    'speakers':{speaker:('Speaker '+str(int(names[0].split('_')[-1])+1)
                        if len(names)==1 else 'Overlapping speakers')},
                    'inference_seconds':time.perf_counter()-tick,
                    'audio_received_seconds':received/16000})
            committed = end
            mx.clear_cache()
        # Keep only uncommitted waveform and bounded turn history.
        retain = max(committed, available-10)
        # Do not discard an uncommitted continuous speech region.
        pending = [t['start'] for t in turns if t['end'] > committed]
        if pending:
            retain = min(retain, max(committed, min(pending)))
        drop = max(0, min(len(audio), round(retain*16000)-base))
        audio = audio[drop:].copy(); base += drop
        turns = [dict(t, start=max(t['start'], committed)) for t in turns if t['end'] > committed]

    for line in sys.stdin:
        message = json.loads(line)
        final = message['type']=='stop'
        if not final and message['type']!='audio':
            raise ValueError('Unexpected live worker command.')
        chunk = np.empty(0, dtype=np.float32) if final else np.frombuffer(
            base64.b64decode(message['pcm'], validate=True), dtype='<f4').copy()
        if len(chunk)>16000 or not np.isfinite(chunk).all():
            raise ValueError('Invalid live audio packet.')
        audio = np.concatenate((audio, chunk)); received += len(chunk)
        tick = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            output, state = diar.feed(chunk, state, sample_rate=16000, final=final,
                threshold=.5, min_duration=0, merge_gap=0)
        for segment in output.segments:
            speaker = f'speaker_{segment.speaker}'
            # Feed boundaries are transport boundaries, not new utterances.
            previous = next((t for t in reversed(turns) if t['speaker']==speaker), None)
            if previous and segment.start-previous['end'] <= .20:
                previous['end'] = max(previous['end'], segment.end)
            else:
                turns.append({'start':segment.start,'end':segment.end,'speaker':speaker})
        diar_seconds=time.perf_counter()-tick
        commit(final)
        emit({'type':'progress','processed_seconds':state.frames_processed*.01,
              'received_seconds':received/16000,'diarization_seconds':diar_seconds,
              'peak_mlx_bytes':mx.get_peak_memory(),'buffer_samples':len(audio),
              'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if final:
            emit({'type':'finished','duration':received/16000,'peak_mlx_bytes':mx.get_peak_memory()})
            return


if __name__=='__main__':
    try:
        run(json.loads(sys.argv[1]))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({'type':'error','error':str(error)})
        sys.exit(1)
