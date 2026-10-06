"""Synthetic JSONL peers only: no capture APIs, model imports or inference."""
import array
import base64
import json
import sys
import time
from pathlib import Path

role, folder, scenario = sys.argv[1:4]
language = sys.argv[4] if len(sys.argv)>4 else 'en'
generation = 0
folder = Path(folder)


def emit(message):
    print(json.dumps(message), flush=True)


def record(message):
    with (folder / f'{role}-commands.jsonl').open('a') as log:
        log.write(json.dumps(message) + '\n')


def audio(source, seconds, value, count):
    samples = array.array('f', [value] * count)
    if sys.byteorder != 'little':
        samples.byteswap()
    emit({'type': 'audio', 'source': source, 'time': seconds,
          'pcm': base64.b64encode(samples.tobytes()).decode()})


if role == 'worker':
    (folder / 'worker-started').touch()
    if scenario == 'startup':
        while not (folder / 'worker-release').exists():
            time.sleep(.01)
    emit({'type': 'ready'})
    if scenario == 'backlog':
        while not (folder / 'worker-release').exists():
            time.sleep(.01)
    received = 0
    for line in sys.stdin:
        message = json.loads(line)
        record({key:value for key,value in message.items() if key!='pcm'})
        if message['type'] == 'audio':
            start=received
            received += len(base64.b64decode(message['pcm'])) // 4
            if scenario in ('language_inflight','language_wrong_epoch','language_browser'):
                if start==0 and scenario!='language_browser':
                    (folder/'worker-inflight').touch()
                    while not (folder/'worker-release').exists():time.sleep(.01)
                emit({'type':'segment','speakers':{'speaker_0':'Speaker 1'},'segment':{
                    'id':f'synthetic-{start}','start':start/16000,'end':received/16000,
                    'speaker':'speaker_0','text':f'Synthetic {language} words.',
                    'language':'en' if scenario=='language_wrong_epoch' and generation else language,
                    'language_mode':language,'language_generation':generation}})
            emit({'type': 'progress', 'processed_seconds': received / 16000})
            if scenario in ('worker_error', 'worker_error_stalled'):
                emit({'type': 'error', 'error': 'Synthetic worker failure.'})
                break
        elif message['type'] == 'language':
            if scenario=='language_browser':
                (folder/'language-inflight').touch()
                while not (folder/'language-release').exists():time.sleep(.01)
            language=message['language'];generation=message['generation']
            emit({'type':'language_registered','language':language,'generation':generation,
                  'start_sample':message['start_sample']})
        elif message['type'] == 'stop':
            if received and scenario not in ('language_inflight','language_wrong_epoch','language_browser'):
                emit({'type': 'segment', 'speakers': {'speaker_0': 'Speaker 1'},
                      'segment': {'id': 'synthetic', 'start': 0, 'end': received / 16000,
                                  'speaker': 'speaker_0', 'text': 'Synthetic transport result.',
                                  'language':language,'language_mode':language,'language_generation':generation}})
            emit({'type': 'finished'})
            break
else:
    first = json.loads(sys.stdin.readline())
    record(first)
    sources = [source for source in ('microphone', 'system') if first[source]]
    emit({'type': 'recording'})
    for source in sources:
        audio(source, 0, 1.2 if scenario == 'source_clip' else .2 if source == 'microphone' else .3, 1600)
    emit({'type': 'clock', 'time': .1})
    duration = .1
    if scenario in ('backlog', 'worker_error', 'worker_error_stalled', 'capture_error', 'capture_error_after_stop'):
        # Keep the stream timeline valid while filling the real transport pipe.
        for n in range(136 if scenario == 'backlog' else 4):
            for source in sources:
                audio(source, duration, .2 if source == 'microphone' else .3, 4000)
            duration += .25
            emit({'type': 'clock', 'time': duration})
        (folder / 'capture-burst-finished').touch()
        if scenario == 'capture_error_after_stop':
            emit({'type': 'stopped', 'time': duration})
        if scenario in ('capture_error', 'capture_error_after_stop'):
            emit({'type': 'error', 'error': 'Synthetic capture failure.'})
    for line in sys.stdin:
        message = json.loads(line)
        record(message)
        if message['type'] == 'pause':
            emit({'type': 'paused', 'time': duration})
        elif message['type'] == 'resume':
            emit({'type': 'recording', 'time': duration})
            for source in sources:
                audio(source, duration, .4 if source == 'microphone' else .1, 4000)
            duration += .25
            emit({'type': 'clock', 'time': duration})
        elif message['type'] == 'stop':
            if scenario == 'worker_error_stalled':
                (folder / 'capture-ignored-stop').touch()
                continue
            if scenario == 'delayed_stop':
                # A start acknowledgement can race a user stop request.
                emit({'type': 'recording', 'time': duration})
                duration = .75
                emit({'type': 'clock', 'time': duration})
                while not (folder / 'capture-release').exists():
                    time.sleep(.01)
            emit({'type': 'stopped', 'time': duration})
            break
