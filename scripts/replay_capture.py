#!/usr/bin/env python3
"""Synthetic PCM transport through the production capture protocol; no AI/capture."""
import array
import base64
import json
import os
from pathlib import Path
import sys
import threading
import time
import wave

def main():
    root=Path(os.environ['SPEAKERDESK_REPLAY_DIR'])
    with wave.open(str(root/'replay.wav'),'rb') as stream:
        if (stream.getnchannels(),stream.getsampwidth(),stream.getframerate())!=(1,2,16000):
            raise ValueError('Replay requires mono PCM16 at 16kHz.')
        pcm=array.array('h',stream.readframes(stream.getnframes()))
    if sys.byteorder!='little':pcm.byteswap()
    config=json.loads((root/'replay.json').read_text());hold=config.get('hold_sample')
    boundaries={len(pcm):'end'}
    if hold is not None and 0<hold<len(pcm):boundaries[hold]='edit'
    condition=threading.Condition();output=threading.Lock()
    state={'cursor':0,'held':False,'paused':False,'stopped':False}
    def emit(value):
        with output:print(json.dumps(value),flush=True)
    def produce():
        try:
            while True:
                with condition:
                    while (state['held'] or state['paused']) and not state['stopped']:condition.wait(.1)
                    if state['stopped']:return
                    start=state['cursor'];end=min(start+4000,*[x for x in boundaries if x>start])
                    if condition.wait((end-start)/16000):continue
                    if state['paused'] or state['stopped']:continue
                    packet=array.array('f',(sample/32768 for sample in pcm[start:end]))
                    if sys.byteorder!='little':packet.byteswap()
                    emit({'type':'audio','source':'microphone','time':start/16000,
                          'pcm':base64.b64encode(packet.tobytes()).decode()})
                    state['cursor']=end;emit({'type':'clock','time':end/16000})
                    if end in boundaries:
                        state['held']=True;(root/('stage-'+boundaries[end])).touch()
        except Exception as error:emit({'type':'error','error':str(error)})
    first=json.loads(sys.stdin.readline())
    if first!={'type':'start','microphone':True,'system':False}:raise ValueError('Only synthetic microphone transport is supported.')
    emit({'type':'recording'})
    thread=threading.Thread(target=produce,daemon=True);thread.start()
    try:
        for line in sys.stdin:
            command=json.loads(line)
            with condition:
                kind=command['type'];now=state['cursor']/16000
                if kind=='pause':state['paused']=True;emit({'type':'paused','time':now})
                elif kind=='resume':state.update(paused=False,held=False);emit({'type':'recording','time':now})
                elif kind=='stop':state['stopped']=True;emit({'type':'stopped','time':now})
                else:raise ValueError('Unexpected transport control.')
                condition.notify_all()
                if kind=='stop':break
    finally:
        with condition:state['stopped']=True;condition.notify_all()
        thread.join(timeout=2)

if __name__=='__main__':main()
