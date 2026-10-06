"""Run the production resident engine with the caller's CPU model substitutions."""
import contextlib
from pathlib import Path
import sys
import tempfile
import wave
import base64
import numpy as np
from live_refinement import Engine, Inbox, Models
from audio import pcm16_bytes


def run_messages(config,messages):
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'audio.wav';prepared=[];cursor=0;epoch=0;language=config['language']
        with wave.open(str(path),'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'))
            for message in messages:
                if message['type']=='audio':
                    pcm=np.frombuffer(base64.b64decode(message['pcm']),dtype='<f4')
                    wav.writeframes(pcm16_bytes(pcm))
                    prepared.append({'type':'audio','start_sample':cursor,'end_sample':cursor+len(pcm),'language_epoch':epoch,'language':language});cursor+=len(pcm)
                else:
                    prepared.append(message)
                    if message['type']=='language':epoch=message['generation'];language=message['language']
        config={**config,'audio_path':str(path),'speech_path':config.get('speech_path','cpu-peer')};events=[];inbox=Inbox()
        with contextlib.redirect_stdout(sys.stderr):models=Models(config)
        engine=Engine(config,models,events.append,inbox)
        for message in prepared:engine.handle(message)
        # Adapter presents the final production document to older contract tests.
        events.extend({'type':'segment','segment':row} for row in engine.document['segments'])
        engine.handle({'type':'shutdown'})
        return events
