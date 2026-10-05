"""Local meeting lifecycle, synchronized source mixing and bounded worker transport."""
import base64
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from flask import abort, jsonify, request
from pipeline import model_config, preflight
from transcript import LANGUAGES

RATE = 16000
BLOCK = 4000
LIVE = ('starting', 'recording', 'paused', 'finishing')


class SourceMixer:
    """Timestamp alignment: watermark protects late packets, memory is bounded.

    Independent tracks are summed with clipping. This is not acoustic echo cancellation.
    """
    def __init__(self, sources, sink):
        import numpy as np
        self.np = np
        self.sources = sources
        self.sink = sink
        self.tiles = {}
        self.cursor = 0
        self.late_samples = 0
        self.levels = {source: 0.0 for source in sources}
        self.last_packet = {source:0. for source in sources}

    def add(self, source, seconds, audio):
        if source not in self.sources or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid capture source or timestamp.')
        samples = self.np.frombuffer(audio, dtype='<f4')
        if len(samples) > RATE or not self.np.isfinite(samples).all():
            raise ValueError('Invalid capture audio packet.')
        self.levels[source] = min(1., float(self.np.sqrt(self.np.mean(samples**2)))) if len(samples) else 0.
        self.last_packet[source] = seconds+len(samples)/RATE
        position = round(seconds*RATE)
        if position > self.cursor + RATE*5:
            raise RuntimeError('Audio capture timing jumped. Recording stopped to preserve its timeline.')
        offset = 0
        if position < self.cursor:
            offset = min(len(samples), self.cursor-position)
            self.late_samples += offset; position += offset
        while offset < len(samples):
            tile_index, within = divmod(position, BLOCK)
            size = min(BLOCK-within, len(samples)-offset)
            tile = self.tiles.setdefault(tile_index, {})
            buffer = tile.setdefault(source, self.np.zeros(BLOCK, dtype=self.np.float32))
            buffer[within:within+size] = samples[offset:offset+size]
            offset += size; position += size

    def flush(self, seconds, final=False):
        for source in self.sources:
            if seconds-self.last_packet[source] > .5:self.levels[source]=0.
        target = round(max(0, seconds if final else seconds-.5)*RATE)
        if not final:
            target = target//BLOCK*BLOCK
        if target > self.cursor+RATE*5:
            raise RuntimeError('Capture clock stopped updating. Audio has been saved.')
        while self.cursor < target:
            tile_index, within = divmod(self.cursor, BLOCK)
            count = min(BLOCK-within, target-self.cursor)
            tile = self.tiles.get(tile_index, {})
            tracks = {s:tile.get(s,self.np.zeros(BLOCK,dtype=self.np.float32))[within:within+count] for s in self.sources}
            if within+count == BLOCK:
                self.tiles.pop(tile_index,None)
            mixed = self.np.clip(sum(tracks.values()),-1,1).astype('<f4')
            self.sink(mixed, tracks)
            self.cursor += count


class MeetingManager:
    def __init__(self, get, put, patch, folder):
        self.get, self.put, self.patch, self.folder = get, put, patch, folder
        self.lock = threading.RLock()
        self.jid = None
        self.capture = self.worker = None
        self.packets = queue.Queue(maxsize=120)  # At most30s audio; recording itself remains independent.
        self.stopped = threading.Event()
        self.levels = {}; self.processed = 0.; self.duration = 0.
        self.error = None

    def helper_path(self):
        configured = os.getenv('SPEAKERDESK_CAPTURE_HELPER')
        return Path(configured) if configured else Path(__file__).resolve().parents[1]/'desktop/capture/speakerdesk-capture'

    def status(self):
        with self.lock:
            job=self.get(self.jid) if self.jid else None
            return {'id':self.jid,'status':job['status'] if job else 'idle','sources':job.get('sources',[]) if job else [],
                    'duration':self.duration,'processed_seconds':self.processed,
                    'pending_seconds':max(0,self.duration-self.processed),'levels':self.levels,'error':self.error,
                    'capture_available':self.helper_path().is_file()}

    def start(self, name, language, sources):
        with self.lock:
            if self.capture or self.worker or (self.jid and self.get(self.jid)['status'] in LIVE):
                abort(409,description='A meeting is already active.')
            if language not in LANGUAGES or not isinstance(name,str) or not name.strip() or len(name)>160:
                raise ValueError('Enter a meeting title and supported language.')
            if not sources or any(s not in ('microphone','system') for s in sources):
                raise ValueError('Choose microphone audio, Mac audio, or both.')
            if not self.helper_path().is_file():
                abort(409,description='Live capture is not included in this build yet. Recording imports remain available.')
            issues = preflight(model_config())
            if issues: abort(409,description='Open Settings to finish local model setup.')
            if shutil.disk_usage(self.folder('')).free < 512*1024**2:
                abort(409,description='Free at least512MB before recording a meeting.')
            self.jid = uuid.uuid4().hex
            self.folder(self.jid).mkdir(mode=0o700)
            job = {'id':self.jid,'name':name.strip(),'language':language,'kind':'meeting',
                   'source_file':'audio.wav','status':'starting','message':'Starting meeting…',
                   'created':time.time(),'duration':0.,'revision':0,'sources':sources,
                   'document':{'schema_version':1,'speakers':{'speaker_0':'Speaker 1'},'segments':[],
                       'provenance':{'kind':'pending_inference','mode':'live_phrase_windows',
                           'models':['nvidia/Nemotron-3-Diarization','CohereLabs/cohere-transcribe-03-2026'],
                           'timing':'NVIDIA speech-region boundaries; Cohere phrase text. No word timestamps.'},
                       'warnings':['Phrase boundaries may cut words. Overlapping speech needs review.']}}
            self.put(job)
            self.duration = self.processed = 0.; self.levels = {}; self.error = None
            self.stopped = threading.Event(); self.packets = queue.Queue(maxsize=120)
            thread = threading.Thread(target=self._run,args=(self.jid,language,sources),daemon=True)
            thread.start()
            return job

    def control(self, jid, action):
        with self.lock:
            if jid!=self.jid: abort(409,description='This meeting is no longer active.')
            status = self.get(jid)['status']
            required = {'pause':'recording','resume':'paused','stop':None}[action]
            if status not in LIVE or (required and status!=required):
                abort(409,description='The meeting is not ready for this action.')
            if not self.capture:
                if action=='stop': self.stopped.set(); self.patch(jid,status='finishing',message='Stopping…'); return
                abort(409,description='Wait for recording to start.')
            self._send(self.capture,{'type':action})
            if action=='stop':
                self.patch(jid,status='finishing',message='Finishing transcript…')
                capture=self.capture
                def stop_stalled_capture():
                    if capture.poll() is None:capture.terminate()
                timer=threading.Timer(15,stop_stalled_capture);timer.daemon=True;timer.start()

    @staticmethod
    def _send(process, message):
        process.stdin.write(json.dumps(message)+'\n'); process.stdin.flush()

    def _run(self, jid, language, sources):
        import numpy as np
        handles = []
        threads = []
        worker_finished = threading.Event()
        failure = []
        packets = self.packets
        stop_event = self.stopped
        capture_finished = False
        wav = None
        tracks = {}
        dest = self.folder(jid)
        def fail(message):
            failure.append(message)
            self.stopped.set()
            if self.capture and self.capture.poll() is None:
                try:self._send(self.capture,{'type':'stop'})
                except (OSError,ValueError):pass
        def consume_results():
            finished = False
            try:
                for line in self.worker.stdout:
                    result = json.loads(line)
                    with self.lock:
                        if result['type']=='segment':
                            job=self.get(jid);document=job['document']
                            document['speakers'].update(result['speakers']);document['segments'].append(result['segment'])
                            document['provenance']['kind']='local_inference'
                            job.update(revision=job['revision']+1);self.put(job)
                        elif result['type']=='progress':self.processed=result['processed_seconds']
                        elif result['type']=='error':fail(result['error']);break
                        elif result['type']=='finished':finished=True;break
            except Exception:fail('Live transcription stopped unexpectedly. Captured audio has been saved.')
            finally:
                if not finished and not failure:fail('Live inference exited before completing. Captured audio has been saved.')
                worker_finished.set()
        def send_audio():
            try:
                while True:
                    try:message=packets.get(timeout=1)
                    except queue.Empty:
                        if stop_event.is_set():return
                        continue
                    self._send(self.worker,message)
                    if message['type']=='stop':return
            except (OSError,ValueError):fail('Live transcription stopped. Captured audio has been saved.')
        try:
            cfg=model_config();cfg['language']=language
            python=os.getenv('SPEAKERDESK_LIVE_PYTHON',str(Path(__file__).resolve().parents[1]/'.venv-package/bin/python'))
            command=([sys.executable,'--live-worker',json.dumps(cfg)] if getattr(sys,'frozen',False)
                     else [python,str(Path(__file__).with_name('live_worker.py')),json.dumps(cfg)])
            log=(dest/'live-worker.log').open('w');handles.append(log)
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            self.worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
            # Startup has a bounded wait so a failed worker cannot strand the meeting.
            ready_queue=queue.Queue()
            threading.Thread(target=lambda:ready_queue.put(self.worker.stdout.readline()),daemon=True).start()
            try:ready=json.loads(ready_queue.get(timeout=60))
            except (queue.Empty,ValueError):raise RuntimeError('Local models could not start. Audio capture did not begin.')
            if ready.get('type')!='ready':raise RuntimeError(ready.get('error','Local models could not start.'))
            if self.stopped.is_set():return
            for target in (consume_results,send_audio):
                thread=threading.Thread(target=target,daemon=True);thread.start();threads.append(thread)
            audio=(dest/'audio.wav').open('w+b');handles.append(audio)
            wav=wave.open(audio,'wb');wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(RATE)
            tracks={}
            for source in sources:
                f=(dest/f'{source}.wav').open('w+b');handles.append(f)
                track=wave.open(f,'wb');track.setnchannels(1);track.setsampwidth(2);track.setframerate(RATE);tracks[source]=track
            def sink(mixed, separate):
                wav.writeframes((mixed*32767).astype('<i2').tobytes())
                for source,track in tracks.items():track.writeframes((separate[source]*32767).astype('<i2').tobytes())
                try:self.packets.put_nowait({'type':'audio','pcm':base64.b64encode(mixed.tobytes()).decode()})
                except queue.Full:raise RuntimeError('Transcription fell30seconds behind. Recording stopped; captured audio has been saved.')
                self.duration=wav.getnframes()/RATE
            mixer=SourceMixer(sources,sink)
            capture_log=(dest/'capture.log').open('w');handles.append(capture_log)
            self.capture=subprocess.Popen([str(self.helper_path())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=capture_log,text=True,bufsize=1)
            self._send(self.capture,{'type':'start','microphone':'microphone' in sources,'system':'system' in sources})
            for line in self.capture.stdout:
                result=json.loads(line);kind=result['type']
                if kind=='audio':
                    mixer.add(result['source'],result['time'],base64.b64decode(result['pcm'],validate=True))
                elif kind=='clock':
                    mixer.flush(result['time']);self.levels=mixer.levels.copy()
                    self.patch(jid,duration=self.duration)
                    for handle in handles:handle.flush()
                    if shutil.disk_usage(dest).free < 256*1024**2:raise RuntimeError('Recording stopped because disk space is low. Audio has been saved.')
                elif kind in ('recording','paused'):
                    self.patch(jid,status=kind,message='Recording' if kind=='recording' else 'Paused')
                    if kind=='paused':mixer.flush(result['time'],final=True)
                elif kind=='stopped':
                    mixer.flush(result['time'],final=True);capture_finished=True;break
                elif kind=='error':raise RuntimeError(result['error'])
            if not capture_finished or (self.capture.poll() is not None and self.capture.returncode):
                raise RuntimeError('Audio capture stopped unexpectedly. Captured audio has been saved.')
            self.patch(jid,status='finishing',message='Finishing transcript…',duration=self.duration)
            wav.close()
            for track in tracks.values():track.close()
            self.packets.put({'type':'stop'},timeout=10)
            if not worker_finished.wait(timeout=60):raise RuntimeError('Transcription took too long to finish. Captured audio has been saved.')
            if failure:raise RuntimeError(failure[0])
            warnings=self.get(jid)['document']['warnings']
            if mixer.late_samples:warnings.append(f'{mixer.late_samples/RATE:.2f}s of late source samples were omitted from the mix. Review audio timing.')
            document=self.get(jid)['document'];document['warnings']=warnings
            self.patch(jid,status='ready',message='Meeting saved on this Mac.',document=document,duration=self.duration)
        except Exception as exc:
            self.error=str(exc)
            self.patch(jid,status='failed',message=self.error,duration=self.duration)
        finally:
            stop_event.set()
            for recording in [wav, *tracks.values()]:
                if recording:
                    try:recording.close()
                    except (OSError,ValueError):pass
            for child in (self.capture,self.worker):
                if child and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
            for thread in threads:
                thread.join(timeout=3)
            for handle in handles:
                handle.close()
            with self.lock:
                self.capture=self.worker=None
                if self.get(jid)['status'] in LIVE:self.patch(jid,status='failed',message='Meeting stopped before capture began.')
                self.jid=None

    def close(self):
        with self.lock:
            self.stopped.set()
            for process in (self.capture,self.worker):
                if process and process.poll() is None:process.terminate()


def register_meetings(app, get, put, patch, folder):
    manager=MeetingManager(get,put,patch,folder)
    app.extensions['speakerdesk']['meetings']=manager
    @app.get('/api/meeting')
    def meeting_status():return jsonify(manager.status())
    @app.post('/api/meetings')
    def start_meeting():
        body=request.get_json();sources=body.get('sources')
        if not isinstance(sources,list) or any(not isinstance(s,str) for s in sources) or len(sources)!=len(set(sources)):
            raise ValueError('Select valid meeting audio sources.')
        return jsonify(manager.start(body.get('name','Meeting'),body.get('language','en'),sources)),201
    @app.post('/api/meetings/<jid>/<action>')
    def control_meeting(jid,action):
        if action not in ('pause','resume','stop'):abort(404)
        manager.control(jid,action);return jsonify(get(jid)),202
