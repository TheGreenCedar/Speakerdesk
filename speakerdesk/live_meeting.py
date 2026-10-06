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
from audio import pcm16_bytes
from pipeline import model_config, preflight
from language_detection import LANGUAGE_CHOICES, LID_CHECKPOINT
from meeting_refinement import RefinementController
from live_language import validate_language_segment
from rolling_refinement import RollingPlan

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
        position = round(seconds*RATE)
        if position > self.cursor + RATE*5:
            raise RuntimeError('Audio capture timing jumped. Recording stopped to preserve its timeline.')
        self.levels[source] = min(1., float(self.np.sqrt(self.np.mean(samples**2)))) if len(samples) else 0.
        self.last_packet[source] = max(self.last_packet[source], seconds+len(samples)/RATE)
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
    def __init__(self, get, put, patch, folder, lock, inference_busy, recognizer=None, *, default_language=None):
        self.get, self.put, self.patch, self.folder = get, put, patch, folder
        self.lock = lock
        self.inference_busy = inference_busy
        self.recognizer = recognizer
        self.default_language=default_language or (lambda:'auto')
        self.jid = None
        self.capture = self.worker = None
        self.packets = queue.Queue(maxsize=120)  # At most30s audio; recording itself remains independent.
        self.stopped = threading.Event()
        self.levels = {}; self.processed = 0.; self.duration = 0.
        self.error = None
        self.thread = None
        self.stop_timer = None
        self.worker_controls = queue.Queue(maxsize=8)
        self.two_pass = False
        self.refining_saved = False
        self.active_language = 'auto'; self.active_epoch = 0
        self.refinement = RefinementController(self)

    def helper_path(self):
        configured = os.getenv('SPEAKERDESK_CAPTURE_HELPER')
        return Path(configured) if configured else Path(__file__).resolve().parents[1]/'desktop/capture/speakerdesk-capture'

    def status(self):
        with self.lock:
            job=self.get(self.jid) if self.jid else None
            return {'id':self.jid,'status':job['status'] if job else 'idle','sources':job.get('sources',[]) if job else [],
                    'duration':self.duration,'processed_seconds':self.processed,
                    'pending_seconds':max(0,self.duration-self.processed),'levels':self.levels,'error':self.error,
                    'capture_available':self.helper_path().is_file(),
                    'language':job.get('language') if job else None,
                    'language_revision':job.get('language_revision',0) if job else 0,
                    'language_acknowledged_revision':job.get('language_acknowledged_revision',0) if job else 0,
                    'language_from_sample':job.get('language_history',[{'start_sample':0}])[-1]['start_sample'] if job else 0,
                    'pause_flush':job.get('pause_flush') if job else None,
                    'refinement_status':job.get('refinement_status','waiting') if job else 'idle'}

    def queue_worker(self, message):
        try:self.worker_controls.put_nowait(message);return True
        except queue.Full:return False

    def change_language(self, jid, language, revision):
        if not isinstance(language,str) or language not in LANGUAGE_CHOICES or type(revision) is not int or revision<0:
            raise ValueError('Choose a supported live language and its current revision.')
        with self.lock:
            if jid!=self.jid:abort(409,description='This meeting is no longer active.')
            job=self.get(jid)
            if job['status'] not in ('recording','paused') or self.stopped.is_set() or not self.worker or self.worker.poll() is not None:
                abort(409,description='Language can change while recording or paused.')
            if revision!=job['language_revision']:abort(409,description='The live language changed in another window. Try again with its current setting.')
            if language==job['language']:
                self.put(job,default_language=language)
                return {**job,'default_language':language}
            issues=preflight(model_config(),language)
            if issues:abort(409,description=' '.join(issues))
            if self.packets.full():abort(429,description='Live transcription is catching up. Try changing language again shortly.')
            epoch={'epoch':revision+1,'generation':revision+1,'language':language,'start_sample':round(self.duration*RATE)}
            job.update(language=language,language_revision=revision+1,language_epoch=revision+1,revision=job['revision']+1)
            job['language_history'].append(epoch)
            job['document']['provenance'].update(language='per_window',language_history=job['language_history'])
            if language=='auto':job['document']['provenance']['language_detector']=LID_CHECKPOINT
            plan=RollingPlan(job['rolling_refinement']);paused=plan.state['cancelled'];plan.cancel()
            from meeting_refinement import remember_unresolved
            remember_unresolved(job,plan.state['completed_sample'],epoch['start_sample'],'language_changed')
            plan.state['completed_sample']=max(plan.state['completed_sample'],epoch['start_sample'])
            if not paused:plan.resume()
            job['rolling_refinement']=plan.snapshot();job.pop('rolling_inflight',None)
            self.put(job,default_language=language)  # Job and preference commit before either producer observes the boundary.
            self.active_language=language;self.active_epoch=revision+1
            self.packets.put_nowait({'type':'language','generation':revision+1,'language_epoch':revision+1,
                                    'language':language,'start_sample':epoch['start_sample']})
            return {**job,'default_language':language}

    def start(self, name, language, sources):
        with self.lock:
            if self.capture or self.worker or (self.jid and self.get(self.jid)['status'] in LIVE):
                abort(409,description='A meeting is already active.')
            if self.inference_busy():
                abort(409,description='Wait for queued import inference to finish before starting a meeting.')
            if language not in LANGUAGE_CHOICES or not isinstance(name,str) or not name.strip() or len(name)>160:
                raise ValueError('Enter a meeting title and supported language.')
            if not sources or any(s not in ('microphone','system') for s in sources):
                raise ValueError('Choose microphone audio, Mac audio, or both.')
            if not self.helper_path().is_file():
                abort(409,description='Live capture is not included in this build yet. Recording imports remain available.')
            issues = preflight(model_config(), language)
            if issues: abort(409,description='Open Settings to finish local model setup.')
            if shutil.disk_usage(self.folder('')).free < 512*1024**2:
                abort(409,description='Free at least512MB before recording a meeting.')
            jid = uuid.uuid4().hex
            dest = self.folder(jid);dest.mkdir(mode=0o700)
            job = {'id':jid,'name':name.strip(),'language':language,'kind':'meeting',
                   'source_file':'audio.wav','status':'starting','message':'Starting meeting…',
                   'created':time.time(),'duration':0.,'revision':0,'sources':sources,
                   'document':{'schema_version':1,'speakers':{'speaker_0':'Speaker 1'},'segments':[],
                       'provenance':{'kind':'pending_inference','mode':'live_phrase_windows',
                           'language':language,'language_detector':LID_CHECKPOINT if language=='auto' else None,
                           'models':['nvidia/Nemotron-3-Diarization','CohereLabs/cohere-transcribe-03-2026'],
                           'timing':'Retained audio regions with NVIDIA speaker labels and Cohere phrase text. No word timestamps.'},
                       'warnings':['Phrase boundaries may cut words. Overlapping speech needs review.'] +
                           (['Uncertain language uses recent established context or a marked supported-language guess when available. Choose the meeting language if Auto has no supported context. Original audio is preserved.'] if language=='auto' else [])}}
            self.refinement.initialize(job)
            try:self.put(job,default_language=language)
            except Exception:
                dest.rmdir()
                raise
            self.jid = jid
            self.duration = self.processed = 0.; self.levels = {}; self.error = None
            self.stopped = threading.Event(); self.packets = queue.Queue(maxsize=120)
            self.worker_controls=queue.Queue(maxsize=8);self.two_pass=False
            self.refining_saved=False
            self.active_language=language;self.active_epoch=0
            self.thread = threading.Thread(target=self._run,args=(self.jid,language,sources),daemon=True)
            self.thread.start()
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
            if action=='stop':
                self.stopped.set()
                self.patch(jid,status='finishing',message='Finishing transcript…')
                self._stop_capture()
            else:
                if action=='pause':
                    job=self.get(jid)
                    if (job.get('pause_flush') or {}).get('state')=='awaiting_capture':
                        # The helper has not acknowledged the first command yet.
                        # Keep its identity and endpoint binding stable.
                        return
                    job['pause_flush']={'request_id':uuid.uuid4().hex,'state':'awaiting_capture'};self.put(job)
                try:self._send(self.capture,{'type':action})
                except (OSError,ValueError):
                    self.stopped.set()
                    self._stop_capture()
                    abort(409,description='Audio capture stopped. Wait for the saved recording.')

    def pause_flush_request(self,jid,through_sample):
        """Bind the capture Pause endpoint to one ordered worker flush."""
        with self.lock:
            if type(through_sample) is not int or through_sample<0:raise ValueError('Invalid Pause audio endpoint.')
            job=self.get(jid);pending=job.get('pause_flush')
            if not pending or pending.get('state')!='awaiting_capture':raise ValueError('Unrequested capture Pause.')
            pending.update(state='pending',through_sample=through_sample)
            self.put(job)
            return {'type':'flush','request_id':pending['request_id'],'through_sample':through_sample}

    def acknowledge_pause_flush(self,jid,result):
        with self.lock:
            job=self.get(jid);pending=job.get('pause_flush') or {}
            if result.get('request_id')!=pending.get('request_id') or pending.get('state')!='pending':return False
            received=result.get('received_sample');available=result.get('available_sample');speech=result.get('speech_observed_sample')
            if (type(received) is not int or received!=pending['through_sample']
                    or type(available) is not int or type(speech) is not int or not 0<=available<=speech<=received
                    or type(result.get('fast_sequence')) is not int
                    or result['fast_sequence']!=job.get('last_fast_sequence',0)):
                raise ValueError('Pause flush acknowledgement differs from received audio or published revisions.')
            pending.update(state='complete',received_sample=received,available_sample=available,
                speech_observed_sample=speech,fast_sequence=result['fast_sequence'],
                deferred_audio=([{'start_sample':available,'end_sample':received}] if available<received else []))
            self.put(job);return True

    def _stop_capture(self):
        """Bound every stop request, including worker errors and a stalled helper."""
        with self.lock:
            capture=self.capture
            if not capture or capture.poll() is not None:return
            try:self._send(capture,{'type':'stop'})
            except (OSError,ValueError):pass
            if self.stop_timer:return
            def stop_stalled_capture():
                if capture.poll() is None:capture.terminate()
            self.stop_timer=threading.Timer(15,stop_stalled_capture)
            self.stop_timer.daemon=True;self.stop_timer.start()

    @staticmethod
    def _send(process, message):
        process.stdin.write(json.dumps(message)+'\n'); process.stdin.flush()

    def _run(self, jid, language, sources):
        import numpy as np
        handles = []
        threads = []
        worker_finished = threading.Event()
        transport_closed = threading.Event()
        worker_stop_sent = threading.Event()
        failure = []
        packets = self.packets
        stop_event = self.stopped
        capture_finished = False
        wav = None
        mixer = None
        tracks = {}
        dest = self.folder(jid)
        terminal = {'status':'failed','message':'Meeting stopped before capture began.'}
        def fail(message):
            with self.lock:
                if not failure:failure.append(message)
                stop_event.set()
                self.error=failure[0]
                terminal.update(status='failed',message=self.error)
                self._stop_capture()
        def consume_results():
            finished = False
            try:
                for line in self.worker.stdout:
                    result = json.loads(line)
                    with self.lock:
                        if result['type']=='segment':
                            job=self.get(jid);document=job['document']
                            validate_language_segment(result['segment'],job['language_history'])
                            append_finalized_segment(document,result)
                            document['provenance']['kind']='local_inference'
                            job.update(revision=job['revision']+1);self.put(job)
                            if self.recognizer:self.recognizer.observe(jid,result['segment']['speaker'])
                        elif result['type']=='provisional_revision':
                            self.refinement.provisional(jid,result)
                            if self.recognizer:self.recognizer.observe(jid)
                        elif result['type']=='canonical_revision':
                            self.refinement.canonical(jid,result)
                            if self.recognizer:self.recognizer.observe(jid)
                        elif result['type']=='language_registered':
                            job=self.get(jid)
                            epoch=next((e for e in job['language_history'] if e['generation']==result.get('generation')),None)
                            if not epoch or any(result.get(key)!=epoch[key] for key in ('generation','language','start_sample')):
                                raise ValueError('Invalid worker language acknowledgement.')
                            job['language_acknowledged_revision']=max(job['language_acknowledged_revision'],epoch['generation']);self.put(job)
                        elif result['type']=='refinement_result':self.refinement.result(jid,result)
                        elif result['type']=='boundary_candidate':
                            job=self.get(jid);job['boundary_candidate']=result
                            key=f"{result['language_epoch']}:{result['start']}:{result['end']}"
                            job.setdefault('boundary_candidates',{})[key]=result;self.put(job)
                        elif result['type']=='capture_finished':
                            if not worker_stop_sent.is_set():fail('Capture finalization arrived before Stop. Audio is preserved.')
                            self.refinement.capture_done(jid,observed_sample=result.get('canonical_observed_sample'))
                        elif result['type']=='progress':
                            self.processed=result['processed_seconds'];self.refinement.schedule(jid,force=bool(result.get('flush')))
                        elif result['type']=='flush_ack':self.acknowledge_pause_flush(jid,result)
                        elif result['type']=='error':fail(result['error']);break
                        elif result['type']=='finished':
                            if not worker_stop_sent.is_set():fail('Live inference finished before recording stopped. Captured audio has been saved.')
                            finished=True;break
            except Exception:fail('Live transcription stopped unexpectedly. Captured audio has been saved.')
            finally:
                if not finished and not failure:fail('Live inference exited before completing. Captured audio has been saved.')
                worker_finished.set()
        def send_audio():
            try:
                while True:
                    try:message=self.worker_controls.get_nowait()
                    except queue.Empty:
                        try:message=packets.get(timeout=.1)
                        except queue.Empty:
                            if transport_closed.is_set():return
                            continue
                    if message['type']=='stop':worker_stop_sent.set()
                    self._send(self.worker,message)
                    if message['type']=='shutdown' or (message['type']=='stop' and not self.two_pass):return
            except (OSError,ValueError):fail('Live transcription stopped. Captured audio has been saved.')
        try:
            cfg=model_config();cfg.update(language=language,language_epoch=0,audio_path=str(dest/'audio.wav'),
                                         job_id=jid,canonical_utterances=True)
            python=os.getenv('SPEAKERDESK_LIVE_PYTHON',str(Path(__file__).resolve().parents[1]/'.venv-package/bin/python'))
            command=([sys.executable,'--live-worker',json.dumps(cfg)] if getattr(sys,'frozen',False)
                     else [python,str(Path(__file__).with_name('live_worker.py')),json.dumps(cfg)])
            log=(dest/'live-worker.log').open('w');handles.append(log)
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            with self.lock:
                if stop_event.is_set():return
                self.worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
            # Startup has a bounded wait so a failed worker cannot strand the meeting.
            ready_queue=queue.Queue()
            thread=threading.Thread(target=lambda:ready_queue.put(self.worker.stdout.readline()),daemon=True)
            thread.start();threads.append(thread)
            deadline=time.monotonic()+60
            while True:
                if stop_event.is_set():return
                try:line=ready_queue.get(timeout=.1);break
                except queue.Empty:
                    if time.monotonic()>=deadline:raise RuntimeError('Local models could not start. Audio capture did not begin.')
            try:ready=json.loads(line)
            except ValueError:raise RuntimeError('Local models could not start. Audio capture did not begin.')
            if ready.get('type')!='ready':raise RuntimeError(ready.get('error','Local models could not start.'))
            self.two_pass=bool(ready.get('two_pass'))
            if ready.get('canonical_utterances'):
                with self.lock:
                    job=self.get(jid);job['canonical_utterances']=True;self.put(job)
            self.refinement.ready(jid,self.two_pass)
            if stop_event.is_set():return
            for target in (consume_results,send_audio):
                thread=threading.Thread(target=target,daemon=True);thread.start();threads.append(thread)
            audio=(dest/'audio.wav').open('w+b');handles.append(audio)
            wav=wave.open(audio,'wb');wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(RATE)
            tracks={}
            for source in sources:
                f=(dest/f'{source}.wav').open('w+b');handles.append(f)
                track=wave.open(f,'wb');track.setnchannels(1);track.setsampwidth(2);track.setframerate(RATE);tracks[source]=track
            def sink(mixed, separate):
                with self.lock:
                    start_sample=wav.getnframes()
                    wav.writeframes(pcm16_bytes(mixed))
                    audio.flush()  # Finalized voice clips must already be readable by the identity worker.
                    for source,track in tracks.items():track.writeframes(pcm16_bytes(separate[source]))
                    self.duration=wav.getnframes()/RATE
                    if not failure:
                        mode,epoch=self.active_language,self.active_epoch
                        message={'type':'audio','start_sample':start_sample,'end_sample':wav.getnframes(),
                                 'language':mode,'language_epoch':epoch}
                        if not self.two_pass:message['pcm']=base64.b64encode(mixed.tobytes()).decode()
                        try:packets.put_nowait(message)
                        except queue.Full:fail('Transcription fell 30 seconds behind. Recording stopped; captured audio has been saved.')
            mixer=SourceMixer(sources,sink)
            capture_log=(dest/'capture.log').open('w');handles.append(capture_log)
            with self.lock:
                if stop_event.is_set():return
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
                    flush=None
                    with self.lock:
                        if kind=='paused':
                            mixer.flush(result['time'],final=True)
                            self.patch(jid,duration=self.duration)
                            if self.two_pass:
                                self.refinement.force=True;flush=self.pause_flush_request(jid,mixer.cursor)
                            else:
                                # Legacy one-pass workers have no flush ACK.
                                # Capture acknowledgement still ends the
                                # duplicate-command guard, without claiming a
                                # processed horizon or completed worker receipt.
                                job=self.get(jid);receipt=job.get('pause_flush') or {}
                                if receipt.get('state')!='awaiting_capture':raise ValueError('Unrequested capture Pause.')
                                receipt.update(state='capture_complete',through_sample=mixer.cursor);self.put(job)
                        if not stop_event.is_set():
                            self.patch(jid,status=kind,message='Recording' if kind=='recording' else 'Paused')
                    if flush:packets.put(flush,timeout=2)
                    if kind=='recording' and not self.refinement.final:self.refinement.force=False
                elif kind=='stopped':
                    mixer.flush(result['time'],final=True);capture_finished=True
                elif kind=='error':raise RuntimeError(result['error'])
            if not capture_finished or (self.capture.poll() is not None and self.capture.returncode):
                raise RuntimeError('Audio capture stopped unexpectedly. Captured audio has been saved.')
            self.patch(jid,status='finishing',message='Finishing transcript…',duration=self.duration)
            wav.close()
            for track in tracks.values():track.close()
            if failure:raise RuntimeError(failure[0])
            packets.put({'type':'stop'},timeout=10)
            finish_budget=None
            if self.two_pass:
                finish_budget=threading.Timer(45,lambda:self.refinement.pause(jid,final=True))
                finish_budget.daemon=True;finish_budget.start()
            try:
                if not worker_finished.wait(timeout=60):raise RuntimeError('Transcription took too long to finish. Captured audio has been saved.')
            finally:
                if finish_budget:finish_budget.cancel()
            if failure:raise RuntimeError(failure[0])
            with self.lock:
                job=self.get(jid)
                if mixer.late_samples:job['document']['warnings'].append(f'{mixer.late_samples/RATE:.2f}s of late source samples were omitted from the mix. Review audio timing.')
                self.put(job)
            # Keep the latest document; an identity decision may finish during cleanup.
            terminal={'status':'ready','message':'Meeting saved on this Mac.'}
        except Exception as exc:
            fail(str(exc))
            # The watermark intentionally holds the latest source packets in memory.
            # Preserve that valid tail on protocol/worker errors as well as normal stop.
            if mixer:
                try:mixer.flush(max(mixer.last_packet.values()),final=True)
                except Exception:pass  # Keep the original diagnostic, e.g. a disk write failure.
            self.error=failure[0]
            terminal={'status':'failed','message':self.error}
        finally:
            stop_event.set()
            transport_closed.set()
            cleanup_failed = False
            def close_handle(handle):
                nonlocal cleanup_failed
                try:handle.close()
                except (OSError,ValueError):cleanup_failed=True
            for recording in [wav, *tracks.values()]:
                if recording:close_handle(recording)
            for child in (self.capture,self.worker):
                if child and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
            for thread in threads:
                thread.join(timeout=3)
            for handle in handles:
                close_handle(handle)
            with self.lock:
                if self.stop_timer:self.stop_timer.cancel();self.stop_timer=None
                for child in (self.capture,self.worker):
                    if child:
                        close_handle(child.stdin);close_handle(child.stdout)
                self.capture=self.worker=None
                if cleanup_failed and terminal['status']=='ready':
                    self.error='Meeting cleanup failed. Check available storage and review the recorded audio.'
                    terminal.update(status='failed',message=self.error)
                try:
                    job=self.get(jid)
                    if self.two_pass and job.get('refinement_status') not in ('complete','paused','unresolved'):
                        self.refinement.pause(jid,final=True);job=self.get(jid)
                    if self.duration>0 and job.get('document'):
                        from review import retain_unassigned_audio
                        retain_unassigned_audio(job['document'],self.duration)
                        job['revision']+=1;self.put(job)
                    self.patch(jid,**terminal,duration=self.duration)
                finally:self.jid=None

    def close(self):
        with self.lock:
            self.stopped.set()
            for process in (self.capture,self.worker):
                if process and process.poll() is None:process.terminate()
            thread=self.thread
        if thread and thread is not threading.current_thread():thread.join(timeout=5)
        if self.recognizer:self.recognizer.close()

    def resume_refinement(self, jid):
        with self.lock:
            if self.jid==jid and self.worker and self.two_pass:
                self.refinement.resume(jid);return self.get(jid)
            if self.jid or self.capture or self.worker or self.inference_busy():abort(409,description='Wait for current local inference to finish.')
            job=self.get(jid)
            if job.get('kind')!='meeting' or not job.get('document') or not (self.folder(jid)/'audio.wav').is_file():
                abort(409,description='This meeting has no saved audio to refine.')
            if preflight(model_config(),job['language']):abort(409,description='Finish local model setup before refining saved audio.')
            from rolling_refinement import RollingPlan
            self.refinement.initialize(job);plan=RollingPlan(job['rolling_refinement'],recover=True)
            if job.get('canonical_utterances'):
                for row in job['rolling_sources'].values():
                    if row.get('canonical_unresolved'):
                        job.setdefault('canonical_refined',{}).pop(row['id'],None)
            plan.cancel();plan.resume();job['rolling_refinement']=plan.snapshot()
            if job['refinement_unresolved']:
                job['rolling_refinement']['completed_sample']=min(job['rolling_refinement']['completed_sample'],min(w['start_sample'] for w in job['refinement_unresolved']))
            job['refinement_status']='waiting';job['message']='Refining saved passages…';self.put(job)
            self.jid=jid;self.duration=job['duration'];self.processed=self.duration
            self.stopped=threading.Event();self.worker_controls=queue.Queue(maxsize=8);self.two_pass=True
            self.refining_saved=True
            self.thread=threading.Thread(target=self._run_saved_refinement,args=(jid,),daemon=True);self.thread.start()
            return job

    def _run_saved_refinement(self,jid):
        dest=self.folder(jid);process=None;timer=None;log=None;sender=None;closed=threading.Event()
        try:
            job=self.get(jid);cfg=model_config();cfg.update(language=job['language'],language_epoch=job.get('language_epoch',0),
                audio_path=str(dest/'audio.wav'),refinement_only=True,job_id=jid,
                canonical_utterances=bool(job.get('canonical_utterances')),language_history=job.get('language_history'),
                fast_sequence=job.get('last_fast_sequence',0))
            python=os.getenv('SPEAKERDESK_LIVE_PYTHON',str(Path(__file__).resolve().parents[1]/'.venv-package/bin/python'))
            command=([sys.executable,'--live-worker',json.dumps(cfg)] if getattr(sys,'frozen',False)
                     else [python,str(Path(__file__).with_name('live_worker.py')),json.dumps(cfg)])
            log=(dest/'live-worker.log').open('a')
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'))
            with self.lock:self.worker=process
            def expire():
                try:self.refinement.pause(jid,final=True)
                finally:
                    if process.poll() is None:process.terminate()
            timer=threading.Timer(60,expire);timer.daemon=True;timer.start()
            ready=json.loads(process.stdout.readline())
            if ready.get('type')!='ready' or not ready.get('two_pass'):raise RuntimeError('Saved refinement worker could not start.')
            self.refinement.ready(jid,True)
            def send():
                while not closed.is_set():
                    try:message=self.worker_controls.get(timeout=.1)
                    except queue.Empty:continue
                    try:self._send(process,message)
                    except (OSError,ValueError):return
                    if message['type']=='shutdown':return
            sender=threading.Thread(target=send,daemon=True);sender.start()
            for line in process.stdout:
                result=json.loads(line)
                if result['type']=='capture_finished':self.refinement.capture_done(jid,observed_sample=result.get('canonical_observed_sample'))
                elif result['type']=='refinement_result':self.refinement.result(jid,result)
                elif result['type']=='finished':break
                elif result['type']=='error':raise RuntimeError(result.get('error','Saved refinement failed.'))
        except Exception:
            with self.lock:
                job=self.get(jid);job.update(refinement_status='paused',refinement_error='Refinement paused. Previous words and saved audio are retained.');self.put(job)
        finally:
            try:
                closed.set()
                if timer:timer.cancel()
                if process and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
                if sender:sender.join(timeout=3)
                if process:
                    for stream in (process.stdin,process.stdout):
                        try:stream.close()
                        except (OSError,ValueError):pass
                if log:log.close()
                with self.lock:
                    job=self.get(jid)
                    if job.get('refinement_status') not in ('complete','unresolved'):job['refinement_status']='paused'
                    job['message']='Meeting saved on this Mac.';self.put(job)
            finally:
                with self.lock:
                    self.worker=None;self.jid=None;self.refining_saved=False


def append_finalized_segment(document, result):
    # The diarizer owns meeting tracks; confirmed/custom names belong to the user.
    for track, name in result['speakers'].items():
        document['speakers'].setdefault(track, name)
    document['segments'].append(result['segment'])


def register_meetings(app, get, put, patch, folder, lock, inference_busy, recognizer=None, *, default_language=None):
    manager=MeetingManager(get,put,patch,folder,lock,inference_busy,recognizer,default_language=default_language)
    app.extensions['speakerdesk']['meetings']=manager
    @app.get('/api/meeting')
    def meeting_status():return jsonify(manager.status())
    @app.post('/api/meetings')
    def start_meeting():
        body=request.get_json();sources=body.get('sources')
        if not isinstance(sources,list) or any(not isinstance(s,str) for s in sources) or len(sources)!=len(set(sources)):
            raise ValueError('Select valid meeting audio sources.')
        return jsonify(manager.start(body.get('name','Meeting'),body.get('language',manager.default_language()),sources)),201
    @app.post('/api/meetings/<jid>/<action>')
    def control_meeting(jid,action):
        if action not in ('pause','resume','stop'):abort(404)
        manager.control(jid,action);return jsonify(get(jid)),202

    @app.patch('/api/meetings/<jid>/language')
    def change_meeting_language(jid):
        body=request.get_json()
        if not isinstance(body,dict):raise ValueError('Choose a live language.')
        return jsonify(manager.change_language(jid,body.get('language'),body.get('language_revision')))

    @app.post('/api/jobs/<jid>/refinement/<action>')
    def refinement_control(jid,action):
        if action=='resume':return jsonify(manager.resume_refinement(jid)),202
        if action!='pause':abort(404)
        with lock:
            if manager.jid!=jid:abort(409,description='This meeting is not being refined.')
            manager.refinement.pause(jid,final=get(jid)['status'] not in LIVE)
            return jsonify(get(jid)),202

    @app.get('/api/jobs/<jid>/refinement/revisions/<window_id>')
    def refinement_history(jid,window_id):
        history=get(jid).get('refinement_history',{}).get(window_id)
        if history is None:abort(404)
        return jsonify(history)

    @app.get('/api/jobs/<jid>/refinement/fast-revisions/<int:start_sample>')
    def fast_history(jid,start_sample):
        history=get(jid).get('fast_history',{}).get(str(start_sample))
        if history is None:abort(404)
        return jsonify(history)

    @app.get('/api/jobs/<jid>/refinement/boundary-candidates')
    def boundary_history(jid):return jsonify(candidates=list(get(jid).get('boundary_candidates',{}).values()))
