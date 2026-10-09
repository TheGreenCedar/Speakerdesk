"""Lazy resident voice child, registered with the app's inference lifecycle."""
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from voice_worker_registry import launch_owned_worker, release_owned_worker, stop_owned_worker
from voice_profiles import ClipEmbedding, normalized

MAX_FRAME_BYTES = 65536
MAX_DIAGNOSTIC_BYTES = 16 * 1024**2
MAX_MLX_BYTES = 256 * 1024**2


class OwnedVoiceBackend:
    def __init__(self, root, config_path, model, updates, audio_root, *, timeout=90):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 90:
            raise ValueError('Voice inference needs a finite bounded timeout.')
        self.root, self.config_path = Path(root).resolve(), Path(config_path).resolve()
        self.model, self.updates = model, updates
        self.audio_root = Path(audio_root).resolve()
        self.timeout = timeout
        self._gate = threading.RLock()
        self._closed = False
        self._cleaned = False
        self._failed = threading.Event()
        self._process = None
        self._responses = queue.Queue(maxsize=2)
        self._readers = []
        self._next_id = 0
        self._owner_thread_id = None

    def require_execution_policy(self):
        if (self._closed or self._failed.is_set()
                or (self._process is not None and self._process.poll() is not None)):
            raise ValueError('The voice inference worker is unavailable. Retry local voice setup.')

    def _read_frames(self):
        try:
            while line := self._process.stdout.readline(MAX_FRAME_BYTES+1):
                if len(line.encode('utf-8')) > MAX_FRAME_BYTES or not line.endswith('\n'):
                    raise ValueError('Voice worker frame exceeded its bound.')
                self._responses.put(json.loads(line), timeout=1)
        except (OSError, ValueError, queue.Full):
            self._failed.set()
        finally:
            # Wake the sole pending caller on EOF; no unbounded queue or buffering.
            try:self._responses.put(None, timeout=1)
            except queue.Full:self._failed.set()

    def _drain_diagnostics(self):
        size = 0
        try:
            while chunk := self._process.stderr.read(4096):
                size += len(chunk.encode('utf-8'))
                if size > MAX_DIAGNOSTIC_BYTES:
                    self._failed.set()
                    stop_owned_worker(self._process)
                    return
        except (OSError, ValueError):
            self._failed.set()

    def _receive(self, deadline):
        while True:
            if self._failed.is_set():
                raise ValueError('Voice worker transport failed.')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise ValueError('Voice recognition exceeded its bounded inference time.')
            try:message = self._responses.get(timeout=min(remaining, .1))
            except queue.Empty:continue
            if not isinstance(message, dict) or 'error' in message:
                raise ValueError('Voice worker could not complete this request.')
            return message

    def _start(self, deadline):
        if self._process is not None:
            return
        init = json.dumps({'model_dir':str(self.root), 'config_path':str(self.config_path),
                           'audio_root':str(self.audio_root), 'maximum_MLX_bytes':MAX_MLX_BYTES})
        command = ([sys.executable, '--voice-worker', init] if getattr(sys, 'frozen', False)
                   else [os.getenv('ASR_PYTHON', sys.executable), '-u',
                         str(Path(__file__).with_name('voice_gpu_worker.py')), init])
        env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'}
        self._process = launch_owned_worker(command, process_group=True, env=env,
            stdin=-1, stdout=-1, stderr=-1, text=True, encoding='utf-8', errors='replace', bufsize=1)
        for name, function in [('voice-output',self._read_frames), ('voice-diagnostics',self._drain_diagnostics)]:
            reader = threading.Thread(target=function, name=name, daemon=True)
            self._readers.append(reader);reader.start()
        if self._receive(deadline) != {'type':'ready', 'model':self.model.payload()}:
            raise ValueError('Voice worker identity did not match the accepted policy.')

    def embed(self, audio, clip):
        lease = self.updates.begin_activity('voice inference')
        try:
            with self._gate:
                self.require_execution_policy()
                from voice_source_audio import validate_voice_audio,clip_pcm_digest
                audio = validate_voice_audio(self.audio_root,audio,clip)
                pcm_digest=clip_pcm_digest(audio,clip) if getattr(clip,'capture_source',None) is not None else None
                deadline = time.monotonic()+self.timeout
                try:
                    self._start(deadline)
                    self._next_id += 1
                    request = {'op':'embed', 'id':self._next_id, 'audio':str(audio), 'clip':clip.payload()}
                    if pcm_digest is not None:request['source_pcm_sha256']=pcm_digest
                    self._process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
                    self._process.stdin.flush()
                    result = self._receive(deadline)
                    if (result.get('type') != 'embedding' or type(result.get('id')) is not int
                            or result['id'] != self._next_id or result.get('model') != self.model.payload()
                            or type(result.get('clean')) is not bool):
                        raise ValueError('Voice worker response does not match its request.')
                    if pcm_digest is not None and (result.get('source_pcm_sha256')!=pcm_digest
                            or clip_pcm_digest(audio,clip)!=pcm_digest):
                        raise ValueError('Voice source PCM changed during inference.')
                    vector = tuple(result['vector']) if result['clean'] else ()
                    if result['clean']:
                        normalized(vector, self.model.dimension)  # Validate without changing the qualified vector.
                        if abs(math.sqrt(sum(v*v for v in vector))-1) > 1e-6:
                            raise ValueError('Voice worker returned an unnormalized embedding.')
                    elif result.get('vector') != []:
                        raise ValueError('Rejected audio must not supply an embedding.')
                    metrics = result.get('metrics')
                    owner = result.get('owner_thread_id')
                    if (type(owner) is not int or owner <= 0
                            or (self._owner_thread_id is not None and owner != self._owner_thread_id)):
                        raise ValueError('Voice inference changed its owner thread.')
                    self._owner_thread_id = owner
                    if result['clean'] and (not isinstance(metrics, dict)
                            or metrics.get('completed_GPU_declarations') != 809
                            or metrics.get('CPU_neural_retry') is not False):
                        raise ValueError('Voice worker did not finish a qualified GPU forward.')
                    return ClipEmbedding(vector, result['clean'],
                        {**metrics, 'inference_owner_thread_id':owner} if metrics is not None else None)
                except BaseException:
                    self._failed.set()
                    self.close()
                    raise
        finally:
            self.updates.finish_activity(lease)

    def close(self):
        with self._gate:
            if self._cleaned:
                return
            self._closed = True
            if self._process is None:
                self._cleaned = True
                return
            try:
                if self._process.poll() is None:
                    try:
                        self._process.stdin.write('{"op":"shutdown"}\n');self._process.stdin.flush()
                        self._process.wait(timeout=3)
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        stop_owned_worker(self._process)
            finally:
                stop_owned_worker(self._process)
                for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                    stream.close()
                for reader in self._readers:
                    reader.join(timeout=3)
                if any(reader.is_alive() for reader in self._readers):
                    raise RuntimeError('Voice transport cleanup has not finished.')
                release_owned_worker(self._process)
                self._cleaned = True
