"""Keep construction and every model call on one joined inference thread."""
from concurrent.futures import ThreadPoolExecutor
import threading


class ThreadBoundVoiceBackend:
    def __init__(self, factory, model, *, require_admission):
        self.model = model
        self._require_admission = require_admission
        self._gate = threading.Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='voice-gpu-owner')
        self.owner_thread_id = None
        try:
            self._backend = self._executor.submit(self._construct, factory).result()
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def _construct(self, factory):
        self._require_admission()
        self.owner_thread_id = threading.get_ident()
        backend = factory()
        if backend.model != self.model:
            raise ValueError('Exact original neural model identity required')
        return backend

    def require_execution_policy(self):
        self._require_admission()
        if self._closed:
            raise ValueError('Voice inference thread is closed')

    def _embed(self, audio, clip):
        self._require_admission()
        if threading.get_ident() != self.owner_thread_id:
            raise ValueError('Voice inference must remain on its owner thread')
        return self._backend.embed(audio, clip)

    def embed(self, audio, clip):
        self.require_execution_policy()
        if threading.get_ident() == self.owner_thread_id:
            raise ValueError('Reentrant model dispatch is not permitted')
        # Also bounds the queue: no next caller submits while a call is pending.
        with self._gate:
            self.require_execution_policy()
            return self._executor.submit(self._embed, audio, clip).result()

    def close(self):
        if threading.get_ident() == self.owner_thread_id:
            raise ValueError('The caller must join the inference thread')
        with self._gate:
            if self._closed:
                return
            self._closed = True
            # A concurrent close must not report success before this join ends.
            self._executor.shutdown(wait=True, cancel_futures=True)

