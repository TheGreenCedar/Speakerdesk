"""Runtime admission and a bounded pipe protocol for the trusted desktop updater.

The native host owns update URLs, signatures and installation. This service only
requests fixed operations and reserves the same lock used for recording work.
"""
import copy
import json
import threading

from flask import abort, jsonify, request

PREFIX = 'SPEAKERDESK_UPDATE='
MAX_FRAME_BYTES = 8192
MUTATING_METHODS = ('POST', 'PUT', 'PATCH', 'DELETE')
STATES = frozenset(('unavailable', 'idle', 'checking', 'current', 'available',
                    'downloading', 'verifying', 'ready', 'preparing', 'preparing_install', 'installing',
                    'stopping', 'shutting_down', 'restart_required', 'cancelled', 'error'))


class UpdateReserved(RuntimeError):
    pass


def decode_frame(line):
    if not isinstance(line, str) or len(line.encode('utf-8')) > MAX_FRAME_BYTES:
        raise ValueError('Update control frame is too large.')
    if not line.startswith(PREFIX):
        raise ValueError('Unknown update control frame.')
    body = json.loads(line[len(PREFIX):])
    if not isinstance(body, dict) or type(body.get('id')) is not int or not 0 <= body['id'] < 2**53:
        raise ValueError('Invalid update attempt.')
    return body


class RuntimeUpdates:
    def __init__(self, lock, busy):
        self.lock, self.busy = lock, busy
        self._activities = {}
        self._next_activity = 0
        self._attempt = 0
        self._reservation = None
        self._native_guarded = False
        self._shutting_down = False
        self._output = None
        self._pending_install = False
        self._editor_ready = False
        self._preparation = None
        self._last_preparation = 0
        self._status = {'id': 0, 'state': 'unavailable',
                        'error': 'App updates are available in the desktop app.'}

    def connect(self, output):
        """Attach the desktop stdout transport; never supplied by an HTTP client."""
        with self.lock:
            self._output = output
            self._status = {'id': self._attempt, 'state': 'idle'}

    def _emit(self, message):
        frame = PREFIX + json.dumps(message, separators=(',', ':'), ensure_ascii=True)
        if len(frame.encode('utf-8')) > MAX_FRAME_BYTES:
            raise ValueError('Update control frame is too large.')
        self._output(frame)

    def begin_activity(self, kind):
        with self.lock:
            if self._reservation is not None or self._shutting_down:
                raise UpdateReserved('Speakerdesk is preparing an update. Retry after it finishes or is cancelled.')
            self._next_activity += 1
            lease = self._next_activity
            self._activities[lease] = kind
            return lease

    def finish_activity(self, lease):
        with self.lock:
            return self._activities.pop(lease, None) is not None

    def submit(self, executor, kind, function, *args, **kwargs):
        """Retain admission while queued, running, or waiting for final cleanup."""
        with self.lock:
            lease = self.begin_activity(kind)
            try:
                future = executor.submit(function, *args, **kwargs)
            except BaseException:
                self.finish_activity(lease)
                raise
            future.add_done_callback(lambda _future: self.finish_activity(lease))
            return future

    def start_thread(self, kind, function, *args, name=None):
        with self.lock:
            lease = self.begin_activity(kind)
            def owned():
                try:
                    function(*args)
                finally:
                    self.finish_activity(lease)
            thread = threading.Thread(target=owned, daemon=True, name=name or kind)
            try:
                thread.start()
            except BaseException:
                self.finish_activity(lease)
                raise
            return thread

    def status(self):
        with self.lock:
            return {**copy.deepcopy(self._status), 'reserved': self._reservation is not None or self._native_guarded}

    def request_operation(self, op, *, attempt=None, preparation=None, error=None):
        with self.lock:
            if self._output is None:
                abort(409, description=self._status['error'])
            state = self._status['state']
            if op in ('check','download','install') and self._native_guarded:
                abort(409,description='Wait for the protected update operation to finish or be safely cancelled.')
            if op in ('download','install') and (self._activities or self.busy()):
                abort(409,description='Finish the active recording, transcription and local setup before downloading or installing an update.')
            if op == 'check':
                if self._reservation is not None or state in ('checking', 'downloading', 'verifying', 'preparing', 'preparing_install', 'stopping', 'installing', 'shutting_down'):
                    abort(409, description='Wait for the current update operation or cancel it.')
                self._attempt += 1
                self._pending_install = False
                self._editor_ready = False
                self._preparation = None
                self._status = {'id': self._attempt, 'state': 'checking'}
            elif op == 'download':
                if state != 'available':
                    abort(409, description='Check for an available update before downloading.')
                self._status.update(state='downloading', downloaded_bytes=0)
            elif op == 'install':
                if state != 'ready' or self._pending_install:
                    abort(409, description='Download and verify the update before installing.')
                self._pending_install = True
                self._editor_ready = False
                self._preparation = None
                self._status.pop('preparation',None)
                self._status.update(state='preparing')
            elif op in ('editor_ready', 'editor_error'):
                if (attempt != self._attempt or preparation != self._preparation or self._preparation is None or
                        not self._pending_install or state not in ('preparing', 'preparing_install')):
                    abort(409, description='This update preparation is no longer current.')
                self._editor_ready = op == 'editor_ready'
            elif op == 'cancel':
                # Native uses 'stopping' while its reservation handshake is still
                # cancellable. Only its validated status can open this interval;
                # committed runtime shutdown always closes it.
                if (self._shutting_down or state in ('installing', 'shutting_down', 'restart_required') or
                        (state == 'stopping' and self._status.get('cancellable') is not True)):
                    abort(409, description='Installation is finishing. Keep Speakerdesk open.')
                if self._attempt == 0:
                    abort(409, description='There is no update to cancel.')
            else:
                raise ValueError('Choose check, download, install or cancel.')
            self._emit({'id': self._attempt, 'op': op,
                        **({'preparation':preparation} if op in ('editor_ready','editor_error') else {}),
                        **({'error': error} if op == 'editor_error' and error else {})})
            return self.status()

    def handle(self, message):
        """Process native commands. A shutdown acknowledgement is not process exit."""
        if not isinstance(message, dict) or type(message.get('id')) is not int:
            return False
        attempt, op = message['id'], message.get('op')
        with self.lock:
            if attempt != self._attempt:
                return False
            if op == 'status':
                if not isinstance(message.get('state'),str) or message['state'] not in STATES:
                    return False
                status = {'id': attempt, 'state': message['state']}
                for key in ('reserved','cancellable'):
                    if key in message and type(message[key]) is not bool:return False
                if 'cancellable' in message:status['cancellable']=message['cancellable']
                for key in ('version', 'error', 'notes'):
                    if key in message:
                        if not isinstance(message[key], str) or len(message[key]) > (4096 if key == 'notes' else 512):
                            return False
                        status[key] = message[key]
                for key in ('downloaded_bytes', 'total_bytes'):
                    if key in message:
                        if type(message[key]) is not int or not 0 <= message[key] < 2**53:
                            return False
                        status[key] = message[key]
                preparation=message.get('preparation')
                new_preparation=False
                if preparation is None:
                    if self._pending_install or self._reservation is not None:return False
                else:
                    if type(preparation) is not int or not 0<preparation<2**53:return False
                    if preparation<self._last_preparation:return False
                    if self._pending_install and self._preparation is None:
                        if preparation<=self._last_preparation or message['state'] not in ('preparing','preparing_install'):
                            return False
                        new_preparation=True
                    elif preparation!=self._preparation:return False
                    status['preparation']=preparation
                if new_preparation:
                    self._preparation=preparation
                    self._last_preparation=preparation
                self._native_guarded=message.get('reserved',self._native_guarded)
                self._status = status
                if message['state'] in ('cancelled', 'error', 'available', 'ready') and self._reservation is None:
                    self._pending_install = False
                return True
            if attempt <= 0 or self._output is None:
                return False
            preparation=message.get('preparation')
            if type(preparation) is not int or preparation!=self._preparation or not 0<preparation<2**53:
                return False
            owner=(attempt,preparation)
            if op == 'reserve':
                allowed = (self._pending_install and self._editor_ready and not self._shutting_down and
                           (self._reservation == owner or
                            (self._reservation is None and not self._activities and not self.busy())))
                if allowed:
                    self._reservation = owner
                self._emit({'id': attempt, 'preparation':preparation, 'op': 'reserved', 'ok': bool(allowed),
                            **({} if allowed else {'error': 'Finish recording, transcription, model setup and local changes before installing.'})})
                return bool(allowed)
            if op == 'release':
                released = self._reservation in (None,owner) and not self._shutting_down
                if released:
                    self._reservation = None
                    self._pending_install = False
                self._emit({'id': attempt, 'preparation':preparation, 'op': 'released', 'ok': released})
                return bool(released)
            if op == 'shutdown':
                if self._reservation != owner or self._shutting_down or self._activities or self.busy():
                    self._emit({'id': attempt, 'preparation':preparation, 'op': 'shutdown_ready', 'ok': False,
                                'error': 'The runtime is still busy. Installation was not started.'})
                    return False
                self._shutting_down = True
                self._status.update(state='shutting_down')
                return True
            return False

    def shutdown_result(self, attempt, preparation, *, ok, error=None):
        with self.lock:
            if (attempt,preparation) != self._reservation or not self._shutting_down:
                return False
            self._emit({'id': attempt, 'preparation':preparation, 'op': 'shutdown_ready', 'ok': bool(ok),
                        **({'error': error} if error else {})})
            return True


def register_updates(app, updates):
    @app.get('/api/updates')
    def update_status():
        return jsonify(updates.status())

    @app.post('/api/updates')
    def update_operation():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError('Choose check, download, install or cancel.')
        op=body.get('op')
        if op in ('editor_ready','editor_error'):
            if (set(body)-{'op','id','preparation','error'} or type(body.get('id')) is not int or
                    type(body.get('preparation')) is not int or not 0<body['preparation']<2**53 or
                    (op=='editor_ready' and 'error' in body) or
                    ('error' in body and (not isinstance(body['error'],str) or len(body['error'])>512))):
                raise ValueError('Supply the current update attempt and a brief editor result.')
        elif set(body)!={'op'} or op not in ('check','download','install','cancel'):
            raise ValueError('Choose check, download, install or cancel.')
        return jsonify(updates.request_operation(op,attempt=body.get('id'),preparation=body.get('preparation'),error=body.get('error'))), 202
