"""One local identity decision per meeting track; profiles are never learned here."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
import uuid
from people import introduction_suggestions
from voice_profiles import propose_match
from source_voice import automatic_clips, clips_current, extract


class RecognitionPreference:
    def __init__(self, db):
        self.db = db
        with db() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL)')

    def enabled(self):
        with self.db() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key='voice_recognition'").fetchone()
        return row is None or json.loads(row['value']) is True

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Choose whether to recognize saved voices.')
        with self.db() as conn:
            conn.execute('INSERT OR REPLACE INTO preferences VALUES (?,?)', ('voice_recognition', json.dumps(enabled)))


class VoiceRecognition:
    def __init__(self, get, put, folder, lock, runtime, store, preference, logger, *, updates):
        self.get, self.put, self.folder, self.lock = get, put, folder, lock
        self.runtime, self.store, self.preference, self.logger = runtime, store, preference, logger
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='voice-recognition')
        self.closed = False
        self.updates = updates

    def observe(self, jid, track=None):
        """Snapshot eligible evidence quickly; model work happens outside the app lock."""
        with self.lock:
            if self.closed or not self.preference.enabled():
                return
            backend, calibration = self.runtime()
            if backend is None or calibration is None or backend.model != calibration.model:
                return
            profiles = [p for p in self.store.profiles() if p['model'] == backend.model.payload()]
            if not profiles:
                return  # No saved voices means no reason to embed meeting audio.
            job = self.get(jid)
            document = job.get('document') or {}
            if document.get('provenance', {}).get('kind') != 'local_inference':
                return
            for candidate in ([track] if track else document.get('speakers', {})):
                if (candidate not in document['speakers'] or job.get('speaker_assignments', {}).get(candidate)
                        or not re.fullmatch(r'Speaker \d+', document['speakers'][candidate])
                        or candidate in job.get('voice_checks', {})):
                    continue
                clips = automatic_clips(job, candidate, calibration.minimum_clips)
                if not clips:
                    continue
                audio = self.folder(jid)/'audio.wav'
                if not audio.is_file():
                    continue
                check_id = uuid.uuid4().hex
                job.setdefault('voice_checks', {})[candidate] = {'id': check_id, 'status': 'checking',
                    'model': backend.model.payload(), 'clips': [c.payload() for c in clips]}
                self.put(job)
                args = (jid, candidate, check_id, backend, calibration, profiles, clips, audio)
                self.updates.submit(self.executor, 'voice recognition', self._check, *args)

    def _check(self, jid, track, check_id, backend, calibration, profiles, clips, audio):
        match, failed = None, False
        try:
            match = propose_match(backend.model, calibration, extract(backend, audio, clips), profiles)
        except Exception as error:
            failed = True
            self.logger.warning('Voice check kept speaker unknown: %s', error)
        with self.lock:
            if self.closed:
                return
            try:
                job = self.get(jid)
            except Exception:
                return  # A completed import may have been deleted while its check ran.
            check = job.get('voice_checks', {}).get(track)
            if not check or check['id'] != check_id or check['status'] != 'checking':
                return
            # New transcript phrases do not invalidate a track; edits to selected evidence do.
            versions = {(p['person_id'], p['version']) for p in self.store.profiles()
                        if p['model'] == backend.model.payload()}
            unchanged = (clips_current(job, clips) and versions == {(p['person_id'], p['version']) for p in profiles}
                         and self.runtime() == (backend, calibration) and self.preference.enabled()
                         and not job.get('speaker_assignments', {}).get(track)
                         and re.fullmatch(r'Speaker \d+', job['document']['speakers'].get(track, '')))
            check.update(status='unknown', checked_at=time.time())
            if failed:
                check['status'] = 'unavailable'
            if not unchanged:
                check['status'] = 'cancelled'
            if unchanged and match:
                person = next((p for p in self.store.list() if p['id'] == match['person_id']), None)
                intro = next((s for s in introduction_suggestions(job) if s['track_id'] == track), None)
                if person and (not intro or intro['name'].casefold() == person['name'].casefold()):
                    job['document']['speakers'][track] = person['name']
                    job.setdefault('speaker_assignments', {})[track] = {
                        'person_id': person['id'], 'meeting_id': jid, 'track_id': track, 'name': person['name'],
                        'source': 'automatic_voice', 'confirmed': False, 'assigned_at': time.time(),
                        'evidence': {'match': match, 'clips': [c.payload() for c in clips]}}
                    check['status'] = 'matched'
            job['revision'] += 1
            self.put(job)

    def close(self, *, wait=False):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=wait, cancel_futures=True)
