"""Reusable local People and confirmed meeting identities, independent of diarizer slots."""
import hashlib
import json
import re
import time
import uuid
from flask import abort, jsonify, request
from voice_profiles import clean_clips, extract, make_profile, propose_match


def person_name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError('Enter a name of 1–100 characters.')
    return value.strip()


class PeopleStore:
    def __init__(self, db):
        self.db = db
        with db() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS voice_profiles (person_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')

    def get(self, pid):
        with self.db() as conn:
            row = conn.execute('SELECT payload FROM people WHERE id=?', (pid,)).fetchone()
        if not row:
            abort(404, description='This person is no longer in People.')
        return json.loads(row['payload'])

    def list(self):
        with self.db() as conn:
            people = [json.loads(r['payload']) for r in conn.execute('SELECT payload FROM people')]
            voices = {r['person_id'] for r in conn.execute('SELECT person_id FROM voice_profiles')}
        return sorted([dict(p, voice_saved=p['id'] in voices) for p in people], key=lambda p: p['name'].casefold())

    def create(self, name):
        person = {'id': uuid.uuid4().hex, 'name': person_name(name), 'created': time.time()}
        with self.db() as conn:
            count = conn.execute('SELECT count(*) FROM people').fetchone()[0]
            if count >= 500:
                raise ValueError('People can hold up to 500 entries.')
            conn.execute('INSERT INTO people VALUES (?,?)', (person['id'], json.dumps(person)))
        return person

    def rename(self, pid, name):
        person = self.get(pid)
        person.update(name=person_name(name), updated=time.time())
        with self.db() as conn:
            conn.execute('UPDATE people SET payload=? WHERE id=?', (json.dumps(person), pid))
        return person

    def profiles(self):
        with self.db() as conn:
            return [json.loads(r['payload']) for r in conn.execute('SELECT payload FROM voice_profiles')]

    def remember(self, pid, profile, consent):
        if consent is not True or profile.get('consent') != 'explicit_remember_voice':
            raise ValueError('Choose Remember voice to save a voice profile.')
        self.get(pid)
        profile = dict(profile, person_id=pid, version=uuid.uuid4().hex, enrolled_at=time.time())
        with self.db() as conn:
            conn.execute('INSERT OR REPLACE INTO voice_profiles VALUES (?,?)', (pid, json.dumps(profile)))

    def forget(self, pid):
        self.get(pid)
        with self.db() as conn:
            conn.execute('DELETE FROM voice_profiles WHERE person_id=?', (pid,))


# Limit discovery to explicit first-person introductions at the start of a turn.
# Addressing someone ("Priya, what do you think?") is never self-identification.
INTRO = re.compile(r"^(?:(?i:hi|hello|hey)[,!]?\s+)?(?i:i['’]m|i am|my name is)\s+"
                   r"(?P<name>[^\W\d_]+(?:[-’'][^\W\d_]+)*(?:\s+[^\W\d_]+(?:[-’'][^\W\d_]+)*){0,2})")


def introduction_suggestions(job):
    document = job.get('document') or {}
    if document.get('provenance', {}).get('kind') != 'local_inference':
        return []
    suggestions = {}
    for segment in document.get('segments', []):
        track = segment['speaker']
        if (not re.fullmatch(r'Speaker \d+', document['speakers'][track])
                or track.startswith('overlap') or segment.get('review') or segment.get('finalized') is False
                or len(segment.get('speaker_candidates', [track])) != 1):
            continue
        match = INTRO.match(segment['text'].strip())
        if not match:
            continue
        # ASR capitalization is imperfect, so uncertain/lowercase names stay unknown.
        words = match['name'].split()
        name_words = []
        for word in words:
            if not word[0].isupper():
                break
            name_words.append(word)
        if not name_words:
            continue
        name = ' '.join(name_words)
        evidence = {'segment_id': segment['id'], 'quote': segment['text'],
                    'start': segment['start'], 'end': segment['end']}
        digest = hashlib.sha256(json.dumps([track, name, evidence], sort_keys=True).encode()).hexdigest()[:32]
        if digest in job.get('dismissed_identity_suggestions', []):
            continue
        suggestions.setdefault(track, {'id': digest, 'track_id': track, 'name': name,
                                       'kind': 'explicit_introduction', 'evidence': evidence})
    return list(suggestions.values())


def reconcile_assignments(job, document, imported=False):
    """Ordinary corrections affect this meeting; they never rename/enroll People."""
    assignments = job.get('speaker_assignments', {})
    job['speaker_assignments'] = {track: value for track, value in assignments.items()
                                  if not imported and document['speakers'].get(track) == value['name']}
    job.pop('voice_suggestions', None)  # Text/timing/track edits invalidate clip evidence.
    for check in job.get('voice_checks', {}).values():
        if check['status'] == 'checking':
            check['status'] = 'cancelled'


def register_people(app, db, get, put, lock, folder, backend=None, calibration=None, voice_busy=lambda: False):
    store = PeopleStore(db)
    app.extensions['speakerdesk']['voice_runtime'] = (backend, calibration)
    app.extensions['speakerdesk']['people'] = store

    def voice_runtime():
        backend, calibration = app.extensions['speakerdesk']['voice_runtime']
        available = backend is not None and calibration is not None and backend.model == calibration.model
        return backend, calibration, available

    def suggestions(job):
        backend, calibration, available = voice_runtime()
        valid_profiles = {p['version']: p for p in store.profiles()}
        people_names = {p['id']: p['name'] for p in store.list()}
        policy = {'dataset_id': calibration.dataset_id, 'threshold': calibration.threshold,
                  'margin': calibration.margin, 'minimum_clips': calibration.minimum_clips} if available else None
        voices = [dict(s, name=people_names[s['person_id']]) for s in job.get('voice_suggestions', [])
                  if available and s['person_id'] in people_names
                  and s['match']['model'] == backend.model.payload() and s['match']['calibration'] == policy
                  and s['match']['profile_version'] in valid_profiles
                  and not job.get('speaker_assignments', {}).get(s['track_id'])
                  and s['id'] not in job.get('dismissed_identity_suggestions', [])]
        # Explicit introduction evidence takes priority over a voice guess.
        intros = introduction_suggestions(job)
        return intros + [s for s in voices if s['track_id'] not in {i['track_id'] for i in intros}]

    def editing_job(jid, body, track=None):
        job = get(jid)
        if job['status'] in ('preparing', 'queued', 'processing', 'finishing') or not job.get('document'):
            abort(409, description='Wait for finalized transcript passages before naming speakers.')
        if body.get('revision') != job['revision']:
            abort(409, description='The transcript changed. Review the current passages and try again.')
        if track is not None and track not in job['document']['speakers']:
            abort(404)
        return job

    @app.get('/api/people')
    def people_list():
        backend, calibration, available = voice_runtime()
        profiles = {p['person_id']: p for p in store.profiles()}
        people = [dict(p, voice_compatible=bool(available and p['id'] in profiles
                                               and profiles[p['id']]['model'] == backend.model.payload())) for p in store.list()]
        return jsonify(people=people, voice_available=available,
                       voice_message=app.extensions['speakerdesk'].get('voice_message', 'Voice recognition is unavailable.'))

    @app.post('/api/people')
    def people_create():
        with lock:
            return jsonify(store.create(request.get_json().get('name'))), 201

    @app.patch('/api/people/<pid>')
    def people_rename(pid):
        with lock:
            return jsonify(store.rename(pid, request.get_json().get('name')))

    @app.delete('/api/people/<pid>/voice')
    def forget_voice(pid):
        with lock:
            store.forget(pid)
        return jsonify(forgotten=True)

    @app.get('/api/jobs/<jid>/identity-suggestions')
    def identity_suggestions(jid):
        with lock:
            job = get(jid)
            return jsonify(revision=job['revision'], suggestions=suggestions(job))

    @app.post('/api/jobs/<jid>/identity-suggestions/<sid>/dismiss')
    def dismiss_suggestion(jid, sid):
        with lock:
            job = editing_job(jid, request.get_json())
            if not any(s['id'] == sid for s in suggestions(job)):
                abort(409, description='This suggestion is no longer current.')
            job['dismissed_identity_suggestions'] = (job.get('dismissed_identity_suggestions', []) + [sid])[-256:]
            job['revision'] += 1
            put(job)
            return jsonify(job)

    @app.post('/api/jobs/<jid>/speakers/<track>/identity')
    def confirm_identity(jid, track):
        body = request.get_json()
        with lock:
            job = editing_job(jid, body, track)
            suggestion = None
            if body.get('suggestion_id'):
                suggestion = next((s for s in suggestions(job) if s['id'] == body['suggestion_id'] and s['track_id'] == track), None)
                if not suggestion:
                    abort(409, description='This suggestion is no longer current.')
            pid = body.get('person_id')
            name = store.get(pid)['name'] if pid else person_name(suggestion['name'] if suggestion else body.get('name'))
            if suggestion and ((suggestion.get('person_id') and pid != suggestion['person_id'])
                               or name.casefold() != suggestion['name'].casefold()):
                raise ValueError('Confirm the suggested person, or make a separate name correction.')
            job['document']['speakers'][track] = name
            job.setdefault('speaker_assignments', {})[track] = {
                'person_id': pid, 'meeting_id': jid, 'track_id': track, 'name': name,
                'confirmed': True, 'confirmed_at': time.time(), 'source': suggestion['kind'] if suggestion else 'manual',
                'evidence': {k: suggestion[k] for k in ('evidence', 'match') if k in suggestion} if suggestion else None}
            job['voice_suggestions'] = [s for s in job.get('voice_suggestions', []) if s['track_id'] != track]
            job.setdefault('voice_checks', {})[track] = {'id': uuid.uuid4().hex, 'status': 'manual'}
            job['revision'] += 1
            put(job)
            return jsonify(job)

    @app.post('/api/people/<pid>/voice')
    def remember_voice(pid):
        body = request.get_json()
        with lock:
            backend, calibration, available = voice_runtime()
            store.get(pid)
            if body.get('consent') is not True:
                raise ValueError('Choose Remember voice to save a local voice profile.')
            if not available:
                abort(409, description='Voice recognition is not available in this build. Names can still be saved in People.')
            if voice_busy():
                abort(409, description='Finish the active recording or transcription before remembering a voice.')
            job = editing_job(body.get('meeting_id', ''), body, body.get('track_id'))
            assignment = job.get('speaker_assignments', {}).get(body.get('track_id'))
            if (not assignment or assignment['person_id'] != pid
                    or assignment.get('source') == 'automatic_voice' or assignment.get('confirmed') is False):
                raise ValueError('Confirm this speaker as the person before remembering their voice.')
            clips = clean_clips(job, body['track_id'], body.get('segment_ids'), calibration.minimum_clips)
            audio = folder(job['id'])/'audio.wav'
            if not audio.is_file():
                abort(409, description='Meeting audio is unavailable.')
            vectors = extract(backend, audio, clips)
            store.remember(pid, make_profile(backend.model, vectors, clips), body['consent'])
            return jsonify(remembered=True)

    @app.post('/api/jobs/<jid>/speakers/<track>/voice-suggestion')
    def voice_suggestion(jid, track):
        body = request.get_json()
        with lock:
            backend, calibration, available = voice_runtime()
            job = editing_job(jid, body, track)
            if not available:
                abort(409, description='Voice recognition is not available in this build.')
            if voice_busy():
                abort(409, description='Finish the active recording or transcription before checking voices.')
            if job.get('speaker_assignments', {}).get(track):
                raise ValueError('This speaker already has a confirmed name.')
            clips = clean_clips(job, track, body.get('segment_ids'), calibration.minimum_clips)
            audio = folder(jid)/'audio.wav'
            if not audio.is_file():
                abort(409, description='Meeting audio is unavailable.')
            match = propose_match(backend.model, calibration, extract(backend, audio, clips), store.profiles())
            job['voice_suggestions'] = [s for s in job.get('voice_suggestions', []) if s['track_id'] != track]
            if match:
                job['voice_suggestions'].append({'id': uuid.uuid4().hex, 'kind': 'voice_match', 'track_id': track,
                    'person_id': match['person_id'], 'name': store.get(match['person_id'])['name'],
                    'match': match, 'evidence': {'clips': [c.payload() for c in clips]}})
            job['revision'] += 1
            put(job)
            return jsonify(job=job, matched=match is not None)
