"""Produce legacy v0.6.6 database fixtures under tests/goldens/legacy-db/.

This is a regeneration tool, not part of build/CI/tests. It runs the real
packaged v0.6.6 sidecar (/Applications/Speakerdesk.app) against an isolated
temporary SPEAKERDESK_HOME whose models/ directory is an APFS clone of the
installed model set. It never touches the real app data directory beyond
reading that one models folder.

System Python 3 (stdlib only), run from the repo root:
    python3 tools/oracle/make_legacy_fixtures.py [fixture ...]

Fixtures: import-ready, manual-transcript, active-run, meeting-canonical,
people-voice. Default is all of them.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.cache' / 'legacy-fixtures'
OUT = ROOT / 'tests' / 'goldens' / 'legacy-db'
RUNTIME = '/Applications/Speakerdesk.app/Contents/MacOS/speakerdesk-runtime'
REAL_MODELS = (Path.home() / 'Library' / 'Application Support'
               / 'com.speakerdesk.desktop' / 'models')
REPLAY_SOURCE = ROOT / '.cache' / 'v066' / 'scripts' / 'replay_capture.py'
AUDIO = WORK / 'audio'

# Two clearly distinct bundled voices for diarization.
VOICE_A = 'Samantha (English (US))'
VOICE_B = 'Fred'

SCRIPT = [
    (VOICE_A, 'Welcome to the planning review. Today we are covering the '
              'release schedule for the next two weeks.'),
    (VOICE_B, 'Thanks. I want to start with the migration timeline, because '
              'the porting work is ahead of where we expected.'),
    (VOICE_A, 'Agreed. The inference pipeline moved over cleanly, and the '
              'remaining work is mostly in the storage layer.'),
    (VOICE_B, 'Good. Then let us also talk about fixtures and golden tests '
              'before we wrap up this review.'),
]

LONG_SCRIPT = [
    (VOICE_A, 'This is a deliberately long synthetic recording. It keeps one '
              'speaker talking so the transcription job stays busy.'),
    (VOICE_B, 'The second speaker adds a few words here so diarization has '
              'more than one voice to separate during the run.'),
] * 3

# One continuous voice: clean single-speaker envelopes qualify as voice clips.
VOICE_SCRIPT = [
    (VOICE_A, 'Hello, my name is Sam, and I am recording a short voice sample '
              'for the enrollment fixture.'),
    (VOICE_A, 'The purpose of this sample is to give the recognizer a few '
              'clean passages of one speaker talking without interruption.'),
    (VOICE_A, 'Each sentence is spoken clearly so the passages stay eligible '
              'as separate voice clips.'),
    (VOICE_A, 'Here is one more passage so the enrollment has enough '
              'independent windows to choose from.'),
]


def log(message):
    print(f'[fixtures] {message}', flush=True)


def say_clip(voice, text, destination):
    """Render one sentence with macOS say and convert to 16 kHz mono WAV."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / 'clip.aiff'
        subprocess.run(['say', '-v', voice, '-o', str(aiff), text],
                       check=True, capture_output=True)
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000',
                        '-c', '1', str(aiff), str(destination)],
                       check=True, capture_output=True)


def pcm_frames(path):
    with wave.open(str(path), 'rb') as stream:
        assert (stream.getnchannels(), stream.getsampwidth(),
                stream.getframerate()) == (1, 2, 16000)
        return stream.readframes(stream.getnframes())


def write_wav(path, frames):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(frames)


def ensure_audio():
    """Synthesize the fixture WAVs once (say + afconvert + wave splice)."""
    AUDIO.mkdir(parents=True, exist_ok=True)
    targets = {
        'import.wav': SCRIPT,       # ~30 s, two alternating voices
        'long.wav': LONG_SCRIPT,    # ~90 s for the mid-run kill fixture
        'replay.wav': SCRIPT,
        'voice.wav': VOICE_SCRIPT,  # ~30 s, one continuous voice for clips
    }
    for name, script in targets.items():
        target = AUDIO / name
        if target.is_file():
            continue
        frames = []
        for i, (voice, text) in enumerate(script):
            clip = AUDIO / f'.clip-{i}.wav'
            say_clip(voice, text, clip)
            clip_frames = pcm_frames(clip)
            if name == 'long.wav' and i % 2 == 0:
                clip_frames *= 3  # stretch speaker A turns for a longer run
            frames.append(clip_frames)
            # Keep gaps short so consecutive same-voice turns stay separate
            # envelopes rather than bridged regions.
            frames.append(b'\x00' * 2 * 8000)  # 0.5 s of silence between turns
            clip.unlink()
        write_wav(target, b''.join(frames))
        log(f'{name}: {len(pcm_frames(target)) // 32000}s synthesized')


class Runtime:
    """One packaged v0.6.6 sidecar process with an isolated SPEAKERDESK_HOME."""

    def __init__(self, name, *, capture_helper=None, replay_dir=None):
        self.name = name
        self.home = WORK / f'home-{name}'
        if self.home.exists():
            shutil.rmtree(self.home)
        self.home.mkdir(parents=True)
        subprocess.run(['cp', '-cR', str(REAL_MODELS), str(self.home / 'models')],
                       check=True)
        env = dict(os.environ)
        env['SPEAKERDESK_HOME'] = str(self.home)
        if capture_helper:
            env['SPEAKERDESK_CAPTURE_HELPER'] = str(capture_helper)
        if replay_dir:
            env['SPEAKERDESK_REPLAY_DIR'] = str(replay_dir)
        # The PyInstaller bootloader forks the real server; put the tree in
        # its own process group so a kill actually reaches every process.
        self.proc = subprocess.Popen(
            [RUNTIME], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, env=env,
            start_new_session=True)
        self.url = None
        deadline = time.monotonic() + 90
        for line in self.proc.stdout:
            match = re.match(r'SPEAKERDESK_URL=(\S+)', line.strip())
            if match:
                self.url = match.group(1)
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f'{name}: runtime never printed SPEAKERDESK_URL')
        if not self.url:
            raise RuntimeError(f'{name}: runtime exited before SPEAKERDESK_URL')
        page = urllib.request.urlopen(f'{self.url}/', timeout=10).read().decode()
        self.token = re.search(r'name="speakerdesk-token" content="([^"]+)"',
                               page).group(1)
        log(f'{name}: runtime up at {self.url} (home {self.home})')

    def api(self, method, path, body=None, files=None, timeout=30):
        headers = {'X-Speakerdesk-Token': self.token}
        data = None
        if files:
            boundary = '----oracle-boundary'
            chunks = []
            for field, value in (body or {}).items():
                chunks.append(f'--{boundary}\r\nContent-Disposition: '
                              f'form-data; name="{field}"\r\n\r\n{value}\r\n')
            for field, filename, content in files:
                chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; '
                              f'name="{field}"; filename="{filename}"\r\n'
                              f'Content-Type: audio/wav\r\n\r\n')
                chunks.append(content)
                chunks.append('\r\n')
            chunks.append(f'--{boundary}--\r\n')
            data = b''.join(part if isinstance(part, bytes)
                            else part.encode() for part in chunks)
            headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        elif body is not None:
            data = json.dumps(body).encode()
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(
            f'{self.url}{path}', data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b'null')
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                return exc.code, json.loads(payload or b'null')
            except json.JSONDecodeError:
                return exc.code, {'raw': payload.decode(errors='replace')}

    def job(self, jid):
        status, body = self.api('GET', f'/api/jobs/{jid}')
        assert status == 200, (status, body)
        return body

    def wait_status(self, jid, wanted, *, timeout=600, interval=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.job(jid)
            if job['status'] in wanted:
                return job
            if job['status'] == 'failed':
                raise RuntimeError(f'{self.name}: job {jid} failed: '
                                   f'{job.get("message")}')
            time.sleep(interval)
        raise RuntimeError(f'{self.name}: job {jid} never reached {wanted}')

    def upload(self, wav_path, *, name, language='en'):
        status, body = self.api(
            'POST', '/api/jobs', body={'language': language},
            files=[('files', name, wav_path.read_bytes())])
        assert status == 201, (status, body)
        return body[0]['id']

    def stop(self, sig=None, grace=10):
        if sig is not None:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass
        else:
            self.proc.stdin.close()
            self.proc.terminate()
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, 9)
            except ProcessLookupError:
                pass
            self.proc.kill()
            self.proc.wait(timeout=10)
        rest = self.proc.stdout.read()
        if rest.strip():
            log(f'{self.name}: runtime tail: {rest.strip()[-400:]}')
        log(f'{self.name}: runtime exited rc={self.proc.returncode}')

    def publish(self, fixture):
        """Copy recordings/jobs.sqlite into tests/goldens/legacy-db/."""
        database = self.home / 'recordings' / 'jobs.sqlite'
        assert database.is_file(), f'{database} missing'
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / f'{fixture}.sqlite'
        shutil.copyfile(database, target)
        with sqlite3.connect(target) as conn:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return target


def scan_sensitive(target):
    """Report any temp-home paths or usernames persisted in the fixture DB."""
    findings = []
    with sqlite3.connect(target) as conn:
        for (table,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                for value in row:
                    if isinstance(value, str):
                        if str(WORK) in value:
                            findings.append((table, 'temp-home path'))
                        if str(Path.home()) in value and 'Application Support' \
                                in value:
                            findings.append((table, 'home-dir path'))
    return findings


def row_counts(target):
    counts = {}
    with sqlite3.connect(target) as conn:
        for (table,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name"):
            counts[table] = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    return counts


def report(target):
    import hashlib
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    log(f'{target.name}: sha256={digest} size={target.stat().st_size} '
        f'rows={row_counts(target)} sensitive={scan_sensitive(target)}')


def fixture_import_ready():
    rt = Runtime('import')
    try:
        started = time.monotonic()
        jid = rt.upload(AUDIO / 'import.wav', name='planning-review.wav')
        rt.wait_status(jid, ('uploaded',))
        status, body = rt.api('POST', f'/api/jobs/{jid}/run')
        assert status == 202, (status, body)
        job = rt.wait_status(jid, ('ready',))
        log(f'import: inference took {time.monotonic() - started:.0f}s; '
            f'{len(job["document"]["segments"])} segments')
        document = job['document']
        row = document['segments'][0]
        row['text'] = row['text'] + ' Edited by fixture.'
        status, body = rt.api('PUT', f'/api/jobs/{jid}/transcript',
                              {'revision': job['revision'],
                               'document': document})
        assert status == 200, (status, body)
        segment = document['segments'][1] if len(document['segments']) > 1 \
            else None
        if segment:
            status, body = rt.api(
                'PATCH', f'/api/jobs/{jid}/segments/{segment["id"]}',
                {'segment_revision': segment.get('machine_revision', 0),
                 'changes': {'speaker': document['speakers'] and segment['speaker']}})
            assert status == 200, (status, body)
        rt.stop()
    finally:
        if rt.proc.poll() is None:
            rt.stop()
    report(rt.publish('import-ready'))


def fixture_manual_transcript():
    rt = Runtime('manual')
    try:
        manual = rt.upload(AUDIO / 'import.wav', name='typed-minutes.wav')
        rt.wait_status(manual, ('uploaded',))
        status, body = rt.api('PUT', f'/api/jobs/{manual}/transcript', {
            'revision': rt.job(manual)['revision'],
            'document': {'speakers': {'speaker_0': 'Person one'},
                         'segments': [
                             {'speaker': 'speaker_0', 'start': 1.0, 'end': 5.0,
                              'text': 'A manually typed passage.'}]}})
        assert status == 200, (status, body)
        imported = rt.upload(AUDIO / 'import.wav', name='pasted-notes.wav')
        rt.wait_status(imported, ('uploaded',))
        status, body = rt.api('PUT', f'/api/jobs/{imported}/transcript', {
            'revision': rt.job(imported)['revision'], 'imported': True,
            'document': {'speakers': {'speaker_0': 'Person one'},
                         'segments': [
                             {'speaker': 'speaker_0', 'start': 2.0, 'end': 6.0,
                              'text': 'An imported passage from notes.'}]}})
        assert status == 200, (status, body)
        rt.stop()
    finally:
        if rt.proc.poll() is None:
            rt.stop()
    report(rt.publish('manual-transcript'))


def fixture_active_run():
    rt = Runtime('active')
    try:
        jid = rt.upload(AUDIO / 'long.wav', name='long-recording.wav')
        rt.wait_status(jid, ('uploaded',))
        status, body = rt.api('POST', f'/api/jobs/{jid}/run')
        assert status == 202, (status, body)
        rt.wait_status(jid, ('processing',))
        time.sleep(8)
        log(f'active: killing runtime mid-run (job status '
            f'{rt.job(jid)["status"]!r})')
        rt.stop(sig=9, grace=15)
    finally:
        if rt.proc.poll() is None:
            rt.stop(sig=9, grace=15)
    report(rt.publish('active-run'))


def replay_home(name):
    """Prepare an executable copy of the v0.6.6 synthetic capture helper."""
    replay = WORK / 'replay'
    replay.mkdir(parents=True, exist_ok=True)
    helper = replay / 'replay_capture.py'
    if not helper.exists():
        shutil.copyfile(REPLAY_SOURCE, helper)
        helper.chmod(0o755)
    shutil.copyfile(AUDIO / 'replay.wav', replay / 'replay.wav')
    (replay / 'replay.json').write_text('{}')
    held = replay / 'stage-end'
    if held.exists():
        held.unlink()
    return replay, helper


def drive_meeting(rt, replay):
    """Record a full canonical meeting through the replay helper."""
    status, body = rt.api('POST', '/api/meetings',
                          {'name': 'Replay Meeting', 'language': 'en',
                           'sources': ['microphone']})
    assert status == 201, (status, body)
    jid = body['id']
    held = replay / 'stage-end'
    deadline = time.monotonic() + 120
    while not held.exists() and time.monotonic() < deadline:
        time.sleep(1)
    assert held.exists(), 'replay helper never reached the end boundary'
    status, body = rt.api('POST', f'/api/meetings/{jid}/stop')
    assert status == 202, (status, body)
    job = rt.wait_status(jid, ('ready',), timeout=600)
    log(f'meeting: finalized {job["status"]}; '
        f'{len(job.get("document", {}).get("segments", []))} segments')
    return job


def fixture_meeting():
    replay, helper = replay_home('meeting')
    rt = Runtime('meeting', capture_helper=helper, replay_dir=replay)
    try:
        drive_meeting(rt, replay)
        rt.stop()
    finally:
        if rt.proc.poll() is None:
            rt.stop()
    report(rt.publish('meeting-canonical'))


def fixture_people_voice():
    # Enrollment needs verified single-speaker audio; canonical meeting
    # segments carry the sealed speaker-activity evidence imports lack.
    replay, helper = replay_home('people')
    rt = Runtime('people', capture_helper=helper, replay_dir=replay)
    try:
        job = drive_meeting(rt, replay)
        jid = job['id']
        status, people = rt.api('GET', '/api/people')
        assert status == 200, (status, people)
        log(f'people: voice_available={people.get("voice_available")} '
            f'message={people.get("voice_message")!r}')
        status, person = rt.api('POST', '/api/people',
                                {'name': 'Fixture Person'})
        assert status == 201, (status, person)
        pid = person['id']
        enrolled = False
        for track in sorted(job['document']['speakers']):
            if not track.startswith('speaker_'):
                continue
            status, body = rt.api(
                'POST', f'/api/jobs/{jid}/speakers/{track}/identity',
                {'revision': rt.job(jid)['revision'], 'person_id': pid})
            assert status == 200, (status, body)
            if not people.get('voice_available') or enrolled:
                continue
            status, clips = rt.api(
                'GET', f'/api/jobs/{jid}/speakers/{track}/voice-clips')
            assert status == 200, (status, clips)
            log(f'people: {track} clip status {clips.get("status")} '
                f'recommended={clips.get("recommended_ids")}')
            if clips.get('status') == 'ready':
                status, body = rt.api('POST', f'/api/people/{pid}/voice', {
                    'consent': True, 'meeting_id': jid, 'track_id': track,
                    'revision': rt.job(jid)['revision'],
                    'segment_ids': clips['recommended_ids']})
                assert status == 200, (status, body)
                log(f'people: voice enrolled from {track}')
                enrolled = True
        if not enrolled:
            log('people: WARNING no track produced usable voice clips')
        status, body = rt.api('PATCH', '/api/preferences/language',
                              {'language': 'de'})
        assert status == 200, (status, body)
        rt.stop()
    finally:
        if rt.proc.poll() is None:
            rt.stop()
    report(rt.publish('people-voice'))


FIXTURES = {
    'import-ready': fixture_import_ready,
    'manual-transcript': fixture_manual_transcript,
    'active-run': fixture_active_run,
    'meeting-canonical': fixture_meeting,
    'people-voice': fixture_people_voice,
}


def main(argv):
    if not Path(RUNTIME).is_file():
        raise SystemExit(f'Packaged v0.6.6 runtime missing: {RUNTIME}')
    if not REAL_MODELS.is_dir():
        raise SystemExit(f'Installed models missing: {REAL_MODELS}')
    ensure_audio()
    wanted = argv or list(FIXTURES)
    for name in wanted:
        log(f'=== {name} ===')
        FIXTURES[name]()
    log('done')


if __name__ == '__main__':
    main(sys.argv[1:])
