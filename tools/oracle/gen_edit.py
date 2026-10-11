"""Generate edit goldens through the real v0.6.6 Flask app.

Each case seeds jobs directly into the app's SQLite store (the same INSERT
the upload route performs), drives the request through the Flask test client
with the CSRF token from GET /, and records the status code, JSON body, and
stored job payload. time.time is frozen so server-set timestamps
(edited_at, accepted_at) are deterministic.

    PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_edit.py
"""
import copy
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_lib as oracle  # noqa: E402

oracle.activate_oracle()
import app as sidecar  # noqa: E402

oracle.imported_from_worktree(sidecar)

FIXED_TIME = 1700000000.0
READY_JOB = 'e' * 31 + '1'
MANUAL_JOB = 'e' * 31 + '2'
ACTIVE_JOB = 'e' * 31 + '3'


def ready_document():
    return {'schema_version': 1,
            'speakers': {'speaker_0': 'Ada', 'speaker_1': 'Ben',
                         'unassigned': 'Unknown speaker'},
            'segments': [
                {'id': 'seg-0', 'speaker': 'speaker_0', 'start': 0.0, 'end': 8.0,
                 'text': 'First passage here.', 'machine_revision': 0,
                 'refinement_state': 'final',
                 'decode_core_plan': {'plan': 'server-owned'},
                 'text_audio_anchor': {'start_sample': 0,
                                       'end_sample': 128000},
                 'alignment': {'score': 0.99},
                 'assembly_provenance': {'kind': 'server'},
                 'bounded_decode_provenance': {'kind': 'server'},
                 'reading_turns': [{'turn': 1}],
                 'reading_turn_provenance': {'turn': 1},
                 'voice_eligible': True,
                 'speaker_candidates': ['speaker_0']},
                {'id': 'seg-1', 'speaker': 'speaker_1', 'start': 9.0,
                 'end': 15.0, 'text': 'Second passage here.',
                 'machine_revision': 2, 'protected_fields': ['text'],
                 'refinement_state': 'edited',
                 'retry_candidate': {'id': 'retry-1', 'language': 'en',
                                     'original_text': 'Second passage here.',
                                     'start': 9.0, 'end': 15.0,
                                     'speaker': 'speaker_1',
                                     'created': 1600000000.0,
                                     'text': 'Corrected retry words.'}},
                {'id': 'seg-2', 'speaker': 'unassigned', 'start': 16.0,
                 'end': 20.0, 'text': 'Unlabeled speech.'}]}


def seed_jobs():
    return {
        READY_JOB: {'id': READY_JOB, 'name': 'Seed Recording', 'kind': 'import',
                    'source_file': 'source.wav', 'status': 'ready',
                    'message': 'Saved locally.', 'language': 'en',
                    'duration': 30.0, 'created': FIXED_TIME,
                    'updated': FIXED_TIME, 'revision': 2,
                    'document': ready_document()},
        MANUAL_JOB: {'id': MANUAL_JOB, 'name': 'Manual Entry', 'kind': 'import',
                     'source_file': 'source.wav', 'status': 'uploaded',
                     'message': 'Ready for local inference or manual editing.',
                     'language': 'en', 'duration': 12.0,
                     'created': FIXED_TIME, 'updated': FIXED_TIME,
                     'revision': 0, 'document': None},
        ACTIVE_JOB: {'id': ACTIVE_JOB, 'name': 'Still Running',
                     'kind': 'import', 'source_file': 'source.wav',
                     'status': 'processing',
                     'message': 'Starting local inference…',
                     'language': 'en', 'duration': 30.0,
                     'created': FIXED_TIME, 'updated': FIXED_TIME,
                     'revision': 0, 'document': ready_document()},
    }


def edited_document(edits):
    document = ready_document()
    for segment_id, changes in edits.items():
        row = next(s for s in document['segments'] if s['id'] == segment_id)
        row.update(changes)
    return document


def run_case(name, seeds, method, path, body):
    """Boot a fresh app, seed jobs, fire one request, record everything."""
    with tempfile.TemporaryDirectory(prefix='oracle-edit-') as data:
        app = sidecar.create_app(data)
        client = app.test_client()
        token = re.search(r'name="speakerdesk-token" content="([^"]+)"',
                          client.get('/').get_data(as_text=True)).group(1)
        database = Path(data) / 'jobs.sqlite'
        with sqlite3.connect(database) as conn:
            conn.executemany(
                'INSERT INTO jobs VALUES (?,?)',
                [(job['id'], json.dumps(job)) for job in seeds.values()])
        headers = {'X-Speakerdesk-Token': token}
        with mock.patch('time.time', return_value=FIXED_TIME):
            response = client.open(path, method=method, json=body,
                                   headers=headers)
        stored = {}
        with sqlite3.connect(database) as conn:
            for job_id in seeds:
                row = conn.execute('SELECT payload FROM jobs WHERE id=?',
                                   (job_id,)).fetchone()
                stored[job_id] = json.loads(row[0]) if row else None
        oracle.dump_golden(
            'edit', name,
            {'seed_jobs': seeds,
             'request': {'method': method, 'path': path, 'body': body}},
            {'status': response.status_code,
             'body': response.get_json(),
             'stored_jobs': stored})


def run():
    jobs = seed_jobs

    run_case(
        'put-text-edit', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 2,
         'document': edited_document(
             {'seg-0': {'text': 'Edited first words.'}})})

    forged = edited_document({'seg-1': {
        'machine_revision': 99, 'protected_fields': ['start', 'speaker'],
        'refinement_state': 'invented', 'alignment': {'forged': True},
        'text_audio_anchor': {'forged': True},
        'canonical_state': 'forged', 'audio_revision': 7,
        'voice_eligible': True, 'speaker_candidates': ['forged'],
        'retry_candidate': {'forged': True}}})
    run_case(
        'put-forged-server-fields', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 2, 'document': forged})

    run_case(
        'put-stale-revision', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 1, 'document': edited_document({})})

    run_case(
        'put-retry-candidate-accepted', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 2,
         'document': edited_document(
             {'seg-1': {'text': 'Corrected retry words.'}})})

    run_case(
        'put-retry-candidate-retained', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 2,
         'document': edited_document(
             {'seg-1': {'text': 'A different human edit.'}})})

    run_case(
        'put-imported', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 2, 'document': edited_document({}), 'imported': True})

    run_case(
        'put-manual-document', {MANUAL_JOB: jobs()[MANUAL_JOB]},
        'PUT', f'/api/jobs/{MANUAL_JOB}/transcript',
        {'revision': 0,
         'document': {'speakers': {'speaker_0': 'Ada'},
                      'segments': [{'speaker': 'speaker_0', 'start': 1.0,
                                    'end': 5.0,
                                    'text': 'Typed in by a person.'}]}})

    invalid = edited_document({'seg-0': {'start': 12.0, 'end': 10.0}})
    run_case(
        'put-invalid-document', {READY_JOB: jobs()[READY_JOB]},
        'PUT', f'/api/jobs/{READY_JOB}/transcript',
        {'revision': 2, 'document': invalid})

    run_case(
        'put-active-job', {ACTIVE_JOB: jobs()[ACTIVE_JOB]},
        'PUT', f'/api/jobs/{ACTIVE_JOB}/transcript',
        {'revision': 0, 'document': edited_document({})})

    run_case(
        'patch-passage-merge', {READY_JOB: jobs()[READY_JOB]},
        'PATCH', f'/api/jobs/{READY_JOB}/segments/seg-0',
        {'segment_revision': 0, 'changes': {'text': 'Rewritten by patch.'}})

    run_case(
        'patch-speaker-change', {READY_JOB: jobs()[READY_JOB]},
        'PATCH', f'/api/jobs/{READY_JOB}/segments/seg-1',
        {'segment_revision': 2, 'changes': {'speaker': 'speaker_0'}})

    run_case(
        'patch-stale-segment', {READY_JOB: jobs()[READY_JOB]},
        'PATCH', f'/api/jobs/{READY_JOB}/segments/seg-1',
        {'segment_revision': 0, 'changes': {'text': 'Out of date.'}})

    run_case(
        'patch-forbidden-field', {READY_JOB: jobs()[READY_JOB]},
        'PATCH', f'/api/jobs/{READY_JOB}/segments/seg-0',
        {'segment_revision': 0, 'changes': {'language': 'de'}})

    run_case(
        'patch-unknown-segment', {READY_JOB: jobs()[READY_JOB]},
        'PATCH', f'/api/jobs/{READY_JOB}/segments/nope',
        {'segment_revision': 0, 'changes': {'text': 'x'}})

    run_case(
        'patch-active-job', {ACTIVE_JOB: jobs()[ACTIVE_JOB]},
        'PATCH', f'/api/jobs/{ACTIVE_JOB}/segments/seg-0',
        {'segment_revision': 0, 'changes': {'text': 'x'}})


if __name__ == '__main__':
    run()
