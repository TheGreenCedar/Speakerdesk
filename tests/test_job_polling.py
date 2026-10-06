"""CPU API/storage contracts; no audio, listener, capture or model runtime."""
import copy
from contextlib import contextmanager
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from app import create_app


class JobPollingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(self.root)
        self.client = self.app.test_client()
        self.manager = self.app.extensions['speakerdesk']['meetings']
        token = re.search(r'name="speakerdesk-token" content="([^"]+)"',
                          self.client.get('/').get_data(as_text=True)).group(1)
        self.headers = {'X-Speakerdesk-Token': token}
        self.jid = 'a' * 32
        self.seed = {'id': self.jid, 'name': 'Synthetic meeting', 'kind': 'meeting',
                     'created': 1., 'updated': 1., 'status': 'ready', 'message': 'Saved.',
                     'language': 'en', 'duration': 10., 'revision': 7,
                     'document': {'schema_version': 1, 'speakers': {'speaker_0': 'Speaker 1'},
                                  'segments': [{'id': 'one', 'start': 0., 'end': 1.,
                                                'speaker': 'speaker_0', 'text': 'Keep these words.',
                                                'machine_revision': 2, 'protected_fields': ['text'],
                                                'canonical_utterance_id': 'canonical-one',
                                                'start_sample': 0, 'end_sample': 16000,
                                                'audio_revision': 4}],
                                  'warnings': [], 'provenance': {'kind': 'local_inference'}},
                     'fast_history': {'old': {'text': 'x' * 20000}},
                     'refinement_history': {'old': {'text': 'y' * 20000}}}
        with self.db() as conn:
            conn.execute('INSERT INTO jobs VALUES (?,?)', (self.jid, json.dumps(self.seed)))
        (self.root / self.jid).mkdir()

    def tearDown(self):
        self.close(self.app)
        self.temp.cleanup()

    @staticmethod
    def close(app):
        app.extensions['speakerdesk']['meetings'].close()
        app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.root / 'jobs.sqlite')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def raw(self):
        with self.db() as conn:
            return conn.execute('SELECT payload FROM jobs WHERE id=?', (self.jid,)).fetchone()[0]

    def detail(self, query=''):
        return self.client.get(f'/api/jobs/{self.jid}{query}')

    def test_list_is_compact_sorted_and_legacy_sql_updates_are_visible(self):
        second = {**self.seed, 'id': 'b' * 32, 'created': 2., 'name': 'Second'}
        with self.db() as conn:
            conn.execute('INSERT INTO jobs VALUES (?,?)', (second['id'], json.dumps(second)))
            changed = {**self.seed, 'name': 'Renamed', 'status': 'failed'}
            conn.execute('UPDATE jobs SET payload=? WHERE id=?', (json.dumps(changed), self.jid))
        response = self.client.get('/api/jobs')
        self.assertEqual([row['id'] for row in response.json], [second['id'], self.jid])
        expected = {'id', 'name', 'kind', 'created', 'updated', 'status',
                    'message', 'language', 'duration', 'revision'}
        self.assertEqual(set(response.json[1]), expected)
        self.assertEqual(response.json[1]['name'], 'Renamed')
        self.assertEqual(response.json[1]['status'], 'failed')
        self.assertLess(len(response.data), 1500)
        self.assertEqual(json.loads(self.raw())['fast_history'], self.seed['fast_history'])

    def test_matching_revision_retains_fresh_metadata_without_document(self):
        self.manager.patch(self.jid, status='paused', language='fr', language_revision=3,
                           pause_flush={'request_id': 'request-one', 'state': 'pending'})
        self.manager.patch(self.jid, duration=12.25)
        self.manager.jid = self.jid
        response = self.detail('?known_revision=7')
        self.assertEqual(response.status_code, 200)
        value = response.json
        self.assertIs(value['unchanged'], True)
        self.assertNotIn('document', value)
        self.assertEqual((value['revision'], value['duration'], value['status'], value['language']),
                         (7, 12.25, 'paused', 'fr'))
        self.assertEqual(value['language_revision'], 3)
        self.assertEqual(value['pause_flush'], {'request_id': 'request-one', 'state': 'pending'})
        self.assertIs(value['inference_owned'], True)
        legacy = self.detail().json
        self.assertNotIn('unchanged', legacy)
        self.assertEqual(legacy['document'], self.seed['document'])
        self.assertEqual(legacy['duration'], 12.25)
        self.assertNotIn('fast_history', legacy)
        self.assertNotIn('refinement_history', legacy)

    def test_passage_edit_changes_revision_and_conditional_document_with_cas(self):
        response = self.client.patch(f'/api/jobs/{self.jid}/segments/one', headers=self.headers,
                                     json={'segment_revision': 2, 'changes': {'text': 'My correction.'}})
        self.assertEqual(response.status_code, 200)
        next_detail = self.detail('?known_revision=7').json
        self.assertIs(next_detail['unchanged'], False)
        self.assertEqual(next_detail['revision'], 8)
        row = next_detail['document']['segments'][0]
        self.assertEqual(row['text'], 'My correction.')
        self.assertEqual(row['canonical_utterance_id'], 'canonical-one')
        self.assertEqual(row['audio_revision'], 4)
        self.assertEqual(self.client.patch(f'/api/jobs/{self.jid}/segments/one', headers=self.headers,
                         json={'segment_revision': 2, 'changes': {'text': 'Stale.'}}).status_code, 409)
        self.assertIs(self.detail('?known_revision=8').json['unchanged'], True)
        exported = self.client.get(f'/api/jobs/{self.jid}/export/json')
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.json['segments'][0]['text'], 'My correction.')

    def test_duration_ticks_do_not_rewrite_job_or_revision_and_ignore_older_ticks(self):
        before = self.raw()
        self.manager.patch(self.jid, duration=12.5)
        first_updated = self.detail().json['updated']
        self.manager.patch(self.jid, duration=11.)
        self.manager.patch(self.jid, duration=12.5)
        self.assertEqual(self.raw(), before)
        current = self.detail().json
        self.assertEqual((current['duration'], current['revision'], current['updated']),
                         (12.5, 7, first_updated))
        self.assertEqual(self.client.get('/api/jobs').json[0]['duration'], 12.5)
        self.assertEqual(self.manager.get(self.jid)['document'], self.seed['document'])

    def test_stale_full_write_folds_latest_checkpoint_and_delete_cleans_projections(self):
        stale = self.manager.get(self.jid)
        self.manager.patch(self.jid, duration=13.)
        stale['message'] = 'Still saved.'
        self.manager.put(stale)
        stored = json.loads(self.raw())
        self.assertEqual(stored['duration'], 13.)
        self.assertEqual(stored['document'], self.seed['document'])
        self.assertEqual(stored['refinement_history'], self.seed['refinement_history'])
        with self.db() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM job_progress').fetchone()[0], 0)
        self.assertEqual(self.client.delete(f'/api/jobs/{self.jid}', headers=self.headers).status_code, 200)
        with self.db() as conn:
            for table in ('jobs', 'job_metadata', 'job_progress'):
                self.assertEqual(conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0], 0)

    def test_restart_recovers_latest_duration_and_interrupted_refinement(self):
        self.manager.patch(self.jid, status='recording', refinement_status='refining',
                           rolling_refinement={'version': 1, 'completed_sample': 0, 'pending': [],
                                               'last_scheduled_at': 0., 'cancelled': False},
                           rolling_inflight={'operation_id': 'interrupted'})
        self.manager.patch(self.jid, duration=14.)
        reopened = create_app(self.root)
        try:
            current = reopened.test_client().get(f'/api/jobs/{self.jid}').json
            self.assertEqual(current['status'], 'failed')
            self.assertEqual(current['duration'], 14.)
            self.assertEqual(current['refinement_status'], 'paused')
            self.assertEqual(current['document'], self.seed['document'])
            stored = json.loads(self.raw())
            self.assertEqual(stored['duration'], 14.)
            self.assertNotIn('rolling_inflight', stored)
            self.assertEqual(stored['fast_history'], self.seed['fast_history'])
            with self.db() as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM job_progress').fetchone()[0], 0)
        finally:
            self.close(reopened)

    def test_original_database_migrates_without_changing_ready_document_or_history(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            conn = sqlite3.connect(root / 'jobs.sqlite')
            try:
                with conn:
                    conn.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
                    conn.execute('INSERT INTO jobs VALUES (?,?)', (self.jid, json.dumps(self.seed)))
            finally:
                conn.close()
            migrated = create_app(root)
            try:
                client = migrated.test_client()
                self.assertEqual(client.get(f'/api/jobs/{self.jid}').json['document'], self.seed['document'])
                self.assertIs(client.get(f'/api/jobs/{self.jid}?known_revision=7').json['unchanged'], True)
                self.assertLess(len(client.get('/api/jobs').data), 750)
                self.assertEqual(migrated.extensions['speakerdesk']['meetings'].get(self.jid)['fast_history'],
                                 self.seed['fast_history'])
            finally:
                self.close(migrated)

    def test_projection_failure_rolls_back_full_job_and_pending_checkpoint(self):
        self.manager.patch(self.jid, duration=15.)
        before = self.raw()
        with self.db() as conn:
            conn.execute("CREATE TRIGGER synthetic_failure BEFORE INSERT ON job_metadata "
                         "BEGIN SELECT RAISE(ABORT,'synthetic metadata failure'); END")
        proposed = copy.deepcopy(self.seed)
        proposed['document']['segments'][0]['text'] = 'Must not commit.'
        with self.assertRaises(sqlite3.IntegrityError):
            self.manager.put(proposed)
        self.assertEqual(self.raw(), before)
        current = self.detail('?known_revision=7').json
        self.assertEqual(current['duration'], 15.)
        self.assertIs(current['unchanged'], True)
        self.assertEqual(self.detail().json['document'], self.seed['document'])

    def test_malformed_revision_and_unknown_job_are_rejected(self):
        for value in ('-1', 'NaN', '1.5', '', '9' * 21, '١'):
            self.assertEqual(self.detail('?known_revision=' + value).status_code, 400)
        self.assertEqual(self.client.get('/api/jobs/' + 'c' * 32 + '?known_revision=7').status_code, 404)


if __name__ == '__main__':
    unittest.main()
