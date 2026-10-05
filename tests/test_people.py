"""CPU-only identity/persistence contracts; synthetic vectors, no model or device access."""
import copy
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from live_meeting import append_finalized_segment
from people import introduction_suggestions
from voice_profiles import Calibration, ClipEmbedding, VoiceModel, propose_match

MODEL = VoiceModel('synthetic-test-model', 'test-revision', 'a'*64, 2)
POLICY = Calibration(MODEL, 'synthetic-test-calibration', .8, .1, 20, 100, 0., .05)


class SyntheticBackend:
    model = MODEL

    def __init__(self):
        self.calls = []
        self.vector = (1., 0.)
        self.clean = True

    def embed(self, audio, clip):
        self.calls.append((audio, clip))
        return ClipEmbedding(self.vector, self.clean)


def document(text="I'm Priya. Welcome to the meeting."):
    return {'schema_version': 1, 'speakers': {'speaker_0': 'Speaker 1'},
            'segments': [{'id': 'intro', 'speaker': 'speaker_0', 'start': 0., 'end': 3.,
                          'text': text, 'review': False},
                         {'id': 'second', 'speaker': 'speaker_0', 'start': 4., 'end': 7.,
                          'text': 'We can start now.', 'review': False}],
            'provenance': {'kind': 'local_inference', 'timing': 'speech regions'}}


class PeopleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backend = SyntheticBackend()
        self.app = create_app(self.root, voice_backend=self.backend, voice_calibration=POLICY)
        self.client = self.app.test_client()
        html = self.client.get('/').get_data(as_text=True)
        self.headers = {'X-Speakerdesk-Token': re.search(r'name="speakerdesk-token" content="([^"]+)"', html)[1]}
        self.jid = 'a'*32
        self.seed(self.jid)

    def close_app(self, app):
        app.extensions['speakerdesk']['meetings'].close()
        app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    def tearDown(self):
        self.close_app(self.app)
        self.temp.cleanup()

    def seed(self, jid, doc=None, status='ready'):
        job = {'id': jid, 'name': 'Meeting', 'kind': 'meeting', 'status': status,
               'created': 1., 'revision': 0, 'duration': 12., 'document': doc or document()}
        with closing(sqlite3.connect(self.root/'jobs.sqlite')) as db:
            db.execute('INSERT INTO jobs VALUES (?,?)', (jid, json.dumps(job)))
            db.commit()
        dest = self.root/jid
        dest.mkdir()
        (dest/'audio.wav').write_bytes(b'synthetic test: backend never reads waveform')
        return job

    def post(self, path, body):
        return self.client.post(path, headers=self.headers, json=body)

    def person(self, name='Priya'):
        response = self.post('/api/people', {'name': name})
        self.assertEqual(response.status_code, 201)
        return response.json['id']

    def job(self, jid=None):
        return self.client.get(f'/api/jobs/{jid or self.jid}').json

    def assign(self, pid=None, jid=None, **extra):
        jid = jid or self.jid
        return self.post(f'/api/jobs/{jid}/speakers/speaker_0/identity',
                         {'revision': self.job(jid)['revision'], 'person_id': pid, 'name': 'Priya', **extra})

    def enroll(self, pid, **extra):
        return self.post(f'/api/people/{pid}/voice', {'consent': True, 'meeting_id': self.jid,
                         'track_id': 'speaker_0', 'revision': self.job()['revision'],
                         'segment_ids': ['intro', 'second'], **extra})

    def proposals(self, jid=None):
        return self.client.get(f'/api/jobs/{jid or self.jid}/identity-suggestions').json['suggestions']

    def test_name_only_people_persist_and_track_slots_do_not_identify_other_meetings(self):
        pid = self.person()
        self.assertEqual(self.assign(pid).status_code, 200)
        self.seed('b'*32)
        self.assertEqual(self.job('b'*32)['document']['speakers']['speaker_0'], 'Speaker 1')
        self.assertNotIn('speaker_assignments', self.job('b'*32))
        reopened = create_app(self.root)
        try:
            saved = reopened.test_client().get('/api/people').json
            self.assertEqual(saved['people'][0]['id'], pid)
            self.assertFalse(saved['people'][0]['voice_saved'])
            self.assertFalse(saved['voice_available'])
        finally:
            self.close_app(reopened)
        self.assertEqual(self.assign(pid, 'b'*32).status_code, 200)
        self.assertEqual(self.job()['speaker_assignments']['speaker_0']['person_id'], pid)
        self.assertEqual(self.job('b'*32)['speaker_assignments']['speaker_0']['person_id'], pid)
        self.assertEqual(self.backend.calls, [])

    def test_people_mutations_require_same_origin_token_and_names_are_validated(self):
        self.assertEqual(self.client.post('/api/people', json={'name': 'Priya'}).status_code, 403)
        for name in ['', ' '*3, None, 'a'*101, {}]:
            self.assertEqual(self.post('/api/people', {'name': name}).status_code, 400)
        self.assertEqual(self.client.get('/api/people').json['people'], [])

    def test_explicit_intro_is_a_pending_meeting_only_name_with_quote_and_time(self):
        suggestion = self.proposals()[0]
        self.assertEqual(suggestion['name'], 'Priya')
        self.assertEqual(suggestion['evidence']['quote'], "I'm Priya. Welcome to the meeting.")
        self.assertEqual(suggestion['evidence']['start'], 0.)
        self.assertEqual(self.job()['document']['speakers']['speaker_0'], 'Speaker 1')
        self.assertEqual(self.assign(suggestion_id=suggestion['id']).status_code, 200)
        job = self.job()
        self.assertEqual(job['document']['speakers']['speaker_0'], 'Priya')
        assignment = job['speaker_assignments']['speaker_0']
        self.assertIsNone(assignment['person_id'])
        self.assertEqual(assignment['source'], 'explicit_introduction')
        self.assertEqual(assignment['meeting_id'], self.jid)
        self.assertEqual(assignment['evidence']['evidence']['segment_id'], 'intro')
        self.assertEqual(self.client.get('/api/people').json['people'], [])
        self.assertEqual(self.backend.calls, [])

    def test_addressed_names_quotes_unfinalized_and_overlap_remain_unknown(self):
        cases = ['Priya, what do you think?', 'Alex said, I’m Priya.', '"I’m Priya," she said.',
                 "I'm happy to join.", 'This is Priya’s proposal.', 'Am I Priya?']
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(introduction_suggestions({'document': document(text)}), [])
        for change in [{'review': True}, {'finalized': False}, {'speaker_candidates': ['speaker_0', 'speaker_1']}]:
            doc = document(); doc['segments'][0].update(change)
            self.assertEqual(introduction_suggestions({'document': doc}), [])
        for kind in ['manual', 'imported', 'pending_inference']:
            doc = document(); doc['provenance']['kind'] = kind
            self.assertEqual(introduction_suggestions({'document': doc}), [])
        for text, name in [("Hello, I'm Priya Patel.", 'Priya Patel'), ('My name is José.', 'José'),
                           ('I’m Anne-Marie.', 'Anne-Marie')]:
            self.assertEqual(introduction_suggestions({'document': document(text)})[0]['name'], name)

    def test_stale_or_dismissed_intro_cannot_be_confirmed(self):
        suggestion = self.proposals()[0]
        response = self.post(f'/api/jobs/{self.jid}/identity-suggestions/{suggestion["id"]}/dismiss', {'revision': 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.proposals(), [])
        self.assertEqual(self.assign(suggestion_id=suggestion['id']).status_code, 409)
        self.assertEqual(self.assign(revision=0).status_code, 409)
        self.assertEqual(self.job()['document']['speakers']['speaker_0'], 'Speaker 1')

    def test_transcript_edits_invalidate_introduction_evidence(self):
        suggestion = self.proposals()[0]
        doc = self.job()['document']; doc['segments'][0]['text'] = 'Priya, what do you think?'
        response = self.client.put(f'/api/jobs/{self.jid}/transcript', headers=self.headers,
                                   json={'revision': 0, 'document': doc})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.proposals(), [])
        self.assertEqual(self.assign(suggestion_id=suggestion['id']).status_code, 409)

    def test_enrollment_requires_consent_confirmed_person_and_multiple_clean_clips(self):
        pid = self.person()
        self.assertEqual(self.enroll(pid).status_code, 400)
        self.assertEqual(self.assign(pid).status_code, 200)
        self.assertEqual(self.enroll(pid, consent=False).status_code, 400)
        for ids in [['intro'], ['intro', 'intro'], ['intro', 'missing']]:
            self.assertEqual(self.enroll(pid, segment_ids=ids).status_code, 400)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.enroll(pid).status_code, 200)
        profile = self.app.extensions['speakerdesk']['people'].profiles()[0]
        self.assertEqual(profile['person_id'], pid)
        self.assertEqual(profile['model'], MODEL.payload())
        self.assertEqual(profile['centroid'], [1., 0.])
        self.assertEqual([c['segment_id'] for c in profile['clips']], ['intro', 'second'])
        self.assertEqual(profile['clips'][0]['meeting_id'], self.jid)
        self.assertEqual(profile['clips'][0]['track_id'], 'speaker_0')
        self.assertEqual(profile['consent'], 'explicit_remember_voice')
        self.assertEqual(len(self.backend.calls), 2)
        self.assertNotIn('centroid', self.client.get('/api/people').get_data(as_text=True))

    def test_backend_quality_failure_saves_no_partial_profile(self):
        pid = self.person(); self.assign(pid)
        self.backend.clean = False
        self.assertEqual(self.enroll(pid).status_code, 400)
        self.assertEqual(self.app.extensions['speakerdesk']['people'].profiles(), [])
        self.backend.clean = True; self.backend.vector = (float('nan'), 0.)
        self.assertEqual(self.enroll(pid).status_code, 400)
        self.assertEqual(self.app.extensions['speakerdesk']['people'].profiles(), [])

    def test_overlap_and_changed_tracks_cannot_be_used_as_clean_enrollment(self):
        pid = self.person(); self.assign(pid)
        job = self.job()
        job['document']['segments'][1]['speaker_candidates'] = ['speaker_0', 'speaker_1']
        with closing(sqlite3.connect(self.root/'jobs.sqlite')) as db:
            db.execute('UPDATE jobs SET payload=? WHERE id=?', (json.dumps(job), self.jid))
            db.commit()
        self.assertEqual(self.enroll(pid).status_code, 400)
        self.assertEqual(self.backend.calls, [])

    def test_missing_approved_backend_does_not_enroll_or_download_anything(self):
        pid = self.person(); self.assign(pid)
        disabled = create_app(self.root)
        try:
            client = disabled.test_client()
            token = re.search(r'name="speakerdesk-token" content="([^"]+)"', client.get('/').get_data(as_text=True))[1]
            response = client.post(f'/api/people/{pid}/voice', headers={'X-Speakerdesk-Token': token},
                                   json={'consent': True})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(disabled.extensions['speakerdesk']['people'].profiles(), [])
        finally:
            self.close_app(disabled)
        self.assertEqual(self.backend.calls, [])

    def test_active_transcription_prevents_voice_loading_or_profile_changes(self):
        pid = self.person(); self.assertEqual(self.assign(pid).status_code, 200)
        self.seed('b'*32, status='processing')
        self.assertEqual(self.enroll(pid).status_code, 409)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.app.extensions['speakerdesk']['people'].profiles(), [])

    def test_voice_match_only_proposes_and_confirmation_does_not_update_profile(self):
        pid = self.person(); self.assign(pid); self.enroll(pid)
        store = self.app.extensions['speakerdesk']['people']
        enrolled = copy.deepcopy(store.profiles())
        other = 'b'*32; self.seed(other, document('Welcome to our next meeting.'))
        response = self.post(f'/api/jobs/{other}/speakers/speaker_0/voice-suggestion',
                             {'revision': 0, 'segment_ids': ['intro', 'second']})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['matched'])
        self.assertEqual(self.job(other)['document']['speakers']['speaker_0'], 'Speaker 1')
        self.assertEqual(store.profiles(), enrolled)
        suggestion = self.proposals(other)[0]
        self.assertEqual(suggestion['kind'], 'voice_match')
        wrong = self.person('Alex')
        self.assertEqual(self.assign(wrong, other, suggestion_id=suggestion['id']).status_code, 400)
        self.assertEqual(self.assign(pid, other, suggestion_id=suggestion['id']).status_code, 200)
        self.assertEqual(store.profiles(), enrolled)
        exported = self.client.get(f'/api/jobs/{other}/export/json').json
        evidence = exported['speaker_assignments']['speaker_0']['evidence']
        self.assertEqual(evidence['match']['model'], MODEL.payload())
        self.assertEqual(evidence['match']['calibration']['dataset_id'], POLICY.dataset_id)
        self.assertNotIn('centroid', json.dumps(exported))

    def test_forgetting_voice_invalidates_pending_match_but_preserves_transcripts(self):
        pid = self.person(); self.assign(pid); self.enroll(pid)
        before = self.job()['document']
        other = 'b'*32; self.seed(other, document('Welcome back.'))
        self.post(f'/api/jobs/{other}/speakers/speaker_0/voice-suggestion',
                  {'revision': 0, 'segment_ids': ['intro', 'second']})
        sid = self.proposals(other)[0]['id']
        response = self.client.delete(f'/api/people/{pid}/voice', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.job()['document'], before)
        self.assertEqual(self.proposals(other), [])
        self.assertEqual(self.assign(pid, other, suggestion_id=sid).status_code, 409)
        self.assertFalse(self.client.get('/api/people').json['people'][0]['voice_saved'])
        self.assertEqual(self.app.extensions['speakerdesk']['people'].profiles(), [])

    def test_name_correction_and_people_rename_keep_voice_and_meeting_snapshots_separate(self):
        pid = self.person(); self.assign(pid); self.enroll(pid)
        profile = copy.deepcopy(self.app.extensions['speakerdesk']['people'].profiles())
        self.client.patch(f'/api/people/{pid}', headers=self.headers, json={'name': 'Dr. Priya'})
        self.assertEqual(self.job()['document']['speakers']['speaker_0'], 'Priya')
        job = self.job(); job['document']['speakers']['speaker_0'] = 'Guest'
        response = self.client.put(f'/api/jobs/{self.jid}/transcript', headers=self.headers,
                                   json={'revision': job['revision'], 'document': job['document']})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['speaker_assignments'], {})
        self.assertEqual(self.client.get('/api/people').json['people'][0]['name'], 'Dr. Priya')
        self.assertEqual(self.app.extensions['speakerdesk']['people'].profiles(), profile)

    def test_import_cannot_claim_confirmed_identity_or_keep_previous_assignment(self):
        pid = self.person(); self.assign(pid)
        job = self.job(); incoming = job['document']
        incoming['speaker_assignments'] = {'speaker_0': {'person_id': pid, 'source': 'voice_match'}}
        response = self.client.put(f'/api/jobs/{self.jid}/transcript', headers=self.headers,
                                   json={'revision': job['revision'], 'document': incoming, 'imported': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['speaker_assignments'], {})
        self.assertNotIn('speaker_assignments', response.json['document'])
        self.assertEqual(self.proposals(), [])

    def test_later_live_results_preserve_confirmed_names_and_new_tracks_stay_unknown(self):
        self.seed('b'*32, status='recording')
        pid = self.person(); response = self.assign(pid, 'b'*32)
        self.assertEqual(response.status_code, 200)
        manager = self.app.extensions['speakerdesk']['meetings']
        job = manager.get('b'*32)
        append_finalized_segment(job['document'], {'speakers': {'speaker_0': 'Speaker 1', 'speaker_1': 'Speaker 2'},
                                  'segment': {'id': 'next-live-segment', 'speaker': 'speaker_0', 'start': 8., 'end': 10., 'text': 'Next topic.'}})
        manager.put(job)
        saved = self.job('b'*32)
        self.assertEqual(saved['document']['speakers'], {'speaker_0': 'Priya', 'speaker_1': 'Speaker 2'})
        self.assertEqual(saved['speaker_assignments']['speaker_0']['person_id'], pid)
        self.assertEqual(saved['document']['segments'][-1]['id'], 'next-live-segment')
        self.assertEqual(self.backend.calls, [])

    def test_duplicate_names_have_separate_person_ids_and_voice_collision_stays_unknown(self):
        first = self.person('Priya'); second = self.person('Priya')
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.client.get('/api/people').json['people']), 2)
        self.assign(first); self.enroll(first)
        self.assign(second); self.enroll(second)
        other = 'b'*32; self.seed(other, document('Welcome back.'))
        response = self.post(f'/api/jobs/{other}/speakers/speaker_0/voice-suggestion',
                             {'revision': 0, 'segment_ids': ['intro', 'second']})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json['matched'])
        self.assertEqual(self.proposals(other), [])
        self.assertEqual(self.job(other)['document']['speakers']['speaker_0'], 'Speaker 1')
        self.assertEqual(self.job()['speaker_assignments']['speaker_0']['person_id'], second)

    def test_new_model_disables_old_profile_and_pending_suggestion_without_erasing_it(self):
        pid = self.person(); self.assign(pid); self.enroll(pid)
        other = 'b'*32; self.seed(other, document('Welcome back.'))
        self.post(f'/api/jobs/{other}/speakers/speaker_0/voice-suggestion',
                  {'revision': 0, 'segment_ids': ['intro', 'second']})
        self.assertEqual(len(self.proposals(other)), 1)
        newer = SyntheticBackend(); newer.model = replace(MODEL, revision='new-revision')
        reopened = create_app(self.root, voice_backend=newer, voice_calibration=replace(POLICY, model=newer.model))
        try:
            client = reopened.test_client()
            self.assertEqual(client.get(f'/api/jobs/{other}/identity-suggestions').json['suggestions'], [])
            person = client.get('/api/people').json['people'][0]
            self.assertTrue(person['voice_saved']); self.assertFalse(person['voice_compatible'])
            self.assertEqual(reopened.extensions['speakerdesk']['people'].profiles()[0]['model'], MODEL.payload())
            self.assertEqual(newer.calls, [])
        finally:
            self.close_app(reopened)

    def test_live_custom_name_correction_survives_next_diarizer_result(self):
        self.seed('b'*32, status='recording')
        self.assertEqual(self.assign(None, 'b'*32, name='Guest presenter').status_code, 200)
        manager = self.app.extensions['speakerdesk']['meetings']; job = manager.get('b'*32)
        append_finalized_segment(job['document'], {'speakers': {'speaker_0': 'Speaker 1'},
                                  'segment': {'id': 'next', 'speaker': 'speaker_0', 'start': 8., 'end': 10., 'text': 'Next topic.'}})
        manager.put(job)
        self.assertEqual(self.job('b'*32)['document']['speakers']['speaker_0'], 'Guest presenter')
        self.assertIsNone(self.job('b'*32)['speaker_assignments']['speaker_0']['person_id'])
        self.assertEqual(self.client.get('/api/people').json['people'], [])


class MatchPolicyTests(unittest.TestCase):
    def profile(self, vector, model=MODEL, pid='saved-person'):
        return {'person_id': pid, 'version': 'profile-v1', 'model': model.payload(), 'centroid': vector}

    def test_unknown_for_insufficient_evidence_bad_clip_low_score_or_close_runner_up(self):
        for vectors, profiles in [([(1., 0.)], [self.profile([1., 0.])]),
                                  ([(1., 0.), (.1, 1.)], [self.profile([1., 0.])]),
                                  ([(0., 1.), (0., 1.)], [self.profile([1., 0.])]),
                                  ([(1., 0.), (1., 0.)], [self.profile([1., 0.]), self.profile([.99, .01], pid='runner-up')]),
                                  ([(1., 0.), (1., 0.)], [])]:
            with self.subTest(vectors=vectors, profiles=profiles):
                self.assertIsNone(propose_match(MODEL, POLICY, vectors, profiles))

    def test_model_revision_artifact_dimension_and_calibration_spaces_must_match(self):
        for model in [replace(MODEL, revision='new'), replace(MODEL, artifact_sha256='b'*64), replace(MODEL, dimension=3)]:
            self.assertIsNone(propose_match(MODEL, POLICY, [(1., 0.), (1., 0.)], [self.profile([1., 0.], model)]))
        self.assertIsNone(propose_match(MODEL, replace(POLICY, model=replace(MODEL, revision='new')),
                                        [(1., 0.), (1., 0.)], [self.profile([1., 0.])]))

    def test_calibration_is_required_and_match_records_threshold_and_margin(self):
        for fields in [{'dataset_id': ''}, {'genuine_trials': 0}, {'impostor_trials': 0},
                       {'minimum_clips': 1}, {'margin': 0.}, {'threshold': float('nan')}]:
            with self.assertRaises(ValueError):
                replace(POLICY, **fields)
        match = propose_match(MODEL, POLICY, [(2., 0.), (3., 0.)], [self.profile([1., 0.])])
        self.assertEqual(match['person_id'], 'saved-person')
        self.assertEqual(match['calibration']['threshold'], .8)
        self.assertEqual(match['calibration']['margin'], .1)
        self.assertAlmostEqual(match['score'], 1.)


if __name__ == '__main__':
    unittest.main()
