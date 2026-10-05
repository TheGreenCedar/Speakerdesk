"""CPU-only track lifecycle/races with synthetic vectors, never native models or audio."""
import copy
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from live_meeting import append_finalized_segment
from voice_profiles import Calibration, ClipEmbedding, VoiceClip, VoiceModel, make_profile

MODEL = VoiceModel('cpu-fixture', 'v1', 'a'*64, 2)
POLICY = Calibration(MODEL, 'cpu-calibration', .8, .1, 10, 10, 0., 0.)


class Backend:
    model = MODEL

    def __init__(self):
        self.calls = []; self.vector = (1., 0.); self.clean = True
        self.started = threading.Event(); self.release = threading.Event(); self.release.set()

    def embed(self, audio, clip):
        self.calls.append(clip); self.started.set()
        if not self.release.wait(timeout=3):raise ValueError('CPU fixture timed out')
        return ClipEmbedding(self.vector, self.clean)


def segment(sid, start, end, **changes):
    return dict(id=sid, speaker='speaker_0', start=start, end=end,
                text='The next project will begin soon.', review=False, **changes)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.backend = Backend()
        self.app = create_app(self.root, voice_backend=self.backend, voice_calibration=POLICY)
        self.client = self.app.test_client()
        self.headers = {'X-Speakerdesk-Token':re.search(r'name="speakerdesk-token" content="([^"]+)"',self.client.get('/').text)[1]}
        ext = self.app.extensions['speakerdesk']
        self.recognizer, self.store = ext['recognition'], ext['people']
        self.jid = 'a'*32

    def tearDown(self):
        self.backend.release.set(); self.drain()
        self.app.extensions['speakerdesk']['meetings'].close()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.temp.cleanup()

    def drain(self):
        self.recognizer.executor.submit(lambda:None).result(timeout=4)

    def seed(self, jid=None, segments=None, **changes):
        jid = jid or self.jid
        job = dict(id=jid,name='CPU meeting',kind='meeting',status='recording',duration=120.,
                   revision=0,created=1.,document={'schema_version':1,'speakers':{'speaker_0':'Speaker 1'},
                   'segments':segments if segments is not None else [segment('one',0.,3.),segment('two',4.,7.)],
                   'provenance':{'kind':'local_inference'},'warnings':[]})
        job.update(changes)
        dest = self.root/jid; dest.mkdir(exist_ok=True); (dest/'audio.wav').write_bytes(b'CPU fixture, not real audio')
        self.recognizer.put(job); return job

    def person(self, name='Priya', model=MODEL):
        person = self.store.create(name)
        clips = [VoiceClip('enrollment','speaker_3',str(i),i*4.,i*4.+3.) for i in range(2)]
        self.store.remember(person['id'],make_profile(model,[(1.,0.),(1.,0.)],clips),True)
        return person['id']

    def job(self, jid=None):return self.recognizer.get(jid or self.jid)
    def pref(self, enabled):return self.client.patch('/api/recognition',headers=self.headers,json={'enabled':enabled})

    def test_default_on_and_off_preference_persists_without_enrollment(self):
        self.assertTrue(self.client.get('/api/recognition').json['enabled'])
        self.assertEqual(self.client.patch('/api/recognition',json={'enabled':False}).status_code,403)
        self.assertEqual(self.pref('false').status_code,400)
        self.assertEqual(self.pref(False).status_code,200)
        self.seed();self.person();self.recognizer.observe(self.jid);self.drain()
        self.assertEqual(self.backend.calls,[])
        reopened = create_app(self.root)
        try:self.assertFalse(reopened.test_client().get('/api/recognition').json['enabled'])
        finally:
            reopened.extensions['speakerdesk']['meetings'].close()
            reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_once_per_track_carries_name_and_checks_same_slot_in_new_meeting(self):
        pid=self.person();self.seed();before=copy.deepcopy(self.store.profiles())
        self.recognizer.observe(self.jid);self.drain()
        job=self.job();assignment=job['speaker_assignments']['speaker_0']
        self.assertEqual(job['document']['speakers']['speaker_0'],'Priya')
        self.assertEqual(assignment['person_id'],pid);self.assertFalse(assignment['confirmed'])
        self.assertEqual(assignment['source'],'automatic_voice')
        self.assertEqual(assignment['evidence']['match']['calibration']['dataset_id'],POLICY.dataset_id)
        for i in range(20):
            job=self.job();append_finalized_segment(job['document'],{'speakers':{'speaker_0':'Speaker 1'},
                'segment':segment('later'+str(i),10.+i*4,13.+i*4)})
            self.recognizer.put(job);self.recognizer.observe(self.jid)
        self.drain();self.assertEqual(len(self.backend.calls),2)
        self.assertEqual(self.job()['document']['speakers']['speaker_0'],'Priya')
        self.seed('b'*32);self.recognizer.observe('b'*32);self.drain()
        self.assertEqual(len(self.backend.calls),4)
        self.assertEqual(self.job('b'*32)['speaker_assignments']['speaker_0']['meeting_id'],'b'*32)
        self.assertEqual(self.store.profiles(),before)

    def test_unknown_is_checked_once_and_no_compatible_profiles_skip_embedding(self):
        self.seed();self.recognizer.observe(self.jid);self.drain();self.assertEqual(self.backend.calls,[])
        self.person(model=VoiceModel('different','v1','b'*64,2))
        self.recognizer.observe(self.jid);self.drain();self.assertEqual(self.backend.calls,[])
        self.person();self.backend.vector=(0.,1.)
        self.recognizer.observe(self.jid);self.drain()
        for _ in range(5):self.recognizer.observe(self.jid)
        self.drain();self.assertEqual(len(self.backend.calls),2)
        self.assertEqual(self.job()['voice_checks']['speaker_0']['status'],'unknown')
        self.assertNotIn('speaker_assignments',self.job())

    def test_multiple_clean_clips_overlap_and_phrase_text_review(self):
        self.person();job=self.seed(segments=[segment('one',0.,6.,voice_eligible=True)])
        job['document']['segments'][0]['review']=True;self.recognizer.put(job)
        self.recognizer.observe(self.jid);self.drain();self.assertEqual(self.backend.calls,[])
        job=self.job();job['document']['segments'].append(segment('overlap',6.,9.,speaker_candidates=['speaker_0','speaker_1'],voice_eligible=True))
        self.recognizer.put(job);self.recognizer.observe(self.jid);self.drain();self.assertEqual(self.backend.calls,[])
        job=self.job();job['document']['segments'].append(segment('two',10.,16.,voice_eligible=True))
        job['document']['segments'][-1]['review']=True;self.recognizer.put(job)
        self.recognizer.observe(self.jid);self.drain()
        self.assertEqual([c.segment_id for c in self.backend.calls],['one','two'])

    def test_long_import_turn_uses_distinct_windows_and_keeps_transcript_intact(self):
        self.person();job=self.seed(segments=[segment('long',0.,25.,voice_eligible=True)],status='ready')
        self.recognizer.observe(self.jid);self.drain()
        self.assertEqual([(c.start,c.end) for c in self.backend.calls],[(0.,6.),(6.,12.)])
        self.assertEqual(self.job()['document']['segments'],job['document']['segments'])
        pid=self.job()['speaker_assignments']['speaker_0']['person_id']
        self.assertEqual(self.client.post(f'/api/jobs/{self.jid}/speakers/speaker_0/identity',headers=self.headers,
            json={'revision':self.job()['revision'],'person_id':pid}).status_code,200)
        response=self.client.post(f'/api/people/{pid}/voice',headers=self.headers,json={'consent':True,
            'meeting_id':self.jid,'track_id':'speaker_0','revision':self.job()['revision'],
            'segment_ids':['long@0:96000','long@96000:192000']})
        self.assertEqual(response.status_code,200)
        self.assertEqual([(c.start,c.end) for c in self.backend.calls[-2:]],[(0.,6.),(6.,12.)])

    def test_explicit_intro_conflict_abstains_and_addressed_name_is_not_an_intro(self):
        self.person('Alex');job=self.seed();job['document']['segments'][0]['text']="I'm Priya. Welcome."
        self.recognizer.put(job);self.recognizer.observe(self.jid);self.drain()
        self.assertNotIn('speaker_assignments',self.job())
        job=self.seed('b'*32);job['document']['segments'][0]['text']='Priya, what do you think?'
        self.recognizer.put(job);self.recognizer.observe('b'*32);self.drain()
        self.assertEqual(self.job('b'*32)['document']['speakers']['speaker_0'],'Alex')

    def test_pending_work_is_outside_lock_and_manual_correction_wins(self):
        pid=self.person();self.seed();self.backend.release.clear();self.recognizer.observe(self.jid)
        self.assertTrue(self.backend.started.wait(timeout=1))
        self.assertTrue(self.recognizer.lock.acquire(timeout=.2));self.recognizer.lock.release()
        response=self.client.post(f'/api/jobs/{self.jid}/speakers/speaker_0/identity',headers=self.headers,
            json={'revision':self.job()['revision'],'name':'A different person'})
        self.assertEqual(response.status_code,200)
        self.backend.release.set();self.drain()
        self.assertEqual(self.job()['document']['speakers']['speaker_0'],'A different person')
        self.assertEqual(self.job()['speaker_assignments']['speaker_0']['source'],'manual')
        self.assertEqual(self.store.profiles()[0]['person_id'],pid)

    def test_switch_off_and_forget_each_cancel_pending_result(self):
        pid=self.person();self.seed();self.backend.release.clear();self.recognizer.observe(self.jid)
        self.assertTrue(self.backend.started.wait(timeout=1));self.pref(False);self.pref(True)
        self.backend.release.set();self.drain();self.assertNotIn('speaker_assignments',self.job())
        self.seed('b'*32);self.backend.started.clear();self.backend.release.clear();self.recognizer.observe('b'*32)
        self.assertTrue(self.backend.started.wait(timeout=1));self.store.forget(pid)
        self.backend.release.set();self.drain();self.assertFalse(self.job('b'*32).get('speaker_assignments'))

    def test_automatic_match_cannot_enroll_until_explicit_name_confirmation(self):
        pid=self.person();self.seed(status='ready');self.recognizer.observe(self.jid);self.drain()
        before=copy.deepcopy(self.store.profiles())
        body={'consent':True,'meeting_id':self.jid,'track_id':'speaker_0','revision':self.job()['revision'],'segment_ids':['one','two']}
        response=self.client.post(f'/api/people/{pid}/voice',headers=self.headers,json=body)
        self.assertEqual(response.status_code,400);self.assertEqual(self.store.profiles(),before)
        self.assertEqual(len(self.backend.calls),2)
        self.assertEqual(self.client.post(f'/api/jobs/{self.jid}/speakers/speaker_0/identity',headers=self.headers,
            json={'revision':self.job()['revision'],'person_id':pid}).status_code,200)
        body['revision']=self.job()['revision']
        self.assertEqual(self.client.post(f'/api/people/{pid}/voice',headers=self.headers,json=body).status_code,200)

    def test_client_cannot_forge_clean_audio_evidence_and_selected_edit_cancels_work(self):
        self.person();job=self.seed(status='ready');job['document']['segments'][0].update(review=True,voice_eligible=False)
        self.recognizer.put(job);doc=copy.deepcopy(job['document']);doc['segments'][0].update(review=False,voice_eligible=True)
        response=self.client.put(f'/api/jobs/{self.jid}/transcript',headers=self.headers,json={'revision':0,'document':doc})
        self.assertEqual(response.status_code,200);self.assertFalse(response.json['document']['segments'][0]['voice_eligible'])
        self.recognizer.observe(self.jid);self.drain();self.assertEqual(self.backend.calls,[])
        job=self.seed('b'*32,status='ready');self.backend.release.clear();self.recognizer.observe('b'*32)
        self.assertTrue(self.backend.started.wait(timeout=1));doc=copy.deepcopy(job['document']);doc['segments'][0]['start']=1.
        response=self.client.put('/api/jobs/'+('b'*32)+'/transcript',headers=self.headers,json={'revision':0,'document':doc})
        self.assertEqual(response.status_code,200)
        self.backend.release.set();self.drain();self.assertFalse(self.job('b'*32).get('speaker_assignments'))


if __name__=='__main__':unittest.main()
