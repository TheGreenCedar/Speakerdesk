"""Real local SQLite/API lifecycle with CPU peers; no models or devices."""
from contextlib import closing
import queue
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from support.meeting_harness import MeetingHarness


class PreferenceTests(MeetingHarness,unittest.TestCase):
    def choose(self,language,**extra):
        return self.client.patch('/api/preferences/language',headers=self.headers,json={'language':language,**extra})
    def default(self):return self.client.get('/api/config').json['default_language']
    def stop(self,jid):
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
    def paused_inflight(self):
        self.scenario='language_inflight';jid=self.start(['microphone'])
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause');self.wait_for(lambda:self.job(jid)['status']=='paused' and self.manager.duration==.1)
        self.wait_for(lambda:(self.root/jid/'worker-inflight').exists());return jid
    def test_default_is_authoritative_for_new_meeting_and_survives_restart_and_auto_reset(self):
        self.assertEqual(self.default(),'auto')
        self.assertEqual(self.choose('fr').json,{'default_language':'fr'});self.assertEqual(self.default(),'fr')
        response=self.client.post('/api/meetings',headers=self.headers,json={'name':'Saved default','sources':['microphone']})
        self.assertEqual(response.status_code,201,response.json);jid=response.json['id']
        self.assertEqual(response.json['language'],'fr');self.wait_for(lambda:self.job(jid)['status']=='recording');self.stop(jid)
        reopened=create_app(self.root)
        try:self.assertEqual(reopened.test_client().get('/api/config').json['default_language'],'fr')
        finally:reopened.extensions['speakerdesk']['meetings'].close();reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
        self.assertEqual(self.choose('auto').json,{'default_language':'auto'});self.assertEqual(self.default(),'auto')
        reopened=create_app(self.root)
        try:self.assertEqual(reopened.test_client().get('/api/config').json['default_language'],'auto')
        finally:reopened.extensions['speakerdesk']['meetings'].close();reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
    def test_settings_mode_changes_active_future_audio_and_default_without_relabeling_old_audio(self):
        jid=self.paused_inflight();response=self.choose('fr')
        self.assertEqual(response.status_code,200,response.json);self.assertEqual(response.json['default_language'],'fr')
        self.assertEqual(response.json['meeting']['language_revision'],1);self.assertEqual(self.default(),'fr')
        self.assertEqual(self.job(jid)['language_history'][-1]['start_sample'],1600)
        (self.root/jid/'worker-release').touch()
        self.wait_for(lambda:self.job(jid)['language_acknowledged_revision']==1)
        self.wait_for(lambda:self.job(jid)['document']['segments'])
        self.control(jid,'resume');self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause');self.wait_for(lambda:self.job(jid)['status']=='paused' and self.manager.duration==.35)
        self.wait_for(lambda:len(self.job(jid)['document']['segments'])==3)
        self.stop(jid);rows=self.job(jid)['document']['segments']
        self.assertEqual([(r['start'],r['end'],r['language']) for r in rows],[(0,.1,'en'),(.1,.25,'fr'),(.25,.35,'fr')])
    def test_direct_active_route_persists_default_and_same_mode_does_not_create_an_epoch(self):
        jid=self.paused_inflight();before=self.job(jid)
        response=self.client.patch(f'/api/meetings/{jid}/language',headers=self.headers,json={'language':'en','language_revision':0})
        self.assertEqual(response.status_code,200,response.json);self.assertEqual(response.json['default_language'],'en')
        self.assertEqual(self.job(jid)['language_history'],before['language_history']);self.assertEqual(self.default(),'en')
        response=self.client.patch(f'/api/meetings/{jid}/language',headers=self.headers,json={'language':'auto','language_revision':0})
        self.assertEqual(response.status_code,200,response.json);self.assertEqual(response.json['default_language'],'auto')
        self.assertEqual(self.default(),'auto');(self.root/jid/'worker-release').touch();self.stop(jid)
    def test_preference_write_failure_rolls_back_job_epoch_default_and_producer_state(self):
        jid=self.paused_inflight();before=self.job(jid);language=self.default()
        with closing(sqlite3.connect(self.root/'jobs.sqlite')) as conn,conn:
            conn.execute("CREATE TRIGGER reject_preference BEFORE INSERT ON preferences BEGIN SELECT RAISE(ABORT,'synthetic disk failure'); END")
        with self.assertLogs(self.app.logger,level='ERROR'):response=self.choose('fr')
        self.assertEqual(response.status_code,500);self.assertEqual(self.job(jid),before);self.assertEqual(self.default(),language)
        self.assertEqual((self.manager.active_language,self.manager.active_epoch),('en',0))
        (self.root/jid/'worker-release').touch();self.stop(jid)
    def test_rejected_validation_cas_preflight_and_queue_leave_default_and_epoch_unchanged(self):
        self.assertEqual(self.client.patch('/api/preferences/language',json={'language':'fr'}).status_code,403)
        self.assertEqual(self.choose('unsupported').status_code,400);self.assertEqual(self.default(),'auto')
        jid=self.paused_inflight();before=self.job(jid)
        self.assertEqual(self.choose('fr',language_revision=99).status_code,409)
        with patch('live_meeting.preflight',return_value=['Synthetic missing model']):self.assertEqual(self.choose('fr').status_code,409)
        with self.manager.lock:
            actual=self.manager.packets;blocked=queue.Queue(maxsize=1);blocked.put({'type':'audio'});self.manager.packets=blocked
            try:self.assertEqual(self.choose('fr').status_code,429)
            finally:self.manager.packets=actual
        self.assertEqual(self.job(jid),before);self.assertEqual(self.default(),'en')
        (self.root/jid/'worker-release').touch();self.stop(jid)


if __name__=='__main__':unittest.main()
