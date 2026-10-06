"""Real resident JSONL, pipes, WAV/SQLite and recovery; CPU models only."""
import json
import threading
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from support.meeting_harness import MeetingHarness, PROTOCOL
from test_live_refinement import wav_file


class RollingLifecycleTests(MeetingHarness,unittest.TestCase):
    def setUp(self):
        super().setUp();self.peers.stop();real_popen=subprocess.Popen
        def peer(command,**kwargs):
            folder=Path(kwargs['stderr'].name).parent
            if len(command)>1:
                worker=Path(__file__).parent/'support/rolling_protocol.py'
                config=json.loads(command[-1])
                if self.scenario=='hold_refinement':config['test_hold_old_refinement']=True
                if self.scenario=='pause_lag':config['test_pause_lag']=True
                process=real_popen([sys.executable,str(worker),json.dumps(config)],**kwargs)
            else:process=real_popen([sys.executable,str(PROTOCOL),'capture',str(folder),self.scenario,'en'],**kwargs)
            self.children.append(process);return process
        self.peers=patch('live_meeting.subprocess.Popen',side_effect=peer);self.peers.start()
    def test_pause_ack_reaches_real_pipes_sqlite_without_requiring_future_lookahead(self):
        self.scenario='pause_lag';jid=self.start(['microphone'])
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.client.post(f'/api/jobs/{jid}/refinement/pause',headers=self.headers)
        response=self.client.post(f'/api/meetings/{jid}/pause',headers=self.headers)
        self.assertEqual(response.status_code,202,response.json);identity=response.json['pause_flush']['request_id']
        job=self.wait_for(lambda:self.job(jid) if self.job(jid).get('pause_flush',{}).get('state')=='complete' else None)
        receipt=job['pause_flush'];self.assertEqual(receipt['request_id'],identity)
        self.assertEqual(receipt['received_sample'],1600);self.assertLess(receipt['available_sample'],1600)
        self.assertEqual(receipt['deferred_audio'],[{'start_sample':receipt['available_sample'],'end_sample':1600}])
        self.assertEqual(self.client.get('/api/meeting').json['pause_flush'],receipt)
        self.control(jid,'resume');self.wait_for(lambda:self.job(jid)['status']=='recording')
        second=self.client.post(f'/api/meetings/{jid}/pause',headers=self.headers)
        self.assertEqual(second.status_code,202,second.json);second_id=second.json['pause_flush']['request_id']
        self.assertNotEqual(second_id,identity)
        self.wait_for(lambda:self.job(jid).get('pause_flush',{}).get('state')=='complete')
        self.assertEqual(self.job(jid)['pause_flush']['received_sample'],5600)
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        self.assertEqual(self.job(jid)['status'],'ready');self.assertEqual(len(self.samples(jid,'audio.wav')),5600)
    def test_short_pause_resume_stop_drains_real_engine_and_refinement_before_shutdown(self):
        # Hold an actual old-epoch refinement response until the language change.
        # Otherwise a faster worker may finish that window before the switch,
        # legitimately leaving no unresolved old audio to assert below.
        self.scenario='hold_refinement'
        jid=self.start(['microphone','system']);self.wait_for(lambda:self.job(jid)['status']=='recording')
        try:
            self.control(jid,'pause');self.wait_for(lambda:self.job(jid)['status']=='paused')
            self.wait_for(lambda:self.job(jid)['document']['segments'])
            self.wait_for(lambda:(self.root/jid/'refinement-response-held').exists())
            self.assertEqual(self.change_language(jid,'fr',0).status_code,200)
        finally:(self.root/jid/'refinement-response-release').touch()
        self.control(jid,'resume');self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        job=self.job(jid);self.assertEqual(job['status'],'ready',job)
        # The previously discarded 100 ms phrase is now retained. Its old
        # language epoch remains eligible for explicit saved-audio refinement.
        self.assertEqual(job['refinement_status'],'unresolved')
        self.assertEqual(job['refinement_unresolved'][0]['reason'],'language_changed')
        rows=[s for s in job['document']['segments'] if s['text']]
        self.assertEqual([(s['start'],s['end'],s['language']) for s in rows],[(0,.1,'en'),(.1,.35,'fr')])
        commands=[json.loads(x)['type'] for x in (self.root/jid/'worker-commands.jsonl').read_text().splitlines()]
        self.assertIn('refine',commands);self.assertLess(commands.index('stop'),commands.index('shutdown'))
        self.assertEqual(len(self.samples(jid,'audio.wav')),5600)
        original=(self.root/jid/'audio.wav').read_bytes();saved=self.manager.get(jid)
        saved['rolling_refinement']['completed_sample']=0;saved['refinement_status']='paused';self.manager.put(saved)
        response=self.client.post(f'/api/jobs/{jid}/refinement/resume',headers=self.headers)
        self.assertEqual(response.status_code,202,response.json);self.wait_for(lambda:self.manager.jid is None)
        restored=self.manager.get(jid)
        self.assertEqual([(s['start'],s['end'],s['language']) for s in restored['document']['segments']],[(0,.1,'en'),(.1,.35,'fr')])
        self.assertEqual((self.root/jid/'audio.wav').read_bytes(),original)
    def change_language(self,jid,language,revision):
        return self.client.patch(f'/api/meetings/{jid}/language',headers=self.headers,json={'language':language,'language_revision':revision})
    def test_saved_refinement_blocks_deletion_until_worker_releases_ownership(self):
        jid='d'*32;(self.root/jid).mkdir();wav_file(self.root/jid/'audio.wav',1)
        self.manager.put({'id':jid,'name':'Saved','kind':'meeting','status':'ready','language':'en','duration':1,'revision':0,
            'document':{'speakers':{},'segments':[],'provenance':{'kind':'local_inference'}}})
        entered=threading.Event();release=threading.Event();original=self.manager._run_saved_refinement
        def held_worker(job_id):
            entered.set();release.wait(timeout=5);original(job_id)
        with patch.object(self.manager,'_run_saved_refinement',side_effect=held_worker):
            try:
                response=self.client.post(f'/api/jobs/{jid}/refinement/resume',headers=self.headers)
                self.assertEqual(response.status_code,202,response.json);self.assertTrue(entered.wait(timeout=2))
                self.assertEqual(self.job(jid)['status'],'ready');self.assertTrue(self.job(jid)['inference_owned'])
                self.assertEqual(self.client.delete(f'/api/jobs/{jid}',headers=self.headers).status_code,409)
                self.assertTrue((self.root/jid/'audio.wav').is_file())
            finally:release.set()
            self.wait_for(lambda:self.manager.jid is None)
        self.assertFalse(self.job(jid)['inference_owned'])
        self.assertEqual(self.client.delete(f'/api/jobs/{jid}',headers=self.headers).status_code,200)
    def test_burst_backlog_final_core_revisions_history_and_explicit_saved_audio_resume(self):
        self.scenario='backlog';jid=self.start(['microphone'])
        self.wait_for(lambda:(self.root/jid/'capture-burst-finished').exists())
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None,timeout=10)
        job=self.manager.get(jid);self.assertEqual(job['status'],'ready',job)
        self.assertEqual(job['refinement_status'],'complete');self.assertTrue(job['refinement_history'])
        self.assertEqual(job['rolling_refinement']['completed_sample'],round(job['duration']*16000))
        before=(self.root/jid/'audio.wav').read_bytes();words=[s['text'] for s in job['document']['segments']]
        # Resume must use the saved WAV and never start the capture helper again.
        job['rolling_refinement']['completed_sample']=0;job['refinement_status']='paused';self.manager.put(job)
        count=len(self.children)
        response=self.client.post(f'/api/jobs/{jid}/refinement/resume',headers=self.headers)
        self.assertEqual(response.status_code,202,response.json);self.wait_for(lambda:self.manager.jid is None,timeout=10)
        self.assertEqual(len(self.children),count+1)
        self.assertEqual(self.manager.get(jid)['refinement_status'],'complete')
        self.assertEqual((self.root/jid/'audio.wav').read_bytes(),before)
        self.assertTrue(all(s['text'] for s in self.manager.get(jid)['document']['segments']))


if __name__=='__main__':unittest.main()
