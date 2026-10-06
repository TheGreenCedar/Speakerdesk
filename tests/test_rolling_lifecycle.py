"""Real resident JSONL, pipes, WAV/SQLite and recovery; CPU models only."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from support.meeting_harness import MeetingHarness, PROTOCOL


class RollingLifecycleTests(MeetingHarness,unittest.TestCase):
    def setUp(self):
        super().setUp();self.peers.stop();real_popen=subprocess.Popen
        def peer(command,**kwargs):
            folder=Path(kwargs['stderr'].name).parent
            if len(command)>1:
                worker=Path(__file__).parent/'support/rolling_protocol.py'
                process=real_popen([sys.executable,str(worker),command[-1]],**kwargs)
            else:process=real_popen([sys.executable,str(PROTOCOL),'capture',str(folder),self.scenario,'en'],**kwargs)
            self.children.append(process);return process
        self.peers=patch('live_meeting.subprocess.Popen',side_effect=peer);self.peers.start()
    def test_short_pause_resume_stop_drains_real_engine_and_refinement_before_shutdown(self):
        jid=self.start(['microphone','system']);self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause');self.wait_for(lambda:self.job(jid)['status']=='paused')
        self.wait_for(lambda:self.job(jid)['document']['segments'])
        self.assertEqual(self.change_language(jid,'fr',0).status_code,200)
        self.control(jid,'resume');self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        job=self.job(jid);self.assertEqual(job['status'],'ready',job)
        self.assertEqual(job['refinement_status'],'complete')
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
