"""Canonical production JSONL/pipes/SQLite with explicit CPU acoustic peers."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from support.meeting_harness import MeetingHarness,PROTOCOL


class CanonicalLifecycle(MeetingHarness,unittest.TestCase):
    def setUp(self):
        super().setUp();self.peers.stop();real_popen=subprocess.Popen
        def peer(command,**kwargs):
            folder=Path(kwargs['stderr'].name).parent
            if len(command)>1:
                config=json.loads(command[-1]);config['test_canonical_peer']=True
                command=[sys.executable,str(Path(__file__).parent/'support/rolling_protocol.py'),json.dumps(config)]
            else:command=[sys.executable,str(PROTOCOL),'capture',str(folder),self.scenario,'en']
            process=real_popen(command,**kwargs);self.children.append(process);return process
        self.peers=patch('live_meeting.subprocess.Popen',side_effect=peer);self.peers.start()

    def test_pause_ack_open_identity_then_stop_drains_whole_utterance_and_persists(self):
        jid=self.start(['microphone']);self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause')
        self.wait_for(lambda:self.job(jid).get('pause_flush',{}).get('state')=='complete')
        paused=self.job(jid);self.assertTrue(paused['canonical_utterances'])
        first=paused['document']['segments'][0]
        self.assertEqual(first['canonical_state'],'open')
        self.control(jid,'resume');self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        job=self.job(jid);self.assertEqual(job['status'],'ready',job)
        speech=[row for row in job['document']['segments'] if row.get('canonical_utterance_id')]
        self.assertEqual(len(speech),1);row=speech[0];self.assertEqual(row['id'],first['id'])
        self.assertEqual(row['canonical_state'],'sealed');self.assertEqual(row['refinement_state'],'refined')
        self.assertEqual(row['end_sample'],5600);self.assertEqual(job['refinement_status'],'complete')
        self.assertEqual(len(self.samples(jid,'audio.wav')),5600)
        inspection=job['capture_admission']
        self.assertEqual(inspection['end_sample'],5600)
        self.assertEqual(inspection['execution_id'],paused['pause_flush']['admission_receipt']['execution_id'])
        commands=[json.loads(line) for line in (self.root/jid/'worker-commands.jsonl').read_text().splitlines()]
        refinement=next(item for item in commands if item['type']=='refine')
        self.assertEqual(refinement['canonical']['id'],row['id'])
        self.assertEqual(refinement['window']['end_sample'],5600)
        self.assertLess(next(i for i,item in enumerate(commands) if item['type']=='stop'),
                        next(i for i,item in enumerate(commands) if item['type']=='shutdown'))
        self.assertTrue(list((self.root/jid).glob('utterance-versions-*.jsonl')))
        # Explicit saved retry must redispatch an attempted unresolved source.
        stored=self.manager.get(jid);sid=row['id']
        stored['rolling_sources'][sid]['canonical_unresolved']='refinement_incomplete'
        stored['refinement_status']='unresolved';self.manager.put(stored)
        before=sum(item['type']=='refine' for item in commands)
        self.assertEqual(self.client.post(f'/api/jobs/{jid}/refinement/resume',headers=self.headers).status_code,202)
        self.wait_for(lambda:self.manager.jid is None)
        after=[json.loads(line) for line in (self.root/jid/'worker-commands.jsonl').read_text().splitlines()]
        self.assertEqual(sum(item['type']=='refine' for item in after),before+1)
        self.assertEqual(self.job(jid)['refinement_status'],'complete')
        self.assertEqual(self.job(jid)['capture_admission'],inspection)


if __name__=='__main__':unittest.main()
