"""CPU-only timing, lifecycle and access regressions; these tests do not run AI or capture devices."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from live_meeting import SourceMixer, RATE


class MixerTests(unittest.TestCase):
    def setUp(self):
        self.blocks=[]
        self.mixer=SourceMixer(['microphone','system'],lambda mixed,tracks:self.blocks.append((mixed.copy(),{k:v.copy() for k,v in tracks.items()})),echo_factory=None)
    def audio(self, value, length):return np.full(length,value,dtype='<f4').tobytes()
    def result(self):return np.concatenate([b[0] for b in self.blocks])
    def test_delayed_source_is_aligned_before_watermark(self):
        self.mixer.add('microphone',0,self.audio(.2,4000))
        self.mixer.flush(.5)
        self.assertEqual(self.blocks,[])
        self.mixer.add('system',.125,self.audio(.3,2000))
        self.mixer.flush(.75)
        np.testing.assert_allclose(self.result()[:2000],.2)
        np.testing.assert_allclose(self.result()[2000:],.5)
    def test_pause_at_partial_block_then_resume_preserves_every_sample(self):
        self.mixer.add('microphone',0,self.audio(.2,1600))
        self.mixer.flush(.1,final=True)
        self.mixer.add('microphone',.1,self.audio(.4,4000))
        self.mixer.flush(.35,final=True)
        self.assertEqual(len(self.result()),5600)
        np.testing.assert_allclose(self.result()[:1600],.2)
        np.testing.assert_allclose(self.result()[1600:],.4)
    def test_silent_source_keeps_timeline_and_sum_clips(self):
        self.mixer.add('microphone',0,self.audio(.8,4000))
        self.mixer.add('system',0,self.audio(.8,2000))
        self.mixer.flush(.5,final=True)
        self.assertEqual(len(self.result()),8000)
        np.testing.assert_allclose(self.result()[:2000],1)
        np.testing.assert_allclose(self.result()[2000:4000],.8)
        np.testing.assert_allclose(self.result()[4000:],0)
    def test_late_source_cannot_overwrite_saved_audio(self):
        self.mixer.flush(.25,final=True)
        self.mixer.add('system',0,self.audio(.7,4000))
        self.assertEqual(self.mixer.late_samples,4000)
        self.assertFalse(self.mixer.tiles)
    def test_long_capture_memory_is_bounded(self):
        for n in range(1200):
            self.mixer.add('microphone',n*.25,self.audio(.1,4000))
            self.mixer.flush((n+1)*.25)
            self.assertLessEqual(len(self.mixer.tiles),3)
        self.mixer.flush(300,final=True)
        self.assertEqual(len(self.result()),300*RATE)
    def test_clock_jump_and_invalid_packets_fail_explicitly(self):
        with self.assertRaises(RuntimeError):self.mixer.add('system',6,self.audio(.1,100))
        with self.assertRaises(ValueError):self.mixer.add('microphone',0,np.array([np.nan],dtype='<f4').tobytes())
        with self.assertRaises(ValueError):self.mixer.add('unknown',0,self.audio(0,100))
        with self.assertRaises(RuntimeError):self.mixer.flush(10)


class MeetingAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.app=create_app(self.root);self.client=self.app.test_client()
        html=self.client.get('/').get_data(as_text=True)
        token=re.search(r'name="speakerdesk-token" content="([^"]+)"',html).group(1)
        self.headers={'X-Speakerdesk-Token':token}
    def tearDown(self):
        self.app.extensions['speakerdesk']['meetings'].close()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
        self.temp.cleanup()
    def test_capture_requires_explicit_authenticated_start(self):
        self.assertEqual(self.client.get('/api/meeting').json['status'],'idle')
        response=self.client.post('/api/meetings',json={'name':'Private','sources':['microphone']})
        self.assertEqual(response.status_code,403)
        self.assertEqual(self.client.get('/api/jobs').json,[])
    def test_missing_helper_creates_no_recording_or_permission_prompt(self):
        self.app.extensions['speakerdesk']['meetings'].helper_path=lambda:self.root/'missing-helper'
        response=self.client.post('/api/meetings',headers=self.headers,json={'name':'Meeting','sources':['microphone'],'language':'en'})
        self.assertEqual(response.status_code,409)
        self.assertEqual(self.client.get('/api/jobs').json,[])
    def test_missing_echo_component_stops_before_models_or_recording(self):
        self.app.extensions['speakerdesk']['meetings'].helper_path=lambda:Path(__file__)
        with patch('capture_echo.library_path',return_value=self.root/'missing-echo.dylib'), \
             patch('live_meeting.preflight',side_effect=AssertionError('Model preflight must not run')):
            response=self.client.post('/api/meetings',headers=self.headers,
                json={'name':'Meeting','sources':['microphone','system'],'language':'en'})
        self.assertEqual(response.status_code,409)
        self.assertIn('echo cancellation is missing',response.json['error'])
        self.assertEqual(self.client.get('/api/jobs').json,[])
    def test_invalid_sources_are_rejected(self):
        for sources in [['microphone','microphone'],['unknown'],[],[{}]]:
            response=self.client.post('/api/meetings',headers=self.headers,json={'name':'Meeting','sources':sources,'language':'en'})
            self.assertEqual(response.status_code,400)
    def test_interrupted_meeting_preserves_transcript_and_audio(self):
        jid='a'*32;folder=self.root/jid;folder.mkdir();audio=folder/'audio.wav';audio.write_bytes(b'preserved')
        document={'schema_version':1,'speakers':{'speaker_0':'Speaker 1'},'segments':[]}
        job={'id':jid,'name':'Meeting','kind':'meeting','status':'recording','created':1,'revision':2,'duration':3,'document':document}
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.root/'jobs.sqlite')) as db:
            db.execute('INSERT INTO jobs VALUES (?,?)',(jid,json.dumps(job)));db.commit()
        # Active documents cannot be edited or deleted while capture owns them.
        self.assertEqual(self.client.delete(f'/api/jobs/{jid}',headers=self.headers).status_code,409)
        self.assertEqual(self.client.put(f'/api/jobs/{jid}/transcript',headers=self.headers,json={'revision':2,'document':document}).status_code,409)
        reopened=create_app(self.root)
        recovered=reopened.test_client().get(f'/api/jobs/{jid}').json
        self.assertEqual(recovered['status'],'failed');self.assertEqual(recovered['document'],document)
        self.assertEqual(audio.read_bytes(),b'preserved')
        reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
        reopened.extensions['speakerdesk']['meetings'].close()


if __name__=='__main__':unittest.main()
