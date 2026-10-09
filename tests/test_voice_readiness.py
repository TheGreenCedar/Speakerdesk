"""Read-only API readiness cannot turn an unqualified runtime into a download remedy."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app


class VoiceReadinessTests(unittest.TestCase):
    def test_runtime_refusal_and_model_missing_have_distinct_actions_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            app=create_app(temporary);ext=app.extensions['speakerdesk'];manager=ext['voice_setup']
            client=app.test_client();job=dict(id='a'*32,name='Owned source fixture',kind='meeting',status='ready',
                duration=7.,created=1.,revision=0,document=dict(schema_version=1,
                speakers={'speaker_0':'Speaker 1'},segments=[dict(id=str(i),speaker='speaker_0',start=i*4.,end=i*4.+3.,
                text='An owned test passage.',review=False,voice_eligible=True) for i in range(2)],warnings=[],provenance={'kind':'local_inference'}))
            ext['recognition'].put(job)
            try:
                cases=[(True,False,False,'runtime_unqualified','updates'),(True,False,True,'runtime_unqualified','updates'),
                       (False,True,True,'runtime_unsupported','updates'),(True,True,False,'model_missing','models'),
                       (True,True,True,'runtime_unavailable','updates')]
                for supported,released,installed,code,action in cases:
                    with self.subTest(code=code,installed=installed),patch.object(manager,'supported',return_value=supported),\
                         patch.object(manager,'released',return_value=released),patch.object(manager,'installed',return_value=installed),\
                         patch.object(manager,'download') as download,patch.object(manager,'activate') as activate:
                        people=client.get('/api/people').json
                        clips=client.get('/api/jobs/'+'a'*32+'/speakers/speaker_0/voice-clips').json
                        self.assertFalse(people['voice_available']);self.assertEqual(clips['status'],'unavailable')
                        for data in (people,clips):
                            self.assertEqual(data['readiness_code'],code);self.assertEqual(data['next_action'],action)
                        if code=='runtime_unqualified':
                            self.assertNotIn('finish local model setup',clips['message'].lower())
                            self.assertIn('qualified',clips['message']);self.assertIn('kept',clips['message'])
                        download.assert_not_called();activate.assert_not_called()
            finally:
                ext['recognition'].close(wait=True);ext['meetings'].close();ext['executor'].shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':unittest.main()
