"""Disposable real-route compatibility states preserve raw stored profile bytes."""
from pathlib import Path
import json
import re
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'speakerdesk'))
from app import create_app
from voice_profiles import VoiceModel,Calibration


class Backend:
    def __init__(self,model):self.model=model
    def embed(self,*args):raise AssertionError('Status must not embed')
    def require_execution_policy(self):pass


class ProfileStatusTests(unittest.TestCase):
    def test_legacy_and_current_profiles_exact_preservation_in_real_GET(self):
        model=VoiceModel('fixture-model','GPU-fixture-v1','a'*64,192)
        policy=Calibration(model,'fixture-only',.7,.1,2,2,0.,0.)
        with tempfile.TemporaryDirectory() as directory:
            app=create_app(directory,voice_backend=Backend(model),voice_calibration=policy)
            client=app.test_client()
            headers={'X-Speakerdesk-Token':re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').text)[1]}
            people=[client.post('/api/people',headers=headers,json={'name':n}).get_json() for n in ('Legacy','Current','None','Partial')]
            db=Path(directory)/'jobs.sqlite'
            legacy={'person_id':people[0]['id'],'version':'legacy-keep','model':{**model.payload(),'revision':'legacy:ALL'},'consent':'explicit_remember_voice','centroid':[1.]+[0.]*191}
            current={**legacy,'person_id':people[1]['id'],'version':'current-keep','model':model.payload()}
            partial={**current,'person_id':people[3]['id'],'version':'partial-keep','model':{k:v for k,v in model.payload().items() if k!='sample_rate'}}
            profiles=(legacy,current,partial)
            raw=[json.dumps(p,indent=3) for p in profiles]
            with sqlite3.connect(db) as conn:
                conn.executemany('INSERT INTO voice_profiles VALUES (?,?)',[(p['person_id'],v) for p,v in zip(profiles,raw)])
            response=client.get('/api/people');self.assertEqual(response.status_code,200)
            states={p['name']:p for p in response.get_json()['people']}
            self.assertEqual(states['Legacy']['voice_profile_state'],'incompatible_runtime')
            self.assertEqual(states['Legacy']['voice_profile_next_action'],'remember_voice')
            self.assertFalse(states['Legacy']['voice_compatible']);self.assertTrue(states['Legacy']['voice_saved'])
            self.assertTrue(states['Current']['voice_compatible']);self.assertEqual(states['None']['voice_profile_state'],'none')
            self.assertEqual(states['Partial']['voice_profile_state'],'unverified')
            self.assertFalse(states['Partial']['voice_compatible'])
            self.assertIsNone(states['Partial']['voice_profile_next_action'])
            with sqlite3.connect(db) as conn:
                actual=dict(conn.execute('SELECT person_id,payload FROM voice_profiles'))
            self.assertEqual(actual,{p['person_id']:v for p,v in zip(profiles,raw)})
            # Removing runtime availability must stop offering re-enrollment.
            app.extensions['speakerdesk']['voice_runtime']=(None,None)
            states={p['name']:p for p in client.get('/api/people').get_json()['people']}
            self.assertEqual(states['Legacy']['voice_profile_state'],'stored_unavailable')
            self.assertIsNone(states['Legacy']['voice_profile_next_action'])
            self.assertEqual(states['Legacy']['name'],'Legacy')
            with sqlite3.connect(db) as conn:
                self.assertEqual(dict(conn.execute('SELECT person_id,payload FROM voice_profiles')),actual)
            app.extensions['speakerdesk']['recognition'].close(wait=True)


if __name__=='__main__':unittest.main()
