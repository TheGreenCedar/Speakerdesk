"""Strict voice-execution regressions; predictors are peers, never neural models."""
import json
import re
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import voice_coreml as core
import voice_setup as setup
from voice_profiles import VoiceClip
LEGACY_CALIBRATION=Path(core.__file__).with_name('voice_calibration.json')


class VoiceAcceleratorPolicyTests(unittest.TestCase):
    def test_every_cpu_inclusive_policy_rejects_before_audio_or_warm_prediction(self):
        for policy in ('ALL','CPU_ONLY','CPU_AND_GPU','CPU_AND_NE'):
            with self.subTest(policy=policy):
                backend=core.ReDimNet2CoreML('unused',policy)
                predictor=Mock();backend._predictor=predictor
                with patch.object(core,'read_clip',return_value=None) as read:
                    with self.assertRaisesRegex(ValueError,'GPU/ANE'):
                        backend.embed('unused',VoiceClip('m','t','s',0.,3.))
                read.assert_not_called();predictor.predict.assert_not_called()

    def test_activation_refuses_legacy_approved_calibration_without_touching_artifacts(self):
        original=LEGACY_CALIBRATION.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(core,'verify_artifact') as verify,patch.object(core,'require_runtime') as runtime:
                with self.assertRaisesRegex(ValueError,'GPU/ANE'):
                    core.load_approved_runtime(LEGACY_CALIBRATION,model_dir=temporary)
                verify.assert_not_called();runtime.assert_not_called()
        self.assertEqual(LEGACY_CALIBRATION.read_bytes(),original)
        self.assertTrue(json.loads(original)['approved_for_recognition'])
        self.assertTrue(core.model_identity('ALL').revision.endswith(':ALL'))

    def test_setup_reports_unavailable_and_preserves_saved_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager=setup.VoiceSetup(temporary,threading.RLock(),Mock(),Mock())
            manager.calibration_path=LEGACY_CALIBRATION
            with patch.object(manager,'installed',return_value=True),patch.object(manager,'supported',return_value=True):
                status=manager.status(available=True)
            self.assertFalse(status['released']);self.assertFalse(status['available']);self.assertFalse(status['can_download'])
            self.assertIn('GPU/ANE',status['message'])
            manager.activate.assert_not_called()

    def test_stale_runtime_cannot_claim_people_availability_or_block_core_setup(self):
        from app import create_app
        import model_setup
        with tempfile.TemporaryDirectory() as temporary,patch.object(model_setup,'SPECS',[]),patch.object(setup.VoiceSetup,'supported',return_value=True),patch.object(setup.VoiceSetup,'installed',return_value=True):
            root=Path(temporary)
            with patch.dict('os.environ',{'SPEAKERDESK_MODELS':str(root/'models'),
                                         'SPEAKERDESK_VOICE_CONFIG':str(LEGACY_CALIBRATION)}):
                app=create_app(root/'data')
            state=app.extensions['speakerdesk']
            try:
                backend=core.ReDimNet2CoreML(root/'models','ALL')
                calibration=core.approved_calibration(core.read_voice_config(LEGACY_CALIBRATION),backend.model)
                state['voice_runtime']=(backend,calibration)
                client=app.test_client()
                store=state['people'];person=store.create('Owned policy fixture')
                saved={'person_id':person['id'],'model':backend.model.payload(),'vector':[1.]+[0.]*191,'version':'fixture'}
                with store.db() as conn:
                    conn.execute('INSERT INTO voice_profiles(person_id,payload) VALUES (?,?)',(person['id'],json.dumps(saved)))
                before=store.profiles()
                cache=root/'models'/'sentinel';cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(b'owned cached fixture')
                people=client.get('/api/people').json;status=client.get('/api/setup').json
                self.assertFalse(people['voice_available']);self.assertFalse(status['voice']['available'])
                self.assertTrue(status['core_ready']);self.assertTrue(status['ready'])
                self.assertEqual(status['status'],'ready');self.assertEqual(status['total_bytes'],0)
                self.assertEqual(store.profiles(),before);self.assertEqual(cache.read_bytes(),b'owned cached fixture')
                self.assertEqual(people['people'][0]['name'],'Owned policy fixture')
                token=re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').get_data(as_text=True))[1]
                with patch.object(setup.urllib.request,'urlopen') as network,patch.object(state['voice_setup'],'download') as download:
                    response=client.post('/api/setup/voice',headers={'X-Speakerdesk-Token':token})
                    self.assertEqual(response.status_code,409);self.assertIn('GPU/ANE',response.json['error'])
                    network.assert_not_called();download.assert_not_called()
            finally:
                state['recognition'].close(wait=True)
                state['meetings'].close();state['executor'].shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':unittest.main()
