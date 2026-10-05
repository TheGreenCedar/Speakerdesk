"""App-managed setup with tiny fake artifacts/HTTP peers; no network, Core ML or enrollment."""
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
import voice_coreml as core
import voice_setup as setup
from voice_profiles import Calibration, VoiceModel


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body); self.status=status; self.headers=headers or {}


def fake_artifact():
    config={'model_type':'redimnet2-b6-speaker-coreml','sample_rate':16000,'input_samples':96000,
            'embedding_dimension':192,'input_name':'audio','output_name':'embedding','output_normalized':True,
            'compiled_model':'fake.mlmodelc','artifact_revision':'cpu-fixture'}
    bodies={'fake.mlmodelc/weights/weight.bin':b'fake-weight-bytes-not-a-model',
            'LICENSE':b'fake fixture notice', 'config.json':json.dumps(config).encode()}
    pin={'model_id':'cpu/fixture','revision':'cpu-fixture','compiled_model':'fake.mlmodelc','artifact_revision':'cpu-fixture',
         'files':{name:{'size':len(body),'sha256':hashlib.sha256(body).hexdigest()} for name,body in bodies.items()},
         'total_bytes':sum(map(len,bodies.values()))}
    return pin,bodies


class VoiceDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.pin,self.bodies=fake_artifact()
        for module in (setup,core):
            replacement=patch.object(module,'PIN',self.pin);replacement.start();self.addCleanup(replacement.stop)
        disk=patch.object(setup.shutil,'disk_usage',return_value=SimpleNamespace(free=50_000_000_000));disk.start();self.addCleanup(disk.stop)
        self.activation=Mock()
        self.manager=setup.VoiceSetup(self.root,threading.RLock(),self.activation,Mock())
        self.manager.released=Mock(return_value=True)
        self.fetch=patch.object(setup.urllib.request,'urlopen',side_effect=self.response).start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        self.temp.cleanup()

    def response(self, request, **kwargs):
        name=request.full_url.split('/cpu-fixture/',1)[1]
        return Response(self.bodies[name])

    def test_interrupted_download_resumes_exact_offset_and_never_enrolls(self):
        name=next(iter(self.bodies));body=self.bodies[name]
        stream=Response(b'')
        stream.read=Mock(side_effect=[body[:7],urllib.error.URLError('offline')])
        self.fetch.side_effect=[stream]
        self.manager.download()
        self.assertEqual(self.manager.state['status'],'failed')
        self.assertIn('connection',self.manager.state['error'])
        self.assertEqual((self.manager.root/(name+'.part')).read_bytes(),body[:7])
        self.activation.assert_not_called()
        requests=[]
        def resumed(request,**kwargs):
            relative=request.full_url.split('/cpu-fixture/',1)[1];requests.append(request)
            if relative==name:
                return Response(body[7:],206,{'Content-Range':f'bytes 7-{len(body)-1}/{len(body)}'})
            return Response(self.bodies[relative])
        self.fetch.side_effect=resumed
        self.manager.download()
        self.assertEqual(requests[0].get_header('Range'),'bytes=7-')
        self.assertEqual(self.manager.state['status'],'ready')
        self.assertEqual(self.manager.state['downloaded_bytes'],self.pin['total_bytes'])
        self.activation.assert_called_once_with(self.manager.root,self.manager.calibration_path)
        self.assertTrue(self.manager.installed())
        self.assertEqual(list(self.manager.root.rglob('*.part')),[])

    def test_bad_range_and_corrupt_full_file_are_rejected_then_retry_recovers(self):
        name=next(iter(self.bodies));body=self.bodies[name];partial=self.manager.root/(name+'.part')
        partial.parent.mkdir(parents=True);partial.write_bytes(body[:7])
        self.fetch.side_effect=[Response(body[7:],206,{'Content-Range':'bytes 0-99/100'})]
        self.manager.download();self.assertEqual(self.manager.state['status'],'failed')
        self.assertFalse(partial.exists());self.activation.assert_not_called()
        self.fetch.side_effect=[Response(b'X'*len(body))]
        self.manager.download();self.assertEqual(self.manager.state['status'],'failed')
        self.assertFalse(partial.exists());self.activation.assert_not_called()
        self.fetch.side_effect=self.response
        self.manager.download();self.assertEqual(self.manager.state['status'],'ready')
        self.activation.assert_called_once()

    def test_server_ignoring_range_restarts_and_verified_cache_detects_tampering(self):
        name=next(iter(self.bodies));body=self.bodies[name];partial=self.manager.root/(name+'.part')
        partial.parent.mkdir(parents=True);partial.write_bytes(body[:7])
        self.manager.download();self.assertTrue(self.manager.installed())
        self.assertEqual((self.manager.root/name).read_bytes(),body)
        self.fetch.reset_mock();self.manager.download();self.fetch.assert_not_called()
        (self.manager.root/name).write_bytes(b'X'*len(body))
        self.assertFalse(self.manager.installed())
        self.manager.download();self.assertTrue(self.manager.installed())

    def test_low_disk_and_symlink_cache_fail_before_network(self):
        with patch.object(setup.shutil,'disk_usage',return_value=SimpleNamespace(free=setup.DISK_RESERVE)):
            self.manager.download()
        self.fetch.assert_not_called();self.activation.assert_not_called()
        self.assertEqual(self.manager.state['status'],'failed')
        self.assertIn('disk',self.manager.state['error'])
        self.manager.root.rmdir();self.manager.root.symlink_to(self.root,target_is_directory=True)
        self.assertFalse(self.manager.installed());self.manager.download()
        self.fetch.assert_not_called();self.activation.assert_not_called()

    def test_pending_release_does_not_claim_download_or_recognition_readiness(self):
        self.manager.released.return_value=False
        self.manager.supported=Mock(return_value=True)
        state=self.manager.status()
        self.assertFalse(state['released']);self.assertFalse(state['can_download']);self.assertFalse(state['available'])
        self.assertEqual(state['license'],'MIT');self.fetch.assert_not_called();self.activation.assert_not_called()

    def test_activation_failure_keeps_verified_cache_and_retry_does_not_redownload(self):
        self.activation.side_effect=ValueError('Internal model configuration detail')
        self.manager.download()
        self.assertEqual(self.manager.state['status'],'failed');self.assertTrue(self.manager.installed())
        self.assertIn('retry setup',self.manager.state['error'])
        self.assertNotIn('configuration',self.manager.state['error'])
        self.fetch.reset_mock();self.activation.side_effect=None
        self.manager.download();self.fetch.assert_not_called()
        self.assertEqual(self.manager.state['status'],'ready')


class ManagedSetupAPITests(unittest.TestCase):
    def test_primary_setup_includes_voice_and_does_not_require_a_second_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);pin,bodies=fake_artifact()
            model=VoiceModel('cpu-test','fixture','a'*64,2)
            policy=Calibration(model,'cpu-fixture',.8,.1,2,2,0.,0.)
            core_spec={'name':'CPU core fixture','directory':'core','bytes':1,'files':['model.safetensors']}
            (root/'models/core').mkdir(parents=True);(root/'models/core/model.safetensors').write_bytes(b'x')
            def response(request,**kwargs):return Response(bodies[request.full_url.split('/cpu-fixture/',1)[1]])
            with (patch.dict('os.environ',{'SPEAKERDESK_MODELS':str(root/'models')}),
                  patch('model_setup.SPECS',[core_spec]),patch.object(core,'PIN',pin),patch.object(setup,'PIN',pin),
                  patch.object(setup.VoiceSetup,'released',return_value=True),
                  patch.object(setup.VoiceSetup,'supported',return_value=True),
                  patch.object(setup.shutil,'disk_usage',return_value=SimpleNamespace(free=50_000_000_000)),
                  patch.object(setup.urllib.request,'urlopen',side_effect=response) as fetch,
                  patch.object(core,'load_approved_runtime',return_value=(SimpleNamespace(model=model),policy))):
                app=create_app(root/'data');client=app.test_client()
                token=re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').text)[1]
                try:
                    initial=client.get('/api/setup').json
                    self.assertTrue(initial['core_ready']);self.assertFalse(initial['ready'])
                    self.assertTrue(initial['voice']['enabled'])
                    self.assertEqual(initial['total_bytes'],1+pin['total_bytes'])
                    self.assertEqual(client.post('/api/setup',headers={'X-Speakerdesk-Token':token}).status_code,202)
                    deadline=time.monotonic()+2
                    while time.monotonic()<deadline and not client.get('/api/setup').json['ready']:time.sleep(.01)
                    state=client.get('/api/setup').json
                    self.assertTrue(state['ready']);self.assertEqual(state['status'],'ready')
                    self.assertEqual([m['name'] for m in state['models']],['CPU core fixture','Voice recognition'])
                    self.assertTrue(state['voice']['available']);self.assertEqual(fetch.call_count,len(bodies))
                    self.assertEqual(client.get('/api/people').json['people'],[])
                finally:
                    app.extensions['speakerdesk']['meetings'].close()
                    app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_authenticated_download_updates_people_and_reopens_without_manual_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);pin,bodies=fake_artifact();started=threading.Event();release=threading.Event()
            model=VoiceModel('cpu-test','fixture','a'*64,2)
            backend=SimpleNamespace(model=model)
            policy=Calibration(model,'cpu-fixture',.8,.1,2,2,0.,0.)
            calls=[]
            def response(request,**kwargs):
                calls.append(request.full_url);started.set();release.wait(timeout=3)
                return Response(bodies[request.full_url.split('/cpu-fixture/',1)[1]])
            with (patch.dict('os.environ',{'SPEAKERDESK_MODELS':str(root/'models')}),
                  patch.object(core,'PIN',pin),patch.object(setup,'PIN',pin),
                  patch.object(setup.VoiceSetup,'released',return_value=True),
                  patch.object(setup.VoiceSetup,'supported',return_value=True),
                  patch.object(setup.shutil,'disk_usage',return_value=SimpleNamespace(free=50_000_000_000)),
                  patch.object(setup.urllib.request,'urlopen',side_effect=response),
                  patch.object(core,'load_approved_runtime',return_value=(backend,policy)) as load):
                app=create_app(root/'data');client=app.test_client()
                token=re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').get_data(as_text=True))[1]
                headers={'X-Speakerdesk-Token':token}
                try:
                    self.assertFalse(client.get('/api/people').json['voice_available'])
                    self.assertEqual(client.post('/api/setup/voice').status_code,403)
                    self.assertEqual(calls,[])
                    person=client.post('/api/people',headers=headers,json={'name':'Synthetic name'}).json
                    self.assertEqual(client.post('/api/setup/voice',headers=headers).status_code,202)
                    self.assertTrue(started.wait(timeout=2))
                    self.assertEqual(client.post('/api/setup/voice',headers=headers).status_code,202)
                    self.assertEqual(client.post('/api/setup',headers=headers).status_code,409)
                    release.set()
                    deadline=time.monotonic()+3
                    while time.monotonic()<deadline and not client.get('/api/people').json['voice_available']:
                        time.sleep(.01)
                    state=client.get('/api/people').json
                    self.assertTrue(state['voice_available']);self.assertEqual(state['people'][0]['id'],person['id'])
                    self.assertFalse(state['people'][0]['voice_saved'])
                    self.assertEqual(len(calls),len(bodies));load.assert_called_once()
                    self.assertTrue(client.get('/api/setup').json['voice']['installed'])
                    reopened=create_app(root/'data')
                    try:
                        self.assertTrue(reopened.test_client().get('/api/people').json['voice_available'])
                        self.assertFalse(reopened.test_client().get('/api/people').json['people'][0]['voice_saved'])
                    finally:
                        reopened.extensions['speakerdesk']['meetings'].close()
                        reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
                finally:
                    release.set();app.extensions['speakerdesk']['meetings'].close()
                    app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_unsupported_platform_refuses_setup_without_requesting_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(setup.urllib.request,'urlopen') as fetch, \
             patch.object(setup,'require_runtime',side_effect=ValueError('CPU fixture unsupported platform')):
            app=create_app(Path(temporary)/'data');client=app.test_client()
            token=re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').get_data(as_text=True))[1]
            try:
                self.assertEqual(client.post('/api/setup/voice',headers={'X-Speakerdesk-Token':token}).status_code,409)
                self.assertIn('Apple Silicon',client.get('/api/setup').json['voice']['message'])
                fetch.assert_not_called()
            finally:
                app.extensions['speakerdesk']['meetings'].close()
                app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':unittest.main()
