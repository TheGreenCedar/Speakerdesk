"""First-use and upgrade setup contracts at the real HTTP/download boundary."""
import hashlib
import importlib.machinery
import io
import json
import logging
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

from flask import Flask
import alignment_setup as setup
import alignment_provider as provider
import model_setup
from update_runtime import RuntimeUpdates


def sha(data):return hashlib.sha256(data).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range='', fail=False):
        super().__init__(data);self.status=status;self.headers={'Content-Range':content_range};self.fail=fail
    def read(self, count=-1):
        if self.fail and self.tell():raise urllib.error.URLError('interrupted')
        return super().read(min(count,2) if self.fail else count)


class Voice:
    def __init__(self,*args):self.state={'status':'idle','total_bytes':0}
    def supported(self):return True
    def released(self):return True
    def installed(self):return True
    def update(self,**changes):self.state.update(changes)
    def download(self):self.update(status='ready')
    def status(self,**kwargs):return {'status':self.state['status'],'installed':True,'available':True,'released':True}


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source={'model.int8.onnx':b'source','tokens.txt':b'tokens','LICENSE':b'license','README.md':b'readme'}
        self.spec=dict(name='Transcript timing',directory='test-timing-provider',optional=False,
            provider_id='test-provider',supported_languages=['en','fr','de','it','es','pt','el','nl','pl','zh','ja','ko','vi','ar'],
            timing_accuracy_calibrated_languages=['en'],bytes=6,weight_file='model.int8.onnx',
            repo='pinned/repo',revision='fixed-revision',sha256=sha(b'source'),files=list(self.source),
            download_bytes=sum(map(len,self.source.values())),file_sources={name:{'bytes':len(data)} for name,data in self.source.items()},
            file_sha256={name:sha(data) for name,data in self.source.items()},
            prepared_files={provider.DERIVED_DIRECTORY+'/weights.npz':{'bytes':7,'sha256':sha(b'weights')}})
        self.spec['file_sources']['tokens.txt'].update(repo='metadata/repo',revision='metadata-revision')
        self.metadata={'converted_weights_bytes':7,'runtime':{'test-runtime':'1'},
                       'provider_identity':{'provider_id':self.spec['provider_id']},
                       'provider_identity_sha256':sha(b'test functional provider behavior')}
        self.receipt={'model_sha256':sha(b'weights'),'manifest_sha256':provider.MANIFEST_SHA256,
                      'provider_id':self.spec['provider_id'],
                      'provider_identity_sha256':self.metadata['provider_identity_sha256'],
                      'runtime':self.metadata['runtime'],'execution':{'backend':'mlx_metal_gpu','evaluated_and_GPU_synchronized':True}}
        self.patches=[patch.dict('os.environ',{'SPEAKERDESK_MODELS':str(self.root)}),
            patch.object(setup,'ALIGNMENT_SPEC',self.spec),patch.object(provider,'ALIGNMENT_SPEC',self.spec),patch.object(model_setup,'ALIGNMENT_SPEC',self.spec),
            patch.object(model_setup,'SPECS',[self.spec]),patch.object(model_setup,'VoiceSetup',Voice),
            patch.object(provider,'MODEL_SHA256',sha(b'weights')),patch.object(setup,'manifest',return_value=self.metadata),patch.object(provider,'manifest',return_value=self.metadata),
            patch.object(setup,'require_runtime'),patch.object(provider,'require_runtime'),patch.object(provider,'require_native_runtime'),
            patch.object(setup,'prepare_model',side_effect=self.prepare),patch.object(setup,'probe',return_value=self.receipt),
            patch.object(setup.urllib.request,'urlopen',side_effect=self.network)]
        for p in self.patches:p.start()
        self.requests=[];self.failure=None;self.busy=False
        self.app=self.app_new();self.client=self.app.test_client()

    def tearDown(self):
        self.wait()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def prepare(self,folder):
        derived=folder/provider.DERIVED_DIRECTORY;derived.mkdir(exist_ok=True)
        (derived/'weights.npz').write_bytes(b'weights')

    def network(self,request,**kwargs):
        name=request.full_url.split('/')[-1].split('?')[0];offset=request.get_header('Range')
        self.requests.append((name,offset))
        if self.failure and name=='model.int8.onnx':
            mode=self.failure;self.failure=None
            if mode=='interrupt':return Response(b'source',fail=True)
            if mode=='corrupt':return Response(b'broken')
            if mode=='wrong-range':return Response(b'urce',206,'bytes 1-5/6')
        if name=='tokens.txt':self.assertIn('/metadata/repo/resolve/metadata-revision/',request.full_url)
        data=self.source[name]
        if offset:
            start=int(offset.split('=')[1].split('-')[0])
            return Response(data[start:],206,f'bytes {start}-{len(data)-1}/{len(data)}')
        return Response(data)

    def app_new(self):
        app=Flask(__name__,instance_path=str(self.root),root_path=str(self.root))
        app.extensions['speakerdesk']={'updates':RuntimeUpdates(threading.RLock(),lambda:self.busy),
                                      'recognition_preference':SimpleNamespace(enabled=lambda:False)}
        model_setup.register_setup(app,lambda *a:None)
        return app

    def wait(self):
        deadline=time.monotonic()+5
        while self.app.extensions['speakerdesk']['updates']._activities:
            if time.monotonic()>deadline:self.fail('setup did not finish')
            time.sleep(.01)

    def start(self,route='/api/setup',body=None):
        response=self.client.post(route,json=body)
        self.assertEqual(response.status_code,202,response.json);self.wait()
        return self.client.get('/api/setup').json

    def test_clean_default_download_enables_verified_timing_and_restart_discovers_it(self):
        before=self.client.get('/api/setup').json
        self.assertFalse(before['alignment']['ready'])
        after=self.start()
        self.assertTrue(after['alignment']['active']);self.assertEqual(len(self.requests),4)
        self.assertEqual(after['alignment']['supported_languages'],self.spec['supported_languages'])
        self.assertEqual(len(after['alignment']['supported_languages']),14)
        self.assertEqual(after['alignment']['timing_accuracy_calibrated_languages'],['en'])
        restart=self.app_new().test_client().get('/api/setup').json
        self.assertTrue(restart['alignment']['ready']);self.assertTrue(restart['alignment']['active'])
        self.assertEqual(self.client.patch('/api/setup/alignment',json={'enabled':False}).status_code,409)
        from pipeline import model_config
        self.assertTrue(model_config()['alignment_enabled'])
        # A previously saved off preference cannot create a degraded route.
        (self.root/'alignment-settings.json').write_text('{"enabled":false}')
        self.assertTrue(model_config()['alignment_enabled'])
        self.assertTrue(self.app_new().test_client().get('/api/setup').json['alignment']['active'])
        self.assertEqual(self.client.post('/api/setup',json={'alignment':False}).status_code,400)
        self.assertTrue(self.client.get('/api/setup').json['core_ready'])

    def test_upgrade_prepares_existing_source_without_network_and_repairs_corrupt_prepared_cache(self):
        folder=self.root/self.spec['directory'];folder.mkdir()
        for name,data in self.source.items():(folder/name).write_bytes(data)
        self.assertFalse(self.client.get('/api/setup').json['alignment']['ready'])
        self.assertTrue(self.start('/api/setup/alignment')['alignment']['active'])
        self.assertEqual(self.requests,[])
        (folder/provider.DERIVED_DIRECTORY/'weights.npz').write_bytes(b'corrupt')
        self.assertFalse(self.client.get('/api/setup').json['alignment']['ready'])
        self.assertEqual(self.client.patch('/api/setup/alignment',json={'enabled':True}).status_code,409)
        self.assertTrue(self.start('/api/setup/alignment')['alignment']['ready']);self.assertEqual(self.requests,[])

    def test_interrupted_download_survives_restart_and_resumes_safe_range(self):
        self.failure='interrupt'
        failed=self.start('/api/setup/alignment')
        self.assertEqual(failed['alignment']['status'],'failed');self.assertFalse(failed['alignment']['ready'])
        self.assertFalse(failed['core_ready'])
        self.app=self.app_new();self.client=self.app.test_client()
        self.assertEqual(self.client.get('/api/setup').json['alignment']['status'],'failed')
        self.assertIn('interrupted',self.client.get('/api/setup').json['alignment']['error'].lower())
        self.assertTrue(self.start('/api/setup/alignment')['alignment']['active'])
        self.assertIn(('model.int8.onnx','bytes=2-'),self.requests)

    def test_corrupt_and_wrong_range_downloads_never_become_ready_and_retry_recovers(self):
        for mode in ('corrupt','wrong-range'):
            with self.subTest(mode=mode):
                folder=self.root/self.spec['directory'];folder.mkdir(exist_ok=True)
                (folder/'model.int8.onnx').unlink(missing_ok=True)
                if mode=='wrong-range':(folder/'model.int8.onnx.part').write_bytes(b'so')
                self.failure=mode
                failed=self.start('/api/setup/alignment')
                self.assertEqual(failed['alignment']['status'],'failed');self.assertFalse(failed['alignment']['ready'])
                self.assertTrue(self.start('/api/setup/alignment')['alignment']['ready'])

    def test_runtime_or_GPU_failure_blocks_readiness_and_explicit_enable(self):
        self.assertTrue(self.start()['alignment']['ready'])
        with patch.object(provider,'require_runtime',side_effect=RuntimeError('runtime drift')),patch.object(setup,'require_runtime',side_effect=RuntimeError('runtime drift')):
            self.assertFalse(self.client.get('/api/setup').json['alignment']['ready'])
            self.assertEqual(self.client.patch('/api/setup/alignment',json={'enabled':True}).status_code,409)
            self.assertEqual(self.start('/api/setup/alignment')['alignment']['status'],'failed')
        self.assertFalse((self.root/self.spec['directory']/'gpu-ready.json').exists())
        with patch.object(setup,'probe',side_effect=RuntimeError('No GPU fallback')):
            failed=self.start('/api/setup/alignment')
            self.assertFalse(failed['alignment']['ready']);self.assertIn('No GPU fallback',failed['alignment']['error'])
        self.assertTrue(self.start('/api/setup/alignment')['alignment']['ready'])

    def test_active_work_and_invalid_preferences_refuse_mutation(self):
        self.busy=True
        self.assertEqual(self.client.post('/api/setup/alignment').status_code,409)
        self.assertEqual(self.client.post('/api/setup').status_code,409)
        self.assertEqual(self.client.patch('/api/setup/alignment',json={'enabled':False}).status_code,409)
        self.assertFalse(self.client.get('/api/setup').json['alignment']['can_download'])
        self.busy=False
        for body in ({'enabled':'yes'},[],None):
            self.assertEqual(self.client.patch('/api/setup/alignment',json=body).status_code,400)
        self.assertEqual(self.requests,[])

    def test_disk_exhaustion_and_recheck_failure_cannot_publish_stale_readiness(self):
        with patch.object(setup.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            failed=self.start('/api/setup/alignment')
            self.assertEqual(failed['alignment']['status'],'failed');self.assertEqual(self.requests,[])
            self.assertIn('Free at least',failed['alignment']['error'])
        self.assertTrue(self.start('/api/setup/alignment')['alignment']['ready'])
        with patch.object(setup,'probe',side_effect=RuntimeError('GPU check failed')):
            self.assertFalse(self.start('/api/setup/alignment')['alignment']['ready'])
        self.assertFalse(self.app_new().test_client().get('/api/setup').json['alignment']['ready'])

    def test_download_admission_blocks_new_inference_but_keeps_readonly_UI_available(self):
        self.app.add_url_rule('/api/meetings',endpoint='start_meeting',view_func=lambda:'unexpected start',methods=['POST'])
        started=threading.Event();release=threading.Event()
        def slow_probe(directory):started.set();release.wait(2);return self.receipt
        with patch.object(setup,'probe',side_effect=slow_probe):
            self.assertEqual(self.client.post('/api/setup/alignment').status_code,202)
            self.assertTrue(started.wait(2))
            try:
                self.assertEqual(self.client.post('/api/meetings').status_code,409)
                state=self.client.get('/api/setup')
                self.assertEqual(state.status_code,200);self.assertEqual(state.json['alignment']['status'],'downloading')
            finally:release.set();self.wait()

    def test_incompatible_required_provider_blocks_transcription_with_actionable_setup(self):
        self.client.post('/api/setup/alignment');self.wait()
        self.assertTrue(setup.available(self.root/self.spec['directory']))
        for reason in ('corrupt manifest','unqualified research provider'):
            error=ValueError('manifest corrupt') if reason=='corrupt manifest' else None
            with self.subTest(reason=reason),patch.object(setup,'manifest',side_effect=error,return_value=self.metadata),patch.object(provider,'manifest',side_effect=error,return_value=self.metadata),patch.dict(self.spec,supported_languages=['en'] if reason=='corrupt manifest' else []):
                app=self.app_new();client=app.test_client();state=client.get('/api/setup').json
                self.assertFalse(state['core_ready']);self.assertFalse(state['ready']);self.assertFalse(state['alignment']['ready'])
                self.assertFalse(state['alignment']['can_download'])
                self.assertIn('Update Speakerdesk',state['alignment']['error'])
                self.assertEqual(client.post('/api/setup/alignment').status_code,409)
                self.assertFalse(setup.available(self.root/self.spec['directory']))


class CheckWorkerTests(unittest.TestCase):
    def test_disposable_check_uses_owned_worker_and_timeout_kills_and_joins_it(self):
        import pipeline
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            events=[]
            class Child:
                returncode=0
                def poll(self):return None if self in pipeline._workers else 0
                def communicate(self,timeout=None):
                    events.append(('communicate',timeout))
                    if timeout is not None:
                        self_test.assertFalse(pipeline.workers_idle())
                        raise subprocess.TimeoutExpired('alignment-check',timeout)
                    return '',''
                def kill(self):events.append(('kill',None))
            self_test=self
            with patch.object(pipeline.subprocess,'Popen',return_value=Child()),patch.object(pipeline,'_stopping',False):
                with self.assertRaisesRegex(RuntimeError,'120-second'):
                    pipeline.run_worker('packaged-python','alignment-check',{'directory':str(folder)},folder,timeout=120)
            self.assertEqual(events,[('communicate',120),('kill',None),('communicate',None)])
            self.assertTrue(pipeline.workers_idle())


if __name__=='__main__':unittest.main()
