"""Updater admission at real HTTP/runtime boundaries; fixture storage and peers."""
import io
import json
from pathlib import Path
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from app import create_app
from update_runtime import PREFIX, MAX_FRAME_BYTES, decode_frame
from support.meeting_harness import MeetingHarness


class UpdateFixture:
    def connect_updates(self):
        self.updates = self.app.extensions['speakerdesk']['updates']
        self.frames = []
        self.preparation = 0
        self.updates.connect(lambda frame: self.frames.append(decode_frame(frame)))

    def operation(self, op, **fields):
        return self.client.post('/api/updates', headers=self.headers, json={'op': op, **fields})

    def prepare_update(self):
        response = self.operation('check')
        self.assertEqual(response.status_code, 202, response.json)
        attempt = response.json['id']
        self.assertTrue(self.updates.handle({'id': attempt, 'op': 'status', 'state': 'available', 'version': '0.6.2'}))
        self.assertEqual(self.operation('download').status_code, 202)
        self.assertTrue(self.updates.handle({'id': attempt, 'op': 'status', 'state': 'ready'}))
        self.assertEqual(self.operation('install').status_code, 202)
        self.begin_preparation(attempt)
        self.assertEqual(self.operation('editor_ready', id=attempt, preparation=self.preparation).status_code, 202)
        return attempt

    def begin_preparation(self, attempt):
        self.preparation += 1
        self.assertTrue(self.updates.handle({'op':'status','id':attempt,'state':'preparing','preparation':self.preparation}))

    def native(self, op, attempt, *, preparation=None, **fields):
        return self.updates.handle({'op':op,'id':attempt,'preparation':self.preparation if preparation is None else preparation,**fields})

    def reserve(self, attempt):
        return self.native('reserve',attempt)


class UpdateApiTests(UpdateFixture, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = patch.dict('os.environ', {'SPEAKERDESK_MODELS': str(self.root/'models')})
        self.models.start()
        self.app = create_app(self.root/'recordings')
        self.client = self.app.test_client()
        token = re.search(r'name="speakerdesk-token" content="([^"]+)"', self.client.get('/').text)[1]
        self.headers = {'X-Speakerdesk-Token': token}

    def tearDown(self):
        self.app.extensions['speakerdesk']['meetings'].close()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.models.stop()
        self.temp.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if predicate(): return
            time.sleep(.005)
        self.fail('Fixture work did not finish.')

    def test_token_fixed_operations_and_clean_editor_acknowledgement_are_required(self):
        self.assertEqual(self.operation('check').status_code, 409)
        self.connect_updates()
        self.assertEqual(self.client.post('/api/updates', json={'op': 'check'}).status_code, 403)
        for body in ({'op': 'check', 'url': 'https://example.test'}, {'op': 'install', 'path': '/tmp/app'}, {'op': 'downgrade'}):
            self.assertEqual(self.client.post('/api/updates', headers=self.headers, json=body).status_code, 400)
        self.assertEqual(self.frames, [])
        attempt = self.operation('check').json['id']
        self.updates.handle({'op': 'status', 'id': attempt, 'state': 'available'})
        self.operation('download')
        self.assertTrue(self.updates.handle({'op':'status','id':attempt,'state':'verifying'}))
        self.assertEqual(self.operation('check').status_code,409)
        self.assertEqual(self.operation('install').status_code,409)
        self.updates.handle({'op': 'status', 'id': attempt, 'state': 'ready'})
        self.operation('install')
        self.assertFalse(self.reserve(attempt))
        self.assertEqual(self.operation('editor_ready',id=attempt,preparation=1).status_code,409)
        self.begin_preparation(attempt)
        self.assertEqual(self.operation('editor_ready', id=attempt-1,preparation=self.preparation).status_code, 409)
        self.assertEqual(self.operation('editor_error', id=attempt,preparation=self.preparation, error='Save failed.').status_code, 202)
        self.assertFalse(self.reserve(attempt))
        self.assertEqual(self.operation('editor_ready', id=attempt,preparation=self.preparation).status_code, 202)
        self.assertTrue(self.reserve(attempt))

    def test_reservation_blocks_real_mutation_routes_and_stale_release_does_not_open_them(self):
        self.connect_updates()
        attempt = self.prepare_update()
        self.assertTrue(self.reserve(attempt))
        routes = [('/api/people', {'name': 'Fixture'}), ('/api/meetings', {'name': 'Fixture', 'sources': ['microphone']}),
                  ('/api/jobs', {}), ('/api/setup', {}), ('/api/setup/voice', {}),
                  ('/api/jobs/'+'a'*32+'/retry', {}), ('/api/people/'+'b'*32+'/voice', {'consent': True})]
        for path, body in routes:
            with self.subTest(path=path):
                response = self.client.post(path, headers=self.headers, json=body)
                self.assertEqual(response.status_code, 409, response.json)
                self.assertIn('preparing an update', response.json['error'])
        self.assertEqual(self.client.patch('/api/recognition', headers=self.headers, json={'enabled': False}).status_code, 409)
        for method,path in (('put','/api/jobs/'+'a'*32+'/transcript'),
                            ('patch','/api/jobs/'+'a'*32+'/segments/fixture'),
                            ('delete','/api/jobs/'+'a'*32),
                            ('patch','/api/preferences/language')):
            self.assertEqual(getattr(self.client,method)(path,headers=self.headers,json={}).status_code,409)
        self.assertEqual(self.client.get('/api/jobs').json, [])
        self.assertFalse(self.native('release',attempt-1))
        self.assertTrue(self.client.get('/api/updates').json['reserved'])
        self.assertTrue(self.native('release',attempt))
        self.assertTrue(self.native('release',attempt))
        self.updates.handle({'op': 'status', 'id': attempt, 'state': 'cancelled'})
        self.assertEqual(self.client.post('/api/people', headers=self.headers, json={'name': 'Fixture'}).status_code, 201)
        next_attempt = self.prepare_update()
        self.assertGreater(next_attempt, attempt)
        self.assertTrue(self.reserve(next_attempt))
        self.assertFalse(self.native('release',attempt))
        self.assertTrue(self.client.get('/api/updates').json['reserved'])

    def test_cancelled_preparation_cannot_acknowledge_or_release_retry_on_same_attempt(self):
        self.connect_updates()
        attempt=self.prepare_update()
        old_preparation=self.preparation
        self.assertTrue(self.reserve(attempt))
        self.assertEqual(self.operation('cancel').status_code,202)
        self.assertTrue(self.native('release',attempt))
        self.assertTrue(self.native('status',attempt,state='ready'))
        self.assertEqual(self.operation('install').status_code,202)
        # The previous preparation's late ready/cancel status cannot reset the
        # new pending install before its native token arrives.
        self.assertFalse(self.native('status',attempt,preparation=old_preparation,state='ready'))
        self.begin_preparation(attempt)
        new_preparation=self.preparation
        self.assertGreater(new_preparation,old_preparation)
        self.assertEqual(self.operation('editor_error',id=attempt,preparation=new_preparation,error='New local edits are not saved.').status_code,202)
        self.assertEqual(self.operation('editor_ready',id=attempt,preparation=old_preparation).status_code,409)
        self.assertFalse(self.native('reserve',attempt,preparation=old_preparation))
        self.assertFalse(self.reserve(attempt))
        self.assertEqual(self.operation('editor_ready',id=attempt,preparation=new_preparation).status_code,202)
        self.assertTrue(self.reserve(attempt))
        self.assertFalse(self.native('release',attempt,preparation=old_preparation))
        self.assertFalse(self.native('shutdown',attempt,preparation=old_preparation))
        self.assertTrue(self.client.get('/api/updates').json['reserved'])
        self.assertEqual(self.client.post('/api/people',headers=self.headers,json={'name':'Still protected'}).status_code,409)
        self.assertTrue(self.native('release',attempt,preparation=new_preparation))
        self.assertEqual(self.client.post('/api/people',headers=self.headers,json={'name':'Editable again'}).status_code,201)

    def test_native_guard_preserves_save_and_cancel_then_keeps_unknown_error_protected(self):
        self.connect_updates()
        attempt=self.prepare_update()
        self.assertTrue(self.native('status',attempt,state='preparing',reserved=True,cancellable=True))
        status=self.client.get('/api/updates').json
        self.assertTrue(status['reserved'])
        self.assertTrue(status['cancellable'])
        # Native exit protection is already held before the editor flushes its
        # save. It must not impersonate the runtime's later mutation lock.
        jid='d'*32
        manager=self.app.extensions['speakerdesk']['meetings']
        manager.put({'id':jid,'name':'Save fixture','status':'ready','duration':1.,'revision':0,'created':1.,
            'document':{'schema_version':1,'speakers':{'speaker_0':'Speaker 1'},'segments':[
                {'id':'words','speaker':'speaker_0','start':0.,'end':1.,'text':'Old words','review':False}]}})
        document=self.client.get('/api/jobs/'+jid).json['document']
        document['segments'][0]['text']='Saved during preparation'
        response=self.client.put('/api/jobs/'+jid+'/transcript',headers=self.headers,json={'revision':0,'document':document})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['document']['segments'][0]['text'],'Saved during preparation')
        self.assertEqual(self.operation('cancel').status_code,202)
        self.assertTrue(self.native('status',attempt,state='error',reserved=True,cancellable=False,error='Reservation acknowledgement is unknown.'))
        self.assertTrue(self.client.get('/api/updates').json['reserved'])
        self.assertFalse(self.client.get('/api/updates').json['cancellable'])
        self.assertEqual(self.operation('check').status_code,409)
        # A malformed/stale native report must not clear the protection flag.
        self.assertFalse(self.native('status',attempt,state='ready',reserved='false',cancellable=False))
        self.assertFalse(self.native('status',attempt,state='ready',reserved=False,cancellable=1))
        self.assertTrue(self.client.get('/api/updates').json['reserved'])
        self.assertTrue(self.native('release',attempt))
        self.assertTrue(self.native('status',attempt,state='ready',reserved=False,cancellable=False))
        self.assertFalse(self.client.get('/api/updates').json['reserved'])
        self.assertEqual(self.operation('check').status_code,202)

    def test_native_reservation_cancel_matches_status_until_shutdown_commits(self):
        self.connect_updates()
        attempt = self.prepare_update()
        self.assertTrue(self.native('status', attempt, state='stopping', reserved=True, cancellable=True))
        self.assertEqual(self.operation('cancel').status_code, 202)
        self.assertEqual(self.frames[-1]['op'], 'cancel')
        # An acknowledged reservation alone is still releasable. Keep all
        # mutations guarded until native actually releases its reservation.
        self.assertTrue(self.reserve(attempt))
        self.assertEqual(self.operation('cancel').status_code, 202)
        self.assertEqual(self.client.post('/api/people', headers=self.headers, json={'name':'Guarded'}).status_code, 409)
        self.assertTrue(self.native('status', attempt, state='stopping', reserved=True, cancellable=False))
        self.assertEqual(self.operation('cancel').status_code, 409)
        # Even a delayed cancellable status cannot reopen committed shutdown.
        self.assertTrue(self.native('status', attempt, state='stopping', reserved=True, cancellable=True))
        self.assertTrue(self.native('shutdown', attempt))
        self.assertTrue(self.native('status', attempt, state='stopping', reserved=True, cancellable=True))
        self.assertEqual(self.operation('cancel').status_code, 409)

    def test_import_preparation_lease_survives_queued_and_running_work(self):
        self.connect_updates()
        attempt = self.prepare_update()
        entered, release = threading.Event(), threading.Event()
        def prepare_audio(source, destination):
            entered.set()
            self.assertTrue(release.wait(3))
            destination.write_bytes(b'fixture audio')
            return 1.
        with patch('app.normalize', side_effect=prepare_audio):
            try:
                response = self.client.post('/api/jobs', headers=self.headers,
                                            data={'files': (io.BytesIO(b'fixture source'), 'fixture.wav')})
                self.assertEqual(response.status_code, 201, response.json)
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.reserve(attempt))
                self.assertFalse(self.native('shutdown',attempt))
                self.assertEqual(self.client.get('/api/jobs').json[0]['status'], 'preparing')
            finally:
                release.set()
                self.app.extensions['speakerdesk']['executor'].submit(lambda: None).result(3)
        self.assertEqual(self.client.get('/api/jobs').json[0]['status'], 'uploaded')
        self.assertTrue(self.reserve(attempt))

    def test_voice_setup_thread_holds_admission_until_download_returns(self):
        self.connect_updates()
        attempt = self.prepare_update()
        voice = self.app.extensions['speakerdesk']['voice_setup']
        entered, release = threading.Event(), threading.Event()
        def download():
            entered.set()
            self.assertTrue(release.wait(3))
            voice.update(status='ready')
        with patch.object(voice, 'supported', return_value=True), patch.object(voice, 'download', side_effect=download):
            try:
                self.assertEqual(self.client.post('/api/setup/voice', headers=self.headers).status_code, 202)
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.reserve(attempt))
                self.assertFalse(self.native('shutdown',attempt))
            finally: release.set()
            self.wait_for(lambda: voice.state['status'] == 'ready')
            self.wait_for(lambda: self.reserve(attempt))

    def test_core_model_setup_holds_admission_before_a_download_is_scheduled(self):
        from types import SimpleNamespace
        self.connect_updates()
        attempt=self.prepare_update()
        voice=self.app.extensions['speakerdesk']['voice_setup']
        entered,release=threading.Event(),threading.Event()
        def disk_check(_root):
            entered.set()
            self.assertTrue(release.wait(3))
            return SimpleNamespace(free=0)
        with patch.object(voice,'supported',return_value=True), patch('model_setup.shutil.disk_usage',side_effect=disk_check):
            try:
                self.assertEqual(self.client.post('/api/setup',headers=self.headers,json={}).status_code,202)
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.reserve(attempt))
            finally:release.set()
            self.wait_for(lambda:self.reserve(attempt))
        self.assertEqual(list((self.root/'models').rglob('*.safetensors')),[])

    def test_automatic_voice_check_on_completed_recording_blocks_reservation(self):
        from test_voice_recognition import Backend, POLICY, MODEL, segment
        from voice_profiles import VoiceClip, make_profile
        backend=Backend()
        backend.release.clear()
        self.app.extensions['speakerdesk']['voice_runtime']=(backend,POLICY)
        store=self.app.extensions['speakerdesk']['people']
        recognizer=self.app.extensions['speakerdesk']['recognition']
        person=store.create('Fixture person')
        clips=[VoiceClip('enrollment','speaker_3',str(i),i*4.,i*4.+3.) for i in range(2)]
        store.remember(person['id'],make_profile(MODEL,[(1.,0.),(1.,0.)],clips),True)
        jid='c'*32
        folder=self.root/'recordings'/jid
        folder.mkdir()
        (folder/'audio.wav').write_bytes(b'CPU fixture, not real recording')
        recognizer.put({'id':jid,'name':'Fixture','kind':'meeting','status':'ready','duration':120.,
            'revision':0,'created':1.,'document':{'schema_version':1,'speakers':{'speaker_0':'Speaker 1'},
                'segments':[segment('one',0.,3.),segment('two',4.,7.)],
                'provenance':{'kind':'local_inference'},'warnings':[]}})
        self.connect_updates()
        attempt=self.prepare_update()
        try:
            recognizer.observe(jid)
            self.assertTrue(backend.started.wait(2))
            self.assertFalse(self.reserve(attempt))
            self.assertEqual(recognizer.get(jid)['status'],'ready')
        finally:
            backend.release.set()
            recognizer.executor.submit(lambda:None).result(3)
        self.assertTrue(self.reserve(attempt))
        self.assertEqual((folder/'audio.wav').read_bytes(),b'CPU fixture, not real recording')


class UpdateMeetingTests(UpdateFixture, MeetingHarness, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.connect_updates()

    def test_recording_paused_and_finishing_refuse_update_without_changing_saved_audio(self):
        self.scenario = 'delayed_stop'
        attempt = self.prepare_update()
        jid = self.start(['microphone'])
        self.wait_for(lambda: self.job(jid)['status'] == 'recording')
        self.assertFalse(self.reserve(attempt))
        self.control(jid, 'pause')
        self.wait_for(lambda: self.job(jid)['status'] == 'paused')
        paused_audio = (self.root/jid/'audio.wav').read_bytes()
        paused_document = self.job(jid)['document']
        self.assertFalse(self.reserve(attempt))
        self.assertFalse(self.native('shutdown',attempt))
        self.assertEqual((self.root/jid/'audio.wav').read_bytes(), paused_audio)
        self.assertEqual(self.job(jid)['document'], paused_document)
        self.assertTrue(any(child.poll() is None for child in self.children))
        self.control(jid, 'resume')
        self.wait_for(lambda: self.job(jid)['status'] == 'recording')
        self.control(jid, 'stop')
        self.wait_for(lambda: self.job(jid)['status'] == 'finishing')
        self.assertFalse(self.reserve(attempt))
        (self.root/jid/'capture-release').touch()
        self.wait_for(lambda: self.manager.jid is None and not self.manager.thread.is_alive())
        saved_audio = (self.root/jid/'audio.wav').read_bytes()
        saved_document = self.job(jid)['document']
        self.assertTrue(self.reserve(attempt))
        self.assertEqual((self.root/jid/'audio.wav').read_bytes(), saved_audio)
        self.assertEqual(self.job(jid)['document'], saved_document)
        self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_capture_start_and_update_reservation_admit_exactly_one_owner(self):
        attempt = self.prepare_update()
        barrier = threading.Barrier(2)
        results = {}
        def start():
            with self.app.test_client() as client:
                barrier.wait(2)
                results['capture'] = client.post('/api/meetings', headers=self.headers,
                    json={'name': 'Race fixture', 'language': 'en', 'sources': ['microphone']})
        worker = threading.Thread(target=start)
        worker.start()
        barrier.wait(2)
        results['update'] = self.reserve(attempt)
        worker.join(3)
        self.assertFalse(worker.is_alive())
        response = results['capture']
        self.assertEqual(int(response.status_code == 201)+int(results['update']), 1)
        if results['update']:
            self.assertEqual(response.status_code, 409)
            self.assertEqual(self.client.get('/api/jobs').json, [])
            self.assertEqual(self.children, [])
        else:
            self.assertEqual(response.status_code, 201, response.json)
            jid = response.json['id']
            self.wait_for(lambda: self.job(jid)['status'] == 'recording')
            self.control(jid, 'stop')
            self.wait_for(lambda: self.manager.jid is None)

    def test_model_startup_owns_runtime_admission_before_capture_exists(self):
        self.scenario='startup'
        attempt=self.prepare_update()
        jid=self.start(['microphone'])
        self.assertEqual(self.job(jid)['status'],'starting')
        self.assertIsNone(self.manager.capture)
        self.assertFalse(self.reserve(attempt))
        self.assertFalse(self.native('shutdown',attempt))
        self.control(jid,'stop')
        self.wait_for(lambda:self.manager.jid is None and not self.manager.thread.is_alive())
        self.assertTrue(self.reserve(attempt))

    def test_check_runs_during_capture_but_download_and_install_do_not_emit_native_operations(self):
        jid=self.start(['microphone'])
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        attempt=self.operation('check').json['id']
        self.updates.handle({'op':'status','id':attempt,'state':'available'})
        self.assertEqual(self.operation('download').status_code,409)
        self.assertEqual(self.frames,[{'id':attempt,'op':'check'}])
        self.control(jid,'pause')
        self.wait_for(lambda:self.job(jid)['status']=='paused')
        self.updates.handle({'op':'status','id':attempt,'state':'ready'})
        audio=(self.root/jid/'audio.wav').read_bytes()
        self.assertEqual(self.operation('install').status_code,409)
        self.assertEqual(self.frames,[{'id':attempt,'op':'check'}])
        self.assertEqual((self.root/jid/'audio.wav').read_bytes(),audio)
        self.control(jid,'resume')
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'stop')
        self.wait_for(lambda:self.manager.jid is None and not self.manager.thread.is_alive())
        self.assertEqual(self.operation('install').status_code,202)


class SidecarUpdateProcessTests(unittest.TestCase):
    def test_validated_idle_shutdown_acknowledgement_is_followed_by_real_process_exit(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            # Preserve a bounded live stack if hosted startup stalls after its
            # imports. run_path executes the same entry point and control pipe.
            diagnostic = ('import faulthandler,runpy,sys; '
                          'faulthandler.dump_traceback_later(5); '
                          'runpy.run_path(sys.argv[1],run_name="__main__")')
            process = subprocess.Popen([sys.executable, '-c', diagnostic,
                                        str(Path(__file__).resolve().parents[1]/'packaging/sidecar.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, 'SPEAKERDESK_HOME': temporary, 'PYTHONDONTWRITEBYTECODE': '1',
                     'PYTHONPROFILEIMPORTTIME': '1'})
            # Drain stderr during cold imports and HTTP requests; a full pipe
            # must not block startup. Keep bounded failure/import diagnostics.
            stderr_tail = ''
            import_costs = []
            def drain_stderr():
                nonlocal stderr_tail
                for line in process.stderr:
                    stderr_tail = (stderr_tail + line)[-65536:]
                    match = re.match(r'import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)', line)
                    if match:
                        import_costs.append((int(match[2]), match[3]))
                        import_costs.sort(reverse=True)
                        del import_costs[20:]
            stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
            stderr_reader.start()
            try:
                def read_line(timeout=5):
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdout, selectors.EVENT_READ)
                        if not selector.select(timeout):
                            self.fail(f'Sidecar output timed out after {timeout}s; exit={process.poll()}; stderr={stderr_tail}')
                    return process.stdout.readline()
                # Cold source imports on the package runner have a separate
                # finite budget; all protocol acknowledgements retain 5s.
                line = read_line(timeout=30)
                print(f'Sidecar startup_seconds={time.monotonic()-started:.3f}; '
                      f'top_cumulative_import_us={sorted(import_costs, reverse=True)[:5]}')
                if not line.startswith('SPEAKERDESK_URL='):
                    # Werkzeug reports denied bind/listen as one stderr line
                    # and exits. Preserve that actual cause instead of an empty
                    # stdout assertion; this does not retry a denied launch.
                    if line == '':
                        stderr_reader.join(timeout=1)
                        startup_error = stderr_tail
                    else:
                        startup_error = 'Unexpected startup output: ' + repr(line)
                    self.fail(f'Sidecar bootstrap failed (exit={process.poll()}): {startup_error[:65536]}')
                base = line.strip().split('=', 1)[1]
                with urllib.request.urlopen(base, timeout=3) as response:
                    token = re.search(r'name="speakerdesk-token" content="([^"]+)"', response.read().decode())[1]
                def post(op, **fields):
                    body=json.dumps({'op':op, **fields}).encode()
                    req=urllib.request.Request(base+'/api/updates', body,
                        {'X-Speakerdesk-Token':token, 'Content-Type':'application/json'})
                    with urllib.request.urlopen(req, timeout=3) as response: return json.load(response)
                def command(op, **fields):
                    process.stdin.write(PREFIX+json.dumps({'op':op, 'id':attempt, **fields})+'\n')
                    process.stdin.flush()
                attempt=post('check')['id']
                self.assertEqual(decode_frame(read_line()), {'op':'check', 'id':attempt})
                command('status', state='available')
                # HTTP requests and stdin commands arrive on independent threads.
                def wait_state(state):
                    deadline=time.monotonic()+3
                    while time.monotonic()<deadline:
                        with urllib.request.urlopen(base+'/api/updates', timeout=3) as response:
                            if json.load(response)['state']==state:return
                        time.sleep(.005)
                    self.fail('Sidecar status did not arrive.')
                wait_state('available'); post('download'); read_line()
                command('status', state='ready'); wait_state('ready')
                post('install'); read_line()
                preparation=1
                command('status',state='preparing',preparation=preparation)
                # The HTTP install result is already preparing; wait for the
                # native token rather than assuming its pipe command arrived.
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    with urllib.request.urlopen(base+'/api/updates',timeout=3) as response:
                        if json.load(response).get('preparation')==preparation:break
                    time.sleep(.005)
                else:self.fail('Native preparation token did not arrive.')
                post('editor_ready', id=attempt,preparation=preparation); read_line()
                # An oversized line and a stale command cannot grant a reservation.
                process.stdin.write(PREFIX+'x'*(MAX_FRAME_BYTES+1)+'\n')
                process.stdin.write(PREFIX+json.dumps({'op':'reserve','id':attempt-1})+'\n')
                process.stdin.write(PREFIX+json.dumps({'op':'status','id':attempt,'state':{}})+'\n')
                process.stdin.flush()
                command('shutdown',preparation=preparation)
                rejected=decode_frame(read_line())
                self.assertEqual((rejected['op'],rejected['ok']), ('shutdown_ready',False))
                self.assertIsNone(process.poll())
                command('reserve',preparation=preparation)
                self.assertTrue(decode_frame(read_line())['ok'])
                command('shutdown',preparation=preparation)
                acknowledgement=decode_frame(read_line())
                self.assertEqual(acknowledgement, {'id':attempt,'preparation':preparation,'op':'shutdown_ready','ok':True})
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process.poll() is None:process.kill()
                process.wait(timeout=5)
                stderr_reader.join(timeout=2)
                process.stdin.close()
                process.stdout.close()
                if not stderr_reader.is_alive():
                    process.stderr.close()
                self.assertFalse(stderr_reader.is_alive(), 'Sidecar stderr reader did not join within its cleanup budget.')


if __name__ == '__main__': unittest.main()
