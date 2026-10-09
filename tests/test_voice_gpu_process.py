"""Real disposable processes protect admission, framing, timeout and reaping."""
from concurrent.futures import ThreadPoolExecutor
import json
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import pipeline
import voice_worker_registry as registry
import voice_gpu_process as process
from voice_gpu_runtime import RELEASE_CALIBRATION, release_policy
from voice_profiles import VoiceClip
from update_runtime import RuntimeUpdates, UpdateReserved
from update_runtime import PREFIX


class OwnedVoiceProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clip = VoiceClip('a'*32, 'speaker_0', 'segment-0', 0., 3.)
        self.audio = self.root/self.clip.meeting_id/'audio.wav'
        self.audio.parent.mkdir();self.audio.write_bytes(b'disposable nonaudio IPC fixture')
        self.model, _ = release_policy()
        for name, value in [('_workers',set()), ('_stopping',False)]:
            p = patch.object(pipeline, name, value);p.start();self.addCleanup(p.stop)
        p = patch.object(registry, '_owned_workers', set());p.start();self.addCleanup(p.stop)
        original = registry.launch_owned_worker
        def launch(command, **options):
            return original([sys.executable, '-u', str(Path(__file__).parent/'voice_process_fixture.py'), command[-1]], **options)
        p = patch.object(process, 'launch_owned_worker', side_effect=launch);p.start();self.addCleanup(p.stop)
        self.updates = RuntimeUpdates(threading.RLock(), lambda:False)
        self.backend = process.OwnedVoiceBackend(self.root, RELEASE_CALIBRATION, self.model,
                                                self.updates, self.root, timeout=2)
        self.addCleanup(self.backend.close)

    def assert_reaped(self, worker):
        self.assertIsNotNone(worker.poll())
        self.assertNotIn(worker, pipeline._workers)
        self.assertNotIn(worker, registry._owned_workers)
        self.assertTrue(pipeline.workers_idle())
        with self.assertRaises(ProcessLookupError):os.killpg(worker.pid, 0)

    def test_resident_worker_handles_distinct_callers_then_joins_on_close(self):
        self.assertTrue(pipeline.workers_idle())
        first = self.backend.embed(self.audio, self.clip)
        worker = self.backend._process
        with ThreadPoolExecutor(max_workers=1) as caller:
            second = caller.submit(self.backend.embed, self.audio, self.clip).result(timeout=5)
        self.assertEqual(first.vector, second.vector)
        self.assertEqual(second.metrics['fixture_request_count'], 2)
        self.assertTrue(second.metrics['zero_neural_fixture'])
        self.assertIs(self.backend._process, worker)
        self.assertIn(worker, pipeline._workers)
        self.assertFalse(pipeline.workers_idle())
        self.backend.close()
        self.assert_reaped(worker)
        with self.assertRaises(ValueError):self.backend.embed(self.audio, self.clip)

    def test_idle_residence_allows_update_reservation_which_blocks_new_dispatch(self):
        self.backend.embed(self.audio, self.clip)
        frames = [];self.updates.connect(frames.append)
        self.updates.request_operation('check')
        self.updates.handle({'id':1, 'op':'status', 'state':'available'})
        self.updates.request_operation('download')
        self.updates.handle({'id':1, 'op':'status', 'state':'ready'})
        self.updates.request_operation('install')
        self.updates.handle({'id':1, 'op':'status', 'state':'preparing_install', 'preparation':1})
        self.updates.request_operation('editor_ready', attempt=1, preparation=1)
        self.assertTrue(self.updates.handle({'id':1, 'op':'reserve', 'preparation':1}))
        with self.assertRaises(UpdateReserved):self.backend.embed(self.audio, self.clip)
        self.updates.handle({'id':1, 'op':'release', 'preparation':1})
        result = self.backend.embed(self.audio, self.clip)
        self.assertEqual(result.metrics['fixture_request_count'], 2)
        worker = self.backend._process
        pipeline.shutdown_workers()
        self.backend.close()
        self.assert_reaped(worker)

    def test_transport_failure_and_hang_reap_child_and_never_retry(self):
        for mode in ('wrong_id', 'oversized', 'exit', 'hang'):
            with self.subTest(mode=mode), patch.dict(os.environ, {'SPEAKERDESK_FAKE_VOICE_MODE':mode}):
                backend = process.OwnedVoiceBackend(self.root, RELEASE_CALIBRATION, self.model,
                                                    self.updates, self.root, timeout=2)
                with self.assertRaises(ValueError):backend.embed(self.audio, self.clip)
                worker = backend._process
                self.assert_reaped(worker)
                with self.assertRaises(ValueError):backend.embed(self.audio, self.clip)
                self.assertIs(backend._process, worker)
                backend.close()

    def control_app(self):
        from app import create_app
        import voice_gpu_runtime as gpu
        with (patch.dict(os.environ, {'SPEAKERDESK_VOICE_CONFIG':str(RELEASE_CALIBRATION),
                                     'SPEAKERDESK_MODELS':str(self.root/'models')}),
              patch.object(gpu, 'require_runtime'), patch.object(gpu, 'verify_artifact')):
            app = create_app(self.root)
        backend, _ = app.extensions['speakerdesk']['voice_runtime']
        self.assertTrue(app.test_client().get('/api/people').json['voice_available'])
        backend.embed(self.audio, self.clip)
        spec = importlib.util.spec_from_file_location('voice_test_sidecar',
            Path(__file__).resolve().parents[1]/'packaging/sidecar.py')
        sidecar = importlib.util.module_from_spec(spec);spec.loader.exec_module(sidecar)
        self.addCleanup(app.extensions['speakerdesk']['executor'].shutdown, wait=True, cancel_futures=True)
        self.addCleanup(app.extensions['speakerdesk']['close_voice'])
        return app, backend, sidecar

    def test_ordinary_control_pipe_reaps_registered_voice_child(self):
        app, backend, sidecar = self.control_app()
        worker = backend._process
        from unittest.mock import Mock
        server = Mock()
        sidecar.listen_for_control(app, server, io.StringIO('shutdown\n'))
        server.shutdown.assert_called_once()
        self.assert_reaped(worker)

    def test_ordinary_shutdown_cancels_inflight_work_before_waiting_app_lock(self):
        self.shutdown_inflight('hang_second')

    def test_ordinary_shutdown_cancels_pipe_holding_bootloader_descendant(self):
        self.shutdown_inflight('bootloader_hang_second')

    def shutdown_inflight(self, mode):
        with patch.dict(os.environ, {'SPEAKERDESK_FAKE_VOICE_MODE':mode}):
            app, backend, sidecar = self.control_app()
        worker = backend._process
        lock = app.extensions['speakerdesk']['updates'].lock
        def enrollment():
            with lock:return backend.embed(self.audio, self.clip)
        from unittest.mock import Mock
        with ThreadPoolExecutor(max_workers=2) as callers:
            active = callers.submit(enrollment)
            deadline = time.monotonic()+2
            while backend._next_id < 2 and time.monotonic() < deadline:time.sleep(.005)
            self.assertEqual(backend._next_id, 2)
            stopped = callers.submit(sidecar.listen_for_control, app, Mock(), io.StringIO('shutdown\n'))
            try:
                stopped.result(timeout=2)
            finally:
                registry.stop_owned_worker(worker)  # Bound old-order descendant cleanup too.
            with self.assertRaises(ValueError):active.result(timeout=2)
        self.assert_reaped(worker)

    def test_update_acknowledges_only_after_voice_child_is_reaped(self):
        app, backend, sidecar = self.control_app()
        worker = backend._process
        updates = app.extensions['speakerdesk']['updates']
        frames = []
        from unittest.mock import Mock
        def acknowledge(frame):
            message = json.loads(frame[len(PREFIX):]);frames.append(frame)
            if message['op'] == 'shutdown_ready' and message['ok']:
                self.assert_reaped(worker)
        updates.connect(acknowledge)
        updates.request_operation('check')
        updates.handle({'id':1, 'op':'status', 'state':'available'})
        updates.request_operation('download')
        updates.handle({'id':1, 'op':'status', 'state':'ready'})
        updates.request_operation('install')
        updates.handle({'id':1, 'op':'status', 'state':'preparing_install', 'preparation':1})
        updates.request_operation('editor_ready', attempt=1, preparation=1)
        control = ''.join(PREFIX+json.dumps({'id':1, 'op':op, 'preparation':1})+'\n'
                          for op in ('reserve','shutdown'))
        server = Mock()
        sidecar.listen_for_control(app, server, io.StringIO(control))
        server.shutdown.assert_called_once()
        result = json.loads(frames[-1][len(PREFIX):])
        self.assertEqual((result['op'], result['ok']), ('shutdown_ready',True))


if __name__ == '__main__':unittest.main()
