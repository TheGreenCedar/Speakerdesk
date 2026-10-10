"""Device-free app/worker/source boundary controls, not acoustic qualification.

These tests refuse trained-module imports. Synthetic probabilities and text
exercise the production routing, persistence and canonical/CAS boundaries.
"""
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class NoNeural:
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx', 'mlx_audio', 'mlx_speech', 'torch', 'transformers'):
            self.attempts.append(fullname)
            raise RuntimeError('Trained neural imports forbidden: ' + fullname)


GUARD = NoNeural()
sys.meta_path.insert(0, GUARD)
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'speakerdesk'), str(Path(__file__).parent)]

import numpy as np
from app import create_app
from capture_sources import RATE, SOURCE_IDS, catalog, binding
from live_refinement import Inbox, Models
from live_meeting import SourceMixer, processed_microphone_mixer, validate_processing_boundary
from source_runtime import SourceRuntime
from speech_admission import FRAME, SpeechSession
from speech_conditioning import PRODUCTION_UNITY_POLICY, resolve
from capture_processing import (ProducerRegistry, EXPORT_CONTRACT, CAPABILITY_VERSION, BASENAME, PROCESSED, LEGACY,
                                file_digest, install, configure)


def write_audio(path, values):
    samples = np.clip(np.rint(np.asarray(values) * 32768), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, RATE, 0, 'NONE', 'none'))
        stream.writeframes(samples.tobytes())


def producer_packet(job_id, helper):
    """Hypothetical qualified CPU producer; not retained Apple tap evidence."""
    return {'type': 'capture_processing', 'schema_version': 1, 'job_id': job_id,
            'helper_sha256': file_digest(helper), 'processor': 'apple_voice_processing_io',
            'export_contract': EXPORT_CONTRACT, 'output': 'processed_microphone',
            'stage': 'retained_source_pcm', 'postprocessing': 'identity', 'stream_epoch': 0,
            'format': {'sample_rate': RATE, 'channels': 1, 'layout': 'mono'},
            'capability_version': CAPABILITY_VERSION,
            'route': {'node': 'inputNode', 'bus': 0, 'tap_scope': 'output'},
            'state_phase': 'after_start', 'channel_mapping': 'mono_direct',
            'callback_format': {'sample_rate': 48000, 'channels': 1, 'layout': 'mono'},
            'io_formats': {key: {'sample_rate': 48000, 'channels': 1, 'layout': 'mono'}
                           for key in ('input_output', 'output_input')},
            'readbacks': {'input_enabled': True, 'output_enabled': True,
                          'engine_running': True, 'bypassed': False, 'manual_rendering': False,
                          'input_muted': False, 'agc_enabled': False}}


def qualified_registry(helper):
    # Generated CPU helper capability, never retained Apple acoustic proof.
    entry = {'helper_sha256': file_digest(helper), 'export_contract': EXPORT_CONTRACT,
             'capability_version': CAPABILITY_VERSION}
    Path(str(helper)+'.capability.json').write_text(json.dumps(entry))
    return ProducerRegistry.for_helper(helper)


class ShapeVAD:
    """Explicit CPU probability peer: low raw stays negative, gain exposes it."""
    normalized_view = True

    def initial_state(self):
        return 0

    def feed(self, samples, state):
        return (.9 if float(np.max(np.abs(samples))) > .02 else .01), state + 1

    def session(self, start_sample=0, *, archive=None, conditioning=None):
        return SpeechSession(self, start_sample, archive=archive, conditioning=conditioning)

    def inspect_frames(self, samples, start_sample=0, *, archive=None, conditioning=None):
        session = self.session(start_sample, archive=archive, conditioning=conditioning)
        session.feed(samples, start_sample, final=True)
        return session.evidence


class DiarPeer:
    def init_streaming_state(self):
        return SimpleNamespace(frames_processed=0, received=0)

    def feed(self, samples, state, **kwargs):
        start = state.received
        state.received += len(samples)
        state.frames_processed = state.received // 160
        turns = ([SimpleNamespace(start=start/RATE, end=state.received/RATE, speaker=0)]
                 if len(samples) else [])
        return SimpleNamespace(segments=turns), state


def model_owner(config):
    """Reuse the real Models.for_source constructor without neural loading."""
    from final_asr_reuse import FinalAsrReuse
    owner = object.__new__(Models)
    owner.config = config
    owner.diar = DiarPeer()
    owner.speech = ShapeVAD()
    owner.asr = object()
    owner.detector = None
    owner.coarse_aligner = None
    owner.final_asr_reuse = FinalAsrReuse({}, mode='observe')
    owner.language_epoch = 0
    return owner


def cpu_text(model, samples, language, names, overlap=False):
    model.cohere_calls += 1
    return [{'start': 0., 'end': len(samples)/RATE, 'text': 'CPU quiet phrase.',
             'cohere_raw_text': 'CPU quiet phrase.', 'language': language}]


class CaptureProcessingAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = create_app(self.root / 'recordings')
        self.client = self.app.test_client()
        self.manager = self.app.extensions['speakerdesk']['meetings']
        token = re.search(r'name="speakerdesk-token" content="([^"]+)"', self.client.get('/').get_data(as_text=True)).group(1)
        self.headers = {'X-Speakerdesk-Token': token}
        helper = self.root / 'no-device-helper'
        helper.write_text('Not executable; never launched.')
        self.manager.helper_path = lambda: helper
        self.addCleanup(self.shutdown)
        self.patchers = [patch('live_meeting.preflight', return_value=[]),
                         patch('live_meeting.EchoMixer'),
                         patch.object(self.manager.updates, 'start_thread', return_value=None)]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def shutdown(self):
        self.manager.jid = None
        self.manager.close()
        self.app.extensions['speakerdesk']['close_voice']()
        self.app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    def start(self, **extra):
        response = self.client.post('/api/meetings', headers=self.headers,
                                   json={'name': 'CPU capture contract', 'language': 'en',
                                         'sources': ['microphone', 'system'], **extra})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json

    def collect_config(self, job, *, saved=False):
        observed = []

        def before_worker(command, **kwargs):
            observed.append(json.loads(command[-1]))
            raise OSError('CPU boundary: no worker or device launched.')

        self.manager.jid = job['id']
        self.manager.stopped.clear()
        with patch('live_meeting.subprocess.Popen', side_effect=before_worker):
            if saved:
                self.manager._run_saved_refinement(job['id'])
            else:
                self.manager._run(job['id'], job['language'], job['sources'])
        self.assertEqual(len(observed), 1)
        return observed[0]

    def test_http_client_cannot_select_processor_or_supply_contract(self):
        job = self.start(experimental_apple_vad_gain=True,
                         capture_processing_contract={'processor': 'apple_voice_processing_io', 'mode': 'active'},
                         capture_source_catalog={'client': 'untrusted'},
                         capture_processor='apple_voice_processing_io',
                         capture_processing_registry={'helper_sha256': 'f'*64, 'capability_version': CAPABILITY_VERSION},
                         source_processing_metadata={'classification': PROCESSED},
                         capture_processing_reference={'basename': BASENAME, 'sha256': 'f'*64})
        persisted = self.manager.get(job['id'])
        self.assertNotIn('experimental_apple_vad_gain', persisted)
        self.assertNotIn('capture_processing_contract', persisted)
        self.assertNotIn('capture_processor', persisted)
        for field in ('capture_processing_registry', 'source_processing_metadata', 'capture_processing_reference'):
            self.assertNotIn(field, persisted)
        self.assertNotEqual(persisted['capture_source_catalog'], {'client': 'untrusted'})
        disk_catalog = json.loads((self.manager.folder(job['id']) / 'capture-sources.json').read_text())
        self.assertEqual(disk_catalog, persisted['capture_source_catalog'])
        self.assertEqual(GUARD.attempts, [])

    def test_default_live_and_saved_configs_keep_legacy_gain(self):
        job = self.start()
        live = self.collect_config(job)
        persisted = self.manager.get(job['id'])
        saved = self.collect_config(persisted, saved=True)
        for config in (live, saved):
            self.assertEqual(config['job_id'], job['id'])
            self.assertEqual(config['capture_source_catalog'], job['capture_source_catalog'])
            self.assertIsNone(resolve(config))
            self.assertFalse(config.get('experimental_apple_vad_gain', False))
        self.assertTrue(saved['refinement_only'])
        self.assertEqual(live['audio_path'], saved['audio_path'])
        self.assertEqual(GUARD.attempts, [])

    def test_owned_producer_event_persists_identical_live_and_saved_worker_identity(self):
        self.manager.source_startup_hold = False
        self.manager.source_innovation = False
        self.manager.capture_processing_registry = qualified_registry(self.manager.helper_path())
        job = self.start()
        packet = producer_packet(job['id'], self.manager.helper_path())
        control = self.manager.capture_processing(job['id'], packet)
        persisted = self.manager.get(job['id'])
        self.assertEqual(control['type'], 'capture_processing')
        self.assertEqual(control['source_processing_metadata']['classification'], PROCESSED)
        self.assertNotIn('source_processing_metadata', persisted)
        live = self.collect_config(persisted)
        saved = self.collect_config(self.manager.get(job['id']), saved=True)
        for config in (live, saved):
            self.assertEqual(config['source_processing_metadata'], control['source_processing_metadata'])
            self.assertEqual(config['capture_processing_reference'], persisted['capture_processing'])
            mic = {**config, 'capture_source': binding(job['capture_source_catalog'], 'microphone_clean')}
            system = {**config, 'capture_source': binding(job['capture_source_catalog'], 'system')}
            self.assertEqual(resolve(mic).input_policy, PRODUCTION_UNITY_POLICY)
            self.assertIsNone(resolve(system))
        self.assertEqual(GUARD.attempts, [])

    def test_registry_selection_uses_identity_catalog_and_shipping_empty_keeps_native_catalog(self):
        first = self.start()
        self.assertEqual(first['capture_source_catalog']['schema_version'], 3)
        self.assertTrue(self.manager.source_innovation)
        self.assertTrue(self.manager.source_startup_hold)
        first['status'] = 'ready'
        self.manager.put(first)
        self.manager.jid = None
        self.manager.capture_processing_registry = qualified_registry(self.manager.helper_path())
        second = self.start()
        self.assertEqual(second['capture_source_catalog']['schema_version'], 1)
        self.assertNotIn('derivation', second['capture_source_catalog'])
        self.assertFalse(self.manager.capture_pcm_seen)
        self.assertEqual(GUARD.attempts, [])

    def test_actual_capture_pipe_refuses_metadata_after_first_subblock_pcm(self):
        self.manager.capture_processing_registry = qualified_registry(self.manager.helper_path())
        job = self.start()
        packet = producer_packet(job['id'], self.manager.helper_path())
        protocol = Path(__file__).parent / 'support/meeting_protocol.py'
        real_popen = subprocess.Popen
        children = []
        capture_script = """import array,base64,json,sys
json.loads(sys.stdin.readline())
print(json.dumps({'type':'audio','source':'microphone','time':0,'pcm':base64.b64encode(array.array('f',[.01]*100).tobytes()).decode()}),flush=True)
print(sys.argv[1],flush=True)
print(json.dumps({'type':'stopped','time':100/16000}),flush=True)
"""

        def peer(command, **kwargs):
            if len(command) > 1:
                actual = [sys.executable, '-S', str(protocol), 'worker',
                          str(self.manager.folder(job['id'])), 'normal', 'en']
            else:
                actual = [sys.executable, '-S', '-c', capture_script, json.dumps(packet)]
            process = real_popen(actual, **kwargs)
            children.append(process)
            return process

        try:
            # This boundary test exercises capture protocol, not the old DSP.
            with patch('live_meeting.subprocess.Popen', side_effect=peer), patch('live_meeting.SourceMixer',
                    side_effect=lambda sources, sink, **kw: SourceMixer(sources, sink, echo_factory=None, **kw)):
                self.manager._run(job['id'], 'en', job['sources'])
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.poll() is not None for child in children))
            persisted = self.manager.get(job['id'])
            self.assertEqual(persisted['status'], 'failed')
            self.assertIn('recording boundary', persisted['message'])
            self.assertNotIn('capture_processing', persisted)
            self.assertFalse((self.manager.folder(job['id'])/BASENAME).exists())
            self.assertTrue(self.manager.capture_pcm_seen)
            self.assertIsNone(self.manager.capture)
            self.assertIsNone(self.manager.worker)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=3)
        self.assertEqual(GUARD.attempts, [])

    def test_auto_load_owned_helper_capability_and_explicit_empty_override(self):
        helper = self.manager.helper_path()
        self.assertIsNone(ProducerRegistry.for_helper(helper).qualification(file_digest(helper)))
        expected = qualified_registry(helper).qualification(file_digest(helper))
        for explicit in (False, True):
            with self.subTest(explicit_empty=explicit), patch('live_meeting.MeetingManager.helper_path', return_value=helper):
                kwargs = {'capture_processing_registry': ProducerRegistry()} if explicit else {}
                app = create_app(self.root / ('explicit' if explicit else 'automatic'), **kwargs)
                manager = app.extensions['speakerdesk']['meetings']
                try:
                    actual = manager.capture_processing_registry.qualification(file_digest(helper))
                    self.assertEqual(actual, None if explicit else expected)
                finally:
                    manager.close(); app.extensions['speakerdesk']['close_voice']()
                    app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.assertEqual(GUARD.attempts, [])


class CaptureProcessingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.jid = 'a'*32
        self.catalog = catalog(self.jid)
        (self.folder / 'capture-sources.json').write_text(json.dumps(self.catalog))
        self.waveform = np.tile(np.linspace(-.001, .001, FRAME, dtype=np.float32), 188)[:6*RATE]
        for name in ('audio', 'microphone_clean', 'system'):
            write_audio(self.folder / (name+'.wav'), self.waveform)
        self.config = {'job_id': self.jid, 'audio_path': str(self.folder / 'audio.wav'),
                       'capture_source_catalog': self.catalog, 'canonical_utterances': True, 'language': 'en'}
        self.events = []
        self.runtime = None
        for target, value in (('transcribe', cpu_text), ('metrics', lambda _self: {})):
            patcher = patch.object(Models, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def construct(self, config=None):
        config = config or self.config
        self.runtime = SourceRuntime(config, model_owner(config), self.events.append, Inbox())
        return self.runtime

    def processor_config(self, *, observed=False):
        helper = self.folder / 'generated-processed-helper'
        helper.write_bytes(b'CPU controlled processed-mono export; not Apple hardware proof.')
        registry = qualified_registry(helper)
        packet = producer_packet(self.jid, helper)
        if observed:
            packet['output'] = 'observed_microphone_tap'
        job = {'id': self.jid, 'kind': 'meeting', 'capture_source_catalog': self.catalog}
        install(job, self.folder, packet, registry, helper)
        return configure(self.config, job, self.folder, registry)

    def feed(self):
        for start in range(0, 6*RATE, RATE):
            self.runtime.handle({'type': 'audio', 'start_sample': start, 'end_sample': start+RATE,
                                 'language': 'en', 'language_epoch': 0})
        self.runtime.handle({'type': 'stop', 'request_id': 'cpu-stop'})

    def revisions(self):
        return [event for event in self.events if event['type'] == 'canonical_revision']

    def test_real_source_factory_legacy_quiet_reaches_original_canonical_rows(self):
        runtime = self.construct()
        self.feed()
        rows = {event['candidate']['capture_source']['source_id']: event['candidate']
                for event in self.revisions() if event['candidate']['text'].strip()}
        self.assertEqual(set(rows), set(SOURCE_IDS))
        self.assertEqual({row['text'] for row in rows.values()}, {'CPU quiet phrase.'})
        self.assertEqual({row['end_sample'] for row in rows.values()}, {6*RATE})
        for source, engine in runtime.engines.items():
            self.assertIsNone(engine.models.speech_conditioning)
            evidence = engine.models.speech_live.evidence
            frames = evidence.archive.query(0, evidence.end_sample) + evidence.frames
            self.assertTrue(frames)
            self.assertGreater(min(frame[4]['gain'] for frame in frames), 200.)
            self.assertEqual(engine.models.speech_live.state, engine.models.speech_live.normalized_state)
            self.assertEqual(engine.models.config['capture_source'], binding(self.catalog, source))
        self.assertEqual(GUARD.attempts, [])

    def test_host_cas_keeps_source_identity_and_refuses_cross_source_candidate(self):
        self.construct()
        self.feed()
        app = create_app(self.folder / 'app')
        manager = app.extensions['speakerdesk']['meetings']
        try:
            manager.duration = 6
            job = {'id': self.jid, 'name': 'CPU source CAS', 'status': 'ready', 'kind': 'meeting',
                   'created': 1, 'language': 'en', 'revision': 0, 'duration': 6,
                   'canonical_utterances': True, 'capture_source_catalog': self.catalog,
                   'document': {'speakers': {}, 'segments': [], 'warnings': [],
                                'provenance': {'kind': 'local_inference'}}}
            manager.refinement.initialize(job)
            manager.put(job)
            for event in self.revisions():
                manager.refinement.canonical(self.jid, event)
            persisted = manager.get(self.jid)
            self.assertEqual(len(persisted['document']['segments']), 2)
            row = copy.deepcopy(persisted['document']['segments'][0])
            other = 'system' if row['capture_source']['source_id'] == 'microphone_clean' else 'microphone_clean'
            row['capture_source'] = binding(self.catalog, other)
            with self.assertRaisesRegex(ValueError, 'identity'):
                manager.refinement.canonical(self.jid, {'candidate': row,
                    'fast_sequence': persisted['last_fast_sequence']+1})
            self.assertEqual(manager.get(self.jid)['document'], persisted['document'])
        finally:
            manager.close()
            app.extensions['speakerdesk']['close_voice']()
            app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        self.assertEqual(GUARD.attempts, [])

    def test_qualified_generated_mic_uses_unity_system_keeps_quiet_gain(self):
        runtime = self.construct(self.processor_config())
        self.feed()
        rows = [event['candidate'] for event in self.revisions() if event['candidate']['text'].strip()]
        self.assertEqual({row['capture_source']['source_id'] for row in rows}, {'system'})
        for source, engine in runtime.engines.items():
            peer = engine.models
            evidence = peer.speech_live.evidence
            frames = evidence.archive.query(0, evidence.end_sample) + evidence.frames
            if source == 'microphone_clean':
                self.assertEqual(peer.speech_conditioning.input_policy, PRODUCTION_UNITY_POLICY)
                self.assertEqual({frame[4]['gain'] for frame in frames}, {1.})
                self.assertEqual(peer.admission_execution['input_policy'], PRODUCTION_UNITY_POLICY)
            else:
                self.assertIsNone(peer.speech_conditioning)
                self.assertGreater(min(frame[4]['gain'] for frame in frames), 200.)
            peer.begin_refinement(self.waveform, 0)
            historical = peer.speech_historical
            expected = PRODUCTION_UNITY_POLICY if source == 'microphone_clean' else evidence.input_policy
            self.assertEqual(historical.input_policy, expected)
            self.assertEqual(peer.speech_live.state, peer.speech_live.normalized_state)
        self.assertEqual(GUARD.attempts, [])

    def test_owned_control_rebinds_before_pcm_with_new_typed_source_execution_ids(self):
        from admission_receipt import execution, validate_execution
        runtime = self.construct()
        before = copy.deepcopy(runtime.admission_executions)
        config = self.processor_config()
        runtime.handle({'type': 'capture_processing',
                        'source_processing_metadata': config['source_processing_metadata'],
                        'capture_processing_reference': config['capture_processing_reference']})
        ready = self.events[-1]
        self.assertEqual(ready['type'], 'processing_ready')
        actual = ready['source_admission_executions']
        self.assertEqual(set(actual), set(SOURCE_IDS))
        self.assertEqual(len({value['execution_id'] for value in actual.values()}), 2)
        for source, value in actual.items():
            self.assertNotEqual(value['execution_id'], before[source]['execution_id'])
            self.assertEqual(validate_execution(value, self.jid), value)
            profile = runtime.engines[source].models.speech_conditioning
            self.assertEqual(value, execution(self.jid, value['execution_id'], conditioning=profile))
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            runtime.handle({'type': 'capture_processing',
                            'source_processing_metadata': config['source_processing_metadata'],
                            'capture_processing_reference': config['capture_processing_reference']})
        self.assertEqual(runtime.received, 0)
        self.assertEqual(GUARD.attempts, [])

    def test_qualified_positive_quiet_is_preserved_and_worker_receipt_mutation_refused(self):
        config = self.processor_config()
        stronger = self.waveform * 40
        for name in ('audio', 'microphone_clean', 'system'):
            write_audio(self.folder/(name+'.wav'), stronger)
        self.construct(config)
        self.feed()
        rows = [event['candidate'] for event in self.revisions() if event['candidate']['text'].strip()]
        self.assertEqual({row['capture_source']['source_id'] for row in rows}, set(SOURCE_IDS))
        self.assertEqual({row['text'] for row in rows}, {'CPU quiet phrase.'})
        for mutate in (lambda source: source.update(source_revision=True),
                       lambda source: source.update(unexpected='client')):
            changed = {**config, 'audio_path': str(self.folder/'system.wav'),
                       'capture_source': binding(self.catalog, 'system')}
            mutate(changed['capture_source'])
            with self.assertRaises(ValueError):
                resolve(changed)
        changed = copy.deepcopy(config)
        changed['source_processing_metadata']['capture_source']['source_id'] = 'system'
        with self.assertRaises(ValueError):
            SourceRuntime(changed, model_owner(changed), lambda event: None, Inbox())
        path = self.folder/BASENAME
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'changed|identity|receipt'):
            SourceRuntime(config, model_owner(config), lambda event: None, Inbox())
        self.assertEqual(GUARD.attempts, [])

    def test_observed_tap_metadata_keeps_legacy_and_rebinding_after_pcm_is_refused(self):
        config = self.processor_config(observed=True)
        self.assertEqual(config['source_processing_metadata']['classification'], LEGACY)
        runtime = self.construct(config)
        runtime.handle({'type': 'audio', 'start_sample': 0, 'end_sample': RATE,
                        'language': 'en', 'language_epoch': 0})
        for engine in runtime.engines.values():
            self.assertIsNone(engine.models.speech_conditioning)
        with self.assertRaises(ValueError):
            runtime.handle({'type': 'capture_processing',
                            'source_processing_metadata': config['source_processing_metadata'],
                            'capture_processing_reference': config['capture_processing_reference']})
        self.assertEqual(runtime.received, RATE)
        self.assertEqual(GUARD.attempts, [])


class CaptureProcessingReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.jid = 'b'*32
        self.helper = self.folder / 'qualified-cpu-helper'
        self.helper.write_bytes(b'CPU control: never launched, no physical qualification.')
        self.registry = qualified_registry(self.helper)
        self.job = {'id': self.jid, 'kind': 'meeting', 'capture_source_catalog': catalog(self.jid)}
        self.config = {'job_id': self.jid, 'capture_source_catalog': self.job['capture_source_catalog']}

    def test_qualification_is_server_owned_and_runtime_readbacks_derive_classification(self):
        packet = producer_packet(self.jid, self.helper)
        value = install(self.job, self.folder, packet, self.registry, self.helper)
        self.assertEqual(value['classification'], PROCESSED)
        self.assertEqual(value['capture_source'], binding(self.job['capture_source_catalog'], 'microphone_clean'))
        self.assertEqual(set(self.job['capture_processing']), {'basename', 'sha256'})
        config = configure({**self.config, 'experimental_apple_vad_gain': True,
                            'capture_processing_contract': {'spoof': True},
                            'source_processing_metadata': {'spoof': True}},
                           self.job, self.folder, self.registry)
        self.assertEqual(config['source_processing_metadata'], value)
        self.assertNotIn('experimental_apple_vad_gain', config)
        self.assertNotIn('capture_processing_contract', config)
        self.assertEqual((self.folder / BASENAME).stat().st_mode & 0o777, 0o600)
        self.assertEqual(GUARD.attempts, [])

    def test_observed_unqualified_bypass_postprocessed_and_nonmono_exports_stay_legacy(self):
        mutations = [('unqualified', lambda packet: None, ProducerRegistry()),
                     ('observed tap', lambda packet: packet.update(output='observed_microphone_tap'), self.registry),
                     ('bypass', lambda packet: packet['readbacks'].update(bypassed=True), self.registry),
                     ('postprocessed', lambda packet: packet.update(postprocessing='webrtc_and_innovation'), self.registry),
                     ('nonmono', lambda packet: packet['format'].update(channels=9, layout='unknown'), self.registry)]
        for name, mutate, registry in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                job = copy.deepcopy(self.job)
                packet = producer_packet(self.jid, self.helper)
                mutate(packet)
                value = install(job, Path(folder), packet, registry, self.helper)
                self.assertEqual(value['classification'], LEGACY)
                self.assertEqual(configure(self.config, job, Path(folder), registry)['source_processing_metadata'], value)
        for startup in (False, True):
            with self.subTest(custom_dsp_catalog=startup), tempfile.TemporaryDirectory() as folder:
                job = {**self.job, 'capture_source_catalog': catalog(self.jid, innovation=True, startup=startup)}
                value = install(job, Path(folder), producer_packet(self.jid, self.helper), self.registry, self.helper)
                self.assertEqual(value['classification'], LEGACY)
        self.assertEqual(GUARD.attempts, [])

    def test_mutated_receipt_cross_job_catalog_and_untrusted_registry_are_refused(self):
        install(self.job, self.folder, producer_packet(self.jid, self.helper), self.registry, self.helper)
        original = (self.folder / BASENAME).read_bytes()
        with self.assertRaisesRegex(ValueError, 'registry'):
            configure(self.config, self.job, self.folder, ProducerRegistry())
        for key, replacement in [('id', 'c'*32),
                                 ('capture_source_catalog', catalog('c'*32))]:
            changed = copy.deepcopy(self.job)
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                configure(self.config, changed, self.folder, self.registry)
        (self.folder / BASENAME).write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            configure(self.config, self.job, self.folder, self.registry)
        self.assertEqual(GUARD.attempts, [])

    def test_path_escape_symlink_and_missing_receipt_are_refused(self):
        install(self.job, self.folder, producer_packet(self.jid, self.helper), self.registry, self.helper)
        changed = copy.deepcopy(self.job)
        changed['capture_processing']['basename'] = '../capture-processing.json'
        with self.assertRaisesRegex(ValueError, 'server owned'):
            configure(self.config, changed, self.folder, self.registry)
        path = self.folder / BASENAME
        retained = self.folder / 'retained.json'
        path.rename(retained)
        path.symlink_to(retained)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            configure(self.config, self.job, self.folder, self.registry)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            configure(self.config, self.job, self.folder, self.registry)
        self.assertEqual(GUARD.attempts, [])

    def test_owned_helper_identity_and_typed_readback_must_match_before_publication(self):
        for name, mutate in [('helper', lambda packet: packet.update(helper_sha256='e'*64)),
                             ('job', lambda packet: packet.update(job_id='c'*32)),
                             ('readback', lambda packet: packet['readbacks'].update(engine_running=1)),
                             ('epoch', lambda packet: packet.update(stream_epoch=1))]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                packet = producer_packet(self.jid, self.helper)
                mutate(packet)
                with self.assertRaises(ValueError):
                    install(copy.deepcopy(self.job), Path(folder), packet, self.registry, self.helper)
                self.assertFalse((Path(folder) / BASENAME).exists())
        with self.assertRaisesRegex(ValueError, 'Imports'):
            install({**self.job, 'kind': 'import'}, self.folder,
                    producer_packet(self.jid, self.helper), self.registry, self.helper)
        self.assertEqual(GUARD.attempts, [])

    def test_capability_manifest_missing_stale_malformed_symlink_and_oversize(self):
        manifest = Path(str(self.helper)+'.capability.json')
        original = manifest.read_bytes()
        manifest.unlink()
        self.assertIsNone(ProducerRegistry.for_helper(self.helper).qualification(file_digest(self.helper)))
        for name, data in [('stale', json.dumps({'helper_sha256': 'd'*64,
                'export_contract': EXPORT_CONTRACT, 'capability_version': CAPABILITY_VERSION}).encode()),
                ('old-proof', json.dumps({'helper_sha256': file_digest(self.helper),
                'export_contract': EXPORT_CONTRACT, 'proof_sha256': 'd'*64}).encode()),
                ('unknown-version', json.dumps({'helper_sha256': file_digest(self.helper),
                'export_contract': EXPORT_CONTRACT, 'capability_version': 'future'}).encode()),
                ('malformed', b'{'), ('oversize', b' '*4097)]:
            with self.subTest(name=name):
                manifest.write_bytes(data)
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    ProducerRegistry.for_helper(self.helper)
        manifest.write_bytes(original)
        retained = self.folder/'capability-original.json'; manifest.rename(retained)
        manifest.symlink_to(retained)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            ProducerRegistry.for_helper(self.helper)
        manifest.unlink(); manifest.write_bytes(original)
        self.helper.write_bytes(b'mutated helper bytes')
        with self.assertRaisesRegex(ValueError, 'differs'):
            ProducerRegistry.for_helper(self.helper)
        self.assertEqual(GUARD.attempts, [])

    def test_poststart_route_io_mapping_and_runtime_state_are_required(self):
        def nine(packet):
            fmt = {'sample_rate': 48000, 'channels': 9, 'layout': 'discrete_float32_noninterleaved'}
            packet.update(callback_format=copy.deepcopy(fmt), channel_mapping='nine_discrete_exact_replicas',
                          io_formats={key: copy.deepcopy(fmt) for key in ('input_output', 'output_input')})
        # This is a server receipt-schema control, not proof of real callback channel equality.
        for label, mutate in [('mono', lambda p: None), ('replica-declaration', nine),
                              ('agc-true', lambda p: p['readbacks'].update(agc_enabled=True))]:
            with self.subTest(eligible=label), tempfile.TemporaryDirectory() as folder:
                packet=producer_packet(self.jid,self.helper); mutate(packet)
                value=install(copy.deepcopy(self.job),Path(folder),packet,self.registry,self.helper)
                self.assertEqual(value['classification'],PROCESSED)
        mutations=[('prestart',lambda p:p.update(state_phase='before_start')),
            ('wrong-node',lambda p:p['route'].update(node='outputNode')),
            ('wrong-bus',lambda p:p['route'].update(bus=1)),
            ('bool-bus',lambda p:p['route'].update(bus=False)),
            ('input-tap',lambda p:p['route'].update(tap_scope='input')),
            ('unknown-capability',lambda p:p.update(capability_version='unknown')),
            ('io-mismatch',lambda p:p['io_formats']['output_input'].update(sample_rate=44100)),
            ('callback-mismatch',lambda p:p['callback_format'].update(sample_rate=44100)),
            ('nine-distinct',lambda p:(nine(p),p.update(channel_mapping='nine_distinct'))),
            ('unknown-mapping',lambda p:p.update(channel_mapping='unknown'))]
        for key in ('input_enabled','output_enabled','engine_running'):
            mutations.append((key,lambda p,k=key:p['readbacks'].update({k:False})))
        for key in ('bypassed','manual_rendering','input_muted'):
            mutations.append((key,lambda p,k=key:p['readbacks'].update({k:True})))
        for label,mutate in mutations:
            with self.subTest(legacy=label),tempfile.TemporaryDirectory() as folder:
                packet=producer_packet(self.jid,self.helper);mutate(packet)
                value=install(copy.deepcopy(self.job),Path(folder),packet,self.registry,self.helper)
                self.assertEqual(value['classification'],LEGACY)
        for label,mutate in [('agc-type',lambda p:p['readbacks'].update(agc_enabled=1)),
                ('missing-output-format',lambda p:p['io_formats'].pop('output_input')),
                ('missing-muted',lambda p:p['readbacks'].pop('input_muted')),
                ('rate-bool',lambda p:p['callback_format'].update(sample_rate=True))]:
            with self.subTest(refused=label),tempfile.TemporaryDirectory() as folder:
                packet=producer_packet(self.jid,self.helper);mutate(packet)
                with self.assertRaises(ValueError):
                    install(copy.deepcopy(self.job),Path(folder),packet,self.registry,self.helper)
                self.assertFalse((Path(folder)/BASENAME).exists())
        self.assertEqual(GUARD.attempts, [])



class ProcessedMicrophoneTransportTests(unittest.TestCase):
    def test_normal_quiet_and_partial_tail_are_exact_without_second_dsp(self):
        for scale in (.25, .00004):
            with self.subTest(scale=scale):
                count = 2*RATE+137
                mic = np.linspace(-scale, scale, count, dtype=np.float32)
                left = np.linspace(.03, -.04, count, dtype=np.float32)
                stereo = np.column_stack((left, left)).astype(np.float32)
                published = []
                raw = []
                mixer = processed_microphone_mixer(['microphone', 'system'],
                    lambda mixed, separate: published.append((mixed.copy(), {k: v.copy() for k, v in separate.items()})),
                    raw_sink=lambda separate: raw.append({k: v.copy() for k, v in separate.items()}))
                self.assertIsNone(mixer.echo)
                self.assertIsNone(mixer.innovation)
                for start in range(0, count, 1365):
                    end = min(start+1365, count)
                    mixer.add('microphone', start/RATE, mic[start:end].tobytes())
                    mixer.add('system', start/RATE, stereo[start:end].tobytes(), channels=2)
                    mixer.flush(end/RATE)
                mixer.flush(count/RATE, final=True)
                selected = np.concatenate([tracks['microphone_clean'] for _, tracks in published])
                raw_mic = np.concatenate([tracks['microphone'] for tracks in raw])
                raw_stereo = np.concatenate([tracks['system_reference'] for tracks in raw])
                mixed = np.concatenate([values for values, _ in published])
                np.testing.assert_array_equal(selected, mic)
                np.testing.assert_array_equal(raw_mic, mic)
                np.testing.assert_array_equal(raw_stereo, stereo)
                np.testing.assert_array_equal(mixed, np.clip(mic+left, -1, 1).astype(np.float32))
                self.assertEqual(mixer.cursor, count)
                self.assertEqual(mixer.late_samples, 0)
        self.assertEqual(GUARD.attempts, [])

    def test_qualified_pause_and_format_refuse_after_exact_tail_drain_legacy_stays_allowed(self):
        mic = np.linspace(-.002, .002, 137, dtype=np.float32)
        published = []
        mixer = processed_microphone_mixer(['microphone', 'system'],
            lambda mixed, separate: published.append(separate['microphone_clean'].copy()), raw_sink=lambda separate: None)
        mixer.add('microphone', 0, mic.tobytes())
        mixer.add('system', 0, np.zeros_like(mic).tobytes())
        mixer.flush(len(mic)/RATE, final=True)
        np.testing.assert_array_equal(np.concatenate(published), mic)
        for kind in ('paused', 'format_changed'):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'epoch changed'):
                validate_processing_boundary(True, kind)
            validate_processing_boundary(False, kind)
        for kind in ('recording', 'stopped', 'clock'):
            validate_processing_boundary(True, kind)
        self.assertEqual(mixer.cursor, len(mic))
        self.assertEqual(GUARD.attempts, [])


if __name__ == '__main__':
    unittest.main()
