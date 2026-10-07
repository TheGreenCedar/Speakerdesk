"""Controlled CPU peers through resident, saved job API and shipped renderer.

Not acoustic accuracy evidence: timing and speaker outputs are fabricated.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = Path(os.environ.get('SEQUENTIAL_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'speakerdesk'), str(ROOT / 'tests')]
from test_canonical_runtime import Peer, RATE
from live_refinement import Engine, Inbox
from reading_turns import CALIBRATION, MODEL
from app import create_app

TEXT = 'Alpha. Still alpha. Beta. Still beta.'


class SequentialPeer(Peer):
    def __init__(self, overlap=False, routed=False):
        super().__init__()
        self.overlap = overlap
        self.routed = routed

    def feed(self, audio, final=False):
        start = self.received
        _, observed = super().feed(audio, final)
        turns = []
        for a, b, speaker in [(0, 6 if self.overlap else 5, 'speaker_0'),
                              (4 if self.overlap else 5, 10, 'speaker_1')]:
            a, b = max(start, a * RATE), min(self.received, b * RATE)
            if a < b:
                turns.append({'start': a / RATE, 'end': b / RATE, 'speaker': speaker})
        return turns, observed

    def transcribe(self, audio, language, names, overlap=False):
        if self.routed:
            return [{'start': a / RATE, 'end': b / RATE, 'text': text, 'cohere_raw_text': text,
                     'language': 'en', 'language_detection': {'mode': 'auto', 'reason': 'detected'},
                     'acoustic_evidence': {'source': 'silero_v6', 'complete': True,
                         'decision': 'speech', 'start_sample': a, 'end_sample': b,
                         'uncertain_regions': [], 'speech_regions': [{'start_sample': a, 'end_sample': b}]}}
                    for a, b, text in [(0, 4 * RATE, 'Alpha. Still alpha.'),
                                       (4 * RATE, len(audio), 'Beta. Still beta.')]]
        return [{'start': 0, 'end': len(audio) / RATE, 'text': TEXT,
                 'cohere_raw_text': TEXT, 'language': 'en',
                 'language_detection': {'mode': 'manual', 'reason': 'override'}}]

    def alignment_supported(self, language):
        return language == 'en'

    def align_canonical(self, request, text, language='en'):
        times = [.75, 1.75, 4.5 if self.overlap else 3.75, 5.5, 5.85, 6.5]
        if text.startswith('Beta.'):
            times = times[3:]
        words = [{'text': unit.group(), 'start_char': unit.start(), 'end_char': unit.end(),
                  'start_sample': round(t * RATE), 'end_sample': round((t + .15) * RATE),
                  'status': 'aligned'} for unit, t in zip(re.finditer(r'\S+', text), times)]
        return {'raw_text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'audio_anchor': request, 'model_sha256': MODEL,
                'timing_kind': 'ctc_emission_cell_envelope',
                'frame_calibration_id': CALIBRATION, 'score_calibration_id': CALIBRATION,
                'words': words}


class SequentialProductPathTests(unittest.TestCase):
    def render_job(self, *, sealed, overlap=False, timing=True, routed=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / 'synthetic.wav'
            with wave.open(str(audio), 'wb') as wav:
                wav.setparams((1, 2, RATE, 0, 'NONE', 'none'))
                wav.writeframes(b'\x01\x00' * 10 * RATE)
            events = []
            jid = 'a' * 32
            language = 'auto' if routed else 'en'
            peer = SequentialPeer(overlap, routed)
            if not timing:
                peer.align_canonical = lambda *args, **kwargs: None
            engine = Engine({'audio_path': str(audio), 'language': language, 'job_id': jid,
                             'canonical_utterances': True}, peer, events.append, Inbox())
            for n in range(10 if sealed else 9):
                engine.handle({'type': 'audio', 'start_sample': n * RATE,
                               'end_sample': (n + 1) * RATE, 'language': language, 'language_epoch': 0})
            if sealed:
                engine.handle({'type': 'stop'})
            app = create_app(root / 'home')
            manager = app.extensions['speakerdesk']['meetings']
            job = {'id': jid, 'created': 1, 'status': 'ready' if sealed else 'recording',
                   'kind': 'meeting', 'name': 'Synthetic A then B', 'language': language,
                   'duration': 10, 'revision': 0, 'canonical_utterances': True,
                   'document': {'speakers': {'speaker_0': 'A', 'speaker_1': 'B'},
                                'segments': [], 'provenance': {}, 'warnings': []}}
            try:
                manager.duration = 10
                manager.refinement.initialize(job)
                manager.put(job)
                for event in events:
                    if event['type'] == 'canonical_revision':
                        manager.refinement.canonical(jid, event)
                response = app.test_client().get('/api/jobs/' + jid)
                self.assertEqual(response.status_code, 200)
                if not timing:
                    exported = app.test_client().get('/api/jobs/' + jid + '/export/txt')
                    self.assertEqual(exported.status_code, 200)
                    self.assertNotIn('Overlapping speakers', exported.get_data(as_text=True),
                                     'candidate union must not produce an overlap warning in TXT export')
                payload = root / 'job.json'
                payload.write_text(json.dumps(response.json))
                result = subprocess.run(['node', str(Path(__file__).with_name('sequential_product_path.test.cjs')),
                                         '--render', str(payload), 'unknown' if not timing else
                                         'overlap' if overlap else 'sequential'],
                                        env={**os.environ, 'FRONTEND_SOURCE_ROOT': str(ROOT)},
                                        text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            finally:
                manager.close()
                app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)

    def test_open_sequential_activity_reaches_readable_turns(self):
        self.render_job(sealed=False)

    def test_exact_five_second_sequential_turns_keep_sentences_together(self):
        self.render_job(sealed=True)

    def test_actual_concurrent_activity_is_localized(self):
        self.render_job(sealed=True, overlap=True)

    def test_missing_timing_keeps_words_without_false_overlap_warning(self):
        self.render_job(sealed=True, timing=False)

    def test_automatic_adjacent_english_routes_reach_readable_turns(self):
        self.render_job(sealed=True, routed=True)
