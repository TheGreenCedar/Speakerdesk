"""Synthetic local UI fixture. No capture, downloads, or trained model execution."""
import json
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
import wave

from flask import abort, jsonify, request
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from language_detection import LANGUAGE_CHOICES
from voice_profiles import Calibration, ClipEmbedding, VoiceModel


class SyntheticVoice:
    model = VoiceModel('browser-fixture', 'v1', 'a'*64, 2)

    def embed(self, audio, clip):
        if self.failure:
            raise ValueError('This synthetic passage could not be used. Choose another passage or retry.')
        self.calls.append(clip.payload())
        return ClipEmbedding((1., 0.), True)


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='speakerdesk-people-') as temporary:
        data = Path(temporary)
        backend = SyntheticVoice()
        backend.calls = []
        backend.failure = False
        policy = Calibration(backend.model, 'browser-fixture', .8, .1, 2, 2, 0., 0.)
        app = create_app(data, voice_backend=backend, voice_calibration=policy)
        for key, name, intervals in [('a', 'Short spoken turns', [(0., .8), (1., 1.9)]),
                                     ('b', 'Clean spoken passages', [(0., 3.), (4., 7.)])]:
            jid = key*32
            job = {'id': jid, 'name': name, 'kind': 'meeting', 'status': 'ready',
                   'message': 'Synthetic UI fixture — no real meeting audio.', 'language': 'auto',
                   'created': 1., 'revision': 0, 'duration': 12.,
                   'document': {'schema_version': 1, 'speakers': {'speaker_0': 'Speaker 1'},
                                'provenance': {'kind': 'local_inference'},
                                'segments': [{'id': f'clip-{i}', 'speaker': 'speaker_0', 'start': start,
                                              'end': end, 'text': 'Synthetic passage for the voice setup preview.',
                                              'review': True, 'voice_eligible': True,
                                              'speaker_candidates': ['speaker_0'], 'language': 'en'}
                                             for i, (start, end) in enumerate(intervals)]}}
            with sqlite3.connect(data/'jobs.sqlite') as db:
                db.execute('INSERT INTO jobs VALUES (?,?)', (jid, json.dumps(job)))
            (data/jid).mkdir()
            with wave.open(str(data/jid/'audio.wav'), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\0\0'*16000*12)
        app.view_functions['config'] = lambda: jsonify(
            languages=LANGUAGE_CHOICES, default_language='auto',
            readiness={'configured': True, 'automatic_language': True}, decoder='Synthetic fixture')
        app.view_functions['status'] = lambda: jsonify(
            models=[], status='ready', ready=True, core_ready=True, supported=True,
            total_bytes=1, downloaded_bytes=1, error=None,
            voice={'status': 'ready', 'available': app.extensions['speakerdesk']['voice_runtime'][0] is not None, 'enabled': True,
                   'message': 'Synthetic fixture: saved voices stay on this Mac.'})
        clip_endpoint = app.view_functions['voice_clips']
        fixture_state = {'clip_error': False, 'model_missing': False}
        # Synthetic readiness facts; never inspect or modify installed models.
        managed_voice = app.extensions['speakerdesk']['voice_setup']
        managed_voice.supported = lambda: True
        managed_voice.released = lambda: fixture_state['model_missing']
        managed_voice.installed = lambda: not fixture_state['model_missing']

        def clip_state(*args, **kwargs):
            if fixture_state['clip_error']:
                abort(409, description='Synthetic passage lookup failed.')
            return clip_endpoint(*args, **kwargs)
        app.view_functions['voice_clips'] = clip_state

        @app.post('/fixture/voice-state')
        def voice_state():
            body = request.get_json()
            app.extensions['speakerdesk']['voice_runtime'] = (None, None) if body.get('unavailable') else (backend, policy)
            app.extensions['speakerdesk']['voice_message'] = 'Voice saving needs a verified GPU/ANE runtime in this build. Saved names and voices are kept.'
            backend.failure = bool(body.get('embedding_error'))
            fixture_state['clip_error'] = bool(body.get('clip_error'))
            fixture_state['model_missing'] = bool(body.get('model_missing'))
            if body.get('revision'):
                manager = app.extensions['speakerdesk']['meetings']
                job = manager.get('b'*32); job['revision'] += 1; manager.put(job)
            return jsonify(updated=True)

        @app.get('/fixture/report')
        def fixture_report():
            return jsonify(embedding_calls=backend.calls,
                           profiles=app.extensions['speakerdesk']['people'].profiles())
        server = make_server('127.0.0.1', 0, app)
        (output/'port.txt').write_text(str(server.server_port))
        print(f'Synthetic People UI fixture: http://127.0.0.1:{server.server_port}', flush=True)
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            app.extensions['speakerdesk']['meetings'].close()
            app.extensions['speakerdesk']['recognition'].close(wait=True)
            app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
