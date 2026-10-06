"""Synthetic local UI fixture. No capture, downloads, or trained model execution."""
import json
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
import wave

from flask import jsonify
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from language_detection import LANGUAGE_CHOICES
from voice_profiles import Calibration, ClipEmbedding, VoiceModel


class SyntheticVoice:
    model = VoiceModel('browser-fixture', 'v1', 'a'*64, 2)

    def embed(self, audio, clip):
        return ClipEmbedding((1., 0.), True)


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='speakerdesk-people-') as temporary:
        data = Path(temporary)
        backend = SyntheticVoice()
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
            voice={'status': 'ready', 'available': True, 'enabled': True,
                   'message': 'Synthetic fixture: saved voices stay on this Mac.'})
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
            app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
