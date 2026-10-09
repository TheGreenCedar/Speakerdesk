"""Model-free publication → SQLite → Flask fixture for the shipped browser UI.

Only speech/model observations are synthetic. Utterance identities, partial
revision policy, host reconciliation, edits and conditional GETs are production.
No capture process, model, original recording or installed app is used.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace

from flask import jsonify, request
from werkzeug.serving import make_server

SOURCE_ROOT = Path(os.environ.get('FRONTEND_SOURCE_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(SOURCE_ROOT / 'speakerdesk'))
from app import create_app
from canonical_runtime import CanonicalRuntime
from utterances import UtteranceBook, RATE

JID = 'd' * 32


class Timeline:
    def __init__(self, app):
        self.app = app
        self.manager = app.extensions['speakerdesk']['meetings']
        self.book = UtteranceBook(JID)
        self.runtime = CanonicalRuntime.__new__(CanonicalRuntime)
        self.runtime.engine = SimpleNamespace(language_at=lambda _: {'language': 'en'})
        self.sequence = 0
        self.events = []
        self.names = {}
        job = dict(id=JID, created=1, name='Publication timeline · synthetic observations',
            status='recording', kind='meeting', language='en', duration=0, revision=0,
            sources=['microphone'], message='Model-free publication fixture',
            canonical_utterances=True, refinement_status='refining',
            document=dict(speakers={}, segments=[], provenance={}, warnings=[]))
        self.manager.refinement.initialize(job)
        self.manager.put(job)
        # Delayed first phrase then ten processed phrases: a real publication
        # backlog, not a manufactured sorted document.
        for i in range(11):
            row = self.speech(2)
            self.decode(row, f'Earlier phrase {i}.', 'live')
            self.silence(1)
            if i:
                self.decode(row, f'Processed phrase {i}.', 'refined')
            else:
                self.names['delayed'] = row['id']
        self.names['tail'] = self.speech(6)['id']
        self.decode(self.book.rows[self.names['tail']], 'We should review the latest sentence', 'live')

    def speech(self, seconds):
        a, b = self.book.cursor, self.book.cursor + round(seconds * RATE)
        self.book.observe(dict(start_sample=a, end_sample=b, complete=True,
            decision='speech', speech_regions=[dict(start_sample=a, end_sample=b)]))
        self.manager.duration = b / RATE
        return self.book.rows[self.book.active]

    def silence(self, seconds):
        a, b = self.book.cursor, self.book.cursor + round(seconds * RATE)
        self.book.observe(dict(start_sample=a, end_sample=b, complete=True,
            decision='no_speech', speech_regions=[]))
        self.manager.duration = b / RATE
        self.publish_all()

    def publish(self, row, candidate=None):
        row['speaker_candidates'] = ['speaker_0']
        candidate = candidate or self.runtime.project(row)
        self.sequence += 1
        self.manager.refinement.canonical(JID, dict(candidate=candidate, fast_sequence=self.sequence))
        self.events.append(dict(sequence=self.sequence, candidate=copy.deepcopy(candidate)))

    def publish_all(self):
        for row in self.book.rows.values():
            self.publish(row)

    def decode(self, row, text, stage, complete=True):
        self.book.apply_model(row['id'], row['machine_revision'], text,
            start_sample=row['start_sample'], end_sample=row['end_sample'],
            stage=stage, complete=complete)
        self.publish(row)

    def step(self, name):
        tail = self.book.rows[self.names['tail']]
        if name == 'grow':
            self.speech(24)
            self.decode(tail, 'We should review the latest sentence ' +
                'and keep every evolving clause visible café 👩🏽‍💻 ' * 90 + 'LATEST TAIL', 'live')
        elif name == 'partial':
            self.speech(3)
            self.decode(tail, '', 'live', complete=False)
        elif name == 'empty-publication':
            # Exercise host protection independently of the book retaining an
            # incomplete candidate: a valid newer empty growing-audio output.
            self.speech(3)
            candidate = self.runtime.project(tail)
            candidate.update(text='', canonical_machine_revision=tail['machine_revision'] + 1)
            self.publish(tail, candidate)
            # Subsequent nonempty publication must own another text revision;
            # the host correctly rejects two texts with the same source CAS.
            tail['machine_revision'] += 2
        elif name == 'silence':
            self.silence(2)
        elif name == 'interjection':
            short = self.speech(.7)
            self.names['interjection'] = short['id']
            self.decode(short, 'Yes.', 'live')
            self.silence(1)
            self.decode(short, 'Yes.', 'refined')
        elif name == 'delayed':
            self.decode(self.book.rows[self.names['delayed']], 'Earlier phrase finally processed.', 'refined')
        elif name == 'refine-tail':
            self.decode(tail, tail['text'] + ' FINAL REVISION', 'refined')
        elif name == 'late-live':
            self.decode(tail, tail['text'] + ' LATE FAST REVISION', 'live')
        elif name == 'stale-publication':
            old = self.events[0]
            self.manager.refinement.canonical(JID, dict(candidate=old['candidate'], fast_sequence=old['sequence']))
        elif name == 'stop':
            self.book.finish()
            self.publish_all()
            self.manager.refinement.capture_done(JID, observed_sample=self.book.cursor, uncertain_samples=0)
            job = self.manager.get(JID)
            job.update(status='ready', revision=job['revision'] + 1)
            self.manager.put(job)
        elif name == 'complete':
            for row in self.book.rows.values():
                self.decode(row, row['text'], 'refined')
            job = self.manager.get(JID)
            job.update(refinement_status='complete', revision=job['revision'] + 1)
            self.manager.put(job)
        elif name == 'remove':
            # Canonical row removal uses the same public saved-document PUT
            # available to the editor, so the next publication stays removed.
            client = self.app.test_client()
            job = self.manager.get(JID)
            document = copy.deepcopy(job['document'])
            document['segments'] = [r for r in document['segments'] if r['id'] != tail['id']]
            import re
            token = re.search(r'name="speakerdesk-token" content="([^"]+)"', client.get('/').text)[1]
            response = client.put(f'/api/jobs/{JID}/transcript',
                headers={'X-Speakerdesk-Token': token}, json=dict(revision=job['revision'], document=document))
            assert response.status_code == 200, response.json
            self.publish(tail)
        else:
            raise ValueError('Unknown fixture step')
        return self.state(name)

    def state(self, name='initial'):
        job = self.manager.get(JID)
        return dict(step=name, ids=self.names, revision=job['revision'],
            expected_order=[row['id'] for row in sorted(job['document']['segments'], key=lambda r: (r['start'], r['end'], r['id']))],
            words={row['id']: row['text'] for row in job['document']['segments']},
            observations='Synthetic speech/model results; production book/projection/reconciliation/SQLite/Flask')


def main():
    with tempfile.TemporaryDirectory(prefix='speakerdesk-timeline-') as home:
        app = create_app(Path(home))
        timeline = Timeline(app)
        held = dict(next=False, response=None, release=threading.Event())

        @app.after_request
        def delay_selected_response(response):
            if held['next'] and request.path == f'/api/jobs/{JID}':
                held['next'] = False
                held['response'] = response.get_json()
                if not held['release'].wait(20):
                    raise RuntimeError('Fixture response was not released')
            return response

        @app.post('/fixture/hold')
        def hold():
            held.update(next=True, response=None)
            held['release'].clear()
            return jsonify(ok=True)

        @app.get('/fixture/held')
        def held_state():
            return jsonify(held=held['response'] is not None)

        @app.post('/fixture/release')
        def release():
            held['release'].set()
            return jsonify(ok=True)

        @app.get('/fixture/state')
        def state():
            return jsonify(timeline.state())

        @app.post('/fixture/step/<name>')
        def step(name):
            return jsonify(timeline.step(name))

        server = make_server('127.0.0.1', 0, app, threaded=True)
        print(json.dumps(dict(base=f'http://127.0.0.1:{server.server_port}', jid=JID)), flush=True)
        try:
            server.serve_forever()
        finally:
            timeline.manager.close()
            app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
