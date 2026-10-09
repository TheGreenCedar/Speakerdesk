"""Serial resident execution of declared capture sources on one sample clock.

The existing Engine owns admission, text, calibrated timing and CAS per source.
Only publication sequencing and completion barriers are shared. No lexical
deduplication or source-to-person inference occurs here.
"""
import copy
import wave
from capture_sources import (RATE, SOURCE_IDS, binding, source_path, speaker_id,
                             validate_catalog, validate_binding)


class SourceTracks:
    def __init__(self, models, source_id):
        self.models, self.source_id = models, source_id

    def __getattr__(self, name):
        return getattr(self.models, name)

    def feed(self, audio, final=False):
        turns, horizon = self.models.feed(audio, final=final)
        return [dict(turn, speaker=speaker_id(self.source_id, turn['speaker']))
                for turn in turns], horizon

    # Offline track labels remain local until the existing temporal mapper
    # reconciles them against this source's measured streaming references.


class SourceRuntime:
    def __init__(self, config, models, emit, inbox):
        from live_refinement import Engine
        self.config, self.emit, self.inbox = config, emit, inbox
        self.catalog = validate_catalog(config['capture_source_catalog'], config['job_id'])
        if not config.get('canonical_utterances'):
            raise ValueError('Source execution requires canonical utterances.')
        self.fast_sequence = config.get('fast_sequence', 0)
        self.engines = {}; self.pending = {}; self.canonical = True
        self.received = 0; self.capture_finished = False
        for source_id in SOURCE_IDS:
            source_config = {**config, 'audio_path': str(source_path(config['audio_path'], source_id)),
                             'capture_source': binding(self.catalog, source_id)}
            peer = SourceTracks(models.for_source(source_config), source_id)
            self.engines[source_id] = Engine(source_config, peer,
                lambda event, source_id=source_id: self.publication(source_id, event), inbox)

    @property
    def admission_executions(self):
        return {source_id: engine.models.admission_execution
                for source_id, engine in self.engines.items()}

    def publication(self, source_id, event):
        event = copy.deepcopy(event)
        if event['type'] in ('canonical_revision', 'refinement_result'):
            candidate = event.get('candidate', event.get('canonical_candidate'))
            if candidate is not None:
                actual = validate_binding(self.catalog, candidate)
                if actual['source_id'] != source_id:
                    raise ValueError('Publication crossed its capture source.')
                self.fast_sequence += 1
                event['fast_sequence'] = self.fast_sequence
            if event['type'] == 'refinement_result':
                event['capture_source'] = binding(self.catalog, source_id)
            self.emit(event)
        else:
            self.pending.setdefault(event['type'], {})[source_id] = event

    def restore_saved(self):
        lengths = []
        for engine in self.engines.values():
            with wave.open(str(engine.path), 'rb') as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE):
                    raise ValueError('Saved capture input format differs.')
                lengths.append(audio.getnframes())
                engine.received = audio.getnframes(); engine.capture_finished = True
        with wave.open(self.config['audio_path'], 'rb') as audio:
            lengths.append(audio.getnframes())
        if len(set(lengths)) != 1:
            raise ValueError('Saved capture sources differ from the shared clock.')
        self.received = lengths[0]; self.capture_finished = True
        self.emit({'type': 'capture_finished', 'duration': self.received/RATE})

    def handle(self, message):
        kind = message['type']; self.pending = {}
        if kind == 'refine':
            actual = validate_binding(self.catalog, message.get('canonical', {}))
            if message.get('capture_source') != actual:
                raise ValueError('Refinement dispatch differs from its source.')
            return self.engines[actual['source_id']].handle(message)
        if kind == 'audio':
            if message['start_sample'] != self.received:
                raise ValueError('Capture sources require contiguous ordered audio.')
            # Fail before either source advances if any declared PCM is absent,
            # corrupt, truncated, or outside the shared capture endpoint.
            for engine in self.engines.values():
                if engine.path.is_symlink():raise ValueError('Capture input changed to a symbolic link.')
                with wave.open(str(engine.path), 'rb') as audio:
                    if ((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE)
                            or audio.getnframes() < message['end_sample']):
                        raise ValueError('Capture input has not reached the shared endpoint.')
        for engine in self.engines.values():
            engine.handle(message)
        endpoints = {engine.received for engine in self.engines.values()}
        if len(endpoints) != 1:
            raise ValueError('Capture source clocks diverged.')
        self.received = endpoints.pop()
        if 'progress' in self.pending:
            events = self.pending['progress'].values()
            self.emit({'type': 'progress', 'received_seconds': self.received/RATE,
                       'processed_seconds': min(e['processed_seconds'] for e in events),
                       **self.engines[SOURCE_IDS[0]].models.metrics()})
        if kind == 'flush' and 'request_id' in message:
            acknowledgements = self.pending['flush_ack']
            self.emit({'type': 'flush_ack', 'request_id': message['request_id'],
                       'received_sample': self.received, 'fast_sequence': self.fast_sequence,
                       'available_sample': min(e['available_sample'] for e in acknowledgements.values()),
                       'speech_observed_sample': min(e['speech_observed_sample'] for e in acknowledgements.values()),
                       'source_admission_receipts': {s: e['admission_receipt'] for s, e in acknowledgements.items()}})
        elif kind == 'stop':
            finished = self.pending['capture_finished']; self.capture_finished = True
            self.emit({'type': 'capture_finished', 'duration': self.received/RATE,
                       'canonical_observed_sample': min(e['canonical_observed_sample'] for e in finished.values()),
                       'canonical_uncertain_samples': max(e['canonical_uncertain_samples'] for e in finished.values()),
                       'source_admission_receipts': {s: e['admission_receipt'] for s, e in finished.items()}})
        elif kind == 'language':
            registered = list(self.pending['language_registered'].values())
            if registered[0] != registered[1]:raise ValueError('Capture source language clocks diverged.')
            self.emit(registered[0])
        elif kind == 'shutdown':
            self.emit({'type': 'finished', 'duration': self.received/RATE}); return False
        return True
