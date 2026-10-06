"""Speech evidence independent of language, speaker identity and Cohere text.

The neural boundary uses the pinned MLX Silero implementation, not a transcript
generator. Missing, incomplete or failed evidence never becomes no-speech.
"""
import hashlib
import json
import math
from pathlib import Path

RATE = 16000
FRAME = 512
SILERO_SPEC = {
    'name': 'Speech detection', 'repo': 'mlx-community/silero-vad-v6',
    'revision': '2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06',
    'directory': 'silero-speech', 'bytes': 1237860,
    'sha256': '65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff',
    'files': ['config.json', 'model.safetensors', 'README.md'],
    'file_sha256': {
        'model.safetensors': '65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff',
        'config.json': '9fe1befb9692a0d4135adadc33f8075ef6d350bd2391b88d750f2c233f97fa0b',
        'README.md': '34990444751e0f8340ea3189ddffb3ef6300c8ba346c458d1d811690d76350ce',
    },
}


class SpeechFrames:
    """An ordered ledger of actual neural frames; partial audio stays pending."""
    def __init__(self, start_sample=0):
        if type(start_sample) is not int or start_sample < 0:
            raise ValueError('Invalid speech-evidence origin.')
        self.start_sample = self.end_sample = start_sample
        self.frames = []
        self.speaking = False

    def append(self, start, end, probability):
        if (type(start) is not int or type(end) is not int or start != self.end_sample
                or not start < end <= start+FRAME or type(probability) not in (int, float)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('Invalid or noncontiguous Silero evidence.')
        # Standard Silero onset/release hysteresis; no minimum speech duration.
        # A single positive32ms frame is retained rather than discarding a word.
        if probability >= .5:
            self.speaking = True
        elif probability < .35:
            self.speaking = False
        self.frames.append((start, end, float(probability), self.speaking))
        self.end_sample = end

    def admission(self, start, end):
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('Invalid speech-admission range.')
        selected = [frame for frame in self.frames if frame[0] < end and frame[1] > start]
        covered = (start >= self.start_sample and end <= self.end_sample and bool(selected))
        result = {'source': 'silero_v6', 'start_sample': start, 'end_sample': end,
                  'observed_until_sample': self.end_sample, 'complete': covered,
                  'decision': 'pending', 'speech_regions': []}
        if not covered:
            return result
        result['maximum_probability'] = max(frame[2] for frame in selected)
        for a, b, _, speaking in selected:
            if not speaking:
                continue
            a, b = max(a, start), min(b, end)
            if result['speech_regions'] and result['speech_regions'][-1]['end_sample'] == a:
                result['speech_regions'][-1]['end_sample'] = b
            else:
                result['speech_regions'].append({'start_sample': a, 'end_sample': b})
        result['decision'] = 'speech' if result['speech_regions'] else 'no_speech'
        return result


class SileroModel:
    """Strict16k branch of the reviewed dependency; explicit recurrent state."""
    def __init__(self, path):
        path = Path(path)
        for name, expected in SILERO_SPEC['file_sha256'].items():
            with (path/name).open('rb') as content:
                if hashlib.file_digest(content, 'sha256').hexdigest() != expected:
                    raise ValueError('Speech-detection checkpoint verification failed.')
        if (path/'model.safetensors').stat().st_size != SILERO_SPEC['bytes']:
            raise ValueError('Incomplete speech-detection checkpoint.')
        config = json.loads((path/'config.json').read_text())
        expected_branch = {'sample_rate': 16000, 'filter_length': 256, 'hop_length': 128,
                           'pad': 64, 'cutoff': 129, 'context_size': 64, 'chunk_size': 512}
        if config.get('branch_16k') != expected_branch or config.get('version') != 'v6':
            raise ValueError('Incompatible speech-detection architecture.')
        import mlx.core as mx
        from mlx_audio.vad.models.silero_vad.config import BranchConfig
        from mlx_audio.vad.models.silero_vad.silero_vad import SileroVADBranch
        self.mx = mx
        self.branch = SileroVADBranch(BranchConfig.from_dict(expected_branch))
        # The v6 conversion contains16k only. Loading a two-branch Model with
        # strict=False would leave an uninitialized8k branch; do not do that.
        weights = mx.load(str(path/'model.safetensors'))
        if not weights or any(not key.startswith('vad_16k.') for key in weights):
            raise ValueError('Unexpected speech-detection weight inventory.')
        self.branch.load_weights([(key.removeprefix('vad_16k.'), value)
                                  for key, value in weights.items()], strict=True)
        mx.eval(self.branch.parameters())

    def initial_state(self):
        return None, self.mx.zeros((1, 64), dtype=self.mx.float32)

    def feed(self, chunk, state):
        recurrent, context = state
        audio = self.mx.array(chunk, dtype=self.mx.float32)[None, :]
        if audio.shape != (1, FRAME):
            raise ValueError('Silero requires512 new samples.')
        probability, recurrent = self.branch(self.mx.concatenate([context, audio], axis=-1), state=recurrent)
        self.mx.eval(probability, recurrent)
        return float(probability.item()), (recurrent, audio[:, -64:])

    def session(self, start_sample=0):
        return SpeechSession(self, start_sample)

    def inspect_frames(self, audio, start_sample=0):
        session = self.session(start_sample)
        session.feed(audio, start_sample, final=True)
        return session.evidence

    def inspect(self, audio, start_sample=0):
        return self.inspect_frames(audio, start_sample).admission(start_sample, start_sample+len(audio))


class SpeechSession:
    """Carry64 context samples and LSTM state across contiguous capture packets."""
    def __init__(self, model, start_sample=0):
        import numpy as np
        self.model, self.state = model, model.initial_state()
        self.evidence = SpeechFrames(start_sample)
        self.received = start_sample
        self.pending = np.empty(0, dtype=np.float32)
        self.closed = False
        self.failed = False

    def feed(self, audio, start_sample, *, final=False):
        import numpy as np
        pcm = np.asarray(audio, dtype=np.float32)
        if (self.closed or self.failed or type(final) is not bool
                or type(start_sample) is not int or start_sample != self.received
                or pcm.ndim != 1 or not np.isfinite(pcm).all()):
            raise ValueError('Speech evidence requires contiguous finite mono audio.')
        self.received += len(pcm)
        self.pending = np.concatenate([self.pending, pcm])
        while len(self.pending) >= FRAME or (final and len(self.pending)):
            count = min(FRAME, len(self.pending))
            chunk = np.pad(self.pending[:count], (0, FRAME-count))
            begin = self.evidence.end_sample
            try:
                probability, state = self.model.feed(chunk, self.state)
                self.evidence.append(begin, begin+count, probability)
            except Exception:
                # Recurrent state may have advanced inside the neural adapter.
                # Do not retry this packet or label the unobserved tail silent.
                self.failed = True
                raise
            self.state = state
            self.pending = self.pending[count:]
        if final:
            self.closed = True
        return self.evidence
