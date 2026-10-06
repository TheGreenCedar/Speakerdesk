"""Speech evidence independent of language, speaker identity and Cohere text.

The neural boundary uses the pinned MLX Silero implementation, not a transcript
generator. Missing, incomplete or failed evidence never becomes no-speech.
"""
import hashlib
import json
import math
import os
from pathlib import Path

RATE = 16000
FRAME = 512
HOT_FRAMES = 1875  #60 seconds, exceeding any single admitted decode window.
INPUT_POLICY = 'raw_and_peak025_gaincap256_per512_v1'
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


class FrameArchive:
    """Append original probabilities before pruning hot memory; query by block."""
    def __init__(self, path):
        self.path = Path(path)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self.index = []

    def append(self, frames):
        if not frames:return
        block = (json.dumps(frames, separators=(',', ':'))+'\n').encode()
        with self.path.open('r+b') as output:
            output.seek(0, os.SEEK_END)
            offset = output.tell()
            try:
                output.write(block);output.flush();os.fsync(output.fileno())
            except Exception:
                output.truncate(offset)
                raise
        self.index.append((frames[0][0], frames[-1][1], offset, len(block)))

    def query(self, start, end):
        result = []
        with self.path.open('rb') as source:
            for a, b, offset, size in self.index:
                if b <= start:continue
                if a >= end:break
                source.seek(offset)
                result.extend(frame for frame in json.loads(source.read(size))
                              if frame[0] < end and frame[1] > start)
        return result


class SpeechFrames:
    """An ordered ledger of actual neural frames; partial audio stays pending."""
    def __init__(self, start_sample=0, *, max_frames=None, archive=None):
        if type(start_sample) is not int or start_sample < 0:
            raise ValueError('Invalid speech-evidence origin.')
        self.start_sample = self.end_sample = start_sample
        self.origin_sample = start_sample
        if max_frames is not None and (type(max_frames) is not int or max_frames < 1 or archive is None):
            raise ValueError('Bounded speech evidence requires durable archival.')
        self.max_frames, self.archive = max_frames, archive
        self.frames = []
        self.speaking = False

    def append(self, start, end, probability, *, observation=None):
        if (type(start) is not int or type(end) is not int or start != self.end_sample
                or not start < end <= start+FRAME or type(probability) not in (int, float)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('Invalid or noncontiguous Silero evidence.')
        if self.max_frames is not None and len(self.frames) == self.max_frames:
            count = min(256, self.max_frames)
            self.archive.append(self.frames[:count])
            self.frames = self.frames[count:]
            self.start_sample = self.frames[0][0] if self.frames else start
        # Standard Silero onset/release hysteresis; no minimum speech duration.
        # A single positive32ms frame is retained rather than discarding a word.
        if probability >= .5:
            self.speaking = True
        elif probability < .35:
            self.speaking = False
        frame=(start, end, float(probability), self.speaking)
        self.frames.append(frame+(observation,) if observation is not None else frame)
        self.end_sample = end

    def admission(self, start, end):
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('Invalid speech-admission range.')
        selected = [frame for frame in self.frames if frame[0] < end and frame[1] > start]
        if self.archive is not None and start < self.start_sample:
            selected = self.archive.query(start, min(end, self.start_sample))+selected
        cursor = start
        for frame in selected:
            a,b=frame[:2]
            if a > cursor:break
            cursor = max(cursor, b)
        covered = start >= self.origin_sample and end <= self.end_sample and cursor >= end
        result = {'source': 'silero_v6', 'start_sample': start, 'end_sample': end,
                  'model_revision': SILERO_SPEC['revision'],
                  'observed_until_sample': self.end_sample, 'complete': covered,
                  'decision': 'pending', 'speech_regions': []}
        if not covered:
            return result
        conditioned=all(len(frame)==5 and frame[4].get('input_policy')==INPUT_POLICY for frame in selected)
        if conditioned:
            result['input_policy']=INPUT_POLICY
            # Uncertainty belongs to the fixed neural frame ledger, including
            # negative frames in a query that also contains admitted speech.
            # Query/notification grouping cannot change the final receipt.
            result['uncertain_regions']=[]
            for frame in selected:
                if frame[3] or frame[4].get('constant_value') is not None:continue
                a,b=max(start,frame[0]),min(end,frame[1])
                if result['uncertain_regions'] and result['uncertain_regions'][-1]['end_sample']==a:
                    result['uncertain_regions'][-1]['end_sample']=b
                else:result['uncertain_regions'].append({'start_sample':a,'end_sample':b})
        result['maximum_probability'] = max(frame[2] for frame in selected)
        for frame in selected:
            a,b,_,speaking=frame[:4]
            if not speaking:
                continue
            a, b = max(a, start), min(b, end)
            if result['speech_regions'] and result['speech_regions'][-1]['end_sample'] == a:
                result['speech_regions'][-1]['end_sample'] = b
            else:
                result['speech_regions'].append({'start_sample': a, 'end_sample': b})
        result['decision'] = 'speech' if result['speech_regions'] else 'no_speech'
        if conditioned and not result['speech_regions']:
            # A model-negative waveform is not known silence. Exact digital
            # zero/DC is deterministic non-speech; other negative input stays
            # uncertain and cannot clear words or authorize a Cohere decode.
            constant=not result['uncertain_regions']
            if not constant:result['decision']='uncertain'
            result['negative_signal_state']='exact_constant' if constant else 'model_negative_uncertain'
        return result

    def persist(self):
        if self.archive is not None and self.frames:
            self.archive.append(self.frames)
            self.frames=[]
            self.start_sample=self.end_sample


def checkpoint_files(path):
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
    return expected_branch


def speech_issues(path):
    try:
        checkpoint_files(path)
        return []
    except (OSError,ValueError,TypeError):
        return ['Speech detection needs its verified local model. Download models in Settings.']


class SileroModel:
    """Strict16k branch of the reviewed dependency; explicit recurrent state."""
    def __init__(self, path):
        self.normalized_view=True
        path = Path(path)
        expected_branch = checkpoint_files(path)
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

    def session(self, start_sample=0, *, archive=None):
        return SpeechSession(self, start_sample, max_frames=HOT_FRAMES if archive else None, archive=archive)

    def inspect_frames(self, audio, start_sample=0, *, archive=None):
        session = self.session(start_sample,archive=archive)
        session.feed(audio, start_sample, final=True)
        return session.evidence

    def inspect(self, audio, start_sample=0):
        return self.inspect_frames(audio, start_sample).admission(start_sample, start_sample+len(audio))


class SpeechSession:
    """Carry64 context samples and LSTM state across contiguous capture packets."""
    def __init__(self, model, start_sample=0, *, max_frames=None, archive=None):
        import numpy as np
        self.model, self.state = model, model.initial_state()
        self.normalized_state=model.initial_state() if getattr(model,'normalized_view',False) else None
        self.evidence = SpeechFrames(start_sample, max_frames=max_frames, archive=archive)
        self.received = start_sample
        self.pending = np.empty(0, dtype=np.float32)
        self.closed = False
        self.failed = False
        self.previous_sample = None
        self.inspected_pcm = hashlib.sha256()
        self.inspected_counts = {'speech_samples':0,'uncertain_samples':0,'negative_constant_samples':0}

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
                observation=None
                if getattr(self.model,'normalized_view',False):
                    raw=chunk[:count];peak=float(np.max(np.abs(raw)))
                    # Conditioning is only for a second VAD view. Original PCM,
                    # NVIDIA and Cohere input are untouched. Gain is independent
                    # of labels, window size, packets and language; no clipping.
                    gain=min(256.,max(1.,.25/peak)) if peak else 1.
                    normalized_probability,normalized_state=self.model.feed(chunk*gain,self.normalized_state)
                    if (not math.isfinite(normalized_probability) or not 0<=normalized_probability<=1
                            or not math.isfinite(probability) or not 0<=probability<=1):
                        raise ValueError('Invalid neural speech probability.')
                    observation={'input_policy':INPUT_POLICY,'raw_probability':probability,
                        'normalized_probability':normalized_probability,'gain':gain,
                        'constant_value':float(raw[0]) if (bool(np.all(raw==raw[0]))
                            and (self.previous_sample is None or self.previous_sample==float(raw[0]))) else None}
                    probability=max(probability,normalized_probability)
                    self.normalized_state=normalized_state
                self.evidence.append(begin, begin+count, probability,observation=observation)
                # Hash only original physical samples after successful neural
                # inspection. Virtual final-frame padding/normalized VAD input
                # never enters the retained-PCM identity.
                self.inspected_pcm.update(np.clip(np.rint(chunk[:count]*32768),-32768,32767).astype('<i2').tobytes())
                speaking=self.evidence.speaking
                key=('speech_samples' if speaking else 'uncertain_samples'
                     if observation is None or observation.get('constant_value') is None else 'negative_constant_samples')
                self.inspected_counts[key]+=count
            except Exception:
                # Recurrent state may have advanced inside the neural adapter.
                # Do not retry this packet or label the unobserved tail silent.
                self.failed = True
                raise
            self.state = state
            self.previous_sample=float(chunk[count-1])
            self.pending = self.pending[count:]
        if final:
            try:self.evidence.persist()
            except Exception:
                self.failed=True
                raise
            self.closed = True
        return self.evidence

    def inspection(self):
        if self.failed or not getattr(self.model,'normalized_view',False):
            raise ValueError('Verified neural inspection is unavailable.')
        return {'inspection_state':'observed_prefix','start_sample':self.evidence.origin_sample,
                'end_sample':self.evidence.end_sample,'received_sample':self.received,'closed':self.closed,
                'audio_encoding':'pcm_s16le','pcm_sha256':self.inspected_pcm.hexdigest(),
                **self.inspected_counts,
                'decision':'speech' if self.inspected_counts['speech_samples'] else
                    'uncertain' if self.inspected_counts['uncertain_samples'] else 'no_speech'}
