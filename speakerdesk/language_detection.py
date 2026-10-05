"""Local audio language routing. Only Cohere generates transcript text.

Whisper's dedicated language-token pass scores all 99 languages. Unsupported
or uncertain speech is retained for review instead of being forced to English.
Policy thresholds are conservative defaults, pending real bilingual evaluation.
"""
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from transcript import LANGUAGES

LANGUAGE_CHOICES = {'auto': 'Automatic', **LANGUAGES}
LID_SPEC = {
    'name': 'Automatic language detection',
    'repo': 'mlx-community/whisper-tiny-asr-fp16',
    'revision': '77fa3f52b482ec80d086df55f893065a9baab172',
    'directory': 'whisper-language', 'bytes': 74385959,
    'sha256': '1267601753d2996d68dc065d1de786895a7ad9a874d1d3548a5274c1bc067444',
    'files': ['config.json', 'added_tokens.json', 'model.safetensors', 'README.md'],
    'file_sha256': {
        'model.safetensors': '1267601753d2996d68dc065d1de786895a7ad9a874d1d3548a5274c1bc067444',
        'config.json': 'cf7def105e7cc5ba77a0c8fae7e97aac1ca896b70b288dec3d64fad6d9c08f7a',
        'added_tokens.json': '9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1',
        'README.md': 'd68f867f75885b5fb22073dee9befb9bcb5586a4ba1cdfe39153f0335b2ca9e4'},
}
LID_CHECKPOINT = LID_SPEC['repo'] + '@' + LID_SPEC['revision']


def language_tokens(tokens, vocab_size):
    """Minimal adapter for the pinned language-only decoding API; no tokenizer."""
    languages = {token[2:-2]: value for token, value in tokens.items()
                 if re.fullmatch(r'<\|[a-z]{2,3}\|>', token)}
    ids = list(languages.values())
    sot = tokens.get('<|startoftranscript|>')
    if (len(languages) != 99 or not {'en', 'fr'} <= languages.keys()
            or any(type(value) is not int or not 0 <= value < vocab_size for value in ids)
            or len(set(ids)) != 99 or sorted(ids) != list(range(min(ids), min(ids)+99))
            or type(sot) is not int or sot != min(ids)-1):
        raise ValueError('Incompatible multilingual language-token metadata.')
    # `language` is only the dependency's capability guard; the decoder receives
    # SOT alone and scores ALL language tokens, without an English prefix.
    return SimpleNamespace(language='en', language_token=languages['en'],
        sot_sequence=(sot, languages['en']), sot=sot,
        all_language_codes=tuple(languages), all_language_tokens=tuple(ids))


def detector_files(path):
    path = Path(path)
    for name in ('config.json', 'added_tokens.json'):
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != LID_SPEC['file_sha256'][name]:
            raise ValueError('Language metadata verification failed.')
    config = json.loads((path/'config.json').read_text())
    # Restrict this loader to the reviewed multilingual tiny architecture.
    expected = {'model_type': 'whisper', 'num_mel_bins': 80, 'max_source_positions': 1500,
                'd_model': 384, 'encoder_attention_heads': 6, 'encoder_layers': 4,
                'vocab_size': 51865, 'max_target_positions': 448,
                'decoder_attention_heads': 6, 'decoder_layers': 4}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Incompatible Whisper tiny language checkpoint.')
    tokens = language_tokens(json.loads((path/'added_tokens.json').read_text()), config['vocab_size'])
    if (path/'model.safetensors').stat().st_size != LID_SPEC['bytes']:
        raise ValueError('Missing or incomplete language checkpoint.')
    return config, tokens


def detector_issues(path):
    try:
        detector_files(path)
        return []
    except (OSError, ValueError, TypeError, AttributeError):
        return ['Automatic language detection needs its local model. Download models in Settings, or choose a language override.']


class WhisperLanguageDetector:
    def __init__(self, path):
        config, self.tokens = detector_files(path)
        weights = Path(path)/'model.safetensors'
        with weights.open('rb') as content:
            digest = hashlib.file_digest(content, 'sha256').hexdigest()
        if digest != LID_SPEC['sha256']:
            raise RuntimeError('Language checkpoint verification failed. Download it again in Settings.')
        import mlx.core as mx
        from mlx_audio.stt.models.whisper.whisper import Model, ModelConfig

        self.model = Model(ModelConfig.from_dict(config))
        self.model.load_weights(list(self.model.sanitize(mx.load(str(weights))).items()), strict=True)
        mx.eval(self.model.parameters())

    def detect(self, audio):
        import mlx.core as mx
        from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES
        from mlx_audio.stt.models.whisper.decoding import detect_language

        # Whisper's 30-second context must be padded in waveform space. Numeric
        # zero Mel features do not represent the frontend's log-Mel silence.
        mel = log_mel_spectrogram(pad_or_trim(mx.array(audio), N_SAMPLES),
                                 n_mels=self.model.dims.n_mels).astype(self.model.dtype)
        _, probabilities = detect_language(self.model, mel, tokenizer=self.tokens)
        return probabilities


class LanguagePolicy:
    """Track each diarization speaker separately; never reuse a disputed language."""
    def __init__(self):
        self.current = {}
        self.pending = {}

    def reset(self, speaker):
        for name in speaker:
            self.current.pop((name,), None)
            self.pending.pop((name,), None)

    def decide(self, probabilities, speaker):
        if (not probabilities or any(not isinstance(code, str) or not math.isfinite(float(p))
                                    or not 0 <= float(p) <= 1 for code, p in probabilities.items())
                or not .98 <= sum(float(p) for p in probabilities.values()) <= 1.02):
            raise ValueError('Invalid language detection probabilities.')
        ranked = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
        winner, probability = ranked[0][0], float(ranked[0][1])
        margin = probability - (float(ranked[1][1]) if len(ranked) > 1 else 0.)
        decision = {'language': None, 'language_detection': {
            'mode': 'auto', 'probability': probability, 'margin': margin,
            'candidates': [{'language': code, 'probability': float(p)} for code, p in ranked[:3]],
            'reason': 'uncertain'}, 'review': True}
        if winner not in LANGUAGES:
            self.reset(speaker)
            decision['language_detection']['reason'] = 'unsupported'
            return decision
        previous = self.current.get(speaker)
        strong = probability >= .95 and margin >= .30
        confident = probability >= .90 and margin >= .20
        # A prior language does not rescue weak acoustic evidence. Real compact
        # TTS probes exposed a moderate-confidence wrong-language admission.
        if confident and previous and winner != previous and not strong:
            candidate, count = self.pending.get(speaker, (None, 0))
            count = count + 1 if candidate == winner else 1
            self.pending[speaker] = (winner, count)
            if count < 2:
                decision['language_detection']['reason'] = 'change_pending'
                return decision
        elif not confident:
            self.reset(speaker)
            return decision
        self.pending.pop(speaker, None)
        self.current[speaker] = winner
        decision.update(language=winner, review=False)
        decision['language_detection']['reason'] = 'detected'
        return decision


class SpeechTranscriber:
    """Shared import/live routing with bounded windows and original-language ASR."""
    def __init__(self, asr, language, detector_path=None):
        if language not in LANGUAGE_CHOICES:
            raise ValueError('Unsupported language mode.')
        self.asr, self.language = asr, language
        self.detector = WhisperLanguageDetector(detector_path) if language == 'auto' else None
        self.policy = LanguagePolicy()

    def transcribe(self, audio, sample_rate, speaker):
        import numpy as np

        if sample_rate != 16000 or np.asarray(audio).ndim != 1 or not np.isfinite(audio).all():
            raise ValueError('Language routing requires finite 16 kHz mono speech audio.')
        count = max(1, math.ceil(len(audio)/(3*sample_rate))) if self.detector else 1
        edges = np.linspace(0, len(audio), count+1, dtype=int)
        windows = []
        for begin, end in zip(edges, edges[1:]):
            pcm = audio[begin:end]
            if not self.detector:
                decision = {'language': self.language, 'review': False,
                            'language_detection': {'mode': 'manual', 'reason': 'override'}}
            elif len(pcm) < .75*sample_rate or float(np.sqrt(np.mean(pcm.astype('float64')**2))) < .001:
                self.policy.reset(speaker)
                decision = {'language': None, 'review': True,
                            'language_detection': {'mode': 'auto', 'reason': 'insufficient_speech'}}
            elif len(speaker) > 1:
                self.policy.reset(speaker)
                decision = {'language': None, 'review': True,
                            'language_detection': {'mode': 'auto', 'reason': 'overlapping_speech'}}
            else:
                decision = self.policy.decide(self.detector.detect(pcm), speaker)
            window = {'begin': int(begin), 'end_sample': int(end), **decision}
            # Join adjacent confident windows of the same language up to six
            # seconds, retaining the least confident language observation.
            if (windows and window['language'] and not window['review']
                    and not windows[-1]['review'] and windows[-1]['language'] == window['language']
                    and end-windows[-1]['begin'] <= 6*sample_rate):
                previous = windows[-1]
                previous['end_sample'] = int(end)
                if window['language_detection'].get('probability', 1) < previous['language_detection'].get('probability', 1):
                    previous['language_detection'] = window['language_detection']
            else:
                windows.append(window)
        results = []
        for window in windows:
            text = ''
            transcription_review = None
            if window['language']:
                try:
                    result = self.asr.transcribe(audio[window['begin']:window['end_sample']],
                        sample_rate=sample_rate, language=window['language'], max_new_tokens=448)
                    text = result.text.strip()
                    if len(result.tokens) >= 448:
                        transcription_review = {'reason': 'token_limit', 'partial_text': True}
                    elif not text:
                        transcription_review = {'reason': 'empty_result', 'partial_text': False}
                except (RuntimeError, ValueError, OSError):
                    # Keep earlier successful passages. Hardware/resource guards remain
                    # outside this per-passage model boundary and stop owned workers.
                    transcription_review = {'reason': 'transcription_failed', 'partial_text': False}
            if transcription_review:
                window['review'] = True
            results.append({'start': window['begin']/sample_rate,
                            'end': window['end_sample']/sample_rate, 'text': text,
                            **{key: window[key] for key in ('language', 'language_detection', 'review')},
                            **({'transcription_review': transcription_review} if transcription_review else {})})
        return results
