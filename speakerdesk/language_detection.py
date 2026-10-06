"""Local audio language routing. Only Cohere generates transcript text.

Whisper's language-token pass scores all 99 languages. Uncertain supported
speech is attempted with recent context or a marked best-effort language.
Unsupported speech and genuine ASR failures retain audio for review.
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
    """Describe acoustic evidence without making speaker identity an ASR gate."""
    def __init__(self):
        self.current = {}

    def reset(self, speaker):
        self.current.pop(speaker, None)

    def decide(self, probabilities, speaker):
        if (not probabilities or any(not isinstance(code, str) or not math.isfinite(float(p))
                                    or not 0 <= float(p) <= 1 for code, p in probabilities.items())
                or not .98 <= sum(float(p) for p in probabilities.values()) <= 1.02):
            raise ValueError('Invalid language detection probabilities.')
        ranked = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
        winner, probability = ranked[0][0], float(ranked[0][1])
        margin = probability - (float(ranked[1][1]) if len(ranked) > 1 else 0.)
        confident = probability >= .90 and margin >= .20
        decision = {'language': None, 'language_detection': {
            'mode': 'auto', 'probability': probability, 'margin': margin,
            'candidates': [{'language': code, 'probability': float(p)} for code, p in ranked[:3]],
            'reason': 'needs_language'}, 'review': True}
        if winner not in LANGUAGES:
            if confident:
                self.reset(speaker)
                decision['language_detection']['reason'] = 'unsupported'
            return decision
        decision['language'] = winner
        decision['language_detection']['reason'] = 'detected' if confident else 'best_effort'
        decision['review'] = not confident
        if confident:self.current[speaker] = winner
        return decision


class SpeechTranscriber:
    """Shared routing with successful, recent supported-language context.

    Context is an explicit snapshot preceding this audio, not an ASR confidence
    score. Callers processing historical audio must seed only preceding context
    from the same language epoch. All positions are absolute 16 kHz samples.
    """
    CONTEXT_SAMPLES = 60 * 16000

    def __init__(self, asr, language, detector_path=None, *, detector=None, context=None):
        self.asr,self.detector_path,self.detector=asr,detector_path,detector
        self.set_language(language)
        if context is not None:
            if (not isinstance(context,dict) or not isinstance(context.get('language'),str)
                    or context['language'] not in LANGUAGES
                    or type(context.get('end_sample')) is not int or context['end_sample']<0):
                raise ValueError('Invalid preceding language context.')
            self.context = {'language':context['language'],'end_sample':context['end_sample']}

    def set_language(self,language):
        if language not in LANGUAGE_CHOICES:raise ValueError('Unsupported language mode.')
        if language=='auto' and self.detector is None:self.detector=WhisperLanguageDetector(self.detector_path)
        if getattr(self,'language',None)!=language:
            self.language=language
            self.policy=LanguagePolicy()
            self.context=None

    def recent_context(self,start_sample):
        if self.context and 0 <= start_sample-self.context['end_sample'] <= self.CONTEXT_SAMPLES:
            return self.context
        return None

    def transcribe(self, audio, sample_rate, speaker, *, max_asr_seconds=6, allow_overlap=False, start_sample=0):
        import numpy as np

        if (not .75 <= max_asr_seconds <= 24.5 or len(audio) > 24.5*sample_rate
                or not len(audio) or sample_rate != 16000 or np.asarray(audio).ndim != 1 or not np.isfinite(audio).all()
                or type(start_sample) is not int or start_sample<0):
            raise ValueError('Language routing requires finite 16 kHz mono speech audio and its sample position.')
        automatic=self.language=='auto'
        count = max(1, math.ceil(len(audio)/(3*sample_rate))) if automatic else 1
        edges = np.linspace(0, len(audio), count+1, dtype=int)
        probes=[]
        for begin,end in zip(edges,edges[1:]):
            pcm=audio[begin:end]
            if not len(pcm) or not np.any(pcm):
                decision={'language':None if automatic else self.language,'review':True,
                          'language_detection':{'mode':'auto' if automatic else 'manual','reason':'insufficient_speech'},
                          'audio_state':'digital_silence'}
            elif not automatic:
                decision={'language':self.language,'review':False,
                          'language_detection':{'mode':'manual','reason':'override'}}
            else:
                try:
                    decision=self.policy.decide(self.detector.detect(pcm),speaker)
                except (RuntimeError,ValueError,OSError):
                    decision={'language':None,'review':True,
                              'language_detection':{'mode':'auto','reason':'needs_language','detector_error':True}}
            probes.append({'begin':int(begin),'end_sample':int(end),**decision})
        results=[]
        index=0
        while index<len(probes):
            window=probes[index];index+=1
            detection=window['language_detection']
            if automatic and detection['reason']=='unsupported':
                self.context=None
            elif (automatic and detection['reason']=='detected' and self.context
                  and self.context['language']!=window['language']):
                # A clear language change invalidates old context even if ASR
                # fails; only successful new speech can establish its successor.
                self.context=None
            elif automatic and detection['reason'] in ('best_effort','needs_language'):
                context=self.recent_context(start_sample+window['begin'])
                if context:
                    window['language']=context['language']
                    detection.update(reason='recent_context',context_end_sample=context['end_sample'])
            # Join adjacent confident windows while preserving whole audio crops.
            while (not window['review'] and index<len(probes) and not probes[index]['review']
                   and probes[index]['language']==window['language']
                   and probes[index]['end_sample']-window['begin']<=max_asr_seconds*sample_rate):
                following=probes[index];index+=1;window['end_sample']=following['end_sample']
                if following['language_detection'].get('probability',1)<detection.get('probability',1):
                    detection=following['language_detection'];window['language_detection']=detection
            text='';transcription_review=None
            if window['language'] and detection['reason']!='insufficient_speech':
                try:
                    result=self.asr.transcribe(audio[window['begin']:window['end_sample']],
                        sample_rate=sample_rate,language=window['language'],max_new_tokens=448)
                    text=result.text.strip()
                    if len(result.tokens)>=448:
                        transcription_review={'reason':'token_limit','partial_text':True}
                    elif not text:
                        transcription_review={'reason':'empty_result','partial_text':False}
                except (RuntimeError,ValueError,OSError):
                    transcription_review={'reason':'transcription_failed','partial_text':False}
            if automatic and detection['reason']=='detected' and text and not transcription_review:
                self.context={'language':window['language'],'end_sample':start_sample+window['end_sample']}
            if transcription_review or len(speaker)>1:window['review']=True
            results.append({'start':window['begin']/sample_rate,'end':window['end_sample']/sample_rate,'text':text,
                            **{key:window[key] for key in ('language','language_detection','review')},
                            **({'audio_state':window['audio_state']} if 'audio_state' in window else {}),
                            **({'transcription_review':transcription_review} if transcription_review else {})})
        return results
