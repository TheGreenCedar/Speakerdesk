"""Align supplied Cohere text; no ASR, audio reads, downloads or model loading.

This isolated prototype consumes normalized CTC scores. Sample spans represent
observed CTC emission cells, not phonetic word edges or proof of speech. Without
an approved frame calibration and score policy, text remains unresolved.
"""
from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import math
import re
import sys
import unicodedata

RATE = 16000
MAX_AUDIO_SAMPLES = 392000  # Existing 24.5 second refinement bound.
MAX_TEXT_CHARACTERS = 4096
MAX_TARGET_TOKENS = 1024
MODEL_SHA256 = 'e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c'


@dataclass(frozen=True)
class TargetToken:
    token_id: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class PreparedText:
    raw_text: str
    targets: tuple
    unsupported: tuple
    text_sha256: str
    acoustic_aliases: tuple = ()


@dataclass(frozen=True)
class TokenAnchor:
    token_index: int
    token_id: int
    frame: int
    log_probability: float
    preceding_blank_frame: object = None
    preceding_blank_log_probability: object = None


@dataclass(frozen=True)
class FrameClock:
    hop_samples: int
    origin_samples: object = None
    calibration_id: object = None
    model_sha256: str = MODEL_SHA256


@dataclass(frozen=True)
class AcceptancePolicy:
    minimum_log_probability: float
    calibration_id: str
    model_sha256: str = MODEL_SHA256


@dataclass(frozen=True)
class AlignmentRequest:
    utterance_id: str
    machine_revision: int
    audio_revision: int
    raw_text: str
    start_sample: int
    end_sample: int

    @classmethod
    def from_utterance(cls, row):
        """Snapshot the 1599e815 canonical text/audio contract without PCM reads."""
        start, end = row['start_sample'], row['end_sample']
        if (not isinstance(row['id'], str) or not row['id']
                or type(row['machine_revision']) is not int or row['machine_revision'] < 0
                or type(row['audio_revision']) is not int or row['audio_revision'] < 0
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end or end-start > MAX_AUDIO_SAMPLES
                or not isinstance(row['text'], str) or not row['text'].strip()
                or len(row['text']) > MAX_TEXT_CHARACTERS
                or row.get('text_audio_anchor') != {'start_sample': start, 'end_sample': end}):
            raise ValueError('Cohere text and current utterance audio must share an exact anchor.')
        return cls(row['id'], row['machine_revision'], row['audio_revision'], row['text'], start, end)


def parse_vocabulary(contents):
    """Parse exported token IDs, preserving the literal space token."""
    by_id = {}
    for line in contents.splitlines():
        token, separator, index = line.rpartition(' ')
        if not separator or not token or not index.isdecimal():
            raise ValueError('Invalid CTC vocabulary line.')
        number = int(index)
        if number in by_id:
            raise ValueError('Duplicate CTC token ID.')
        by_id[number] = token
    if set(by_id) != set(range(len(by_id))) or len(set(by_id.values())) != len(by_id):
        raise ValueError('CTC vocabulary IDs and tokens must be unique and contiguous.')
    if [by_id.get(i) for i in range(5)] != ['<s>', '<pad>', '</s>', '<unk>', ' ']:
        raise ValueError('Unexpected Omnilingual reserved tokens.')
    return tuple(by_id[i] for i in range(len(by_id)))


def _normalized_characters(raw_text):
    """NFC scalar mapping, including compositions spanning raw code points."""
    normalized = ''
    mapped = []
    for index, character in enumerate(raw_text):
        updated = unicodedata.normalize('NFC', normalized + character)
        common = 0
        while common < min(len(normalized), len(updated)) and normalized[common] == updated[common]:
            common += 1
        begin = min((a for _, a, _ in mapped[common:]), default=index)
        mapped[common:] = [(c, begin, index+1) for c in updated[common:]]
        normalized = updated
    return mapped


def prepare_text(raw_text, vocabulary):
    """Keep raw text, digits, repeats and offsets; never drop unknown letters."""
    if not isinstance(raw_text, str) or len(raw_text) > MAX_TEXT_CHARACTERS:
        raise ValueError('Supplied text exceeds the bounded alignment contract.')
    lookup = {token: index for index, token in enumerate(vocabulary) if index >= 4}
    targets, unsupported, aliases = [], [], []
    for character, begin, end in _normalized_characters(raw_text):
        category = unicodedata.category(character)
        if character.isspace():
            if targets and targets[-1].text == ' ':
                previous = targets.pop()
                targets.append(TargetToken(previous.token_id, ' ', previous.start_char, end))
                continue
            character = ' '
        elif category.startswith('P'):
            # Punctuation remains in raw display units; it is not an acoustic
            # target. Numbers such as 12.5 remain one unchanged display unit.
            continue
        if character not in lookup:
            lower = character.lower()
            # This export omits the capital letters seen in actual Cohere
            # snapshots. Map only a supported one-scalar lowercase alias;
            # retain the exact raw span/hash and never expand/romanize text.
            if lower != character and len(lower) == 1 and lower in lookup:
                aliases.append({'start_char':begin,'end_char':end,
                    'raw_text':raw_text[begin:end],'acoustic_token':lower,
                    'kind':'supported_single_character_lowercase_alias'})
                character = lower
        if character not in lookup:
            unsupported.append({'start_char': begin, 'end_char': end, 'text': raw_text[begin:end]})
            continue
        targets.append(TargetToken(lookup[character], character, begin, end))
    if len(targets) > MAX_TARGET_TOKENS:
        raise ValueError('Supplied text exceeds the bounded CTC target limit.')
    while targets and targets[0].text == ' ':
        targets.pop(0)
    while targets and targets[-1].text == ' ':
        targets.pop()
    return PreparedText(raw_text, tuple(targets), tuple(unsupported),
                        hashlib.sha256(raw_text.encode('utf-8')).hexdigest(),tuple(aliases))


def _utf16_offset(text, index):
    return len(text[:index].encode('utf-16-le')) // 2


def _display_units(prepared):
    units = []
    for match in re.finditer(r'\S+', prepared.raw_text):
        a, b = match.span()
        units.append({'text': match.group(), 'start_char': a, 'end_char': b,
                      'start_utf16': _utf16_offset(prepared.raw_text, a),
                      'end_utf16': _utf16_offset(prepared.raw_text, b),
                      'unit_kind': 'whitespace_run', 'status': 'unresolved',
                      'start_sample': None, 'end_sample': None})
    return units


def _valid_log_probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value <= 0


def materialize(prepared, anchors, *, clock, policy, audio_start_sample,
                audio_num_samples, model_sha256=MODEL_SHA256):
    """Validate acoustic anchors before converting them to absolute sample cells.

    An unresolved unit has null sample positions. Consumers may attach complete
    results to a matching utterance revision; partial results are review data.
    Whitespace runs are display units, not a linguistic tokenizer for CJK text.
    """
    if (type(audio_start_sample) is not int or audio_start_sample < 0
            or type(audio_num_samples) is not int or not 0 < audio_num_samples <= MAX_AUDIO_SAMPLES
            or model_sha256 != MODEL_SHA256):
        raise ValueError('Invalid audio range or acoustic model provenance.')
    result = {'schema_version': 1, 'text_source': 'cohere', 'raw_text': prepared.raw_text,
              'text_sha256': prepared.text_sha256, 'model_sha256': model_sha256,
              'offset_unit': 'unicode_codepoint', 'timing_kind': 'ctc_emission_cell_envelope',
              'audio_anchor': {'start_sample': audio_start_sample,
                               'end_sample': audio_start_sample+audio_num_samples},
              'status': 'unresolved', 'complete': False, 'reason': None,
              'words': _display_units(prepared), 'characters': [],
              'unsupported': list(prepared.unsupported),
              'acoustic_aliases': list(prepared.acoustic_aliases), 'non_speech_proof': False}
    if prepared.unsupported:
        result['reason'] = 'unsupported_text'
        return result
    if not any(not t.text.isspace() for t in prepared.targets):
        result['reason'] = 'no_acoustic_target'
        return result
    if (clock is None or type(clock.hop_samples) is not int or clock.hop_samples <= 0
            or type(clock.origin_samples) is not int
            or not isinstance(clock.calibration_id, str) or not clock.calibration_id
            or clock.model_sha256 != model_sha256):
        result['reason'] = 'frame_clock_unverified'
        return result
    if (policy is None or not isinstance(policy.calibration_id, str) or not policy.calibration_id
            or policy.model_sha256 != model_sha256
            or not _valid_log_probability(policy.minimum_log_probability)):
        result['reason'] = 'acceptance_policy_unverified'
        return result
    result.update(frame_calibration_id=clock.calibration_id, score_calibration_id=policy.calibration_id)
    anchors = tuple(anchors)
    previous_frame = -1
    if len(anchors) != len(prepared.targets):
        result['reason'] = 'incomplete_anchor_sequence'
        return result
    for index, (target, anchor) in enumerate(zip(prepared.targets, anchors)):
        if (type(anchor.token_index) is not int or anchor.token_index != index
                or anchor.token_id != target.token_id
                or type(anchor.frame) is not int or anchor.frame <= previous_frame
                or not _valid_log_probability(anchor.log_probability)):
            result['reason'] = 'invalid_anchor_sequence'
            return result
        previous_frame = anchor.frame
    for index, (target, anchor) in enumerate(zip(prepared.targets, anchors)):
        begin = audio_start_sample+clock.origin_samples+anchor.frame*clock.hop_samples
        end = begin+clock.hop_samples
        accepted = (audio_start_sample <= begin < end <= audio_start_sample+audio_num_samples
                    and anchor.log_probability >= policy.minimum_log_probability)
        if index and target.token_id == prepared.targets[index-1].token_id:
            accepted = accepted and (type(anchor.preceding_blank_frame) is int
                and anchors[index-1].frame < anchor.preceding_blank_frame < anchor.frame
                and _valid_log_probability(anchor.preceding_blank_log_probability)
                and anchor.preceding_blank_log_probability >= policy.minimum_log_probability)
        result['characters'].append({'text': target.text, 'token_id': target.token_id,
            'raw_text':prepared.raw_text[target.start_char:target.end_char],
            'start_char': target.start_char, 'end_char': target.end_char,
            'start_utf16': _utf16_offset(prepared.raw_text, target.start_char),
            'end_utf16': _utf16_offset(prepared.raw_text, target.end_char),
            'frame': anchor.frame, 'log_probability': anchor.log_probability,
            'status': 'aligned' if accepted else 'unresolved',
            'start_sample': begin if accepted else None, 'end_sample': end if accepted else None})
    for word in result['words']:
        aliases = [item for item in prepared.acoustic_aliases
                   if word['start_char'] <= item['start_char'] < item['end_char'] <= word['end_char']]
        if aliases:
            word['acoustic_aliases'] = aliases
        characters = [c for c in result['characters']
                      if word['start_char'] <= c['start_char'] < c['end_char'] <= word['end_char']]
        if not characters:
            word['status'] = 'not_acoustic'
        elif all(c['status'] == 'aligned' for c in characters):
            word.update(status='aligned', start_sample=characters[0]['start_sample'],
                        end_sample=characters[-1]['end_sample'],
                        minimum_token_log_probability=min(c['log_probability'] for c in characters))
    result['complete'] = (all(c['status'] == 'aligned' for c in result['characters'])
                          and all(w['status'] == 'aligned' for w in result['words']))
    result['status'] = 'aligned' if result['complete'] else 'partial'
    result['reason'] = None if result['complete'] else 'insufficient_acoustic_evidence'
    return result


def attachment_for(result, request, current_utterance):
    """Produce a CAS-bound attachment only for complete raw-text coverage."""
    if AlignmentRequest.from_utterance(current_utterance) != request:
        raise ValueError('Utterance text, audio or revisions changed during alignment.')
    expected_hash = hashlib.sha256(request.raw_text.encode('utf-8')).hexdigest()
    if (result.get('status') != 'aligned' or result.get('complete') is not True
            or result.get('raw_text') != request.raw_text or result.get('text_sha256') != expected_hash
            or result.get('model_sha256') != MODEL_SHA256
            or result.get('audio_anchor') != {'start_sample': request.start_sample, 'end_sample': request.end_sample}):
        raise ValueError('Only complete matching Cohere alignment can attach.')
    if (result.get('timing_kind')!='ctc_emission_cell_envelope'
            or any(not isinstance(result.get(key),str) or not result[key].strip()
                   for key in ('frame_calibration_id','score_calibration_id'))):
        raise ValueError('Alignment timing kind and calibration identifiers are required.')
    words = result.get('words')
    raw_units = list(re.finditer(r'\S+', request.raw_text))
    if not isinstance(words, list) or not words or len(words) != len(raw_units):
        raise ValueError('Alignment does not cover the raw text display units.')
    previous = request.start_sample
    for word, raw in zip(words, raw_units):
        a, b = word.get('start_sample'), word.get('end_sample')
        if (word.get('status') != 'aligned' or word.get('text') != raw.group()
                or type(word.get('start_char')) is not int or type(word.get('end_char')) is not int
                or (word.get('start_char'), word.get('end_char')) != raw.span()
                or type(a) is not int or type(b) is not int or not previous <= a < b <= request.end_sample):
            raise ValueError('Alignment has unresolved, reordered or out-of-range text.')
        previous = b
    return {'id': request.utterance_id, 'machine_revision': request.machine_revision,
            'audio_revision': request.audio_revision, 'text_sha256': expected_hash,
            'words': [dict(word, model_sha256=MODEL_SHA256,
                           timing_kind=result['timing_kind'],
                           frame_calibration_id=result['frame_calibration_id'],
                           score_calibration_id=result['score_calibration_id']) for word in words],
            'model_sha256': MODEL_SHA256,
            'timing_kind': result['timing_kind'],
            'frame_calibration_id': result['frame_calibration_id'],
            'score_calibration_id': result['score_calibration_id']}


def annotate_speakers(result, activity):
    """Preserve concurrent/partial activity rather than inventing a word owner."""
    intervals = []
    for row in activity:
        a, b, speaker = row['start_sample'], row['end_sample'], row['speaker']
        if type(a) is not int or type(b) is not int or not 0 <= a < b or not isinstance(speaker, str) or not speaker:
            raise ValueError('Invalid absolute speaker activity.')
        intervals.append((a, b, speaker))
    words = []
    for original in result['words']:
        word = dict(original)
        word.update(speaker='unknown', speaker_state='unknown', speaker_candidates=[])
        if word['status'] == 'aligned':
            a, b = word['start_sample'], word['end_sample']
            hits = [(max(a, x), min(b, y), s) for x, y, s in intervals if x < b and y > a]
            names = sorted({s for _, _, s in hits})
            word['speaker_candidates'] = names
            if len(names) > 1:
                word['speaker_state'] = 'multiple_speakers'
            elif names:
                cursor = a
                for x, y, _ in sorted(hits):
                    if x > cursor:
                        break
                    cursor = max(cursor, y)
                if cursor >= b:
                    word.update(speaker=names[0], speaker_state='single_speaker')
        words.append(word)
    return words


def align_ctc_scores(raw_text, log_probabilities, vocabulary, *, clock, policy,
                     audio_start_sample, audio_num_samples, model_sha256=MODEL_SHA256):
    """Source-only runtime adapter for precomputed, normalized CTC scores.

    This does not load or invoke an acoustic model. The source-only test suite
    never calls the optional CTC package or invents a calibrated production policy.
    """
    prepared = prepare_text(raw_text, vocabulary)
    unresolved = materialize(prepared, (), clock=clock, policy=policy,
        audio_start_sample=audio_start_sample, audio_num_samples=audio_num_samples,
        model_sha256=model_sha256)
    if unresolved['reason'] != 'incomplete_anchor_sequence':
        return unresolved
    import numpy as np
    scores = np.asarray(log_probabilities)
    if (len(vocabulary) != 9812 or scores.ndim != 2 or scores.shape[1] != len(vocabulary)
            or not 0 < scores.shape[0] <= audio_num_samples//clock.hop_samples+2
            or scores.dtype != np.float32 or scores.nbytes > 64*1024**2
            or not np.isfinite(scores).all() or (scores > 0).any()
            or (np.abs(np.logaddexp.reduce(scores, axis=1)) > .001).any()):
        unresolved['reason'] = 'invalid_ctc_scores'
        return unresolved
    # Check native-extension presence without importing the package: its import
    # fallback otherwise compiles Cython dynamically on the user's machine.
    try:
        package = importlib.util.find_spec('ctc_segmentation')
        native = package and importlib.machinery.PathFinder.find_spec(
            'ctc_segmentation.ctc_segmentation_dyn', package.submodule_search_locations)
        if (native is None or not isinstance(native.loader, importlib.machinery.ExtensionFileLoader)
                or importlib.metadata.version('ctc-segmentation') != '1.7.4'):
            raise ImportError('Pinned native alignment extension unavailable.')
        # Load the native extension before the package. A missing/invalid binary
        # then fails here, before the Python core can invoke its JIT fallback.
        if native.name not in sys.modules:
            extension = importlib.util.module_from_spec(native)
            native.loader.exec_module(extension)
            sys.modules[native.name] = extension
        import ctc_segmentation as ctc
    except (ImportError, OSError, ValueError, importlib.metadata.PackageNotFoundError):
        unresolved['reason'] = 'pinned_precompiled_aligner_unavailable'
        return unresolved
    expanded, positions, separators = [], [], {}
    for index, target in enumerate(prepared.targets):
        if index and target.token_id == prepared.targets[index-1].token_id:
            separators[index] = 2+len(expanded)
            expanded.append(0)
        positions.append(2+len(expanded))
        expanded.append(target.token_id)
    config = ctc.CtcSegmentationParameters(char_list=list(vocabulary), blank=0,
        index_duration=1.0, self_transition='<alignment-stay>',
        start_of_ground_truth='<alignment-start>', excluded_characters='')
    try:
        ground_truth, _ = ctc.prepare_token_list(config, [np.array(expanded, dtype=np.int64)])
        timings, _, states = ctc.ctc_segmentation(config, scores, ground_truth)
        anchors = []
        for index, (target, position) in enumerate(zip(prepared.targets, positions)):
            time = float(timings[position])
            if not math.isfinite(time) or abs(time-round(time)) > 1e-7:
                raise ValueError('Non-integral CTC frame anchor.')
            frame = round(time)
            if not 0 <= frame < len(states) or states[frame] != target.text:
                raise ValueError('CTC frame state does not match the supplied target.')
            blank_frame, blank_score = None, None
            if index in separators:
                time = float(timings[separators[index]])
                if not math.isfinite(time) or abs(time-round(time)) > 1e-7:
                    raise ValueError('Invalid repeated-token blank anchor.')
                blank_frame = round(time)
                if not 0 <= blank_frame < len(states) or states[blank_frame] != vocabulary[0]:
                    raise ValueError('Repeated-token separator lacks blank evidence.')
                blank_score = float(scores[blank_frame, 0])
            anchors.append(TokenAnchor(index, target.token_id, frame,
                float(scores[frame, target.token_id]), blank_frame, blank_score))
        result = materialize(prepared, anchors, clock=clock, policy=policy,
            audio_start_sample=audio_start_sample, audio_num_samples=audio_num_samples,
            model_sha256=model_sha256)
        result['alignment_backend'] = 'ctc-segmentation-1.7.4'
        return result
    except (AssertionError, IndexError, ValueError):
        unresolved['reason'] = 'ctc_alignment_failed'
        return unresolved
