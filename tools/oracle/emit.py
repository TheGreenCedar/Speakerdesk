#!/usr/bin/env python3
"""Dev-only golden emitter for the v0.6.6 Python modules.

Inserts ``<repo>/speakerdesk`` on ``sys.path`` and writes JSON under
``tests/goldens/``. Imports no Rust. Not used by the build, CI, or packaging.
"""
import ast
import copy
import hashlib
import json
import sqlite3
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'speakerdesk'))

from alignment_text import (  # noqa: E402
    FUNCTIONAL_POLICY_ID,
    LEGACY_UNIT_POLICY_ID,
    NORMALIZATION_POLICY_ID,
    PROVIDER_ID,
    TIMING_UNIT_POLICY_ID,
    UPSTREAM_NORMALIZATION_REVISION,
    raw_units,
)
from job_store import JobStore  # noqa: E402
from language_detection import LanguagePolicy  # noqa: E402
from people import PeopleStore, introduction_suggestions  # noqa: E402
from pipeline import TURN_GAP_SECONDS, crop_context, decode_regions  # noqa: E402
from reading_turns import (  # noqa: E402
    CALIBRATION,
    MODEL,
    bind_words,
    project_turns,
    validated_turns,
)
from transcript import export, validate  # noqa: E402
from word_alignment import parse_vocabulary, prepare_text  # noqa: E402

GOLDENS = REPO / 'tests' / 'goldens'


def write_json(relative, payload):
    path = REPO / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    alternate = path.with_name(path.name.replace('.skipped.json', '.json') if path.name.endswith('.skipped.json')
                               else path.name.replace('.json', '.skipped.json'))
    if alternate != path and alternate.exists() and alternate.suffixes == ['.json']:
        # Drop a stale sibling only when switching between golden and skipped.
        if path.name.endswith('.skipped.json') or alternate.name.endswith('.skipped.json'):
            alternate.unlink()
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(text, encoding='utf-8')
    return relative, len(text.encode('utf-8'))


def record(name, function_name, producer):
    relative = f'tests/goldens/{name}.json'
    skipped = f'tests/goldens/{name}.skipped.json'
    try:
        payload = producer()
    except Exception as exc:
        traceback.print_exc()
        file_name, size = write_json(skipped, {
            'error': f'{type(exc).__name__}: {exc}',
            'exception_text': str(exc),
        })
        ready = REPO / relative
        if ready.exists():
            ready.unlink()
        return {'file': file_name, 'function': function_name, 'byte_size': size, 'skipped': True}
    stale = REPO / skipped
    if stale.exists():
        stale.unlink()
    file_name, size = write_json(relative, payload)
    return {'file': file_name, 'function': function_name, 'byte_size': size}


def emit_decode_regions():
    cases = []

    def add(name, turns, **kwargs):
        regions = decode_regions(turns, **kwargs) if kwargs else decode_regions(turns)
        cases.append({
            'name': name,
            'turns': turns,
            'arguments': kwargs,
            'regions': regions,
        })

    add('overlap_gap', [
        {'start': 0.0, 'end': 3.0, 'speaker': 'speaker_0'},
        {'start': 2.5, 'end': 6.0, 'speaker': 'speaker_1'},
        {'start': 6.0, 'end': 9.0, 'speaker': 'speaker_2'},
    ])
    end = 3.5
    add('same_speaker_brief_gap', [
        {'start': 0.0, 'end': 1.0, 'speaker': 'speaker_0'},
        {'start': 1.25, 'end': 2.25, 'speaker': 'speaker_0'},
        {'start': 2.5, 'end': end, 'speaker': 'speaker_0'},
    ], coverage=(0.0, end))
    add('same_speaker_long_pause', [
        {'start': 0.0, 'end': 1.0, 'speaker': 'speaker_0'},
        {'start': 2.0, 'end': 3.0, 'speaker': 'speaker_0'},
    ], coverage=(0.0, 3.0))
    add('region_longer_than_default_max_seconds', [
        {'start': 0.0, 'end': 30.0, 'speaker': 'speaker_0'},
    ])
    add('region_longer_than_18_seconds', [
        {'start': 0.0, 'end': 20.0, 'speaker': 'speaker_0'},
    ], max_seconds=18)
    return {
        'function': 'speakerdesk.pipeline.decode_regions',
        'turn_gap_seconds': TURN_GAP_SECONDS,
        'default_max_seconds': 24.5,
        'cases': cases,
    }


def emit_crop_context():
    turns = [
        {'start': 0.1, 'end': 1.0, 'speaker': 'speaker_0'},
        {'start': 4.0, 'end': 5.0, 'speaker': 'speaker_1'},
    ]
    duration = 12.0
    source = decode_regions(turns)
    cases = []
    for padding in (0.22, 0):
        chunks = copy.deepcopy(source)
        returned = crop_context(chunks, duration, padding=padding)
        cases.append({
            'padding': padding,
            'duration': duration,
            'turns': turns,
            'chunks_before': source,
            'return_value': returned,
            'chunks_after': chunks,
        })
    return {
        'function': 'speakerdesk.pipeline.crop_context',
        'cases': cases,
    }


def emit_raw_units():
    samples = [
        {'name': 'english', 'text': 'Hello, world.'},
        {'name': 'japanese', 'text': 'こんにちは、世界。'},
        {'name': 'japanese_long_vowel', 'text': 'コーヒー'},
        {'name': 'arabic', 'text': 'مرحبا بالعالم'},
        {'name': 'arabic_marks', 'text': 'مَرْحَبًا'},
        {'name': 'mixed', 'text': 'Hello 東京 مرحبا'},
    ]
    cases = []
    for sample in samples:
        text = sample['text']
        default_units = raw_units(text)
        legacy_units = raw_units(text, LEGACY_UNIT_POLICY_ID)
        timing_units = raw_units(text, TIMING_UNIT_POLICY_ID)
        cases.append({
            **sample,
            'default_policy_id': LEGACY_UNIT_POLICY_ID,
            'default_units': default_units,
            'explicit_legacy_units': legacy_units,
            'default_matches_legacy_policy': default_units == legacy_units,
            'timing_policy_id': TIMING_UNIT_POLICY_ID,
            'timing_units': timing_units,
        })
    normalization_error = None
    try:
        raw_units(samples[0]['text'], NORMALIZATION_POLICY_ID)
    except ValueError as exc:
        normalization_error = str(exc)
    if normalization_error is None:
        raise RuntimeError('NORMALIZATION_POLICY_ID was accepted by raw_units.')
    return {
        'function': 'speakerdesk.alignment_text.raw_units',
        'policy_ids': {
            'LEGACY_UNIT_POLICY_ID': LEGACY_UNIT_POLICY_ID,
            'TIMING_UNIT_POLICY_ID': TIMING_UNIT_POLICY_ID,
            'NORMALIZATION_POLICY_ID': NORMALIZATION_POLICY_ID,
            'FUNCTIONAL_POLICY_ID': FUNCTIONAL_POLICY_ID,
            'PROVIDER_ID': PROVIDER_ID,
            'UPSTREAM_NORMALIZATION_REVISION': UPSTREAM_NORMALIZATION_REVISION,
        },
        'normalization_policy_accepted': False,
        'normalization_policy_error': normalization_error,
        'cases': cases,
    }


def emit_prepare_text():
    characters = ['h', 'e', 'l', 'o', 'w', 'r', 'd', 'م', 'ر', 'ح', 'ب', 'ا', 'ع', 'コ', 'ヒ']
    tokens = ['<s>', '<pad>', '</s>', '<unk>', ' '] + characters
    if len(tokens) != len(set(tokens)):
        raise RuntimeError('Synthetic vocabulary tokens must be unique.')
    contents = '\n'.join(f'{token} {index}' for index, token in enumerate(tokens)) + '\n'
    vocabulary = parse_vocabulary(contents)
    samples = [
        {'name': 'english_default_language', 'text': 'Hello, world.', 'language': None},
        {'name': 'english', 'text': 'Hello, world.', 'language': 'en'},
        {'name': 'japanese_long_vowel', 'text': 'コーヒー', 'language': 'ja'},
        {'name': 'arabic_marks', 'text': 'مَرْحَبًا', 'language': 'ar'},
        {'name': 'unsupported_scalar', 'text': 'Hello Ω', 'language': 'en'},
    ]
    prepared_cases = []
    for sample in samples:
        language = sample['language']
        if language is None:
            prepared = prepare_text(sample['text'], vocabulary)
        else:
            prepared = prepare_text(sample['text'], vocabulary, language=language)
        prepared_cases.append({
            'name': sample['name'],
            'text': sample['text'],
            'language': language,
            'prepared': asdict(prepared) if is_dataclass(prepared) else prepared,
        })
    return {
        'function': 'speakerdesk.word_alignment.prepare_text',
        'vocabulary_source': 'in-memory string passed to parse_vocabulary; no vocabulary file and no model file',
        'vocabulary': list(vocabulary),
        'cases': prepared_cases,
    }


def emit_exports():
    document = {
        'speakers': {'speaker_0': ' Ada Lovelace '},
        'segments': [
            {'id': 'later', 'start': 4.0, 'end': 5.5, 'speaker': 'speaker_0', 'text': 'Second line.'},
            {'id': 'earlier', 'start': 0.5, 'end': 2.25, 'speaker': 'speaker_0', 'text': 'Hello, oracle.'},
        ],
    }
    duration = 10.0
    validated = validate(copy.deepcopy(document), duration)
    exports = {}
    for kind in ('txt', 'srt', 'vtt', 'json'):
        body, media_type = export(validated, kind)
        exports[kind] = {'body': body, 'media_type': media_type}
    bad = {
        'speakers': {'speaker_0': 'Ada Lovelace'},
        'segments': [{
            'id': 'late', 'start': 0.0, 'end': 99.0, 'speaker': 'speaker_0', 'text': 'Too late.',
        }],
    }
    error = None
    try:
        validate(bad, duration)
    except ValueError as exc:
        error = str(exc)
    if error is None:
        raise RuntimeError('validate() accepted the bad document.')
    return {
        'function': 'speakerdesk.transcript.validate; speakerdesk.transcript.export',
        'duration': duration,
        'document': document,
        'validated': validated,
        'exports': exports,
        'validate_error': {'error': error, 'document': bad, 'duration': duration},
    }


def emit_reading_turns():
    """Smallest synthetic row that project_turns and validated_turns both accept.

    Words are display units from the legacy whitespace policy. Sample spans sit
    inside one speaker region and inside the legacy uncertainty margin. The
    receipt hashes are the hashes validated_turns recomputes; the turn text
    itself comes from project_turns.
    """
    text = 'go next'
    row = {
        'id': 'seg-1',
        'canonical_utterance_id': 'seg-1',
        'machine_revision': 1,
        'canonical_machine_revision': 1,
        'audio_revision': 1,
        'language_epoch': 0,
        'start_sample': 0,
        'end_sample': 32000,
        'start': 0.0,
        'end': 2.0,
        'text': text,
        'text_audio_anchor': {'start_sample': 0, 'end_sample': 32000},
        'speaker': 'multiple_speakers',
        'speaker_candidates': ['speaker_0', 'speaker_1'],
        'speaker_activity': {
            'audio_revision': 1,
            'regions': [
                {'start_sample': 0, 'end_sample': 16000, 'speakers': ['speaker_0']},
                {'start_sample': 16000, 'end_sample': 32000, 'speakers': ['speaker_1']},
            ],
        },
    }
    words = [
        {'text': 'go', 'start_char': 0, 'end_char': 2, 'start_sample': 4000, 'end_sample': 8000},
        {'text': 'next', 'start_char': 3, 'end_char': 7, 'start_sample': 20000, 'end_sample': 24000},
    ]
    bind_words(row, words)
    if 'reading_word_evidence' not in row:
        raise RuntimeError('bind_words did not attach reading_word_evidence.')
    turns = project_turns(row)
    if not turns:
        raise RuntimeError('project_turns returned no turns for the synthetic row.')
    segment = copy.deepcopy(row)
    segment['reading_turns'] = turns
    segment['reading_turn_provenance'] = {
        'utterance_id': segment['id'],
        'machine_revision': segment['machine_revision'],
        'canonical_machine_revision': segment['canonical_machine_revision'],
        'audio_revision': segment['audio_revision'],
        'language_epoch': segment['language_epoch'],
        'text_sha256': hashlib.sha256(segment['text'].encode()).hexdigest(),
        'audio_anchor': segment['text_audio_anchor'],
        'activity_sha256': hashlib.sha256(json.dumps(
            segment['speaker_activity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'turns_sha256': hashlib.sha256(json.dumps(
            turns, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
        'calibration_id': CALIBRATION,
        'model_sha256': MODEL,
    }
    checked = validated_turns(segment)
    if checked != turns:
        raise RuntimeError('validated_turns rejected the receipt built from project_turns.')
    return {
        'function': 'speakerdesk.reading_turns.project_turns; speakerdesk.reading_turns.validated_turns',
        'model_sha256': MODEL,
        'calibration_id': CALIBRATION,
        'synthetic_words': words,
        'row_after_bind_words': row,
        'project_turns': turns,
        'validated_turns': checked,
    }


def emit_language_policy():
    low_margin_winner = .55
    low_margin_second = low_margin_winner - .20
    low_margin_rest = 1 - low_margin_winner - low_margin_second
    cases = [
        {
            'name': 'supported_above_both_thresholds',
            'speaker': 'speaker_0',
            'probabilities': {'en': .95, 'de': .03, 'fr': .02},
        },
        {
            'name': 'supported_probability_below_0_90',
            'speaker': 'speaker_0',
            'probabilities': {'en': .89, 'fr': .11},
        },
        {
            'name': 'supported_margin_below_0_20',
            'speaker': 'speaker_0',
            'probabilities': {'en': .52, 'fr': .48},
        },
        {
            'name': 'supported_probability_at_0_90',
            'speaker': 'speaker_0',
            'probabilities': {'en': .90, 'fr': .10},
        },
        {
            'name': 'supported_margin_at_0_20_probability_below_0_90',
            'speaker': 'speaker_0',
            'probabilities': {'en': low_margin_winner, 'fr': low_margin_second, 'de': low_margin_rest},
        },
        {
            'name': 'unsupported_above_both_thresholds',
            'speaker': 'speaker_1',
            'probabilities': {'ru': .96, 'en': .04},
        },
        {
            'name': 'unsupported_below_thresholds',
            'speaker': 'speaker_1',
            'probabilities': {'ru': .55, 'en': .45},
        },
    ]
    results = []
    for case in cases:
        policy = LanguagePolicy()
        decision = policy.decide(dict(case['probabilities']), case['speaker'])
        results.append({
            'name': case['name'],
            'speaker': case['speaker'],
            'probabilities': case['probabilities'],
            'decision': decision,
            'current_after': policy.current,
        })
    return {
        'function': 'speakerdesk.language_detection.LanguagePolicy.decide',
        'detector': 'LanguagePolicy.decide accepts a probability mapping directly. No detector object is required.',
        'cases': results,
    }


def _job(job_id, kind, status, name, message, text):
    return {
        'id': job_id,
        'name': name,
        'kind': kind,
        'source_file': f'{kind}-source',
        'created': 1700000000.0,
        'status': status,
        'message': message,
        'language': 'en',
        'duration': 12.0,
        'revision': 1,
        'document': {
            'schema_version': 1,
            'speakers': {'speaker_0': 'Speaker 1'},
            'segments': [{
                'id': 'seg-1',
                'speaker': 'speaker_0',
                'start': 0.0,
                'end': 1.5,
                'text': text,
            }],
            'provenance': {'kind': 'local_inference'},
        },
    }


def _assigned_value(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f'{name} was not found in {path}.')


def emit_legacy_jobs():
    live = _assigned_value(REPO / 'speakerdesk' / 'live_meeting.py', 'LIVE')
    if live != ('starting', 'recording', 'paused', 'finishing'):
        raise RuntimeError(f'Unexpected live_meeting.LIVE: {live!r}')
    active = ('preparing', 'queued', 'processing') + live
    with tempfile.TemporaryDirectory(prefix='speakerdesk-oracle-') as folder:
        database = Path(folder) / 'jobs.sqlite'

        @contextmanager
        def db():
            conn = sqlite3.connect(database, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

        store = JobStore(db)
        store.initialize()
        people = PeopleStore(db)
        store.put(_job('job-import', 'import', 'ready', 'Imported notes', 'Saved.',
                       'Imported hello.'))
        store.put(_job('job-meeting', 'meeting', 'ready', 'Weekly sync', 'Saved.',
                       'Meeting hello.'))
        created_person = people.create('Priya Natarajan')
        store.put(_job('job-processing', 'meeting', 'processing', 'Live notes',
                       'Starting local inference.', 'Still working.'))
        before = store.get('job-processing')
        recover_return = store.recover(active)
        after = store.get('job-processing')
        if after.get('status') != 'failed':
            raise RuntimeError('recover() did not mark the processing job failed.')
        with db() as conn:
            schema = [dict(row) for row in conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name")]
            tables = {}
            for entry in schema:
                if entry['type'] != 'table':
                    continue
                name = entry['name']
                if not name.replace('_', '').isalnum():
                    raise RuntimeError(f'Unexpected table name {name!r}.')
                tables[name] = [dict(row) for row in conn.execute(
                    f'SELECT * FROM {name} ORDER BY rowid')]
        blob = database.read_bytes()
    home = str(Path.home())
    if home.encode() in blob or home in json.dumps(tables):
        raise RuntimeError('Legacy dump contains a home directory path.')
    return {
        'function': 'speakerdesk.job_store.JobStore; speakerdesk.people.PeopleStore',
        'sqlite_sha256': hashlib.sha256(blob).hexdigest(),
        'sqlite_bytes': len(blob),
        'schema': schema,
        'tables': tables,
        'people_create': created_person,
        'recover': {
            'active': list(active),
            'return_value': recover_return,
            'before': before,
            'after': after,
        },
    }


def emit_introductions():
    def job(text, **segment):
        body = {
            'id': 'intro',
            'speaker': 'speaker_0',
            'start': 0.0,
            'end': 3.0,
            'text': text,
            'review': False,
            'finalized': True,
            'speaker_candidates': ['speaker_0'],
        }
        body.update(segment)
        return {
            'document': {
                'schema_version': 1,
                'speakers': {'speaker_0': 'Speaker 1'},
                'segments': [body],
                'provenance': {'kind': 'local_inference'},
            },
        }

    jobs = [
        {'name': 'clear_introduction', 'job': job("I'm Priya. Welcome to the meeting.")},
        {'name': 'addressed_name_not_an_introduction', 'job': job('Priya, what do you think?')},
        {'name': 'lowercase_name_not_suggested', 'job': job("I'm priya.")},
    ]
    return {
        'function': 'speakerdesk.people.introduction_suggestions',
        'cases': [{**item, 'suggestions': introduction_suggestions(item['job'])} for item in jobs],
    }


def manifest_text(entries, self_size):
    document = {
        'description': 'Committed goldens produced by tools/oracle/emit.py from the v0.6.6 Python modules. The harness imports no Rust.',
        'files': list(entries) + [{
            'file': 'tests/goldens/manifest.json',
            'function': 'tools/oracle/emit.py',
            'byte_size': self_size,
        }],
    }
    return json.dumps(document, ensure_ascii=False, indent=2) + '\n'


def main():
    sections = [
        ('decode_regions', 'speakerdesk.pipeline.decode_regions', emit_decode_regions),
        ('crop_context', 'speakerdesk.pipeline.crop_context', emit_crop_context),
        ('raw_units', 'speakerdesk.alignment_text.raw_units', emit_raw_units),
        ('prepare_text', 'speakerdesk.word_alignment.prepare_text', emit_prepare_text),
        ('exports', 'speakerdesk.transcript.validate; speakerdesk.transcript.export', emit_exports),
        ('reading_turns', 'speakerdesk.reading_turns.project_turns; speakerdesk.reading_turns.validated_turns', emit_reading_turns),
        ('language_policy', 'speakerdesk.language_detection.LanguagePolicy.decide', emit_language_policy),
        ('introductions', 'speakerdesk.people.introduction_suggestions', emit_introductions),
        ('legacy/jobs.sqlite', 'speakerdesk.job_store.JobStore; speakerdesk.people.PeopleStore', emit_legacy_jobs),
    ]
    entries = [record(*section) for section in sections]
    size = 0
    text = ''
    for _ in range(8):
        text = manifest_text(entries, size)
        updated = len(text.encode('utf-8'))
        if updated == size:
            break
        size = updated
    else:
        raise RuntimeError('manifest.json byte size did not converge.')
    manifest_path = GOLDENS / 'manifest.json'
    manifest_path.write_text(text, encoding='utf-8')
    home = str(Path.home())
    for path in GOLDENS.rglob('*.json'):
        body = path.read_text(encoding='utf-8')
        if home and home in body:
            raise RuntimeError(f'{path} contains a home directory path.')
        json.loads(body)
    print(f'wrote {manifest_path.relative_to(REPO)} ({size} bytes)')
    for entry in entries:
        print(f"{entry['file']} {entry['byte_size']} skipped={entry.get('skipped', False)}")


if __name__ == '__main__':
    main()
