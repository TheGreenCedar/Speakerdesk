"""CPU core-seam controls and optional replay of retained real model receipts.

No model/audio replay. The external case uses only the completed public crop's
immutable provider output; synthetic cells are explicitly separate controls.
"""
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from authoritative_tail import assemble, reading_words, requests
from reading_turns import bind_words, project_turns, validated_turns
from app import create_app
from test_authoritative_tail import part, VOCAB
from test_canonical_assembly import book_for
import test_canonical_runtime as runtime_fixtures


class ReadingCoreSeamTests(unittest.TestCase):
    def decode_retained_parts(self, book, parts, vocabulary, regions, language='en'):
        case = runtime_fixtures.CanonicalTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        self.runtime_fixture = case
        case.engine.config['language'] = language
        case.engine.timeline[0]['language'] = language
        runtime = case.engine.canonical
        runtime.book = book
        row = book.rows[book.order[0]]
        book.attach_activity(row['id'], row['audio_revision'], regions, observed_end_sample=row['end_sample'])
        case.peer.align_canonical = lambda *args, **kwargs: self.fail('No provider/model call is permitted')
        case.peer.alignment_supported = lambda language: language == 'en'
        case.peer.coarse_aligner = SimpleNamespace(vocabulary=vocabulary)
        runtime.decode_long_parts = lambda *args, **kwargs: copy.deepcopy(parts)
        runtime.refinement_request = {'operation_id': 'retained-receipts-only'}
        self.assertTrue(runtime.decode(row, 'refined'))
        return row, runtime.project(row)

    def synthetic(self, regions=None):
        book, supplied = book_for(seconds=28)
        req = requests(supplied)
        parts = [part(req[0], 'Head  go tail', (2, 13.8, 15.5)),
                 part(req[1], 'Different Go after', (11.9, 13.8, 16))]
        if regions is None:
            regions = [(0, 17, ['speaker_0']), (17, 28, ['speaker_1'])]
        regions = [dict(start_sample=round(a * 16000), end_sample=round(b * 16000), speakers=names)
                   for a, b, names in regions]
        return self.decode_retained_parts(book, parts, VOCAB, regions)

    def test_artificial_core_seam_retains_single_owner_without_claiming_core_timing(self):
        row, candidate = self.synthetic()
        self.assertEqual(candidate['text'], 'Head  go after')
        self.assertEqual([(turn['speaker'], turn['attribution']) for turn in candidate['reading_turns']],
                         [('speaker_0', 'single')])
        seam = row['assembly_provenance']['words'][1]
        self.assertEqual(seam['core_ownership'], 'unknown')
        self.assertIsNone(seam['start_sample'])
        self.assertIsNone(seam['end_sample'])
        self.assertFalse(row['assembly_provenance']['alignment_complete'])

    def test_real_speaker_transition_at_core_seam_remains_unknown(self):
        row, candidate = self.synthetic([(0, 14, ['speaker_0']), (14, 28, ['speaker_1'])])
        self.assertEqual([(turn['speaker'], turn['attribution']) for turn in candidate['reading_turns']],
                         [('speaker_0', 'single'), ('unassigned', 'unknown'), ('speaker_1', 'single')])
        self.assertEqual(candidate['reading_turns'][1]['text'], 'go ')

    def test_actual_overlap_at_core_seam_remains_local_overlap(self):
        row, candidate = self.synthetic([(0, 10, ['speaker_0']), (10, 17, ['speaker_0', 'speaker_1']),
                                         (17, 28, ['speaker_1'])])
        self.assertEqual([turn['attribution'] for turn in candidate['reading_turns']], ['single', 'overlap'])
        self.assertEqual(candidate['reading_turns'][1]['speaker'], 'overlap_speaker_0_speaker_1')
        self.assertEqual(candidate['reading_turns'][1]['text'], 'go after')

    def test_emission_in_gap_at_core_seam_remains_unknown(self):
        row, candidate = self.synthetic([(0, 13.7, ['speaker_0']), (13.7, 14.5, []),
                                         (14.5, 28, ['speaker_1'])])
        self.assertEqual(candidate['reading_turns'][1]['attribution'], 'unknown')
        self.assertEqual(candidate['reading_turns'][1]['text'], 'go ')

    def test_unobserved_uncertainty_window_at_core_seam_remains_unknown(self):
        row, candidate = self.synthetic()
        row['speaker_activity']['observed_end_sample'] = round(13.9 * 16000)
        turns = project_turns(row)
        self.assertEqual(turns[-1]['attribution'], 'unknown')
        self.assertEqual(turns[-1]['text'], 'go after')

    def test_conflicting_source_envelopes_cannot_replace_retained_word_timings(self):
        for start, end in ((150, 250), (250, 350)):
            words = [{'text': 'before', 'start_sample': 100, 'end_sample': 200},
                     {'text': 'seam', 'start_sample': None, 'end_sample': None, 'core_ownership': 'unknown',
                      'source_emission_envelope': {'start_sample': start, 'end_sample': end}},
                     {'text': 'after', 'start_sample': 300, 'end_sample': 400}]
            original = copy.deepcopy(words)
            actual = reading_words(words)
            self.assertIsNone(actual[1]['start_sample'])
            self.assertEqual([(word['start_sample'], word['end_sample']) for word in actual],
                             [(100, 200), (None, None), (300, 400)])
            self.assertEqual(words, original)

    def test_missing_provider_timing_has_no_source_envelope_to_recover(self):
        book, supplied = book_for(seconds=7)
        req = requests(supplied)
        result = assemble(supplied, req, [{'request': req[0], 'text': 'Untimed raw words',
                                          'complete': True, 'alignment': None}])
        self.assertTrue(result['complete'])
        words = reading_words(result['words'])
        self.assertTrue(all(word['start_sample'] is None for word in words))
        self.assertTrue(all('source_emission_envelope' not in word for word in words))

    def test_human_protected_text_cannot_bind_restored_envelopes(self):
        row, candidate = self.synthetic()
        row.update(text='Human correction', protected_fields=['text'])
        row.pop('reading_word_evidence', None)
        bind_words(row, reading_words(row['assembly_provenance']['words']))
        self.assertNotIn('reading_word_evidence', row)
        self.assertEqual(project_turns(row), [])

    @unittest.skipUnless(os.environ.get('PIERS_RETAINED_EVIDENCE'), 'Explicit retained public receipts required')
    def test_retained_piers_model_receipts_recover_only_six_artificial_core_seam_words(self):
        evidence = Path(os.environ['PIERS_RETAINED_EVIDENCE'])
        report = json.loads((evidence / 'report.json').read_text())
        self.assertTrue(report['complete'])
        self.assertFalse(report['private_audio'])
        self.assertEqual(report['source_commit'], 'fedd5bb510b229173ae9af57bb4364a6b7862bc8')
        job = json.loads((evidence / 'final-job.json').read_text())
        saved = job['document']['segments'][0]
        folder = evidence / 'home' / job['id']
        events = [json.loads(line) for path in folder.glob('utterance-versions-*.jsonl')
                  for line in path.read_text().splitlines()]
        event = next(event for event in reversed(events)
                     if event['type'] == 'authoritative_machine_version' and event['stage'] == 'refined')
        parts = []
        for retained in event['parts']:
            reference = retained['authority_reference']
            path = Path(reference['path'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), reference['sha256'])
            selected = json.loads(path.read_text())['part']
            selected['request'] = retained['request']
            parts.append(selected)
        # Reconstruct only the token IDs needed by these supplied receipts;
        # this does not load or claim to recreate the full model vocabulary.
        tokens = {}
        for selected in parts:
            for cell in selected['alignment']['characters']:
                self.assertEqual(tokens.setdefault(cell['token_id'], cell['text']), cell['text'])
        vocabulary = [f'<unused:{index}>' for index in range(max(tokens) + 1)]
        vocabulary[:4] = ['<s>', '<pad>', '</s>', '<unk>']
        for token_id, token in tokens.items():
            vocabulary[token_id] = token
        book, supplied = book_for(seconds=30)
        row = book.rows[supplied['id']]
        book.rows.clear()
        row.update(id=saved['id'], audio_revision=event['audio_revision'], machine_revision=event['base_revision'],
                   language=saved['language'])
        book.rows[row['id']] = row
        book.order = [row['id']]
        row, candidate = self.decode_retained_parts(book, parts, vocabulary, saved['speaker_activity']['regions'],
                                                   language=report['language_mode'])
        self.assertEqual(candidate['text'], saved['text'])
        self.assertEqual(candidate['assembly_provenance']['rollover_receipts'], saved['assembly_provenance']['rollover_receipts'])
        self.assertEqual([(word['start_sample'], word['end_sample'], word['core_ownership'])
                          for word in candidate['assembly_provenance']['words']],
                         [(word['start_sample'], word['end_sample'], word['core_ownership'])
                          for word in saved['assembly_provenance']['words']])
        counts = Counter()
        for turn in candidate['reading_turns']:
            counts[turn['speaker']] += len(turn['text'].split())
        self.assertEqual(counts, {'speaker_0': 11, 'speaker_1': 77,
                                  'overlap_speaker_0_speaker_1': 1, 'unassigned': 6})
        self.assertEqual(''.join(turn['text'] for turn in candidate['reading_turns']), saved['text'])
        # Persist the real receipt-derived publication through the actual host
        # API, then render it with the already-existing CPU DOM fixture.
        case_folder = self.runtime_fixture.root
        app = create_app(case_folder / 'saved-home')
        manager = app.extensions['speakerdesk']['meetings']
        manager.duration = 30
        jid = job['id']
        controlled_job = {'id': jid, 'created': 1, 'status': 'ready', 'kind': 'meeting',
                          'name': 'Retained public crop CPU projection', 'language': report['language_mode'],
                          'duration': 30, 'revision': 0, 'canonical_utterances': True,
                          'document': {'speakers': {'speaker_0': 'Speaker 1', 'speaker_1': 'Speaker 2'},
                                       'segments': [], 'provenance': {}, 'warnings': []}}
        try:
            manager.refinement.initialize(controlled_job)
            manager.put(controlled_job)
            manager.refinement.canonical(jid, {'type': 'canonical_revision', 'candidate': candidate, 'fast_sequence': 1})
            response = app.test_client().get('/api/jobs/' + jid)
            self.assertEqual(response.status_code, 200)
            persisted = response.json['document']['segments'][0]
            self.assertEqual(validated_turns(persisted), candidate['reading_turns'])
            payload = case_folder / 'public-candidate-job.json'
            payload.write_text(json.dumps(response.json))
            node = """
const assert=require('node:assert/strict'),fs=require('node:fs');
const [root,payload]=process.argv.slice(1);
const {frontend}=require(root+'/tests/support/frontend_dom.cjs');
const job=JSON.parse(fs.readFileSync(payload)),f=frontend(root);f.seed(0);f.context.actualJob=job;
f.run('selected=structuredClone(actualJob);doc=structuredClone(selected.document);window.original=JSON.stringify(doc);renderSegments()');
const host=f.document.getElementById('segments'),labels=host.querySelectorAll('.reading-turn-speaker').map(p=>p.textContent);
assert(labels.includes('Speaker 1'));assert(labels.includes('Speaker 2'));assert(labels.includes('Overlapping speakers'));
assert.equal(host.querySelectorAll('.reading-turn-words').map(p=>p.textContent).join(''),job.document.segments[0].text);
assert.equal(host.querySelectorAll('.reading-turn-unassigned').length,0,'artificial core seams must not add uncertain inline words');
assert.equal(f.run('JSON.stringify(doc)===original'),true);
"""
            subprocess.run(['node', '-e', node, str(Path(__file__).resolve().parents[1]), str(payload)],
                           check=True, capture_output=True, text=True, timeout=20)
            if os.environ.get('PIERS_CANDIDATE_JSON'):
                Path(os.environ['PIERS_CANDIDATE_JSON']).write_text(json.dumps(response.json, indent=2) + '\n')
        finally:
            manager.close()
            app.extensions['speakerdesk']['executor'].shutdown(wait=True, cancel_futures=True)
        print(json.dumps({'retained_real_receipt_word_counts': dict(counts), 'raw_text_unchanged': True,
                          'canonical_core_timings_unchanged': True, 'rollover_receipts_unchanged': True}))


if __name__ == '__main__':
    unittest.main()
