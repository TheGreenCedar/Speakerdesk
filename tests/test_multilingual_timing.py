"""Functional supplied-text safety; invented cells are not accuracy evidence."""
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from word_alignment import AcceptancePolicy, FrameClock, TokenAnchor, materialize, prepare_text

VOCAB = ('<s>', '<pad>', '</s>', '<unk>', ' ', 'a', 'م', 'س', 'タ', 'ネ', '日', '本', 'é')


def cells(raw, *, language):
    prepared = prepare_text(raw, VOCAB, language=language)
    anchors = [TokenAnchor(i, target.token_id, 30 + i * 3, -.1,
                29 + i * 3 if i and target.token_id == prepared.targets[i-1].token_id else None,
                -.1 if i and target.token_id == prepared.targets[i-1].token_id else None)
               for i, target in enumerate(prepared.targets)]
    return materialize(prepared, anchors, clock=FrameClock(320, 0, 'invented-functional-clock'),
                       policy=AcceptancePolicy(-1, 'invented-functional-clock'),
                       audio_start_sample=0, audio_num_samples=64000)


class MultilingualTimingTests(unittest.TestCase):
    def test_mixed_script_english_route_reuses_complete_functional_evidence(self):
        import numpy as np
        from coarse_alignment import CoarseAlignment
        from alignment_cache import AlignmentCache
        from alignment_model import CALIBRATION_ID,TOKEN_SHA256
        from alignment_provider import manifest
        from alignment_text import NORMALIZATION_POLICY_ID,TIMING_UNIT_POLICY_ID,PROVIDER_ID
        from word_alignment import MODEL_SHA256
        identity=manifest()['provider_identity_sha256']
        for raw in ('a 日本','a مُس','a ターネ'):
            with self.subTest(raw=raw):
                provider=CoarseAlignment.__new__(CoarseAlignment)
                provider.model=object();provider.actual_gpu_forwards=0;provider.vocabulary={}
                provider.conversion={'source_model_sha256':'source-peer'}
                provider.provider_identity_sha256=identity
                provider.cache=AlignmentCache((MODEL_SHA256,TOKEN_SHA256,CALIBRATION_ID,
                    'ctc_emission_cell_envelope',320,0,-20,'mlx','native',
                    NORMALIZATION_POLICY_ID,TIMING_UNIT_POLICY_ID,PROVIDER_ID,identity))
                result=cells(raw,language='en')
                result.update(frame_calibration_id=CALIBRATION_ID,score_calibration_id=CALIBRATION_ID)
                calls=[]
                def forward(*args,execution):
                    calls.append(True);execution.update(backend='mlx_metal_gpu',evaluated_and_GPU_synchronized=True)
                    return np.zeros((1,1),dtype=np.float32)
                audio=np.zeros(64000,dtype=np.float32)
                with patch('mlx_ctc_forward.forward_scores',side_effect=forward),patch('coarse_alignment.align_ctc_scores',return_value=result):
                    first=provider.align(audio,raw,start_sample=0,language='en')
                    second=provider.align(audio,raw,start_sample=0,language='en')
                self.assertEqual(len(calls),1)
                self.assertEqual(first,second)
                self.assertEqual(second['qualified_scope'],'multilingual_functional_ctc_units')
                self.assertEqual(second['timing_accuracy_calibrated_languages'],[])

    def test_null_core_keeps_qualified_cjk_peers_without_inheriting_time(self):
        from alignment_provider import manifest
        from alignment_text import FUNCTIONAL_POLICY_ID,PROVIDER_ID,raw_units,TIMING_UNIT_POLICY_ID
        from reading_turns import alignment_words,bind_words,project_turns
        result=cells('日本',language='ja')
        result.update(frame_calibration_id=FUNCTIONAL_POLICY_ID,score_calibration_id=FUNCTIONAL_POLICY_ID,
            provider_id=PROVIDER_ID,provider_identity_sha256=manifest()['provider_identity_sha256'])
        words=alignment_words(result,'日本',0,64000)
        fallback=[dict(unit,start_char=unit['start_char']+3,end_char=unit['end_char']+3,
            start_sample=None,end_sample=None) for unit in raw_units('a🙂',TIMING_UNIT_POLICY_ID)]
        row=dict(text='日本 a🙂',start_sample=0,end_sample=128000,machine_revision=1,audio_revision=1,
            text_audio_anchor={'start_sample':0,'end_sample':128000},speaker_activity={
                'audio_revision':1,'observed_end_sample':128000,'regions':[
                    {'start_sample':0,'end_sample':128000,'speakers':['speaker_0']}]})
        bind_words(row,words+fallback)
        turns=project_turns(row)
        self.assertEqual([(t['text'],t['attribution']) for t in turns],[('日本 ','single'),('a🙂','unknown')])
        self.assertIsNone(turns[-1]['start']);self.assertIsNone(turns[-1]['end'])
        bad=dict(row);bad.pop('reading_word_evidence')
        bind_words(bad,words+[dict(fallback[0],start_sample=32000,end_sample=32320)])
        self.assertNotIn('reading_word_evidence',bad)

    def test_supported_mixed_probes_do_not_claim_english_calibration(self):
        from canonical_runtime import CanonicalRuntime
        from types import SimpleNamespace
        runtime=CanonicalRuntime.__new__(CanonicalRuntime)
        result={'qualified_scope':'bounded_ami_english_coarse_envelopes','timing_accuracy_calibrated_languages':['en']}
        runtime.engine=SimpleNamespace(config={},models=SimpleNamespace(alignment_supported=lambda language:language in ('en','de'),
            align_canonical=lambda *args,**kw:result))
        passage={'language':'en','language_detection':{'reason':'mixed_languages','probes':[
            {'start_sample':0,'end_sample':16000,'language':'en','decision':{'reason':'detected'}},
            {'start_sample':16000,'end_sample':32000,'language':'de','decision':{'reason':'detected'}}]}}
        actual=runtime.align_reading({'start_sample':0,'end_sample':32000},'a a',passage)
        self.assertEqual(actual['reading_timing_qualification'],'supported_decoder_and_functional_ctc_units')
        self.assertEqual(actual['timing_accuracy_calibrated_languages'],[])
        self.assertEqual(result['timing_accuracy_calibrated_languages'],['en'])

    def test_cjk_import_projection_keeps_repeated_text_and_unknown_speaker_boundary(self):
        from alignment_provider import manifest
        from alignment_text import (PROVIDER_ID, FUNCTIONAL_POLICY_ID, NORMALIZATION_POLICY_ID,
                                    TIMING_UNIT_POLICY_ID)
        from pipeline import attach_import_reading_turns
        from reading_turns import validated_turns, alignment_words
        raw = '日本日本。'
        prepared=prepare_text(raw,VOCAB,language='ja')
        anchors=[TokenAnchor(i,target.token_id,30+i*30,-.1) for i,target in enumerate(prepared.targets)]
        result=materialize(prepared,anchors,clock=FrameClock(320,0,FUNCTIONAL_POLICY_ID),
            policy=AcceptancePolicy(-1,FUNCTIONAL_POLICY_ID),audio_start_sample=0,audio_num_samples=64000)
        result.update(provider_id=PROVIDER_ID,provider_identity_sha256=manifest()['provider_identity_sha256'],
                      audio_float32_sha256='0'*64)
        segment=dict(id='supplied-fixture',text=raw,language='ja',start=0.,end=4.,decode_context=True,
            audio_float32_sha256='0'*64,reading_alignment=result,activity_regions=[
                dict(start=0.,end=1.3,speakers=['speaker_0']),dict(start=1.3,end=4.,speakers=['speaker_1'])])
        attach_import_reading_turns(segment)
        turns=validated_turns(segment)
        self.assertEqual(''.join(turn['text'] for turn in turns),raw)
        self.assertEqual([(turn['text'],turn['attribution'],turn['speaker']) for turn in turns],
                         [('日','single','speaker_0'),('本','unknown','unassigned'),('日本。','single','speaker_1')])
        for field in ('provider_identity_sha256','normalization_policy_id','timing_unit_policy_id'):
            changed=dict(result,**{field:'wrong'})
            self.assertEqual(alignment_words(changed,raw,0,64000),[])

    def test_behavior_verification_is_model_free_and_old_receipts_cannot_activate_it(self):
        import alignment_provider as provider
        before={name for name in sys.modules if name.startswith(('mlx.','mlx_ctc_components.models'))}
        metadata=provider.manifest()
        after={name for name in sys.modules if name.startswith(('mlx.','mlx_ctc_components.models'))}
        self.assertEqual(before,after)
        receipt=dict(model_sha256=provider.MODEL_SHA256,manifest_sha256=provider.MANIFEST_SHA256,
            runtime=metadata['runtime'],provider_id=metadata['provider_identity']['provider_id'],
            provider_identity_sha256=metadata['provider_identity_sha256'],
            execution=dict(backend='mlx_metal_gpu',evaluated_and_GPU_synchronized=True))
        provider.validate_receipt(receipt,metadata)
        for field in ('provider_id','provider_identity_sha256'):
            old=dict(receipt);old.pop(field)
            with self.subTest(field=field),self.assertRaises(ValueError):provider.validate_receipt(old,metadata)
            with self.assertRaises(ValueError):provider.validate_receipt(dict(receipt,**{field:'wrong'}),metadata)
        with patch.object(provider,'digest_file',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'implementation differs'):provider.manifest()

    def test_existing_models_route_each_supported_language_when_provider_is_configured(self):
        from live_refinement import Models
        from alignment_artifact import ALIGNMENT_SPEC
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ALIGNMENT_SPEC['files']:
                (folder / name).touch()
            models = Models.__new__(Models)
            models.config = {'alignment_path': str(folder)}
            for language in ('en','de','fr','it','es','pt','el','nl','pl','vi','zh','ar','ja','ko'):
                with self.subTest(language=language):
                    self.assertTrue(models.alignment_supported(language))
            self.assertFalse(models.alignment_supported('xx'))
            models.config = {}
            self.assertFalse(models.alignment_supported('ja'))

    def test_attached_marks_keep_raw_offsets_but_never_independent_timestamps(self):
        for raw, language, mark in [('مُس','ar','ُ'), ('ターネ','ja','ー'), ('a مُس ターネ','en','ー')]:
            with self.subTest(raw=raw):
                result = cells(raw, language=language)
                self.assertEqual(result['raw_text'], raw)
                self.assertEqual(result['text_sha256'], hashlib.sha256(raw.encode()).hexdigest())
                self.assertTrue(result['complete'])
                alias = next(item for item in result['normalization_spans'] if item['raw_text'] == mark)
                self.assertEqual(raw[alias['start_char']:alias['end_char']], mark)
                self.assertIsNone(alias['start_sample'])
                self.assertIsNone(alias['end_sample'])
                self.assertFalse(alias['independent_acoustic_timestamp'])
                for character in result['characters']:
                    if character['raw_text'] == 'ー':
                        self.assertIsNone(character['start_sample'])
                        self.assertIsNone(character['end_sample'])
                        self.assertIsNotNone(character['target_cell_start_sample'])

    def test_cjk_units_are_explicit_characters_with_lossless_punctuation_and_code_switching(self):
        result = cells('「日本」、a ターネ。', language='ja')
        self.assertEqual([unit['text'] for unit in result['words']], ['「日','本」、','a','ター','ネ。'])
        self.assertEqual([unit['unit_kind'] for unit in result['words']],
                         ['ctc_character','ctc_character','whitespace_run','ctc_character','ctc_character'])
        for unit in result['words']:
            a, b = unit['start_char'], unit['end_char']
            self.assertEqual(result['raw_text'][a:b], unit['text'])
            self.assertEqual(unit['start_utf16'], len(result['raw_text'][:a].encode('utf-16-le')) // 2)
        spans = [(unit['start_sample'], unit['end_sample']) for unit in result['words']]
        self.assertTrue(all(a < b for a, b in spans))
        self.assertTrue(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])))

    def test_standalone_marks_and_unknown_letters_remain_usefully_unresolved(self):
        for raw in ('ُم','ーa','日🙂本','a\u0307'):
            with self.subTest(raw=raw):
                result = cells(raw, language='ja')
                self.assertEqual(result['reason'], 'unsupported_text')
                self.assertFalse(result['complete'])
                self.assertEqual(result['raw_text'], raw)
                self.assertTrue(all(unit['start_sample'] is None for unit in result['words']))


if __name__ == '__main__':
    unittest.main()
