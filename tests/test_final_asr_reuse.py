"""Real canonical refinement with synthetic model peers; no neural inference."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import wave

import numpy as np
from pcm_peer import SpeechEvidencePeer, varying_pcm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from final_asr_reuse import FinalAsrReuse, digest
from language_probe_cache import LanguageProbeCache
from live_refinement import Models
from rolling_refinement import segment_version
from test_authoritative_tail import book_for, part, VOCAB
import test_canonical_runtime as fixtures

IDENTITY={'device':'gpu','weights_sha256':'synthetic_peer_only','runtime':'synthetic_test'}


class RoutingPeer(fixtures.Peer):
    transcribe=Models.transcribe
    set_language_context=Models.set_language_context
    set_language_epoch=Models.set_language_epoch
    begin_asr_request=Models.begin_asr_request
    finish_asr_request=Models.finish_asr_request
    def __init__(self,mode):
        super().__init__();self.config={};self.check_memory=lambda:None
        self.mx=SimpleNamespace(clear_cache=lambda:None,
            default_device=lambda:SimpleNamespace(type='synthetic_gpu'),gpu='synthetic_gpu');self.detector=None
        self.language_probe_cache=LanguageProbeCache();self.language_epoch=0
        self.language_context=None;self.transcription_start_sample=0;self.cohere_calls=0
        self.coarse_aligner=SimpleNamespace(vocabulary=VOCAB)
        self.speech_historical=None;self.historical_calls=0;self.align_calls=0;self.batch_calls=0
        self.final_asr_reuse=FinalAsrReuse(IDENTITY,mode=mode,qualification=digest(IDENTITY))
        self.asr=SimpleNamespace(config={'fixture':1},model=SimpleNamespace(training=False),
            feature_extractor=SimpleNamespace(dither=0),tokenizer=SimpleNamespace())
        def recognize(audio,**kw):
            raw='Head go tail' if self.transcription_start_sample==0 else 'Different Go after'
            return SimpleNamespace(text=raw,tokens=[1,2,3],language=kw['language'])
        self.asr.transcribe=Mock(side_effect=recognize)
    def begin_refinement(self,pcm,start):
        self.historical_calls+=1;self.speech_historical=SpeechEvidencePeer()
    def end_refinement(self):self.speech_historical=None
    def alignment_supported(self,language):return language=='en'
    def align_canonical(self,request,text,language='en'):
        self.align_calls+=1
        times=(2,13.8,15.5) if request['start_sample']==0 else (11.9,13.8,16)
        return part(request,text,times)['alignment']
    def batch_turns(self,pcm):self.batch_calls+=1;return super().batch_turns(pcm)


class FinalAsrTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CanonicalTests();self.fixture.setUp()
        self.engine=self.fixture.engine;self.peer=RoutingPeer('reuse');self.engine.models=self.peer
        self.peer.speech_live.evidence=SpeechEvidencePeer()
        with wave.open(str(self.fixture.path),'wb') as audio:
            audio.setparams((1,2,16000,0,'NONE','none'));audio.writeframes(b'\x01\x00\xff\xff'*(60*16000//2))
        book,row=book_for(seconds=28);self.engine.canonical.book=book
        self.row=book.rows[row['id']]
        book.attach_activity(row['id'],row['audio_revision'],[
            {'start_sample':0,'end_sample':14*16000,'speakers':['speaker_0']},
            {'start_sample':14*16000,'end_sample':28*16000,'speakers':['speaker_1']}])
        self.engine.canonical.decode(self.row,'live')
        self.assertIsNone(self.row.get('canonical_unresolved'))
        self.assertEqual(self.peer.cohere_calls,2)
    def tearDown(self):self.fixture.tearDown()
    def request(self,operation='final',purpose='automatic_final'):
        row=self.engine.canonical.project(self.row)
        return {'operation_id':operation,'language_epoch':0,'language':'en','canonical':row,
            'asr_purpose':purpose,'expected':{row['id']:segment_version(row)},
            'window':{'id':row['id'],'start_sample':0,'end_sample':28*16000},'references':[]}
    def decisions(self):
        import json
        rows=[json.loads(line) for line in self.engine.canonical.archive.path.read_text().splitlines()]
        return [d for row in rows if row['type']=='final_asr_reuse_decisions' and row['stage']=='refined'
                for d in row['decisions']]
    def test_selected_raw_results_skip_only_asr_and_keep_final_vad_alignment_assembly(self):
        live=copy.deepcopy(self.engine.canonical.project(self.row))
        aligned=self.peer.align_calls
        result=self.engine.canonical.refine(self.request())
        self.assertEqual(self.peer.cohere_calls,2)
        self.assertEqual(self.peer.asr.transcribe.call_count,2)
        self.assertEqual(self.peer.historical_calls,2)
        self.assertEqual(self.peer.align_calls-aligned,2)
        candidate=result['canonical_candidate']
        self.assertEqual(candidate['text'],live['text'])
        self.assertEqual(candidate['text_audio_anchor'],live['text_audio_anchor'])
        self.assertEqual(candidate['speaker_activity'],live['speaker_activity'])
        self.assertEqual(candidate['reading_turns'],live['reading_turns'])
        self.assertNotIn('canonical_unresolved',candidate)
        self.assertTrue(all(d['reused'] and not d['provider_executed'] for d in self.decisions()))
        self.assertTrue(all(d['consumer_binding']['text_revision']==live['canonical_machine_revision']
                            for d in self.decisions()))
    def test_observer_executes_current_model_and_compares_both_tokens_and_raw_words(self):
        self.peer.final_asr_reuse.mode='observe'
        self.engine.canonical.refine(self.request())
        self.assertEqual(self.peer.cohere_calls,4)
        self.assertTrue(all(d['eligible'] and d['shadow_text_equal'] and d['shadow_tokens_equal']
                            and not d['reused'] for d in self.decisions()))
    def test_retry_missing_purpose_missing_cas_and_unqualified_identity_force_fresh_asr(self):
        for field in ('explicit_retry','purpose_missing','expected_missing','qualification_missing'):
            with self.subTest(field=field):
                request=self.request(field)
                if field=='explicit_retry':request['asr_purpose']='explicit_retry'
                elif field=='purpose_missing':request.pop('asr_purpose')
                elif field=='expected_missing':request['expected']={}
                else:self.peer.final_asr_reuse.qualified=False
                before=self.peer.cohere_calls
                self.engine.canonical.refine(request)
                self.assertEqual(self.peer.cohere_calls-before,2)
    def test_stale_text_revision_cancels_before_reuse_and_protected_text_survives_fresh_final(self):
        request=self.request();book=self.engine.canonical.book
        book.edit(self.row['id'],self.row['machine_revision'],'  Human go go café  ')
        result=self.engine.canonical.refine(request)
        self.assertTrue(result['cancelled']);self.assertEqual(self.peer.cohere_calls,2)
        request=self.request('after-edit')
        result=self.engine.canonical.refine(request)
        self.assertEqual(self.peer.cohere_calls,4)
        self.assertEqual(result['canonical_candidate']['text'],'  Human go go café  ')
    def test_corrupted_selected_authority_reruns_instead_of_inheriting_old_result(self):
        cache=self.peer.final_asr_reuse
        import json
        saved=json.loads(next(iter(cache.entries.values())))
        Path(saved['authority']['path']).write_text('{}')
        self.engine.canonical.refine(self.request())
        self.assertEqual(self.peer.cohere_calls,3)
        self.assertEqual(sum(d['reused'] for d in self.decisions()),1)
    def test_cancel_during_historical_inspection_restores_prior_text_without_acknowledging(self):
        before=copy.deepcopy(self.row);original=self.peer.begin_refinement
        def cancel(pcm,start):
            original(pcm,start);self.fixture.inbox.push({'type':'cancel_refinement','operation_id':'cancelled'})
        self.peer.begin_refinement=cancel
        result=self.engine.canonical.refine(self.request('cancelled'))
        self.assertTrue(result['cancelled'])
        self.assertEqual(self.engine.canonical.book.rows[self.row['id']],before)

    def test_short_final_keeps_batch_diarization_and_does_not_inherit_long_reuse_scope(self):
        book,row=book_for(seconds=7);self.engine.canonical.book=book;self.row=book.rows[row['id']]
        self.engine.canonical.decode(self.row,'live');before=self.peer.cohere_calls
        request=self.request('short');request['window']['end_sample']=7*16000
        self.engine.canonical.refine(request)
        self.assertEqual(self.peer.batch_calls,1)
        self.assertEqual(self.peer.cohere_calls-before,1)
        self.assertFalse(self.peer.final_asr_reuse.active)

    def test_new_negative_admission_prevents_cached_asr_from_becoming_speech(self):
        def negative(pcm,start):
            self.peer.historical_calls+=1;self.peer.speech_historical=SpeechEvidencePeer('no_speech')
        self.peer.begin_refinement=negative
        self.engine.canonical.refine(self.request('negative'))
        self.assertEqual(self.peer.historical_calls,2)
        self.assertEqual(self.peer.cohere_calls,2)
        self.assertEqual(self.decisions(),[])

    def test_device_or_training_mode_changes_cannot_silently_inherit_gpu_results(self):
        self.peer.asr.model.training=True
        before=self.peer.cohere_calls
        self.engine.canonical.refine(self.request('training'))
        self.assertEqual(self.peer.cohere_calls-before,2)
        self.peer.mx.default_device=lambda:SimpleNamespace(type='synthetic_cpu')
        before=self.peer.cohere_calls
        with self.assertRaisesRegex(RuntimeError,'CPU fallback is disabled'):
            self.engine.canonical.refine(self.request('cpu'))
        self.assertEqual(self.peer.cohere_calls,before)


class InputValidityTests(unittest.TestCase):
    def test_failed_empty_and_token_limited_results_remain_fresh_retryable_work(self):
        settings={'sample_rate':16000,'language':'en','punctuation':True,'itn':False,'max_new_tokens':448}
        scope={'stage':'live','purpose':'live','eligible':True,'binding':
            {'utterance_id':'u','language_epoch':0,'expected_source_sha256':'a'*64}}
        provider=SimpleNamespace(model=SimpleNamespace(training=False))
        for fault in ('empty','token_limit','failure'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as folder:
                cache=FinalAsrReuse(IDENTITY,mode='reuse',qualification=digest(IDENTITY))
                raw=SimpleNamespace(text='' if fault=='empty' else 'Raw words',
                    tokens=list(range(448)) if fault=='token_limit' else [7],language='en')
                compute=Mock(side_effect=RuntimeError('synthetic model failure')) if fault=='failure' else Mock(return_value=raw)
                cache.begin(scope)
                if fault=='failure':
                    with self.assertRaisesRegex(RuntimeError,'synthetic model failure'):
                        cache.resolve(provider,varying_pcm(16000),settings,{},compute)
                else:cache.resolve(provider,varying_pcm(16000),settings,{},compute)
                import json
                path=Path(folder)/'authority.json'
                path.write_text(json.dumps({'part':{'complete':True,'text':raw.text,'passages':[{}]}}))
                cache.finish({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                self.assertEqual(len(cache.entries),0)
                compute.side_effect=None;compute.return_value=SimpleNamespace(text='New retry words',tokens=[8],language='en')
                cache.begin(dict(scope,stage='refined',purpose='automatic_final'))
                result,decision=cache.resolve(provider,varying_pcm(16000),settings,{},compute)
                self.assertEqual(result.text,'New retry words');self.assertFalse(decision['reused'])
                self.assertEqual(compute.call_count,2)

    def test_entry_eviction_preserves_fresh_decoder_fallback(self):
        settings={'sample_rate':16000,'language':'en','punctuation':True,'itn':False,'max_new_tokens':448}
        scope={'stage':'live','purpose':'live','eligible':True,'binding':
            {'utterance_id':'u','language_epoch':0,'expected_source_sha256':'a'*64}}
        provider=SimpleNamespace(model=SimpleNamespace(training=False))
        raw=SimpleNamespace(text='Raw words',tokens=[7],language='en');compute=Mock(return_value=raw)
        with tempfile.TemporaryDirectory() as folder:
            import json
            path=Path(folder)/'authority.json'
            path.write_text(json.dumps({'part':{'complete':True,'text':raw.text,'passages':[{}]}}))
            authority={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            cache=FinalAsrReuse(IDENTITY,mode='reuse',qualification=digest(IDENTITY),max_entries=1)
            for start in (0,1):
                cache.begin(scope);cache.resolve(provider,varying_pcm(16000),settings,{'start_sample':start},compute)
                cache.finish(authority)
            self.assertEqual(len(cache.entries),1);self.assertLessEqual(cache.bytes,cache.max_bytes)
            cache.begin(dict(scope,stage='refined',purpose='automatic_final'))
            _,decision=cache.resolve(provider,varying_pcm(16000),settings,{'start_sample':0},compute)
            self.assertFalse(decision['reused']);self.assertEqual(compute.call_count,3)

    def test_changed_route_language_decoder_pcm_model_and_padding_cannot_alias(self):
        # Actual entry point has no ASR text prompt/context. We conservatively
        # require matching entire routing evidence as well as decoder arguments.
        base_settings={'sample_rate':16000,'language':'en','punctuation':True,'itn':False,'max_new_tokens':448}
        base_route={'start_sample':0,'end_sample':16000,'padding':[0,0],
                    'language_detection':{'reason':'recent_context','context_end_sample':0}}
        for fault in ('language','max_tokens','punctuation','itn','routing_context','pcm','padding','model','epoch'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as folder:
                cache=FinalAsrReuse(IDENTITY,mode='reuse',qualification=digest(IDENTITY))
                provider=SimpleNamespace(config={'version':1},model=SimpleNamespace(training=False))
                raw=SimpleNamespace(text='  Raw go go café  ',tokens=[7],language='en')
                compute=Mock(return_value=raw);pcm=varying_pcm(16000)
                scope={'stage':'live','purpose':'live','eligible':True,'binding':
                    {'utterance_id':'u','language_epoch':0,'expected_source_sha256':'a'*64}}
                cache.begin(scope);cache.resolve(provider,pcm,base_settings,base_route,compute)
                import json
                path=Path(folder)/'authority.json'
                path.write_text(json.dumps({'part':{'complete':True,'text':raw.text,'passages':[{}]}}))
                cache.finish({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                settings=copy.deepcopy(base_settings);route=copy.deepcopy(base_route);scope=copy.deepcopy(scope)
                scope.update(stage='refined',purpose='automatic_final')
                if fault=='language':settings['language']='fr'
                elif fault=='max_tokens':settings['max_new_tokens']=447
                elif fault=='punctuation':settings['punctuation']=False
                elif fault=='itn':settings['itn']=True
                elif fault=='routing_context':route['language_detection']['context_end_sample']=1
                elif fault=='pcm':pcm[10]+=.01
                elif fault=='padding':route['padding']=[3200,0]
                elif fault=='model':provider.config={'version':2}
                else:scope['binding']['language_epoch']=1
                cache.begin(scope);cache.resolve(provider,pcm,settings,route,compute)
                self.assertEqual(compute.call_count,2)


class HostPurposeTests(unittest.TestCase):
    def test_automatic_dispatch_and_explicit_resume_have_distinct_trusted_purpose(self):
        from test_unknown_speaker_identity import source_job
        with source_job() as context:
            manager=context['manager'];jid=context['jid'];control=manager.refinement
            control.ready(jid,True);control.schedule(jid)
            first=manager.worker_controls.get_nowait()
            self.assertEqual(first['asr_purpose'],'automatic_final')
            control.pause(jid);control.resume(jid)
            cancellation=manager.worker_controls.get_nowait()
            self.assertEqual(cancellation['type'],'cancel_refinement')
            resumed=manager.worker_controls.get_nowait()
            self.assertEqual(resumed['asr_purpose'],'explicit_retry')
            self.assertTrue(manager.get(jid)['canonical_force_fresh_asr'])


if __name__=='__main__':unittest.main()
