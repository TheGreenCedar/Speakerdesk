"""Exact language-score reuse with synthetic peers; no neural inference."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from pcm_peer import SpeechEvidencePeer, varying_pcm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from language_detection import LanguageProbeCache, SpeechTranscriber
from live_refinement import Engine, Inbox, Models

RATE=16000
MODEL_IDENTITY=('synthetic_fixture_model_v1',)


class LanguageProbeReuseTests(unittest.TestCase):
    def transcriber(self, detector, cache, *, context=None, epoch=0, evidence=None):
        asr=Mock()
        asr.transcribe.side_effect=lambda audio,**kw:SimpleNamespace(
            text='  Raw '+kw['language']+' go go café  ',tokens=[1])
        return SpeechTranscriber(asr,'auto',detector=detector,context=context,
            speech_evidence=evidence or SpeechEvidencePeer(),probe_cache=cache,language_epoch=epoch)

    def test_resident_repeated_anchor_reuses_probes_but_runs_every_cohere_revision(self):
        # Exercise Engine's epoch propagation and the real Models.transcribe
        # owner, substituting only the model and resource boundaries.
        models=Models.__new__(Models)
        models.config={};models.check_memory=lambda:None;models.mx=SimpleNamespace(clear_cache=lambda:None)
        models.detector=Mock();models.detector.detect.return_value={'en':.97,'fr':.03}
        models.language_probe_cache=LanguageProbeCache();models.language_epoch=0
        models.cohere_calls=0;models.asr_padding=(0,0)
        models.speech_historical=None;models.speech_live=SimpleNamespace(evidence=SpeechEvidencePeer())
        models.asr=Mock();models.asr.transcribe.side_effect=[
            SimpleNamespace(text='  live go go café  ',tokens=[1]),
            SimpleNamespace(text='  final changed words  ',tokens=[1]),
            SimpleNamespace(text='  fresh epoch words  ',tokens=[1])]
        engine=Engine({'audio_path':'unused','language':'auto'},models,lambda event:None,Inbox())
        pcm=varying_pcm(6*RATE)
        live=engine.decode(pcm,'auto',['speaker_0'],0,0)
        # The final pass still uses independent historical admission.
        historical=SpeechEvidencePeer();historical.admission=Mock(wraps=historical.admission)
        models.speech_historical=historical
        final=engine.decode(pcm.copy(),'auto',['speaker_0'],0,0)
        self.assertEqual(models.detector.detect.call_count,2)
        self.assertEqual(models.cohere_calls,2)
        self.assertEqual(live[0]['cohere_raw_text'],'  live go go café  ')
        self.assertEqual(final[0]['cohere_raw_text'],'  final changed words  ')
        self.assertGreaterEqual(historical.admission.call_count,3)
        self.assertEqual(live[0]['language_detection'],final[0]['language_detection'])
        engine.decode(pcm,'auto',['speaker_0'],0,1)
        self.assertEqual(models.language_epoch,1)
        self.assertEqual(models.detector.detect.call_count,4)
        self.assertEqual(models.cohere_calls,3)
        for call in models.asr.transcribe.call_args_list:np.testing.assert_array_equal(call.args[0],pcm)

    def test_same_weak_scores_reroute_from_each_frozen_context(self):
        detector=Mock();detector.detect.return_value={'en':.6,'fr':.4}
        cache=LanguageProbeCache();pcm=varying_pcm(RATE)
        first=self.transcriber(detector,cache,context={'language':'fr','end_sample':RATE})
        second=self.transcriber(detector,cache,context={'language':'en','end_sample':RATE})
        a=first.transcribe(pcm,RATE,('speaker_0',),start_sample=2*RATE)[0]
        b=second.transcribe(pcm,RATE,('speaker_1',),start_sample=2*RATE)[0]
        self.assertEqual(detector.detect.call_count,1)
        self.assertEqual((a['language'],b['language']),('fr','en'))
        self.assertEqual((a['language_detection']['reason'],b['language_detection']['reason']),
                         ('recent_context','recent_context'))
        self.assertEqual(a['cohere_raw_text'],'  Raw fr go go café  ')
        self.assertEqual(b['cohere_raw_text'],'  Raw en go go café  ')
        self.assertTrue(a['language_review']);self.assertTrue(b['language_review'])

    def test_crop_pcm_epoch_dtype_model_and_provider_changes_never_alias(self):
        pcm=varying_pcm(RATE);changed=pcm.copy();changed[10]+=.001
        cases=[{'audio':changed},{'audio':pcm[:-1]},{'audio':pcm.astype(np.float64)},
               {'start_sample':1},{'language_epoch':1},{'model_identity':('different_model_config',)}]
        for changes in cases:
            with self.subTest(changes=list(changes)):
                detector=Mock();detector.detect.return_value={'en':.97,'fr':.03}
                cache=LanguageProbeCache();kw=dict(audio=pcm,start_sample=0,language_epoch=0,model_identity=MODEL_IDENTITY)
                cache.detect(detector,**kw);cache.detect(detector,**{**kw,**changes})
                self.assertEqual(detector.detect.call_count,2)
        replacement=Mock();replacement.detect.return_value={'fr':.97,'en':.03}
        self.assertEqual(cache.detect(replacement,**kw),{'fr':.97,'en':.03})
        replacement.detect.assert_called_once()

    def test_eviction_and_mutated_returned_scores_do_not_change_future_results(self):
        detector=Mock();detector.detect.return_value={'en':.97,'fr':.03};pcm=varying_pcm(RATE)
        cache=LanguageProbeCache(max_entries=1,max_bytes=64)
        result=cache.detect(detector,pcm,start_sample=0,language_epoch=0,model_identity=MODEL_IDENTITY);result['en']=0
        self.assertEqual(cache.detect(detector,pcm,start_sample=0,language_epoch=0,model_identity=MODEL_IDENTITY),{'en':.97,'fr':.03})
        detector.detect.return_value={'en':.97,'fr':.03}
        cache.detect(detector,pcm,start_sample=RATE,language_epoch=0,model_identity=MODEL_IDENTITY)
        cache.detect(detector,pcm,start_sample=0,language_epoch=0,model_identity=MODEL_IDENTITY)
        self.assertEqual(detector.detect.call_count,3)
        self.assertLessEqual(len(cache.entries),1);self.assertLessEqual(cache.bytes,64)
        uncached=LanguageProbeCache(max_bytes=1)
        for _ in range(2):uncached.detect(detector,pcm,start_sample=0,language_epoch=0,model_identity=MODEL_IDENTITY)
        self.assertEqual(detector.detect.call_count,5);self.assertEqual(uncached.bytes,0)

    def test_failed_or_invalid_probe_retries_without_promoting_stale_language(self):
        for failure in (RuntimeError('synthetic detector failure'),{'en':float('nan')}):
            with self.subTest(failure=type(failure).__name__):
                detector=Mock();detector.detect.side_effect=[failure,{'en':.97,'fr':.03}]
                cache=LanguageProbeCache();pcm=varying_pcm(RATE)
                first=self.transcriber(detector,cache).transcribe(pcm,RATE,('speaker_0',))[0]
                second=self.transcriber(detector,cache).transcribe(pcm,RATE,('speaker_0',))[0]
                self.assertTrue(first['language_detection']['detector_error'])
                self.assertEqual(first['text'],'');self.assertEqual(second['language'],'en')
                self.assertEqual(detector.detect.call_count,2)

    def test_cached_language_cannot_bypass_new_negative_speech_evidence(self):
        detector=Mock();detector.detect.return_value={'en':.97,'fr':.03}
        cache=LanguageProbeCache();pcm=varying_pcm(RATE)
        self.transcriber(detector,cache).transcribe(pcm,RATE,('speaker_0',))
        transcriber=self.transcriber(detector,cache,evidence=SpeechEvidencePeer('no_speech'))
        negative=transcriber.transcribe(pcm,RATE,('speaker_0',))[0]
        self.assertEqual(negative['audio_state'],'model_non_speech');self.assertEqual(negative['text'],'')
        transcriber.asr.transcribe.assert_not_called();self.assertEqual(detector.detect.call_count,1)


if __name__=='__main__':unittest.main()
