"""CPU-only exact evidence/cache contracts; provider peers are not acoustic proof."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from alignment_cache import AlignmentCache
from coarse_alignment import CoarseAlignment,CALIBRATION_ID,TOKEN_SHA256
from word_alignment import MODEL_SHA256

IDENTITY=(MODEL_SHA256,TOKEN_SHA256,CALIBRATION_ID,'ctc_emission_cell_envelope',320,0,-20,'mlx-0.32.2-cached-float32-gpu','ctc-segmentation-1.7.4')

def evidence(key):
    _,text,audio,start,end,_=key
    return {'status':'aligned','complete':True,'text_source':'cohere','text_sha256':text,
        'audio_float32_sha256':audio,'audio_anchor':{'start_sample':start,'end_sample':end},
        'model_sha256':MODEL_SHA256,'frame_calibration_id':CALIBRATION_ID,
        'score_calibration_id':CALIBRATION_ID,'timing_kind':'ctc_emission_cell_envelope',
        'qualified_scope':'bounded_ami_english_coarse_envelopes','words':[{'text':'café','start_sample':start,'end_sample':end}]}

class CacheTests(unittest.TestCase):
    def test_worker_disk_reuses_25_exact_results_after_hot_eviction(self):
        with tempfile.TemporaryDirectory() as folder:
            c=AlignmentCache(IDENTITY,directory=Path(folder)/'worker',max_entries=1,max_bytes=1000)
            keys=[c.key(str(i),b'pcm',i,i+1,'en') for i in range(25)]
            for key in keys:self.assertTrue(c.put(key,evidence(key)))
            for key in keys:
                got=c.get(key);self.assertEqual(got,evidence(key));got['words'][0]['text']='changed'
            self.assertEqual(c.disk_hits,25);self.assertLessEqual(c.bytes,1000)
            self.assertEqual(len(c.disk),25);self.assertEqual(c.get(keys[0]),evidence(keys[0]))
            self.assertEqual(c.provider_calls,0)
    def test_disk_keys_preserve_raw_unicode_audio_clock_policy_and_worker_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            c=AlignmentCache(IDENTITY,directory=Path(folder)/'worker',max_entries=1)
            key=c.key('é é',b'pcm',0,8,'en');c.put(key,evidence(key));c.entries.clear();c.bytes=0
            for args in [('e\u0301 é',b'pcm',0,8,'en'),('é é',b'other',0,8,'en'),
                         ('é é',b'pcm',1,9,'en'),('é é',b'pcm',0,9,'en'),('é é',b'pcm',0,8,'fr')]:
                self.assertIsNone(c.get(c.key(*args)))
            other=AlignmentCache((*IDENTITY[:-1],'other'),directory=Path(folder)/'other')
            self.assertIsNone(c.get(other.key('é é',b'pcm',0,8,'en')))
            self.assertIsNone(other.get(key));self.assertEqual(c.get(key),evidence(key))
    def test_disk_corruption_missing_file_and_failed_optional_write_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as folder:
            c=AlignmentCache(IDENTITY,directory=Path(folder)/'worker',max_entries=1)
            key=c.key('words',b'pcm',0,8,'en');c.put(key,evidence(key));c.entries.clear();c.bytes=0
            path=c.disk[key][0];data=path.read_bytes();path.write_bytes(data.replace(b'cafe',b'xxxx')+b' ')
            self.assertIsNone(c.get(key));path.unlink();self.assertIsNone(c.get(key))
            with patch.object(Path,'open',side_effect=OSError('disk unavailable')):
                self.assertTrue(c.put(key,evidence(key)))
            self.assertEqual(c.get(key),evidence(key))
    def test_disk_entry_and_byte_eviction_partial_exclusion_and_oversized_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            c=AlignmentCache(IDENTITY,directory=Path(folder)/'worker',max_entries=1,
                             max_bytes=1000,disk_max_entries=2,disk_max_bytes=2000)
            keys=[c.key(str(i),b'pcm',i,i+1,'en') for i in range(3)]
            for k in keys:c.put(k,evidence(k))
            self.assertIsNone(c.get(keys[0]));self.assertIsNotNone(c.get(keys[1]))
            self.assertEqual(len(list(c.directory.glob('*.json'))),2)
            large=dict(evidence(keys[0]),extra='x'*2001);self.assertFalse(c.put(keys[0],large))
            self.assertFalse(c.put(keys[0],dict(evidence(keys[0]),complete=False)))
            self.assertLessEqual(c.disk_bytes,2000);self.assertEqual(c.disk_bytes,sum(p.stat().st_size for p in c.directory.glob('*.json')))
    def test_optional_directory_failure_preserves_hot_cache_and_provider_fallback(self):
        with patch.object(Path,'mkdir',side_effect=OSError('unavailable')):
            c=AlignmentCache(IDENTITY,directory='unavailable')
        key=c.key('words',b'pcm',0,8,'en');self.assertTrue(c.put(key,evidence(key)))
        self.assertEqual(c.get(key),evidence(key));self.assertTrue(c.disk_write_disabled)
        self.assertIsNone(c.directory)
    def test_failed_eviction_remains_counted_and_prevents_more_disk_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            c=AlignmentCache(IDENTITY,directory=Path(folder)/'worker',disk_max_entries=2)
            keys=[c.key(str(i),b'pcm',i,i+1,'en') for i in range(8)]
            with patch.object(Path,'unlink',side_effect=OSError('blocked cleanup')):
                for key in keys:self.assertTrue(c.put(key,evidence(key)))
            files=list(c.directory.glob('*.json'))
            self.assertEqual(len(files),2);self.assertEqual(len(c.disk),2)
            self.assertEqual(c.disk_bytes,sum(p.stat().st_size for p in files));self.assertTrue(c.disk_write_disabled)
            self.assertEqual(c.get(keys[-1]),evidence(keys[-1]))
    def test_partial_write_cleanup_failure_is_counted_and_stops_disk_growth(self):
        with tempfile.TemporaryDirectory() as folder:
            c=AlignmentCache(IDENTITY,directory=Path(folder)/'worker')
            keys=[c.key(str(i),b'pcm',i,i+1,'en') for i in range(8)];original=Path.open
            class Partial:
                def __init__(self,path):self.stream=original(path,'xb')
                def __enter__(self):return self
                def write(self,data):self.stream.write(data[:17]);self.stream.flush();raise OSError('partial write')
                def __exit__(self,*args):self.stream.close()
            with patch.object(Path,'open',lambda path,*args,**kw:Partial(path)),patch.object(Path,'unlink',side_effect=OSError('blocked cleanup')):
                for key in keys:self.assertTrue(c.put(key,evidence(key)))
            files=list(c.directory.glob('*.json'));self.assertEqual(len(files),1)
            self.assertEqual(c.disk_bytes,17);self.assertTrue(c.disk_write_disabled)
            c.entries.clear();c.bytes=0;self.assertIsNone(c.get(keys[0]))
    def test_exact_evidence_hit_and_mutation_isolation(self):
        c=AlignmentCache(IDENTITY);key=c.key('café café',b'audio',100,200,'en');original=evidence(key)
        self.assertTrue(c.put(key,original));original['words'][0]['text']='changed'
        got=c.get(key);self.assertEqual(got['words'][0]['text'],'café')
        got['words'][0]['start_sample']=None
        self.assertEqual(c.get(key)['words'][0]['start_sample'],100)
        self.assertEqual(c.metrics()['cache_hits'],2)
    def test_exact_text_pcm_bounds_language_identity_keys(self):
        c=AlignmentCache(IDENTITY);key=c.key('é é',b'pcm',0,8,'en');c.put(key,evidence(key))
        for args in [('e\u0301 é',b'pcm',0,8,'en'),('é é',b'other',0,8,'en'),
                     ('é é',b'pcm',1,9,'en'),('é é',b'pcm',0,9,'en'),('é é',b'pcm',0,8,'fr')]:
            self.assertIsNone(c.get(c.key(*args)))
        changed=AlignmentCache((*IDENTITY[:-1],'different-runtime'))
        self.assertIsNone(c.get(changed.key('é é',b'pcm',0,8,'en')))
    def test_uncalibrated_partial_failed_and_forged_receipts_not_cached(self):
        c=AlignmentCache(IDENTITY);key=c.key('words',b'pcm',10,20,'en')
        changes=[{'complete':False},{'complete':1},{'status':'partial'},{'text_source':'other'},
                 {'text_sha256':'bad'},{'audio_float32_sha256':'bad'},{'audio_anchor':{'start_sample':0,'end_sample':20}},
                 {'model_sha256':'bad'},{'frame_calibration_id':'bad'},{'score_calibration_id':'bad'},
                 {'timing_kind':'unverified'},{'qualified_scope':'other'}]
        for change in changes:self.assertFalse(c.put(key,dict(evidence(key),**change)))
        self.assertFalse(c.put(key,None));self.assertEqual(c.metrics()['cache_entries'],0)
        bad=evidence(key);bad['words'][0]['start_sample']=float('nan')
        self.assertFalse(c.put(key,bad))
    def test_entry_lru_and_byte_limits_and_oversized_rejection(self):
        c=AlignmentCache(IDENTITY,max_entries=2,max_bytes=10000)
        keys=[c.key(str(i),b'audio',0,20,'en') for i in range(3)]
        for k in keys[:2]:self.assertTrue(c.put(k,evidence(k)))
        c.get(keys[0]);c.put(keys[2],evidence(keys[2]))
        self.assertIsNone(c.get(keys[1]));self.assertIsNotNone(c.get(keys[0]))
        large=evidence(keys[0]);large['extra']='x'*10001
        self.assertFalse(c.put(keys[0],large));self.assertLessEqual(c.bytes,10000)
        small=AlignmentCache(IDENTITY,max_bytes=1000)
        for k in keys:small.put(k,evidence(k))
        self.assertLessEqual(small.bytes,1000)
        self.assertEqual(small.bytes,sum(len(v) for v in small.entries.values()))
    def test_actual_provider_wrapper_hits_before_session_and_returns_fresh_result(self):
        import numpy as np
        class ForwardPeer:
            def __init__(self):self.calls=0
            def __call__(self,*args,execution):
                self.calls+=1;execution.update(backend='mlx_metal_gpu',evaluated_and_GPU_synchronized=True);return np.zeros((3,4),dtype=np.float32)
        forward=ForwardPeer();provider=CoarseAlignment.__new__(CoarseAlignment);provider.model=object();provider.conversion={'source_model_sha256':'e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c'};provider.actual_gpu_forwards=0;provider.vocabulary={};provider.cache=AlignmentCache(IDENTITY)
        audio=np.array([.1,.2,.3],dtype=np.float32)
        key=provider.cache.key('café',audio.astype('<f4').tobytes(),100,103,'en')
        result=evidence(key)
        with patch('mlx_ctc_forward.forward_scores',side_effect=forward), patch('coarse_alignment.align_ctc_scores',return_value=result):
            first=provider.align(audio,'café',start_sample=100,language='en')
            first['words'][0]['start_sample']=None
            second=provider.align(audio,'café',start_sample=100,language='en')
        self.assertEqual(forward.calls,1);self.assertEqual(second['words'][0]['start_sample'],100)
        self.assertEqual(provider.cache.metrics()['provider_calls'],1)
        self.assertEqual(provider.cache.metrics()['cache_hits'],1)
    def test_partial_provider_results_repeat_and_unsupported_language_never_runs(self):
        import numpy as np
        class ForwardPeer:
            def __init__(self):self.calls=0
            def __call__(self,*args,execution):
                self.calls+=1;execution.update(backend='mlx_metal_gpu',evaluated_and_GPU_synchronized=True);return np.zeros((3,4),dtype=np.float32)
        forward=ForwardPeer();p=CoarseAlignment.__new__(CoarseAlignment);p.model=object();p.conversion={'source_model_sha256':'e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c'};p.actual_gpu_forwards=0;p.vocabulary={};p.cache=AlignmentCache(IDENTITY)
        audio=np.array([.1,.2,.3],dtype=np.float32)
        key=p.cache.key('words',audio.astype('<f4').tobytes(),0,3,'en')
        with patch('mlx_ctc_forward.forward_scores',side_effect=forward), patch('coarse_alignment.align_ctc_scores',return_value=dict(evidence(key),status='partial',complete=False)):
            p.align(audio,'words',start_sample=0,language='en');p.align(audio,'words',start_sample=0,language='en')
        self.assertEqual(forward.calls,2);self.assertEqual(p.cache.bytes,0)
        self.assertIsNone(p.align(audio,'words',start_sample=0,language='fr'));self.assertEqual(forward.calls,2)
