"""Current namespace admission and exact source separation; CPU protocol peers."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import asr_reuse_profile as subject
import test_asr_reuse_profile as fixtures
from capture_sources import catalog,binding,utterance_id
from final_asr_reuse import FinalAsrReuse,digest


class CurrentProfileTests(unittest.TestCase):
    def setUp(self):
        self.base=fixtures.ProfileTests();self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.value=copy.deepcopy(self.base.profile)
        self.value.update(namespace='source_bound_canonical_v1',
            lineage={'legacy_profile_sha256':subject.PROFILE_SHA256,
                'reuse_algorithm_sha256':'a'*64},
            qualification={'current_namespace_execution_qualified':False})
        self.base.profile['identity']['policy_source_sha256']['final_asr_reuse.py']='a'*64
        self.base.write(self.base.package/'final_asr_reuse.py',b'reuse algorithm')
        self.base.profile['identity']['policy_source_sha256']['final_asr_reuse.py']=subject.file_digest(self.base.package/'final_asr_reuse.py')
        self.value['identity']['policy_source_sha256']=copy.deepcopy(self.base.profile['identity']['policy_source_sha256'])
        self.value['lineage']['reuse_algorithm_sha256']=self.base.profile['identity']['policy_source_sha256']['final_asr_reuse.py']
        legacy=self.base.package/'asr-reuse-profile.json';legacy.write_text(json.dumps(self.base.profile))
        patch.object(subject,'PROFILE_SHA256',subject.file_digest(legacy)).start()
        self.value['lineage']['legacy_profile_sha256']=subject.PROFILE_SHA256
        self.save()
    def save(self):
        path=self.base.package/'asr-source-reuse-profile.json';path.write_text(json.dumps(self.value))
        patch.object(subject,'SOURCE_PROFILE_SHA256',subject.file_digest(path)).start()
    def activate(self,mode='reuse'):
        return self.base.activate(asr_final_reuse_mode=mode)
    def test_unqualified_current_profile_observes_but_cannot_reuse(self):
        self.assertIsNone(self.activate()[0])
        cache,status=self.activate('observe')
        self.assertFalse(cache.qualified);self.assertEqual(cache.mode,'observe')
        self.assertFalse(status['final_namespace_execution_qualified'])
    def test_sealed_qualified_current_profile_activates_process_local_cache(self):
        self.value['qualification']['current_namespace_execution_qualified']=True;self.save()
        cache,status=self.activate();self.assertTrue(cache.qualified)
        self.assertTrue(status['final_namespace_execution_qualified']);self.assertEqual(len(cache.entries),0)
    def test_current_profile_corruption_and_backend_lineage_changes_refuse(self):
        p=self.base.package/'asr-source-reuse-profile.json';p.write_text('{}')
        self.assertIsNone(self.activate('observe')[0]);self.save()
        self.value['identity']['device']='cpu';self.save()
        self.assertIsNone(self.activate('observe')[0])
    def test_even_sealed_bad_legacy_lineage_or_source_bytes_refuse(self):
        self.value['lineage']['legacy_profile_sha256']='f'*64;self.save()
        self.assertIsNone(self.activate('observe')[0])
        self.value['lineage']['legacy_profile_sha256']=subject.PROFILE_SHA256;self.save()
        (self.base.package/'canonical_runtime.py').write_bytes(b'changed')
        self.assertIsNone(self.activate('observe')[0])


class SourceSeparationTests(unittest.TestCase):
    def test_same_pcm_cannot_cross_source_or_mutated_authority(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            job='b'*32;c=catalog(job,innovation=True,startup=True)
            near=binding(c,'microphone_clean');far=binding(c,'system')
            identity={'device':'gpu','scope':'CPU peer only'}
            cache=FinalAsrReuse(identity,mode='reuse',qualification=digest(identity))
            provider=SimpleNamespace(model=SimpleNamespace(training=False),tokenizer=object(),config={},feature_extractor=SimpleNamespace())
            settings={'language':'en','max_new_tokens':448};audio=np.arange(4000,dtype=np.float32)/4000
            def scope(source,stage):return {'stage':stage,'purpose':'live' if stage=='live' else 'automatic_final','eligible':True,
                'binding':{'utterance_id':utterance_id(job,0,0,source),'language_epoch':0,'expected_source_sha256':'a'*64}}
            calls=[]
            def compute():calls.append(1);return SimpleNamespace(text='Exact public peer words',tokens=[1,2],language='en')
            cache.begin(scope(near,'live'));raw,_=cache.resolve(provider,audio,settings,{},compute)
            path=Path(folder)/'authority.json';path.write_text(json.dumps({'part':{'complete':True,'text':raw.text,'passages':[{}]}}))
            authority={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};cache.finish(authority)
            cache.begin(scope(near,'refined'));_,d=cache.resolve(provider,audio,settings,{},compute);cache.finish()
            self.assertTrue(d['reused']);self.assertEqual(len(calls),1)
            cache.begin(scope(far,'refined'));_,d=cache.resolve(provider,audio,settings,{},compute);cache.finish()
            self.assertFalse(d['reused']);self.assertEqual(len(calls),2)
            changed=audio.copy();changed[0]=.01
            cache.begin(scope(near,'refined'));_,d=cache.resolve(provider,changed,settings,{},compute);cache.finish()
            self.assertFalse(d['reused']);self.assertEqual(len(calls),3)
            path.write_text('{}')
            cache.begin(scope(near,'refined'));_,d=cache.resolve(provider,audio,settings,{},compute);cache.finish()
            self.assertFalse(d['reused']);self.assertEqual(len(calls),4)
