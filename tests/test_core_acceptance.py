"""Policy tests with explicitly fabricated traces; never actual-model acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'scripts'))
import core_acceptance as core
import promote_release
import packaged_replay

class CoreAcceptanceTests(unittest.TestCase):
    def test_discovery_includes_mandatory_speech_checkpoint_and_metadata(self):
        pins=core.models(self.root);files=core.model_file_pins(self.root)
        self.assertEqual(set(pins),{'nemotron','cohere-speech','whisper-language','silero-speech','coarse-alignment'})
        self.assertEqual(files['coarse-alignment']['model.int8.onnx'],pins['coarse-alignment']['sha256'])
        self.assertEqual(files['silero-speech']['model.safetensors'],pins['silero-speech']['sha256'])
        self.assertEqual(set(files['silero-speech']),set(pins['silero-speech']['files']))
        path=self.root/'speakerdesk/model_setup.py'
        path.write_text(path.read_text().replace('    SILERO_SPEC,','    UNKNOWN_MODEL_SPEC,'))
        with self.assertRaisesRegex(ValueError,'Unresolved model identity'):core.models(self.root)
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.base=Path(self.temporary.name);self.root=self.base/'source';self.root.mkdir()
        paths=list(core.HARNESSES)+['scripts/promote_release.py','speakerdesk/model_setup.py','speakerdesk/language_detection.py','speakerdesk/speech_admission.py','speakerdesk/alignment_artifact.py','tests/acceptance/recipes.json','tests/acceptance/holdout-recipes.json','tests/acceptance/model-metadata.json']
        for name in paths:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,target)
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        subprocess.run(['git','add','.'],cwd=self.root,check=True)
        subprocess.run(['git','-c','user.name=Acceptance Contract Test','-c','user.email=contract@example.invalid','commit','-qm','Synthetic policy fixture'],cwd=self.root,check=True)
        self.head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        self.evidence=self.base/'evidence';self.evidence.mkdir();self.report_path=self.evidence/'core.json';self.manifest_path=self.evidence/'artifact-manifest.json'
        archive=self.evidence/'Speakerdesk_0.0.0_AppleSilicon.app.zip';runtime=b'Synthetic contract bytes, not an executable'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('Speakerdesk.app/Contents/MacOS/speakerdesk-runtime',runtime)
        self.manifest={'product':'Speakerdesk','source_commit':self.head,'signing':'developer-id','notarized':True,'files':[{'filename':archive.name,'bytes':archive.stat().st_size,'sha256':core.digest_file(archive)}]}
        cases,identity=core.suite(self.root)
        with zipfile.ZipFile(archive) as z:bundle=core.bundle_identity(core.zip_bundle(z))
        self.report={'schema_version':1,'scope':'packaged_production_replay','runtime_scope':'frozen_packaged_backend_live_and_refinement',
             'source_commit':self.head,'status':'passed','complete':True,'models_executed':True,'private_audio':False,'native_capture':False,
             'model_substitution':False,'probabilities_modified':False,'model_pins':core.models(self.root),'suite_sha256':identity,
             'harness_sha256':{name:core.digest_file(self.root/name) for name in core.HARNESSES},'package_sha256':core.digest_file(archive),
             'runtime_sha256':hashlib.sha256(runtime).hexdigest(),'bundle_sha256':bundle,'signature_verified':True,'app_cdhash':'a'*40,
             'owned_processes_stopped':True,'model_files':core.model_file_pins(self.root),'cases':[],'budget':{'lease':{'admitted':True,'seconds':600,'floor_plus_other_reserve_bytes':40_000_000_000},
             'wall_seconds':200,'peak_owned_rss_bytes':2_000_000_000,'minimum_free_memory_percent':40,'minimum_free_disk_bytes':41_000_000_000}}
        self.recipes=cases
        for recipe in cases:
            expected=recipe['expect'];text=recipe.get('speech',' '.join(expected.get('terms',[])))
            if expected.get('minimum_thank_you_count'):text='Thank you. Thank you. '+text
            if expected.get('protected_correction'):text=recipe['correction']
            if expected.get('mixed_evidence'):text='Two synthetic speakers'
            count=expected.get('reading_turns',1);rows=[]
            for i in range(count if text else 0):
                rows.append({'id':'row-'+str(i),'text':text if i==0 else 'additional request','language':expected.get('language','en'),
                  'speaker_candidates':['a','b'] if expected.get('mixed_evidence') else ['a'],'voice_eligible':False,
                  'protected_fields':['text'] if expected.get('protected_correction') else [],
                  'language_detection':{'reason':'detected' if recipe.get('preceding_context_case') else 'recent_context'}})
            job={'id':'f'*32,'last_fast_sequence':1,'document':{'segments':rows,'provenance':{'kind':'local_inference'}},
                 'status':'ready','refinement_status':'complete','refinement_unresolved':[],'rolling_refinement':{'completed_sample':1,'cancelled':False}}
            provisional=copy.deepcopy(job);provisional.update(status='paused');provisional['rolling_refinement']['cancelled']=True
            provisional['pause_flush']={'request_id':'synthetic-pause','state':'complete','through_sample':1,
                'received_sample':1,'available_sample':0,'speech_observed_sample':0,'fast_sequence':1,
                'deferred_audio':[{'start_sample':0,'end_sample':1}]}
            synthesis={'recipe_id':recipe['id']}
            pcm_hash=hashlib.sha256(b'\0\0').hexdigest()
            preceding={'id':job['id'],'language_epoch':0,'document':{'segments':[{'start':0,'end':1,'language_epoch':0,'language':'en','text':'Successful English predecessor','language_detection':{'reason':'detected'}}]}}
            trace={'job_id':job['id'],'pause_request_id':'synthetic-pause','input_frames':1,'slice_start_sample':32000,'saved_audio':{'frames':1,'pcm_sha256':pcm_hash,'expected_pcm_sha256':pcm_hash},'synthesis':synthesis,'provisional':provisional,'refined':job,
                   'edit':{'id':'row-0','machine_revision':1},'preceding':{'case':recipe.get('preceding_context_case'),'job_id':job['id'],'context':{'language':'en','end_sample':16000},'observed':preceding}}
            folder=self.evidence/recipe['id'];folder.mkdir();(folder/'trace.json').write_text(json.dumps(trace));(folder/'synthesis.json').write_text(json.dumps(synthesis))
            with wave.open(str(folder/'input.wav'),'wb') as pcm:pcm.setnchannels(1);pcm.setsampwidth(2);pcm.setframerate(16000);pcm.writeframes(b'\0\0')
            case={'id':recipe['id'],'partition':recipe['partition'],'groups':recipe['groups'],'status':'passed','stages':['provisional','refined'],'assertions':core.evaluate(recipe,trace)}
            for key,name in (('input_wav','input.wav'),('saved_wav','input.wav'),('trace','trace.json'),('synthesis','synthesis.json')):
                path=folder/name;case[key]={'file':str(path.relative_to(self.evidence)),'bytes':path.stat().st_size,'sha256':core.digest_file(path)}
            self.report['cases'].append(case)
        self.save()
    def tearDown(self):self.temporary.cleanup()
    def save(self):
        self.report_path.write_text(json.dumps(self.report));self.manifest_path.write_text(json.dumps(self.manifest))
    def validate(self):self.save();return core.validate(self.report_path,self.manifest_path,root=self.root)
    def test_complete_contract_is_accepted_not_model_accuracy(self):
        self.assertEqual(len(self.validate()['cases']),15)
    def canonical_negative_trace(self,uncertain=0):
        # Entirely fabricated receipt contract, never model/candidate evidence.
        sys.path.insert(0,str(SOURCE/'speakerdesk'))
        from admission_receipt import execution
        recipe=core.suite()[0][0]
        trace=json.loads((self.evidence/recipe['id']/'trace.json').read_text())
        identity=execution(trace['job_id'],'b'*32)
        empty=hashlib.sha256(b'').hexdigest();whole=hashlib.sha256(b'\0\0').hexdigest()
        trace['admission_pcm']={'provisional':empty,'refined':whole}
        for stage in ('provisional','refined'):
            job=trace[stage];job.update(canonical_utterances=True,last_fast_sequence=0,admission_execution=identity)
            job['document']['provenance']['kind']='pending_inference'
        pause=trace['provisional']['pause_flush'];pause['fast_sequence']=0
        pause['admission_receipt']={**identity,'phase':'pause','request_id':trace['pause_request_id'],
            'inspection_state':'observed_prefix','start_sample':0,'end_sample':0,'received_sample':1,
            'closed':False,'audio_encoding':'pcm_s16le','pcm_sha256':empty,
            'speech_samples':0,'uncertain_samples':0,'negative_constant_samples':0,'model_negative_samples':0,'decision':'no_speech'}
        final=trace['refined'];final.update(canonical_observed_sample=1,canonical_uncertain_samples=uncertain,
            capture_inspection_request={'through_sample':1,'request_id':'fixture-stop'},rolling_sources={})
        final['capture_admission']={**identity,'phase':'stop','request_id':'fixture-stop',
            'inspection_state':'observed_prefix','start_sample':0,'end_sample':1,'received_sample':1,
            'closed':True,'audio_encoding':'pcm_s16le','pcm_sha256':whole,
            'speech_samples':0,'uncertain_samples':uncertain,'negative_constant_samples':1-uncertain,'model_negative_samples':0,
            'decision':'uncertain' if uncertain else 'no_speech'}
        if uncertain:final['refinement_status']='unresolved';final['rolling_refinement']['completed_sample']=0
        return recipe,trace
    def test_canonical_no_words_requires_bound_inspection_instead_of_fake_text_revision(self):
        recipe,trace=self.canonical_negative_trace()
        self.assertTrue(all(core.evaluate(recipe,trace).values()))
        for stage,receipt_key in [('provisional','pause_flush'),('refined','capture_admission')]:
            broken=copy.deepcopy(trace)
            if stage=='provisional':broken[stage][receipt_key].pop('admission_receipt')
            else:broken[stage].pop(receipt_key)
            self.assertFalse(core.evaluate(recipe,broken)[stage+'_observed'])
        for key,value in [('inspection_state','failed'),('model_revision','wrong'),('execution_id','c'*32),
                          ('pcm_sha256','0'*64),('end_sample',0),('closed',False),
                          ('negative_constant_samples',2),('decision','speech'),('schema_version',True)]:
            broken=copy.deepcopy(trace);broken['refined']['capture_admission'][key]=value
            with self.subTest(key=key):self.assertFalse(core.evaluate(recipe,broken)['negative_audio_inspected'])
        trace['refined']['document']['segments']=[{'text':'Invented thank you'}]
        self.assertFalse(core.evaluate(recipe,trace)['refined_no_words'])
    def test_uncertain_negative_receipt_can_never_claim_complete_or_bypass_speech_gate(self):
        recipe,trace=self.canonical_negative_trace(1)
        self.assertTrue(core.evaluate(recipe,trace)['negative_audio_inspected'])
        trace['refined']['refinement_status']='complete'
        self.assertFalse(core.evaluate(recipe,trace)['negative_audio_inspected'])
        trace['refined']['refinement_status']='unresolved'
        speech_recipe=copy.deepcopy(recipe);speech_recipe['expect']={'terms':['blue']}
        checks=core.evaluate(speech_recipe,trace)
        self.assertFalse(checks['refinement_executed']);self.assertFalse(checks['refined_coverage'])
    def test_rehashed_canonical_receipt_pcm_is_rechecked_from_saved_audio(self):
        recipe,trace=self.canonical_negative_trace()
        directory=self.evidence/recipe['id'];(directory/'trace.json').write_text(json.dumps(trace))
        case=self.report['cases'][0];case['assertions']=core.evaluate(recipe,trace)
        case['trace']['bytes']=(directory/'trace.json').stat().st_size
        case['trace']['sha256']=core.digest_file(directory/'trace.json')
        self.assertTrue(self.validate())
        trace['admission_pcm']['refined']='1'*64;trace['refined']['capture_admission']['pcm_sha256']='1'*64
        (directory/'trace.json').write_text(json.dumps(trace));case['assertions']=core.evaluate(recipe,trace)
        case['trace']['bytes']=(directory/'trace.json').stat().st_size
        case['trace']['sha256']=core.digest_file(directory/'trace.json')
        with self.assertRaisesRegex(ValueError,'Admission receipt PCM'):self.validate()

    def test_pause_receipt_accepts_actual_lookahead_but_rejects_stale_or_unobserved_endpoint(self):
        job={'last_fast_sequence':2,'pause_flush':{'request_id':'current','state':'complete','through_sample':16000,
            'received_sample':16000,'available_sample':12000,'speech_observed_sample':15872,'fast_sequence':2,
            'deferred_audio':[{'start_sample':12000,'end_sample':16000}]}}
        self.assertTrue(core.pause_acknowledged(job,'current',16000))
        self.assertFalse(core.pause_acknowledged(job,'old',16000))
        self.assertFalse(core.pause_acknowledged(job,'current',32000))
        for field,value in [('state','pending'),('available_sample',16001),('speech_observed_sample',11000),('deferred_audio',[]),('fast_sequence',1)]:
            changed=copy.deepcopy(job);changed['pause_flush'][field]=value
            self.assertFalse(core.pause_acknowledged(changed,'current',16000))
    def test_raw_evaluation_binds_pause_to_the_post_request(self):
        recipe=core.suite()[0][0]
        trace=json.loads((self.evidence/recipe['id']/'trace.json').read_text())
        self.assertTrue(core.evaluate(recipe,trace)['provisional_pause_acknowledged'])
        for identity in (None,'different-post'):
            trace['pause_request_id']=identity
            self.assertFalse(core.evaluate(recipe,trace)['provisional_pause_acknowledged'])
    def test_missing_report_blocks_before_external_request(self):
        self.report_path.unlink()
        with patch.object(promote_release,'api') as network:
            with self.assertRaisesRegex(ValueError,'report missing'):
                promote_release.prepare(self.report_path,self.manifest_path,self.base/'no-qa',self.base/'no-notes')
            network.assert_not_called()
    def test_source_mock_skipped_and_partial_receipts_fail_closed(self):
        for key,value in [('scope','source_replay'),('models_executed',False),('complete',False),('status','skipped'),('model_substitution',True),('owned_processes_stopped',False)]:
            original=self.report[key];self.report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
            self.report[key]=original
    def test_changed_package_source_model_suite_and_harness_fail_closed(self):
        for key in ('source_commit','package_sha256','runtime_sha256','bundle_sha256','model_pins','suite_sha256','harness_sha256'):
            original=self.report[key];self.report[key]='stale'
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
            self.report[key]=original
    def test_missing_stage_case_or_assertion_fails_closed(self):
        original=copy.deepcopy(self.report)
        for change in ('case','stage','assertion'):
            self.report=copy.deepcopy(original)
            if change=='case':self.report['cases'].pop()
            elif change=='stage':self.report['cases'][0]['stages']=['refined']
            else:self.report['cases'][0]['assertions']={'fake_pass':True}
            with self.subTest(change=change),self.assertRaises(ValueError):self.validate()
    def test_rehashed_hallucination_cannot_hide_behind_passed_assertions(self):
        case=self.report['cases'][0];path=self.evidence/case['trace']['file'];trace=json.loads(path.read_text())
        trace['refined']['document']['segments']=[{'text':'Thank you.'}];path.write_text(json.dumps(trace))
        case['trace'].update(bytes=path.stat().st_size,sha256=core.digest_file(path))
        with self.assertRaisesRegex(ValueError,'Raw core evidence fails'):self.validate()
    def test_failed_resource_budget_blocks_promotion(self):
        self.report['budget']['peak_owned_rss_bytes']=5*1024**3
        with self.assertRaisesRegex(ValueError,'Memory budget failed'):self.validate()
    def test_holdouts_frozen_and_short_phrases_checked(self):
        case=next(r for r in self.recipes if r['id']=='holdout_en_lamp_continuity_v1')
        trace={'provisional':{'document':{'segments':[{'text':'Move small lamp closer read final line'}]}},'refined':{}}
        self.assertFalse(core.evaluate(case,trace)['provisional_phrase_0'])
        brief=next(r for r in self.recipes if r['id']=='holdout_en_quiet_single_blue_v1')
        trace['provisional']['document']['segments']=[{'text':'Blue. You.'}]
        self.assertFalse(core.evaluate(brief,trace)['provisional_exact_text'])
    def test_backend_teardown_error_still_closes_supervisor(self):
        from unittest.mock import Mock
        backend=Mock();backend.close.side_effect=OSError('fixture teardown error')
        budget=Mock();budget.close.return_value=True
        stopped,errors=packaged_replay.cleanup(backend,budget)
        self.assertFalse(stopped);self.assertTrue(errors);budget.close.assert_called_once()
    def test_constructor_failure_still_closes_supervisor(self):
        from unittest.mock import Mock
        budget=Mock();budget.close.return_value=True
        self.assertEqual(packaged_replay.cleanup(None,budget),(True,[]));budget.close.assert_called_once()
    def test_denied_owned_metrics_never_count_as_zero_memory(self):
        from unittest.mock import Mock
        process=Mock(pid=12345)
        for result in (subprocess.CompletedProcess([],1,'','Operation not permitted'),subprocess.CompletedProcess([],0,'','')):
            with patch.object(packaged_replay.subprocess,'run',return_value=result),patch.object(packaged_replay.os,'killpg'):
                with self.assertRaisesRegex(ValueError,'metrics unavailable'):packaged_replay.owned_rss(process)
        empty=subprocess.CompletedProcess([],1,'','')
        with patch.object(packaged_replay.subprocess,'run',return_value=empty),patch.object(packaged_replay.os,'killpg',side_effect=ProcessLookupError):
            self.assertEqual(packaged_replay.owned_rss(process),0)
        measured=subprocess.CompletedProcess([],0,'12345 1024\n12345 2048\n','')
        with patch.object(packaged_replay.subprocess,'run',return_value=measured):
            self.assertEqual(packaged_replay.owned_rss(process),3072*1024)
    def test_existing_lightweight_and_annotated_tags_must_match_source(self):
        plan={'tag':'v0.0.0','source_commit':'a'*40}
        with patch.object(promote_release,'api',return_value=[{'ref':'refs/tags/v0.0.0','object':{'type':'commit','sha':'b'*40}}]):
            with self.assertRaisesRegex(ValueError,'different source'):promote_release.check_tag(plan)
        with patch.object(promote_release,'api',side_effect=[[{'ref':'refs/tags/v0.0.0','object':{'type':'tag','sha':'c'*40}}],{'object':{'type':'commit','sha':'a'*40}}]):
            promote_release.check_tag(plan,required=True)
    def test_trusted_signer_archive_rejects_relabelled_manifest(self):
        manifest=copy.deepcopy(self.manifest);manifest.update(producer_run_id=11,input_artifact_id=12,input_artifact_sha256='1'*64,
                 signer_run_id='21',signer_commit='2'*40)
        manifest['files'].append({'filename':'Speakerdesk_0.0.0_AppleSilicon.dmg','bytes':3,'sha256':hashlib.sha256(b'dmg').hexdigest()})
        archive=self.base/'original-signer.zip'
        with zipfile.ZipFile(archive,'w') as zipped:
            zipped.writestr('artifact-manifest.json',json.dumps(manifest));zipped.writestr('SHA256SUMS','synthetic contract sums')
            zipped.write(self.evidence/self.manifest['files'][0]['filename'],self.manifest['files'][0]['filename'])
            zipped.writestr(manifest['files'][1]['filename'],b'dmg')
        def run(repo_id,repo,path,sha):return {'repository':{'id':repo_id,'full_name':repo},'head_repository':{'id':repo_id},'path':path,
             'head_branch':'main','event':'workflow_dispatch','status':'completed','conclusion':'success','head_sha':sha}
        source_origin={'id':11,'head_sha':self.head,'repository_id':1406057260,'head_repository_id':1406057260,'head_branch':'main'}
        signer_origin={'id':21,'head_sha':'2'*40,'repository_id':1406130411,'head_repository_id':1406130411,'head_branch':'main'}
        replies=[run(1406057260,promote_release.REPOSITORY,'.github/workflows/apple-build.yml',self.head),
                 run(1406130411,'TheGreenCedar/AppleRelease','.github/workflows/sign.yml','2'*40),
                 {'digest':'sha256:'+'1'*64,'workflow_run':source_origin},
                 {'artifacts':[{'name':'Speakerdesk-'+self.head+'-notarized-candidate-21','expired':False,'workflow_run':signer_origin,'digest':'sha256:'+core.digest_file(archive)}]}]
        with patch.object(promote_release,'api',side_effect=copy.deepcopy(replies)):
            promote_release.verify_signing(manifest,archive)
        relabelled=copy.deepcopy(manifest);relabelled['version']='9.9.9'
        with patch.object(promote_release,'api',side_effect=copy.deepcopy(replies)):
            with self.assertRaisesRegex(ValueError,'manifest differs'):promote_release.verify_signing(relabelled,archive)

if __name__=='__main__':unittest.main()
