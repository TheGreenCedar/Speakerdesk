"""Authenticated signer inventory and client-key binding, with synthetic bytes."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import promote_release as promotion
import windows_signing


class UpdaterReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.stem='Speakerdesk_0.6.2_AppleSilicon'
        self.key=base64.b64encode(b'Public contract fixture, not a signing key').decode()
        self.signature=base64.b64encode(b'Synthetic signature transport, not cryptographic proof').decode()
        payloads={self.stem+'.dmg':b'dmg',self.stem+'.app.zip':b'appzip',
                  self.stem+'.app.tar.gz':b'final archive fixture',self.stem+'.app.tar.gz.sig':self.signature.encode()}
        self.manifest={'version':'0.6.2','source_commit':'a'*40,'producer_run_id':11,'input_artifact_id':12,
                       'input_artifact_sha256':'1'*64,'signer_run_id':'21','signer_commit':'2'*40,
                       'files':[{'filename':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for name,value in payloads.items()],
                       'updater':{'platform':'darwin-aarch64','filename':self.stem+'.app.tar.gz',
                                  'signature_filename':self.stem+'.app.tar.gz.sig','signature':self.signature,
                                  'public_key':self.key,'require_signed_version':True}}
        for name,value in payloads.items():(self.root/name).write_bytes(value)
        self.config={'plugins':{'updater':{'pubkey':self.key,'requireSignedVersion':True,'allowDowngrades':False}}}

    def tearDown(self):self.temporary.cleanup()

    def native_candidate(self):
        self.manifest.update(source_repository=promotion.REPOSITORY,runtime_kind='native',signing='developer-id',notarized=True,
                             notarization=[{'kind':'app','status':'Accepted'}, {'kind':'dmg','status':'Accepted'}])
        self.config['version']=self.manifest['version']
        config=self.root/'desktop/src-tauri';config.mkdir(parents=True)
        (config/'tauri.conf.json').write_text(json.dumps(self.config))
        manifest=self.root/'artifact-manifest.json';manifest.write_text(json.dumps(self.manifest))
        (self.root/'SHA256SUMS').write_text(''.join(f'{x["sha256"]}  {x["filename"]}\n' for x in self.manifest['files']))
        verification=self.root/'native-verification.md'
        verification.write_text('Synthetic operator QA fixture; not real inference or app proof.\n')
        qa={'passed':True,'source_commit':self.manifest['source_commit'],'version':self.manifest['version'],
            'app_archive_sha256':promotion.digest_file(self.root/(self.stem+'.app.zip')),
            'verification':str(verification),'verification_sha256':promotion.digest_file(verification)}
        qa_path=self.root/'native-qa.json';qa_path.write_text(json.dumps(qa))
        notes=self.root/'notes.md';notes.write_text('Source: '+self.manifest['source_commit'])
        return manifest,qa_path,notes,self.root/'signer.zip'

    def test_native_preparation_binds_qa_and_payloads_before_authenticating_signer(self):
        args=self.native_candidate()
        with patch.object(promotion,'ROOT',self.root),patch.object(promotion,'verify_signing') as signer:
            plan=promotion.prepare(*args)
        signer.assert_called_once_with(self.manifest,args[-1])
        self.assertEqual(plan['tag'],'v0.6.2')
        self.assertEqual(set(plan['files']),promotion.payload_names(self.manifest)|{'artifact-manifest.json','SHA256SUMS'})

    def test_native_qa_for_another_candidate_or_changed_report_fails_before_network(self):
        args=self.native_candidate();qa=json.loads(args[1].read_text())
        changes={'passed':False,'source_commit':'b'*40,'version':'0.7.1','app_archive_sha256':'f'*64,'verification_sha256':'f'*64}
        with patch.object(promotion,'ROOT',self.root),patch.object(promotion,'verify_signing',side_effect=AssertionError('No network before local acceptance')):
            for key,value in changes.items():
                with self.subTest(key=key):
                    args[1].write_text(json.dumps(dict(qa,**{key:value})))
                    with self.assertRaises(ValueError):promotion.prepare(*args)

    def test_native_promotion_rejects_modified_bytes_and_incomplete_notarization(self):
        args=self.native_candidate()
        with patch.object(promotion,'ROOT',self.root),patch.object(promotion,'verify_signing',side_effect=AssertionError('No network before local acceptance')):
            for kind in ('payload','checksums','notarization','source','inventory','symlink'):
                with self.subTest(kind=kind):
                    payload=self.root/(self.stem+'.dmg');original=payload.read_bytes()
                    sums=(self.root/'SHA256SUMS').read_text();manifest=copy.deepcopy(self.manifest)
                    if kind=='payload':payload.write_bytes(b'modified')
                    elif kind=='checksums':(self.root/'SHA256SUMS').write_text('wrong checksums')
                    elif kind=='notarization':manifest['notarization'].pop()
                    elif kind=='source':manifest['source_repository']='another/Speakerdesk'
                    elif kind=='inventory':manifest['files'].append(manifest['files'][0])
                    elif kind=='symlink':
                        target=self.root/'other.dmg';target.write_bytes(original);payload.unlink();payload.symlink_to(target)
                    args[0].write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):promotion.prepare(*args)
                    if payload.is_symlink():payload.unlink()
                    payload.write_bytes(original);(self.root/'SHA256SUMS').write_text(sums)

    def authenticated_archive(self,manifest,extra=None):
        archive=self.root/'signer.zip'
        with zipfile.ZipFile(archive,'w') as zipped:
            zipped.writestr('artifact-manifest.json',json.dumps(manifest));zipped.writestr('SHA256SUMS','fixture checksums')
            for item in manifest['files']:zipped.write(self.root/item['filename'],item['filename'])
            if extra:zipped.writestr(extra,b'Unexpected authentic signer member')
        def run(repo_id,repo,path,sha):return {'repository':{'id':repo_id,'full_name':repo},'head_repository':{'id':repo_id},'path':path,
             'head_branch':'main','event':'workflow_dispatch','status':'completed','conclusion':'success','head_sha':sha}
        producer={'id':11,'head_sha':'a'*40,'repository_id':1406057260,'head_repository_id':1406057260,'head_branch':'main'}
        signer={'id':21,'head_sha':'2'*40,'repository_id':1406130411,'head_repository_id':1406130411,'head_branch':'main'}
        replies=[run(1406057260,promotion.REPOSITORY,'.github/workflows/apple-build.yml','a'*40),
                 run(1406130411,'TheGreenCedar/AppleRelease','.github/workflows/sign.yml','2'*40),
                 {'digest':'sha256:'+'1'*64,'workflow_run':producer},
                 {'artifacts':[{'name':'Speakerdesk-'+'a'*40+'-notarized-candidate-21','expired':False,
                                'workflow_run':signer,'digest':'sha256:'+promotion.digest_file(archive)}]}]
        with patch.object(promotion,'api',side_effect=replies):promotion.verify_signing(manifest,archive)

    def test_canonical_identity_remains_bound_when_repository_is_public(self):
        repo={'id':promotion.REPOSITORY_ID,'full_name':promotion.REPOSITORY,'private':False}
        promotion.verify_repository(repo)
        for field,value in [('id',1),('full_name','another/Speakerdesk')]:
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'identity differs'):
                promotion.verify_repository(dict(repo,**{field:value}))

    def test_exact_six_member_updater_signer_archive_is_accepted(self):
        self.authenticated_archive(self.manifest)
        promotion.verify_updater_files(self.manifest,self.root,self.config)

    def test_authentic_archive_still_rejects_missing_duplicate_or_extra_members(self):
        for extra in (self.stem+'.app.tar.gz.sig','unapproved.pkg'):
            with self.subTest(extra=extra),self.assertRaisesRegex(ValueError,'inventory'):
                self.authenticated_archive(self.manifest,extra)
        missing=copy.deepcopy(self.manifest);missing['files'].pop()
        with self.assertRaisesRegex(ValueError,'inventory'):self.authenticated_archive(missing)

    def test_legacy_signer_contract_remains_exactly_four_members(self):
        legacy=copy.deepcopy(self.manifest);legacy.pop('updater');legacy['files']=legacy['files'][:2]
        legacy['version']='0.6.1'
        for item in legacy['files']:
            original=self.root/item['filename'];item['filename']=item['filename'].replace('0.6.2','0.6.1')
            (self.root/item['filename']).write_bytes(original.read_bytes())
        self.authenticated_archive(legacy)
        promotion.verify_updater_files(legacy,self.root,{'plugins':{}})
        with self.assertRaisesRegex(ValueError,'requires signed updater'):
            promotion.verify_updater_files(legacy,self.root,self.config)

    def test_expanded_062_release_refuses_unconfigured_or_missing_updater(self):
        manifest=copy.deepcopy(self.manifest);manifest.pop('updater')
        for config in ({'plugins':{}},self.config):
            with self.subTest(config=config),self.assertRaisesRegex(ValueError,'0.6.2.*requires configured'):
                promotion.verify_updater_files(manifest,self.root,config)
        with self.assertRaisesRegex(ValueError,'0.6.2.*requires configured'):
            promotion.verify_updater_files(self.manifest,self.root,{'plugins':{}})

    def test_final_signature_bytes_and_compiled_public_key_must_match(self):
        (self.root/self.manifest['updater']['signature_filename']).write_text(self.signature+'tampered')
        with self.assertRaisesRegex(ValueError,'signature metadata differs'):
            promotion.verify_updater_files(self.manifest,self.root,self.config)
        (self.root/self.manifest['updater']['signature_filename']).write_text(self.signature)
        for key in (base64.b64encode(b'Another public fixture').decode(),):
            config=copy.deepcopy(self.config);config['plugins']['updater']['pubkey']=key
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'verification key differs'):
                promotion.verify_updater_files(self.manifest,self.root,config)

    def test_update_version_architecture_and_signed_version_policy_are_required(self):
        for field,value in [('platform','darwin-x86_64'),('filename','Speakerdesk_0.6.1_AppleSilicon.app.tar.gz'),('require_signed_version',False)]:
            manifest=copy.deepcopy(self.manifest);manifest['updater'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'platform/version binding'):
                promotion.verify_updater_files(manifest,self.root,self.config)
        config=copy.deepcopy(self.config);config['plugins']['updater']['allowDowngrades']=True
        with self.assertRaisesRegex(ValueError,'version policy'):
            promotion.verify_updater_files(self.manifest,self.root,config)

    def publication_fixture(self):
        args=self.native_candidate()
        with patch.object(promotion,'ROOT',self.root),patch.object(promotion,'verify_signing'):
            plan=promotion.prepare(*args)
        release={'id':41,'tag_name':plan['tag'],'target_commitish':plan['source_commit'],
                 'prerelease':False,'draft':True,'body':plan['notes'],'published_at':None,
                 'html_url':'https://github.com/'+promotion.REPOSITORY+'/releases/tag/'+plan['tag'],
                 'assets':[{'id':index,'name':name,'state':'uploaded','size':item['bytes'],
                            'digest':'sha256:'+item['sha256']} for index,(name,item) in enumerate(plan['files'].items(),1)]}
        state={'created':False,'published':False,'before':[],'listed':[copy.deepcopy(release)],
               'release':release,'refs':[],'created_sha':plan['source_commit'],'mutations':[]}
        base='repos/'+promotion.REPOSITORY
        def command(arguments,**kwargs):
            if arguments[:3]==['gh','release','create']:
                self.assertIn('--draft',arguments);state['created']=True;state['mutations'].append('draft')
                return subprocess.CompletedProcess(arguments,0,release['html_url']+'\n','')
            self.assertEqual(arguments[:2],['gh','api'])
            endpoint=arguments[2]
            method=arguments[arguments.index('--method')+1] if '--method' in arguments else 'GET'
            body=json.loads(kwargs['input']) if kwargs.get('input') else None
            if method=='GET':
                if endpoint==base:reply={'id':promotion.REPOSITORY_ID,'full_name':promotion.REPOSITORY}
                elif endpoint==base+'/commits/main':reply={'sha':plan['source_commit']}
                elif endpoint==base+'/git/matching-refs/tags/'+plan['tag']:reply=state['refs']
                elif endpoint==base+'/releases?per_page=100':reply=state['listed'] if state['created'] else state['before']
                elif endpoint==base+'/releases/'+str(release['id']):reply=state['release']
                elif endpoint==base+'/releases/tags/'+plan['tag']:
                    return subprocess.CompletedProcess(arguments,1,'{"message":"Not Found","status":"404"}','gh: Not Found (HTTP 404)')
                else:raise AssertionError('Unexpected read: '+endpoint)
            elif method=='POST' and endpoint==base+'/git/refs':
                self.assertEqual(body,{'ref':'refs/tags/'+plan['tag'],'sha':plan['source_commit']})
                state['mutations'].append('tag')
                state['refs']=[{'ref':body['ref'],'object':{'type':'commit','sha':state['created_sha']}}]
                reply=state['refs'][0]
            elif method=='PATCH' and endpoint==base+'/releases/'+str(release['id']):
                self.assertEqual(body,{'draft':False,'prerelease':False,'make_latest':'true'})
                self.assertEqual(state['refs'],[{'ref':'refs/tags/'+plan['tag'],'object':{'type':'commit','sha':plan['source_commit']}}])
                state['mutations'].append('publish');state['published']=True
                reply=dict(state['release'],draft=False,published_at='2026-10-12T00:00:00Z')
            else:raise AssertionError('Unexpected mutation: '+method+' '+endpoint)
            return subprocess.CompletedProcess(arguments,0,json.dumps(reply),'')
        def download(asset_id):
            item=plan['files'][release['assets'][asset_id-1]['name']]
            return item['bytes'],item['sha256']
        return plan,state,command,download

    def test_publish_finds_created_draft_by_id_and_creates_verified_tag_before_publication(self):
        plan,state,command,download=self.publication_fixture()
        with patch.object(promotion.subprocess,'run',side_effect=command),patch.object(promotion,'downloaded_digest',side_effect=download) as downloads:
            result=promotion.publish(plan)
        self.assertTrue(result['downloads_verified'])
        self.assertEqual(state['mutations'],['draft','tag','publish'])
        self.assertEqual(downloads.call_count,6)

    def test_publish_preserves_an_existing_exact_source_tag(self):
        plan,state,command,download=self.publication_fixture()
        state['refs']=[{'ref':'refs/tags/'+plan['tag'],'object':{'type':'commit','sha':plan['source_commit']}}]
        with patch.object(promotion.subprocess,'run',side_effect=command),patch.object(promotion,'downloaded_digest',side_effect=download):
            promotion.publish(plan)
        self.assertEqual(state['mutations'],['draft','publish'])

    def test_draft_identity_metadata_and_tag_fail_closed_before_publication(self):
        plan,state,command,download=self.publication_fixture()
        original=copy.deepcopy(state)
        cases={'existing draft':'Existing release/draft','missing draft':'Created draft identity',
               'ambiguous draft':'Created draft identity','reused identity':'Created draft identity',
               'invalid listed id':'Created draft identity','wrong existing tag':'Existing release tag points',
               'wrong fetched id':'Created draft identity','wrong fetched tag':'Release metadata',
               'wrong uploaded digest':'Uploaded release bytes','already published':'Unexpected release visibility',
               'wrong created tag':'Existing release tag points'}
        for case,message in cases.items():
            with self.subTest(case=case):
                state.clear();state.update(copy.deepcopy(original))
                if case=='existing draft':state['before']=[state['release']]
                elif case=='missing draft':state['listed']=[]
                elif case=='ambiguous draft':state['listed'].append(dict(state['release'],id=42))
                elif case=='reused identity':state['before']=[dict(state['release'],tag_name='v0.6.1')]
                elif case=='invalid listed id':state['listed'][0]['id']='41'
                elif case=='wrong existing tag':state['refs']=[{'ref':'refs/tags/'+plan['tag'],'object':{'type':'commit','sha':'b'*40}}]
                elif case=='wrong fetched id':state['release']['id']=42
                elif case=='wrong fetched tag':state['release']['tag_name']='v0.6.1'
                elif case=='wrong uploaded digest':state['release']['assets'][0]['digest']='sha256:'+'f'*64
                elif case=='already published':state['release']['draft']=False
                elif case=='wrong created tag':state['created_sha']='b'*40
                with patch.object(promotion.subprocess,'run',side_effect=command),patch.object(promotion,'downloaded_digest',side_effect=download) as downloads:
                    with self.assertRaisesRegex(ValueError,message):promotion.publish(plan)
                self.assertFalse(state['published']);downloads.assert_not_called()
                if case!='wrong created tag':self.assertNotIn('tag',state['mutations'])
                if case in ('existing draft','wrong existing tag'):self.assertFalse(state['created'])


class WindowsPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(promotion, 'ROOT', self.root).start()
        patch.object(windows_signing, 'ROOT', self.root).start()
        self.network = patch.object(promotion, 'api', side_effect=AssertionError('Windows preparation must remain local')).start()
        config = self.root / 'desktop/src-tauri'; config.mkdir(parents=True)
        self.version, self.source = '0.7.7', 'a' * 40
        self.key = base64.b64encode(b'Public synthetic updater key').decode()
        (config / 'tauri.conf.json').write_text(json.dumps(dict(version=self.version,
            plugins=dict(updater=dict(pubkey=self.key, requireSignedVersion=True, allowDowngrades=False)))))
        (config / 'tauri.windows.conf.json').write_text('{}')
        self.candidate = self.root / 'candidate'; self.candidate.mkdir()
        name = f'Speakerdesk_{self.version}_x64-setup.exe'
        self.installer = self.candidate / name
        self.installer.write_bytes(b'Synthetic installer bytes; not native installation proof')
        self.signature = self.candidate / (name + '.sig')
        self.signature.write_text(base64.b64encode(b'Synthetic signature transport').decode())
        receipts = self.candidate / 'signing'; (receipts / 'payloads').mkdir(parents=True)
        payloads = {}
        for role, filename in dict(capture='speakerdesk-capture.exe', echo='speakerdesk_echo.dll',
                app='speakerdesk.exe', uninstaller='uninstall.exe', installer=name).items():
            payload = receipts / 'payloads' / filename
            payload.write_bytes(self.installer.read_bytes() if role == 'installer' else role.encode())
            digest = promotion.digest_file(payload)
            (receipts / (role + '.json')).write_text(json.dumps(dict(role=role, disposition='owned_signed',
                sha256=digest, authenticode_status='Valid', timestamped=True, publisher_subject='CN=Albert Najjar')))
            if role != 'installer': payloads[filename] = digest
        runtime = self.root / 'crates/models/src/windows-runtime.json'; runtime.parent.mkdir(parents=True)
        runtime.write_text(json.dumps({'files': {'onnxruntime.dll': {'sha256': 'c' * 64}}}))
        payloads.update({'runtime/onnxruntime.dll': 'c' * 64,
            'speakerdesk-capture.exe.capability.json': 'd' * 64, 'THIRD_PARTY_NOTICES.txt': 'e' * 64})
        self.manifest = dict(platform='windows-x86_64', version=self.version, source_commit=self.source,
            source_dirty=False, public_ready=False, authenticode_status='Valid', installed_payloads=payloads,
            installed_payloads_observed=False, signing=dict(kind='azure-artifact-signing', receipts='signing', payloads_observed_at_hook=True),
            installer=dict(name=name, bytes=self.installer.stat().st_size, sha256=promotion.digest_file(self.installer)),
            updater_signature=dict(filename=name, signature_filename=self.signature.name, signature=self.signature.read_text(),
                verified=True, require_signed_version=True, **{k:v for k,v in windows_signing.updater_configuration().items() if k != 'version'}))
        self.manifest_path = self.candidate / 'artifact-manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        # These are operator-report contract fixtures, not claims of native trust or cryptographic success.
        self.install_report = self.root / 'installed-report.json'
        self.install_report.write_text(json.dumps(dict(kind='signed-standard-user-install', passed=True,
            standard_user=True, installer_sha256=self.manifest['installer']['sha256'], installed_payloads=payloads,
            authenticode_verified=True, save_reopen=True, clean_shutdown=True, uninstall_reinstall=True, data_preserved=True)))
        self.updater_report = self.root / 'updater-report.json'
        self.updater_report.write_text(json.dumps(dict(kind='nsis-installer-updater-verification',
            payload_sha256=self.manifest['installer']['sha256'], signature_sha256=promotion.digest_file(self.signature),
            version=self.version, public_key=self.key, signature_key_version_verified=True, payload_unchanged=True)))
        self.qa = dict(passed=True, platform='windows-x86_64', version=self.version, source_commit=self.source,
            artifact_manifest_sha256=promotion.digest_file(self.manifest_path),
            verification=str(self.install_report), verification_sha256=promotion.digest_file(self.install_report),
            updater_verification=str(self.updater_report), updater_verification_sha256=promotion.digest_file(self.updater_report))
        self.qa_path = self.root / 'native-qa.json'; self.qa_path.write_text(json.dumps(self.qa))
        self.notes = self.root / 'notes.md'; self.notes.write_text('Source: ' + self.source)

    def prepare(self):
        return promotion.prepare(self.manifest_path, self.qa_path, self.notes, None)

    def test_local_windows_preparation_binds_signed_bytes_and_operator_reports_without_network(self):
        result = self.prepare()
        self.assertEqual(result['provenance'], 'local-operator-controlled')
        self.assertFalse(result['publication_enabled'])
        self.assertEqual(result['source_commit'], self.source)
        self.assertEqual(set(result['files']), {self.installer.name, self.signature.name, 'artifact-manifest.json'})
        self.network.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Windows publication is disabled'):
            promotion.publish(result)
        self.network.assert_not_called()

    def test_windows_preparation_rejects_unqualified_candidates_and_changed_evidence(self):
        originals = {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        cases = [
            (self.manifest_path, dict(self.manifest, source_dirty=True)),
            (self.manifest_path, dict(self.manifest, qualification_fixture='Renamed helper, NOT an installer')),
            (self.manifest_path, dict(self.manifest, installed_payloads={name:value for name,value in
                self.manifest['installed_payloads'].items() if name != 'runtime/onnxruntime.dll'})),
            (self.manifest_path, dict(self.manifest, updater_signature=dict(self.manifest['updater_signature'], pending=True, verified=False))),
            (self.manifest_path, dict(self.manifest, updater_signature=dict(self.manifest['updater_signature'], configuration_sha256='0' * 64))),
            (self.manifest_path, dict(self.manifest, updater_signature=dict(self.manifest['updater_signature'], windows_configuration_sha256='0' * 64))),
            (self.installer, b'tampered installer'), (self.signature, b'tampered signature'),
            (self.qa_path, dict(self.qa, source_commit='b' * 40)),
            (self.qa_path, dict(self.qa, fixture='Synthetic QA, not installed proof')),
            (self.qa_path, dict(self.qa, qualification_fixture='Synthetic QA, not installed proof')),
            (self.qa_path, dict(self.qa, artifact_manifest_sha256='0' * 64)),
            (self.qa_path, dict(self.qa, verification_sha256='0' * 64)),
            (self.candidate / 'signing/payloads/speakerdesk.exe', b'changed signed snapshot'),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=str(value)[:60]):
                path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
                with self.assertRaises((ValueError, KeyError)):
                    self.prepare()
                for saved, data in originals.items(): saved.write_bytes(data)
        self.signature.unlink()
        with self.assertRaises((ValueError, OSError)):
            self.prepare()

    def test_matching_report_digest_cannot_hide_wrong_install_or_updater_bindings(self):
        for path, field, value, digest_field in [
            (self.install_report, 'fixture', 'Synthetic report', 'verification_sha256'),
            (self.install_report, 'qualification_fixture', 'Synthetic report', 'verification_sha256'),
            (self.install_report, 'standard_user', False, 'verification_sha256'),
            (self.install_report, 'installed_payloads', {}, 'verification_sha256'),
            (self.install_report, 'authenticode_verified', False, 'verification_sha256'),
            (self.updater_report, 'kind', 'renamed-helper-fixture', 'updater_verification_sha256'),
            (self.updater_report, 'fixture', 'Synthetic report', 'updater_verification_sha256'),
            (self.updater_report, 'qualification_fixture', 'Synthetic report', 'updater_verification_sha256'),
            (self.updater_report, 'version', '0.7.6', 'updater_verification_sha256'),
            (self.updater_report, 'public_key', 'different', 'updater_verification_sha256'),
            (self.updater_report, 'payload_sha256', '0' * 64, 'updater_verification_sha256'),
            (self.updater_report, 'signature_sha256', '0' * 64, 'updater_verification_sha256'),
        ]:
            original = path.read_bytes()
            with self.subTest(field=field):
                report = json.loads(original); report[field] = value; path.write_text(json.dumps(report))
                qa = dict(self.qa); qa[digest_field] = promotion.digest_file(path); self.qa_path.write_text(json.dumps(qa))
                with self.assertRaises(ValueError): self.prepare()
            path.write_bytes(original); self.qa_path.write_text(json.dumps(self.qa))

    def test_powershell_utf8_bom_reports_keep_raw_hash_binding(self):
        qa = dict(self.qa)
        for path, field in ((self.install_report, 'verification_sha256'),
                            (self.updater_report, 'updater_verification_sha256')):
            path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
            qa[field] = promotion.digest_file(path)
        self.qa_path.write_bytes(b'\xef\xbb\xbf' + json.dumps(qa).encode())
        result = self.prepare()
        self.assertEqual(result['native_gate_sha256'], promotion.digest_file(self.qa_path))
        self.assertEqual(result['verification_sha256'], promotion.digest_file(self.install_report))
        self.install_report.write_bytes(self.install_report.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'operator report changed'):
            self.prepare()


if __name__=='__main__':unittest.main()
