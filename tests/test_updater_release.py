"""Authenticated signer inventory and client-key binding, with synthetic bytes."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import promote_release as promotion


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


if __name__=='__main__':unittest.main()
