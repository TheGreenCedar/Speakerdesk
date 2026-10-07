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
