"""Promote an authenticated native candidate after recorded operator QA.

Local QA and payload checks precede read-only signer authentication. --publish
creates and verifies a draft before publication; existing releases are preserved.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import os
import selectors
import signal
import time
import zipfile
import windows_signing

REPOSITORY='TheGreenCedar/Speakerdesk'
REPOSITORY_ID=1406057260
ROOT=Path(__file__).resolve().parents[1]


def require(value,message):
    if not value:raise ValueError(message)

def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024**2):digest.update(block)
    return digest.hexdigest()

def payload_names(manifest):
    """Exact signed payload contract; legacy candidates retain two payloads."""
    version=manifest['version']
    require(re.fullmatch(r'\d+\.\d+\.\d+',version),'Invalid release version')
    stem=f'Speakerdesk_{version}_AppleSilicon'
    expected={stem+'.dmg',stem+'.app.zip'}
    if 'updater' in manifest:
        update=manifest['updater']
        require(isinstance(update,dict) and set(update)=={'platform','filename','signature_filename','signature','public_key','require_signed_version'},'Invalid updater metadata contract')
        archive=stem+'.app.tar.gz'
        require(update['platform']=='darwin-aarch64' and update['filename']==archive and
                update['signature_filename']==archive+'.sig' and update['require_signed_version'] is True,'Updater platform/version binding differs')
        for key in ('signature','public_key'):
            value=update[key]
            require(isinstance(value,str) and 0<len(value)<=8192,'Missing updater '+key)
            try:base64.b64decode(value.strip(),validate=True)
            except (ValueError,TypeError):raise ValueError('Invalid updater '+key)
        expected|={archive,archive+'.sig'}
    return expected

def verify_updater_files(manifest,directory,config):
    """Bind authenticated signer output to this client's compiled public key."""
    public_key=config.get('plugins',{}).get('updater',{}).get('pubkey','')
    update=manifest.get('updater')
    if tuple(map(int,manifest['version'].split('.')))>=(0,6,2):
        require(update is not None and public_key,'Speakerdesk 0.6.2+ requires configured and signed updater artifacts')
    require(not public_key or update is not None,'Updater-enabled client requires signed updater artifacts')
    if update is None:return
    payload_names(manifest)
    require(public_key and update['public_key']==public_key,'Updater verification key differs from the compiled client')
    plugin=config['plugins']['updater']
    require(plugin.get('requireSignedVersion') is True and plugin.get('allowDowngrades',False) is False,'Updater client version policy differs')
    signature=directory/update['signature_filename']
    require(signature.stat().st_size<=8192 and not signature.is_symlink(),'Unsafe updater signature file')
    require(signature.read_text().strip()==update['signature'].strip(),'Updater signature metadata differs from final signer bytes')

def prepare(manifest_path, qa_path, notes, signer_archive):
    require(manifest_path.name=='artifact-manifest.json' and manifest_path.is_file() and not manifest_path.is_symlink(),
            'Validate the exact manifest that will be uploaded')
    manifest=json.loads(manifest_path.read_text());qa=json.loads(qa_path.read_text(encoding='utf-8-sig'))
    if manifest.get('platform')=='windows-x86_64':
        require(signer_archive is None,'Local Windows preparation does not use an Apple signer archive')
        return prepare_windows(manifest_path,qa_path,notes,manifest,qa)
    require(signer_archive is not None,'Apple preparation requires the original signer archive')
    require(manifest.get('source_repository')==REPOSITORY and
            re.fullmatch(r'[0-9a-f]{40}',manifest.get('source_commit','')),'Signer source repository or commit differs')
    require(manifest.get('runtime_kind')=='native','Native signer candidate required; Python acceptance is retired')
    require(manifest.get('signing')=='developer-id' and manifest.get('notarized') is True and
            all(n.get('status')=='Accepted' for n in manifest.get('notarization',[])) and
            {n.get('kind') for n in manifest.get('notarization',[])}=={'app','dmg'},'App and DMG notarization required')
    config=json.loads((ROOT/'desktop/src-tauri/tauri.conf.json').read_text())
    version=manifest['version'];require(version==config['version'],'Candidate version differs')
    expected=payload_names(manifest);files={}
    for item in manifest['files']:
        filename=item['filename'];require(filename in expected and filename not in files,'Unexpected package payload inventory')
        path=manifest_path.parent/filename
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes'] and
                digest_file(path)==item['sha256'],'Signed payload bytes differ')
        files[filename]=dict(item,path=str(path))
    require(set(files)==expected,'Unexpected package payload inventory')
    require(qa.get('passed') is True and qa.get('source_commit')==manifest['source_commit'] and qa.get('version')==version,
            'Exact native package QA required')
    require(qa.get('app_archive_sha256')==files[f'Speakerdesk_{version}_AppleSilicon.app.zip']['sha256'],
            'Native QA executed a different package')
    # The report records operator acceptance; signer provenance is authenticated independently below.
    require(digest_file(Path(qa['verification']))==qa.get('verification_sha256'),'Native verification changed')
    verify_updater_files(manifest,manifest_path.parent,config)
    for name in ('artifact-manifest.json','SHA256SUMS'):
        path=manifest_path.parent/name;require(path.is_file() and not path.is_symlink(),'Release metadata missing')
        files[name]={'filename':name,'bytes':path.stat().st_size,'sha256':digest_file(path),'path':str(path)}
    sums=''.join(f'{x["sha256"]}  {x["filename"]}\n' for x in manifest['files'])
    require((manifest_path.parent/'SHA256SUMS').read_text()==sums,'Signed checksum inventory differs')
    body=notes.read_text();require(manifest['source_commit'] in body,'Notes must identify exact source')
    verify_signing(manifest,signer_archive)
    return {'repository':REPOSITORY,'tag':'v'+version,'source_commit':manifest['source_commit'],'version':version,
            'files':files,'notes_file':str(notes),'notes':body,'native_gate_sha256':digest_file(qa_path)}

def prepare_windows(manifest_path,qa_path,notes,manifest,qa):
    """Bind local operator observations without claiming Actions provenance."""
    regular=windows_signing.regular
    require(not {'fixture','qualification_fixture'} & manifest.keys(),'Helper fixtures are not Windows release candidates')
    require(not {'fixture','qualification_fixture'} & qa.keys(),'Fixture QA is not signed installed evidence')
    source=manifest.get('source_commit','');version=manifest.get('version','')
    configuration=windows_signing.updater_configuration()
    require(re.fullmatch(r'[0-9a-f]{40}',source) and manifest.get('source_dirty') is False,'Clean exact Windows source required')
    require(version==configuration['version'] and re.fullmatch(r'\d+\.\d+\.\d+',version),'Windows candidate version differs')
    require(manifest.get('public_ready') is False and manifest.get('authenticode_status')=='Valid'
            and manifest.get('signing')=={'kind':'azure-artifact-signing','payloads_observed_at_hook':True,'receipts':'signing'},
            'Final Authenticode-signed Windows candidate required')
    name=f'Speakerdesk_{version}_x64-setup.exe';pin=manifest['installer'];update=manifest['updater_signature']
    installer=regular(manifest_path.parent/name);signature=regular(manifest_path.parent/(name+'.sig'))
    require(pin=={'name':name,'bytes':installer.stat().st_size,'sha256':digest_file(installer)},'Windows installer pin differs')
    require(signature.stat().st_size<=8192,'Windows updater signature exceeds its bound')
    signature_text=signature.read_text().strip()
    require(re.fullmatch(r'[A-Za-z0-9+/]+={0,2}',signature_text),'Invalid updater signature encoding')
    expected_update={'filename':name,'signature_filename':name+'.sig','signature':signature_text,
                     'verified':True,'require_signed_version':True,
                     **{key:value for key,value in configuration.items() if key!='version'}}
    require(update==expected_update,'Final verified Windows updater/configuration metadata required')
    installed=manifest['installed_payloads']
    require(isinstance(installed,dict) and installed and all(
        re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*',name) and
        not {'.','..'} & set(name.split('/')) and re.fullmatch(r'[0-9a-f]{64}',value)
        for name,value in installed.items()),'Invalid installed payload inventory')
    runtime=json.loads(regular(ROOT/'crates/models/src/windows-runtime.json').read_text())['files']
    require({'speakerdesk-capture.exe.capability.json','THIRD_PARTY_NOTICES.txt'}<=installed.keys()
            and all(installed.get('runtime/'+name)==pin['sha256'] for name,pin in runtime.items()),
            'Installed inventory omits or changes pinned Windows runtime resources')
    signed={}
    for role,filename in {'capture':'speakerdesk-capture.exe','echo':'speakerdesk_echo.dll',
                          'app':'speakerdesk.exe','uninstaller':'uninstall.exe','installer':name}.items():
        receipt=json.loads(regular(manifest_path.parent/'signing'/(role+'.json')).read_text(encoding='utf-8-sig'))
        snapshot=regular(manifest_path.parent/'signing/payloads'/filename)
        expected=pin['sha256'] if role=='installer' else installed.get(filename)
        require(receipt.get('role')==role and receipt.get('disposition')=='owned_signed'
                and receipt.get('authenticode_status')=='Valid' and receipt.get('timestamped') is True
                and re.match(r'^CN=Albert Najjar(?:,|$)',receipt.get('publisher_subject',''))
                and receipt.get('sha256')==expected==digest_file(snapshot),'Signed Windows payload inventory differs: '+role)
        signed[filename]=expected
    require(qa.get('passed') is True and qa.get('platform')=='windows-x86_64' and qa.get('source_commit')==source
            and qa.get('version')==version and qa.get('artifact_manifest_sha256')==digest_file(regular(manifest_path)),
            'Exact signed Windows installed QA required')
    reports={}
    for field in ('verification','updater_verification'):
        data=regular(qa[field]).read_bytes()
        require(hashlib.sha256(data).hexdigest()==qa.get(field+'_sha256'),'Windows operator report changed: '+field)
        reports[field]=json.loads(data.decode('utf-8-sig'))
    observed=reports['verification'];verified=reports['updater_verification']
    require(not {'fixture','qualification_fixture'} & observed.keys()
            and observed.get('kind')=='signed-standard-user-install' and observed.get('passed') is True
            and observed.get('installer_sha256')==pin['sha256'] and observed.get('installed_payloads')==installed
            and all(observed.get(key) is True for key in ('standard_user','authenticode_verified','save_reopen',
                                                         'clean_shutdown','uninstall_reinstall','data_preserved')),
            'Signed standard-user installation observations required')
    public=json.loads(regular(ROOT/'desktop/src-tauri/tauri.conf.json').read_text())['plugins']['updater']['pubkey']
    require(verified.get('kind')=='nsis-installer-updater-verification' and not {'fixture','qualification_fixture'} & verified.keys()
            and verified.get('payload_sha256')==pin['sha256'] and verified.get('signature_sha256')==digest_file(signature)
            and verified.get('version')==version and verified.get('public_key')==public
            and verified.get('signature_key_version_verified') is True and verified.get('payload_unchanged') is True,
            'Actual final-installer updater verification receipt required')
    body=regular(notes).read_text();require(source in body,'Notes must identify exact source')
    files={path.name:{'filename':path.name,'bytes':path.stat().st_size,'sha256':digest_file(path),'path':str(path)}
           for path in (installer,signature,regular(manifest_path))}
    return {'repository':REPOSITORY,'platform':'windows-x86_64','provenance':'local-operator-controlled',
            'app_driven_update_observed':observed.get('app_driven_update_observed') is True,
            'tag':'v'+version,'version':version,'source_commit':source,
            'files':files,'signed_payloads':signed,'notes_file':str(notes),'notes':body,
            'native_gate_sha256':digest_file(regular(qa_path)),
            'verification_sha256':qa['verification_sha256'],'updater_verification_sha256':qa['updater_verification_sha256']}

def api(endpoint,body=None,*,method='PATCH'):
    command=['gh','api',endpoint]
    if body is not None:command+=['--method',method,'--input','-']
    result=subprocess.run(command,input=json.dumps(body) if body is not None else None,text=True,capture_output=True,timeout=60)
    require(result.returncode==0,'GitHub request failed; preserve partial draft for inspection')
    return json.loads(result.stdout)

def verify_signing(manifest,archive):
    """Read-only service provenance plus original signed artifact bytes.

Local labels cannot authenticate a relabelled older package. This independently
binds the exact signed manifest/payload to the trusted successful signer artifact,
and its input to the successful main producer run. No credential changes.
"""
    producer=api('repos/'+REPOSITORY+'/actions/runs/'+str(manifest['producer_run_id']))
    signer=api('repos/TheGreenCedar/AppleRelease/actions/runs/'+str(manifest['signer_run_id']))
    for run,repo_id,name,path,head in ((producer,REPOSITORY_ID,REPOSITORY,'.github/workflows/apple-build.yml',manifest['source_commit']),
              (signer,1406130411,'TheGreenCedar/AppleRelease','.github/workflows/sign.yml',manifest['signer_commit'])):
        require(run.get('repository',{}).get('id')==repo_id and run.get('head_repository',{}).get('id')==repo_id and
                run['repository'].get('full_name')==name and run.get('path')==path and run.get('head_branch')=='main' and
                run.get('event')=='workflow_dispatch' and run.get('status')=='completed' and run.get('conclusion')=='success' and
                run.get('head_sha')==head,'Trusted producer/signer run provenance differs')
    input_artifact=api('repos/'+REPOSITORY+'/actions/artifacts/'+str(manifest['input_artifact_id']))
    origin=input_artifact.get('workflow_run',{})
    require(input_artifact.get('digest')=='sha256:'+manifest['input_artifact_sha256'] and origin.get('id')==int(manifest['producer_run_id']) and
            origin.get('head_sha')==manifest['source_commit'] and origin.get('repository_id')==REPOSITORY_ID and origin.get('head_repository_id')==REPOSITORY_ID and
            origin.get('head_branch')=='main','Signed candidate producer artifact differs')
    candidates=api('repos/TheGreenCedar/AppleRelease/actions/runs/'+str(manifest['signer_run_id'])+'/artifacts?per_page=100')['artifacts']
    name=f'Speakerdesk-{manifest["source_commit"]}-notarized-candidate-{manifest["signer_run_id"]}'
    candidates=[a for a in candidates if a.get('name')==name];require(len(candidates)==1,'Trusted signer artifact missing/ambiguous')
    candidate=candidates[0];origin=candidate.get('workflow_run',{})
    require(candidate.get('expired') is False and origin.get('id')==int(manifest['signer_run_id']) and origin.get('repository_id')==1406130411 and
            origin.get('head_repository_id')==1406130411 and origin.get('head_sha')==manifest['signer_commit'] and origin.get('head_branch')=='main','Signer artifact provenance differs')
    require(archive.is_file() and not archive.is_symlink() and candidate.get('digest')=='sha256:'+digest_file(archive),'Original signer archive differs from authenticated service digest')
    with zipfile.ZipFile(archive) as zipped:
        members=[i for i in zipped.infolist() if not i.is_dir()]
        require(json.loads(zipped.read('artifact-manifest.json'))==manifest,'Validated source manifest differs from trusted signer output')
        names=payload_names(manifest)|{'artifact-manifest.json','SHA256SUMS'}
        require({i.filename for i in members}==names and len(members)==len(names) and
                len(manifest['files'])==len(names)-2,'Signer artifact inventory differs')
        for item in manifest['files']:
            info=zipped.getinfo(item['filename']);require(info.file_size==item['bytes'],'Trusted signed payload size differs')
            with zipped.open(info) as stream:require(hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256'],'Trusted signed payload hash differs')

def verify(release,plan):
    require(release['tag_name']==plan['tag'] and release['target_commitish']==plan['source_commit'] and release['prerelease'] is False and release['body']==plan['notes'],'Release metadata differs')
    require(len(release['assets'])==len(plan['files']) and {a['name'] for a in release['assets']}==set(plan['files']),'Release inventory differs')
    for item in release['assets']:
        expected=plan['files'][item['name']]
        require(item['state']=='uploaded' and item['size']==expected['bytes'] and item['digest']=='sha256:'+expected['sha256'],'Uploaded release bytes differ')

def check_tag(plan,required=False):
    refs=api('repos/'+REPOSITORY+'/git/matching-refs/tags/'+plan['tag'])
    refs=[r for r in refs if r['ref']=='refs/tags/'+plan['tag']]
    require(len(refs)<=1 and (refs or not required),'Expected release Git tag missing/ambiguous')
    if not refs:return False
    obj=refs[0]['object']
    for _ in range(10):
        if obj['type']=='commit':break
        require(obj['type']=='tag','Release tag must resolve to a commit')
        obj=api('repos/'+REPOSITORY+'/git/tags/'+obj['sha'])['object']
    require(obj['type']=='commit' and obj['sha']==plan['source_commit'],'Existing release tag points to different source')
    return True

def downloaded_digest(asset_id):
    process=subprocess.Popen(['gh','api','-H','Accept: application/octet-stream',
                 'repos/'+REPOSITORY+'/releases/assets/'+str(asset_id)],stdout=subprocess.PIPE,
                 stderr=subprocess.DEVNULL,start_new_session=True)
    digest=hashlib.sha256();size=0;deadline=time.monotonic()+180
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout,selectors.EVENT_READ)
            while True:
                require(time.monotonic()<deadline,'Published download timed out')
                if not ready.select(timeout=min(1,max(.01,deadline-time.monotonic()))):continue
                block=os.read(process.stdout.fileno(),1024**2)
                if not block:break
                digest.update(block);size+=len(block)
            process.wait(timeout=3);require(process.returncode==0,'Published download request failed')
        return size,digest.hexdigest()
    finally:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
        process.stdout.close()

def verify_repository(repo):
    require(repo['id']==REPOSITORY_ID and repo['full_name']==REPOSITORY,'Repository identity differs')

def publish(plan):
    windows=plan.get('platform')=='windows-x86_64'
    if windows:
        require(plan.get('app_driven_update_observed') is True,'Windows publication requires actual app-driven update observations')
    verify_repository(api('repos/'+REPOSITORY))
    main=api('repos/'+REPOSITORY+'/commits/main')
    if windows:
        comparison=api('repos/'+REPOSITORY+'/compare/'+plan['source_commit']+'...'+main['sha'])
        ancestor=comparison.get('merge_base_commit',{})
        tree=main.get('commit',{}).get('tree',{}).get('sha','')
        require(ancestor.get('sha')==plan['source_commit'] and re.fullmatch(r'[0-9a-f]{40}',tree)
                and ancestor.get('commit',{}).get('tree',{}).get('sha')==tree,
                'Windows candidate must be an ancestor of reviewed main with the same Git tree')
    else:
        require(main['sha']==plan['source_commit'],'Candidate is not the current reviewed main')
    check_tag(plan)
    releases=api('repos/'+REPOSITORY+'/releases?per_page=100')
    require(not any(r['tag_name']==plan['tag'] for r in releases),'Existing release/draft must be inspected; never overwrite')
    result=subprocess.run(['gh','release','create',plan['tag'],'--repo',REPOSITORY,'--target',plan['source_commit'],
          '--title','Speakerdesk '+plan['version']+(' for Windows' if windows else ''),'--notes-file',plan['notes_file'],'--draft']+
          [item['path'] for item in plan['files'].values()],text=True,capture_output=True,timeout=180)
    require(result.returncode==0,'Draft upload failed; preserve any partial draft')
    # The tag endpoint exposes published releases; drafts are listed by identity.
    drafts=[r for r in api('repos/'+REPOSITORY+'/releases?per_page=100') if r['tag_name']==plan['tag']]
    require(len(drafts)==1 and type(drafts[0].get('id')) is int and drafts[0]['id']>0 and
            drafts[0]['id'] not in {r['id'] for r in releases},'Created draft identity missing/ambiguous')
    release_id=drafts[0]['id']
    release=api('repos/'+REPOSITORY+'/releases/'+str(release_id))
    require(release.get('id')==release_id,'Created draft identity differs');verify(release,plan)
    require(release['draft'] is True,'Unexpected release visibility')
    if not check_tag(plan):
        api('repos/'+REPOSITORY+'/git/refs',{'ref':'refs/tags/'+plan['tag'],'sha':plan['source_commit']},method='POST')
    check_tag(plan,required=True)
    if windows:
        require(api('repos/'+REPOSITORY+'/commits/main')['sha']==main['sha'],
                'Reviewed main changed during Windows upload; preserve the draft for inspection')
    release=api('repos/'+REPOSITORY+'/releases/'+str(release_id),{'draft':False,'prerelease':False,'make_latest':'false' if windows else 'true'})
    require(release.get('id')==release_id,'Published release identity differs')
    verify(release,plan);require(release['draft'] is False and release['published_at'],'Publication incomplete')
    check_tag(plan,required=True)
    # Rehash authenticated actual downloads without storing a duplicate installer.
    for item in release['assets']:
        size,digest=downloaded_digest(item['id'])
        require(size==plan['files'][item['name']]['bytes'] and digest==plan['files'][item['name']]['sha256'],'Published download verification failed')
    return {'release_url':release['html_url'],'published_at':release['published_at'],'source_commit':plan['source_commit'],'downloads_verified':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--native-qa',type=Path,required=True);parser.add_argument('--notes',type=Path,required=True);parser.add_argument('--publish',action='store_true')
    parser.add_argument('--signer-archive',type=Path)
    args=parser.parse_args();plan=prepare(args.manifest,args.native_qa,args.notes,args.signer_archive)
    print(json.dumps(publish(plan) if args.publish else {k:v for k,v in plan.items() if k!='notes'},indent=2))

if __name__=='__main__':main()
