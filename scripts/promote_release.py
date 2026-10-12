"""Retained signer/updater verification; native release promotion is unsupported.

The retired Python replay gate cannot qualify a native package. Preparation fails
before any network call until a native promotion gate is implemented.
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

REPOSITORY='TheGreenCedar/Speakerdesk'
REPOSITORY_ID=1406057260


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

def prepare(report, manifest_path, qa_path, notes, signer_archive=None):
    raise ValueError('Python acceptance reports are retired; native release promotion is not implemented')

def api(endpoint,body=None):
    command=['gh','api',endpoint]
    if body is not None:command+=['--method','PATCH','--input','-']
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
    if not refs:return
    obj=refs[0]['object']
    for _ in range(10):
        if obj['type']=='commit':break
        require(obj['type']=='tag','Release tag must resolve to a commit')
        obj=api('repos/'+REPOSITORY+'/git/tags/'+obj['sha'])['object']
    require(obj['type']=='commit' and obj['sha']==plan['source_commit'],'Existing release tag points to different source')

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
    verify_repository(api('repos/'+REPOSITORY))
    require(api('repos/'+REPOSITORY+'/commits/main')['sha']==plan['source_commit'],'Candidate is not the current reviewed main')
    check_tag(plan)
    releases=api('repos/'+REPOSITORY+'/releases?per_page=100')
    require(not any(r['tag_name']==plan['tag'] for r in releases),'Existing release/draft must be inspected; never overwrite')
    result=subprocess.run(['gh','release','create',plan['tag'],'--repo',REPOSITORY,'--target',plan['source_commit'],
          '--title','Speakerdesk '+plan['version'],'--notes-file',plan['notes_file'],'--draft']+
          [item['path'] for item in plan['files'].values()],text=True,capture_output=True,timeout=180)
    require(result.returncode==0,'Draft upload failed; preserve any partial draft')
    release=api('repos/'+REPOSITORY+'/releases/tags/'+plan['tag']);verify(release,plan)
    check_tag(plan,required=True)
    require(release['draft'] is True,'Unexpected release visibility')
    release=api('repos/'+REPOSITORY+'/releases/'+str(release['id']),{'draft':False,'prerelease':False,'make_latest':'true'})
    verify(release,plan);require(release['draft'] is False and release['published_at'],'Publication incomplete')
    check_tag(plan,required=True)
    # Rehash authenticated actual downloads without storing a duplicate installer.
    for item in release['assets']:
        size,digest=downloaded_digest(item['id'])
        require(size==plan['files'][item['name']]['bytes'] and digest==plan['files'][item['name']]['sha256'],'Published download verification failed')
    return {'release_url':release['html_url'],'published_at':release['published_at'],'source_commit':plan['source_commit'],'downloads_verified':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--native-qa',type=Path,required=True);parser.add_argument('--notes',type=Path,required=True);parser.add_argument('--publish',action='store_true')
    parser.add_argument('--signer-archive',type=Path,required=True)
    args=parser.parse_args();plan=prepare(args.report,args.manifest,args.native_qa,args.notes,args.signer_archive)
    print(json.dumps(publish(plan) if args.publish else {k:v for k,v in plan.items() if k!='notes'},indent=2))

if __name__=='__main__':main()
