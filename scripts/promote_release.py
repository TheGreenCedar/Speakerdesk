"""Supported private release promotion: exact acoustics AND native export gates.

Local core/native validation runs before any network call. Default preparation
verifies signing provenance read-only, then prints a reviewable plan. --publish
executes an already-authorized release; it is never CI-driven.
An existing release/draft is preserved, not overwritten or auto-resumed.
"""
import argparse
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
from core_acceptance import ROOT, artifact, digest_file, require, validate

REPOSITORY='TheGreenCedar/Speakerdesk'
REPOSITORY_ID=1406057260

def prepare(report, manifest_path, qa_path, notes, signer_archive=None):
    require(manifest_path.name=='artifact-manifest.json','Validate the exact manifest that will be uploaded')
    core=validate(report,manifest_path)
    manifest=json.loads(manifest_path.read_text());qa=json.loads(qa_path.read_text())
    require(manifest.get('source_repository')==REPOSITORY,'Signer source repository differs')
    require(all(n.get('status')=='Accepted' for n in manifest.get('notarization',[])) and
            {n.get('kind') for n in manifest.get('notarization',[])}=={'app','dmg'},'App and DMG notarization required')
    require(manifest.get('signer_run_id') and manifest.get('producer_run_id') and manifest.get('input_artifact_sha256'),'Trusted signing provenance missing')
    config=json.loads((ROOT/'desktop/src-tauri/tauri.conf.json').read_text())
    version=manifest['version'];require(version==config['version'] and re.fullmatch(r'\d+\.\d+\.\d+',version),'Candidate version differs')
    require(qa.get('passed') is True and qa.get('source_commit')==manifest['source_commit'] and qa.get('version')==version,'Exact native export QA required')
    require(qa.get('candidate_cdhash')==core['app_cdhash'],'Native and acoustic QA executed different apps')
    for key in ('all_four_native_output_formats_match_expected_bytes','root_independently_rehashed_saved_exports','staging_clean','native_UI_responsive'):
        require(qa.get(key) is True,'Native export regression incomplete: '+key)
    require(qa.get('native_UI_save_cancel_retry')=='passed' and qa.get('native_UI_save_cancel_retry_replacement')=='passed','Native Save/Cancel/retry/replacement incomplete')
    require(digest_file(Path(qa['verification']))==qa.get('verification_sha256'),'Native verification changed')
    exports=qa.get('rehashed_exports',[])
    require({x['kind'] for x in exports}=={'txt','srt','vtt','json'},'Native format coverage missing')
    for item in exports:artifact(Path(item['path']),item)
    files={}
    for item in manifest['files']:
        filename=item['filename'];require(Path(filename).name==filename,'Unsafe artifact filename')
        path=manifest_path.parent/filename;artifact(path,item);files[filename]=dict(item,path=str(path))
    require(set(files)=={f'Speakerdesk_{version}_AppleSilicon.dmg',f'Speakerdesk_{version}_AppleSilicon.app.zip'},'Unexpected package payload inventory')
    for name in ('artifact-manifest.json','SHA256SUMS'):
        path=manifest_path.parent/name;require(path.is_file() and not path.is_symlink(),'Release metadata missing')
        files[name]={'filename':name,'bytes':path.stat().st_size,'sha256':digest_file(path),'path':str(path)}
    sums=''.join(f'{x["sha256"]}  {x["filename"]}\n' for x in manifest['files'])
    require((manifest_path.parent/'SHA256SUMS').read_text()==sums,'Signed checksum inventory differs')
    body=notes.read_text();require(manifest['source_commit'] in body,'Notes must identify exact source')
    require(signer_archive is not None,'Authenticated original signer artifact archive required')
    verify_signing(manifest,signer_archive)
    return {'repository':REPOSITORY,'tag':'v'+version,'source_commit':manifest['source_commit'],'version':version,
            'files':files,'notes_file':str(notes),'notes':body,'core_report_sha256':digest_file(report),'native_gate_sha256':digest_file(qa_path)}

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
        require({i.filename for i in members}=={x['filename'] for x in manifest['files']}|{'artifact-manifest.json','SHA256SUMS'} and len(members)==4,'Signer artifact inventory differs')
        require(json.loads(zipped.read('artifact-manifest.json'))==manifest,'Validated source manifest differs from trusted signer output')
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

def publish(plan):
    repo=api('repos/'+REPOSITORY);require(repo['id']==REPOSITORY_ID and repo['private'] is True,'Repository privacy/identity differs')
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
