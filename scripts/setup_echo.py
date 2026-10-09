#!/usr/bin/env python3
"""Build the approved pinned, project-local CPU echo component. No device/model access.

Official source archives only: no redirects, repository hooks, external build
scripts, gclient, browser checkout, downloads of toolchains, or global install.
"""
import base64
import hashlib
import io
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'.cache/echo-source/src'
BUILD = ROOT/'.cache/echo-build'
LOCK = ROOT/'desktop/capture/echo/source-lock.json'
RECEIVE_LIMIT = 32*1024**2
EXTRACT_LIMIT = 128*1024**2

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise RuntimeError('Echo source redirect refused.')


def git_hash(kind,data):
    return hashlib.sha1(f'{kind} {len(data)}\0'.encode()+data).digest()


def tree_hash(folder):
    entries=[]
    for p in folder.iterdir():
        if p.is_symlink():raise RuntimeError(f'Unsafe echo source: {p}')
        if p.is_dir():mode,oid,key='40000',tree_hash(p),p.name+'/'
        else:mode,oid,key=('100755' if p.stat().st_mode&0o111 else '100644'),git_hash('blob',p.read_bytes()),p.name
        entries.append((key.encode(),mode.encode()+b' '+p.name.encode()+b'\0'+oid))
    return git_hash('tree',b''.join(v for _,v in sorted(entries)))


def sources():
    lock_digest=hashlib.sha256(LOCK.read_bytes()).hexdigest()
    verified=SOURCE.parent/'VERIFIED-LOCK.json'
    if verified.is_file():
        cached=json.loads(verified.read_text())
        if cached.get('lock_sha256')==lock_digest and cached.get('all_source_hashes_verified'):
            return {**cached,'received_bytes':0,'extracted_bytes':0,'fetched_files':0,'verified_source_cache_reused':True}
    lock=json.loads(LOCK.read_text())
    opener=urllib.request.build_opener(NoRedirect)
    received=extracted=count=0
    def get(url):
        nonlocal received
        def expired(signum,frame):raise TimeoutError('Echo source request exceeded 30 seconds.')
        previous=signal.signal(signal.SIGALRM,expired)
        signal.alarm(30)
        try:
            with opener.open(url,timeout=30) as response:data=response.read(RECEIVE_LIMIT-received+1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM,previous)
        received+=len(data)
        if received>RECEIVE_LIMIT:raise RuntimeError('Echo source receive budget exceeded.')
        return data
    SOURCE.mkdir(parents=True,exist_ok=True)
    for relative,entry in lock['archives'].items():
        folder=SOURCE/relative
        if not folder.exists():
            data=get(entry['url'])
            # Extract to an owned temporary path; publish only after Git-tree verification.
            temp=folder.with_name(folder.name+'.incoming')
            if temp.exists():raise RuntimeError(f'Incomplete source fetch: {temp}; inspect before retry.')
            temp.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
                for member in tar:
                    rel=Path(member.name)
                    if rel.is_absolute() or '..' in rel.parts or not (member.isdir() or member.isfile()):
                        raise RuntimeError('Unsafe echo archive entry.')
                    p=temp/rel
                    if member.isdir():p.mkdir(parents=True,exist_ok=True);continue
                    extracted+=member.size;count+=1
                    if extracted>EXTRACT_LIMIT or count>10000:raise RuntimeError('Echo source extraction budget exceeded.')
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('xb') as out:out.write(tar.extractfile(member).read())
                    p.chmod(0o755 if member.mode&0o111 else 0o644)
            if tree_hash(temp).hex()!=entry['git_tree']:raise RuntimeError(f'Echo source tree mismatch: {relative}')
            temp.rename(folder)
        if tree_hash(folder).hex()!=entry['git_tree']:raise RuntimeError(f'Modified pinned echo source: {relative}')
    for relative,entry in lock['files'].items():
        path=SOURCE/relative
        if path.is_symlink():raise RuntimeError(f'Unsafe echo source file: {relative}')
        if not path.exists():
            data=base64.b64decode(get(entry['url']),validate=True)
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise RuntimeError(f'Echo source file mismatch: {relative}')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as out:out.write(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError(f'Modified pinned echo file: {relative}')
    receipt={'lock_sha256':lock_digest,'received_bytes':received,'extracted_bytes':extracted,
             'fetched_files':count,'all_source_hashes_verified':True}
    verified.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def build():
    if (platform.system(),platform.machine())!=('Darwin','arm64'):
        raise RuntimeError('The capture echo component requires Apple Silicon macOS.')
    receipt=sources();BUILD.mkdir(parents=True,exist_ok=True)
    guard=f'{sys.executable};{ROOT}/scripts/echo_compile_guard.py'
    compiler=subprocess.check_output(['xcrun','--find','clang'],text=True).strip()
    cxx=subprocess.check_output(['xcrun','--find','clang++'],text=True).strip()
    configure=['cmake','-S',str(ROOT/'desktop/capture/echo'),'-B',str(BUILD),'-G','Ninja',
        f'-DWEBRTC_SOURCE={SOURCE}',f'-DCMAKE_C_COMPILER={compiler}',f'-DCMAKE_CXX_COMPILER={cxx}',
        f'-DCMAKE_C_COMPILER_LAUNCHER={guard}',f'-DCMAKE_CXX_COMPILER_LAUNCHER={guard}',
        '-DCMAKE_BUILD_TYPE=Release']
    subprocess.run(configure,check=True,timeout=60)
    start=time.monotonic();reason=None
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    with (BUILD/'BUILD.log').open('w') as log:
        child=subprocess.Popen(['cmake','--build',str(BUILD),'--target','speakerdesk_echo','--parallel','2'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        (BUILD/'ACTIVE-BUILD.json').write_text(json.dumps({'owned_pid':child.pid,'phase':'compiling'})+'\n')
        try:
            while child.poll() is None:
                if time.monotonic()-start>600:reason='Native build wall budget';break
                if sum(p.stat().st_size for p in BUILD.rglob('*') if p.is_file())>512*1024**2:reason='Native build output budget';break
                time.sleep(.25)
        finally:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
            (BUILD/'ACTIVE-BUILD.json').write_text(json.dumps({'owned_pid':child.pid,'phase':'joined','exit_code':child.returncode})+'\n')
    receipt.update(exit_code=child.returncode,stop_reason=reason,elapsed_seconds=time.monotonic()-start,
                   peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                   parallel_compilers=2,output_bytes=sum(p.stat().st_size for p in BUILD.rglob('*') if p.is_file()))
    if child.returncode or reason:
        print('\n'.join((BUILD/'BUILD.log').read_text().splitlines()[-30:]),file=sys.stderr)
        raise RuntimeError(f'Echo component build failed: {reason or child.returncode}')
    built=BUILD/'libspeakerdesk_echo.dylib'
    target=ROOT/'desktop/capture/echo/libspeakerdesk_echo.dylib'
    receipt['library_sha256']=hashlib.sha256(built.read_bytes()).hexdigest()
    shutil.copyfile(built,target)
    (BUILD/'BUILD-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':build()
