"""Bounded local acceptance of exact signed runtime through production live APIs.

Does not capture, patch model classes, import the source server, or publish. Missing
resources/voice/checkpoints/cases never count as passing. Run after source freeze
and a fresh coordinated lease; retaining failed results is intentional.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import wave
import zipfile
from acceptance_fixtures import generate, read_wav, write_wav
from core_acceptance import ROOT, HARNESSES, artifact, bundle_identity, digest_file, established_context, evaluate, model_file_pins, models, pause_acknowledged, require, suite, transport_pcm, zip_bundle

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def owned_rss(process):
    # Only the PG created by this harness. Never retry the denied shared-process
    # inventory or inspect unrelated arguments/credentials. A denied owned read
    # aborts acceptance rather than requesting a bypass.
    result=subprocess.run(['/bin/ps','-g',str(process.pid),'-o','pgid=,rss='],text=True,capture_output=True,timeout=3)
    if result.returncode or not result.stdout.strip():
        try:os.killpg(process.pid,0)
        except ProcessLookupError:
            require(result.returncode in (0,1) and not result.stderr.strip() and not result.stdout.strip(),'Owned resource metrics unavailable')
            return 0
        raise ValueError('Owned resource metrics unavailable for running group')
    return sum(int(size)*1024 for group,size in (line.split() for line in result.stdout.splitlines()) if int(group)==process.pid)

def free_memory():
    text=subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=3)
    return int(re.search(r'memory free percentage:\s*(\d+)%',text)[1])

def terminate_group(process):
    """Check the whole owned PG, including children reparented after leader exit."""
    def alive():
        try:os.killpg(process.pid,0);return True
        except ProcessLookupError:return False
    try:
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:pass
            end=time.monotonic()+3
            while alive() and time.monotonic()<end:time.sleep(.1)
            if not alive():break
        try:process.wait(timeout=1)
        except subprocess.TimeoutExpired:return False
        return not alive()
    except Exception:
        # A failed census must not prevent terminating our own recorded PG.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:process.wait(timeout=1)
        except subprocess.TimeoutExpired:pass
        return False

def cleanup(backend,budget):
    errors=[];backend_clean=True;budget_clean=False
    try:
        if backend:backend_clean=backend.close()
    except Exception as exc:backend_clean=False;errors.append('backend: '+str(exc))
    finally:
        try:budget_clean=budget.close()
        except Exception as exc:errors.append('supervisor: '+str(exc))
    return backend_clean and budget_clean,errors

class Budget:
    def __init__(self, lease, output):
        value=json.loads(lease.read_text());now=datetime.datetime.now(datetime.timezone.utc)
        stamp=datetime.datetime.fromisoformat(value['checked_utc'])
        require(value.get('admitted') is True and 0<=(now-stamp).total_seconds()<=600,'Fresh coordinated lease required')
        self.floor=value['floor_plus_other_reserve_bytes'];self.seconds=value['seconds'];self.reserve=value['output_reserve_bytes']
        require(type(self.floor) is int and self.floor>=40_000_000_000,'Shared floor missing')
        require(type(self.seconds) is int and 60<=self.seconds<=1200,'Bounded 60–1200 second window required')
        require(type(self.reserve) is int and 64*1024**2<=self.reserve<=128*1024**2,'64–128 MiB output reservation required')
        require(shutil.disk_usage(output.parent).free>=self.floor+self.reserve and free_memory()>=35,'Insufficient admitted resources')
        require(value.get('other_model_build_operations_joined') is True,'Coordinated other-owner completion required; no shared process inventory is attempted')
        self.output=output;self.started=time.monotonic();self.process=None;self.peak=0;self.minimum_disk=shutil.disk_usage(output.parent).free;self.minimum_memory=100;self.last_memory=0;self.available=100
        self.processes=[];self.failure=None;self.closed=threading.Event();self.guard=None
        self.lease=value
    def check(self):
        if self.failure:raise RuntimeError(self.failure)
        require(time.monotonic()-self.started<self.seconds,'Acceptance deadline exceeded')
        free=shutil.disk_usage(self.output).free;self.minimum_disk=min(free,self.minimum_disk)
        require(free>=self.floor,'Shared disk floor reached')
        require(sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file() and not p.is_symlink())<=self.reserve,'Output reservation exceeded')
        if time.monotonic()-self.last_memory>=2:self.available=free_memory();self.last_memory=time.monotonic()
        self.minimum_memory=min(self.minimum_memory,self.available);require(self.available>=25,'Memory pressure')
        if self.processes:
            total=sum(owned_rss(process) for process in tuple(self.processes));self.peak=max(total,self.peak)
            require(total<=4*1024**3,'Owned inference RSS exceeds 4 GiB')
    def start(self):
        def supervise():
            while not self.closed.wait(.25):
                try:self.check()
                except Exception as exc:
                    self.failure=str(exc)
                    for process in tuple(self.processes):terminate_group(process)
                    return
        self.guard=threading.Thread(target=supervise,daemon=True);self.guard.start()
    def command(self,arguments,timeout):
        self.check();process=subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        self.processes.append(process)
        try:
            stdout,stderr=process.communicate(timeout=min(timeout,max(.1,self.seconds-(time.monotonic()-self.started))))
            require(process.returncode==0,'Bounded synthesis command failed');self.check()
            return subprocess.CompletedProcess(arguments,process.returncode,stdout,stderr)
        finally:
            if terminate_group(process):self.processes.remove(process)
    def close(self):
        self.closed.set()
        if self.guard:self.guard.join(timeout=8)
        results=[terminate_group(process) for process in tuple(self.processes)]
        return all(results) and not (self.guard and self.guard.is_alive())
    def wait(self, callback, seconds=65):
        end=time.monotonic()+seconds
        while True:
            self.check();result=callback()
            if result:return result
            require(time.monotonic()<end,'Production route did not settle before deadline');time.sleep(.2)

class Backend:
    def __init__(self,budget,runtime,home,helper,log):
        env=dict(os.environ,SPEAKERDESK_HOME=str(home),SPEAKERDESK_CAPTURE_HELPER=str(helper),
                 SPEAKERDESK_REPLAY_DIR=str(home/'replay'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        for key in ('DIAR_MODEL_PATH','COHERE_MODEL_PATH','LID_MODEL_PATH','SPEAKERDESK_LIVE_PYTHON','SPEAKERDESK_MODELS','PYTHONPATH'):
            env.pop(key,None)
        self.budget=budget;self.log=log.open('w');lines=queue.Queue()
        self.process=subprocess.Popen([str(runtime)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,start_new_session=True)
        budget.process=self.process;budget.processes.append(self.process)
        def collect():
            for line in self.process.stdout:lines.put(line);self.log.write(line);self.log.flush()
        self.reader=threading.Thread(target=collect,daemon=True);self.reader.start()
        def url():
            require(self.process.poll() is None,'Packaged server exited')
            while not lines.empty():
                line=lines.get_nowait()
                if line.startswith('SPEAKERDESK_URL='):return line.strip().split('=',1)[1]
        self.url=budget.wait(url,30)
        require(re.fullmatch(r'http://127\.0\.0\.1:\d+',self.url) is not None,'Unexpected server address')
        self.http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        page=self.http.open(self.url,timeout=5).read().decode()
        self.token=re.search(r'name="speakerdesk-token" content="([^"]+)"',page)[1]
    def api(self,path,body=None,method=None):
        self.budget.check()
        data=json.dumps(body).encode() if body is not None else None
        request=urllib.request.Request(self.url+path,data=data,method=method,
                 headers={'Content-Type':'application/json','X-Speakerdesk-Token':self.token})
        with self.http.open(request,timeout=5) as response:return json.load(response)
    def close(self):
        process=getattr(self,'process',None)
        if process is None:
            if getattr(self,'log',None):self.log.close()
            return False
        if process.poll() is None:
            try:process.stdin.write('shutdown\n');process.stdin.flush();process.wait(timeout=8)
            except (OSError,subprocess.TimeoutExpired):pass
        clean=terminate_group(process)
        if getattr(self,'reader',None):self.reader.join(timeout=2)
        self.log.close();return clean

def document_slice(job,start_sample):
    selected=copy.deepcopy(job)
    selected['document']['segments']=[row for row in selected['document']['segments'] if round(row['end']*16000)>start_sample]
    # Any row crossing the held predecessor boundary is an explicit failure, not
    # permission to splice generated text or invent word alignment.
    require(all(round(row['start']*16000)>=start_sample for row in selected['document']['segments']),'Candidate row crosses context fixture boundary')
    return selected

def case_run(recipe,all_recipes,backend,home,directory,budget):
    frames=generate(recipe,directory,budget.command);start_sample=0;preceding=None
    if recipe.get('preceding_context_case'):
        previous=all_recipes[recipe['preceding_context_case']]
        prefix_dir=directory/'preceding';generate(previous,prefix_dir,budget.command)
        prefix=read_wav(prefix_dir/'replay.wav')+[0]*19200
        start_sample=len(prefix);values=prefix+read_wav(directory/'replay.wav');write_wav(directory/'replay.wav',values);frames=len(values)
        write(directory/'replay.json',{'hold_sample':start_sample})
    replay=home/'replay'
    if replay.exists():shutil.rmtree(replay)
    shutil.copytree(directory,replay)
    job=backend.api('/api/meetings',{'name':'Acceptance '+recipe['id'],'language':recipe['language'],'sources':['microphone']})
    jid=job['id'];get=lambda:backend.api('/api/jobs/'+jid)
    budget.wait(lambda:get() if get()['status']=='recording' else None)
    backend.api('/api/jobs/'+jid+'/refinement/pause',{},'POST')
    edit=None
    if recipe.get('correction') or start_sample:
        budget.wait(lambda:(replay/'stage-edit').exists(),60)
        pause_request=backend.api('/api/meetings/'+jid+'/pause',{},'POST')['pause_flush']['request_id']
        def flushed():
            observed=get()
            point=json.loads((replay/'replay.json').read_text())['hold_sample']
            return observed if observed['status']=='paused' and pause_acknowledged(observed,pause_request,point) else None
        observed=budget.wait(flushed,40)
        if start_sample:
            context=established_context(observed,start_sample)
            require(context and context['language']=='en','Actual confident successful English predecessor unavailable')
            preceding={'case':recipe['preceding_context_case'],'job_id':jid,'context':context,'observed':observed}
        else:
            row=next(r for r in observed['document']['segments'] if r['text'].strip())
            result=backend.api('/api/jobs/'+jid+'/segments/'+row['id'],{'segment_revision':row.get('machine_revision',0),'changes':{'text':recipe['correction']}},'PATCH')
            edit=result['segment']
        backend.api('/api/meetings/'+jid+'/resume',{},'POST')
    budget.wait(lambda:(replay/'stage-end').exists(),80)
    pause_request=backend.api('/api/meetings/'+jid+'/pause',{},'POST')['pause_flush']['request_id']
    def settled():
        current=get()
        return current if current['status']=='paused' and pause_acknowledged(current,pause_request,frames) else None
    provisional=budget.wait(settled,40)
    backend.api('/api/jobs/'+jid+'/refinement/resume',{},'POST')
    backend.api('/api/meetings/'+jid+'/stop',{},'POST')
    def finished():
        current=get()
        require(current['status']!='failed','Production meeting failed: '+current.get('message',''))
        return current if current['status']=='ready' else None
    final=budget.wait(finished,70)
    if final['refinement_status'] in ('paused','waiting'):
        backend.api('/api/jobs/'+jid+'/refinement/resume',{},'POST')
        final=budget.wait(lambda:get() if get().get('refinement_status') in ('complete','unresolved') and backend.api('/api/meeting')['status']=='idle' else None,70)
    with wave.open(str(home/'recordings'/jid/'audio.wav'),'rb') as recorded:
        recorded_frames=recorded.getnframes();saved_pcm=recorded.readframes(recorded_frames)
    # Production mixer receives exactly representable PCM16/32768 as float32.
    # Check every saved PCM byte; equal frame count is insufficient.
    expected_pcm=transport_pcm(directory/'replay.wav')
    trace={'job_id':jid,'input_frames':frames,'slice_start_sample':start_sample,'synthesis':json.loads((directory/'synthesis.json').read_text()),'provisional':document_slice(provisional,start_sample),
           'refined':document_slice(final,start_sample),'edit':edit,'preceding':preceding,
           'saved_audio':{'frames':recorded_frames,'sha256':digest_file(home/'recordings'/jid/'audio.wav'),
                          'pcm_sha256':hashlib.sha256(saved_pcm).hexdigest(),'expected_pcm_sha256':hashlib.sha256(expected_pcm).hexdigest()}}
    write(directory/'trace.json',trace)
    assertions=evaluate(recipe,trace)
    return {'id':recipe['id'],'partition':recipe['partition'],'groups':recipe['groups'],'status':'passed' if all(assertions.values()) else 'failed',
            'stages':['provisional','refined'],'assertions':assertions}

def main(args):
    require(sys.platform=='darwin','Apple Silicon macOS required')
    require(not args.output.exists(),'Preserve any previous acceptance attempt')
    manifest=json.loads(args.manifest.read_text());cases,identity=suite();pins=models()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(head==manifest['source_commit'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Freeze exact clean candidate source before acceptance')
    require(manifest.get('signing')=='developer-id' and manifest.get('notarized') is True,'Signed notarized candidate required')
    archive=next(item for item in manifest['files'] if item['filename'].endswith('.app.zip'));artifact(args.manifest.parent/archive['filename'],archive)
    runtime=args.app/'Contents/MacOS/speakerdesk-runtime'
    runtime_hash=digest_file(runtime)
    with zipfile.ZipFile(args.manifest.parent/archive['filename']) as zipped:
        inventory=zip_bundle(zipped)
        for name,expected in inventory.items():
            path=args.app.joinpath(*Path(name).parts[1:]);require(path.is_file() and digest_file(path)==expected,'Executed bundle file differs: '+name)
        require({str(p.relative_to(args.app)) for p in args.app.rglob('*') if p.is_file()}==
                {str(Path(n).relative_to(Path(n).parts[0])) for n in inventory},'Executed bundle inventory differs')
        member=next(n for n in zipped.namelist() if n.endswith('/Contents/MacOS/speakerdesk-runtime'))
        with zipped.open(member) as stream:require(hashlib.file_digest(stream,'sha256').hexdigest()==runtime_hash,'Executed runtime differs from candidate ZIP')
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(args.app)],check=True,capture_output=True,timeout=20)
    details=subprocess.run(['/usr/bin/codesign','-dv','--verbose=4',str(args.app)],check=True,capture_output=True,text=True,timeout=10).stderr
    require('flags=0x10000(runtime)' in details and 'Authority=Developer ID Application:' in details,'Signed hardened app required')
    cdhash=re.search(r'CDHash=([0-9a-f]+)',details)[1]
    budget=Budget(args.lease,args.output);args.output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'scope':'packaged_production_replay','runtime_scope':'frozen_packaged_backend_live_and_refinement','source_commit':head,
            'package_sha256':archive['sha256'],'bundle_sha256':bundle_identity(inventory),'runtime_sha256':runtime_hash,'app_cdhash':cdhash,'signature_verified':True,
            'model_pins':pins,'suite_sha256':identity,'harness_sha256':{name:digest_file(ROOT/name) for name in HARNESSES},
            'status':'incomplete','complete':False,'models_executed':False,'model_substitution':False,'probabilities_modified':False,
            'private_audio':False,'native_capture':False,'cases':[],'owned_processes_stopped':False,'model_files':{}}
    report_path=args.output/'core-acceptance.json';write(report_path,report)
    home=args.output/'home';home.mkdir();(home/'models').mkdir()
    helper=args.output/'replay-helper';import shlex
    helper.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(ROOT/'scripts/replay_capture.py'))+'\n');helper.chmod(0o700)
    backend=None
    try:
        budget.start();expected_metadata=model_file_pins()
        for directory,spec in pins.items():
            folder=args.models/directory;require(folder.is_dir(),'Cached pinned model missing: '+directory)
            artifact(folder/'model.safetensors',spec)
            hashes={name:digest_file(folder/name) for name in spec['files']}
            require(hashes==expected_metadata[directory],'Loaded model/configuration differs: '+directory)
            report['model_files'][directory]=hashes;(home/'models'/directory).symlink_to(folder.resolve(),target_is_directory=True)
        backend=Backend.__new__(Backend)
        backend.__init__(budget,runtime,home,helper,args.output/'runtime.log')
        by_id={c['id']:c for c in cases}
        for recipe in cases:
            directory=args.output/recipe['id'];print('Actual packaged replay: '+recipe['id'],flush=True)
            result=case_run(recipe,by_id,backend,home,directory,budget)
            for key,file in (('input_wav','replay.wav'),('trace','trace.json'),('synthesis','synthesis.json')):
                path=directory/file;result[key]={'file':str(path.relative_to(args.output)),'bytes':path.stat().st_size,'sha256':digest_file(path)}
            saved=home/'recordings'/json.loads((directory/'trace.json').read_text())['job_id']/'audio.wav'
            result['saved_wav']={'file':str(saved.relative_to(args.output)),'bytes':saved.stat().st_size,'sha256':digest_file(saved)}
            report['cases'].append(result);report['models_executed']=True;write(report_path,report)
        report.update(complete=True,status='passed' if all(c['status']=='passed' for c in report['cases']) else 'failed')
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        report['owned_processes_stopped'],errors=cleanup(backend,budget)
        if errors or not report['owned_processes_stopped']:report.update(status='failed',cleanup_errors=errors)
        if budget.failure:report.update(status='failed',error=budget.failure)
        report['budget']={'lease':budget.lease,'peak_owned_rss_bytes':budget.peak,'minimum_free_disk_bytes':budget.minimum_disk,
                          'minimum_free_memory_percent':budget.minimum_memory,'wall_seconds':time.monotonic()-budget.started}
        write(report_path,report)
    require(report['status']=='passed' and report['owned_processes_stopped'],'Core acceptance failed; preserve results and block promotion')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--app',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True);parser.add_argument('--lease',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
