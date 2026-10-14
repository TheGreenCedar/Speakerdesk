const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');

test('polling keeps uploaded progress through queued work and hides it on completion or failure',async()=>{
  const f=frontend(root);f.seed(0,'uploaded');f.document.getElementById('job-progress').hidden=true;
  f.run("selected.kind='upload';window.next=structuredClone(selected);api=async path=>path==='/api/jobs'?[next]:structuredClone(next)");
  for(const status of ['uploaded','preparing','queued','processing','ready','failed']) {
    f.context.status=status;
    f.run("next.status=status;next.message='Current stage';next.revision++");
    await f.run('poll()');
    assert.equal(f.document.getElementById('job-progress').hidden,!['preparing','queued','processing'].includes(status),status);
    assert.equal(f.document.getElementById('job-progress').getAttribute('value'),null);
    assert.equal(f.document.getElementById('job-message').textContent,'Current stage');
  }
});

test('Transcribe shows progress before the request finishes and recovers on rejection',async()=>{
  const f=frontend(root);f.seed(0,'uploaded');f.document.getElementById('job-progress').hidden=true;
  f.run("wire();window.request=null;window.calls=0;api=()=>{calls++;return new Promise((resolve,reject)=>{request={resolve,reject}})};setStatus()");
  const run=f.document.getElementById('run');run.click();
  assert.equal(f.document.getElementById('job-progress').hidden,false);
  assert.equal(f.document.getElementById('job-message').textContent,'Starting transcription…');
  assert.equal(run.disabled,true);
  const pending=run.lastEventResults;
  run.click();assert.equal(f.run('calls'),1);
  f.run("request.reject(Error('Could not start'))");await Promise.all(pending);
  assert.equal(f.document.getElementById('job-progress').hidden,true);
  assert.equal(run.disabled,false);
  assert.match(f.document.getElementById('notice').textContent,/Could not start/);
  run.click();
  f.run("request.resolve({...fixture,status:'queued',message:'Queued'})");
  await Promise.all(run.lastEventResults);
  assert.equal(f.document.getElementById('job-progress').hidden,false);
  assert.equal(f.document.getElementById('job-message').textContent,'Queued');
});

test('late Transcribe replies preserve newer polls and navigation, including reopening the same upload',async()=>{
  for(const navigation of ['poll','other','reopen','home']) {
    const f=frontend(root);f.seed(0,'uploaded');
    f.run("wire();window.request=null;api=path=>path.endsWith('/run')?new Promise(resolve=>{request={resolve}}):Promise.resolve(path==='/api/jobs'?[newer]:structuredClone(newer));setStatus()");
    const run=f.document.getElementById('run');run.click();
    f.context.navigation=navigation;
    f.run(`window.newer={...fixture,status:'ready',message:'Complete',revision:3};
      if(navigation==='other')newer.id='other';
      if(navigation!=='poll') {
        selected=navigation==='home'?null:structuredClone(newer);selectionGeneration++;
        if(selected)setStatus();
      }`);
    if(navigation==='poll')await f.run('poll()');
    f.run("request.resolve({...fixture,status:'queued',message:'Queued',revision:2})");
    await Promise.all(run.lastEventResults);
    assert.equal(f.run('selected?.id'),navigation==='home'?undefined:navigation==='other'?'other':f.context.fixture.id,navigation);
    if(navigation!=='home') {
      assert.equal(f.run('selected.revision'),3,navigation);
      assert.equal(f.document.getElementById('job-progress').hidden,true,navigation);
    }
  }
});

test('passage retry progress follows pending work and terminal candidates without freezing editing or showing on meetings',async()=>{
  for(const result of ['candidate','error']) {
    const f=frontend(root);f.seed(1);
    f.document.getElementById('job-progress').hidden=true;
    f.run("selected.kind='upload';selected.message='Ready';flushSave=async()=>{};window.request=null;api=()=>new Promise(resolve=>{request={resolve}});setStatus();renderSegments()");
    const retry=f.document.getElementById('segments').querySelector('.passage-retry button');
    retry.click();await Promise.resolve();await Promise.resolve();
    assert.equal(f.document.getElementById('job-progress').hidden,false,result);
    assert.match(f.document.getElementById('job-message').textContent,/Retrying passage/);
    f.run("request.resolve({...selected,passage_retry:{id:'pending',segment_id:'r0'}})");
    await Promise.all(retry.lastEventResults);
    assert.equal(f.document.getElementById('job-progress').hidden,false,result);
    assert.equal(f.document.getElementById('editor').inert,false,result);
    assert.match(f.document.getElementById('job-message').textContent,/Retrying passage/);
    f.context.result=result;
    f.run(`window.next=structuredClone(selected);next.revision++;delete next.passage_retry;
      next.document.segments[0].retry_candidate=result==='error'?{id:'pending',error:'Retry failed'}:{id:'pending',text:'Candidate words',language:'en'};
      api=async path=>path==='/api/jobs'?[next]:structuredClone(next)`);
    await f.run('poll()');
    assert.equal(f.document.getElementById('job-progress').hidden,true,result);
    assert.equal(f.document.getElementById('job-message').textContent,'Ready',result);
    assert.equal(f.run('doc.segments[0].text'),'Words 0',result);
    assert.match(f.document.getElementById('segments').textContent,result==='error'?/Retry failed/:/Candidate words/);
  }
  const f=frontend(root);f.seed(0);
  for(const status of ['starting','recording','paused','finishing','processing']) {
    f.context.status=status;
    f.run("selected.kind='meeting';selected.status=status;setStatus()");
    assert.equal(f.document.getElementById('job-progress').hidden,true,status);
  }
});

test('voice separation progress identifies the request and active model work',async()=>{
  for(const [path,message] of [['/separation','Starting voice separation…'],['/cancel','Stopping voice separation…'],['/separation/episode/restore','Restoring the original passage…']]) {
    const f=frontend(root);f.seed(1);
    f.run("selected.kind='upload';selected.message='Ready';flushSave=async()=>{};window.request=null;api=()=>new Promise(resolve=>{request={resolve}})");
    if(path!=='/separation')f.run("selected.passage_separation={id:'pending',segment_id:'r0',phase:'Separating voices and transcribing each estimate…'}");
    f.context.path=path;
    const pending=f.run('separationRequest(path)');await Promise.resolve();await Promise.resolve();
    assert.equal(f.document.getElementById('job-message').textContent,message,path);
    assert.equal(f.document.getElementById('job-progress').hidden,false,path);
    f.run("window.next=structuredClone(selected);next.revision++;next.passage_separation={id:'pending',segment_id:'r0',phase:'Separating voices and transcribing each estimate…'};request.resolve(next)");
    await pending;
    assert.equal(f.document.getElementById('job-message').textContent,'Separating voices and transcribing each estimate…',path);
    assert.equal(f.document.getElementById('job-progress').hidden,false,path);
    assert.equal(f.document.getElementById('editor').inert,false,path);
    f.run("delete selected.passage_separation;setStatus()");
    assert.equal(f.document.getElementById('job-message').textContent,'Ready',path);
    assert.equal(f.document.getElementById('job-progress').hidden,true,path);
  }
});
