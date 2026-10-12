const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');

test('failed empty capture never promises retained audio or pending playback',()=>{
  const f=frontend(root);f.seed(0,'failed');
  f.run("selected.kind='meeting';selected.duration=0;selected.rolling_refinement={};selected.refinement_status='paused';selected.capture_audio={state:'empty',frames:0};setStatus();renderRefinementStatus()");
  assert.equal(f.document.getElementById('duration').textContent,'No audio');
  assert.equal(f.document.getElementById('refinement-status').textContent,'No audio was recorded.');
  assert.equal(f.document.getElementById('refinement-toggle').hidden,true);
  assert.equal(f.document.querySelector('.player-panel').hidden,true);
  f.run('delete selected.capture_audio;setStatus();renderRefinementStatus()');
  assert.equal(f.document.getElementById('refinement-status').textContent,'No audio was recorded.');
});

test('failed retained capture and live startup keep their existing audio controls',()=>{
  const f=frontend(root);f.seed(0,'failed');
  f.run("selected.kind='meeting';selected.duration=3;selected.rolling_refinement={};selected.refinement_status='paused';selected.capture_audio={state:'retained',frames:48000};setStatus();renderRefinementStatus()");
  assert.equal(f.document.querySelector('.player-panel').hidden,false);
  assert.equal(f.document.getElementById('refinement-status').textContent,'Accuracy pass paused · audio kept for later');
  assert.equal(f.document.getElementById('refinement-toggle').hidden,false);
  f.run("selected.status='starting';selected.duration=0;delete selected.capture_audio;setStatus()");
  assert.equal(f.document.getElementById('duration').textContent,'Preparing…');
  assert.equal(f.document.querySelector('.player-panel').hidden,true);
});

test('unreadable retained audio is reported as unverified rather than kept or empty',()=>{
  const f=frontend(root);f.seed(0,'failed');
  f.run("selected.kind='meeting';selected.duration=0;selected.rolling_refinement={};selected.refinement_status='paused';selected.capture_audio={state:'unverified'};renderRefinementStatus()");
  assert.equal(f.document.getElementById('refinement-status').textContent,'Recorded audio could not be verified.');
  assert.equal(f.document.getElementById('refinement-toggle').hidden,true);
});

test('saved accuracy controls start, pause and retry through the existing routes',async()=>{
  const f=frontend(root);f.seed(1,'ready');
  f.run("selected.kind='meeting';selected.saved_refinement_available=true;selected.refinement_status='unavailable';delete selected.rolling_refinement;window.calls=[];api=async(path,options)=>{calls.push({path,method:options.method});return {};};poll=async()=>{};");
  for(const [status,label,action] of [
    ['unavailable','Start saved accuracy pass','resume'],
    ['refining','Pause accuracy pass','pause'],
    ['unresolved','Retry unresolved phrases','resume'],
  ]) {
    f.context.nextStatus=status;
    f.run("selected.refinement_status=nextStatus;if(nextStatus!=='unavailable')selected.rolling_refinement={mode:'saved'};renderRefinementStatus()");
    assert.equal(f.document.getElementById('refinement-controls').hidden,false);
    assert.equal(f.document.getElementById('refinement-toggle').textContent,label);
    if(status==='refining')assert.equal(f.document.getElementById('refinement-status').textContent,'Rechecking saved passages…');
    assert.match(f.document.getElementById('refinement-status').title,/captured Auto or explicit language/);
    assert.match(f.document.getElementById('refinement-status').title,/uncertain or mixed-language/);
    assert.match(f.document.getElementById('refinement-status').title,/one confidently routed language/);
    assert.match(f.document.getElementById('refinement-status').title,/Non-English word times are uncalibrated estimates/);
    await f.run('toggleRefinement()');
    assert.equal(f.run('calls.at(-1).path'),`/api/jobs/${'d'.repeat(32)}/refinement/${action}`);
    assert.equal(f.run('calls.at(-1).method'),'POST');
    assert.equal(f.document.getElementById('refinement-toggle').disabled,false);
  }
});


test('live accuracy controls preserve recording and target the accuracy routes',async()=>{
  const f=frontend(root);f.seed(1,'recording');
  f.run("selected.kind='meeting';selected.rolling_refinement={mode:'live'};window.calls=[];api=async(path,options)=>{calls.push({path,method:options.method});return {};};poll=async()=>{};");
  for(const [status,action] of [['waiting','pause'],['refining','pause'],['paused','resume']]) {
    f.context.nextStatus=status;f.run('selected.refinement_status=nextStatus;renderRefinementStatus()');
    assert.equal(f.document.getElementById('refinement-controls').hidden,false);
    assert.equal(f.document.getElementById('refinement-toggle').hidden,false);
    assert.match(f.document.getElementById('refinement-status').title,/while recording continues/);
    if(status==='refining')assert.equal(f.document.getElementById('refinement-status').textContent,'Rechecking sealed passages…');
    await f.run('toggleRefinement()');
    assert.equal(f.run('calls.at(-1).path'),`/api/jobs/${'d'.repeat(32)}/refinement/${action}`);
    assert.equal(f.run('selected.status'),'recording');
  }
});
