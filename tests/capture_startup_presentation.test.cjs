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
