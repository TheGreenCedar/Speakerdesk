// Existing editor controller with a synthetic DOM, not desktop acceptance.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {frontend} = require('./support/frontend_dom.cjs');

test('explicit-language import can run while native automatic language and alignment are unavailable', () => {
  const f = frontend(path.resolve(__dirname, '..'));f.seed(0, 'uploaded');
  f.run("config.readiness={configured:true,automatic_language:false,word_alignment:false};setupState={core_ready:true,alignment:{ready:false}};setStatus();");
  assert.equal(f.document.getElementById('run').disabled, false);
  f.run("selected.language='auto';setStatus();");
  assert.equal(f.document.getElementById('run').disabled, true);
  f.run("selected.language='en';config.readiness.configured=false;setStatus();");
  assert.equal(f.document.getElementById('run').disabled, true);
});

test('required alignment on the existing runtime still blocks import until ready', () => {
  const f = frontend(path.resolve(__dirname, '..'));f.seed(0, 'uploaded');
  f.run("setupState={core_ready:true,alignment:{ready:false}};setStatus();");
  assert.equal(f.document.getElementById('run').disabled, true);
  f.run("setupState.alignment.ready=true;setStatus();");
  assert.equal(f.document.getElementById('run').disabled, false);
});

test('setup polling enables native import after initialization with no download total', async () => {
  const f = frontend(path.resolve(__dirname, '..'));f.seed(0, 'uploaded');
  f.run("config.readiness.configured=false;setStatus();");
  const progress=f.document.getElementById('model-progress');let value=0;
  // WebKit rejects non-finite HTMLProgressElement values; the lightweight DOM does not.
  Object.defineProperty(progress,'value',{get:()=>value,set:next=>{
    if(!Number.isFinite(next))throw new TypeError('The provided value is non-finite');value=next;
  }});
  f.context.fetch=async url=>({ok:true,json:async()=>url==='/api/setup'?{
    core_ready:true,ready:true,status:'ready',automatic_language:false,models:[],
    total_bytes:0,downloaded_bytes:0,supported:true,can_download:false,
    voice:{status:'unavailable',available:false,enabled:false,message:'Unavailable'}
  }:{languages:{en:'English',auto:'Automatic'},default_language:'en',
    readiness:{configured:true,automatic_language:false,word_alignment:false,live_capture:false}}});
  await f.run('refreshSetup()');
  assert.equal(f.document.getElementById('model-status').textContent,'Ready to transcribe on this Mac.');
  assert.equal(f.document.getElementById('run').disabled,false);
  assert.equal(progress.value,0);
  assert.equal(f.document.getElementById('install-models').hidden,true);
});

test('setup completion refreshes timing capability even when manual transcription was already ready', async () => {
  const f=frontend(path.resolve(__dirname,'..')); f.seed(0,'uploaded');
  f.run('config.readiness={configured:true,automatic_language:false,word_alignment:false};');
  let configReads=0;
  f.context.fetch=async url=>({ok:true,json:async()=>url==='/api/setup'?{
    core_ready:true,ready:true,status:'ready',automatic_language:false,models:[],
    total_bytes:6,downloaded_bytes:6,supported:true,can_download:true,can_cancel:false,
    alignment:{id:'coarse-alignment',status:'ready',ready:true,supported_languages:['en'],timing_accuracy_calibrated_languages:['en'],total_bytes:6,stored_bytes:6,required_free_bytes:6,message:'Ready'},
    voice:{status:'unavailable',available:false,enabled:false,message:'Unavailable'}
  }:(configReads++,{languages:{en:'English',auto:'Automatic'},default_language:'en',readiness:{configured:true,automatic_language:false,word_alignment:true,live_capture:false}})});
  await f.run('refreshSetup()');
  assert.equal(configReads,1);
  assert.equal(f.run('config.readiness.word_alignment'),true);
  assert.equal(f.document.getElementById('cancel-model-setup').hidden,true);
});
