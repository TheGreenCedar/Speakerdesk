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

test('first setup keeps native diagnostics in Details and failures stay recoverable', async () => {
  const raw='Speech detection: Silero: No such file or directory (os error 2) Speaker diarization: Nemotron: No such file or directory (os error 2) Transcription: Cohere: No such file or directory (os error 2)';
  for(const [status,download_required,error,expected,action,supported=true] of [
    ['missing',true,raw,'Download the local models once. Meetings are transcribed offline on this Mac.','Download models · 2.2 GB'],
    ['failed',true,'HTTP 503 while downloading model weights','Model setup did not finish. Try again, or open Details for more information.','Retry setup'],
    ['failed',true,'Model checksum differs from the pinned artifact','Model setup did not finish. Try again, or open Details for more information.','Retry setup'],
    ['failed',true,'No space left on device (os error 28)','Model setup did not finish. Try again, or open Details for more information.','Retry setup'],
    ['failed',true,'Model setup cancelled. Downloaded files can be resumed.','Model setup did not finish. Try again, or open Details for more information.','Retry setup'],
    ['failed',false,'Metal model initialization failed','Model setup did not finish. Try again, or open Details for more information.','Retry setup'],
    ['missing',true,raw,'This Mac can’t run the local models. Speakerdesk needs Apple Silicon and macOS 15 or later.','Download models · 2.2 GB',false],
  ]) {
    const f=frontend(path.resolve(__dirname,'..'));f.seed(0,'uploaded');
    f.document.getElementById('setup-card-install').append(f.document.createElement('span'));
    f.context.setupFixture={status,download_required,error,message:error,core_ready:false,ready:false,supported,
      automatic_language:false,total_bytes:2.2e9,downloaded_bytes:0,can_download:true,models:[],
      voice:{status:'unavailable',available:false,enabled:false}};
    f.run("config.readiness={configured:false,automatic_language:false,word_alignment:false};$('language').value='en';api=async()=>structuredClone(setupFixture)");
    await f.run('refreshSetup()');
    const el=id=>f.document.getElementById(id);
    assert.equal(el('setup-card-text').textContent,expected,status);
    assert.equal(el('setup-card-status').textContent,'',status);
    assert.equal(el('model-status').textContent,expected,status);
    assert.equal(el('setup-card-install').querySelector('span').textContent,action,status);
    assert.equal(el('install-models').textContent,action,status);
    assert.equal(el('setup-card-install').disabled,!supported,status);
    assert.equal(el('model-error').textContent,error,status);
    assert.equal(el('model-error-details').hidden,false,status);
    assert.equal(el('start-meeting').disabled,true,status);
  }
});

test('model loading hides download actions and progress across the start card and Settings', async () => {
  const f=frontend(path.resolve(__dirname,'..'));f.seed(0,'uploaded');
  f.document.getElementById('setup-card-install').append(f.document.createElement('span'));
  f.run("config.readiness={configured:false,automatic_language:false,word_alignment:false,live_capture:true};meeting={capture_available:true};$('language').value='en';");
  const state={core_ready:false,ready:false,status:'loading',phase:'Opening installed artifacts',
    automatic_language:false,models:[{id:'silero-speech',name:'Speech detection',bytes:100,installed:false}],
    total_bytes:100,downloaded_bytes:0,required_free_bytes:200,supported:true,can_download:false,can_cancel:true,
    alignment:{id:'coarse-alignment',status:'loading',ready:false,can_download:false,supported_languages:['en'],
      timing_accuracy_calibrated_languages:['en'],total_bytes:100,stored_bytes:100,required_free_bytes:200,
      message:'Use Download models below to finish setup for all local models.'},
    voice:{status:'unavailable',available:false,enabled:false,message:'Unavailable'}};
  f.context.fetch=async url=>({ok:true,json:async()=>url==='/api/setup'?state:{languages:{en:'English',auto:'Automatic'},
    default_language:'en',readiness:{configured:state.core_ready,automatic_language:false,word_alignment:state.alignment.ready,live_capture:true}}});
  const el=id=>f.document.getElementById(id);
  await f.run('refreshSetup()');
  assert.equal(el('setup-card').hidden,true);
  assert.equal(el('setup-card-install').hidden,true);
  assert.equal(el('install-models').hidden,true);
  assert.equal(el('install-alignment').hidden,true);
  for(const id of ['model-progress','alignment-progress','setup-disk-note','alignment-cost'])
    assert.equal(el(id).hidden,true,id);
  for(const id of ['start-hint','alignment-status'])
    assert.match(el(id).textContent,/Loading models…/,id);
  assert.equal(el('model-status').textContent,state.phase);
  assert.doesNotMatch(el('alignment-message').textContent,/download/i);
  assert.match(el('model-list').textContent,/Loading/);
  assert.equal(el('start-meeting').disabled,true);
  f.run('setStatus()');
  assert.equal(el('run').title,'Loading models…');
  assert.equal(el('cancel-model-setup').hidden,false);

  state.phase=state.alignment.phase='Cancelling model setup';
  await f.run('refreshSetup()');
  assert.equal(el('model-status').textContent,'Cancelling model setup');
  assert.equal(el('alignment-status').textContent,'Cancelling model setup');
  assert.equal(el('setup-card').hidden,true);
  assert.equal(el('start-hint').textContent,'Loading models…');

  // A real missing-file transfer still exposes its progress and cancel action.
  Object.assign(state,{status:'downloading',phase:'Downloading speech detection',downloaded_bytes:25});
  state.alignment.status='downloading';
  state.alignment.phase=state.phase;
  await f.run('refreshSetup()');
  assert.equal(el('setup-card').hidden,false);
  assert.equal(el('install-models').hidden,false);
  assert.equal(el('setup-card-install').hidden,false);
  assert.equal(el('model-progress').hidden,false);
  assert.equal(el('model-progress').value,25);
  assert.match(el('model-status').textContent,/Downloading speech detection · 25%/);
  assert.equal(el('cancel-model-setup').hidden,false);

  // Initializer failure does not establish that the installed files need another download.
  Object.assign(state,{status:'failed',error:'Model initialization failed.',can_download:true,can_cancel:false});
  state.alignment.status='idle';
  await f.run('refreshSetup()');
  assert.equal(el('setup-card').hidden,false);
  assert.equal(el('install-models').textContent,'Retry setup');
  assert.equal(el('setup-card-install').querySelector('span').textContent,'Retry setup');
  assert.match(el('model-status').textContent,/setup did not finish/);
  assert.equal(el('model-error').textContent,state.error);
  assert.equal(el('cancel-model-setup').hidden,true);

  // A catalog presence check can positively distinguish a missing download from a load failure.
  Object.assign(state,{status:'missing',download_required:true,error:'Required model files are missing.'});
  await f.run('refreshSetup()');
  assert.match(el('install-models').textContent,/^Download models · /);
  assert.match(el('setup-card-install').querySelector('span').textContent,/^Download models · /);
  assert.equal(el('setup-disk-note').hidden,false);

  Object.assign(state,{status:'ready',error:null,core_ready:true,ready:true,download_required:false});
  Object.assign(state.alignment,{status:'ready',ready:true});
  await f.run('refreshSetup()');
  assert.equal(el('start-meeting').disabled,false);
  assert.equal(el('setup-card').hidden,true);
  assert.equal(el('install-models').hidden,true);
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
