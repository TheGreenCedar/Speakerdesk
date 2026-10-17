// Exercise the real setup renderer with a controlled API response; no listener.
const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

test('all shipped models use required setup labels, including stale optional flags',async()=>{
  const f=frontend(root);f.seed();f.document.getElementById('setup-card-install').append(f.document.createElement('span'));
  f.context.setupFixture={status:'idle',ready:false,core_ready:false,supported:true,total_bytes:1024,downloaded_bytes:0,
    voice:{status:'ready',available:false,enabled:false},separation:{installed:false,ready:false,active:false},models:[
      {id:'cohere-speech',name:'Speech',bytes:1024,installed:true},
      {id:'mossformer2-16k',name:'Voice separation',bytes:2048,installed:false,optional:true}
    ]};
  f.run("api=async path=>path==='/api/config'?{...config,readiness:{...config.readiness,configured:false}}:structuredClone(setupFixture)");
  await f.run('refreshSetup()');
  const rows=f.document.getElementById('model-list').children;
  assert.equal(rows[0].querySelector('span').textContent,'Installed');
  assert.equal(rows[1].querySelector('span').textContent,f.run('formatBytes(2048)'));
  assert.equal(rows[1].querySelector('button'),null);
  assert.equal(f.document.getElementById('setup-card').hidden,false);
  assert.equal(f.document.getElementById('start-meeting').disabled,true);
});

test('required timing remains actionable and prevents fixed-language admission before readiness',async()=>{
  const f=frontend(root);f.seed();f.document.getElementById('setup-card-install').append(f.document.createElement('span'));
  f.run("config.languages.ar='Arabic';config.languages.ja='Japanese'");
  f.context.setupFixture={status:'idle',ready:false,core_ready:false,supported:true,total_bytes:100,downloaded_bytes:100,
    voice:{status:'ready',available:false,enabled:false},models:[{id:'coarse-alignment',name:'Transcript timing',bytes:100,installed:false,optional:false}],
    alignment:{id:'coarse-alignment',status:'idle',ready:false,enabled:true,active:false,can_download:true,total_bytes:100,stored_bytes:500,required_free_bytes:600,supported_languages:['en','ar','ja'],timing_accuracy_calibrated_languages:['en'],message:'Download and enable.'}};
  f.context.requests=[];
  f.run("api=async (path,options)=>{if(path==='/api/config')return {...config,readiness:{...config.readiness,configured:setupFixture.core_ready}};if(options){requests.push({path,...options});setupFixture.alignment.status='downloading';setupFixture.alignment.can_download=false;}return structuredClone(setupFixture)}");
  f.run('wire()');await f.run('refreshSetup()');
  const action=f.document.getElementById('install-alignment');
  assert.equal(f.document.getElementById('install-models').hidden,false);
  assert.equal(f.run('languageReady("en")'),false);
  assert.equal(f.run('languageReady("auto")'),false);
  assert.equal(action.hidden,false);assert.equal(action.disabled,false);
  assert.equal(f.document.getElementById('alignment-coverage').textContent,'Supported languages: English, Arabic, Japanese. Timing accuracy measured against references: English. Accuracy in other supported languages has not been independently measured.');
  action.click();await Promise.all(action.lastEventResults);
  assert.equal(f.context.requests[0].path,'/api/setup/alignment');assert.equal(f.context.requests[0].method,'POST');
  f.run("setupFixture.alignment={...setupFixture.alignment,status:'failed',error:'Interrupted',can_download:true}");await f.run('refreshSetup()');
  assert.equal(action.textContent,'Retry setup');assert.equal(f.document.getElementById('alignment-status').textContent,'Interrupted');
  f.run("setupFixture.ready=true;setupFixture.core_ready=true;setupFixture.alignment={...setupFixture.alignment,status:'ready',error:null,ready:true,enabled:true}");await f.run('refreshSetup()');
  assert.equal(action.hidden,true);
  assert.equal(f.run('languageReady("en")'),true);
  assert.equal(f.context.requests.length,1);
  const template=require('node:fs').readFileSync(path.join(root,'speakerdesk/templates/index.html'),'utf8');
  assert(!template.includes('id="enable-alignment"'));
});


test('cached unqualified voice weights are unavailable rather than usable',async()=>{
  const f=frontend(root);f.seed();f.document.getElementById('setup-card-install').append(f.document.createElement('span'));
  f.context.setupFixture={status:'ready',ready:true,supported:true,total_bytes:1,downloaded_bytes:1,
    voice:{status:'unavailable',available:false,enabled:true,released:false},models:[
      {name:'Voice recognition',bytes:30,installed:true,optional:true}]};
  f.run("api=async()=>structuredClone(setupFixture)");await f.run('refreshSetup()');
  assert.equal(f.document.getElementById('model-list').children[0].querySelector('span').textContent,'Unavailable');
});

for(const platform of ['macos','windows'])for(const scenario of ['fresh','upgrade','failed','interrupted'])test(`${platform}: ${scenario} separation uses the shared required setup and continues after repair`,async()=>{
  const f=frontend(root,{platform});f.seed();f.document.getElementById('setup-card-install').append(f.document.createElement('span'));
  f.context.scenario=scenario;
  f.run(`window.requests=[];window.setupFixture={status:['failed','interrupted'].includes(scenario)?'failed':'idle',ready:false,core_ready:false,supported:true,can_download:true,total_bytes:670353371,downloaded_bytes:0,
    operation:['failed','interrupted'].includes(scenario)?{target:'separation',status:scenario,error:'Connection interrupted'}:null,
    voice:{status:'ready',available:false,enabled:false},separation:{installed:scenario==='failed',ready:false,active:false},
    models:[{id:'cohere-speech',name:'Transcription',installed:scenario!=='fresh',bytes:100},{id:'mossformer2-16k',name:'Voice separation',installed:scenario==='failed',optional:true,bytes:670353271}]};
    api=async(path,options)=>{
      if(path==='/api/config')return {...config,readiness:{...config.readiness,configured:setupFixture.core_ready}};
      if(options){requests.push({path,options});setupFixture.status='downloading';setupFixture.separation.active=true;}
      return structuredClone(setupFixture);
    };wire();`);
  await f.run('refreshSetup()');
  assert.equal(f.run('requests.length'),0,'setup remains an explicit download action');
  assert.equal(f.run('languageReady("en")'),false);
  const action=f.document.getElementById('install-models');
  assert.equal(action.hidden,false);assert.equal(action.disabled,false);
  if(['failed','interrupted'].includes(scenario)){
    assert.equal(action.textContent,'Retry setup');
    assert.match(f.document.getElementById('model-error').textContent,/Connection interrupted/);
  }
  action.click();await Promise.all(action.lastEventResults);
  assert.equal(f.run('requests[0].path'),'/api/setup');assert.equal(f.run('requests[0].options.method'),'POST');
  assert.equal(action.disabled,true);assert.equal(f.document.getElementById('start-meeting').disabled,true);
  f.run("setupFixture.status='ready';setupFixture.ready=true;setupFixture.core_ready=true;setupFixture.separation={installed:true,ready:true,active:false};setupFixture.models.forEach(model=>model.installed=true)");
  await f.run('refreshSetup()');
  assert.equal(action.hidden,true);assert.equal(f.document.getElementById('setup-card').hidden,true);
  assert.equal(f.document.getElementById('start-meeting').disabled,false);
  assert.equal(f.run('languageReady("en")'),true);
  assert.equal(f.run('requests.length'),1);
  assert.match(f.document.getElementById('start-separation-readiness').textContent,/is ready/);
});

test('installed but uninitialized separation needs shared setup',async()=>{
  const f=frontend(root);f.seed();f.document.getElementById('setup-card-install').append(f.document.createElement('span'));f.run("setupState={separation:{installed:true,ready:false,active:false}};renderSeparationReadiness()");
  assert.match(f.document.getElementById('start-separation-readiness').textContent,/needs setup/);
  assert.doesNotMatch(f.document.getElementById('meeting-separation-readiness').textContent,/is ready/);
  assert.equal(f.run('languageReady("en")'),false);
});

test('current readiness wins over retained failed setup history after restart',async()=>{
  for(const target of ['separation','models']) {
    const f=frontend(root);f.seed();f.document.getElementById('setup-card-install').append(f.document.createElement('span'));f.context.fixtureTarget=target;
    f.run(`window.setupFixture={status:'ready',ready:true,core_ready:true,supported:true,total_bytes:100,downloaded_bytes:100,
      operation:{target:fixtureTarget,status:'failed',error:'Previous initialization failed'},
      voice:{status:'ready',available:false,enabled:false},separation:{installed:true,ready:true,active:false},
      models:[{id:'mossformer2-16k',name:'Voice separation',installed:true,optional:true,bytes:100}]};
      api=async()=>structuredClone(setupFixture)`);
    await f.run('refreshSetup()');
    assert.equal(f.document.getElementById('model-list').querySelector('span').textContent,'Installed');
    assert.equal(f.document.getElementById('install-models').hidden,true);
    assert.equal(f.document.getElementById('model-status').textContent,'Ready to transcribe on this Mac.');
    assert.match(f.document.getElementById('model-error').textContent,/Previous setup attempt: Previous initialization failed/);
  }
});
