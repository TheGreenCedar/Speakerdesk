// Exercise the real setup renderer with a controlled API response; no listener.
const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

test('setup distinguishes missing optional models without changing installed or required labels',async()=>{
  const f=frontend(root);f.seed();
  f.context.setupFixture={status:'idle',ready:true,supported:true,total_bytes:1024,downloaded_bytes:0,
    voice:{status:'idle',available:false,enabled:false},models:[
      {name:'Required',bytes:1024,installed:false,optional:false},
      {name:'Optional alignment',bytes:2048,installed:false,optional:true},
      {name:'Installed alignment',bytes:2048,installed:true,optional:true},
      {name:'Legacy required',bytes:1024,installed:false}
    ]};
  f.run("api=async path=>{if(path!=='/api/setup')throw Error('Unexpected route');return structuredClone(setupFixture)}");
  await f.run('refreshSetup()');
  const labels=f.document.getElementById('model-list').children.map(row=>row.querySelector('span').textContent);
  assert.equal(labels[0],f.run('formatBytes(1024)'));
  assert.equal(labels[1],'Optional · '+f.run('formatBytes(2048)'));
  assert.equal(labels[2],'Installed');assert.equal(labels[3],labels[0]);
  assert.equal(f.run('setupState.models[1].installed'),false);
  assert.equal(f.run('setupState.models[1].optional'),true);
  assert.equal(f.document.getElementById('model-status').textContent,'Ready to transcribe on this Mac.');
});

test('required timing remains actionable and prevents fixed-language admission before readiness',async()=>{
  const f=frontend(root);f.seed();
  f.run("config.languages.ar='Arabic';config.languages.ja='Japanese'");
  f.context.setupFixture={status:'idle',ready:false,core_ready:false,supported:true,total_bytes:100,downloaded_bytes:100,
    voice:{status:'ready',available:false,enabled:false},models:[{id:'coarse-alignment',name:'Transcript timing',bytes:100,installed:false,optional:false}],
    alignment:{id:'coarse-alignment',status:'idle',ready:false,enabled:true,active:false,can_download:true,total_bytes:100,stored_bytes:500,required_free_bytes:600,supported_languages:['en','ar','ja'],timing_accuracy_calibrated_languages:['en'],message:'Download and enable.'}};
  f.context.requests=[];
  f.run("api=async (path,options)=>{if(options){requests.push({path,...options});setupFixture.alignment.status='downloading';setupFixture.alignment.can_download=false;}return structuredClone(setupFixture)}");
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
  const f=frontend(root);f.seed();
  f.context.setupFixture={status:'ready',ready:true,supported:true,total_bytes:1,downloaded_bytes:1,
    voice:{status:'unavailable',available:false,enabled:true,released:false},models:[
      {name:'Voice recognition',bytes:30,installed:true,optional:true}]};
  f.run("api=async()=>structuredClone(setupFixture)");await f.run('refreshSetup()');
  assert.equal(f.document.getElementById('model-list').children[0].querySelector('span').textContent,'Unavailable');
});
