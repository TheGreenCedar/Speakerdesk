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
