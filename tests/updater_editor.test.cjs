// Real editor/save/update controller with synthetic DOM and HTTP transport only.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {frontend} = require('./support/frontend_dom.cjs');
const root = path.resolve(__dirname, '..');

function editor() {
  const f = frontend(root);f.seed();
  f.context.setTimeout = setTimeout;f.context.clearTimeout = clearTimeout;
  f.context.AbortController = AbortController;
  const calls = [];
  f.context.fetch = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;calls.push({url,body});
    if (url.endsWith('/transcript')) {
      return {ok:true,json:async()=>({...f.context.fixture,revision:2,document:body.document})};
    }
    return {ok:true,json:async()=>({id:1,state:'preparing',preparation:1,reserved:false})};
  };
  f.run("updateState={id:1,state:'preparing',preparation:1,reserved:false};");
  return {...f,calls};
}

test('install acknowledges only after real transcript autosave completes', async()=>{
  const f=editor();f.run("doc.segments[0].text='My exact saved words';changed();");
  await f.run('prepareEditorUpdate(1,1)');
  assert.deepEqual(f.calls.map(c=>c.body?.op || c.url),['/api/jobs/'+ 'd'.repeat(32) +'/transcript','editor_ready']);
  assert.equal(f.calls[0].body.document.segments[0].text,'My exact saved words');
  assert.equal(f.run('dirty || saving || localMutations > 0'),false);
  assert.equal(f.run('updateFrozen'),true,'The editor remains protected until native completion/cancellation.');
});

test('failed save retains edited words and never acknowledges installation', async()=>{
  const f=editor(),transport=f.context.fetch;
  f.context.fetch=async(url,options)=>url.endsWith('/transcript')?{ok:false,json:async()=>({error:'Fixture disk write refused'})}:transport(url,options);
  f.run("doc.segments[0].text='Still my words';changed();");
  await f.run('prepareEditorUpdate(1,1)');
  assert.deepEqual(f.calls.map(c=>c.body?.op),['editor_error']);
  assert.equal(f.run('dirty'),true);assert.equal(f.run('doc.segments[0].text'),'Still my words');
  assert.equal(f.run('updateFrozen'),true);
  f.context.fetch=async()=>({ok:true,json:async()=>({id:1,state:'error',reserved:false,error:'Fixture disk write refused'})});
  await f.run('refreshUpdates()');
  assert.equal(f.run('updateFrozen'),false,'Confirmed native refusal restores editing.');
  assert.equal(f.run('dirty'),true);
  f.run('clearTimeout(autosaveTimer)');
});

test('draft, retained correction, pending passage save, undo and open name edit are preserved',async()=>{
  for(const seed of [
    "passageDrafts.set('r0',{text:'Local draft'});",
    "recoverablePassageDrafts.set('r0',{text:'Retained correction'});",
    "passageSaveOperations.set(selected.id+':r0',{jid:selected.id});",
    "undoRemoval={jid:selected.id,timer:null};",
    "const dialog=document.createElement('dialog');dialog.setAttribute('id','name-dialog');dialog.setAttribute('open','');document.body.append(dialog);"
  ]) {
    const f=editor();f.run(seed);const before=f.run('JSON.stringify(doc)');
    await f.run('prepareEditorUpdate(1,1)');
    assert.deepEqual(f.calls.map(c=>c.body?.op),['editor_error'],seed);
    assert.equal(f.run('JSON.stringify(doc)'),before);
    assert.equal(f.run('passageDrafts.size + recoverablePassageDrafts.size + passageSaveOperations.size + Number(!!undoRemoval) + Number(!!document.querySelector("dialog[open]"))'),1);
  }
});

test('recording, paused and finishing captures refuse both download and installation',async()=>{
  for(const state of ['starting','recording','paused','finishing']) {
    const f=editor();f.context.captureState=state;f.run('meeting={status:captureState};');
    await assert.rejects(f.run("requestUpdate('download')"),/Finish the recording/);
    await f.run('prepareEditorUpdate(1,1)');
    assert.deepEqual(f.calls.map(c=>c.body?.op),['editor_error']);
  }
});

test('edit epoch changed during save refuses the current install attempt',async()=>{
  const f=editor(),transport=f.context.fetch;
  f.context.fetch=async(url,options)=>{
    const result=await transport(url,options);
    if(url.endsWith('/transcript')) f.run("editGeneration++;doc.segments[1].text='Newer words';");
    return result;
  };
  // Avoid retrying a save forever: the save response clears only its own epoch.
  f.run('dirty=true;');
  const original=f.context.fetch;
  let once=true;
  f.context.fetch=async(url,options)=>{
    if(url.endsWith('/transcript') && once){once=false;return original(url,options);}
    return transport(url,options);
  };
  await f.run('prepareEditorUpdate(1,1)');
  assert.equal(f.calls.some(c=>c.body?.op==='editor_ready'),false);
  assert.equal(f.calls.at(-1).body.op,'editor_error');
});

test('late save after cancellation does not acknowledge a newer attempt',async()=>{
  const f=editor(),transport=f.context.fetch;let finish;
  f.context.fetch=async(url,options)=>{
    if(url.endsWith('/transcript')) await new Promise(resolve=>{finish=resolve;});
    return transport(url,options);
  };
  f.run('dirty=true;');const pending=f.run('prepareEditorUpdate(1,1)');
  await Promise.resolve();
  f.run("updateState={id:2,state:'available',reserved:false};");finish();await pending;
  assert.equal(f.calls.some(c=>['editor_ready','editor_error'].includes(c.body?.op)),false);
});

test('late save cannot acknowledge a newer preparation of the same downloaded update',async()=>{
  const f=editor(),transport=f.context.fetch;let finish;
  f.context.fetch=async(url,options)=>{
    if(url.endsWith('/transcript')) await new Promise(resolve=>{finish=resolve;});
    return transport(url,options);
  };
  f.run('dirty=true;');const pending=f.run('prepareEditorUpdate(1,1)');await Promise.resolve();
  f.run("updateState={id:1,state:'preparing',preparation:2,reserved:false};");finish();await pending;
  assert.equal(f.calls.some(c=>['editor_ready','editor_error'].includes(c.body?.op)),false);
});

test('interrupted save times out without losing edits or acknowledging installation',async()=>{
  const f=editor(),transport=f.context.fetch;let finish;
  f.context.setTimeout=(callback,ms)=>setTimeout(callback,ms===15000?5:ms);
  f.context.fetch=async(url,options)=>{
    if(url.endsWith('/transcript')) await new Promise(resolve=>{finish=resolve;});
    return transport(url,options);
  };
  f.run("doc.segments[0].text='Unfinished save words';dirty=true;");
  await f.run('prepareEditorUpdate(1,1)');
  assert.equal(f.calls.some(c=>c.body?.op==='editor_ready'),false);
  assert.equal(f.calls.at(-1).body.op,'editor_error');
  assert.equal(f.run('dirty && saving'),true);
  assert.equal(f.run('doc.segments[0].text'),'Unfinished save words');
  finish();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.calls.some(c=>c.body?.op==='editor_ready'),false,'A late successful save cannot resume the aborted preparation.');
});

test('expired preparation never starts a later save or clears a new Undo window',async()=>{
  const f=editor();f.context.setTimeout=(callback,ms)=>setTimeout(callback,ms===15000?5:ms);
  f.run('dirty=true;localMutations=1;');await f.run('prepareEditorUpdate(1,1)');
  f.context.fetch=async()=>({ok:true,json:async()=>({id:1,state:'ready',preparation:1,reserved:false})});
  await f.run('refreshUpdates()');
  f.run("doc.segments[0].text='New local words';undoRemoval={jid:selected.id,timer:null};localMutations=0;");
  await new Promise(resolve=>setTimeout(resolve,65));
  assert.equal(f.calls.some(c=>c.url.endsWith('/transcript')),false);
  assert.equal(f.run('dirty && !!undoRemoval'),true);
  assert.equal(f.run('doc.segments[0].text'),'New local words');
});

test('refusing an update preserves the actual passage Undo action and exact words',async()=>{
  const f=editor();const before=f.run('JSON.stringify(doc)');
  f.run('removePassage(doc.segments[0]);');
  const undo=f.document.querySelector('#notice .notice-action');assert.ok(undo);
  await f.run('prepareEditorUpdate(1,1)');
  assert.equal(f.document.querySelector('#notice .notice-action'),undo);
  f.context.fetch=async()=>({ok:true,json:async()=>({id:1,state:'ready',preparation:1,reserved:false})});
  await f.run('refreshUpdates()');f.run('closeSettings()');undo.click();
  assert.equal(f.run('JSON.stringify(doc)'),before);
  assert.equal(f.run('undoRemoval'),null);f.run('clearTimeout(autosaveTimer)');
});

test('a stalled refusal keeps the actual Undo action and recording controls usable',async()=>{
  const f=editor();const before=f.run('JSON.stringify(doc)');let finish;
  f.run('removePassage(doc.segments[0]);');
  const undo=f.document.querySelector('#notice .notice-action');assert.ok(undo);
  f.context.fetch=async()=>{
    await new Promise(resolve=>{finish=resolve;});
    return {ok:true,json:async()=>({id:1,state:'ready',preparation:1,reserved:false})};
  };
  const refusing=f.run('prepareEditorUpdate(1,1)');await Promise.resolve();
  assert.equal(f.run('updateFrozen'),false,'No installation permission was given; local edit controls must remain usable.');
  assert.equal(!!f.document.getElementById('settings-close').disabled,false);
  f.run('closeSettings()');undo.click();
  assert.equal(f.run('JSON.stringify(doc)'),before);assert.equal(f.run('undoRemoval'),null);
  finish();await refusing;f.run('clearTimeout(autosaveTimer)');
  for(const state of ['recording','paused','finishing']) {
    const live=editor();live.context.captureState=state;live.run('meeting={status:captureState};');
    await live.run('prepareEditorUpdate(1,1)');assert.equal(live.run('updateFrozen'),false);
  }
});

test('reserved runtime and shutdown errors retain UI protection; successful cancellation releases it',async()=>{
  const f=editor();f.run('freezeForUpdate(true);');
  f.context.fetch=async()=>({ok:true,json:async()=>({id:1,state:'error',reserved:true,error:'Shutdown timeout'})});
  await f.run('refreshUpdates()');assert.equal(f.run('updateFrozen'),true);
  await assert.rejects(f.run("api('/api/jobs',{method:'POST',body:'{}'})"),/Cancel the update/);
  f.context.fetch=async()=>({ok:true,json:async()=>({id:1,state:'ready',reserved:false})});
  await f.run('refreshUpdates()');assert.equal(f.run('updateFrozen'),false);
});

test('native preparation keeps Cancel available while its save is protected',async()=>{
  const f=editor();f.run("freezeForUpdate(true);updateState={id:1,state:'preparing',preparation:1,reserved:true,cancellable:true};renderUpdates();");
  assert.equal(f.document.getElementById('update-cancel').disabled,false);
  assert.equal(f.document.getElementById('settings-close').disabled,true);
  f.run("updateState={id:1,state:'stopping',preparation:1,reserved:true,cancellable:false};renderUpdates();");
  assert.equal(f.document.getElementById('update-cancel').disabled,true);
});

test('stalled update POST times out and leaves Cancel usable for retry',async()=>{
  const f=editor();let signal;
  f.run("updateState={id:1,state:'available',reserved:false};");
  f.context.setTimeout=(callback,ms)=>setTimeout(callback,ms===10000?5:ms);
  f.context.fetch=async(_url,options)=>{signal=options.signal;return await new Promise(()=>{});};
  await assert.rejects(f.run("requestUpdate('download')"),/did not respond/);
  assert.equal(signal.aborted,true);assert.equal(f.run('updateRequesting'),false);
  const operations=[];
  f.context.fetch=async(_url,options={})=>{
    if(options.body)operations.push(JSON.parse(options.body).op);
    return {ok:true,json:async()=>({id:1,state:'available',reserved:false})};
  };
  await f.run("requestUpdate('cancel')");assert.deepEqual(operations,['cancel']);
});

test('stalled update response body cannot block later status polls',async()=>{
  const f=editor();f.context.setTimeout=(callback,ms)=>setTimeout(callback,ms===10000?5:ms);
  f.context.fetch=async()=>({ok:true,json:()=>new Promise(()=>{})});
  await f.run('refreshUpdates()');assert.equal(f.run('updatePolling'),false);
  f.context.fetch=async()=>({ok:true,json:async()=>({id:1,state:'ready',reserved:false})});
  await f.run('refreshUpdates()');assert.equal(f.run('updateState.state'),'ready');
});
