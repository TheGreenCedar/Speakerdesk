// Real app actions, editor and updater with synthetic DOM/HTTP; no native install.
const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');
function app() {
  const f=frontend(root);f.seed();const calls=[],handlers=[];
  f.context.AbortController=AbortController;
  f.context.loadPeople=async()=>{};
  f.document.addEventListener=(type,fn)=>{if(type==='keydown')handlers.push(fn);};
  f.Element.prototype.select=function(){this.setSelectionRange(0,this.value.length);};
  let status={id:0,state:'idle',dismissed_version:''};
  f.context.fetch=async(url,options={})=>{
    const body=options.body?JSON.parse(options.body):null;calls.push({url,body});
    if(url.endsWith('/transcript'))return {ok:true,json:async()=>({...f.context.fixture,revision:2,document:body.document})};
    if(url==='/api/preferences/update-notice')status={...status,dismissed_version:body.version};
    if(body?.op==='check')status={id:status.id+1,state:'checking',dismissed_version:status.dismissed_version};
    return {ok:true,json:async()=>structuredClone(status)};
  };
  f.run("$('setup').hidden=true;$('empty').hidden=true;wire();renderSaveState();");
  f.document.querySelector('main').append(f.document.getElementById('workspace'));
  f.document.getElementById('workspace').append(f.document.getElementById('search'),f.document.getElementById('save'));
  for(const id of ['new-meeting','files','meeting-search','setup-toggle'])f.document.querySelector('.sidebar').append(f.document.getElementById(id));
  const key=(key,overrides={})=>{
    const event={type:'keydown',key,metaKey:true,ctrlKey:false,altKey:false,shiftKey:false,repeat:false,isComposing:false,defaultPrevented:false,preventDefault(){this.defaultPrevented=true;},...overrides};
    handlers.forEach(fn=>fn(event));return event;
  };
  return {...f,calls,key,status:next=>{status=next;}};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('startup checks once through existing transport; available notice never downloads or installs automatically',async()=>{
  const f=app();f.context.speakerdeskNativeUpdater=true;
  await f.run('initializeUpdates()');await f.run('initializeUpdates()');
  assert.deepEqual(f.calls.filter(c=>c.body).map(c=>c.body.op),['check']);
  f.status({id:1,state:'available',version:'0.7.0',dismissed_version:''});await f.run('refreshUpdates()');
  assert.equal(f.document.getElementById('update-notice').hidden,false);
  assert.equal(f.document.getElementById('update-notice-text').textContent,'Speakerdesk 0.7.0 is available.');
  assert.equal(f.document.getElementById('setup').hidden,true);
  await f.run("dispatchAction('download-update')");
  assert.equal(f.document.getElementById('setup').hidden,false);
  assert.deepEqual(f.calls.filter(c=>c.body).map(c=>c.body.op),['check','download']);
});

test('current/error/unavailable are quiet and manual check exposes the existing result',async()=>{
  for(const state of ['current','error','unavailable']) {
    const f=app();f.context.speakerdeskNativeUpdater=true;
    f.status({id:1,state,error:state==='error'?'Fixture offline':''});await f.run('initializeUpdates()');
    assert.equal(f.document.getElementById('update-notice').hidden,true);
    assert.equal(f.document.getElementById('setup').hidden,true);
    assert.equal(f.calls.some(c=>c.body),false);
    if(state!=='unavailable') {
      await f.run("dispatchAction('check-updates')");assert.equal(f.calls.at(-2).body.op,'check');
      assert.equal(f.document.getElementById('setup').hidden,false);
    }
  }
});

test('dismissed version remains suppressed across polls; newer version reappears; Undo stays intact',async()=>{
  const f=app();f.run('removePassage(doc.segments[0]);');
  const undo=f.document.querySelector('#notice .notice-action');
  f.status({id:1,state:'available',version:'0.7.0',dismissed_version:''});await f.run('refreshUpdates()');
  await f.run('dismissUpdateNotice()');await f.run('refreshUpdates()');
  assert.equal(f.document.getElementById('update-notice').hidden,true);
  assert.equal(f.document.querySelector('#notice .notice-action'),undo);
  f.status({id:1,state:'available',version:'0.7.0',dismissed_version:''});await f.run('refreshUpdates()');
  assert.equal(f.document.getElementById('update-notice').hidden,true,'A stale pre-dismissal poll cannot nag again.');
  f.status({id:2,state:'available',version:'0.8.0',dismissed_version:'0.7.0'});await f.run('refreshUpdates()');
  assert.equal(f.document.getElementById('update-notice').hidden,false);
  assert.equal(f.calls.some(c=>c.body?.op==='install'),false);
});

test('recording disables update action while still announcing availability',async()=>{
  for(const status of ['starting','recording','paused','finishing']) {
    const f=app();f.context.capture=status;f.run('meeting={status:capture};');
    f.status({id:1,state:'available',version:'0.7.0'});await f.run('refreshUpdates()');
    assert.equal(f.document.getElementById('update-notice').hidden,false);
    assert.equal(f.document.getElementById('update-notice-action').disabled,true);
    assert.equal(await f.run("dispatchAction('download-update')"),false);
    assert.equal(f.calls.some(c=>c.body),false);
  }
});

test('Command comma and N dispatch actual settings/new meeting actions without browser defaults',async()=>{
  const f=app();assert.equal(f.key(',').defaultPrevented,true);await tick();
  assert.equal(f.document.getElementById('setup').hidden,false);
  assert.equal(f.document.activeElement.id,'settings-close');
  f.run('closeSettings();');assert.equal(f.key('n').defaultPrevented,true);await tick();
  assert.equal(f.document.getElementById('workspace').hidden,true);
  assert.equal(f.document.activeElement.id,'meeting-title');
  assert.equal(f.run('selected'),null);
});

test('New Meeting preserves recordings, disabled state, passage drafts and failed saves',async()=>{
  for(const state of ['starting','recording','paused','finishing']) {
    const f=app();const before=f.run('JSON.stringify(doc)');f.context.capture=state;f.run('meeting={status:capture};');
    assert.equal(await f.run("dispatchAction('new-meeting')"),false);
    assert.equal(f.run('JSON.stringify(doc)'),before);assert.equal(f.calls.length,0);
  }
  for(const seed of ["$('new-meeting').disabled=true;","passageSaveOperations.set(selected.id+':r0',{jid:selected.id});","undoRemoval={jid:selected.id};"]) {
    const f=app();f.run(seed);assert.equal(await f.run("dispatchAction('new-meeting')"),false);assert.ok(f.run('selected'));
  }
  for(const seed of ["passageDrafts.set('r0',{text:'My draft'});","recoverablePassageDrafts.set('r0',{text:'My retained draft'});"]) {
    const f=app();f.context.confirm=()=>false;f.run(seed);
    assert.equal(await f.run("dispatchAction('new-meeting')"),false);assert.ok(f.run('selected'));
    assert.equal(f.run('passageDrafts.size+recoverablePassageDrafts.size'),1);
  }
  const f=app();f.context.confirm=()=>false;f.run("doc.segments[0].text='My words';dirty=true;");
  f.context.fetch=async()=>({ok:false,json:async()=>({error:'Fixture disk refused'})});
  assert.equal(await f.run("dispatchAction('new-meeting')"),false);
  assert.equal(f.run('dirty'),true);assert.equal(f.run('doc.segments[0].text'),'My words');
});

test('New Meeting saves exact edits and rechecks capture state after asynchronous save',async()=>{
  const f=app();f.run("doc.segments[0].text='Exact edits';dirty=true;");
  assert.equal(await f.run("dispatchAction('new-meeting')"),true);
  assert.equal(f.calls[0].body.document.segments[0].text,'Exact edits');
  const late=app(),transport=late.context.fetch;
  late.run("doc.segments[0].text='Saved while capture starts';dirty=true;");
  late.context.fetch=async(...args)=>{const response=await transport(...args);late.run("meeting={status:'recording'};");return response;};
  assert.equal(await late.run("dispatchAction('new-meeting')"),false);assert.ok(late.run('selected'));
});

test('Find is contextual, Save uses real autosave, and Import uses the existing picker',async()=>{
  const f=app();f.document.getElementById('search').value='Words';f.key('f');await tick();
  assert.equal(f.document.activeElement.id,'search');assert.equal(f.document.activeElement.selectionEnd,5);
  f.run("doc.segments[0].text='Save shortcut words';changed();");f.key('s');await tick();
  assert.equal(f.calls[0].body.document.segments[0].text,'Save shortcut words');
  f.run("$('workspace').hidden=true;");f.key('f');await tick();assert.equal(f.document.activeElement.id,'meeting-search');
  let picked=0;f.document.getElementById('files').click=()=>picked++;f.key('o');await tick();assert.equal(picked,1);
});

test('modal/inert/disabled and frozen states block menu and keyboard alike; editing keys remain native',async()=>{
  for(const setup of ["openSettings('updates');","updateFrozen=true;","document.querySelector('main').inert=true;","$('search').disabled=true;"]) {
    const f=app();f.run(setup);const before=f.document.activeElement;f.key('f');await tick();
    assert.equal(f.document.activeElement===before,true,setup);
  }
  const f=app();const dialog=f.document.createElement('dialog');dialog.setAttribute('open','');f.document.body.append(dialog);
  for(const action of ['settings','new-meeting','find','save','import-recording','check-updates'])assert.equal(await f.run(`dispatchAction('${action}')`),false);
  for(const key of ['a','c','v','x','z'])assert.equal(f.key(key).defaultPrevented,false);
  for(const override of [{shiftKey:true},{altKey:true},{isComposing:true},{metaKey:false}])assert.equal(f.key('n',override).defaultPrevented,false);
});

test('native Command events stay uncancelled and dispatch only through the menu; browser repeats are suppressed',async()=>{
  const f=app();f.context.speakerdeskNativeMenu=true;
  const focused=f.document.activeElement;
  for(const key of [',','n','o','f','s']) {
    assert.equal(f.key(key).defaultPrevented,false,`Native Command+${key} must reach the menu.`);
    await tick();
    assert.ok(f.run('selected'));assert.equal(f.document.getElementById('setup').hidden,true);
    assert.equal(f.document.activeElement,focused);assert.equal(f.calls.length,0);
  }
  assert.equal(await f.context.speakerdeskAction('new-meeting'),true);assert.equal(f.run('selected'),null);
  const repeat=app();assert.equal(repeat.key('n',{repeat:true}).defaultPrevented,true);await tick();assert.ok(repeat.run('selected'));
});
