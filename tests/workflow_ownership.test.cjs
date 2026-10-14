const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');

test('upload completion saves edits made while the upload was pending',async()=>{
  const f=frontend(root);f.seed();
  f.context.FormData=class{append(){}};
  f.run("window.release=null;window.saved=[];api=(path,options)=>{if(options?.method==='POST')return new Promise(resolve=>release=resolve);if(options?.method==='PUT'){const body=JSON.parse(options.body);saved.push(body.document.segments[0].text);return Promise.resolve({...selected,revision:2,document:body.document});}return Promise.resolve(path==='/api/jobs'?[fixture]:structuredClone(fixture));}");
  const uploading=f.run('upload([{}])');await Promise.resolve();await Promise.resolve();
  f.run("doc.segments[0].text='Words typed during upload';changed();release([{id:fixture.id}])");
  await uploading;
  assert.deepEqual(Array.from(f.run('saved')),['Words typed during upload']);
});

test('delete completion preserves a different recording and its new draft',async()=>{
  const f=frontend(root);f.seed();f.run("wire();window.release=null;api=(path,options)=>options?.method==='DELETE'?new Promise(resolve=>release=resolve):Promise.resolve([])");
  const button=f.document.getElementById('delete');button.click();const pending=button.lastEventResults;
  f.run("selected={...selected,id:'b'.repeat(32)};selectionGeneration++;doc.segments[0].text='Keep my newer recording draft';changed();release({})");
  await Promise.all(pending);
  assert.equal(f.run('selected.id'),'b'.repeat(32));
  assert.equal(f.run('doc.segments[0].text'),'Keep my newer recording draft');assert.equal(f.run('dirty'),true);
});

test('failed selection retains the visible transcript and recoverable corrections',async()=>{
  const f=frontend(root);f.seed();f.run("renderSegments();recoverablePassageDrafts.set('old',{text:'Retained correction'});api=async()=>{throw Error('Temporary read failure')}");
  const card=f.document.getElementById('segments').firstChild;
  await assert.rejects(f.run("select('b'.repeat(32))"),/Temporary read failure/);
  assert.equal(f.document.getElementById('segments').firstChild,card);
  assert.equal(f.run("recoverablePassageDrafts.get('old').text"),'Retained correction');
});

test('declining loss of a recoverable correction prevents selection',async()=>{
  const f=frontend(root);f.seed();f.run("recoverablePassageDrafts.set('old',{text:'Retained correction'});window.reads=0;confirm=()=>false;api=async()=>{reads++;return fixture}");
  await f.run("select('b'.repeat(32))");
  assert.equal(f.run('reads'),0);assert.equal(f.run('recoverablePassageDrafts.size'),1);
});

test('upload completion does not navigate over a newer selection',async()=>{
  const f=frontend(root);f.seed();f.context.FormData=class{append(){}};
  f.run("window.release=null;api=(path,options)=>options?.method==='POST'?new Promise(resolve=>release=resolve):Promise.resolve([])");
  const uploading=f.run('upload([{}])');await Promise.resolve();await Promise.resolve();
  f.run("selected={...selected,id:'b'.repeat(32)};selectionGeneration++;doc.segments[0].text='Words in the new selection';changed();release([{id:'a'.repeat(32)}])");
  await uploading;assert.equal(f.run('selected.id'),'b'.repeat(32));assert.equal(f.run('dirty'),true);
});

test('invalid timing stays out of the document and blocks navigation until reset',async()=>{
  const f=frontend(root);f.seed();f.run('renderSegments();showInspector(doc.segments[0],$("segments").firstChild,true)');
  const input=f.document.getElementById('segment-details').querySelector('input');input.value='';input.dispatchEvent({type:'change'});
  assert.equal(f.run('doc.segments[0].start'),0);assert.equal(f.run('dirty'),false);
  assert.equal(input.getAttribute('aria-invalid'),'true');assert.equal(await f.run('saveBeforeLeaving()'),false);
  f.document.getElementById('segment-details').querySelectorAll('button').find(button=>button.textContent==='Reset').click();
  assert.equal(f.run('timingDrafts.size'),0);assert.equal(await f.run('saveBeforeLeaving()'),true);
  input.value='11';input.dispatchEvent({type:'change'});assert.equal(f.run('doc.segments[0].start'),0);
  input.value='1';input.dispatchEvent({type:'change'});assert.equal(f.run('doc.segments[0].start'),1);assert.equal(f.run('dirty'),true);
});

test('saved accuracy retains playback and revision-protected timing and speaker edits',async()=>{
  const f=frontend(root);f.seed();f.run("selected.refinement_status='refining';renderSegments();window.requests=[];api=async(path,options)=>{requests.push(JSON.parse(options.body));return {segment:{...doc.segments[0],...JSON.parse(options.body).changes,machine_revision:2},revision:2}};poll=async()=>{}");
  const row=f.document.getElementById('segments').firstChild;assert.equal(row.querySelector('.seek').disabled,false);
  row.querySelector('.saved-passage-details').click();
  const input=f.document.getElementById('segment-details').querySelector('input');input.value='1';input.dispatchEvent({type:'change'});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.run('requests[0].segment_revision'),1);assert.equal(f.run('requests[0].changes.start'),1);
  assert.equal(f.run('doc.segments[0].start'),1);assert.equal(f.run('dirty'),false);
});

test('header reports live and recoverable corrections as unsaved',()=>{
  const f=frontend(root);f.seed(3,'recording');f.run('renderSegments()');
  const text=f.document.getElementById('segments').firstChild.querySelector('textarea');text.value='My correction';text.dispatchEvent({type:'input'});
  assert.equal(f.document.getElementById('save').textContent,'Unsaved corrections');assert.equal(f.document.getElementById('save').hidden,false);
  f.run("recoverablePassageDrafts.set('r0',passageDrafts.get('r0'));passageDrafts.clear();renderSaveState()");
  assert.equal(f.document.getElementById('save').textContent,'Unsaved corrections');
});

test('model polling retains focused controls and holds their in-flight disabled state',async()=>{
  const f=frontend(root);f.seed();f.context.loadPeople=async()=>{};
  f.run("window.state={models:[{id:'mossformer2-16k',name:'Separation',bytes:1,optional:true}],status:'ready',ready:true,supported:true,separation:{bytes:1,can_download:true},voice:{available:false,enabled:false,status:'ready'}};window.release=null;api=(path,options)=>options?.method==='POST'?new Promise(resolve=>release=resolve):Promise.resolve(structuredClone(state))");
  await f.run('refreshSetup()');const button=f.document.getElementById('model-list').querySelector('button');button.focus();button.click();
  await f.run('refreshSetup()');assert.equal(f.document.getElementById('model-list').querySelector('button'),button);assert.equal(f.document.activeElement,button);assert.equal(button.disabled,true);
  f.run('release({})');await Promise.all(button.lastEventResults);assert.equal(button.disabled,false);
});

test('cancelled imports show pending ownership until the worker releases',async()=>{
  const f=frontend(root);f.seed(0,'preparing');f.run("selected.kind='import';selected.inference_owned=true;wire();setStatus();window.next={...selected,status:'uploaded',inference_owned:true};api=async(path,options)=>options?.method==='POST'?{}:path==='/api/jobs'?[next]:structuredClone(next)");
  const cancel=f.document.getElementById('cancel-job');assert.equal(cancel.hidden,false);cancel.click();await Promise.all(cancel.lastEventResults);
  assert.equal(cancel.textContent,'Stopping processing…');assert.equal(cancel.disabled,true);assert.equal(cancel.hidden,false);
  f.run('next.inference_owned=false');await f.run('poll()');assert.equal(cancel.hidden,true);
});

test('native service loss preserves drafts and replaces stale capture status',async()=>{
  const f=frontend(root);f.seed(3,'recording');f.run("meeting={id:selected.id,status:'recording'};renderSegments();passageDrafts.set('r0',{text:'Local words',revision:1,segment:doc.segments[0]});window.speakerdeskHostUnavailable('The service stopped')");
  for(const id of ['microphone-level','system-level']){const meter=f.document.createElement('div');meter.className='audio-meter';meter.append(f.document.createElement('label'),f.document.getElementById(id));f.document.body.append(meter);}
  f.document.getElementById('pause-meeting').append(f.document.createElement('span'));
  assert.equal(f.document.getElementById('pause-meeting').disabled,true);assert.equal(f.document.getElementById('stop-meeting').disabled,true);
  assert.equal(f.document.getElementById('microphone-level').value,0);assert.match(f.document.getElementById('listening').textContent,/unavailable/);
  assert.equal(f.run("passageDrafts.get('r0').text"),'Local words');
  f.run("api=async()=>({id:selected.id,status:'paused',duration:30,levels:{},sources:['microphone']})");await f.run('refreshMeeting()');
  assert.equal(f.document.getElementById('connection-status').hidden,true);assert.equal(f.document.getElementById('pause-meeting').disabled,false);
  assert.equal(f.run("passageDrafts.get('r0').text"),'Local words');
});

test('initial config failure can retry without duplicate control wiring',async()=>{
  const f=frontend(root);f.seed();f.context.loadPeople=async()=>{};
  f.context.location={search:''};
  f.run("window.attempt=0;window.calls=[];refreshMeeting=async()=>{};refreshSetup=async()=>{};initializeUpdates=async()=>{};api=async path=>{calls.push(path);if(path==='/api/config'){if(!attempt++)throw Error('Startup unavailable');return {languages:{auto:'Automatic',en:'English'},theme:'dark',default_language:'en',readiness:{configured:true,automatic_language:true}};}return [];}");
  await f.run('init()');assert.equal(f.document.getElementById('connection-status').hidden,false);
  await f.run('init()');await f.run('init()');
  assert.equal(f.document.getElementById('new-meeting').events.get('click').length,1);
  assert.equal(f.document.getElementById('language').children.length,2);assert.equal(f.run('initialized'),true);
});
