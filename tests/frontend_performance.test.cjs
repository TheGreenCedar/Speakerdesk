const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

test('saved reconciliation retains editable nodes and binds handlers to the current document',()=>{
  const f=frontend(root);f.seed(360);f.run('renderSegments()');
  const first=f.document.getElementById('segments').children[0],text=first.querySelector('textarea');text.focus();text.setSelectionRange(3,3);
  f.run('doc=structuredClone(doc);renderSegments()');
  assert.ok(f.document.getElementById('segments').children[0]===first,'Unchanged saved card must be retained');
  assert.ok(f.document.activeElement===text);assert.equal(text.selectionStart,3);
  text.value='My retained edit';text.dispatchEvent({type:'input'});
  assert.equal(f.run('doc.segments[0].text'),'My retained edit');
  assert.ok(first.querySelector('select').getAttribute('aria-label'));
  f.run("$('search').value='Words';renderSegments()");
  assert.equal(f.document.getElementById('segments').children.length,359);
});

test('textarea fitting batches height resets before reads and final writes',()=>{
  const f=frontend(root);f.seed();f.run('renderSegments()');
  const sizing=f.trace.filter(x=>['auto','scroll','height'].includes(x.kind));
  const autos=sizing.map((x,i)=>x.kind==='auto'?i:-1).filter(i=>i>=0),reads=sizing.map((x,i)=>x.kind==='scroll'?i:-1).filter(i=>i>=0),writes=sizing.map((x,i)=>x.kind==='height'?i:-1).filter(i=>i>=0);
  assert.equal(autos.length,3);assert.equal(reads.length,3);assert.equal(writes.length,3);
  assert.ok(Math.max(...autos)<Math.min(...reads),'All resets must precede height reads');
  assert.ok(Math.max(...reads)<Math.min(...writes),'All reads must precede final heights');
  f.trace.length=0;f.run('renderSegments()');assert.equal(f.trace.filter(x=>x.kind==='height').length,0,'Unchanged cards must keep fitted heights');
});

test('saved changed row defers replacement until focus leaves without replacing neighboring rows',async()=>{
  const f=frontend(root);f.seed();f.run('renderSegments()');
  const host=f.document.getElementById('segments'),card=host.children[0],neighbor=host.children[1],text=card.querySelector('textarea');text.focus();text.setSelectionRange(2,2);
  f.run("doc=structuredClone(doc);doc.segments[0].text='New returned words';doc.segments[0].end=12;renderSegments()");
  assert.ok(host.children[0]===card);assert.equal(text.value,'Words 0');assert.equal(text.selectionStart,2);assert.ok(host.children[1]===neighbor);
  text.blur();card.dispatchEvent(new f.context.Event('focusout'));await Promise.resolve();
  assert.equal(host.children[0].querySelector('textarea').value,'New returned words');assert.ok(host.children[1]===neighbor);
});

test('unchanged live rows avoid rewriting names; provisional heading and focused draft survive arrival',()=>{
  const f=frontend(root);f.seed(12,'recording');f.run("doc.segments[0].refinement_state='provisional';doc.segments[1].refinement_state='provisional';renderSegments()");
  const host=f.document.getElementById('segments'),names=host.querySelectorAll('.name-speaker'),assignments=names.map(n=>n.textAssignments);
  f.run('renderSegments()');assert.deepEqual(names.map(n=>n.textAssignments),assignments);
  assert.equal(host.querySelectorAll('.live-section-start').length,1);assert.equal(host.querySelectorAll('.live-provisional').length,2);
  const text=host.children[0].querySelector('textarea');text.focus();text.value='Protected local words';text.setSelectionRange(5,5);text.dispatchEvent({type:'input'});
  f.run("doc.segments[0].text='New machine words';doc.segments[0].machine_revision=2;renderSegments()");
  assert.ok(host.children[0].querySelector('textarea')===text);assert.equal(text.value,'Protected local words');assert.equal(text.selectionStart,5);
  assert.equal(host.children[0].querySelector('.keep-correction').dataset.revision,'2');
  assert.equal(host.children[0].querySelector('.latest-machine-words').textContent,'New machine words');
});

test('playback keeps overlap and backward seek highlights without per-card linear searches',()=>{
  const f=frontend(root);f.seed(120);f.run('doc.segments[0].end=25;renderSegments();wire();window.findCalls=0;doc.segments.find=new Proxy(doc.segments.find,{apply(fn,thisArg,args){findCalls++;return Reflect.apply(fn,thisArg,args)}})');
  f.run("$('player').currentTime=15;$('player').dispatchEvent(new Event('timeupdate'))");
  const ids=()=>f.document.getElementById('segments').querySelectorAll('.playing').map(c=>c.dataset.segmentId);
  assert.deepEqual(ids(),['r0','r1']);
  f.run("$('player').currentTime=2;$('player').dispatchEvent(new Event('timeupdate'))");assert.deepEqual(ids(),['r0']);
  assert.ok(f.run('findCalls')<=1,'Time updates must not find each visible row in the full transcript');
  f.run("doc.segments[0].start=30;doc.segments[0].end=40;changed();$('player').dispatchEvent(new Event('timeupdate'))");assert.deepEqual(ids(),[]);
});

test('actual Flask conditional fixtures retain only document and adopt fresh metadata or corrections',async()=>{
  const f=frontend(root);f.seed();f.context.responses=require('./support/job_responses.json');
  f.run('selected=structuredClone(responses.legacy_detail.body);doc=structuredClone(selected.document);window.serverDocument=selected.document;selected.staleMetadata=true;window.paths=[];window.reply="matching_revision";renderJobs=()=>{};setStatus=()=>{};renderSegments=()=>{};refreshIdentitySuggestions=()=>{};updateNameControls=()=>{};api=async path=>{paths.push(path);return structuredClone(path==="/api/jobs"?responses.list.body:responses[reply].body);}');
  await f.run('poll()');
  assert.equal(f.run('selected.document===serverDocument'),true);assert.equal(f.run('selected.duration'),12.25);assert.equal(f.run('selected.language'),'fr');assert.equal(f.run('selected.inference_owned'),true);
  assert.equal(f.run('selected.language_revision'),3);assert.equal(f.run('selected.pause_flush.request_id'),'request-one');assert.equal(f.run('selected.pause_flush.state'),'pending');assert.equal(f.run('Object.hasOwn(selected,"staleMetadata")'),false);
  assert.ok(f.run('paths.at(-1)').endsWith('?known_revision=7'));
  f.run('reply="changed_revision"');await f.run('poll()');
  assert.equal(f.run('doc.segments[0].text'),'My correction.');assert.equal(f.run('doc.segments[0].machine_revision'),3);assert.equal(f.run('doc.segments[0].protected_fields[0]'),'text');
  f.run('window.correctedDocument=selected.document;reply="matching_after_edit"');await f.run('poll()');assert.equal(f.run('selected.document===correctedDocument'),true);assert.ok(f.run('paths.at(-1)').endsWith('?known_revision=8'));
});

test('late conditional response cannot cross a selection or overwrite an edit',async()=>{
  const f=frontend(root);f.seed();f.run('renderJobs=()=>{};setStatus=()=>{};window.hold=null;api=path=>path==="/api/jobs"?Promise.resolve([]):new Promise(resolve=>{hold=resolve});window.pending=poll()');
  await Promise.resolve();f.run('dirty=true;doc.segments[0].text="New local edit";hold({id:selected.id,revision:2,status:"ready",document:{segments:[]},unchanged:false})');await f.run('pending');
  assert.equal(f.run('doc.segments[0].text'),'New local edit');assert.equal(f.run('selected.revision'),1);
  f.run('dirty=false;window.pending=poll()');await Promise.resolve();f.run('selectionGeneration++;selected={id:"e".repeat(32),revision:7};hold({id:"d".repeat(32),revision:1,unchanged:true})');await f.run('pending');assert.equal(f.run('selected.revision'),7);
});

test('library summaries retain unchanged buttons and focus while status labels update',()=>{
  const f=frontend(root);f.seed();f.run('jobs=[{id:selected.id,name:"Meeting",created:1,status:"ready"}];renderJobs()');
  const button=f.document.getElementById('jobs').querySelector('.job');button.focus();f.run('jobs=structuredClone(jobs);renderJobs()');assert.ok(f.document.getElementById('jobs').querySelector('.job')===button,'Unchanged summary button must be retained');assert.ok(f.document.activeElement===button);
  f.run('jobs[0].status="recording";renderJobs()');assert.ok(f.document.getElementById('jobs').querySelector('.job')===button);assert.ok(button.querySelector('small').textContent.includes('Recording'));
});

test('active French and Auto changes adopt persisted defaults without rewriting earlier words',async()=>{
  const f=frontend(root);f.seed(3,'recording');f.run("config.default_language='en';$('language').value='en';meeting={id:selected.id,status:'recording',language:'en',language_revision:0,language_acknowledged_revision:0};renderStartState=()=>{};refreshMeeting=async()=>{};window.beforeWords=JSON.stringify(doc);api=async(path,options)=>{const b=JSON.parse(options.body);return {language:b.language,default_language:b.language,language_revision:b.language_revision+1,language_acknowledged_revision:0,language_history:[{start_sample:1600}]};};$('live-language').value='fr'");
  await f.run('changeLiveLanguage()');assert.equal(f.run('config.default_language'),'fr');assert.equal(f.run("$('language').value"),'fr');assert.equal(f.run('meeting.language'),'fr');assert.equal(f.run('JSON.stringify(doc)===beforeWords'),true);
  f.run("$('live-language').value='auto'");await f.run('changeLiveLanguage()');assert.equal(f.run('config.default_language'),'auto');assert.equal(f.run("$('language').value"),'auto');assert.equal(f.run('meeting.language_revision'),2);
});

test('failed active change restores both controls and leaves the prior default intact',async()=>{
  const f=frontend(root);f.seed(3,'recording');f.run("config.default_language='en';$('language').value='en';meeting={id:selected.id,status:'recording',language:'en',language_revision:4};renderStartState=()=>{};refreshMeeting=async()=>{};api=async()=>{throw Error('409 language conflict')};$('live-language').value='fr'");
  await assert.rejects(f.run('changeLiveLanguage()'),/409 language conflict/);
  assert.equal(f.run('config.default_language'),'en');assert.equal(f.run("$('language').value"),'en');assert.equal(f.run("$('live-language').value"),'en');assert.equal(f.run('meeting.language_revision'),4);
});

test('restart uses server default over old local storage and new-meeting request uses that mode',async()=>{
  const f=frontend(root);f.seed();f.context.localStorage.getItem=()=> 'en';
  f.run("selected=null;doc=null;refreshMeeting=async()=>{meeting={status:'idle'}};refreshSetup=async()=>{};window.newMeetingMode=null;api=async(path,options)=>{if(path==='/api/config')return {languages:{auto:'Automatic',en:'English',fr:'French'},default_language:'fr',readiness:{configured:true,automatic_language:true}};if(path==='/api/jobs')return [];if(path==='/api/meetings'){newMeetingMode=JSON.parse(options.body).language;return {id:'new'};}throw Error(path);};select=async()=>{}");
  await f.run('init()');assert.equal(f.run("$('language').value"),'fr');
  f.run("$('meeting-title').value='Meeting';$('capture-microphone').checked=true;$('meeting-form').dispatchEvent(new Event('submit'))");
  await Promise.all(f.document.getElementById('meeting-form').lastEventResults);assert.equal(f.run('newMeetingMode'),'fr');
});

test('Settings persists the same state; active Settings selection uses the live epoch route',async()=>{
  const f=frontend(root);f.seed();f.run("config.default_language='en';renderStartState=()=>{};refreshMeeting=async()=>{};window.requests=[];api=async(path,options)=>{requests.push(path);const b=JSON.parse(options.body);return path==='/api/preferences/language'?{default_language:b.language}:{language:b.language,default_language:b.language,language_revision:1,language_acknowledged_revision:0,language_history:[{start_sample:3200}]};};$('language').value='fr'");
  await f.run('changeDefaultLanguage()');assert.equal(f.run('config.default_language'),'fr');assert.equal(f.run('requests[0]'),'/api/preferences/language');
  f.run("meeting={id:selected.id,status:'paused',language:'fr',language_revision:0};$('language').value='auto'");await f.run('changeDefaultLanguage()');assert.equal(f.run('config.default_language'),'auto');assert.ok(f.run('requests[1]').endsWith('/language'));assert.equal(f.run('meeting.language'),'auto');
  f.run("meeting=null;api=async()=>{throw Error('persist failed')};$('language').value='fr'");await assert.rejects(f.run('changeDefaultLanguage()'),/persist failed/);assert.equal(f.run("$('language').value"),'auto');
});

test('a late config fetch cannot restore the old default after a successful selection',async()=>{
  const f=frontend(root);f.seed();f.run("config.default_language='en';renderStartState=()=>{};window.release=null;api=path=>path==='/api/config'?new Promise(resolve=>{release=resolve}):Promise.resolve({default_language:'fr'});window.pendingConfig=refreshConfig();$('language').value='fr'");
  await f.run('changeDefaultLanguage()');f.run("release({languages:{auto:'Automatic',en:'English',fr:'French'},default_language:'en',readiness:{configured:true,automatic_language:true}})");await f.run('pendingConfig');assert.equal(f.run('config.default_language'),'fr');
});

test('pending preference commit disables both language controls and blocks new meeting or import',async()=>{
  const f=frontend(root);f.seed();f.run("config.default_language='en';meeting=null;saveBeforeLeaving=async()=>true;window.requests=[];window.release=null;api=(path,options)=>{requests.push(path);return new Promise(resolve=>release=resolve)};wire();$('language').value='fr';window.pendingChange=changeDefaultLanguage()");
  assert.equal(f.run("$('language').disabled"),true);assert.equal(f.run("$('live-language').disabled"),true);assert.equal(f.run("$('start-meeting').disabled"),true);
  await assert.rejects(f.run('upload([{}])'),/Wait for the language change/);
  f.run("$('meeting-form').dispatchEvent(new Event('submit'))");await Promise.all(f.document.getElementById('meeting-form').lastEventResults);assert.equal(f.run('requests.length'),1);
  f.run("release({default_language:'fr'})");await f.run('pendingChange');assert.equal(f.run("$('language').disabled"),false);assert.equal(f.run('config.default_language'),'fr');
});
