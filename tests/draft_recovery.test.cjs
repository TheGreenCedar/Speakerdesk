const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');
function editor(){const f=frontend(root);f.seed();f.Element.prototype.select=function(){this.setSelectionRange(0,this.value.length);};vm.runInContext(fs.readFileSync(path.join(root,'speakerdesk/static/recovery.js'),'utf8'),f.context);return f;}
test('native snapshot copies every unsaved class without changing canonical authority',()=>{
 const f=editor();f.run("doc.segments[0].text='Unsaved whole transcript';dirty=true;passageDrafts.set('r1',{text:'Passage correction',revision:1,segment:doc.segments[1]});recoverablePassageDrafts.set('old',{text:'Recoverable correction',revision:1,segment:doc.segments[0]});timingDrafts.set(selected.id+':r0:start','');passageSaveOperations.set(selected.id+':r2',{jid:selected.id,segment:doc.segments[2],changes:{speaker:'s1'}})");
 const before=f.run('JSON.stringify(selected.document)'),snapshot=f.context.speakerdeskRecoverySnapshot();
 assert.equal(snapshot.has_unsaved,true);assert.equal(snapshot.document.segments[0].text,'Unsaved whole transcript');assert.equal(snapshot.passage_drafts[0][1].text,'Passage correction');assert.equal(snapshot.recoverable_drafts[0][1].text,'Recoverable correction');assert.equal(snapshot.timing_drafts[0][1],'');
 snapshot.document.segments[0].text='Changed copy';assert.equal(f.run('doc.segments[0].text'),'Unsaved whole transcript');assert.equal(f.run('JSON.stringify(selected.document)'),before);
 assert.match(f.run('recoveryText(window.speakerdeskRecoverySnapshot())'),/Pending passage changes/);
});
test('exact close freezes edits; failed persistence reopens the same drafts for copying',async()=>{
 const f=editor();f.run("$('setup').hidden=true;doc.segments[0].text='Words from the final keystroke';dirty=true");
 const snapshot=f.context.speakerdeskRecoverySnapshot(true);assert.equal(snapshot.document.segments[0].text,'Words from the final keystroke');assert.equal(f.document.querySelector('main').inert,true);
 await assert.rejects(f.run("api('/api/jobs',{method:'POST'})"),/recovery copy/);
 f.context.speakerdeskRecoveryFailed('Disk unavailable',true);assert.equal(f.document.querySelector('main').inert,false);assert.equal(f.run('recoveryClosing'),false);assert.equal(f.run('dirty'),true);
 f.document.getElementById('recovery-copy-open').click();assert.match(f.document.getElementById('recovery-text').value,/Words from the final keystroke/);assert.equal(f.document.activeElement.id,'recovery-text');
});
test('reviewing and discarding a persisted copy never imports its words',async()=>{
 const f=editor(),id='a'.repeat(32)+'-'+'b'.repeat(32);f.context.recoveryId=id;
 f.run("window.requests=[];api=async(path,options)=>{requests.push({path,method:options?.method || 'GET'});if(path==='/api/recovery')return {receipts:[{id:recoveryId,name:'Old meeting'}]};if(options?.method==='DELETE')return {discarded:true};return {snapshot:{name:'Old meeting',job_id:'b'.repeat(32),revision:3,document:{segments:[{start:0,speaker:'Speaker 1',text:'Old unsaved correction'}]},passage_drafts:[],recoverable_drafts:[]}};}");
 const before=f.run('JSON.stringify(doc)');await f.run('loadRecoveryCopies()');
 const row=f.document.getElementById('recovery-list').firstChild;row.querySelector('button').click();await Promise.all(row.querySelector('button').lastEventResults);
 assert.match(f.document.getElementById('recovery-text').value,/Old unsaved correction/);assert.equal(f.run('JSON.stringify(doc)'),before);
 const discard=row.querySelectorAll('button')[1];discard.click();await Promise.all(discard.lastEventResults);assert.equal(f.run('requests.some(request=>request.method==="PUT" || request.method==="PATCH")'),false);assert.equal(f.run('JSON.stringify(doc)'),before);
});
test('late close failure cannot unfreeze a replacement page that owns no close attempt',()=>{
 const f=editor();f.run("recoveryClosing=false;updateFrozen=true;document.querySelector('main').inert=true;$('recovery-copies').hidden=true;doc.segments[0].text='New page correction';dirty=true");
 f.context.speakerdeskRecoveryFailed('Earlier page timed out',true);
 assert.equal(f.document.querySelector('main').inert,true);assert.equal(f.document.getElementById('recovery-copies').hidden,true);assert.equal(f.run('doc.segments[0].text'),'New page correction');
});
test('progressive processing allows passage CAS, keeps new rows, and resumes only checkpointed work',async()=>{
 const f=editor();f.run("selected.kind='import';selected.status='processing';selected.import_resumable=true;selected.import_progress={completed:1,total:3};selected.inference_owned=true;setStatus();renderSegments();window.requests=[];window.server=structuredClone(selected);api=async(path,options)=>{if(options?.method==='PATCH'){const body=JSON.parse(options.body);requests.push(body);server.document.segments[0]={...server.document.segments[0],...body.changes,machine_revision:2};server.revision++;return {segment:server.document.segments[0],revision:server.revision};}return path==='/api/jobs'?[server]:structuredClone(server);}");
 assert.equal(f.document.getElementById('editor').inert,false);assert.equal(f.document.getElementById('add').hidden,true);assert.equal(f.document.getElementById('import').disabled,true);
 const row=f.document.getElementById('segments').firstChild,text=row.querySelector('textarea');text.value='Protected partial correction';text.dispatchEvent({type:'input'});const save=row.querySelector('.save-passage');save.click();await Promise.all(save.lastEventResults);
 assert.equal(f.run('requests[0].segment_revision'),1);assert.equal(f.run('doc.segments[0].text'),'Protected partial correction');
 f.run("server.status='uploaded';server.inference_owned=false;server.revision++");await f.run('poll()');
 assert.equal(f.document.getElementById('run').hidden,false);assert.equal(f.document.getElementById('run').disabled,false);assert.equal(f.document.getElementById('run').textContent,'Resume transcription');
 f.run("server.import_resumable=false;server.status='ready';server.revision++");await f.run('poll()');assert.equal(f.document.getElementById('run').hidden,true);
});
