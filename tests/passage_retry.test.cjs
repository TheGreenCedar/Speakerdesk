// Shipped renderer contract; router/model behavior is exercised separately.
const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
test('pending retry prevents duplicate retries while the saved transcript stays editable',()=>{
  const f=frontend(path.resolve(__dirname,'..'));f.seed(1);
  f.run("selected.passage_retry={id:'pending',segment_id:'r0'};setStatus();renderSegments()");
  const retry=()=>f.document.getElementById('segments').querySelectorAll('button').find(b=>b.textContent==='Retry passage locally');
  assert.equal(retry().disabled,true);assert.equal(f.document.getElementById('editor').inert,false);
  assert.equal(f.document.getElementById('segments').querySelectorAll('button').filter(b=>b.textContent==='Cancel passage retry').length,1);
  f.run('delete selected.passage_retry;renderSegments()');
  assert.equal(retry().disabled,false);
});
for(const update of ['retry candidate','voice metadata'])test(`${update} can advance revision before dirty words autosave without stranding or overwriting them`,async()=>{
  for(const conflict of [null,'text','speaker','boundary','name','status','duration','second_save']) {
    const f=frontend(path.resolve(__dirname,'..'));f.seed(1);
    f.context.conflict=conflict;f.context.update=update;
    f.run(`selected.kind='import';selected.status='ready';
      if(update==='retry candidate')selected.passage_retry={id:'pending',segment_id:'r0'};
      window.latest=structuredClone(selected);latest.revision++;delete latest.passage_retry;
      if(update==='retry candidate')latest.document.segments[0].retry_candidate={id:'pending',start:0,end:10,speaker:'s0',text:'Model candidate',language:'en'};
      else latest.voice_checks={s0:{status:'unknown'}};
      if(conflict==='text')latest.document.segments[0].text='New authoritative words';
      if(conflict==='speaker')latest.document.segments[0].speaker='s1';
      if(conflict==='boundary')latest.document.segments[0].end=9;
      if(conflict==='name')latest.document.speakers.s0='Changed name';
      if(conflict==='status')latest.status='processing';
      if(conflict==='duration')latest.duration++;
      doc.segments[0].text='My exact unsaved words';changed();renderSegments();
      document.getElementById('segments').querySelector('textarea').focus();
      window.puts=[];fetch=async(path,options)=>{
        if(options.method!=='PUT')return {ok:true,status:200,json:async()=>structuredClone(latest)};
        const body=JSON.parse(options.body);puts.push(body);
        if(body.revision!==latest.revision || conflict==='second_save')return {ok:false,status:409,json:async()=>({error:'This transcript changed.'})};
        latest.document=body.document;
        if(update==='retry candidate')latest.document.segments[0].retry_candidate={id:'pending',start:0,end:10,speaker:'s0',text:'Model candidate',language:'en'};
        latest.revision++;return {ok:true,status:200,json:async()=>structuredClone(latest)};
      };`);
    if(conflict) {
      await assert.rejects(f.run('save({quiet:true})'),/changed/i);
      assert.equal(f.run('dirty'),true);
      assert.equal(f.run('puts.length'),conflict==='second_save'?2:1);
    } else {
      await f.run('save({quiet:true})');assert.equal(f.run('dirty'),false);
      assert.equal(f.run('latest.document.segments[0].text'),'My exact unsaved words');
      assert.deepEqual(Array.from(f.run('puts.map(p=>p.revision)')),[1,2]);
      if(update==='retry candidate')assert.equal(f.run('doc.segments[0].retry_candidate.text'),'Model candidate');
      else assert.equal(f.run('selected.voice_checks.s0.status'),'unknown');
    }
    assert.equal(f.run('doc.segments[0].text'),'My exact unsaved words');
  }
});
test('cancel uses the existing job route, retains words and freezes only the request',async()=>{
  const f=frontend(path.resolve(__dirname,'..'));f.seed(1);
  f.run("selected.passage_retry={id:'pending',segment_id:'r0'};flushSave=async()=>{};window.request=null;api=async(path,options)=>{window.request={path,options};if(!$('editor').inert)throw Error('request must be frozen');const next=structuredClone(selected);delete next.passage_retry;return next;};setStatus();renderSegments()");
  const cancel=f.document.getElementById('segments').querySelectorAll('button').find(b=>b.textContent==='Cancel passage retry');
  cancel.click();await Promise.all(cancel.lastEventResults);
  assert.equal(f.run('request.path'),'/api/jobs/'+'d'.repeat(32)+'/cancel');
  assert.equal(f.run('request.options.method'),'POST');assert.equal(f.run('doc.segments[0].text'),'Words 0');
  assert.equal(f.document.getElementById('editor').inert,false);assert.equal(f.run('!!selected.passage_retry'),false);
});
