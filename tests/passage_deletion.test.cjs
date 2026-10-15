const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

test('a poll started before deletion cannot recreate the deleted passage after its PATCH acknowledgement',async()=>{
  const f=frontend(root);f.seed(2);
  f.run(`window.before=structuredClone(selected);before.revision=2;flushSave=async()=>{};
    api=async(path,options={})=>{
      if(options.method==='PATCH'){const segment=structuredClone(doc.segments[0]);segment.deleted=true;segment.machine_revision++;return {segment,revision:3};}
      if(path==='/api/jobs')return [];
      return new Promise(resolve=>window.releaseOldPoll=()=>resolve(before));
    };window.pendingPoll=poll();`);
  await new Promise(resolve=>setImmediate(resolve));
  await f.run('removePassage(doc.segments[0])');
  f.run('releaseOldPoll()');await f.run('pendingPoll');
  assert.equal(f.run('doc.segments[0].deleted'),true);
  assert.equal(f.document.getElementById('segments').children.length,1);
});

test('passage deletion is available during processing, uses passage CAS and Undo restores its words',async()=>{
  for(const status of ['ready','recording','processing']) {
    const f=frontend(root);f.seed(2,status);
    f.run(`selected.import_resumable=true;selected.rolling_refinement={version:1};selected.refinement_status='refining';
      window.requests=[];flushSave=async()=>{};poll=async()=>{};
      api=async(path,options)=>{const body=JSON.parse(options.body);requests.push({path,body});
        const segment=structuredClone(doc.segments[0]);segment.deleted=body.changes.deleted;segment.machine_revision++;
        segment.protected_fields=['text'];segment.refinement_state='edited';return {segment,revision:2};};renderSegments();`);
    const button=f.document.getElementById('segments').children[0].querySelector('.remove-segment');
    assert.ok(button,status);assert.equal(button.hidden,false);
    await f.run('removePassage(doc.segments[0])');
    assert.equal(f.document.getElementById('segments').children.length,1);
    assert.equal(f.run('doc.segments[0].text'),'Words 0');
    assert.equal(f.run('requests[0].body.segment_revision'),1);
    assert.equal(f.run('requests[0].body.changes.deleted'),true);
    const undo=f.document.querySelector('#notice .notice-action');assert.ok(undo);
    undo.click();await new Promise(resolve=>setImmediate(resolve));
    assert.equal(f.run('requests[1].body.segment_revision'),2);
    assert.equal(f.run('requests[1].body.changes.deleted'),false);
    assert.equal(f.document.getElementById('segments').children.length,2);
  }
});
