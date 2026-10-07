// Uses the existing CPU DOM and the actual shipped renderer. No GUI or models.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

if(process.argv[2]==='--render') {
  const f=frontend(root);f.seed(0);
  f.context.job=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
  f.run("selected=structuredClone(job);doc=structuredClone(job.document);renderSegments()");
  const host=f.document.getElementById('segments');
  const turns=host.querySelectorAll('.reading-turn');
  const words=turns.map(t=>t.querySelector('.reading-turn-words').textContent).join('');
  if(process.argv[4]==='unknown') {
    assert.equal(host.querySelector('textarea').value,'Alpha. Still alpha. Beta. Still beta.');
    assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),false,'disjoint activity is not overlap when timing is missing');
    assert.doesNotMatch(host.querySelectorAll('.review-tag').map(t=>t.textContent).join(''),/Overlapping speakers/);
    process.exit(0);
  }
  assert.equal(words,'Alpha. Still alpha. Beta. Still beta.','server-persisted words must reach readable turns');
  const labels=turns.map(t=>t.querySelector('.reading-turn-speaker').textContent);
  if(process.argv[4]==='sequential') {
    assert.deepEqual(labels,['A','B'],'one continuous readable turn per speaker, including both sentences');
    assert(turns.every(t=>t.dataset.attribution==='single'),'disjoint activity must never mean overlap');
  } else {
    assert.equal(labels[0],'A');assert.equal(labels.at(-1),'B');
    assert(labels.includes('Overlapping speakers'),'concurrent activity must have a local overlap label');
  }
} else {
  const test=require('node:test');
  test('Use latest retains a replaced human correction after Stop',()=>{
    const f=frontend(root);f.seed(2,'recording');f.run('renderSegments()');
    const host=f.document.getElementById('segments');
    const card=host.querySelector('[data-segment-id="r0"]'),text=card.querySelector('textarea');
    text.focus();text.value='Recoverable human correction';text.dispatchEvent({type:'input'});
    card.querySelector('.use-latest').click();
    f.run("doc.segments.splice(0,1);selected.status='ready';selected.refinement_status='complete';renderSegments()");
    const retained=host.querySelector('[data-segment-id="r0"]');
    assert(retained,'the replaced correction must remain accessible after Stop');
    assert.equal(retained.classList.contains('live-provisional'),false);
    retained.querySelector('.recover-correction').click();
    const copies=retained.querySelectorAll('textarea').map(t=>t.value);
    assert(copies.includes('Recoverable human correction'),'the retained card must expose the exact correction');
  });
  for(const status of ['recording','paused','ready'])test(`refined and pending words form distinct sections while ${status}`,()=>{
    const f=frontend(root);f.seed(4,status);
    f.run("doc.segments.forEach((s,i)=>s.refinement_state=i%2?'refined':'provisional');window.before=JSON.stringify(doc);renderSegments()");
    const host=f.document.getElementById('segments');
    let live=false;
    for(const card of host.children) {
      if(card.classList.contains('live-provisional'))live=true;
      else assert.equal(live,false,'a refined passage appeared inside the live section');
    }
    assert.deepEqual(host.children.map(c=>c.dataset.segmentId),['r1','r3','r0','r2']);
    assert.equal(host.querySelectorAll('.live-section-label').filter(e=>!e.hidden).length,1);
    assert.equal(f.run('JSON.stringify(doc)===before'),true,'presentation must not mutate stored order');
    f.run("doc.segments[0].text='Refined replacement';doc.segments[0].refinement_state='refined';doc.segments[0].machine_revision++;renderSegments()");
    assert.equal(host.children.filter(c=>c.dataset.segmentId==='r0').length,1,'refinement must replace the existing passage');
    assert.deepEqual(host.children.filter(c=>c.classList.contains('live-provisional')).map(c=>c.dataset.segmentId),['r2']);
    assert.equal(host.querySelectorAll('textarea').filter(e=>e.value==='Words 0').length,0,'stale live words must disappear');
  });
}
