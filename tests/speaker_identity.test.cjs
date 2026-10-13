const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

function fixture(status='ready') {
  const f=frontend(root);f.seed(4,status);
  f.run(`doc.speakers={speaker_0:'Ada',speaker_1:'Ada',speaker_2:'Ada',speaker_3:'Ada'};
    doc.segments.forEach((s,i)=>s.speaker='speaker_'+i);
    selected.speaker_assignments={
      speaker_0:{person_id:'person-a',name:'Ada',source:'manual',confirmed:true},
      speaker_1:{person_id:'person-a',name:'Ada',source:'automatic_voice',confirmed:false},
      speaker_2:{person_id:'person-b',name:'Ada',source:'manual',confirmed:true},
      speaker_3:{person_id:null,name:'Ada',source:'manual',confirmed:true}};
    selected.document=structuredClone(doc);window.original=JSON.stringify(doc);renderEditor()`);
  return f;
}
const cards=f=>f.document.getElementById('segments').children;
function wireNaming(f) {
  f.run(fs.readFileSync(path.join(root,'speakerdesk/static/people.js'),'utf8'));
  f.Element.prototype.showModal=function(){this.open=true;};
  f.Element.prototype.close=function(){this.open=false;this.dispatchEvent({type:'close'});};
  f.Element.prototype.select=function(){this.setSelectionRange(0,this.value.length);};
  f.context.requests=[];
  f.run(`api=async (url,options)=>{
    if(url==='/api/people')return {people:[{id:'person-a',name:'Ada'}],voice_available:false};
    if(url==='/api/jobs')return [];
    if(url.endsWith('/voice-clips'))return {revision:selected.revision,status:'unavailable',clips:[],excluded:[],recommended_ids:[],minimum_clips:2,message:'Voice model unavailable'};
    if(!options)return structuredClone(window.freshJob || selected);
    requests.push({url,body:JSON.parse(options.body)});return structuredClone(selected);
  };wirePeople();renderEditor()`);
}

for(const status of ['ready','recording'])test(`${status}: saved person IDs unify visible identity while acoustic tracks remain separate`,()=>{
  const f=fixture(status),rows=cards(f);
  assert.equal(f.document.getElementById('speaker-list').children.length,3,'One saved person, a different person with the same name, and a local name');
  assert.equal(rows[0].querySelector('.speaker-avatar').dataset.color,rows[1].querySelector('.speaker-avatar').dataset.color);
  assert.deepEqual(rows.map(c=>c.classList.contains('speaker-continuation')),[false,true,false,false]);
  assert.deepEqual(rows.map(c=>c.dataset.speaker),['speaker_0','speaker_1','speaker_2','speaker_3']);
  assert.equal(rows[0].querySelector('.recognized-speaker'),null);
  assert.equal(rows[1].querySelector('.recognized-speaker').textContent,'Recognized','Confirmation remains track-specific');
  if(status==='ready')for(const row of rows) {
    const options=row.querySelector('select').children;
    assert.equal(options.length,3);
    assert.equal(options.filter(o=>o.selected).length,1);
    assert.equal(options.find(o=>o.selected).value,row.dataset.speaker,'Rendering must not move the passage to another acoustic track');
  }
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
  // Reopening a persisted API result reconstructs the same identity without a cache.
  f.run('doc=structuredClone(selected.document);renderEditor()');
  assert.equal(f.document.getElementById('speaker-list').children.length,3);
  assert.equal(cards(f)[1].classList.contains('speaker-continuation'),true);
});

test('live snapshots and equal-name reassignment refresh identity without discarding a focused correction',async()=>{
  const f=fixture('recording'),row=cards(f)[1],text=row.querySelector('textarea');
  text.focus();text.value='My exact correction';text.dispatchEvent({type:'input'});text.setSelectionRange(2,8);
  f.run("selected.speaker_assignments.speaker_1={person_id:'person-b',name:'Ada',source:'manual',confirmed:true};renderSegments()");
  assert.equal(cards(f)[1],row);assert.equal(f.document.activeElement,text);
  assert.equal(text.value,'My exact correction');assert.equal(text.selectionStart,2);assert.equal(text.selectionEnd,8);
  assert.equal(row.classList.contains('speaker-continuation'),false);
  assert.equal(cards(f)[2].classList.contains('speaker-continuation'),true);
  assert.equal(row.querySelector('.speaker-avatar').dataset.color,cards(f)[2].querySelector('.speaker-avatar').dataset.color);
  assert.equal(row.querySelector('.recognized-speaker'),null);
  f.run(`window.next=structuredClone(selected);next.revision++;next.document.segments.push({...next.document.segments[0],id:'later',start:50,end:60});
    api=async url=>url==='/api/jobs'?[]:structuredClone(next)`);
  await f.run('poll()');
  assert.equal(f.document.getElementById('speaker-list').children.length,3,'Future snapshots keep one entry per saved person');
  assert.equal(cards(f)[1],row);assert.equal(text.value,'My exact correction');
  f.run("selected.speaker_assignments.speaker_1={person_id:null,name:'Ada',source:'manual',confirmed:true};renderEditor()");
  assert.equal(f.document.getElementById('speaker-list').children.length,4,'Unlinking a passage track restores a separate identity');
  assert.equal(cards(f)[2].classList.contains('speaker-continuation'),false);
});

test('reading turns combine the same saved person and split after an equal-name identity correction',()=>{
  const f=fixture();
  f.run(`doc.segments=doc.segments.slice(0,1);const s=doc.segments[0];s.text='First. Second.';
    s.reading_turns=[{text:'First. ',speaker:'speaker_0',attribution:'single',start_offset:0,end_offset:7,start:null,end:null},
      {text:'Second.',speaker:'speaker_1',attribution:'single',start_offset:7,end_offset:14,start:null,end:null}];
    window.original=JSON.stringify(doc);renderSegments()`);
  assert.equal(cards(f)[0].querySelectorAll('.reading-turn').length,1);
  assert.equal(cards(f)[0].querySelector('.reading-turn-words').textContent,'First. Second.');
  assert.equal(cards(f)[0].querySelector('.reading-turn .review-tag').textContent,'Includes recognized voice','A mixed group keeps the automatic match hint without claiming every track was recognized');
  f.run("selected.speaker_assignments.speaker_1.person_id='person-b';renderSegments()");
  assert.equal(cards(f)[0].querySelectorAll('.reading-turn').length,2);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

for(const status of ['ready','recording'])test(`${status}: the person strip requests a group change while passage details keep track scope`,async()=>{
  const f=fixture(status);wireNaming(f);
  f.document.getElementById('speaker-list').querySelector('button').click();
  await f.run('Promise.resolve()');await f.run('Promise.resolve()');
  assert.match(f.document.getElementById('name-title').textContent,/person/i);
  assert.match(f.document.getElementById('name-message').textContent,/all passages/i);
  f.document.getElementById('name-form').dispatchEvent({type:'submit',preventDefault(){}});
  await f.run('Promise.resolve()');await f.run('Promise.resolve()');
  assert.equal(f.context.requests[0].body.all_person_tracks,true);
  assert.equal(f.document.getElementById('name-dialog').open,false,'A successful group apply closes its old membership context');
  cards(f)[1].querySelector('.passage-details-toggle').click();
  const details=status==='ready'?f.document.getElementById('segment-details'):cards(f)[1].querySelector('.rolling-review');
  details.querySelector('.passage-name-control').click();
  await f.run('Promise.resolve()');await f.run('Promise.resolve()');
  assert.equal(f.document.getElementById('name-voice-section').hidden,false,'Track-specific voice clips stay reachable on a continuation');
  f.document.getElementById('name-form').dispatchEvent({type:'submit',preventDefault(){}});
  await f.run('Promise.resolve()');await f.run('Promise.resolve()');
  assert.equal(f.context.requests[1].url.endsWith('/speakers/speaker_1/identity'),true);
  assert.equal(f.context.requests[1].body.all_person_tracks,false);
});

test('live group naming refreshes an explicit revision without losing the typed name',async()=>{
  const f=fixture('recording');wireNaming(f);
  await f.run("openNamePicker('speaker_0',true)");
  f.document.getElementById('name-value').value='My meeting name';
  f.document.getElementById('name-value').dispatchEvent({type:'input'});
  f.run('window.freshJob=structuredClone(selected);freshJob.revision++');
  await f.run('poll()');
  assert.equal(f.document.getElementById('name-apply').disabled,true);
  assert.match(f.document.getElementById('name-message').textContent,/transcript changed/i);
  const refresh=f.document.getElementById('refresh-voice-clips');
  assert.equal(refresh.hidden,false);assert.equal(refresh.disabled,false);
  refresh.click();await Promise.all(refresh.lastEventResults);
  assert.equal(f.document.getElementById('name-apply').disabled,false);
  assert.equal(f.document.getElementById('name-value').value,'My meeting name');
  const form=f.document.getElementById('name-form');form.dispatchEvent({type:'submit',preventDefault(){}});
  await Promise.all(form.lastEventResults);await f.run('Promise.resolve()');
  assert.equal(f.context.requests[0].body.revision,f.context.freshJob.revision);
  assert.equal(f.context.requests[0].body.name,'My meeting name');
  assert.equal(f.context.requests[0].body.person_id,null);
});

test('group refresh refuses an equal-name correction to the source person',async()=>{
  const f=fixture('recording');wireNaming(f);
  await f.run("openNamePicker('speaker_0',true)");
  f.run(`window.freshJob=structuredClone(selected);freshJob.revision++;
    freshJob.speaker_assignments.speaker_0.person_id='person-b'`);
  f.run('selected.revision++;updateNameControls()');
  const refresh=f.document.getElementById('refresh-voice-clips');
  assert.equal(refresh.hidden,false);
  refresh.click();await Promise.all(refresh.lastEventResults);
  assert.equal(f.document.getElementById('name-apply').disabled,true,'An explicit refresh cannot retarget a person with the same name');
  assert.match(f.document.getElementById('name-message').textContent,/person.*changed|assignment.*changed/i);
  assert.equal(f.context.requests.length,0);
  assert.equal(refresh.textContent,'Choose person again');
  refresh.click();await Promise.all(refresh.lastEventResults);
  assert.equal(f.document.getElementById('name-dialog').open,false);
  assert.equal(f.context.requests.length,0);
});
