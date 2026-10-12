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
for(const update of ['retry candidate','separation candidate','voice metadata'])test(`${update} can advance revision before dirty words autosave without stranding or overwriting them`,async()=>{
  for(const conflict of [null,'text','speaker','boundary','name','status','duration','second_save']) {
    const f=frontend(path.resolve(__dirname,'..'));f.seed(1);
    f.context.conflict=conflict;f.context.update=update;
    f.run(`selected.kind='import';selected.status='ready';
      if(update==='retry candidate')selected.passage_retry={id:'pending',segment_id:'r0'};
      window.latest=structuredClone(selected);latest.revision++;delete latest.passage_retry;
      if(update==='retry candidate')latest.document.segments[0].retry_candidate={id:'pending',start:0,end:10,speaker:'s0',text:'Model candidate',language:'en'};
      else if(update==='separation candidate')latest.document.segments[0].separation_candidate={id:'pending',original:structuredClone(selected.document.segments[0]),voices:[{text:'Voice one',language:'en'},{text:'Voice two',language:'en'}]};
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
        else if(update==='separation candidate')latest.document.segments[0].separation_candidate={id:'pending',original:structuredClone(selected.document.segments[0]),voices:[{text:'Voice one',language:'en'},{text:'Voice two',language:'en'}]};
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
      else if(update==='separation candidate')assert.equal(f.run('doc.segments[0].separation_candidate.voices[0].text'),'Voice one');
      else assert.equal(f.run('selected.voice_checks.s0.status'),'unknown');
    }
    assert.equal(f.run('doc.segments[0].text'),'My exact unsaved words');
  }
});

function overlappingFixture(f) {
  f.seed(2);
  f.run(`setupState={separation:{installed:true}};doc.segments[0].end=2;doc.segments[0].audio_revision=0;
    doc.segments[0].speaker_activity={audio_revision:0,observed_end_sample:32000,regions:[{start_sample:0,end_sample:32000,speakers:['speaker_0','speaker_1']}]};
    selected.document=structuredClone(doc);`);
}
function button(f,label) { return f.document.getElementById('segments').querySelectorAll('button').find(button=>button.textContent===label); }
test('separation needs actual two-track concurrency and leaves ordinary passages uncluttered',()=>{
  const f=frontend(path.resolve(__dirname,'..'));overlappingFixture(f);f.run('renderSegments()');
  assert.equal(button(f,'Retry voice separation').disabled,false);
  assert.equal(button(f,'Retry voice separation').closest('details').open===true,false);
  assert.equal(f.document.getElementById('segments').children[1].querySelector('.passage-separation').textContent,'');
  f.run("doc.segments[0].speaker_activity.regions.push({start_sample:16000,end_sample:32000,speakers:['speaker_2']});renderSegments()");
  assert.equal(button(f,'Retry voice separation').disabled,true);
  assert.match(f.document.getElementById('segments').textContent,/exactly two voices/);
  f.run("doc.segments[0].speaker_activity.regions=[{start_sample:0,end_sample:16000,speakers:['speaker_0']},{start_sample:16000,end_sample:32000,speakers:['speaker_1']}];renderSegments()");
  assert.equal(button(f,'Retry voice separation'),undefined);
  f.run("doc.segments[0].speaker_activity.regions=[{start_sample:0,end_sample:32000,speakers:['speaker_0','speaker_1']}];doc.segments[0].end=25;renderSegments()");
  assert.equal(button(f,'Retry voice separation').disabled,true);
  f.run("doc.segments[0].end=2+1/256;doc.segments[0].speaker_activity.observed_end_sample=32062;doc.segments[0].speaker_activity.regions[0].end_sample=32062;renderSegments()");
  assert.equal(button(f,'Retry voice separation').disabled,false);
  f.run("doc.segments[0].speaker_activity.audio_revision=1;renderSegments()");
  assert.equal(button(f,'Retry voice separation').disabled,true);
});
test('separation begin saves first, retains editable words during work, and cancels through the existing route',async()=>{
  const f=frontend(path.resolve(__dirname,'..'));overlappingFixture(f);
  f.run(`window.requests=[];flushSave=async()=>requests.push('save');api=async(path,options)=>{
    if(!$('editor').inert)throw Error('request must be frozen');requests.push({path,options});
    const next=structuredClone(selected);next.revision++;
    if(path.endsWith('/cancel'))delete next.passage_separation;
    else next.passage_separation={id:'e'.repeat(32),segment_id:'r0',phase:'Separating voices…'};
    return next;};setStatus();renderSegments();`);
  const begin=button(f,'Retry voice separation');begin.click();await Promise.all(begin.lastEventResults);
  assert.equal(f.run('requests[0]'),'save');assert.equal(f.run('requests[1].path'),'/api/jobs/'+'d'.repeat(32)+'/separation');
  assert.equal(f.run('JSON.parse(requests[1].options.body).segment_id'),'r0');
  assert.equal(f.run('doc.segments[0].text'),'Words 0');assert.equal(f.document.getElementById('editor').inert,false);
  assert.equal(button(f,'Retry passage locally').disabled,true);
  const cancel=button(f,'Cancel separation');cancel.click();await Promise.all(cancel.lastEventResults);
  assert.equal(f.run('requests[2].path'),'/api/jobs/'+'d'.repeat(32)+'/cancel');assert.equal(f.run('doc.segments[0].text'),'Words 0');
});
test('Cancel separation bypasses saves and retains unsaved live-card drafts and pending passage saves',async()=>{
  for(const pendingSave of [false,true]) {
    const f=frontend(path.resolve(__dirname,'..'));overlappingFixture(f);f.context.pendingSave=pendingSave;
    f.run(`selected.passage_separation={id:'e'.repeat(32),segment_id:'r0',phase:'Separating voices…'};
      doc.segments[1].refinement_state='provisional';selected.document=structuredClone(doc);renderSegments();
      window.requests=[];flushSave=async()=>{throw Error('Cancel must not save text');};api=async(path,options)=>{
        requests.push({path,options});const next=structuredClone(selected);next.revision++;delete next.passage_separation;return next;};
      if(pendingSave)passageSaveOperations.set(selected.id+':r1',{jid:selected.id,segment:structuredClone(doc.segments[1])});`);
    const text=f.document.getElementById('segments').children[0].querySelector('textarea');
    text.focus();text.value='My exact unsaved draft';text.dispatchEvent({type:'input'});
    const cancel=button(f,'Cancel separation');cancel.click();await Promise.all(cancel.lastEventResults);
    assert.equal(f.run('requests.length'),1);assert.equal(f.run('requests[0].path'),'/api/jobs/'+'d'.repeat(32)+'/cancel');
    assert.equal(f.run('requests[0].options.method'),'POST');assert.equal(f.run('!!selected.passage_separation'),false);
    assert.equal(f.run("passageDrafts.get('r0').text"),'My exact unsaved draft');assert.equal(text.value,'My exact unsaved draft');
    assert.equal(f.run('hasPendingPassageSaves()'),pendingSave);assert.equal(f.document.getElementById('editor').inert,false);
  }
});
function candidateFixture(f) {
  overlappingFixture(f);
  f.run(`doc.segments[0].start=20;doc.segments[0].end=22;
    doc.segments[0].separation_candidate={id:'e'.repeat(32),original:structuredClone(doc.segments[0]),voices:[{text:'First voice',language:'en'},{text:'Second voice',language:'fr'}]};
    selected.document=structuredClone(doc);renderSegments();`);
}
test('retained voice result stays collapsed and preserves authoritative text and caret with correct audio clocks',()=>{
  const f=frontend(path.resolve(__dirname,'..'));overlappingFixture(f);f.run('renderSegments()');
  const textarea=f.document.getElementById('segments').children[0].querySelector('textarea');textarea.focus();textarea.setSelectionRange(2,5);
  f.run(`doc.segments[0].separation_candidate={id:'e'.repeat(32),original:structuredClone(doc.segments[0]),voices:[{text:'First voice',language:'en'},{text:'Second voice',language:'fr'}]};renderSegments()`);
  assert.equal(f.document.activeElement,textarea);assert.equal(textarea.value,'Words 0');assert.equal(textarea.selectionStart,2);assert.equal(textarea.selectionEnd,5);
  assert.match(f.document.getElementById('segments').textContent,/Separated voice estimates · check both/);
  const g=frontend(path.resolve(__dirname,'..'));candidateFixture(g);
  const recovery=g.document.getElementById('segments').children[0].querySelector('.separation-comparison');
  assert.equal(recovery.tagName,'DETAILS');assert.equal(recovery.open===true,false);
  const audio=g.document.getElementById('segments').children[0].querySelectorAll('audio');assert.equal(audio.length,3);
  audio.forEach(player=>player.dispatchEvent({type:'loadedmetadata'}));
  assert.deepEqual(audio.map(player=>player.currentTime),[20,0,0]);
  assert.match(audio[0].src,/passages\/r0\/audio$/);assert.match(audio[1].src,/separation\/.+\/0\/audio$/);assert.match(audio[2].src,/separation\/.+\/1\/audio$/);
  assert.equal(g.run('doc.segments[0].text'),'Words 0');
});
for(const action of ['Keep original','Use both voices'])test(`${action} invokes its explicit server action and adopts only the returned document`,async()=>{
  const f=frontend(path.resolve(__dirname,'..'));candidateFixture(f);
  f.run(`window.requests=[];flushSave=async()=>requests.push('save');api=async(path,options)=>{
    requests.push({path,options});const next=structuredClone(selected);delete next.document.segments[0].separation_candidate;return next;};`);
  const choose=button(f,action);assert.equal(choose.disabled===true,false);choose.click();await Promise.all(choose.lastEventResults);
  assert.equal(f.run('requests[0]'),'save');assert.match(f.run('requests[1].path'),new RegExp('/separation/.+/'+(action==='Keep original'?'discard':'accept')+'$'));
  assert.equal(f.run('requests[1].options.method'),'POST');assert.equal(f.run('doc.segments[0].text'),'Words 0');assert.equal(f.run('doc.segments.length'),2);
});
test('changed or unsupported candidate words cannot replace the original',()=>{
  const f=frontend(path.resolve(__dirname,'..'));candidateFixture(f);
  f.run("doc.segments[0].text='My correction';renderSegments()");assert.equal(button(f,'Use both voices').disabled,true);
  f.run("doc.segments[0].text='Words 0';doc.segments[0].separation_candidate.voices[1].language='unknown';renderSegments()");assert.equal(button(f,'Use both voices').disabled,true);
});
test('overlap audio without original words can expose separation and its completed comparison',()=>{
  const f=frontend(path.resolve(__dirname,'..'));overlappingFixture(f);
  f.run("doc.segments[0].text='';doc.segments[0].transcription_review={reason:'overlapping_speech'};renderSegments()");
  assert.notEqual(f.document.getElementById('retained-audio-review').querySelectorAll('button').find(button=>button.textContent==='Retry voice separation'),undefined);
  f.run("doc.segments[0].separation_candidate={id:'e'.repeat(32),original:structuredClone(doc.segments[0]),voices:[{text:'First voice',language:'en'},{text:'Second voice',language:'en'}]};renderSegments()");
  assert.equal(f.document.getElementById('segments').children[0].dataset.segmentId,'r0');assert.equal(button(f,'Use both voices').disabled,false);
});
test('automatically published voices are ordinary editable rows with hidden generated players and collapsed original history',async()=>{
  for(const edited of [false,true]) {
    const f=frontend(path.resolve(__dirname,'..'));candidateFixture(f);f.context.edited=edited;
    f.run(`const candidate=doc.segments[0].separation_candidate;window.historyFixture={original:candidate.original,accepted:candidate.voices.map((voice,channel)=>({...candidate.original,id:'e'+channel,text:voice.text,speaker:'unassigned',separation:{episode_id:candidate.id,channel,origin_sample:320000,length_samples:32000}}))};
      doc.segments.splice(0,1,...structuredClone(historyFixture.accepted));delete doc.segments[0].protected_fields;doc.segments[1].protected_fields=[];historyFixture.accepted[0].protected_fields=[];
      if(edited)doc.segments[1].text='Keep this correction';
      selected.document=structuredClone(doc);api=async()=>structuredClone(historyFixture);renderSegments();`);
    const row=f.document.getElementById('segments').children[0],audio=row.querySelector('.separated-player');
    assert.match(audio.src,/separation\/.+\/0\/audio$/);assert.equal(audio.hidden,true);assert.equal(audio.controls,false);
    audio.dispatchEvent({type:'loadedmetadata'});assert.equal(audio.currentTime,0);
    row.querySelector('.seek').click();assert.equal(audio.currentTime,0);assert.match(row.textContent,/Check separated voices/);
    assert.equal(row.querySelector('.separated-voice'),null);assert.equal(row.querySelector('select').getAttribute('aria-label'),'Segment speaker');
    assert.equal(row.querySelector('textarea').readOnly===true,false);assert.equal(row.querySelector('textarea').value,'First voice');
    assert.equal(button(f,'Use both voices'),undefined);assert.equal(button(f,'Keep original'),undefined);
    const history=row.querySelector('.separation-history');assert.equal(history.tagName,'DETAILS');assert.equal(history.open===true,false);
    row.querySelectorAll('button').find(button=>button.textContent==='Play original recording').click();
    assert.match(f.document.getElementById('player').getAttribute('src'),/\/original\/audio$/);assert.equal(f.document.getElementById('player').currentTime,20);
    const original=button(f,'Show original passage');original.click();await Promise.all(original.lastEventResults);
    assert.equal(row.querySelector('.separation-original textarea').value,'Words 0');
    if(edited){assert.equal(button(f,'Restore original'),undefined);assert.match(row.textContent,/Copy the original words/);assert.equal(f.run('doc.segments[1].text'),'Keep this correction');}
    else assert.notEqual(button(f,'Restore original'),undefined);
  }
});
test('published voices in the draft renderer keep normal naming and play their own audio without dropping an older draft',()=>{
  const f=frontend(path.resolve(__dirname,'..'));overlappingFixture(f);
  f.run(`doc.segments[0].start=20;doc.segments[0].end=22;
    doc.segments[0].separation_candidate={id:'e'.repeat(32),original:structuredClone(doc.segments[0]),voices:[{text:'First voice',language:'en'},{text:'Second voice',language:'fr'}]};
    doc.segments[1].refinement_state='provisional';renderSegments();`);
  const text=f.document.getElementById('segments').children.find(row=>row.dataset.segmentId==='r0').querySelector('textarea');text.value='My retained local correction';text.dispatchEvent({type:'input'});
  f.run(`window.original=structuredClone(doc.segments[0]);
    const candidate=original.separation_candidate;
    doc.segments.splice(0,1,...candidate.voices.map((voice,channel)=>({...candidate.original,id:'e'+channel,text:voice.text,speaker:channel?'s1':'s0',separation:{episode_id:candidate.id,channel,origin_sample:320000,length_samples:32000}})));
    renderSegments();`);
  const rows=f.document.getElementById('segments').children,voice=rows.find(row=>row.dataset.segmentId==='e0'),old=rows.find(row=>row.dataset.segmentId==='r0');
  assert.equal(voice.querySelector('.name-speaker').textContent,'Maya');
  const audio=voice.querySelector('.separated-player');assert.equal(audio.hidden,true);voice.querySelector('.seek').click();assert.equal(audio.currentTime,0);assert.match(audio.src,/\/0\/audio$/);
  assert.equal(old.querySelector('textarea').value,'My retained local correction');assert.equal(f.run("passageDrafts.get('r0').text"),'My retained local correction');
  assert.equal(button(f,'Use both voices')?.closest('.segment'),old);
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
