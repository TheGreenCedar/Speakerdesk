// CPU DOM contracts using shipped renderers. No browser/listener/model/device.
const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

test('finished accuracy exceptions keep their reason beside retained words without implying active or interrupted work',()=>{
  const f=frontend(root);f.seed(2);
  f.run(`selected.rolling_refinement={version:1};selected.refinement_status='complete_with_exceptions';doc.segments[0].refinement_state='provisional';
    doc.segments[0].refinement_issue={kind:'accuracy',message:'The bounded decode returned no complete words.'};
    renderSegments();renderRefinementStatus();`);
  assert.match(f.document.getElementById('refinement-status').textContent,/finished.*exception/i);
  assert.match(f.document.getElementById('segments').children[0].textContent,/bounded decode/);
  assert.equal(f.document.getElementById('segments').children[0].querySelector('.passage-issue').closest('.rolling-review'),null);
  assert.equal(f.document.getElementById('pending-phrases').hidden,true);
  assert.match(f.document.getElementById('refinement-toggle').textContent,/retry/i);
});

test('an empty supported-language passage keeps its accuracy exception and repair visible after the pass finishes',()=>{
  const f=frontend(root);f.seed(1);
  f.run(`selected.rolling_refinement={version:1};selected.refinement_status='complete_with_exceptions';
    Object.assign(doc.segments[0],{text:'',canonical_utterance_id:'r0',canonical_state:'sealed',refinement_state:'provisional',language:'en',audio_state:'insufficient_speech',refinement_issue:{kind:'accuracy',message:'No complete words returned.'}});
    renderSegments();renderRefinementStatus();`);
  const card=f.document.getElementById('segments').children[0];assert.ok(card);
  assert.match(card.querySelector('.passage-issue').textContent,/No complete words/);
  assert.ok(card.querySelector('.passage-repair'));
  assert.equal(card.querySelector('.passage-issue').closest('.rolling-review'),null);
  assert.equal(f.document.getElementById('pending-phrases').hidden,true);
});

function fixture(rows,status='recording') {
  const f=frontend(root);f.seed(1,status);
  f.context.rows=rows.map((row,index)=>({id:'negative-'+index,start:index,end:index+1,speaker:'s0',
    text:'',review:true,refinement_state:'provisional',machine_revision:1,...row}));
  f.run('doc.segments=structuredClone(rows);window.original=JSON.stringify(doc);renderSegments()');
  return f;
}
const negativeRows=[
  {audio_state:'digital_silence'},
  {audio_state:'model_non_speech',language_detection:{mode:'auto',reason:'insufficient_speech'},
    acoustic_evidence:{decision:'no_speech',start_sample:0,end_sample:16000}},
  {audio_state:'model_non_speech',language_detection:{mode:'manual',reason:'insufficient_speech'}},
  {audio_state:'insufficient_speech'},
  {language_detection:{mode:'auto',reason:'insufficient_speech'}},
  {language_detection:{mode:'manual',reason:'insufficient_speech'}},
  {transcription_review:{reason:'insufficient_speech',partial_text:false}}
];

test('current and legacy empty negative hints add no transcript cards, audio-review ranges or pending words',()=>{
  const f=fixture(negativeRows);
  assert.equal(f.document.getElementById('segments').children.length,0);
  assert.equal(f.document.getElementById('retained-audio-review').hidden,true);
  assert.equal(f.document.getElementById('pending-phrases').hidden,true);
  assert.equal(f.run('doc.segments.length'),negativeRows.length);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

test('saved negative placeholders remain internal without inventing acoustic evidence or clearance',()=>{
  const f=fixture(negativeRows,'ready');
  assert.equal(f.document.getElementById('segments').children.length,0);
  assert.equal(f.document.getElementById('retained-audio-review').hidden,true);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
  assert.equal(f.run('doc.segments.filter(s=>s.acoustic_evidence).length'),1);
  assert.equal(f.run('doc.segments.every(s=>s.review===true)'),true);
  assert.equal(f.run('doc.segments.some(s=>s.review_resolution)'),false);
});

test('nonempty quiet, short, uncertain and corrected words override every negative presentation hint',()=>{
  const f=fixture(negativeRows.map((row,index)=>({...row,text:'Human or intelligible words '+index,
    ...(index===0?{protected_fields:['text'],refinement_state:'edited'}:{})})));
  assert.equal(f.document.getElementById('segments').children.length,negativeRows.length);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
  assert.deepEqual(f.document.getElementById('segments').children.map(c=>c.querySelector('textarea').value),
    negativeRows.map((row,index)=>'Human or intelligible words '+index));
});

test('explicit ASR errors, candidates, partial words and unassigned coverage remain reviewable',()=>{
  const exceptions=[
    {transcription_review:{reason:'transcription_failed'}},
    {transcription_review:{reason:'empty_result'}},
    {transcription_review:{reason:'token_limit',partial_text:true}},
    {transcription_review:{reason:'unassigned_audio'}},
    {transcription_review:{reason:'short_acoustic_context',candidate_text:'Exact candidate words.'}},
    {transcription_review:{reason:'insufficient_speech',candidate_text:'Retained uncertain candidate.'}},
    {transcription_review:{reason:'insufficient_speech',partial_text:true}}
  ];
  const f=fixture(exceptions.map(row=>({audio_state:'model_non_speech',...row})));
  assert.equal(f.document.getElementById('retained-audio-review').querySelectorAll('.retained-audio-range').length,exceptions.length);
  assert.equal(f.document.getElementById('retained-audio-review').querySelectorAll('textarea').length,2);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

test('positive, uncertain and pending acoustic receipts and structural short-input failures are not legacy silence',()=>{
  const rows=[...['speech','uncertain','pending'].map(decision=>({audio_state:'model_non_speech',
    language_detection:{mode:'auto',reason:'insufficient_speech'},acoustic_evidence:{decision}})),
    {audio_state:'insufficient_acoustic_context',language_detection:{mode:'auto',reason:'insufficient_speech'}},
    {audio_state:'speech_evidence_pending',language_detection:{mode:'auto',reason:'insufficient_speech'}}];
  const f=fixture(rows);
  assert.equal(f.document.getElementById('retained-audio-review').querySelectorAll('.retained-audio-range').length,rows.length);
  assert.equal(f.document.getElementById('pending-phrases').hidden,false);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

test('focused draft, recoverable correction, protected row and pending save survive empty negative refinements',()=>{
  const f=fixture([{text:'Machine words',audio_state:'model_non_speech'}]);
  const host=f.document.getElementById('segments'),text=host.children[0].querySelector('textarea');
  text.focus();text.value='Exact local correction';text.setSelectionRange(2,7);text.dispatchEvent({type:'input'});
  f.run("doc.segments[0].text='';doc.segments[0].machine_revision=2;renderLiveSegments()");
  assert.equal(host.children[0].querySelector('textarea'),text);
  assert.equal(text.value,'Exact local correction');assert.equal(text.selectionStart,2);assert.equal(text.selectionEnd,7);
  text.blur();f.run("recoverablePassageDrafts.set('negative-0',structuredClone(passageDrafts.get('negative-0')));passageDrafts.clear();renderLiveSegments()");
  assert.equal(host.children.length,1);assert.equal(host.children[0].querySelector('.recover-correction').disabled,false);
  f.run("recoverablePassageDrafts.clear();doc.segments[0].protected_fields=['text'];renderLiveSegments()");
  assert.equal(host.children.length,1);
  f.run("delete doc.segments[0].protected_fields;passageSaveOperations.set(selected.id+':negative-0',{jid:selected.id,segment:structuredClone(doc.segments[0])});renderLiveSegments()");
  assert.equal(host.children.length,1);assert.equal(host.children[0].querySelector('.save-passage').disabled,true);
  f.run('passageSaveOperations.clear();renderLiveSegments()');
  assert.equal(host.children.length,0);assert.equal(f.document.getElementById('retained-audio-review').hidden,true);
  assert.equal(f.run('doc.segments.length'),1);
});

for(const state of ['unresolved','provisional'])test(`stopped ${state} language gaps stay in reading order with a visible repair`,async()=>{
  const f=fixture([
    {text:'Words before the gap.',refinement_state:'refined'},
    {language_detection:{mode:'auto',reason:'uncertain'},refinement_state:state},
    {audio_state:'digital_silence',refinement_state:'refined'},
    {text:'Words after the gap.',refinement_state:'refined'}
  ],'ready');
  f.run("selected.rolling_refinement={version:1};selected.refinement_status='unresolved';renderSegments();renderRefinementStatus()");
  const host=f.document.getElementById('segments');
  assert.deepEqual(host.children.map(c=>c.dataset.segmentId),['negative-0','negative-1','negative-3']);
  const gap=host.children[1];
  assert.match(gap.querySelector('.passage-issue').textContent,/language/i);
  const repair=gap.querySelector('.passage-repair');
  assert.equal(repair.open,true);
  assert.equal(repair.closest('.rolling-review'),null,'The repair must not sit inside hidden evidence');
  assert.equal(f.document.getElementById('retained-audio-review').hidden,true);
  assert.match(f.document.getElementById('refinement-status').textContent,/1 passage needs a language/);
  await f.run('toggleRefinement()');
  assert.equal(f.document.activeElement,repair.querySelector('select'));
  assert.equal(f.run('JSON.stringify(doc)===original'),true,'Visibility never manufactures words or resolves evidence');
});

for(const state of ['unresolved','provisional'])test(`a failed ${state} language recheck surfaces its repair beside retained words`,async()=>{
  const f=fixture([
    {text:'These words are already readable.',language:'en',language_detection:{mode:'auto',reason:'detected'},refinement_state:state,
      refinement_issue:{kind:'language',reason:'unsupported',message:'Language detection selected an unsupported language.'}},
    {text:'A successful best-effort sentence.',language:'en',language_detection:{mode:'auto',reason:'uncertain'},refinement_state:'refined'}
  ],'ready');
  f.run("selected.rolling_refinement={version:1};selected.refinement_status='unresolved';renderSegments();renderRefinementStatus()");
  const cards=f.document.getElementById('segments').children,repair=cards[0].querySelector('.passage-repair');
  assert.match(cards[0].querySelector('.passage-issue')?.textContent || '',/unsupported language|language is not supported/i);
  assert.equal(repair.open,true);
  assert.equal(repair.closest('.rolling-review'),null);
  assert.equal(cards[0].querySelector('textarea').value,'These words are already readable.');
  assert.equal(cards[1].querySelector('.passage-issue'),null,'Accepted readable uncertainty creates no repair obligation');
  assert.match(f.document.getElementById('refinement-status').textContent,/1 passage needs a language/);
  await f.run('toggleRefinement()');
  assert.equal(f.document.activeElement,repair.querySelector('select'));
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
  f.run("delete doc.segments[0].refinement_issue;doc.segments[0].refinement_state='refined';selected.refinement_status='complete';renderSegments();renderRefinementStatus()");
  assert.equal(f.document.getElementById('refinement-status').textContent,'Accuracy pass complete');
  f.document.activeElement?.blur();f.run('renderSegments()');
  assert.equal(f.document.getElementById('segments').children[0].querySelector('.passage-issue'),null);
});

test('a language failure arriving during an edit becomes visible without replacing the focused passage',()=>{
  const f=fixture([{text:'Readable original words.',language:'en',language_detection:{reason:'detected'}}],'ready');
  f.run("selected.rolling_refinement={version:1};selected.refinement_status='refining';renderSegments()");
  const card=f.document.getElementById('segments').children[0],text=card.querySelector('textarea');
  text.focus();text.value='My correction in progress';text.setSelectionRange(3,8);text.dispatchEvent({type:'input'});
  f.run("doc.segments[0].refinement_issue={kind:'language',reason:'uncertain'};selected.refinement_status='unresolved';renderSegments();renderRefinementStatus()");
  assert.equal(f.document.getElementById('segments').children[0],card);
  assert.equal(card.querySelector('textarea'),text);
  assert.equal(f.document.activeElement,text);
  assert.equal(text.value,'My correction in progress');
  assert.equal(text.selectionStart,3);assert.equal(text.selectionEnd,8);
  assert.equal(card.querySelector('.passage-issue').closest('.rolling-review'),null);
  assert.equal(card.querySelector('.passage-repair').open,true);
  assert.equal(card.querySelector('.passage-retry button').disabled,true);
  assert.match(card.querySelector('.rolling-repair').textContent,/Save or discard/);
  assert.equal(f.run('doc.segments[0].text'),'Readable original words.');
});

for(const state of ['unresolved','provisional'])test(`a filtered ${state} language gap is reachable and long retry limits have usable timing controls`,async()=>{
  const f=fixture([{end:30,language_detection:{mode:'auto',reason:'uncertain'},refinement_state:state}], 'ready');
  f.run("selected.duration=30;selected.rolling_refinement={version:1};selected.refinement_status='unresolved';$('search').value='unrelated words';renderSegments();renderRefinementStatus()");
  assert.equal(f.document.getElementById('segments').children.length,0);
  await f.run('toggleRefinement()');
  const card=f.document.getElementById('segments').children[0],repair=card.querySelector('.passage-repair');
  assert.equal(f.document.getElementById('search').value,'');
  assert.equal(f.document.activeElement,repair.querySelector('select'));
  assert.equal(repair.querySelector('.passage-retry button').disabled,true);
  assert.match(repair.textContent,/up to 24.5 seconds/);
  const timing=repair.querySelectorAll('button').find(button=>button.textContent==='Adjust passage timing');
  timing.click();
  assert.equal(f.document.getElementById('inspector').hidden,false);
  const end=f.document.getElementById('segment-details').querySelectorAll('input')[1];
  end.value='20';end.dispatchEvent({type:'change'});
  assert.equal(f.run('doc.segments[0].end'),20);
  assert.equal(f.document.getElementById('segments').children[0].querySelector('.passage-retry button').disabled,false);
});

test('repair polling retains a focused correction and clears the blocker only when words arrive',()=>{
  const f=fixture([{language_detection:{mode:'auto',reason:'uncertain'}}], 'ready');
  f.run("selected.rolling_refinement={version:1};selected.refinement_status='unresolved';renderSegments()");
  const host=f.document.getElementById('segments'),text=host.children[0].querySelector('textarea');
  text.focus();text.value='My correction';text.setSelectionRange(2,5);text.dispatchEvent({type:'input'});
  f.run("doc.segments[0].machine_revision=2;renderLiveSegments()");
  assert.equal(host.children[0].querySelector('textarea'),text);
  assert.equal(text.value,'My correction');assert.equal(text.selectionStart,2);assert.equal(text.selectionEnd,5);
  assert.equal(host.children[0].querySelector('.passage-retry button').disabled,true);
  text.blur();f.run("passageDrafts.clear();doc.segments[0].text='Accepted words';doc.segments[0].language='fr';doc.segments[0].refinement_state='edited';doc.segments[0].protected_fields=['text'];selected.refinement_status='complete';renderSegments();renderRefinementStatus()");
  assert.equal(host.children[0].querySelector('.passage-issue'),null);
  assert.equal(host.children[0].querySelector('textarea').value,'Accepted words');
  assert.equal(f.document.getElementById('refinement-status').textContent,'Accuracy pass complete');
});

test('timing repair rechecks drafts and pending saves created after opening the inspector',async()=>{
  for(const pending of ['draft','save']) {
    const f=fixture([{end:30,language_detection:{mode:'auto',reason:'uncertain'}},{text:'Another passage.'}], 'ready');
    f.run("selected.duration=40;selected.rolling_refinement={version:1};selected.refinement_status='unresolved';renderSegments()");
    const card=f.document.getElementById('segments').children[0];
    card.querySelectorAll('button').find(button=>button.textContent==='Adjust passage timing').click();
    const end=f.document.getElementById('segment-details').querySelectorAll('input')[1];
    if(pending==='draft') {
      const text=f.document.getElementById('segments').children[1].querySelector('textarea');
      text.value='Unsaved correction';text.dispatchEvent({type:'input'});
    } else f.run("passageSaveOperations.set(selected.id+':negative-1',{jid:selected.id,segment:structuredClone(doc.segments[1])})");
    end.value='20';end.dispatchEvent({type:'change'});
    assert.equal(f.run('doc.segments[0].end'),30,pending);
    await f.run('toggleRefinement()');
    const current=f.document.getElementById('segments').children[0];
    assert.equal(current.querySelector('.passage-repair select').disabled,true);
    assert.equal(f.document.activeElement,current,'A disabled language selector must not be the navigation target');
    assert.match(current.querySelector('.passage-repair').textContent,/Save or discard/);
  }
});

test('a replaced saved passage cannot be edited through its old repair timing inspector',()=>{
  const f=fixture([{end:30,language_detection:{mode:'auto',reason:'uncertain'},refinement_state:'unresolved'}], 'ready');
  f.run('selected.duration=30;renderSegments()');
  const card=f.document.getElementById('segments').children[0];
  card.querySelectorAll('button').find(button=>button.textContent==='Adjust passage timing').click();
  const end=f.document.getElementById('segment-details').querySelectorAll('input')[1];
  f.run('doc.segments[0]=structuredClone(doc.segments[0]);doc.segments[0].machine_revision++;renderSegments()');
  end.value='20';end.dispatchEvent({type:'change'});
  assert.equal(f.run('doc.segments[0].end'),30);
  assert.equal(f.run('dirty'),false,'A detached inspector must not schedule a whole-document save');
  assert.match(f.document.getElementById('notice').textContent,/open the current passage/);
});
