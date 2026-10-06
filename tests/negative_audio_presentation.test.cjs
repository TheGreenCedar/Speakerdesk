// CPU DOM contracts using shipped renderers. No browser/listener/model/device.
const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');

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
