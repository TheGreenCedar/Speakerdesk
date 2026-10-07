// Shipped renderer CPU contracts; not browser/layout or model accuracy evidence.
const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const fs=require('node:fs');
const {frontend}=require('./support/frontend_dom.cjs');
const root=process.env.FRONTEND_SOURCE_ROOT||path.resolve(__dirname,'..');
function projection(parts) {
  let cursor=0;
  return parts.map(part=>{const start_offset=cursor;cursor+=Array.from(part.text).length;
    return {start_offset,end_offset:cursor,start:null,end:null,attribution:'single',...part};});
}
function fixture(parts,status='ready') {
  const f=frontend(root);f.seed(1,status);
  f.context.turns=projection(parts);
  f.run("doc.speakers={multiple_speakers:'Multiple speakers',speaker_0:'Albert',speaker_1:'Maya'};Object.assign(doc.segments[0],{text:turns.map(t=>t.text).join(''),speaker:'multiple_speakers',speaker_candidates:['speaker_0','speaker_1'],reading_turns:structuredClone(turns),start:0,end:339,audio_anchor:{start_sample:0,end_sample:5424000}});window.original=JSON.stringify(doc);renderSegments()");
  return f;
}
const parts=[{text:'Hello 🙂, français.  ',speaker:'speaker_0',start:0,end:3},
  {text:'The next reply.\n',speaker:'speaker_1',start:3,end:6},
  {text:'Last words stay exact.',speaker:'speaker_0',start:6,end:9}];
const card=f=>f.document.getElementById('segments').children[0];

test('canonical unassigned words use current temporal concurrency, never candidate union, for overlap warnings',()=>{
  const f=fixture(parts);
  f.run(`(()=>{const s=doc.segments[0];delete s.reading_turns;
    Object.assign(s,{canonical_utterance_id:s.id,speaker:'unassigned',review:true,audio_revision:3,
      start_sample:0,end_sample:160000,text_audio_anchor:{start_sample:0,end_sample:160000},
      speaker_activity:{audio_revision:3,regions:[
        {start_sample:0,end_sample:80000,speakers:['speaker_0']},
        {start_sample:80000,end_sample:160000,speakers:['speaker_1']}]}});})()`);
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),false);
  assert.equal(f.run('passageReviewReason(doc.segments[0])'),'Speaker uncertain');
  f.run("doc.segments[0].review_resolution='words_reviewed'");
  assert.equal(f.run('passageReviewReason(doc.segments[0])'),'Speaker uncertain');
  f.run('delete doc.segments[0].review_resolution');
  f.run("doc.segments[0].speaker_activity.regions[0].speakers.push('speaker_1')");
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),true);
  assert.match(f.run('passageReviewReason(doc.segments[0])'),/Overlapping speakers/);
  f.run('doc.segments[0].speaker_activity.audio_revision=2');
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),false);
  f.run("delete doc.segments[0].speaker_activity;doc.segments[0].speaker='overlap_speaker_0_speaker_1'");
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),true);
  f.run("doc.segments[0].speaker='speaker_0';delete doc.segments[0].canonical_utterance_id");
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),true);
});

test('saved sequential reading turns preserve Unicode, spaces, one parent ID and one whole-parent editor',()=>{
  const f=fixture(parts),c=card(f);
  assert.equal(f.document.getElementById('segments').children.length,1);assert.equal(c.dataset.segmentId,'r0');
  assert.equal(c.querySelector('textarea').hidden,true);
  assert.equal(c.querySelector('textarea').value,parts.map(t=>t.text).join(''));
  assert.deepEqual(c.querySelectorAll('.reading-turn-words').map(e=>e.textContent),parts.map(t=>t.text));
  assert.deepEqual(c.querySelectorAll('.reading-turn-speaker').map(e=>e.textContent),['Albert','Maya','Albert']);
  assert.equal(c.querySelectorAll('.edit-whole-passage').length,1);
  assert.equal(c.querySelectorAll('.reading-turn button,.reading-turn textarea,.reading-turn select').length,0);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),false);
  assert.equal(f.run('passageReviewReason(doc.segments[0])'),'');
});

test('unknown text has no invented time and actual overlap is labeled independently of sequential owners',()=>{
  const f=fixture([{text:'Uncertain words. ',speaker:'unassigned',attribution:'unknown'},
    {text:'Simultaneous words. ',speaker:'overlap_speaker_0_speaker_1',attribution:'overlap',start:4,end:7},
    {text:'Clear words.',speaker:'speaker_0',start:7,end:9}]);
  const rows=card(f).querySelectorAll('.reading-turn');
  assert.deepEqual(rows.map(e=>e.dataset.attribution),['unknown','overlap','single']);
  assert.deepEqual(rows.map(e=>e.querySelector('.reading-turn-speaker').textContent),['Unknown speaker','Overlapping speakers','Albert']);
  assert.equal(rows[0].querySelector('.reading-turn-time'),null);
  assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),true);
  assert.match(f.run('passageReviewReason(doc.segments[0])'),/Overlapping speakers/);
});

test('partially known time bounds preserve attributed words without manufacturing a complete time',()=>{
  const f=fixture([{text:'Known start only. ',speaker:'speaker_0',start:2,end:null},
    {text:'Known end only.',speaker:'unassigned',attribution:'unknown',start:null,end:8}]);
  assert.equal(card(f).querySelectorAll('.reading-turn').length,2);
  assert.equal(card(f).querySelectorAll('.reading-turn-time').length,0);
  assert.equal(card(f).querySelectorAll('.reading-turn-words').map(e=>e.textContent).join(''),'Known start only. Known end only.');
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

test('adjacent word slices with the same owner form a continuous turn without per-word controls',()=>{
  const f=fixture([{text:'One ',speaker:'speaker_0',start:0,end:1},{text:'two ',speaker:'speaker_0',start:1,end:2},
    {text:'three. ',speaker:'speaker_0',start:2,end:3},{text:'Reply.',speaker:'speaker_1',start:3,end:4}]);
  assert.equal(card(f).querySelectorAll('.reading-turn').length,2);
  assert.deepEqual(card(f).querySelectorAll('.reading-turn-words').map(e=>e.textContent),['One two three. ','Reply.']);
  assert.equal(f.run('doc.segments[0].reading_turns.length'),4);
});

test('a 339-second 1061-word canonical passage reads as eight turns with one shared editor',()=>{
  const words=Array.from({length:1061},(_,index)=>({text:'word'+index+(index===1060?'':' '),
    speaker:'speaker_'+(Math.floor(index/150)%2),start:index*339/1061,end:(index+1)*339/1061}));
  const f=fixture(words),c=card(f);
  assert.equal(f.document.getElementById('segments').children.length,1);
  assert.equal(c.querySelectorAll('.reading-turn').length,8);
  assert.equal(c.querySelectorAll('.edit-whole-passage').length,1);
  const reading=c.querySelectorAll('.reading-turn-words').map(e=>e.textContent).join('');
  assert.equal(reading,words.map(w=>w.text).join(''));assert.equal(reading.split(' ').length,1061);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

test('missing, stale, malformed and UTF16-offset projections fall back to exact full-parent text',()=>{
  const mutations=["delete doc.segments[0].reading_turns", "doc.segments[0].reading_turns[0].text='Wrong'",
    "doc.segments[0].reading_turns[1].start_offset++", "doc.segments[0].reading_turns[0].end_offset++",
    "doc.segments[0].reading_turns[0].attribution='invented'", "doc.segments[0].reading_turns[0].start=-1"];
  for(const mutation of mutations){const f=fixture(parts);f.run(mutation+';renderSegments()');
    assert.equal(card(f).querySelector('textarea').hidden,false);assert.equal(card(f).querySelector('textarea').value,parts.map(t=>t.text).join(''));
    assert.ok(!card(f).querySelector('.reading-turn-view') || card(f).querySelector('.reading-turn-view').hidden);}
});

test('explicit editing retains the saved textarea and hides attribution as soon as words change',()=>{
  const f=fixture(parts),c=card(f),text=c.querySelector('textarea');
  c.querySelector('.edit-whole-passage').click();assert.equal(f.document.activeElement,text);assert.equal(text.hidden,false);
  c.querySelector('.edit-whole-passage').click();assert.equal(text.hidden,true);
  c.querySelector('.edit-whole-passage').click();text.value='My corrected whole passage';text.setSelectionRange(3,9);text.dispatchEvent({type:'input'});
  assert.equal(c.querySelector('.reading-turn-view').hidden,true);assert.equal(c.querySelector('.edit-whole-passage').hidden,true);
  assert.equal(f.run('doc.segments[0].text'),'My corrected whole passage');
  f.run('doc=structuredClone(doc);renderSegments()');assert.equal(card(f).querySelector('textarea'),text);
  assert.equal(text.selectionStart,3);assert.equal(text.selectionEnd,9);
});

test('live refinement preserves focused draft, caret, parent audio anchor and displayed CAS revision',()=>{
  const f=fixture(parts,'recording'),c=card(f),text=c.querySelector('textarea');
  c.querySelector('.edit-whole-passage').click();text.value='My live correction';text.setSelectionRange(2,8);text.dispatchEvent({type:'input'});
  f.run("doc.segments[0].text='New machine words';doc.segments[0].machine_revision=2;renderLiveSegments()");
  assert.equal(card(f),c);assert.equal(card(f).querySelector('textarea'),text);
  assert.equal(text.value,'My live correction');assert.equal(text.selectionStart,2);assert.equal(text.selectionEnd,8);
  assert.equal(c.querySelector('.reading-turn-view').hidden,true);
  assert.equal(c.querySelector('.keep-correction').dataset.revision,'2');
  assert.equal(f.run('doc.segments[0].audio_anchor.end_sample'),5424000);
  assert.equal(f.run('passageDrafts.get("r0").revision'),1);
});

test('speaker-name and exact-word search keep the canonical passage; renaming updates labels',()=>{
  const f=fixture(parts);f.run("$('search').value='maya';renderSegments()");assert.equal(f.document.getElementById('segments').children.length,1);
  f.run("$('search').value='français';renderSegments()");assert.equal(f.document.getElementById('segments').children.length,1);
  f.run("$('search').value='';doc.speakers.speaker_1='Chris';renderSegments()");
  assert.deepEqual(card(f).querySelectorAll('.reading-turn-speaker').map(e=>e.textContent),['Albert','Chris','Albert']);
  f.run("$('search').value='does not exist';renderSegments()");assert.equal(f.document.getElementById('segments').children.length,0);
});

test('parent playback and explicit review do not reinterpret temporal speaker union as overlap',()=>{
  const f=fixture(parts);card(f).querySelector('.seek').click();
  assert.equal(f.document.getElementById('player').currentTime,0);assert.equal(f.run('passageEnd'),339);
  assert.equal(f.run('markReviewable(doc.segments[0])'),true);
  f.run("doc.segments[0].review_resolution='words_reviewed'");assert.equal(f.run('passageReviewReason(doc.segments[0])'),'');
  f.run('delete doc.segments[0].reading_turns');assert.equal(f.run('hasOverlappingSpeakers(doc.segments[0])'),false);
});

test('A/unknown/A reads as one explicitly partly unassigned paragraph without timing or altered words',()=>{
  const parts=[{text:'First words.  ',speaker:'speaker_0',start:0,end:2},
    {text:'uncertain 🙂 ',speaker:'unassigned',attribution:'unknown'},
    {text:'Same speaker continues.',speaker:'speaker_0',start:4,end:7}];
  const f=fixture(parts),rows=card(f).querySelectorAll('.reading-turn');
  assert.equal(rows.length,1);assert.equal(rows[0].dataset.attribution,'single_with_unassigned');
  assert.equal(rows[0].querySelector('.reading-turn-speaker').textContent,'Albert');
  assert.equal(rows[0].querySelector('.reading-unassigned-note').textContent,'Includes unassigned words');
  assert.equal(rows[0].querySelector('.reading-turn-words').textContent,parts.map(t=>t.text).join(''));
  assert.equal(rows[0].querySelector('.reading-turn-time'),null);
  const span=rows[0].querySelector('.reading-turn-unassigned');
  assert.equal(span.textContent,parts[1].text);assert.equal(span.dataset.speaker,'unassigned');
  assert.equal(span.dataset.startOffset,String(Array.from(parts[0].text).length));
  assert.match(span.title,/Unassigned words/);assert.equal(span.getAttribute('aria-label'),'Unassigned words: '+parts[1].text);
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
  card(f).querySelector('.edit-whole-passage').click();
  assert.equal(card(f).querySelector('textarea').value,parts.map(t=>t.text).join(''));
});

test('A/unknown/B, long unknown, standalone unknown and actual overlap remain separate',()=>{
  for(const middle of [{text:'Unknown words. ',speaker:'unassigned',attribution:'unknown'},
    {text:'Simultaneous words. ',speaker:'overlap_speaker_0_speaker_1',attribution:'overlap',start:2,end:3}]){
    const f=fixture([{text:'First. ',speaker:'speaker_0',start:0,end:2},middle,
      {text:'Following words.',speaker:middle.attribution==='overlap'?'speaker_0':'speaker_1',start:3,end:6}]);
    assert.equal(card(f).querySelectorAll('.reading-turn').length,3);
    assert.equal(card(f).querySelectorAll('.reading-unassigned-note').length,0);
    assert.equal(f.run('JSON.stringify(doc)===original'),true);
  }
  const f=fixture([{text:'First. ',speaker:'speaker_0',start:0,end:2},
    {text:'This longer unassigned passage has more than six words. ',speaker:'unassigned',attribution:'unknown'},
    {text:'Same owner.',speaker:'speaker_0',start:6,end:8}]);
  assert.equal(card(f).querySelectorAll('.reading-turn').length,3);
  assert.equal(card(f).querySelectorAll('.reading-turn-speaker')[1].textContent,'Unknown speaker');
  const alone=fixture([{text:'Unassigned alone.',speaker:'unassigned',attribution:'unknown'}]);
  assert.equal(card(alone).querySelector('.reading-turn-speaker').textContent,'Unknown speaker');
});

test('repeated brief unknown bridges retain every uncertainty span and suppress incomplete timing',()=>{
  const f=fixture([{text:'A ',speaker:'speaker_0',start:0,end:1},
    {text:'u ',speaker:'unassigned',attribution:'unknown',start:1,end:2},
    {text:'B ',speaker:'speaker_0',start:2,end:3},
    {text:'v ',speaker:'unassigned',attribution:'unknown',start:3,end:null},
    {text:'C',speaker:'speaker_0',start:4,end:5}]);
  assert.equal(card(f).querySelectorAll('.reading-turn').length,1);
  assert.equal(card(f).querySelectorAll('.reading-turn-unassigned').length,2);
  assert.equal(card(f).querySelector('.reading-turn-time'),null);
  assert.equal(card(f).querySelector('.reading-turn-words').textContent,'A u B v C');
  assert.equal(f.run('JSON.stringify(doc)===original'),true);
});

test('read-only actual source projection reduces presentation fragmentation without changing any canonical word or slice',
  {skip:!process.env.READING_TURN_JOB},()=>{
    const job=JSON.parse(fs.readFileSync(process.env.READING_TURN_JOB,'utf8'));
    const f=frontend(root);f.seed();f.context.actualJob=job;
    f.run('selected=structuredClone(actualJob);doc=structuredClone(selected.document);window.original=JSON.stringify(doc);renderSegments()');
    const segment=job.document.segments.reduce((largest,row)=>(row.reading_turns?.length||0)>(largest.reading_turns?.length||0)?row:largest);
    const c=f.document.getElementById('segments').children.find(c=>c.dataset.segmentId===segment.id);
    assert.equal(segment.reading_turns.length,112);assert.equal(segment.text.split(/\s+/).length,1061);
    const paragraphs=c.querySelectorAll('.reading-turn'),notes=c.querySelectorAll('.reading-unassigned-note');
    assert.ok(paragraphs.length<112);assert.ok(notes.length>0);
    assert.equal(c.querySelectorAll('.reading-turn-words').map(e=>e.textContent).join(''),segment.text);
    assert.equal(c.querySelectorAll('.edit-whole-passage').length,1);
    assert.equal(f.run('JSON.stringify(doc)===original'),true);
    console.log(JSON.stringify({actualCanonicalSlices:112,readingParagraphs:paragraphs.length,paragraphsWithUnassignedWords:notes.length,
      inlineUnassignedSpans:c.querySelectorAll('.reading-turn-unassigned').length,words:1061}));
  });
