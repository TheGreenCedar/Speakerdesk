// Uncluttered transcript reading and on-demand evidence verification using controlled CPU fixture outputs. Uses installed Chromium;
// starts no devices or model workers and downloads no browser dependencies.
const assert = require('node:assert/strict');
const {spawn,execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const {mkdtemp, readFile, readdir, rm, mkdir, writeFile} = require('node:fs/promises');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {setTimeout: delay} = require('node:timers/promises');
const [base, chromium, output, mode='after'] = process.argv.slice(2);
async function main() {
  assert(base && chromium && output, 'Pass fixture URL, installed Chromium path, output and optional before mode.');
  await mkdir(output,{recursive:true});
  const profile=await mkdtemp(join(tmpdir(),'speakerdesk-density-'));
  const browser=spawn(chromium,['--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
    '--disable-background-networking','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});
  let ws;
  try {
    let port;
    for(let i=0;i<100;i++){try{port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await delay(100);}}
    assert(port,'Chromium did not start');
    const targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
    let id=0;const pending=new Map(),errors=[];
    ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);
      if(pending.has(m.id)){const [resolve,reject]=pending.get(m.id);pending.delete(m.id);m.error?reject(new Error(m.error.message)):resolve(m.result);}};
    const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,[resolve,reject]);ws.send(JSON.stringify({id:key,method,params}));});
    const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
      if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
    const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(100);}throw new Error(`Timed out: ${expression}`);};
    const screenshot=async name=>{const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(join(output,name),Buffer.from(shot.data,'base64'));};
    await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:`${base}/?meeting=${'d'.repeat(32)}`});
    await wait('typeof selected !== "undefined" && selected?.id && !meetingPoll && !polling');
    const checks=[];
    const source={head:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),sha256:{}};
    for(const file of ['speakerdesk/static/app.js','speakerdesk/static/live_editor.js'])source.sha256[file]=createHash('sha256').update(await readFile(file)).digest('hex');
    await evaluate("(async()=>{await api('/fixture/candidate-review',{method:'POST'});await poll();await refreshMeeting();polling=true;meetingPoll=true;notice('')})()");
    const original=await evaluate('structuredClone(doc)');
    await evaluate(`(()=>{doc.segments.find(s=>s.id==='quiet-positive').review=true;
      doc.segments.find(s=>s.id==='quiet-positive').confidence=.38;
      doc.segments.find(s=>s.id==='quiet-positive').language_detection.probability=.42;
      window.readingDocument=JSON.stringify(doc);selected.status='recording';selected.rolling_refinement=true;selected.refinement_status='unresolved';
      setStatus();renderEditor();$('transcript-pane').scrollTop=0;})()`);
    const assertQuietReading=async()=>{
      assert.equal(await evaluate(`document.querySelectorAll('#segments .review-tag:not(.routine-state),#segments .passage-context-note').length`),0);
      const reading=await evaluate(`$('segments').innerText`);
      assert.doesNotMatch(reading,/Needs review|Language needs review|Best-effort|Language inferred|short audio fragment|Audio to review|Play this passage to review/);
      assert.equal(await evaluate(`$('retained-audio-review').querySelector(':scope > summary').textContent`),'Audio details');
      assert.doesNotMatch(await evaluate(`$('segment-count').textContent+' '+$('refinement-status').textContent`),/to review|need.*review/i);
      assert.equal(await evaluate(`JSON.stringify(doc)===readingDocument`),true);
    };
    await assertQuietReading();
    assert.equal(await evaluate(`document.querySelectorAll('#segments .segment').length`),3);
    assert.equal(await evaluate(`document.querySelector('[data-segment-id=brief-positive] textarea').value`),'Brief words stay visible.');
    await screenshot('live-reading-clean-1280x800.png');
    checks.push('Live reading retains quiet/100ms/250ms words without badges, confidence/language prose, review counts or canonical row changes.');
    await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .passage-details-toggle').click()`);
    assert.equal(await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .rolling-review').checkVisibility()`),true);
    assert.match(await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .rolling-review').innerText`),/Best-effort/);
    await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .passage-evidence').open=true`);
    const metadata=JSON.parse(await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .passage-evidence pre').textContent`));
    assert.equal(metadata.review,true);assert.equal(metadata.confidence,.38);assert.equal(metadata.language_detection.probability,.42);assert.equal(metadata.recording_provenance.kind,'local_inference');
    await screenshot('live-passage-evidence-1280x800.png');
    checks.push('Explicit live Details exposes exact confidence, language evidence and recording provenance without changing review state.');
    await evaluate(`(()=>{selected.status='ready';selected.refinement_status='complete';setStatus();renderEditor();$('transcript-pane').scrollTop=0})()`);
    await assertQuietReading();
    assert.equal(await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .seek').disabled`),false);
    await evaluate(`document.querySelector('[data-segment-id=quiet-positive] textarea').focus()`);
    assert.equal(await evaluate(`$('inspector').hidden`),true,'Ordinary editing must not open review prose.');
    await screenshot('saved-reading-clean-1280x800.png');
    await evaluate(`document.querySelector('[data-segment-id=quiet-positive] .passage-details-toggle').click()`);
    assert.equal(await evaluate(`$('inspector').hidden`),false);
    assert.match(await evaluate(`$('segment-details').innerText`),/Best-effort/);
    const savedMetadata=JSON.parse(await evaluate(`document.querySelector('#segment-details .passage-evidence pre').textContent`));
    assert.deepEqual(savedMetadata,metadata);
    checks.push('Saved transcript keeps playback and editing; only explicit Details opens inspector evidence, with metadata identical to live.');
    await evaluate(`(()=>{const row=doc.segments.find(s=>s.id==='brief-positive');showInspector(row,document.querySelector('[data-segment-id=brief-positive]'),true);})()`);
    assert.match(await evaluate(`$('segment-details').innerText`),/short audio fragment/);
    assert.equal(await evaluate(`document.querySelector('#segment-details .review-action')!==null`),true);
    await screenshot('saved-passage-evidence-1280x800.png');
    for(const kind of ['txt','srt','vtt']) {
      const result=await evaluate(`fetch('/api/jobs/'+selected.id+'/export/${kind}').then(response=>response.text())`);
      assert.equal(result.includes('Brief words stay visible.'),true);assert.equal(result.includes('Quiet but intelligible speech remains visible.'),true);
      assert.equal(result.includes('Unconfirmed tail words'),false);
    }
    const project=await evaluate("fetch('/api/jobs/'+selected.id+'/export/json').then(response=>response.json())");
    assert.equal(project.segments.length,original.segments.length);
    assert.deepEqual(project.segments.find(s=>s.id==='brief-positive').transcription_review,original.segments.find(s=>s.id==='brief-positive').transcription_review);
    checks.push('Real export routes preserve short/quiet words, candidate omission and internal review metadata; UI changes do not alter export semantics.');
    assert.deepEqual(errors,[]);
    for(const file of Object.keys(source.sha256))assert.equal(createHash('sha256').update(await readFile(file)).digest('hex'),source.sha256[file]);
    await writeFile(join(output,'reading-review-browser-report.json'),JSON.stringify({fixture:'Synthetic CPU fixture, real API/assets; no models/capture/installed app',source,checks,pageErrors:errors},null,2));
    console.log(JSON.stringify({source,checks,pageErrors:errors},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
