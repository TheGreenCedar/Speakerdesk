// Run against support/live_timeline_server.py: production publication and Flask,
// synthetic model observations, installed Chromium, no capture or neural model.
// node tests/live_timeline_browser.cjs <base> <chromium> <output> [before]
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const {mkdtemp, readFile, rm, mkdir, writeFile} = require('node:fs/promises');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {createHash}=require('node:crypto');
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
    let id=0;const pending=new Map(),errors=[],dialogs=[];
    ws.onmessage=event=>{const m=JSON.parse(event.data);
      if(m.method==='Page.javascriptDialogOpening'){
        dialogs.push(m.params.message);
        const replacement=m.params.type==='confirm' && m.params.message==='Replace the transcript of Rolling context · synthetic CPU fixture with meeting.json?';
        if(!replacement)errors.push(`Unexpected dialog: ${m.params.message}`);
        send('Page.handleJavaScriptDialog',{accept:replacement}).catch(error=>errors.push(error.message));
      }
      if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);
      if(pending.has(m.id)){const [resolve,reject]=pending.get(m.id);pending.delete(m.id);m.error?reject(new Error(m.error.message)):resolve(m.result);}};
    const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,[resolve,reject]);ws.send(JSON.stringify({id:key,method,params}));});
    const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
      if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
    const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(100);}throw new Error(`Timed out: ${expression}`);};
    const screenshot=async name=>{const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(join(output,name),Buffer.from(shot.data,'base64'));};
    await send('Runtime.enable');await send('Page.enable');await send('Page.addScriptToEvaluateOnNewDocument',{source:'window.setInterval=()=>0;'});
    await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:`${base}/?meeting=${'d'.repeat(32)}`});
    await wait('typeof selected !== "undefined" && selected?.id && !meetingPoll && !polling');
    const measures=[],violations=[];
    let state=await evaluate("api('/fixture/state')");
    const tail=state.ids.tail,delayed=state.ids.delayed;
    const measure=async step=>{
      const newest=state.expected_order.at(-1);
      const m=await evaluate(`(()=>{const cards=Array.from($('segments').children),pane=$('transcript-pane'),tail=document.querySelector('[data-segment-id="${tail}"]'),newest=document.querySelector('[data-segment-id="${newest}"]');return {
        step:${JSON.stringify(step)},revision:selected.revision,ids:cards.map(c=>c.dataset.segmentId),
        sections:cards.map(c=>c.dataset.transcriptSection || 'processed'),
        starts:cards.map(c=>Number(c.dataset.start ?? doc.segments.find(s=>s.id===c.dataset.segmentId)?.start)),
        tailPresent:!!tail,tailWords:tail?.querySelector('textarea')?.value,
        tailVisible:!!tail && tail.getBoundingClientRect().bottom>pane.getBoundingClientRect().top && tail.getBoundingClientRect().top<pane.getBoundingClientRect().bottom,
        newestVisible:!!newest && newest.getBoundingClientRect().bottom>pane.getBoundingClientRect().top && newest.getBoundingClientRect().top<pane.getBoundingClientRect().bottom,
        following:isLive() && followingLive && !$('segments').contains(document.activeElement),hasDraft:hasPassageDrafts(),
        scrollTop:pane.scrollTop,scrollHeight:pane.scrollHeight,atBottom:Math.abs(pane.scrollHeight-pane.clientHeight-pane.scrollTop)<2,
        labels:Array.from(document.querySelectorAll('.transcript-section-label')).map(e=>e.textContent),
        textClipped:Array.from($('segments').querySelectorAll('textarea')).some(t=>t.checkVisibility() && t.scrollHeight>t.clientHeight+1),
        overflow:document.body.scrollWidth>innerWidth}})()`);
      measures.push(m);
      if(JSON.stringify(m.ids)!==JSON.stringify(state.expected_order))violations.push(`${step}: publication chronology changed in presentation`);
      if(!m.tailPresent)violations.push(`${step}: nonempty canonical tail disappeared`);
      if(m.following && !m.newestVisible)violations.push(`${step}: newest published sentence disappeared from the followed viewport`);
      if(!m.hasDraft && m.tailWords!==state.words[tail])violations.push(`${step}: rendered tail differs from the current published words`);
      if(m.textClipped)violations.push(`${step}: visible textarea clips words`);
      if(m.overflow)violations.push(`${step}: horizontal overflow`);
      return m;
    };
    const step=async name=>{
      state=await evaluate(`api('/fixture/step/${name}',{method:'POST'})`);
      await evaluate('poll()');await delay(50);return measure(name);
    };
    await evaluate('followingLive=true;renderSegments()');await measure('initial');
    await screenshot('timeline-initial.png');
    await step('grow');await screenshot('timeline-long-sentence.png');
    const longWords=measures.at(-1).tailWords;
    await step('partial');assert.equal(measures.at(-1).tailWords,longWords);
    await step('empty-publication');assert.equal(measures.at(-1).tailWords,longWords);
    await step('silence');await step('interjection');
    await evaluate(`document.querySelector('[data-segment-id="${tail}"] textarea').focus();window.tailText=document.activeElement;tailText.value='Exact human correction';tailText.dispatchEvent(new Event('input',{bubbles:true}));tailText.setSelectionRange(3,9,'backward');window.anchorTop=tailText.getBoundingClientRect().top;window.tailCard=tailText.closest('article');`);
    await step('delayed');await step('refine-tail');await step('late-live');
    const draft=await evaluate(`({same:document.activeElement===tailText && tailText.closest('article')===tailCard,value:tailText.value,caret:[tailText.selectionStart,tailText.selectionEnd,tailText.selectionDirection],latest:tailCard.querySelector('.latest-machine-words').textContent,anchorDrift:Math.abs(tailText.getBoundingClientRect().top-anchorTop)})`);
    assert.equal(draft.same,true);assert.equal(draft.value,'Exact human correction');assert.deepEqual(draft.caret,[3,9,'backward']);
    assert(draft.latest.endsWith('LATE FAST REVISION'));
    await screenshot('timeline-draft-revisions.png');
    const revision=state.revision;await step('stale-publication');assert.equal(state.revision,revision);
    await evaluate('tailText.blur();tailCard.querySelector(".keep-correction").click()');
    await wait('!hasPendingPassageSaves() && !hasPassageDrafts() && !polling');
    assert.equal(await evaluate(`doc.segments.find(s=>s.id==='${tail}').text`),'Exact human correction');
    assert.equal(await evaluate(`doc.segments.find(s=>s.id==='${tail}').protected_fields.includes('text')`),true);
    await step('stop');await step('complete');
    assert.equal(measures.at(-1).tailWords,'Exact human correction');
    assert.equal(await evaluate("document.querySelectorAll('.live-provisional,.live-section-label').length"),0);
    await screenshot('timeline-complete.png');
    // Hold an actual conditional Flask GET after its snapshot is formed, then
    // finish another real publication. The older reply must not roll it back.
    await evaluate("(async()=>{await api('/fixture/hold',{method:'POST'});window.oldPoll=poll()})()");
    await wait("api('/fixture/held').then(s=>s.held)");
    state=await evaluate("api('/fixture/step/late-live',{method:'POST'})");
    await evaluate("(async()=>{const newest=await api('/api/jobs/'+selected.id);selected=newest;doc=structuredClone(newest.document);renderSegments();await api('/fixture/release',{method:'POST'});await oldPoll})()");
    assert.equal(await evaluate('selected.revision'),state.revision);
    assert.equal(await evaluate(`doc.segments.find(s=>s.id==='${tail}').text`),'Exact human correction');
    for(const [width,theme] of [[760,'dark'],[1280,'light']]){
      await send('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
      await evaluate(`document.documentElement.dataset.theme='${theme}';$('transcript-pane').scrollTop=0`);
      await screenshot(`timeline-complete-${width}-${theme}.png`);
    }
    // A replaced ID with an unsaved correction remains separately recoverable;
    // the machine timeline honors the real PUT removal and later publication.
    await evaluate(`doc.segments.find(s=>s.id==='${delayed}').refinement_state='provisional';renderSegments();tailText=document.querySelector('[data-segment-id="${tail}"] textarea');tailText.focus();tailText.value='Retained removed-row draft';tailText.dispatchEvent(new Event('input',{bubbles:true}));`);
    state=await evaluate("api('/fixture/step/remove',{method:'POST'})");await evaluate('poll()');
    assert.equal(await evaluate(`doc.segments.some(s=>s.id==='${tail}')`),false);
    assert.equal(await evaluate("document.querySelector('[data-transcript-section=corrections] textarea').value"),'Retained removed-row draft');
    assert.equal(await evaluate('document.activeElement===tailText'),true);
    await screenshot('timeline-retained-correction.png');
    const source={};for(const file of ['app.js','live_editor.js','style.css'])source[file]=createHash('sha256').update(await readFile(join(process.env.FRONTEND_SOURCE_ROOT || join(__dirname,'..'),'speakerdesk/static',file))).digest('hex');
    const report={surface:'Shipped HTML/JS/CSS in installed Chromium; actual model-free Flask publication fixture',source,
      observations:'Synthetic speech/model results; production canonical IDs, revision policy, host reconciliation, SQLite, edits and conditional GET',
      neuralInferenceExecuted:false,activeAppTouched:false,mode,measures,draft,violations,pageErrors:errors,
      olderConditionalFlaskReplyRejected:true,realPassagePatchProtectedWords:true,realDocumentRemovalHonored:true};
    await writeFile(join(output,'timeline-browser-report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({mode,steps:measures.length,violations,pageErrors:errors,draft:{...draft,latestCharacters:draft.latest.length,latest:undefined}},null,2));
    if(mode!=='before')assert.deepEqual(violations,[]);
    assert.deepEqual(errors,[]);
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
