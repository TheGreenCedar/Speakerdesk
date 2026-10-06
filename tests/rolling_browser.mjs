// CPU/browser fixture check, using an existing Chromium executable. No npm install.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm, mkdir, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';

const [base, chromium, output, mode='after'] = process.argv.slice(2);
assert(base && chromium && output, 'Pass fixture URL, installed Chromium path, screenshot directory, and optional before mode.');
await mkdir(output, {recursive:true});
const profile = await mkdtemp(join(tmpdir(), 'speakerdesk-browser-'));
const browser = spawn(chromium, ['--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], {stdio:'ignore'});
let ws;
try {
  let port;
  for (let i=0;i<100;i++) {
    try { port=(await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
    catch { await delay(100); }
  }
  assert(port, 'Chromium did not start.');
  const targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0;const pending=new Map();
  ws.onmessage=event=>{const message=JSON.parse(event.data);if(pending.has(message.id)){
    const [resolve,reject]=pending.get(message.id);pending.delete(message.id);
    message.error?reject(new Error(message.error.message)):resolve(message.result);
  }};
  const send=(method,params={})=>new Promise((resolve,reject)=>{
    const key=++id;pending.set(key,[resolve,reject]);ws.send(JSON.stringify({id:key,method,params}));
  });
  const evaluate=async expression=>{
    const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails)throw new Error(result.exceptionDetails.text+' '+JSON.stringify(result.exceptionDetails.exception));
    return result.result.value;
  };
  const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(100);}throw new Error(`Timed out: ${expression}`);};
  const click=async selector=>{
    const point=await evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});el.scrollIntoView({block:'center'});const r=el.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});
  };
  const input=(selector,value)=>evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});el.value=${JSON.stringify(value)};el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  const screenshot=async name=>{
    await evaluate('if(document.querySelector("#name-dialog").open)document.querySelector("#name-dialog").scrollTop=0');
    const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    await writeFile(join(output,name),Buffer.from(shot.data,'base64'));
  };
  const errors=[];ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);});
  await send('Runtime.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1280,height:mode==='before'?1050:800,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:`${base}/?meeting=${'d'.repeat(32)}`});
  await wait('typeof selected !== "undefined" && selected?.id');
  await wait('!meetingPoll && !polling');
  await evaluate('(async()=>{meetingPoll=true;polling=true;await api("/fixture/reset",{method:"POST"});meeting=null;meetingPoll=false;polling=false;await select(selected.id);await refreshMeeting();})()');
  await wait('meeting.language_revision === 0');
  await wait('document.querySelectorAll(".rolling-segment").length === 24');
  assert.equal(await evaluate('document.querySelectorAll(".review-tag, .passage-review").length'),0);
  assert.equal(await evaluate('document.querySelector("#pending-phrases").hidden'),false);
  await evaluate('window.draftNode=document.querySelector(".rolling-segment textarea");draftNode.focus()');
  await input('.rolling-segment textarea','My unsaved correction');
  await evaluate('api("/fixture/advance",{method:"POST",body:JSON.stringify({start:0})}).then(()=>poll())');
  await wait('doc.segments[0].end === 6');
  assert.equal(await evaluate('document.activeElement === draftNode'),true);
  assert.equal(await evaluate('draftNode.value'),'My unsaved correction');
  await click('.rolling-actions button');
  await wait('document.querySelector(".passage-save-status").textContent.includes("Compare both versions")');
  await wait('!isPassageSaving("row-0")');
  assert.equal(await evaluate('draftNode.value'),'My unsaved correction');
  await click('.rolling-actions .text-button');
  await wait('document.querySelector(".rolling-segment textarea").value.includes("larger-context")');
  await input('.rolling-segment textarea','My saved correction');
  await click('.rolling-actions button');
  await wait('!hasPassageDrafts()');
  await evaluate('api("/fixture/advance",{method:"POST",body:JSON.stringify({start:0})}).then(()=>poll())');
  const persisted=await evaluate(`api('/api/jobs/${'d'.repeat(32)}')`);
  assert.equal(persisted.document.segments[0].text,'My saved correction');
  assert.deepEqual(persisted.document.segments[0].protected_fields,['text']);
  assert.equal(persisted.document.speakers.speaker_0,'Confirmed Albert');
  await evaluate('document.activeElement.blur();followingLive=false;(()=>{const pane=$("transcript-pane"),card=document.querySelector("[data-segment-id=row-10]");pane.scrollTop+=card.getBoundingClientRect().top-pane.getBoundingClientRect().top;})()');
  const before=await evaluate('document.querySelector("[data-segment-id=row-10]").getBoundingClientRect().top');
  await evaluate('api("/fixture/advance",{method:"POST",body:JSON.stringify({start:6})}).then(()=>poll())');
  await wait('doc.segments.some(s=>s.start===6 && s.end===12)');
  const after=await evaluate('document.querySelector("[data-segment-id=row-10]").getBoundingClientRect().top');
  assert(Math.abs(before-after)<3,`Audio anchor moved ${after-before}px during merge.`);
  await click('#refinement-toggle');
  await wait('document.querySelector("#refinement-status").textContent.includes("paused")');
  await click('#refinement-toggle');
  await wait('document.querySelector("#refinement-status").textContent.includes("queued")');
  await evaluate('(()=>{const el=$("live-language");el.value="fr";el.dispatchEvent(new Event("change"));})()');
  await wait('meeting.language_revision === 1');
  const routed=await evaluate(`api('/api/jobs/${'d'.repeat(32)}')`);
  assert.equal(routed.language_revision,1);assert.equal(routed.language,'fr');
  assert.notEqual(await evaluate('$("notice").className'),'error');
  assert.equal(await evaluate('$("language").value'),'auto');
  assert.match(await evaluate('$("live-language-status").textContent'),/02:24.00/);
  assert.equal(await evaluate('doc.segments[0].text'),'My saved correction');
  await screenshot('rolling-anchor-preserved.png');
  await evaluate('speakerdeskTheme.set("light");$("transcript-pane").scrollTop=0');
  await screenshot('rolling-context-light.png');
  await evaluate('speakerdeskTheme.set("dark")');await screenshot('rolling-context-dark.png');
  await evaluate('api("/fixture/saved-owned",{method:"POST",body:JSON.stringify({owned:true})}).then(()=>poll())');
  await wait('selected.status === "ready" && selected.inference_owned');
  assert.equal(await evaluate('$("delete").disabled'),true);
  await evaluate('api("/fixture/saved-owned",{method:"POST",body:JSON.stringify({owned:false})}).then(()=>poll())');
  await wait('selected.status === "ready" && !selected.inference_owned');
  assert.equal(await evaluate('$("delete").disabled'),false);
  await evaluate('api("/fixture/review-edges",{method:"POST",body:JSON.stringify({live:true})}).then(()=>poll())');
  await wait('document.querySelectorAll(".review-audio-ranges").length === 2');
  const ranges=await evaluate('Array.from(document.querySelectorAll(".review-audio-ranges"),el=>el.textContent)');
  assert.match(ranges[0],/00:39\.8000–00:39\.8100 \(10 ms\)/);
  assert.match(ranges[1],/00:49\.5585–00:49\.5800 \(21\.5 ms\)/);
  assert.equal(await evaluate('doc.segments.find(s=>s.id === "unknown-audio").refinement_state'),'unresolved');
  assert.equal(await evaluate('document.querySelector("#segments [data-segment-id=unknown-audio]") === null'),true);
  assert.equal(await evaluate('document.querySelector("#retained-audio-review [data-segment-id=unknown-audio]") !== null'),true);
  await evaluate('document.querySelectorAll(".rolling-review").forEach(el=>{el.hidden=false;el.open=true;});$("retained-audio-review").open=true;$("transcript-pane").scrollTop=0');
  await screenshot('rolling-live-review-ranges.png');
  await evaluate('api("/fixture/review-edges",{method:"POST",body:JSON.stringify({live:false})}).then(()=>poll())');
  await wait('selected.status === "ready" && document.querySelectorAll(".rolling-segment").length === 0');
  const savedRanges=[];
  for(const sid of ['english-tail','french-tail']) {
    await evaluate(`document.querySelector('[data-segment-id=${sid}] .passage-details-toggle').click()`);
    savedRanges.push(await evaluate('document.querySelector("#segment-details .review-audio-ranges").textContent'));
  }
  assert.deepEqual(savedRanges,ranges);
  assert.equal(await evaluate('document.querySelector("#segments [data-segment-id=unknown-audio]") === null'),true);
  assert.match(await evaluate('document.querySelector("#retained-audio-review [data-segment-id=unknown-audio]").textContent'),/Overlapping speakers/);
  await screenshot('rolling-review-ranges.png');
  assert.deepEqual(errors,[]);
  await writeFile(join(output,'browser-report.json'),JSON.stringify({kind:'synthetic_cpu_fixture',checks:['collapsed_pending_rows','compact_provisional_states','focused_unsaved_draft_survives_merge','stale_row_save_rejected','saved_edit_protected_from_late_pass','confirmed_name_retained','audio_anchor_scroll_survives_merge','pause_resume_refinement','live_language_boundary_separate_from_default','saved_refinement_ownership_disables_delete','precise_uncovered_ranges','unknown_audio_review_retained'],page_errors:errors},null,2));
  console.log('Passed 12 rolling UI checks with zero page errors. Production assets/API and synthetic CPU outputs; no capture or native model accuracy claim.');

} finally {
  ws?.close();browser.kill('SIGTERM');
  await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
  await rm(profile,{recursive:true,force:true});
}
