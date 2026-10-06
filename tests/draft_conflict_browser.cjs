// Protected-draft conflict recovery using the real CPU fixture API and browser. Uses installed Chromium;
// starts no devices or model workers and downloads no browser dependencies.
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
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
    await evaluate(`(async()=>{await api('/fixture/reset',{method:'POST'});await select(selected.id);await refreshMeeting();
      window.patchResults=[];window.originalFetch=fetch;
      window.fetch=async(input,options)=>{const response=await originalFetch(input,options);if(options?.method==='PATCH' && String(input).includes('/segments/'))patchResults.push({status:response.status,body:JSON.parse(options.body)});return response;};})()`);
    await evaluate(`(()=>{window.draftElement=document.querySelector('[data-segment-id=row-0] textarea');draftElement.focus();draftElement.value='My exact local correction';draftElement.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    await evaluate("(async()=>{await api('/fixture/advance',{method:'POST',body:JSON.stringify({start:0})});await poll()})()");
    assert.equal(await evaluate('draftElement.value'),'My exact local correction');
    await evaluate("document.querySelector('[data-segment-id=row-0] .rolling-actions button').click()");
    await wait('patchResults.length===1 && !polling');
    assert.equal(await evaluate('patchResults[0].status'),409);
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-0] .local-correction-copy').value"),'My exact local correction');
    assert.match(await evaluate("document.querySelector('[data-segment-id=row-0] .latest-machine-words').textContent"),/larger-context/);
    const displayed=await evaluate("Number(document.querySelector('[data-segment-id=row-0] .keep-correction').dataset.revision)");
    // Advance real server revision without updating the displayed comparison.
    await evaluate("api('/fixture/advance',{method:'POST',body:JSON.stringify({start:0})})");
    await evaluate("document.querySelector('[data-segment-id=row-0] .keep-correction').click()");
    await wait('patchResults.length===2 && !polling');
    assert.equal(await evaluate('patchResults[1].status'),409);
    assert.equal(await evaluate('patchResults[1].body.segment_revision'),displayed);
    assert.deepEqual(await evaluate('Object.keys(patchResults[1].body.changes)'),['text']);
    assert.equal(await evaluate('draftElement.value'),'My exact local correction');
    await screenshot('draft-conflict-compare-before-keep.png');
    checks.push('Actual CPU fixture returns409 for stale draft; comparison retains copyable correction and latest machine words; an unseen newer revision still rejects explicit keep against the displayed revision with text-only changes.');
    await evaluate("document.querySelector('[data-segment-id=row-0] .keep-correction').click()");
    await wait('patchResults.length===3 && !hasPassageDrafts()');
    assert.equal(await evaluate('patchResults[2].status'),200);
    await evaluate("(async()=>{await api('/fixture/advance',{method:'POST',body:JSON.stringify({start:0})});await poll()})()");
    const saved=await evaluate("api('/api/jobs/'+selected.id)");
    const protectedRow=saved.document.segments.find(s=>s.id==='row-0');
    assert.equal(protectedRow.text,'My exact local correction');assert.deepEqual(protectedRow.protected_fields,['text']);
    assert.equal(protectedRow.start,0);assert.equal(protectedRow.end,6);assert.equal(protectedRow.speaker,'speaker_0');
    checks.push('Explicit keep succeeds against the compared revision and further actual late refinement preserves protected words, speaker and audio bounds.');
    await evaluate(`(()=>{window.recoveryElement=document.querySelector('[data-segment-id=row-2] textarea');recoveryElement.focus();recoveryElement.value='Recover this local correction';recoveryElement.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    await evaluate("(async()=>{await api('/fixture/advance',{method:'POST',body:JSON.stringify({start:6})});await poll()})()");
    await evaluate("document.querySelector('[data-segment-id=row-2] .rolling-actions .text-button').click()");
    assert.equal(await evaluate("passageDrafts.has('row-2')"),false);
    assert.equal(await evaluate("recoverablePassageDrafts.get('row-2').text"),'Recover this local correction');
    await evaluate("document.querySelector('[data-segment-id=row-2] .recover-correction').click()");
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-2] textarea').value"),'Recover this local correction');
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-2] .local-correction-copy').value"),'Recover this local correction');
    assert.equal(await evaluate("passageDrafts.has('row-2')"),true);
    await screenshot('draft-recovered-after-use-latest.png');
    checks.push('Use latest retains the replaced local draft; explicit Recover my correction restores it with comparison instead of silently overwriting newer words.');
    await evaluate(`(()=>{const row=doc.segments.find(s=>s.id==='row-2');row.text='';row.machine_revision++;row.refinement_state='unresolved';delete row.protected_fields;renderLiveSegments();
      document.querySelector('[data-segment-id=row-2] .rolling-actions .text-button').click();})()`);
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-2] .recover-correction')!==null"),true);
    await evaluate("document.querySelector('[data-segment-id=row-2] .recover-correction').click()");
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-2] textarea').value"),'Recover this local correction');
    assert.equal(await evaluate("doc.segments.find(s=>s.id==='row-2').text"),'');
    checks.push('Controlled empty-result UI peer cannot hide recovery: Use latest on blank machine words still exposes Recover my correction without inventing server text.');
    await evaluate(`(async()=>{passageDrafts.clear();recoverablePassageDrafts.clear();dirty=false;await api('/fixture/reset',{method:'POST'});await select(selected.id);
      const text=document.querySelector('[data-segment-id=row-0] textarea');text.focus();text.value='Recovery copy retained during pending save';text.dispatchEvent(new Event('input',{bubbles:true}));
      await api('/fixture/advance',{method:'POST',body:JSON.stringify({start:0})});await poll();
      document.querySelector('[data-segment-id=row-0] .use-latest').click();document.querySelector('[data-segment-id=row-0] .recover-correction').click();
      window.heldBaseFetch=fetch;window.heldPatchCount=0;window.replyHeld=false;window.holdSaveReply=true;
      window.fetch=async(input,options)=>{const response=await heldBaseFetch(input,options);
        if(options?.method==='PATCH' && String(input).includes('/segments/')){heldPatchCount++;if(holdSaveReply){holdSaveReply=false;replyHeld=true;await new Promise(resolve=>window.releaseSaveReply=resolve);}}return response;};})()`);
    await evaluate("document.querySelector('[data-segment-id=row-0] .keep-correction').click()");
    await wait('replyHeld');
    assert.equal(await evaluate("['.save-passage','.keep-correction','.use-latest','.recover-correction'].every(selector=>document.querySelector('[data-segment-id=row-0] '+selector).disabled)"),true);
    // Force events even on disabled controls to prove event guards, then rebuild
    // the card while the actual server response remains held in the browser.
    await evaluate(`(()=>{const card=document.querySelector('[data-segment-id=row-0]');
      for(const selector of ['.save-passage','.keep-correction','.use-latest','.recover-correction']){const button=card.querySelector(selector);button.disabled=false;button.click();}
      card.remove();renderLiveSegments();})()`);
    assert.equal(await evaluate('heldPatchCount'),1);
    assert.equal(await evaluate("isPassageSaving('row-0')"),true);
    assert.equal(await evaluate("recoverablePassageDrafts.get('row-0').text"),'Recovery copy retained during pending save');
    assert.equal(await evaluate("['.save-passage','.keep-correction','.use-latest','.recover-correction'].every(selector=>document.querySelector('[data-segment-id=row-0] '+selector).disabled)"),true);
    await evaluate(`(()=>{const text=document.querySelector('[data-segment-id=row-0] textarea');text.focus();text.value='Newer correction typed during pending response';text.dispatchEvent(new Event('input',{bubbles:true}));releaseSaveReply();})()`);
    await wait("!isPassageSaving('row-0') && !polling");
    assert.equal(await evaluate("passageDrafts.get('row-0').text"),'Newer correction typed during pending response');
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-0] textarea').value"),'Newer correction typed during pending response');
    assert.equal((await evaluate("api('/api/jobs/'+selected.id)")).document.segments.find(s=>s.id==='row-0').text,'Recovery copy retained during pending save');
    await evaluate("document.querySelector('[data-segment-id=row-0] .use-latest').click()");
    assert.equal(await evaluate("recoverablePassageDrafts.get('row-0').text"),'Newer correction typed during pending response');
    await evaluate("document.querySelector('[data-segment-id=row-0] .recover-correction').click()");
    assert.equal(await evaluate("document.querySelector('[data-segment-id=row-0] textarea').value"),'Newer correction typed during pending response');
    checks.push('Held actual successful PATCH keeps durable meeting+row busy state across forced control events/card rebuild, blocks overlapping requests and destructive latest/recover choices, retains the recovery copy until acknowledgement, preserves typing during the held response, and re-enables explicit recovery afterward.');
    assert.deepEqual(errors,[]);
    await writeFile(join(output,'draft-conflict-browser-report.json'),JSON.stringify({kind:'real_CPU_fixture_API_and_browser',modelsOrCaptureExecuted:false,checks,actualPatchResults:await evaluate('patchResults'),pageErrors:errors},null,2));
    console.log(JSON.stringify({checks,pageErrors:errors},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
