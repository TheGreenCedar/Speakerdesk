// Candidate word review/export verification using controlled CPU fixture outputs. Uses installed Chromium;
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
    const files=['speakerdesk/static/app.js','speakerdesk/static/live_editor.js','speakerdesk/static/style.css','speakerdesk/transcript.py','speakerdesk/review.py','speakerdesk/language_detection.py'];
    const source={head:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),sha256:{}};
    for(const file of files)source.sha256[file]=createHash('sha256').update(await readFile(file)).digest('hex');
    await evaluate("(async()=>{await api('/fixture/candidate-review',{method:'POST'});await poll();await refreshMeeting();polling=true;meetingPoll=true;notice('')})()");
    assert.equal(await evaluate("document.querySelectorAll('#segments .segment').length"),3);
    assert.equal(await evaluate("document.querySelector('#segments [data-segment-id=candidate-tail]')===null"),true);
    assert.equal(await evaluate("document.querySelector('#segments [data-segment-id=candidate-no-speech]')===null"),true);
    assert.equal(await evaluate("$('retained-audio-review').hidden"),false);
    assert.equal(await evaluate("$('retained-audio-review').open"),false);
    assert.equal(await evaluate("document.querySelectorAll('#retained-audio-review textarea[aria-label=\"Unconfirmed model words\"]').length"),2);
    checks.push('Unconfirmed tail/non-speech candidates have no primary reading cards; three retained audio ranges stay in collapsed review.');
    await evaluate("$('retained-audio-review').open=true;document.querySelectorAll('#retained-audio-review details').forEach(el=>el.open=true)");
    const candidateUI=await evaluate(`[...document.querySelectorAll('#retained-audio-review textarea[aria-label="Unconfirmed model words"]')].map(el=>({id:el.closest('[data-segment-id]').dataset.segmentId,value:el.value,readonly:el.readOnly,
      visible:el.checkVisibility(),fontSize:getComputedStyle(el).fontSize,width:el.clientWidth,height:el.clientHeight,scrollHeight:el.scrollHeight,overflowY:getComputedStyle(el).overflowY}))`);
    const canonical=await evaluate("api('/api/jobs/'+selected.id)");
    for(const row of candidateUI){assert.equal(row.readonly,true);assert.equal(row.visible,true);assert.equal(row.value,canonical.document.segments.find(s=>s.id===row.id).transcription_review.candidate_text);}
    await evaluate(`(()=>{const text=document.querySelector('#retained-audio-review textarea[aria-label="Unconfirmed model words"]');text.focus();text.select();})()`);
    assert.equal(await evaluate(`(()=>{const text=document.querySelector('#retained-audio-review textarea[aria-label="Unconfirmed model words"]');return text.value.slice(text.selectionStart,text.selectionEnd)===text.value})()`),true);
    checks.push('Expanded candidates remain readonly and preserve exact text, punctuation and accents; all words can be selected for copying.');
    await screenshot('candidate-review-expanded-1280x800.png');
    for(const kind of ['txt','srt','vtt']){
      const result=await evaluate(`fetch('/api/jobs/'+selected.id+'/export/${kind}').then(response=>response.text())`);
      assert.equal(result.includes('Unconfirmed tail words'),false);assert.equal(result.includes('without a detected speaker'),false);
      assert.equal(result.includes('Quiet but intelligible speech remains visible.'),true);assert.equal(result.includes('Yes.'),true);
      assert.equal(result.includes('Brief words stay visible.'),true);
    }
    const project=await evaluate("fetch('/api/jobs/'+selected.id+'/export/json').then(response=>response.json())");
    assert.equal(project.segments.length,6);
    for(const row of candidateUI)assert.equal(project.segments.find(s=>s.id===row.id).transcription_review.candidate_text,row.value);
    assert.equal(project.segments.find(s=>s.id==='candidate-tail').text,'');
    checks.push('Real text/SRT/VTT exports omit acoustic candidates but retain quiet,250ms and100ms reviewed/context-fallback speech; JSON retains all six rows and exact candidate metadata.');
    assert.equal(await evaluate("document.querySelector('[data-segment-id=quiet-positive] textarea').value"),'Quiet but intelligible speech remains visible.');
    assert.equal(await evaluate("document.querySelector('[data-segment-id=short-positive] textarea').value"),'Yes.');
    assert.equal(await evaluate("document.querySelector('[data-segment-id=brief-positive] textarea').value"),'Brief words stay visible.');
    assert.equal(await evaluate("document.querySelector('#retained-audio-review [data-segment-id=brief-positive]')===null"),true);
    assert.equal(await evaluate("document.querySelector('#retained-audio-review [data-segment-id=guarded-fragment] textarea[aria-label=\"Unconfirmed model words\"]')===null"),true);
    checks.push('Positive quiet/short words remain editable reading passages; a guarded fragment without a model candidate does not fabricate a candidate control.');
    assert.deepEqual(errors,[]);
    for(const file of files)assert.equal(createHash('sha256').update(await readFile(file)).digest('hex'),source.sha256[file],`Source changed during browser verification: ${file}`);
    await writeFile(join(output,'candidate-browser-report.json'),JSON.stringify({fixture:'Controlled CPU fixture outputs; no models/capture/native dialog',source,candidateUI,readabilityConcerns:candidateUI.filter(row=>parseFloat(row.fontSize)<14).map(row=>`${row.id}: ${row.fontSize} candidate text`),checks,pageErrors:errors},null,2));
    console.log(JSON.stringify({source,candidateUI,checks,pageErrors:errors},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
