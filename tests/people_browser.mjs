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
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1280,height:mode==='before'?1050:1200,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:`${base}/?meeting=${'a'.repeat(32)}`});
  await wait('document.querySelector(".name-speaker") && typeof openNamePicker === "function"');
  await click('.name-speaker');await wait('document.querySelector("#name-dialog").open');
  await input('#name-value','Albert');await click('#name-add-person');
  await wait('document.querySelector("#name-person").value !== ""');
  await click('#name-form button[type=submit]');
  if(mode==='before') {
    await wait('!document.querySelector("#name-dialog").open');
    await click('.name-speaker');await wait('document.querySelector("#name-dialog").open');
    assert.equal(await evaluate('document.querySelector("#voice-clip-list").children.length'),0);
    assert.equal(await evaluate('document.querySelector("#voice-clips").hidden'),false);
    await screenshot('people-before-empty.png');
    console.log('Reproduced: Apply closes the flow; reopening shows an empty clip selector without an availability reason.');
  } else {
    await wait('document.querySelector("#name-message").textContent === "Name applied to this meeting." && !document.querySelector("#refresh-voice-clips").disabled');
    assert.equal(await evaluate('document.querySelector("#name-dialog").open'),true,'Applying a name keeps enrollment discoverable.');
    assert.equal(await evaluate('document.querySelector("#voice-clips").hidden'),true,'An empty selector is not shown.');
    assert.match(await evaluate('document.querySelector("#voice-availability").textContent'),/No usable passages/);
    assert.equal(await evaluate('document.querySelector("#voice-excluded-list").children.length'),2);
    assert.match(await evaluate('document.querySelector("#voice-excluded-list").textContent'),/Shorter than 2 seconds/);
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    assert.equal(await evaluate('document.querySelector("#voice-consent").checked'),false);
    await screenshot('people-after-short-turns-light.png');
    await click('#name-done');
    await evaluate(`select('${'b'.repeat(32)}')`);
    await wait('document.querySelector("#recording-name").textContent === "Clean spoken passages"');
    assert.match(await evaluate('document.querySelector(".name-speaker").getAttribute("aria-label")'),/Name or remember voice for /);
    await screenshot('people-speaker-action-light.png');
    await click('.name-speaker');
    await wait('document.querySelector("#name-dialog").open && !document.querySelector("#name-apply").disabled');
    await evaluate('(()=>{const el=document.querySelector("#name-person");el.selectedIndex=1;el.dispatchEvent(new Event("change"));})()');
    await click('#name-apply');
    await wait('document.querySelector("#name-message").textContent === "Name applied to this meeting." && !document.querySelector("#refresh-voice-clips").disabled');
    assert.equal(await evaluate('document.querySelectorAll("#voice-clip-list input:checked").length'),2);
    assert.equal(await evaluate('document.querySelector("#voice-consent").checked'),false);
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    await click('#voice-clip-list button');
    await wait('!document.querySelector("#voice-preview").paused');
    await evaluate('document.querySelector("#voice-preview").currentTime = 3.1');
    await wait('document.querySelector("#voice-preview").paused');
    assert.equal(await evaluate('document.querySelector("#voice-preview").currentTime'),3,'Preview stops at the exact clip end.');
    await click('#voice-consent');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),false);
    // Changing the person/name invalidates the enrollment action and its consent.
    await input('#name-value','Someone else');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    assert.equal(await evaluate('document.querySelector("#voice-consent").checked'),false);
    await evaluate('(()=>{const el=document.querySelector("#name-person");el.selectedIndex=1;el.dispatchEvent(new Event("change"));})()');
    await screenshot('people-after-ready-light.png');
    await evaluate('speakerdeskTheme.set("dark")');await screenshot('people-after-ready-dark.png');
    await click('#voice-consent');await click('#remember-voice');
    await wait('!document.querySelector("#name-dialog").open');
    const people=await evaluate('api("/api/people")');
    assert.equal(people.people.length,1);assert.equal(people.people[0].voice_saved,true);
    const job=await evaluate(`api('/api/jobs/${'b'.repeat(32)}')`);
    assert.equal(job.document.speakers.speaker_0,'Albert');
    assert.equal(job.document.segments[0].text,'Synthetic passage for the voice setup preview.');
    console.log('Passed: visible naming action, retained Apply flow, actionable empty reasons, selected clips, bounded preview, person-change consent reset, explicit enrollment persisted, light/dark screenshots. Synthetic backend only.');
  }
} finally {
  ws?.close();browser.kill('SIGTERM');
  await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
  await rm(profile,{recursive:true,force:true});
}
