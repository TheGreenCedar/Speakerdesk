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
  const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(100);}await screenshot("failed-state.png");throw new Error(`Timed out: ${expression}; ${await evaluate('JSON.stringify({active:document.activeElement?.id,message:document.querySelector("#name-message")?.textContent,note:document.querySelector("#voice-note")?.textContent,apply:document.querySelector("#name-apply")?.disabled})')}`);};
  const click=async selector=>{
    const point=await evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});el.scrollIntoView({block:'center'});const r=el.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});
  };
  const key=async key=>{const codes={Enter:13,Escape:27,Tab:9,' ':32};await send('Input.dispatchKeyEvent',{type:'keyDown',key,code:key===' '?'Space':key,windowsVirtualKeyCode:codes[key],text:key==='Enter'?'\r':key===' '?' ':undefined});await send('Input.dispatchKeyEvent',{type:'keyUp',key,code:key===' '?'Space':key,windowsVirtualKeyCode:codes[key]});};
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
  await wait('document.querySelector("#name-message").textContent === "Name applied to this meeting." && !document.querySelector("#refresh-voice-clips").disabled');
  assert.equal(await evaluate('document.querySelector("#name-dialog").open'),true);
  assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
  await click('#remember-voice');
  assert.equal((await evaluate('api("/fixture/report")')).embedding_calls.length,0,'Disabled mouse activation never enrolls.');
  if(mode==='before') {
    assert.equal(await evaluate('document.querySelector("#voice-consent").disabled'),true);
    await click('#voice-consent');
    assert.equal(await evaluate('document.querySelector("#voice-consent").checked'),false);
    await screenshot('voice-before-two-disabled-controls.png');
  } else {
    assert.equal(await evaluate('document.querySelector("#voice-consent")'),null,'Consent comes from one explicit action.');
    assert.equal(await evaluate('document.querySelector("#voice-clips").hidden'),true);
    assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/No usable passages/);
    assert.match(await evaluate('document.querySelector("#voice-excluded-list").textContent'),/Shorter than 2 seconds/);
    await screenshot('voice-short-passages.png');
  }
  await key('Escape');
  await wait('!document.querySelector("#name-dialog").open');
  assert.equal((await evaluate('api("/fixture/report")')).profiles.length,0,'Escaping the modal never saves a voice.');
  await evaluate(`select('${'b'.repeat(32)}')`);
  await wait('document.querySelector("#recording-name").textContent === "Clean spoken passages"');
  await click('.name-speaker');
  await wait('document.querySelector("#name-dialog").open && !document.querySelector("#name-apply").disabled');
  await evaluate('(()=>{const el=document.querySelector("#name-person");el.selectedIndex=1;el.dispatchEvent(new Event("change"));})()');
  assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true,'A selected person must first be applied.');
  if(mode!=='before')assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/Apply this saved person/);
  await evaluate('document.querySelector("#name-apply").focus()');await key('Enter');
  await wait('document.querySelector("#name-message").textContent === "Name applied to this meeting." && !document.querySelector("#refresh-voice-clips").disabled');
  assert.equal(await evaluate('document.querySelectorAll("#voice-clip-list input:checked").length'),2);
  if(mode==='before') {
    assert.equal(await evaluate('document.querySelector("#voice-consent").disabled'),false);
    assert.equal(await evaluate('document.querySelector("#voice-consent").checked'),false);
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true,'Even qualified clips require the redundant checkbox.');
    await screenshot('voice-before-ready-checkbox-gate.png');
    console.log('Reproduced current UI: insufficient clips disable both controls; qualified clips leave Remember voice disabled until an additional checkbox is checked. Native readiness is a separate backend gate.');
  } else {
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),false);
    assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/Choosing Remember voice saves Albert.*on this Mac/);
    assert.match(await evaluate('document.querySelector("#remember-voice").getAttribute("aria-describedby")'),/voice-note/);
    await click('#voice-clip-list input');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/Select 2.*nonoverlapping/);
    await evaluate('document.querySelector("#voice-clip-list input").focus()');await key(' ');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),false,'Keyboard selection restores eligible passages.');
    await click('#voice-clip-list button');
    await wait('!document.querySelector("#voice-preview").paused');
    await evaluate('document.querySelector("#voice-preview").currentTime = 3.1');
    await wait('document.querySelector("#voice-preview").paused');
    assert.equal(await evaluate('document.querySelector("#voice-preview").currentTime'),3);
    await input('#name-value','Someone else');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    await evaluate('(()=>{const el=document.querySelector("#name-person");el.selectedIndex=1;el.dispatchEvent(new Event("change"));})()');
    await evaluate('document.querySelector("#name-value").focus()');
    for(let i=0;i<18;i++) { await key('Tab'); assert.equal(await evaluate('!!document.activeElement.closest("#name-dialog")'),true,'Tab stays inside the native modal.'); }
    await evaluate('speakerdeskTheme.set("light")');await screenshot('voice-ready-light.png');
    await evaluate('speakerdeskTheme.set("dark")');await screenshot('voice-ready-dark.png');
    await click('#remember-voice');
    await wait('!document.querySelector("#name-dialog").open');
    let report=await evaluate('api("/fixture/report")');
    assert.equal(report.profiles.length,1);assert.equal(report.embedding_calls.length,2,'One mouse action embeds exactly the two selected synthetic passages.');
    assert.equal(report.profiles[0].consent,'explicit_remember_voice');
    const version=report.profiles[0].version;
    await evaluate('document.querySelector(".name-speaker").focus()');await key('Enter');
    await wait('document.querySelector("#name-dialog").open && !document.querySelector("#remember-voice").disabled');
    assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/replaces Albert/);
    await evaluate('api("/fixture/voice-state",{method:"POST",body:JSON.stringify({revision:true})})');
    await wait('document.querySelector("#voice-note").textContent.includes("transcript changed")');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    await click('#refresh-voice-clips');await wait('!document.querySelector("#remember-voice").disabled');
    await evaluate('api("/fixture/voice-state",{method:"POST",body:JSON.stringify({unavailable:true})})');
    await click('#refresh-voice-clips');
    await wait('!document.querySelector("#refresh-voice-clips").disabled && !document.querySelector("#voice-settings").hidden');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/(?:verified|qualified) GPU\/ANE runtime/);
    await click('#remember-voice');
    report=await evaluate('api("/fixture/report")');assert.equal(report.embedding_calls.length,2);assert.equal(report.profiles[0].version,version);
    await screenshot('voice-runtime-blocked.png');
    const readiness=await evaluate(`api('/api/jobs/${'b'.repeat(32)}/speakers/speaker_0/voice-clips')`);
    if(mode==='after-contract') { assert.equal(readiness.readiness_code,'runtime_unqualified');assert.equal(readiness.next_action,'updates');assert.doesNotMatch(readiness.message,/finish local model setup/i); }
    const destination=readiness.next_action==='updates'?'updates':'voices';
    await click('#voice-settings');await wait('!document.querySelector("#name-dialog").open && !document.querySelector("#setup").hidden');
    assert.equal(await evaluate(`document.querySelector('#panel-${destination}').hidden`),false,'Runtime blocker opens its backend-prescribed destination.');
    await click('#settings-close');
    if(mode==='after-contract') {
      await evaluate('api("/fixture/voice-state",{method:"POST",body:JSON.stringify({unavailable:true,model_missing:true})})');
      await click('.name-speaker');await wait('document.querySelector("#name-dialog").open && !document.querySelector("#voice-settings").hidden && !document.querySelector("#voice-settings").disabled');
      assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
      assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/voice model download/);
      assert.equal(await evaluate('document.querySelector("#voice-settings").textContent'),'Set up voice model');
      await screenshot('voice-model-missing.png');
      await click('#voice-settings');assert.equal(await evaluate('document.querySelector("#panel-models").hidden'),false);
      await click('#settings-close');
    }
    await evaluate('api("/fixture/voice-state",{method:"POST",body:JSON.stringify({clip_error:true})})');
    await click('.name-speaker');
    await wait('document.querySelector("#voice-availability").textContent.includes("Synthetic passage lookup failed")');
    assert.equal(await evaluate('document.querySelector("#remember-voice").disabled'),true);
    assert.match(await evaluate('document.querySelector("#voice-note").textContent'),/Refresh passages/);
    await screenshot('voice-lookup-error.png');
    await evaluate('api("/fixture/voice-state",{method:"POST",body:JSON.stringify({embedding_error:true})})');
    await click('#refresh-voice-clips');await wait('!document.querySelector("#remember-voice").disabled');
    await click('#remember-voice');
    await wait('document.querySelector("#name-message").textContent.includes("synthetic passage could not be used")');
    assert.equal(await evaluate('document.querySelector("#name-dialog").open'),true);
    await wait('!document.querySelector("#remember-voice").disabled');
    report=await evaluate('api("/fixture/report")');assert.equal(report.profiles[0].version,version);
    await screenshot('voice-enrollment-error.png');
    await evaluate('api("/fixture/voice-state",{method:"POST",body:JSON.stringify({})})');
    await evaluate('document.querySelector("#remember-voice").focus()');await key('Enter');
    await wait('!document.querySelector("#name-dialog").open');
    report=await evaluate('api("/fixture/report")');
    assert.equal(report.embedding_calls.length,4,'Keyboard activation sends one further enrollment, not duplicate events.');
    assert.notEqual(report.profiles[0].version,version);
    const job=await evaluate(`api('/api/jobs/${'b'.repeat(32)}')`);
    assert.equal(job.document.speakers.speaker_0,'Albert');
    assert.equal(job.document.segments[0].text,'Synthetic passage for the voice setup preview.');
    await writeFile(join(output,'browser-proof.json'),JSON.stringify({mouse_and_keyboard_enrollment:true,modal_escape_and_focus:true,
      name_selection_and_revision_guards:true,clip_selection_and_preview:true,runtime_and_lookup_blocks:true,truthful_readiness_contract:mode==='after-contract',
      backend_error_preserves_existing_profile:true,explicit_consent:true,synthetic_embedding_calls:report.embedding_calls.length,
      trained_model_calls:0,private_data_access:false},null,2)+'\n');
    console.log('Passed single-action consent, mouse/keyboard enrollment, modal Escape/Tab, naming/revision/clip guards, runtime/settings and lookup/error states, existing-profile preservation; synthetic fixtures only.');
  }

} finally {
  ws?.close();browser.kill('SIGTERM');
  await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
  await rm(profile,{recursive:true,force:true});
}
