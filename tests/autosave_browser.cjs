// Autosave/undo/review regression against the synthetic People fixture. Uses installed Chromium;
// starts no devices or model workers. Usage: node tests/autosave_browser.cjs <people fixture URL> <chromium> <output>
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
  const profile=await mkdtemp(join(tmpdir(),'speakerdesk-autosave-'));
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
    await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
    const A='a'.repeat(32),B='b'.repeat(32),checks=[];
    await send('Page.navigate',{url:`${base}/?meeting=${A}`});
    await wait(`typeof selected!=="undefined" && selected?.id==="${A}" && doc`);
    await evaluate(`(()=>{window.putDelay=0;const original=window.fetch;window.puts=0;
      window.fetch=async(url,options={})=>{if(options.method==='PUT'){puts++;await new Promise(r=>setTimeout(r,putDelay));}return original(url,options);};
      window.edit=text=>{doc.segments[0].text=text;changed();};
      window.saved=async jid=>(await api('/api/jobs/'+jid)).document;})()`);

    await evaluate(`(async()=>{putDelay=400;edit('first');const inFlight=save({quiet:true});
      await new Promise(r=>setTimeout(r,100));edit('second');
      const switching=select("${B}");await inFlight;
      await new Promise(r=>setTimeout(r,100));edit('third');await switching;})()`);
    await wait(`selected?.id==="${B}"`);
    assert.equal((await evaluate(`saved("${A}")`)).segments[0].text,'third');
    checks.push('An edit typed while switching meetings during an autosave is saved, not dropped.');

    await evaluate(`(async()=>{putDelay=0;await select("${A}");})()`);
    await wait(`selected?.id==="${A}" && doc`);
    const revision=await evaluate('selected.revision');
    await evaluate(`removePassage(doc.segments[1])`);
    await delay(2000);
    assert.equal(await evaluate(`(async()=>(await api('/api/jobs/${A}')).revision)()`),revision,'Removal must stay local while Undo is offered.');
    await evaluate(`document.querySelector('#notice .notice-action').click()`);
    await evaluate('flushSave()');
    const restored=(await evaluate(`saved("${A}")`)).segments[1];
    assert.equal(restored.id,'clip-1');assert.equal(restored.voice_eligible,true);assert.deepEqual(restored.speaker_candidates,['speaker_0']);
    await evaluate(`removePassage(doc.segments[1])`);await evaluate('flushSave()');
    await evaluate(`document.querySelector('#notice .notice-action')?.click()`);
    assert.equal((await evaluate(`saved("${A}")`)).segments.length,1,'Undo after a saved removal must not resurrect a stripped passage.');
    checks.push('Undo restores a removed passage with its server-owned metadata; once saved, Undo is no longer offered.');

    await evaluate(`(async()=>{doc.segments[0].transcription_review={reason:'token_limit'};doc.segments[0].review=true;changed();await flushSave();renderSegments();})()`);
    const txt=()=>evaluate(`(async()=>(await fetch('/api/jobs/${A}/export/txt')).text())()`);
    assert.match(await txt(),/Needs review/);
    await evaluate(`document.querySelector('#segments .passage-details-toggle').click();document.querySelector('#segment-details .review-action').click()`);
    await evaluate('flushSave()');
    assert.doesNotMatch(await txt(),/Needs review/);
    checks.push('Mark words reviewed in the explicit inspector clears the actual export warning.');

    await evaluate(`(()=>{window.confirm=()=>false;edit('kept after cancel');document.querySelector('#delete').click();})()`);
    await delay(2500);
    assert.equal((await evaluate(`saved("${A}")`)).segments[0].text,'kept after cancel');
    checks.push('Cancelling Delete keeps autosave scheduled.');

    assert.deepEqual(errors,[]);
    await writeFile(join(output,'autosave-browser-report.json'),JSON.stringify({fixture:'Synthetic CPU fixture; no devices/models',checks,pageErrors:errors},null,2));
    console.log(JSON.stringify({checks,pageErrors:errors},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
