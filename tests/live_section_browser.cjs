// Provisional/refined presentation with controlled CPU fixture outputs. Uses installed Chromium;
// starts no devices or model workers and downloads no browser dependencies.
const assert = require('node:assert/strict');
const {spawn,execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const {mkdtemp, readFile, readdir, rm, mkdir, writeFile} = require('node:fs/promises');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {setTimeout: delay} = require('node:timers/promises');
const [base, chromium, output] = process.argv.slice(2);
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
    const checks=[],measurements=[];
    const source={head:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),sha256:{}};
    for(const file of ['speakerdesk/static/app.js','speakerdesk/static/style.css'])source.sha256[file]=createHash('sha256').update(await readFile(file)).digest('hex');
    await evaluate(`(()=>{polling=true;meetingPoll=true;followingLive=false;
      doc.segments=doc.segments.filter(s=>s.text.trim());
      doc.segments.forEach((s,i)=>{s.refinement_state=i<18?'refined':'provisional'});
      window.originalDocument=JSON.stringify(doc);renderEditor();})()`);
    const state=()=>evaluate(`(()=>{const cards=[...$('segments').children];return {
      tinted:cards.filter(c=>c.classList.contains('live-provisional')).map(c=>c.dataset.segmentId),
      labels:[...document.querySelectorAll('.live-section-label')].filter(e=>e.checkVisibility()).length,
      heading:document.querySelector('.live-section-start')?.dataset.segmentId,
      untouched:JSON.stringify(doc)===originalDocument}})()`);
    assert.deepEqual(await state(),{tinted:['row-18','row-19','row-20','row-21','row-22','row-23'],labels:1,heading:'row-18',untouched:true});
    checks.push('Exact canonical provisional state tints six passages with one Live heading; refined rows remain plain and document is untouched.');
    for(const width of [1280,760])for(const theme of ['light','dark']) {
      await send('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
      await evaluate(`document.documentElement.dataset.theme='${theme}';$('transcript-pane').scrollTop=document.querySelector('[data-segment-id="row-17"]').offsetTop`);
      await delay(100);
      const m=await evaluate(`(()=>{const card=document.querySelector('[data-segment-id="row-18"]'),text=card.querySelector('textarea'),label=card.querySelector('.live-section-label'),timing=card.querySelector('.rolling-time');
        const cs=getComputedStyle(card),ts=getComputedStyle(text),ls=getComputedStyle(label),r=text.getBoundingClientRect(),t=timing.getBoundingClientRect();return {
        background:cs.backgroundColor,ink:ts.color,label:ls.color,font:ts.fontSize,timingFont:getComputedStyle(timing).fontSize,
        overlap:t.right>r.left,overflow:document.body.scrollWidth>innerWidth,headingVisible:label.checkVisibility(),height:card.getBoundingClientRect().height}})()`);
      assert.equal(m.font,'16px');assert.equal(m.timingFont,'12px');assert.equal(m.overlap,false);assert.equal(m.overflow,false);assert.equal(m.headingVisible,true);
      function luminance(value){const c=value.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>{x/=255;return x<=.04045?x/12.92:((x+.055)/1.055)**2.4});return .2126*c[0]+.7152*c[1]+.0722*c[2];}
      function contrast(a,b){a=luminance(a);b=luminance(b);return (Math.max(a,b)+.05)/(Math.min(a,b)+.05);}
      m.textContrast=contrast(m.ink,m.background);m.labelContrast=contrast(m.label,m.background);
      assert(m.textContrast>=4.5);assert(m.labelContrast>=4.5);
      measurements.push({width,theme,...m});await screenshot(`live-section-${width}-${theme}.png`);
    }
    checks.push('Actual wide/narrow light/dark renders retain16px words/12px timing, AA text/label contrast and no timing overlap or body overflow.');
    await evaluate(`(()=>{const text=document.querySelector('[data-segment-id="row-20"] textarea');
      text.focus();text.value='My exact local correction';text.dispatchEvent(new Event('input',{bubbles:true}));text.setSelectionRange(3,9);
      window.focusedText=text;window.beforeScroll=$('transcript-pane').scrollTop;
      doc.segments.find(s=>s.id==='row-20').refinement_state='refined';renderLiveSegments();})()`);
    assert.deepEqual(await evaluate(`({same:document.activeElement===focusedText,value:focusedText.value,caret:[focusedText.selectionStart,focusedText.selectionEnd],tinted:focusedText.closest('article').classList.contains('live-provisional'),labels:[...document.querySelectorAll('.live-section-label')].filter(e=>e.checkVisibility()).length,scroll:Math.abs($('transcript-pane').scrollTop-beforeScroll)})`),
      {same:true,value:'My exact local correction',caret:[3,9],tinted:false,labels:2,scroll:0});
    checks.push('A focused drafted passage becoming refined loses its tint in place without replacing its textarea, correction, caret or scroll position.');
    for(const status of ['paused','ready']) {
      await evaluate(`selected.status='${status}';selected.refinement_status='paused';setStatus();renderSegments()`);
      assert.equal((await state()).labels,2);assert.equal((await state()).tinted.length,5);
    }
    await evaluate(`doc.segments.forEach(s=>s.refinement_state='refined');renderSegments()`);
    assert.equal((await state()).labels,0);assert.equal((await state()).tinted.length,0);
    assert.equal(await evaluate('focusedText.value'),'My exact local correction');
    checks.push('Paused/stopped unfinished words stay tinted by passage state; full refinement removes every tint and Live heading, preserving the draft.');
    assert.deepEqual(errors,[]);
    await writeFile(join(output,'live-section-browser-report.json'),JSON.stringify({fixture:'Synthetic CPU state, real API/assets; no models/capture/installed app',source,checks,measurements,pageErrors:errors},null,2));
    console.log(JSON.stringify({source,checks,measurements,pageErrors:errors},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
