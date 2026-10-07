// Real-browser icon retention regression and trusted keyboard input profile.
// Pass the disposable synthetic fixture URL, an existing Chrome-for-Testing
// executable, and evidence directory. This never starts capture or AI models.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir,rm} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';

const [base,chromium,outArg,mode='candidate']=process.argv.slice(2);
assert(base && chromium && outArg);
const out=resolve(outArg);await mkdir(out,{recursive:true});
const temporary=await mkdtemp(join(out,'chrome-profile-'));
const browser=spawn(chromium,['--headless=new','--no-first-run','--no-default-browser-check',
  '--disable-background-networking','--disable-component-update','--disable-sync',
  '--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',`--user-data-dir=${temporary}`,'about:blank'],{stdio:'ignore'});
let ws;
try{
  let port;
  for(let i=0;i<100;i++){try{port=(await readFile(join(temporary,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await delay(100);}}
  assert(port,'Dedicated Chrome for Testing did not start');
  const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((ok,no)=>{ws.onopen=ok;ws.onerror=no;});
  let id=0;const pending=new Map();
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const [ok,no]=pending.get(m.id);pending.delete(m.id);m.error?no(new Error(m.error.message)):ok(m.result);}};
  const send=(method,params={})=>new Promise((ok,no)=>{const n=++id;pending.set(n,[ok,no]);ws.send(JSON.stringify({id:n,method,params}));});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await send('Page.enable');
  const version=await send('Browser.getVersion');
  await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
  await send('Page.addScriptToEvaluateOnNewDocument',{source:'window.setInterval=()=>1;'});
  await send('Page.navigate',{url:base});
  let initialized=false;
  for(let i=0;i<100;i++){if(await evaluate('typeof config!=="undefined" && config?.languages && setupState!==null')){initialized=true;break;}await delay(50);}
  assert(initialized);
  await evaluate(`(async()=>{await api('/fixture/reset',{method:'POST',body:JSON.stringify({hours:3})});await select('${'b'.repeat(32)}');})()`);
  await evaluate(`window.__inputSamples=[];window.__eventEntries=[];window.__longTasks=[];window.__inputBindings=new WeakMap();
    window.__inputPhase='';
    document.addEventListener('input',e=>{
      if(!e.isTrusted || (e.target.id!=='search' && e.target.tagName!=='TEXTAREA'))return;
      const start=performance.now(),sample={phase:__inputPhase,target:e.target.id||'textarea',trusted:e.isTrusted,
        valueLength:e.target.value.length,dispatchToCaptureMs:start-e.timeStamp};__inputSamples.push(sample);
      __inputBindings.set(e,{sample,start});requestAnimationFrame(()=>{sample.nextAnimationFrameMs=performance.now()-start;});
    },true);
    document.addEventListener('input',e=>{const binding=__inputBindings.get(e);if(binding)binding.sample.handlerMs=performance.now()-binding.start;});
    window.__eventObserver=new PerformanceObserver(list=>__eventEntries.push(...list.getEntries().map(e=>({name:e.name,start:e.startTime,
      durationMs:e.duration,processingMs:e.processingEnd-e.processingStart,inputDelayMs:e.processingStart-e.startTime,interactionId:e.interactionId}))));
    __eventObserver.observe({type:'event',durationThreshold:16,buffered:false});
    window.__longObserver=new PerformanceObserver(list=>__longTasks.push(...list.getEntries().map(e=>({start:e.startTime,durationMs:e.duration}))));
    __longObserver.observe({type:'longtask',buffered:false});
    window.__frame=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));`);
  const type=async ch=>{
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:ch,code:'Key'+ch.toUpperCase(),text:ch,unmodifiedText:ch});
    await send('Input.dispatchKeyEvent',{type:'keyUp',key:ch,code:'Key'+ch.toUpperCase()});
    await evaluate('__frame()');
  };
  await evaluate(`const c=$('segments').children[1];c.querySelector('.edit-whole-passage').click();const t=c.querySelector('textarea');t.focus();t.setSelectionRange(t.value.length,t.value.length);__inputPhase='edit';`);
  for(let i=0;i<5;i++)await type('x');
  await evaluate(`clearTimeout(autosaveTimer);autosaveTimer=null;dirty=false;document.activeElement.blur();`);
  for(let i=0;i<5;i++){
    await evaluate(`__inputPhase='setup';$('search').value='';$('search').dispatchEvent(new Event('input',{bubbles:true}));$('search').focus();__inputPhase='search';`);
    for(const ch of 'milestone')await type(ch);
    await evaluate(`__inputPhase='clear-search';$('search').setSelectionRange(0,$('search').value.length);`);
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Backspace',code:'Backspace',windowsVirtualKeyCode:8,nativeVirtualKeyCode:8});
    await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Backspace',code:'Backspace',windowsVirtualKeyCode:8,nativeVirtualKeyCode:8});
    await evaluate('__frame()');
  }
  const inputs=await evaluate('({samples:__inputSamples,eventTiming:__eventEntries,longTasks:__longTasks})');
  assert.equal(inputs.samples.filter(x=>x.phase==='edit').length,5);
  assert.equal(inputs.samples.filter(x=>x.phase==='search').length,45);
  assert.equal(inputs.samples.filter(x=>x.phase==='clear-search').length,5);
  assert(inputs.samples.every(x=>x.trusted),'All profiled keyboard inputs must be browser-trusted');
  // Payload transformation costs measured separately from input and layout.
  const transforms=await evaluate(`(async()=>{const r=await fetch('/api/jobs/${'b'.repeat(32)}');const text=await r.text(),original=JSON.parse(text);const samples=[];
    for(let i=0;i<7;i++){let t=performance.now();const parsed=JSON.parse(text);const parseMs=performance.now()-t;t=performance.now();const cloned=structuredClone(parsed.document);
      const cloneMs=performance.now()-t;t=performance.now();const body=JSON.stringify({revision:parsed.revision,document:cloned});samples.push({parseMs,cloneMs,stringifyMs:performance.now()-t,
        documentBytes:new TextEncoder().encode(body).length});}return {bodyBytes:new TextEncoder().encode(text).length,samples};})()`);
  // Use an independent clean setup for retention: earlier keyboard edits must
  // not introduce a second changed row in the one-row publication probe.
  await evaluate(`(async()=>{await api('/fixture/reset',{method:'POST',body:JSON.stringify({hours:3})});await select('${'b'.repeat(32)}');})()`);
  // Protect identity of rendered SVGs, and prove placeholders still hydrate.
  const retention=await evaluate(`(()=>{window.__icons=Array.from($('segments').querySelectorAll('svg'));window.__cards=Array.from($('segments').children);
    renderSegments();return {total:__icons.length,retained:__icons.filter(e=>e.isConnected).length,
      cards:__cards.length,retainedCards:__cards.filter(e=>e.isConnected).length,
      placeholders:document.querySelectorAll('i[data-lucide]').length};})()`);
  await evaluate(`api('/fixture/advance',{method:'POST',body:JSON.stringify({hours:3})}).then(()=>poll())`);
  const update=await evaluate(`({cards:$('segments').children.length,retainedCards:__cards.filter(e=>e.isConnected).length,
    retainedIcons:__icons.filter(e=>e.isConnected).length,placeholders:document.querySelectorAll('i[data-lucide]').length,
    pendingMarkers:document.querySelectorAll('svg[data-lucide-pending]').length,
    lastRowIcons:$('segments').lastElementChild.querySelectorAll('svg').length})`);
  const checks={retention,update};
  const expectedRetention=retention.retained===retention.total && retention.total>0;
  const report={mode,version,inputs,transforms,checks,iconRetentionContractPassed:expectedRetention,
    synthetic:true,nativeWebview:false,hardwareInput:false};
  await writeFile(join(out,'input-and-icons-report.json'),JSON.stringify(report,null,2));
  assert.equal(retention.retainedCards,1080);
  assert.equal(update.retainedCards,1079);
  assert.equal(update.placeholders,0,'New row icons must still render');
  assert.equal(update.pendingMarkers,0,'Hydrated icons cannot remain pending');
  assert.equal(update.lastRowIcons,4);
  if(mode==='baseline')assert.equal(expectedRetention,false,'Negative control must reproduce icon replacement');
  else{
    assert.equal(expectedRetention,true,'Unchanged transcript render must retain existing SVG nodes');
    assert.equal(update.retainedIcons,retention.total-4,'Only the changed row may replace four icons');
  }
  console.log(JSON.stringify({mode,iconRetentionContractPassed:expectedRetention,trustedInputs:inputs.samples.length,
    eventTimingEntries:inputs.eventTiming.length,checks,output:out}));
}finally{
  ws?.close();browser.kill('SIGTERM');
  await new Promise(ok=>{
    if(browser.exitCode!==null||browser.signalCode!==null){ok();return;}
    const escalation=setTimeout(()=>browser.kill('SIGKILL'),2000);
    browser.once('exit',()=>{clearTimeout(escalation);ok();});
  });
  await rm(temporary,{recursive:true,force:true});
}
