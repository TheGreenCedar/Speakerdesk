// Disposable browser proof using shipped assets and synthetic HTTP snapshots.
// No installed app, real transcripts, capture, models, or dependency downloads.
// node tests/live_partition_browser.cjs <installed Chromium executable> <evidence directory>
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {createServer}=require('node:http');
const {createHash}=require('node:crypto');
const {mkdtemp,readFile,rm,mkdir,writeFile}=require('node:fs/promises');
const {tmpdir}=require('node:os');
const {join,resolve}=require('node:path');
const {setTimeout:delay}=require('node:timers/promises');
const [chromium,output,mode]=process.argv.slice(2),root=resolve(__dirname,'..'),jid='d'.repeat(32);
async function main(){
  assert(chromium && output,'Pass installed Chromium executable and evidence directory');
  await mkdir(output,{recursive:true});
  let job={id:jid,name:'Synthetic partition fixture',kind:'meeting',created:1,status:'recording',language:'en',duration:60,revision:1,refinement_status:'refining',message:'Synthetic UI fixture',
    document:{schema_version:1,speakers:{s0:'Speaker 1'},segments:Array.from({length:6},(_,i)=>({id:`r${i}`,start:i*10,end:(i+1)*10,speaker:'s0',text:`Synthetic words from epoch ${Math.floor(i/2)}, passage ${i}.`,language_epoch:Math.floor(i/2),machine_revision:1,refinement_state:i%2?'provisional':'refined'}))}};
  job.document.segments.push({...job.document.segments[3],text:'Older duplicate',machine_revision:0});
  job.document.segments.reverse();
  let recoveryReceipt={schema_version:1,snapshot:{schema_version:1,job_id:jid,name:'Recovery fixture',revision:1,has_unsaved:true,document:{segments:[{start:0,speaker:'Speaker 1',text:'Exact recovered human words.'}]},passage_drafts:[],recoverable_drafts:[],timing_drafts:[]}};
  const recoveryId='a'.repeat(32)+'-'+jid;
  const mutations=[];
  // Silent audio keeps the real player usable without touching a recording.
  const pcm=Buffer.alloc(44+60*16000*2);pcm.write('RIFF',0);pcm.writeUInt32LE(pcm.length-8,4);pcm.write('WAVEfmt ',8);pcm.writeUInt32LE(16,16);pcm.writeUInt16LE(1,20);pcm.writeUInt16LE(1,22);pcm.writeUInt32LE(16000,24);pcm.writeUInt32LE(32000,28);pcm.writeUInt16LE(2,32);pcm.writeUInt16LE(16,34);pcm.write('data',36);pcm.writeUInt32LE(pcm.length-44,40);
  let holdNext=false,held=null;
  const server=createServer(async(req,res)=>{
    try{
      const url=new URL(req.url,'http://fixture'),send=value=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));};
      if(url.pathname==='/'){res.setHeader('Content-Type','text/html');res.end((await readFile(join(root,'speakerdesk/templates/index.html'),'utf8')).replace('{{token}}','synthetic-token').replace('<script src="/static/theme.js">','<script>window.setInterval=()=>0;</script><script src="/static/theme.js">'));return;}
      if(/^\/static\/[a-z_.]+$/.test(url.pathname)){
        res.setHeader('Content-Type',url.pathname.endsWith('.css')?'text/css':url.pathname.endsWith('.svg')?'image/svg+xml':'text/javascript');res.end(await readFile(join(root,'speakerdesk',url.pathname)));return;
      }
      if(url.pathname==='/fixture/update'){
        let body='';for await(const part of req)body+=part;job=JSON.parse(body);send({ok:true});return;
      }
      if(url.pathname==='/fixture/hold'){holdNext=true;send({ok:true});return;}
      if(url.pathname==='/fixture/state'){send({held:!!held});return;}
      if(url.pathname==='/fixture/release'){if(held){const pending=held;held=null;pending.res.setHeader('Content-Type','application/json');pending.res.end(JSON.stringify(pending.snapshot));}send({ok:true});return;}
      if(url.pathname==='/api/config'){send({languages:{auto:'Automatic',en:'English',fr:'French'},default_language:'en',readiness:{configured:true,automatic_language:true}});return;}
      if(url.pathname==='/api/jobs'){send([{id:jid,name:job.name,created:1,status:job.status}]);return;}
      if(url.pathname===`/api/jobs/${jid}`){
        const snapshot=structuredClone(job);
        if(holdNext){holdNext=false;held={res,snapshot};return;}
        if(Number(url.searchParams.get('known_revision'))===job.revision){delete snapshot.document;snapshot.unchanged=true;}
        send(snapshot);return;
      }
      if(url.pathname==='/api/meeting'){send({status:'idle',duration:0,levels:{microphone:0,system:0},sources:[]});return;}
      if(url.pathname==='/api/people'){send({people:[],voice_available:false});return;}
      if(url.pathname==='/api/recovery'){send({receipts:recoveryReceipt?[{id:recoveryId,name:'Recovery fixture',job_id:jid,revision:1}]:[]});return;}
      if(url.pathname===`/api/recovery/${recoveryId}`){if(req.method==='DELETE'){recoveryReceipt=null;send({discarded:true});}else send(recoveryReceipt);return;}
      if(url.pathname===`/api/jobs/${jid}/cancel`){job.status='uploaded';job.inference_owned=true;job.revision++;send(job);return;}
      if(url.pathname===`/api/jobs/${jid}/run`){job.status='processing';job.inference_owned=true;job.revision++;send(job);return;}
      if(url.pathname===`/api/jobs/${jid}/segments/r0` && req.method==='PATCH'){
        let body='';for await(const part of req)body+=part;const request=JSON.parse(body),row=job.document.segments.find(row=>row.id==='r0');mutations.push(request);
        if(request.segment_revision!==row.machine_revision){res.statusCode=409;send({error:'This passage changed'});return;}
        Object.assign(row,request.changes);row.machine_revision++;row.refinement_state='edited';job.revision++;send({segment:row,revision:job.revision});return;
      }
      if(url.pathname.endsWith('/identity-suggestions')){send({suggestions:[]});return;}
      if(url.pathname.endsWith('/audio')){res.setHeader('Content-Type','audio/wav');res.end(pcm);return;}
      if(url.pathname==='/api/setup'){send({models:[],status:'ready',ready:true,core_ready:true,supported:true,total_bytes:0,downloaded_bytes:0,voice:{available:false,enabled:false,status:'ready',message:'Synthetic fixture'}});return;}
      res.statusCode=404;send({error:'Synthetic fixture route unavailable'});
    }catch(error){res.statusCode=500;res.end(JSON.stringify({error:error.message}));}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`,profile=await mkdtemp(join(tmpdir(),'speakerdesk-partition-'));
  const browser=spawn(chromium,['--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});
  let ws;
  try{
    let port;for(let i=0;i<100;i++){try{port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await delay(100);}}
    assert(port,'Chromium did not start');
    const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
    let id=0;const pending=new Map(),errors=[];
    ws.onmessage=event=>{const message=JSON.parse(event.data);if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails.text);if(pending.has(message.id)){const [resolve,reject]=pending.get(message.id);pending.delete(message.id);message.error?reject(new Error(message.error.message)):resolve(message.result);}};
    const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,[resolve,reject]);ws.send(JSON.stringify({id:key,method,params}));});
    const evaluate=async expression=>{const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
    const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(100);}throw new Error(`Timed out: ${expression}`);};
    const screenshot=async name=>{await evaluate('document.fonts.ready.then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))))');const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(join(output,name),Buffer.from(shot.data,'base64'));};
    await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:`${base}/?meeting=${jid}`});await wait('typeof selected!=="undefined" && selected?.id && !polling && !meetingPoll');
    const checks=[],measurements=[],source={sha256:{}};
    for(const file of ['speakerdesk/static/app.js','speakerdesk/static/live_editor.js','speakerdesk/static/style.css','speakerdesk/static/recovery.js','speakerdesk/static/theme.js','speakerdesk/templates/index.html'])source.sha256[file]=createHash('sha256').update(await readFile(join(root,file))).digest('hex');
    if(mode==='--overlap-setup') {
      await evaluate(`(async()=>{setupState={models:[],ready:true,core_ready:true,automatic_language:true,status:'failed',supported:true,total_bytes:100,downloaded_bytes:0,voice:{available:false,enabled:false,status:'ready'},operation:{target:'separation',status:'failed',error:'Connection interrupted. Retry this download.'},separation:{installed:true,ready:false,active:false,bytes:670353271,can_download:true}};
        setupState.models=${JSON.stringify(JSON.parse(await readFile(join(root,'crates/host/src/model-artifacts.json'),'utf8')).map(model=>({id:model.id,name:model.name,bytes:Object.values(model.files).reduce((sum,file)=>sum+file.size,0),installed:true})))};setupState.models.push({id:'mossformer2-16k',name:'Voice separation',bytes:670353271,installed:true,optional:true});window.browserSetup=structuredClone(setupState);window.retryRoutes=[];
        window.originalApi=api;api=async(path,options)=>{if(path==='/api/setup'){return structuredClone(browserSetup);}if(path==='/api/setup/separation'){retryRoutes.push(path);return {};}return originalApi(path,options);};await refreshSetup();openSettings('models');})()`);
      assert(!String(await evaluate("$('model-status').textContent")).startsWith('Could not'),await evaluate("$('model-error').textContent"));
      for(const width of [1280,760]) {
        await send('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
        const placement=await evaluate(`(()=>{const card=$('model-list').firstElementChild,b=card.querySelector('button'),e=card.querySelector('.model-failure'),note=card.querySelector('.separation-model-note'),view=document.querySelector('.settings-card');return {button:b.textContent,error:e.textContent,retryVisible:b.getBoundingClientRect().bottom<=view.getBoundingClientRect().bottom,errorBeforeRetry:e.getBoundingClientRect().top<=b.getBoundingClientRect().top,retryBeforeExplanation:b.getBoundingClientRect().bottom<=note.getBoundingClientRect().top,genericHidden:$('install-models').hidden,overflow:document.body.scrollWidth>innerWidth};})()`);
        assert.equal(placement.button,'Retry separation download');assert.equal(placement.retryVisible,true);assert.equal(placement.retryBeforeExplanation,true);assert.equal(placement.genericHidden,true);assert.equal(placement.overflow,false);
        measurements.push({width,...placement});await screenshot(`overlap-setup-${width}.png`);
      }
      await evaluate("$('model-list').querySelector('button').click()");await wait('retryRoutes.length===1');assert.deepEqual(await evaluate('retryRoutes'),['/api/setup/separation']);checks.push('Visible scoped separation retry above explanation at desktop and narrow widths');
      await evaluate(`closeSettings();jobFixture=null;doc.speakers={multiple_speakers:'Multiple speakers',speaker_0:'Voice One',speaker_1:'Voice Two'};let offset=0;const words=['First original sentence. ','Second original sentence. ','Third original sentence.'];const row={id:'r0',start:0,end:10,text:words.join(''),speaker:'multiple_speakers',language:'en',refinement_state:'refined',machine_revision:1,reading_turns:words.map((text,i)=>{const start_offset=offset;offset+=text.length;return {text,start_offset,end_offset:offset,start:i,end:i+1,speaker:'speaker_'+(i%2),attribution:'single'};})};row.separation_candidate={id:'estimate',original:structuredClone(row),voices:[{text:'First voice estimate.',language:'en'},{text:'Second voice estimate.',language:'en'}]};doc.segments=[row];selected.status='recording';selected.document=structuredClone(doc);renderEditor();`);
      for(const width of [1280,760]) {
        await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
        const state=await evaluate(`(()=>{const c=$('segments').firstElementChild,result=c.querySelector('.separation-comparison');return {comparisonOpen:result.open,turns:c.querySelectorAll('.reading-turn').length,labels:Array.from(c.querySelectorAll('.reading-turn-speaker')).map(e=>e.textContent),play:Array.from(result.querySelectorAll('button')).map(b=>b.textContent),acceptDisabled:Array.from(result.querySelectorAll('button')).find(b=>b.textContent==='Use both voices').disabled,diagnostics:Array.from(c.querySelectorAll('.passage-evidence')).map(d=>({open:d.open,label:d.querySelector('summary').textContent})),overflow:document.body.scrollWidth>innerWidth};})()`);
        assert.equal(state.comparisonOpen,false);assert.equal(state.turns,3);assert(state.play.includes('Play Voice A') && state.play.includes('Play Voice B') && state.play.includes('Play Original recording'));assert.equal(state.acceptDisabled,true);assert(state.diagnostics.every(d=>!d.open && d.label==='Technical diagnostics'));assert.equal(state.overflow,false);measurements.push({width,...state});await screenshot(`overlap-candidate-${width}.png`);await evaluate("$('segments').querySelector('.separation-comparison').open=true");await screenshot(`overlap-comparison-${width}.png`);await evaluate("$('segments').querySelector('.separation-comparison').open=false");
      }
      checks.push('Original speaker paragraphs retained beside live candidate playback; acceptance disabled while recording; diagnostics collapsed');
      await send('Runtime.evaluate',{expression:"$('segments').querySelector('.separation-comparison').open=true;window.voicePlayback=$('segments').querySelectorAll('.separation-playback')[1];voicePlayback.querySelector('button').click()",userGesture:true});
      await wait("voicePlayback.querySelector('button').textContent==='Pause Voice A' && !voicePlayback.querySelector('audio').paused");
      await evaluate("voicePlayback.querySelector('button').click()");
      await wait("voicePlayback.querySelector('button').textContent==='Play Voice A' && voicePlayback.querySelector('audio').paused");
      checks.push('Generated voice playback can be started and paused from its visible button');
      assert.deepEqual(errors,[]);await writeFile(join(output,'overlap-browser-report.json'),JSON.stringify({fixture:'Synthetic setup and transcript in disposable headless Chrome',modelsOrCaptureExecuted:false,source,checks,measurements,pageErrors:errors},null,2));return;
    }
    if(mode==='--unknown-speakers') {
      const snapshots=require('./support/unknown_speaker_jobs.json'),partition=structuredClone(job);
      for(const status of ['ready','recording'])for(const [name,snapshot] of Object.entries(snapshots)) {
        job=structuredClone(snapshot);job.id=jid;job.status=status;job.refinement_status='complete';
        job.document.segments.forEach(s=>s.refinement_state='refined');
        await evaluate(`select('${jid}')`);
        const qualified=name.startsWith('qualified');
        const state=await evaluate(`(()=>{const c=$('segments').firstElementChild;return {count:$('segments').children.length,
          words:c.querySelector('textarea').value,labels:Array.from(c.querySelectorAll('.reading-turn-speaker')).map(e=>e.textContent),
          unassigned:c.querySelector('.name-speaker')?.textContent || c.querySelector('select').selectedOptions[0].textContent,
          names:Array.from($('speaker-list').querySelectorAll('button')).map(e=>e.textContent),
          invented:!!doc.speakers.speaker_7 || !!doc.speakers.speaker_8}})()`);
        assert.equal(state.count,1);assert.equal(state.words,'Alpha. Still alpha. Beta. Still beta.');assert.equal(state.invented,false);
        assert(!state.names.includes('Speaker unassigned'));
        if(qualified)assert.deepEqual(state.labels,['Speaker 1','Speaker 2']);
        else {assert.deepEqual(state.labels,[]);assert.equal(state.unassigned,'Speaker unassigned');}
        if(name.endsWith('candidate')) {
          await evaluate("document.querySelector('.passage-details-toggle').click()");
          const receipt=await evaluate(`(()=>{const evidence=document.querySelector('.passage-evidence');return {open:evidence.open,mapping:JSON.parse(evidence.querySelector('pre').textContent).speaker_track_mapping};})()`);
          assert.equal(receipt.open,false);assert.deepEqual(receipt.mapping.mapping,qualified?{speaker_7:'speaker_0',speaker_8:'speaker_1'}:{});
          assert.deepEqual(receipt.mapping.raw_batch_turns.map(t=>t.speaker),['speaker_7','speaker_8']);
          await evaluate(status==='ready'?"$('inspector').hidden=true":"document.querySelector('.passage-details-toggle').click()");
        }
        if(name==='one-live-track-two-batch-slots.candidate')for(const width of [1280,760]) {
          await send('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
          await evaluate(`document.documentElement.dataset.theme='${width===1280?'light':'dark'}';$('transcript-pane').scrollTop=0`);
          await screenshot(`unknown-${status}-${width}.png`);
        }
      }
      checks.push('Eight actual synthetic saved-API snapshots in saved/live views: qualified tracks keep two owners; missing timing and unmatched local slots keep unassigned words, with no invented global IDs or generic naming action.');
      checks.push('Mapped/unmatched batch receipts preserve both local IDs inside collapsed existing Inference evidence; four saved/live wide/light and narrow/dark screenshots.');
      job=partition;await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
      await evaluate(`document.documentElement.dataset.theme='light';select('${jid}')`);
    }
    const ids=()=>evaluate("Array.from($('segments').children).map(c=>c.dataset.segmentId)");
    assert.deepEqual(await ids(),['r0','r1','r2','r3','r4','r5']);
    assert.equal(await evaluate("document.querySelectorAll('.live-section-label').length"),0);
    assert.equal(await evaluate("document.querySelectorAll('.processed-section-label').length"),0);
    assert.equal(await evaluate("document.querySelector('[data-segment-id=r3] textarea').value"),'Synthetic words from epoch 1, passage 3.');
    checks.push('Shipped renderer keeps three epochs chronological across mixed live/processed states; duplicate ID retains newest revision.');
    for(const width of [1280,760])for(const theme of ['light','dark']){
      await send('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
      await evaluate(`document.documentElement.dataset.theme='${theme}';$('transcript-pane').scrollTop=0`);await delay(100);
      const m=await evaluate(`(()=>{const processed=document.querySelector('[data-segment-id=r0]'),live=document.querySelector('[data-segment-id=r1]'),text=live.querySelector('textarea'),timing=live.querySelector('.rolling-time');const cs=getComputedStyle(live),r=text.getBoundingClientRect(),t=timing.getBoundingClientRect();return {background:getComputedStyle(document.querySelector('main')).backgroundColor,rowBackground:cs.backgroundColor,processedBackground:getComputedStyle(processed).backgroundColor,ink:getComputedStyle(text).color,font:getComputedStyle(text).fontSize,overlap:t.right>r.left,overflow:document.body.scrollWidth>innerWidth,liveBelowProcessed:live.getBoundingClientRect().top>=processed.getBoundingClientRect().bottom}})()`);
      assert.equal(m.font,'16px');assert.equal(m.overlap,false);assert.equal(m.overflow,false);assert.equal(m.liveBelowProcessed,true);assert.equal(m.rowBackground,m.processedBackground);
      const luminance=value=>{const c=value.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>{x/=255;return x<=.04045?x/12.92:((x+.055)/1.055)**2.4});return .2126*c[0]+.7152*c[1]+.0722*c[2];};
      const contrast=(a,b)=>{a=luminance(a);b=luminance(b);return(Math.max(a,b)+.05)/(Math.min(a,b)+.05);};
      m.textContrast=contrast(m.ink,m.background);assert(m.textContrast>=4.5);
      measurements.push({width,theme,...m});await screenshot(`partition-${width}-${theme}.png`);
    }
    checks.push('Wide/narrow light/dark screenshots show one chronological reading column,16px words, AA contrast, no timing overlap or body overflow.');
    await evaluate(`(()=>{window.focusedText=document.querySelector('[data-segment-id=r1] textarea');focusedText.focus();focusedText.value='Exact human correction';focusedText.dispatchEvent(new Event('input',{bubbles:true}));focusedText.setSelectionRange(2,8,'backward');window.beforeTop=focusedText.getBoundingClientRect().top;})()`);
    job.document.segments=job.document.segments.filter(s=>s.machine_revision>0);const row=job.document.segments.find(s=>s.id==='r1');row.refinement_state='refined';row.machine_revision=2;row.text='New processed machine words';job.revision++;
    await evaluate('poll()');
    assert.deepEqual(await evaluate(`({same:document.activeElement===focusedText,value:focusedText.value,caret:[focusedText.selectionStart,focusedText.selectionEnd,focusedText.selectionDirection],section:focusedText.closest('article').dataset.transcriptSection,latest:focusedText.closest('article').querySelector('.latest-machine-words').textContent})`),{same:true,value:'Exact human correction',caret:[2,8,'backward'],section:'transcript',latest:'New processed machine words'});
    assert.equal(await evaluate(`Math.abs(focusedText.getBoundingClientRect().top-beforeTop)<1 || ($('transcript-pane').scrollTop===0 && focusedText.getBoundingClientRect().top >= $('transcript-pane').getBoundingClientRect().top)`),true,'Retain editor anchor unless the scroll range clamps at the top');
    checks.push('Refinement during polling keeps chronological position, identical focused textarea, exact draft, backward caret and displayed new machine revision; viewport remains anchored within available scroll range.');
    job.document.segments=job.document.segments.filter(s=>s.id!=='r1');job.revision++;await evaluate('poll()');
    assert.equal(await evaluate("focusedText.closest('article').dataset.transcriptSection"),'corrections');assert.equal(await evaluate('document.activeElement===focusedText'),true);
    assert.equal(await evaluate("document.querySelectorAll('.corrections-section-label').length"),1);
    checks.push('Replacement retains the human correction in a separate region and removes its provisional tint without duplicating live words.');
    await evaluate("focusedText.blur();passageDrafts.clear();renderSegments();followingLive=true");
    job.document.segments.find(s=>s.id==='r5').text='';job.revision++;await evaluate('poll()');
    assert.equal(await evaluate("$('pending-phrases').hidden"),false);assert.equal(await evaluate("$('transcript-pane').scrollTop+$('transcript-pane').clientHeight >= $('transcript-pane').scrollHeight-1"),true);
    job.status='ready';job.refinement_status='paused';job.document.segments.find(s=>s.id==='r5').text='Latest pending tail after Stop';job.revision++;await evaluate('poll()');
    assert.equal(await evaluate("document.querySelector('[data-segment-id=r5]').classList.contains('live-provisional')"),false);assert.equal(await evaluate("$('pending-phrases').hidden"),true);
    await screenshot('partition-final-stop.png');
    checks.push('Live follow reaches latest pending tail; final Stop keeps unfinished words readable without a manual review region.');
    await evaluate("(async()=>{await api('/fixture/hold',{method:'POST'});window.oldPoll=poll();})()");await wait("api('/fixture/state').then(s=>s.held)");
    job.refinement_status='complete';job.document.segments.forEach(s=>s.refinement_state='refined');job.document.segments.sort((a,b)=>a.start-b.start || a.end-b.end || a.id.localeCompare(b.id));job.revision++;
    await evaluate("(async()=>{const next=await api('/api/jobs/'+selected.id);selected=next;doc=structuredClone(next.document);renderSegments();await api('/fixture/release',{method:'POST'});await oldPoll;})()");
    assert.equal(await evaluate('selected.revision'),job.revision);assert.equal(await evaluate("document.querySelectorAll('.live-provisional,.live-section-label').length"),0);
    assert.equal(await evaluate("document.querySelector('[data-segment-id=r5] textarea').value"),'Latest pending tail after Stop');
    checks.push('Held older HTTP poll cannot undo a newer fully processed document; completion preserves the same reading column.');
    const canonicalBeforeRecovery=await evaluate('JSON.stringify(doc)');
    await evaluate("loadRecoveryCopies();$('recovery-copies').open=true");
    await wait("$('recovery-list').querySelector('button')");
    await evaluate("$('recovery-list').querySelector('button').click()");
    await wait("$('recovery-text').value.includes('Exact recovered human words.')");
    assert.equal(await evaluate('JSON.stringify(doc)'),canonicalBeforeRecovery);
    assert.equal(await evaluate("$('recovery-text').readOnly && document.activeElement===$('recovery-text')"),true);
    await screenshot('recovery-review.png');
    await evaluate("$('recovery-copies').open=false;$('recovery-text').blur()");
    checks.push('Persisted recovery words are selectable/read-only and review leaves canonical transcript unchanged.');
    job.kind='import';job.status='processing';job.import_resumable=true;job.inference_owned=true;job.import_progress={completed:2,total:6};job.refinement_status='complete';job.revision++;
    await evaluate('poll()');
    assert.equal(await evaluate("$('editor').inert"),false);assert.equal(await evaluate("$('add').hidden && $('import').disabled"),true);
    await evaluate("(()=>{const row=$('segments').querySelector('[data-segment-id=r0]'),text=row.querySelector('textarea');text.focus();text.value='Protected partial import correction';text.dispatchEvent(new Event('input',{bubbles:true}));row.querySelector('.save-passage').click();})()");
    await wait("!hasPassageDrafts() && !hasPendingPassageSaves() && doc.segments.find(row=>row.id==='r0').text==='Protected partial import correction'");
    assert.equal(mutations.at(-1).changes.text,'Protected partial import correction');
    await evaluate("$('cancel-job').click()");await wait("selected.status==='uploaded' && !polling");
    assert.equal(await evaluate("$('cancel-job').disabled && !$('cancel-job').hidden && $('run').disabled"),true);
    job.inference_owned=false;job.revision++;await evaluate('poll()');
    assert.equal(await evaluate("!$('run').hidden && !$('run').disabled && $('run').textContent==='Resume transcription'"),true);
    await screenshot('progressive-import-resume.png');
    await evaluate("$('run').click()");await wait("selected.status==='processing' && startingJob===null");
    assert.equal(await evaluate("doc.segments.find(row=>row.id==='r0').text"),'Protected partial import correction');
    job.status='ready';job.inference_owned=false;job.import_resumable=false;job.import_progress={completed:6,total:6};job.document.segments.sort((a,b)=>a.start-b.start || a.end-b.end || a.id.localeCompare(b.id));job.revision++;await evaluate('poll()');
    assert.equal(await evaluate("$('run').hidden"),true);
    assert.equal(await evaluate("$('recording-dot').hasAttribute('hidden')"),true,'Lucide SVG recording dot must disappear after capture');
    assert.deepEqual(await evaluate("Array.from($('segments').children).map(row=>Number(row.querySelector('.seek').title.match(/(\\d+):(\\d+)/)[2]))"),[0,20,30,40,50]);
    checks.push('Progressive processing uses passage revision PATCH, retains corrected prefix through Cancel/ownership drain/Resume, then returns to saved controls.');
    await evaluate("(()=>{const text=$('segments').querySelector('[data-segment-id=r0] textarea');text.value='Unsaved words before host failure';text.dispatchEvent(new Event('input',{bubbles:true}));window.speakerdeskHostUnavailable('Synthetic host failure');window.speakerdeskRecoverySnapshot(true);})()");
    assert.equal(await evaluate("document.querySelector('main').inert && $('pause-meeting').disabled && $('stop-meeting').disabled"),true);
    await evaluate("window.speakerdeskRecoveryFailed('Synthetic recovery write failure',true);$('recovery-copy-open').click()");
    assert.equal(await evaluate("!document.querySelector('main').inert && !recoveryClosing"),true);
    assert.equal(await evaluate("$('recovery-text').value.includes('Unsaved words before host failure')"),true);
    await screenshot('recovery-disconnected.png');
    checks.push('Native failure callback removes stale capture controls; exact-close snapshot freezes edits and failed persistence restores the same window for copying.');
    assert.deepEqual(errors,[]);await writeFile(join(output,'partition-browser-report.json'),JSON.stringify({fixture:'Disposable synthetic HTTP server + shipped UI + installed headless Chrome',modelsOrCaptureExecuted:false,source,checks,measurements,pageErrors:errors},null,2));
    console.log(JSON.stringify({checks,measurements,pageErrors:errors},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    held?.res.destroy();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
