// Model-free browser layout/interaction regression. Uses installed Chromium;
// starts no devices or model workers and downloads no browser dependencies.
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const {mkdtemp, readFile, rm, mkdir, writeFile} = require('node:fs/promises');
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
    await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:`${base}/?meeting=${'d'.repeat(32)}`});
    await wait('typeof selected !== "undefined" && selected?.id && !meetingPoll && !polling');
    const checks=[];
    if(mode!=='before'){
      await evaluate(`(async()=>{await refreshMeeting();window.initialWords=JSON.stringify(doc.segments);
        window.originalApi=api;window.heldReady=false;window.holdNext=true;
        api=async(path,options)=>{const result=await originalApi(path,options);
          if(path==='/api/meeting' && holdNext){holdNext=false;heldReady=true;await new Promise(resolve=>window.releaseStatus=resolve);}return result;};
        window.oldStatus=refreshMeeting();})()`);
      await wait('heldReady');
      await evaluate("$('live-language').value='fr';window.languageChange=changeLiveLanguage()");
      await wait("meeting.language==='fr' && !changingLiveLanguage");
      await evaluate('(async()=>{releaseStatus();await oldStatus;await languageChange;api=originalApi})()');
      assert.equal(await evaluate("$('live-language').value"),'fr');
      assert.equal(await evaluate('JSON.stringify(doc.segments)===initialWords'),true);
      assert.equal(await evaluate("$('language').value"),'auto');
      await evaluate("(async()=>{$('live-language').value='auto';await changeLiveLanguage()})()");
      assert.equal(await evaluate("$('live-language').value"),'auto');
      checks.push('Real current-meeting PATCH changes language and Auto while a late status cannot restore the old epoch; earlier words and the persistent default remain unchanged.');
    }
    await evaluate('meetingPoll=true;polling=true;followingLive=false;passageDrafts.clear();speakerdeskTheme.set("light");notice("")');
    const hourSetup=`(()=>{
      doc.speakers={speaker_0:'Albert',speaker_1:'Maya',speaker_2:'Chris'};
      doc.segments=Array.from({length:360},(_,n)=>({id:'hour-'+n,start:n*10,end:n*10+10,
        text:'We should keep the meeting notes clear and review the next steps together. This passage remains individually editable and playable.',
        speaker:'speaker_'+(Math.floor(n/3)%3),language:'en',machine_revision:1,refinement_state:'refined'}));
      selected.name='Product planning · one hour sample';selected.duration=3600;selected.document=structuredClone(doc);
      meeting={...meeting,id:selected.id,language:'en',language_revision:0,language_acknowledged_revision:0};
      selected.refinement_status='complete';selected.rolling_refinement=false;
    })()`;
    await evaluate(hourSetup);
    const measurements={};let hourWithUnassignedGaps;
    for(const state of ['recording','ready']){
      await evaluate(`selected.status=${JSON.stringify(state)};meeting.status=${JSON.stringify(state)};$('capture-footer').hidden=${state!=='recording'};setStatus();renderEditor();$('transcript-pane').scrollTop=0`);
      await delay(200);
      measurements[state]=await evaluate(`(()=>{const pane=$('transcript-pane').getBoundingClientRect(),cards=[...document.querySelectorAll('.segment')];
        const visible=cards.filter(el=>{const r=el.getBoundingClientRect();return r.top>=pane.top && r.bottom<=pane.bottom;});
        const readable=cards.filter(el=>{const r=el.querySelector('textarea').getBoundingClientRect();return r.top>=pane.top && r.bottom<=pane.bottom;});
        return {viewport:{width:innerWidth,height:innerHeight},sampleDuration:3600,totalPassages:cards.length,fullyVisiblePassages:visible.length,
          fullyReadablePassages:readable.length,readableWords:readable.reduce((n,el)=>n+el.querySelector('textarea').value.split(/\\s+/).length,0),
          fontSize:getComputedStyle(cards[0].querySelector('textarea')).fontSize,paneHeight:pane.height,
          defaultVisiblePassageLanguageControls:[...document.querySelectorAll('.passage-retry select')].filter(el=>el.checkVisibility()).length};})()`);
      await screenshot(`${mode}-hour-${state}-1280x800.png`);
    }
    if(mode!=='before'){
      assert.equal(measurements.recording.fontSize,'16px');assert.equal(measurements.ready.fontSize,'16px');
      assert(measurements.recording.fullyReadablePassages>=8,'Live laptop density must retain the baseline eight readable passages.');
      assert(measurements.ready.fullyReadablePassages>=7,'Saved laptop density must retain the baseline seven readable passages.');
      checks.push('Matched live and saved laptop fixtures retain baseline reading density without reducing the16px transcript font.');
      assert.equal(measurements.ready.defaultVisiblePassageLanguageControls,0);
      assert.equal(await evaluate("document.querySelectorAll('.speaker-continuation').length"),240);
      assert.equal(await evaluate("new Set([...document.querySelectorAll('.segment')].map(el=>el.dataset.segmentId)).size"),360);
      checks.push('Hour transcript groups consecutive speakers visually while preserving all 360 distinct segment IDs.');
      assert.equal(await evaluate("[...document.querySelectorAll('.segment-top>select')].filter(el=>el.checkVisibility()).length"),120);
      await evaluate("document.querySelector('[data-segment-id=hour-1] textarea').focus()");
      assert.equal(await evaluate("document.querySelector('[data-segment-id=hour-1] .segment-top>select').checkVisibility()"),true);
      assert.equal(await evaluate("(()=>{const card=document.querySelector('[data-segment-id=hour-1]'),seek=card.querySelector('.seek').getBoundingClientRect(),text=card.querySelector('textarea').getBoundingClientRect();return seek.height>=28 && seek.right<=text.left && card.querySelector('select').getBoundingClientRect().height>=28})()"),true);
      await evaluate("document.activeElement.blur();$('inspector').hidden=true");
      checks.push('One speaker header per consecutive turn; focusing a continuation reveals its own28px speaker control and keeps playback beside the words.');
      await evaluate("document.querySelector('.passage-repair').open=true");
      assert.equal(await evaluate("document.querySelector('.passage-repair select').getClientRects().length>0"),true);
      checks.push('Per-passage language retry is available only after opening contextual repair.');
      await evaluate("document.querySelector('.passage-repair').open=false;selected.status='recording';meeting.status='recording';$('capture-footer').hidden=false;setStatus();renderEditor();$('transcript-pane').scrollTop=0");
      assert.equal(await evaluate("$('live-language').closest('.recording-heading')!==null"),true);
      assert.equal(await evaluate("document.querySelectorAll('#live-language').length"),1);
      assert.equal(await evaluate("$('live-language').getClientRects().length>0"),true);
      checks.push('One current-meeting language selector is visible in the meeting header.');
      await evaluate("followingLive=false;$('transcript-pane').scrollTop=1800;window.oldScroll=$('transcript-pane').scrollTop;renderLiveSegments()");
      assert(Math.abs(await evaluate("$('transcript-pane').scrollTop-oldScroll"))<2);
      await evaluate("followingLive=true;renderLiveSegments()");
      assert(Math.abs(await evaluate("$('transcript-pane').scrollHeight-$('transcript-pane').clientHeight-$('transcript-pane').scrollTop"))<2);
      checks.push('Paused auto-follow preserves reading position; resumed auto-follow reaches the newest passage.');
      await evaluate("speakerdeskTheme.set('dark');$('transcript-pane').scrollTop=0");
      await screenshot('after-hour-recording-dark-1280x800.png');
      await send('Emulation.setDeviceMetricsOverride',{width:900,height:700,deviceScaleFactor:1,mobile:false});
      await delay(200);
      for(const id of ['live-language','pause-meeting','stop-meeting'])assert.equal(await evaluate(`(()=>{const r=$(${JSON.stringify(id)}).getBoundingClientRect();return r.width>0 && r.x>=0 && r.right<=innerWidth && r.bottom<=innerHeight})()`),true,id);
      assert.equal(await evaluate("[...document.querySelectorAll('.rolling-segment textarea')].every(el=>el.scrollHeight<=el.clientHeight+1)"),true);
      await screenshot('after-hour-recording-900x700.png');
      checks.push('Language selector, Pause and Stop fit 900×700 without horizontal overflow.');
      await send('Emulation.setDeviceMetricsOverride',{width:1280,height:720,deviceScaleFactor:1,mobile:false});
      await evaluate(`(()=>{speakerdeskTheme.set('light');followingLive=false;doc.speakers={speaker_0:'Albert',speaker_1:'Maya',speaker_2:'Chris',overlap_unknown:'Unassigned audio'};
        doc.segments=Array.from({length:360},(_,n)=>[
          {id:'speech-'+n,start:n*10,end:n*10+6,speaker:'speaker_'+Math.floor(n/3)%3,language:'en',machine_revision:1,refinement_state:'refined',
            text:'We should keep the meeting notes clear and review the next steps together. This passage remains individually editable and playable.'},
          {id:'gap-'+n,start:n*10+6,end:n*10+10,speaker:'overlap_unknown',text:'',machine_revision:1,refinement_state:'unresolved',transcription_review:{reason:'unassigned_audio'}}
        ]).flat();$('retained-audio-review').open=false;renderLiveSegments();$('transcript-pane').scrollTop=0;})()`);
      await delay(100);
      hourWithUnassignedGaps=await evaluate(`(()=>{const pane=$('transcript-pane'),bounds=pane.getBoundingClientRect(),cards=[...$('segments').children];
        return {viewport:{width:innerWidth,height:innerHeight},sampleDuration:3600,internalRows:doc.segments.length,speechCards:cards.length,
          blankTranscriptCards:cards.filter(el=>!el.querySelector('textarea').value.trim()).length,retainedReviewRanges:$('retained-audio-review').querySelectorAll('.retained-audio-range').length,
          reviewCollapsed:!$('retained-audio-review').open,blankNameControls:$('retained-audio-review').querySelectorAll('.name-speaker').length,
          fullyReadablePassages:cards.filter(el=>{const r=el.querySelector('textarea').getBoundingClientRect();return r.top>=bounds.top && r.bottom<=bounds.bottom}).length,
          paneHeight:pane.clientHeight,scrollHeight:pane.scrollHeight,fontSize:getComputedStyle(cards[0].querySelector('textarea')).fontSize};})()`);
      assert.equal(hourWithUnassignedGaps.internalRows,720);assert.equal(hourWithUnassignedGaps.speechCards,360);
      assert.equal(hourWithUnassignedGaps.blankTranscriptCards,0);assert.equal(hourWithUnassignedGaps.retainedReviewRanges,360);
      assert.equal(hourWithUnassignedGaps.reviewCollapsed,true);assert.equal(hourWithUnassignedGaps.blankNameControls,0);
      assert(hourWithUnassignedGaps.fullyReadablePassages>=6,'720-row laptop fixture must retain six readable speech passages.');
      assert(hourWithUnassignedGaps.scrollHeight<=25003,'Hour fixture must fit the original dense layout scroll budget.');
      await screenshot('after-hour-720rows-unassigned-gaps-1280x720.png');
      await evaluate(`(()=>{window.hourDraft=document.querySelector('[data-segment-id=speech-0] textarea');hourDraft.focus();hourDraft.value='My retained meeting correction';hourDraft.dispatchEvent(new Event('input',{bubbles:true}));
        doc.segments[0].text='';doc.segments[0].machine_revision=2;renderLiveSegments();})()`);
      assert.equal(await evaluate("document.querySelector('[data-segment-id=speech-0] textarea')===hourDraft"),true);
      assert.equal(await evaluate('hourDraft.value'),'My retained meeting correction');
      assert.equal(await evaluate('doc.segments.length'),720);
      await evaluate('passageDrafts.clear();hourDraft.blur()');
      checks.push('The 720-row hour sample at1280×720 keeps360 speech cards and720 internal rows, collapses360 unassigned gaps without claiming silence, adds no blank naming controls, and retains a focused draft across empty refinement.');
      await send('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});
      await evaluate(`(()=>{speakerdeskTheme.set('light');doc.speakers={speaker_0:'Albert',overlap_unknown:'Unknown speaker'};
        doc.segments=[
          {id:'context',start:0,end:10,speaker:'speaker_0',language:'en',text:'Recovered audible words using recent context.',language_detection:{mode:'auto',reason:'recent_context'},refinement_state:'refined'},
          {id:'overlap',start:10,end:20,speaker:'overlap_unknown',language:'en',text:'Audible words are retained even when the speaker is unknown.',refinement_state:'unresolved'},
          {id:'failed',start:20,end:30,speaker:'overlap_unknown',text:'',language_detection:{mode:'auto',reason:'uncertain'},transcription_review:{reason:'transcription_failed'},refinement_state:'unresolved'},
          {id:'empty',start:30,end:40,speaker:'speaker_0',text:'',language_detection:{mode:'auto',reason:'best_effort'},transcription_review:{reason:'empty_result'},refinement_state:'unresolved'}
        ];renderLiveSegments();$('transcript-pane').scrollTop=0;document.querySelectorAll('.rolling-review').forEach(el=>{el.hidden=false;el.open=true;});})()`);
      assert.match(await evaluate("document.querySelector('[data-segment-id=context]').textContent"),/recent meeting context/);
      assert.equal(await evaluate("document.querySelector('[data-segment-id=overlap] textarea').value"),'Audible words are retained even when the speaker is unknown.');
      assert.equal(await evaluate("document.querySelector('#segments [data-segment-id=failed]')===null"),true);
      assert.match(await evaluate("document.querySelector('#retained-audio-review [data-segment-id=failed]').textContent"),/Transcription failed/);
      assert.match(await evaluate("document.querySelector('#retained-audio-review [data-segment-id=empty]').textContent"),/No transcript text returned/);
      await evaluate("$('retained-audio-review').open=true");
      await screenshot('after-recovered-unknown-failed-review-1280x800.png');
      checks.push('Recent-context evidence remains available on demand; unknown-speaker text stays editable; actual ASR failure/empty results remain available in audio details, including overlapping speakers.');
      await evaluate(`(()=>{doc.segments.push(...Array.from({length:40},(_,n)=>({id:'silence-'+n,start:40+n,end:41+n,speaker:'speaker_0',text:'',refinement_state:'unresolved',audio_state:'digital_silence'})));
        doc.segments.push({id:'quiet',start:80,end:90,speaker:'speaker_0',text:'Quiet but intelligible words remain in the transcript.',language:'en',language_detection:{mode:'auto',reason:'best_effort'},refinement_state:'refined'});
        renderLiveSegments();$('transcript-pane').scrollTop=0;})()`);
      assert.equal(await evaluate("document.querySelectorAll('#segments .segment').length"),3);
      assert.equal(await evaluate("document.querySelectorAll('#retained-audio-review .retained-audio-range').length"),2);
      assert.equal(await evaluate("doc.segments.length"),45);
      checks.push('Forty digitally silent pause ranges create no voice cards or review noise; nonblank quiet uncertain words and all internal timeline rows remain.');
      await screenshot('after-silence-pauses-retained-words-1280x800.png');
      await evaluate(`(()=>{window.editedText=document.querySelector('[data-segment-id=quiet] textarea');editedText.focus();editedText.value='My protected audible words';editedText.dispatchEvent(new Event('input',{bubbles:true}));
        const row=doc.segments.find(s=>s.id==='quiet');row.text='';row.machine_revision=2;row.refinement_state='unresolved';renderLiveSegments();})()`);
      assert.equal(await evaluate("document.querySelector('[data-segment-id=quiet] textarea')===editedText"),true);
      assert.equal(await evaluate("editedText.value"),'My protected audible words');
      await evaluate("passageDrafts.clear();editedText.blur();renderLiveSegments()");
      assert.equal(await evaluate("document.querySelector('#segments [data-segment-id=quiet]')===null"),true);
      assert.equal(await evaluate("doc.segments.some(s=>s.id==='quiet')"),true);
      checks.push('An asynchronous empty refinement preserves the focused user draft; after explicitly discarding the draft its blank machine card disappears while the audio/timeline row remains.');
      await evaluate(`(()=>{doc.segments.push({id:'unsupported-provisional',start:90,end:95,speaker:'speaker_0',text:'',refinement_state:'provisional',language_detection:{mode:'auto',reason:'unsupported'}},
        {id:'historical-short',start:95,end:100,speaker:'speaker_0',text:'',refinement_state:'provisional',language_detection:{mode:'auto',reason:'insufficient_speech'}});
        $('retained-audio-review').open=false;renderLiveSegments();})()`);
      assert.equal(await evaluate("document.querySelector('#segments [data-segment-id=unsupported-provisional]')===null"),true);
      assert.match(await evaluate("document.querySelector('#retained-audio-review [data-segment-id=unsupported-provisional]').textContent"),/unsupported/);
      assert.equal(await evaluate("document.querySelector('#retained-audio-review [data-segment-id=historical-short]')!==null"),true);
      assert.equal(await evaluate("$('retained-audio-review').open"),false);
      checks.push('Blank provisional unsupported-language and historical insufficient-speech ranges remain available in collapsed audio review; neither becomes a voice bubble or is mislabeled digital silence.');
      await evaluate(`(async()=>{await api('/fixture/saved-owned',{method:'POST',body:JSON.stringify({owned:false})});
        selected=await api('/api/jobs/'+selected.id);doc=structuredClone(selected.document);dirty=false;meetingPoll=false;await refreshMeeting();meetingPoll=true;setStatus();renderEditor();
        const incoming={schema_version:1,speakers:{speaker_0:'Imported speaker'},segments:[{id:'imported-passage',start:0,end:12,speaker:'speaker_0',text:'Imported transcript words remain intact.'}]};
        const transfer=new DataTransfer();transfer.items.add(new File([JSON.stringify(incoming)],'meeting.json',{type:'application/json'}));
        $('import').files=transfer.files;$('import').dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await wait("!saving && $('notice').textContent==='Imported and saved locally.'");
      assert.deepEqual(dialogs,['Replace the transcript of Rolling context · synthetic CPU fixture with meeting.json?']);
      assert.equal(await evaluate("doc.segments[0].text"),'Imported transcript words remain intact.');
      assert.equal(await evaluate("doc.provenance.kind"),'imported');
      assert.equal(await evaluate("document.querySelector('[data-segment-id=imported-passage] textarea').value"),'Imported transcript words remain intact.');
      assert.equal(await evaluate("$('meeting-language-control').hidden"),true);
      checks.push('Real JSON import control validates and saves locally, keeps imported words/provenance, and hides the current-meeting selector after capture ends.');
      await evaluate(`(()=>{doc.segments=[{id:'verified',start:0,end:10,speaker:'speaker_0',text:'Manually verified words.',transcription_review:{reason:'transcription_failed'},refinement_state:'unresolved',review_resolution:'words_reviewed'}];renderLiveSegments()})()`);
      assert.equal(await evaluate("passageReviewReason(doc.segments[0])"),'');
      assert.equal(await evaluate("document.querySelector('[data-segment-id=verified] .refinement-badge').textContent"),'Words reviewed');
      assert.equal(await evaluate("document.querySelector('[data-segment-id=verified] .rolling-review').hidden"),true);
      await evaluate("doc.segments[0].speaker='overlap_unknown';doc.speakers.overlap_unknown='Unknown speaker';renderLiveSegments()");
      assert.match(await evaluate("passageReviewReason(doc.segments[0])"),/Overlapping speakers/);
      checks.push('Explicitly reviewed nonempty failed-ASR words clear text warnings while unresolved speaker identity stays visible.');
    }
    assert.deepEqual(errors,[]);
    await writeFile(join(output,`${mode}-density-report.json`),JSON.stringify({fixture:'Representative synthetic text only; real renderer/assets; no models/devices',measurements,hourWithUnassignedGaps,checks,pageErrors:errors},null,2));
    console.log(JSON.stringify({measurements,hourWithUnassignedGaps,checks},null,2));
  }finally{
    ws?.close();browser.kill('SIGTERM');await new Promise(resolve=>{if(browser.exitCode!==null)resolve();else{browser.once('exit',resolve);setTimeout(()=>{browser.kill('SIGKILL');resolve();},2000).unref();}});
    await rm(profile,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
