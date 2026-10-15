'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="speakerdesk-token"]').content;
let config, jobs = [], selected = null, doc = null, dirty = false, polling = false, saving = false, editGeneration = 0, selectionGeneration = 0;
let meeting = null, followingLive = true, passageEnd = null, passageStart = null, retrying = false, separationRequestMessage = '', startingJob = null;
let changingLiveLanguage = false, changingDefaultLanguage = false, defaultLanguageGeneration = 0;
let pendingExport = null, autosaveTimer = null, setupState = null, undoRemoval = null;
let updateState = {id: 0, state: 'unavailable'}, updateFrozen = false, updatePolling = false;
let updatePreparation = null, updateRequesting = false, localMutations = 0;
let startupUpdateChecked = false, newMeetingOpening = false;
let dismissedUpdateVersion = '';
let serviceUnavailable=false, initializing=false, initialized=false, wired=false, pollingStarted=false, themeGeneration=0, changingTheme=false;
let recoveryClosing=false;
const cancelingJobs=new Set();
const timingDrafts=new Map();
const savedPassageBindings = new WeakMap();
const readingTurnBindings = new WeakMap();
let playbackDocument = null, playbackIndexDirty = true, playbackRows = [], playbackEnds = [];
let playbackSegments = new Map(), playbackCards = new Map(), playingCards = new Set();
const narrowLayout = () => matchMedia('(max-width: 900px)').matches;
const liveStatuses = ['starting','recording','paused','finishing'];
const isLive = () => selected && liveStatuses.includes(selected.status);
const isProgressiveImport = () => selected?.kind === 'import' && selected.import_resumable === true;
// One binding map supplies keyboard dispatch and every UI shortcut hint.
const actionShortcuts = Object.freeze({settings:',','new-meeting':'n','import-recording':'o',find:'f',save:'s'});

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function renderIcons() {
  // Lucide retains data-lucide on SVGs. Mark only fresh placeholders so a
  // transcript reconciliation does not replace icons in retained rows.
  for(const placeholder of document.querySelectorAll('i[data-lucide]'))placeholder.dataset.lucidePending=placeholder.dataset.lucide;
  lucide.createIcons({nameAttr:'data-lucide-pending'});
  for(const icon of document.querySelectorAll('svg[data-lucide-pending]'))icon.removeAttribute('data-lucide-pending');
}
async function api(path, options = {}) {
  const {updateSave = false, ...requestOptions} = options;
  const mutation = !['GET', 'HEAD'].includes((options.method || 'GET').toUpperCase());
  const updater = path === '/api/updates';
  if(recoveryClosing && mutation)throw new Error('Saving your private recovery copy before closing.');
  if (updateFrozen && mutation && !updater && !updateSave) throw new Error('Speakerdesk is preparing an update. Cancel the update to continue.');
  const headers = {'X-Speakerdesk-Token': token, ...options.headers};
  if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  if (mutation && !updater) localMutations++;
  let updateTimer, controller;
  try {
    if(updater)controller=new AbortController();
    const operation=(async()=>{
      let response;
      try { response=await fetch(path, {...requestOptions, headers,...(controller?{signal:controller.signal}:{})}); }
      catch(error){if(!updater)markServiceUnavailable();throw error;}
      const data = await response.json();
      if (!response.ok) throw Object.assign(new Error(data.error || 'Local request failed.'),{status:response.status});
      return data;
    })();
    if(!updater)return await operation;
    return await Promise.race([operation,new Promise((_,reject)=>{
      updateTimer=setTimeout(()=>{controller.abort();reject(new Error('The local update service did not respond. Check the update status and try again.'));},10000);
    })]);
  } finally { clearTimeout(updateTimer);if (mutation && !updater) localMutations--; }
}
function notice(text, error = false, action = null) {
  const host = $('notice'); host.replaceChildren(); host.hidden = !text;
  host.className = error ? 'error' : '';
  if (!text) return;
  const message = node('span', text); host.append(message);
  if (action) {
    const button = node('button', action.label, 'text-button notice-action'); button.type = 'button';
    button.addEventListener('click', () => { host.hidden = true; action.run(); });
    host.append(button);
  }
  if (!error) setTimeout(() => { if (host.contains(message)) host.hidden = true; }, action ? 8000 : 4500);
}
function markServiceUnavailable(message='The local service is unavailable. Your open corrections are kept here.') {
  serviceUnavailable=true;$('connection-status').hidden=false;$('connection-message').textContent=message;
  $('pause-meeting').disabled=true;$('stop-meeting').disabled=true;$('cancel-job').disabled=true;$('start-meeting').disabled=true;
  $('microphone-level').value=0;$('system-level').value=0;$('live-delay').textContent='Status unavailable';
  $('listening').textContent='Recording status unavailable. Your open words are retained.';
  $('active-meeting').textContent='Recording status unavailable';
  $('recovery-copies').hidden=false;$('recovery-copy-open').hidden=false;
  $('elapsed').textContent='—';$('job-message').textContent='Local service unavailable · open words retained';$('recording-dot').classList.add('paused');
}
window.speakerdeskHostUnavailable=message=>markServiceUnavailable(message || 'The local service stopped. Reopen Speakerdesk to restore access. Your open corrections are retained.');
function markServiceAvailable() {
  if(!serviceUnavailable)return;
  serviceUnavailable=false;$('connection-status').hidden=true;$('listening').textContent='Listening. Your first phrases will appear here.';
  if(selected)setStatus();renderStartState();
}
async function retryConnection() {
  $('connection-retry').disabled=true;
  try {await refreshConfig();await poll();await refreshMeeting();}
  catch(error){markServiceUnavailable(error.message);}
  finally{$('connection-retry').disabled=false;}
}
function renderExportState() {
  document.querySelectorAll('[data-export]').forEach(button=>{button.disabled=updateFrozen || !!pendingExport || !doc?.segments?.length;});
  const notices=document.querySelector('a[href="/api/notices"]');
  if(notices){notices.setAttribute('aria-disabled',String(!!pendingExport));notices.setAttribute('aria-busy',String(!!pendingExport));}
  document.querySelector('.export-menu').setAttribute('aria-busy',String(!!pendingExport));
}
function finishExport(operation) {
  if(pendingExport!==operation)return;
  pendingExport=null;renderExportState();
}
window.speakerdeskExportStatus=(nonce,phase)=>{
  const operation=pendingExport;
  if(!operation?.native || !/^[0-9a-f]{32}$/.test(nonce) || operation.nonce!==nonce)return false;
  const messages={downloading:'Preparing export…',choosing:'Choose where to save the export.',writing:'Saving export…',saved:'Export saved.',cancelled:'Export cancelled.',failed:'Export could not be saved. Try again.'};
  if(!Object.hasOwn(messages,phase))return false;
  notice(messages[phase],phase==='failed');
  if(['saved','cancelled','failed'].includes(phase))finishExport(operation);
  return true;
};
async function downloadExport(kind=null) {
  if(updateFrozen)throw new Error('Cancel the update before exporting.');
  if(pendingExport)return;
  const snapshot={id:selected?.id,name:selected?.name,generation:selectionGeneration};
  if(kind && (!['txt','srt','vtt','json'].includes(kind) || !/^[0-9a-f]{32}$/.test(snapshot.id || '')))throw new Error('Choose a saved meeting to export.');
  const operation={nonce:crypto.randomUUID().replaceAll('-',''),native:window.speakerdeskNativeExport===true};
  pendingExport=operation;renderExportState();
  try {
    const checkMeeting=()=>{
      if(kind && (selected?.id!==snapshot.id || selectionGeneration!==snapshot.generation))throw new Error('The selected meeting changed. Choose Export again.');
      if(kind && (dirty || hasPassageDrafts() || hasRecoverablePassageDrafts() || timingDrafts.size))throw new Error('Save or discard your latest edits before exporting.');
    };
    if(kind){if(hasPassageDrafts() || hasRecoverablePassageDrafts() || timingDrafts.size)throw new Error('Save or discard passage drafts before exporting.');await flushSave();checkMeeting();}
    if(pendingExport!==operation)return;
    const path=kind?`/api/jobs/${snapshot.id}/export/${kind}`:'/api/notices';
    const url=new URL(path,location.origin);url.searchParams.set('download',operation.nonce);
    const response=await fetch(url,{method:'HEAD',headers:{'X-Speakerdesk-Token':token},cache:'no-store',redirect:'error'});
    if(!response.ok)throw new Error(response.status===413?'This export is too large to save. Try another format.':'This export is unavailable. Try again.');
    checkMeeting();if(pendingExport!==operation)return;
    const link=node('a');link.href=url.href;link.download=kind?`${snapshot.name.replace(/\.[^.]+$/, '')}.${kind}`:'Speakerdesk-third-party-notices.txt';
    document.body.append(link);
    try{link.click();}finally{link.remove();}
    if(!operation.native)finishExport(operation);
  }catch(error){finishExport(operation);throw error;}
}
function renderSaveState() {
  const button = $('save');
  const drafts=passageDrafts.size+recoverablePassageDrafts.size,pending=hasPendingPassageSaves();
  button.disabled = updateFrozen || saving || !dirty;
  (button.querySelector('.shortcut-label') || button).textContent = saving || pending ? 'Saving…' : dirty ? 'Save now' : drafts ? 'Unsaved corrections' : timingDrafts.size ? 'Check timing' : 'Saved';
  button.classList.toggle('is-dirty', dirty && !saving);
  button.dataset.shortcutTitle = drafts ? 'Save each passage correction, or recover the retained correction.' : timingDrafts.size ? 'Fix the invalid timing fields before leaving.' : dirty ? 'Edits save automatically. Save now' : 'All changes saved on this Mac';
  button.title = button.dataset.shortcutTitle;
  renderShortcutControl(button);
}
function scheduleAutosave() {
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => {
    autosaveTimer = null;
    if (!dirty || !selected || !doc || isLive()) return;
    // A removal stays local while Undo is offered: the server drops audio
    // anchors and edit protection for passages missing from a saved document.
    if (saving || undoRemoval) { scheduleAutosave(); return; }
    save({quiet: true}).catch(error => notice(`Couldn’t save automatically: ${error.message}`, true));
  }, 1200);
}
async function flushSave({canContinue = () => true} = {}) {
  clearTimeout(autosaveTimer); autosaveTimer = null;
  while (dirty || saving) {
    if(!canContinue())return;
    if (saving) await new Promise(resolve => setTimeout(resolve, 50));
    else await save({quiet: true});
  }
}
function changed() { editGeneration++; dirty = true; playbackIndexDirty = true; renderSaveState(); scheduleAutosave(); }
function time(seconds) {
  const minutes = Math.floor(seconds / 60), rest = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2,'0')}`;
}
function passageTime(seconds) {
  const ms=Math.round(seconds*1000),minutes=Math.floor(ms/60000),rest=ms%60000;
  return `${String(minutes).padStart(2,'0')}:${String(Math.floor(rest/1000)).padStart(2,'0')}.${String(rest%1000).padStart(3,'0')}`;
}
function emptyFailedCapture(job) {
  return job?.kind==='meeting' && job.status==='failed' &&
    (job.capture_audio?.state==='empty' || (!job.capture_audio && !job.duration));
}
function renderJobs() {
  $('job-count').textContent = jobs.length;
  const host=$('jobs'), existing=new Map(Array.from(host.children).map(child=>[child.dataset.jobId || child.dataset.group,child]));
  const retained=new Set();let cursor=host.firstChild;
  const place=child=>{retained.add(child);if(child!==cursor)host.insertBefore(child,cursor);cursor=child.nextSibling;};
  const query=$('meeting-search').value.toLowerCase();let lastDate='';
  const today=new Date(),yesterday=new Date();yesterday.setDate(today.getDate()-1);
  for (const job of jobs) {
    if(query && !job.name.toLowerCase().includes(query))continue;
    const date=new Date(job.created*1000),dateKey=date.toDateString();
    const group=date.toDateString()===today.toDateString()?'Today':date.toDateString()===yesterday.toDateString()?'Yesterday':date.toLocaleDateString(undefined,{month:'short',day:'numeric'});
    if(dateKey!==lastDate){
      const heading=existing.get(dateKey) || node('div',group,'date-group');heading.dataset.group=dateKey;
      if(heading.textContent!==group)heading.textContent=group;place(heading);lastDate=dateKey;
    }
    let button=existing.get(job.id);
    if(!button){button=node('button',undefined,'job');button.dataset.jobId=job.id;button.append(node('strong'),node('small'));button.addEventListener('click',()=>select(job.id).catch(e=>notice(e.message,true)));}
    const signature=JSON.stringify([job.name,job.created,job.status]);
    if(button.dataset.signature!==signature){
      const status=liveStatuses.includes(job.status)?` · ${job.status==='starting'?'Starting':job.status==='finishing'?'Finishing':job.status==='paused'?'Paused':'Recording'}`:job.status==='failed'?' · Interrupted':job.status==='delete_failed'?' · Cleanup failed':job.status==='deleting'?' · Deleting':'';
      button.title=job.name;button.querySelector('strong').textContent=job.name;
      button.querySelector('small').textContent=date.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'})+status;
      button.dataset.signature=signature;
    }
    setPassageClass(button,'active',selected?.id===job.id);place(button);
  }
  for(const child of Array.from(host.children))if(!retained.has(child))child.remove();
}
function languageReady(mode) {
  return config.readiness.configured && (mode !== 'auto' || config.readiness.automatic_language)
    && setupState?.core_ready !== false
    && (config.readiness.word_alignment === false || setupState?.alignment?.ready !== false);
}
function setStatus() {
  $('recording-name').textContent = selected.name; $('recording-name').title = selected.name;
  $('meeting-date').textContent = new Date(selected.created*1000).toLocaleString(undefined,{weekday:'long',month:'long',day:'numeric',hour:'numeric',minute:'2-digit'});
  const starting=startingJob===selected.id;
  const passageMessage=separationRequestMessage || (selected.passage_separation ? selected.passage_separation.phase || 'Separating voices locally…' : retrying || selected.passage_retry ? 'Retrying passage locally…' : '');
  $('job-message').textContent = passageMessage || (starting ? 'Starting transcription…' : selected.message);
  const live=isLive(), busy=live || starting || ['preparing','processing','queued'].includes(selected.status);
  $('job-progress').hidden=selected.kind==='meeting' || !(busy || retrying || selected.passage_retry || selected.passage_separation);
  $('editor').inert=updateFrozen || retrying || (busy && !live && !isProgressiveImport());
  $('recording-dot').hidden=!live;
  if(live)$('recording-dot').removeAttribute('hidden');else $('recording-dot').setAttribute('hidden','');
  $('recording-dot').classList.toggle('paused',selected.status==='paused');
  $('run').hidden = (!!doc?.segments?.length && !isProgressiveImport()) || live; $('run').disabled = serviceUnavailable || busy || !!selected.inference_owned || !languageReady(selected.language);
  $('run').textContent=isProgressiveImport()?'Resume transcription':'Transcribe';
  $('processing-row').hidden=!busy && !serviceUnavailable && !['failed','delete_failed','deleting'].includes(selected.status) && $('run').hidden;
  $('run').title = languageReady(selected.language) ? '' : setupState?.status==='loading' ? 'Loading models…' : 'Finish model setup in Settings → Models first.';
  $('manual').hidden = !!doc || busy || !selected.duration;
  $('delete').disabled = busy || !!selected.inference_owned;
  $('delete').title=selected.status==='delete_failed'?'Retry media cleanup':'Delete recording';
  const cancelPending=cancelingJobs.has(selected.id) && (!!selected.inference_owned || ['preparing','processing','queued'].includes(selected.status));
  if(cancelingJobs.has(selected.id) && !selected.inference_owned && !['preparing','processing','queued'].includes(selected.status))cancelingJobs.delete(selected.id);
  $('cancel-job').hidden=selected.kind==='meeting' || (!['preparing','processing','queued'].includes(selected.status) && !cancelPending);
  $('cancel-job').disabled=serviceUnavailable || cancelPending;
  $('cancel-job').textContent=cancelPending?'Stopping processing…':'Cancel processing';
  $('duration').textContent = selected.duration ? time(selected.duration) : emptyFailedCapture(selected)?'No audio':'Preparing…';
  $('segment-count').textContent = doc && !live ? passageCountLabel() : '';
  $('save').hidden = !doc || (live && !hasPassageDrafts() && !hasRecoverablePassageDrafts() && !hasPendingPassageSaves()); $('search').disabled = !doc;
  $('player').hidden = !selected.duration || live;
  document.querySelector('.player-panel').hidden=live || emptyFailedCapture(selected);
  document.querySelector('.export-menu').inert=live || !doc;
  document.querySelector('.export-menu').hidden=live;
  $('speaker-tools').hidden=live || isProgressiveImport();$('add').hidden=live || isProgressiveImport();$('editor-hints').hidden=live || !doc || isProgressiveImport();
  $('import').disabled=live || isProgressiveImport();
  renderFollowLive();
  $('listening').hidden=!live || !!doc?.segments?.length;
  renderLiveLanguage();
  renderExportState();
  renderRefinementStatus();
  renderShortcutHints();
  if(serviceUnavailable){$('processing-row').hidden=false;$('job-message').textContent='Local service unavailable · open words retained';$('recording-dot').classList.add('paused');}
}
async function select(jid) {
  if(hasPendingPassageSaves()){notice('Wait for the passage save to finish before switching meetings.');return;}
  if(retrying){notice('Wait for the passage retry to start before switching recordings.');return;}
  if (identityBusy) { notice('Wait for the name change to finish before switching meetings.'); return; }
  $('editor').inert = true;
  let result, generation;
  try {
    if (!(await saveBeforeLeaving())) return;
    if ((hasPassageDrafts() || hasRecoverablePassageDrafts()) && !confirm('Discard unsaved or recoverable passage corrections and open another recording?')) return;
    generation = ++selectionGeneration;
    result = await api(`/api/jobs/${jid}`);
  } finally { if (!identityBusy) $('editor').inert = false; }
  if (generation !== selectionGeneration) return;
  passageDrafts.clear();recoverablePassageDrafts.clear();resetPlayback();$('segments').replaceChildren();
  selected = result; doc = selected.document ? structuredClone(selected.document) : null; dirty = false;
  passageEnd=null;passageStart=null;
  $('empty').hidden = true; $('workspace').hidden = false;
  if(!isLive() && selected.duration) $('player').src = `/api/jobs/${jid}/audio`;
  else {$('player').pause();$('player').removeAttribute('src');}
  $('player').playbackRate = Number($('speed').value);
  $('search').value = ''; notice(''); setStatus(); renderJobs(); renderEditor();
}
async function saveBeforeLeaving() {
  if(timingDrafts.size){notice('Fix or reset the invalid passage timing before leaving.',true);return false;}
  if (!dirty && !saving) return true;
  try { await flushSave(); return true; }
  catch (error) { return confirm(`Your latest edits couldn’t be saved (${error.message}). Discard them?`); }
}
function passageCountLabel() {
  const shown = doc.segments.filter(showTranscriptPassage);
  return `${shown.length} passage${shown.length === 1 ? '' : 's'}`;
}
function renderFollowLive() {
  $('follow-live').hidden = !isLive() || followingLive;
}
function nameSpeaker(id,allPersonTracks=false) {
  if(updateFrozen){notice('Cancel the update before changing a speaker name.',true);return;}
  flushSave().then(() => openNamePicker(id,allPersonTracks)).catch(error => notice(error.message, true));
}
function speakerPerson(id) {
  if(id==='unassigned' || id==='multiple_speakers')return null;
  const assignment=selected.speaker_assignments?.[id];
  // A local name draft breaks its saved assignment, just as transcript save does.
  return typeof assignment?.person_id==='string' && assignment.person_id && assignment.name===doc.speakers[id]
    ? assignment.person_id : null;
}
function speakerIdentity(id) {
  const person=speakerPerson(id);
  return person?`person:${person}`:`track:${id}`;
}
function visibleSpeakers() {
  const seen=new Set();
  return Object.keys(doc.speakers).filter(id=>{
    const identity=speakerIdentity(id);if(seen.has(identity))return false;
    seen.add(identity);return true;
  });
}
function speakerLabel(id) {
  if(id==='unassigned')return 'Speaker unassigned';
  const person=speakerPerson(id);
  if(person)id=Object.keys(doc.speakers).find(track=>speakerPerson(track)===person);
  return doc.speakers[id] || 'Unknown speaker';
}
function speakerColor(id) {
  if(id==='unassigned' || id==='multiple_speakers')return -1;
  // Named tracks share a person's color; unlinked track colors survive renaming.
  id=speakerPerson(id)?speakerIdentity(id):id;
  let hash=0;for(const char of id)hash=(hash*31+char.codePointAt(0))>>>0;
  return hash%8;
}
function updateSpeakerAvatar(avatar,id) {
  const color=String(speakerColor(id)),label=speakerLabel(id),number=label.match(/^Speaker (\d+)$/);
  const text=color==='-1'?(id==='multiple_speakers'?'…':'?'):number?`S${number[1]}`:label==='Unknown speaker'?'?':label.trim().split(/\s+/u).slice(0,2).map(part=>Array.from(part)[0]).join('').toLocaleUpperCase();
  if(avatar.textContent!==text)avatar.textContent=text;
  if(avatar.dataset.color!==color)avatar.dataset.color=color;
}
function refreshSpeakerRecognition(card,segment) {
  const existing=card.querySelector('.recognized-speaker');
  const recognized=selected.speaker_assignments?.[segment.speaker]?.source==='automatic_voice';
  if(!recognized){existing?.remove();return;}
  if(existing)return;
  const hint=node('span','Recognized','review-tag routine-state recognized-speaker');
  hint.title='Matched a saved voice. Choose the speaker name to correct it.';
  const top=card.querySelector('.segment-top');top.insertBefore(hint,top.querySelector('.segment-actions'));
}
function speakerNameControl(id,className='name-speaker',allPersonTracks=false) {
  const assigned=id!=='unassigned',label=speakerLabel(id);
  const control=node(assigned?'button':'span',label,className);control.dataset.speaker=id;
  if(assigned) {
    control.type='button';control.title='Name this speaker or remember their voice';
    control.setAttribute('aria-label',`Name or remember voice for ${label}`);
    control.addEventListener('click',()=>nameSpeaker(id,allPersonTracks));
  }else {
    control.title='These words have no assigned speaker.';
    control.setAttribute('aria-label',label);
  }
  return control;
}
async function removePassage(segment) {
  if(updateFrozen || hasPassageDrafts() || hasPendingPassageSaves()){notice('Save or discard passage corrections before deleting a passage.',true);return;}
  const jid=selected.id;
  const update=async deleted=>{
    await flushSave();
    if(selected?.id!==jid)return false;
    const current=doc.segments.find(row=>row.id===segment.id);if(!current)return false;
    const key=`${jid}:${segment.id}`,operation={jid,segment:structuredClone(current)};
    passageSaveOperations.set(key,operation);renderSaveState();
    let result;
    try {result=await api(`/api/jobs/${jid}/segments/${encodeURIComponent(segment.id)}`,{method:'PATCH',body:JSON.stringify({segment_revision:current.machine_revision ?? 0,changes:{deleted}})});}
    finally {passageSaveOperations.delete(key);renderSaveState();}
    if(selected?.id!==jid)return false;
    selectionGeneration++;
    doc.segments[doc.segments.findIndex(row=>row.id===segment.id)]=result.segment;
    const saved=selected.document?.segments.findIndex(row=>row.id===segment.id);
    if(saved>=0)selected.document.segments[saved]=structuredClone(result.segment);
    renderSegments();renderSaveState();return true;
  };
  try {
    if(!await update(true))return;
    clearTimeout(undoRemoval?.timer);
    const removal={jid,timer:setTimeout(()=>{if(undoRemoval===removal)undoRemoval=null;},8000)};undoRemoval=removal;
    notice('Passage deleted. Original audio kept.',false,{label:'Undo',run:async()=>{
      if(undoRemoval!==removal || selected?.id!==jid)return;
      try {if(await update(false)){clearTimeout(removal.timer);undoRemoval=null;notice('Passage restored.');}}
      catch(error){notice(error.message,true);}
    }});
  }catch(error){notice(error.message,true);}
}
function markReviewable(segment) {
  return !!segment.text.trim() && !hasOverlappingSpeakers(segment) && segment.speaker !== 'unassigned';
}
function renderSearchResults(visible, query) {
  $('no-results').hidden = visible > 0 || !query;
  $('no-results-text').textContent = `No passages match “${$('search').value.trim()}”.`;
  $('search-count').textContent = query ? `${visible} ${visible === 1 ? 'match' : 'matches'}` : '';
  document.querySelector('.transcript-search').classList.toggle('has-query', !!query);
}
function speakerOptions(selectEl, current) {
  selectEl.replaceChildren();
  for (const id of visibleSpeakers()) {
    const selected=speakerIdentity(id)===speakerIdentity(current);
    const option = node('option', speakerLabel(id)); option.value = selected?current:id; option.selected = selected; selectEl.append(option);
  }
}
function renderSpeakerList() {
  const host=$('speaker-list'),signature=JSON.stringify([doc.speakers,selected.speaker_assignments]);
  if(host.dataset.signature===signature)return;
  host.dataset.signature=signature;host.replaceChildren();
  visibleSpeakers().forEach(id => {
    const row = node('div', undefined, 'speaker-name');
    const dot = node('span', undefined, 'speaker-dot'); dot.dataset.color = speakerColor(id);
    const group=!!speakerPerson(id) && Object.keys(doc.speakers).some(track=>track!==id && speakerIdentity(track)===speakerIdentity(id));
    const choose = speakerNameControl(id,'speaker-name-action name-speaker',group);
    choose.replaceChildren(dot, node('span', speakerLabel(id)));
    if(id!=='unassigned')choose.append(icon('pencil'));
    row.append(choose); host.append(row);
  });
}
function renderEditor() {
  $('inspector').hidden = true;
  $('editor').hidden = !doc; if (!doc) return;
  renderSaveState();
  $('segment-count').textContent = isLive() ? '' : passageCountLabel();
  $('timing-note').textContent = doc.provenance?.timing || 'User supplied segment boundaries.';
  $('warnings').textContent = (doc.warnings || []).join(' '); $('warnings').hidden = !doc.warnings?.length;
  renderSegments();
  refreshIdentitySuggestions();
  updateNameControls();
}
// Server-owned reading slices are a projection of one canonical passage.
// Offsets use Unicode code points, not textarea/JavaScript UTF-16 positions.
function validReadingTurns(segment) {
  const turns=segment.reading_turns;
  if(!Array.isArray(turns) || !turns.length || typeof segment.text!=='string')return null;
  let cursor=0,joined='';
  for(const turn of turns) {
    if(!turn || typeof turn.text!=='string' || typeof turn.speaker!=='string' ||
       !['single','overlap','unknown'].includes(turn.attribution) ||
       !Number.isSafeInteger(turn.start_offset) || !Number.isSafeInteger(turn.end_offset) ||
       turn.start_offset!==cursor || turn.end_offset<=cursor ||
       turn.end_offset-cursor!==Array.from(turn.text).length)return null;
    if(![turn.start,turn.end].every(value=>value===null || (Number.isFinite(value) && value>=0)) ||
       (turn.start!==null && turn.end!==null && turn.end<turn.start))return null;
    cursor=turn.end_offset;joined+=turn.text;
  }
  return joined===segment.text && cursor===Array.from(segment.text).length ? turns : null;
}
function hasOverlappingSpeakers(segment) {
  if(segment.separation)return false;
  const turns=validReadingTurns(segment);
  if(turns)return turns.some(turn=>turn.attribution==='overlap');
  if(segment.canonical_utterance_id) {
    // Canonical candidates are a union across the whole passage. Only current
    // temporal concurrency or an explicit overlap label establishes overlap.
    const activity=segment.speaker_activity,regions=activity?.regions;
    if(activity?.audio_revision===segment.audio_revision && Array.isArray(regions) && regions.length) {
      let cursor=segment.start_sample,overlap=false;
      for(const region of regions) {
        const names=region?.speakers;
        if(!Number.isSafeInteger(cursor) || !Number.isSafeInteger(region?.start_sample) ||
           !Number.isSafeInteger(region?.end_sample) || region.start_sample!==cursor ||
           region.end_sample<=cursor || region.end_sample>segment.end_sample ||
           !Array.isArray(names) || names.some(name=>typeof name!=='string' || !/^speaker_\d+$/.test(name)) ||
           new Set(names).size!==names.length)return segment.speaker.startsWith('overlap');
        overlap ||= names.length>1;cursor=region.end_sample;
      }
      if(cursor===segment.end_sample)return overlap;
    }
    return segment.speaker.startsWith('overlap');
  }
  return segment.speaker.startsWith('overlap') ||
    (segment.speaker!=='multiple_speakers' && segment.speaker_candidates?.length>1);
}
function readingTurnLabel(turn) {
  if(turn.attribution==='overlap')return 'Overlapping speakers';
  if(turn.attribution==='unknown')return speakerLabel('unassigned');
  return speakerLabel(turn.speaker);
}
function passageSearchText(segment) {
  const names=validReadingTurns(segment)?.map(readingTurnLabel).join(' ') || '';
  return segment.text+' '+speakerLabel(segment.speaker)+' '+names;
}
function refreshReadingTurnView(card,segment) {
  const turns=validReadingTurns(segment);let binding=readingTurnBindings.get(card);
  if(!binding && !turns)return;
  if(!binding) {
    const body=card.querySelector('.segment-body'),text=body.querySelector(':scope > textarea');
    if(!text)return;
    const host=node('div',undefined,'reading-turn-view');host.setAttribute('role','group');
    host.setAttribute('aria-label',`Speaker turns for passage at ${passageTime(segment.start)}`);
    const button=node('button','Edit passage','segment-action edit-whole-passage');button.type='button';
    button.setAttribute('aria-label',`Edit whole passage at ${passageTime(segment.start)}`);
    body.append(host);card.querySelector('.segment-actions').append(button);
    binding={host,text,button,editing:false,signature:null};readingTurnBindings.set(card,binding);
    const current=()=>doc.segments.find(row=>row.id===card.dataset.segmentId) || segment;
    button.addEventListener('click',()=>{
      if(isPassageSaving(card.dataset.segmentId) || passageDrafts.has(card.dataset.segmentId) ||
         recoverablePassageDrafts.has(card.dataset.segmentId))return;
      binding.editing=!binding.editing;if(!binding.editing)text.blur();refreshReadingTurnView(card,current());
      if(binding.editing){text.focus();fitPassageText(text);}
    });
    text.addEventListener('focus',()=>{binding.editing=true;refreshReadingTurnView(card,current());fitPassageText(text);});
    text.addEventListener('input',()=>refreshReadingTurnView(card,current()));
  }
  const {host,text,button}=binding;
  const scope=`Full passage ${passageTime(segment.start)}–${passageTime(segment.end)}`;
  text.setAttribute('aria-label',scope);
  const signature=JSON.stringify([turns,doc.speakers,selected.speaker_assignments]);
  if(signature!==binding.signature) {
    binding.signature=signature;host.replaceChildren();
    const groups=[];
    for(const turn of turns || []) {
      const previous=groups.at(-1);
      if(previous && speakerIdentity(previous.speaker)===speakerIdentity(turn.speaker) && previous.attribution===turn.attribution) {
        previous.text+=turn.text;previous.slices.push(turn);
        if(previous.start!==null && turn.start!==null && turn.end!==null)previous.end=turn.end;
        else previous.start=previous.end=null;
      }else {
        const timed=turn.start!==null && turn.end!==null;
        groups.push({...turn,slices:[turn],start:timed?turn.start:null,end:timed?turn.end:null});
      }
    }
    // Bridge only brief unassigned text between the same singleton owner.
    // This is paragraph layout, never an ownership or timing reassignment.
    const paragraphs=[];
    const shortUnknown=turn=>turn?.attribution==='unknown' && turn.text.trim() &&
      turn.text.trim().split(/\s+/u).length<=6 && Array.from(turn.text.trim()).length<=80;
    for(let index=0;index<groups.length;index++) {
      const turn=groups[index];
      while(turn.attribution==='single' && shortUnknown(groups[index+1]) &&
            groups[index+2]?.attribution==='single' && speakerIdentity(groups[index+2].speaker)===speakerIdentity(turn.speaker)) {
        const unknown=groups[index+1],following=groups[index+2];
        turn.text+=unknown.text+following.text;turn.slices.push(...unknown.slices,...following.slices);
        turn.includesUnassigned=true;
        if(turn.start!==null && unknown.start!==null && following.start!==null)turn.end=following.end;
        else turn.start=turn.end=null;
        index+=2;
      }
      paragraphs.push(turn);
    }
    // A short uncertain tail stays inline under its own explicit attribution.
    // It does not acquire the preceding speaker or an invented timestamp.
    const tail=paragraphs.at(-1),previous=paragraphs.at(-2);
    if(paragraphs.length===2 && previous?.attribution==='single' && previous.start!==null &&
       segment.speaker===previous.speaker && segment.speaker_candidates?.length===1 && shortUnknown(tail)) {
      previous.text+=tail.text;previous.slices.push(...tail.slices);previous.includesUnassigned=true;
      previous.start=previous.end=null;paragraphs.pop();
    }
    for(const turn of paragraphs) {
      const row=node('section',undefined,'reading-turn');
      row.dataset.attribution=turn.includesUnassigned?'single_with_unassigned':turn.attribution;
      const label=node('div',undefined,'reading-turn-meta');label.append(node('span',readingTurnLabel(turn),'reading-turn-speaker'));
      if(turn.attribution==='single' && turn.slices.some(slice=>selected.speaker_assignments?.[slice.speaker]?.source==='automatic_voice')) {
        const allRecognized=turn.slices.every(slice=>selected.speaker_assignments?.[slice.speaker]?.source==='automatic_voice');
        const recognized=node('span',allRecognized?'Recognized':'Includes recognized voice','review-tag');recognized.title='Matched a saved voice. Choose the passage speaker name to confirm or correct it.';label.append(recognized);
      }
      if(turn.includesUnassigned)label.append(node('span','Includes unassigned words','reading-unassigned-note'));
      if(turn.start!==null)label.append(node('span',passageTime(turn.start),'reading-turn-time'));
      const words=node('p',undefined,'reading-turn-words');
      if(turn.includesUnassigned) {
        for(const slice of turn.slices) {
          if(slice.attribution!=='unknown'){words.append(slice.text);continue;}
          const unassigned=node('span',slice.text,'reading-turn-unassigned');
          unassigned.title='Unassigned words; speaker attribution unavailable.';unassigned.setAttribute('role','note');
          unassigned.setAttribute('aria-label',`Unassigned words: ${slice.text}`);
          unassigned.dataset.attribution=slice.attribution;unassigned.dataset.speaker=slice.speaker;
          unassigned.dataset.startOffset=slice.start_offset;unassigned.dataset.endOffset=slice.end_offset;
          words.append(unassigned);
        }
      }else words.textContent=turn.text;
      row.append(label,words);host.append(row);
    }
  }
  const blocked=passageDrafts.has(segment.id) || recoverablePassageDrafts.has(segment.id) || isPassageSaving(segment.id);
  const reading=!!turns && !binding.editing && !blocked && text.value===segment.text && document.activeElement!==text;
  host.hidden=!reading;text.hidden=reading;button.hidden=!turns;
  button.disabled=blocked;button.textContent=reading?`Edit full passage${host.children.length>1?` (${host.children.length} speaker paragraphs)`:''}`:'View speaker turns';
  button.title=scope;
  let explanation=card.querySelector('.whole-passage-note');
  if(!explanation){explanation=node('p','Editing the full passage. Speaker turns are a reading view; editing words may make their attribution unavailable.','whole-passage-note');text.parentElement.insertBefore(explanation,text);}
  explanation.textContent=`${scope}. `+(turns?'Editing the full passage: all speaker paragraphs together. Speaker attribution becomes unavailable when the words change; timing and original audio are kept.':'Speaker-turn attribution is unavailable for these words. This is the full original passage.');
  explanation.hidden=reading;
  setPassageClass(card,'has-reading-turns',!!turns);setPassageClass(card,'editing-reading-turns',!reading);
}
function savedPassageSignature(segment) {
  return JSON.stringify([segment,doc.speakers,doc.provenance,selected.speaker_assignments,
    ['preparing','processing','queued'].includes(selected.status),!!selected.passage_retry,!!selected.passage_separation]);
}
function savedRetryAccepted(segment,saved) {
  const candidate=segment.retry_candidate;
  if(!candidate)return false;
  saved ??= selected.document?.segments.find(row=>row.id===segment.id);
  return saved?.retry_candidate?.id===candidate.id && Number.isFinite(saved.retry_candidate.accepted_at);
}
function refreshAcknowledgedPassageRepair(card,segment,saved) {
  if(!card)return;
  // A focused card keeps its textarea, but a saved acceptance is no longer an action.
  if(!segment.retry_candidate || savedRetryAccepted(segment,saved))card.querySelector('.retry-result')?.remove();
  const body=card.querySelector('.segment-body');
  if(body)refreshPassageLanguageIssue(body,segment);
}
function savedCard(segment) {
  const card = node('article', undefined, 'segment'); card.dataset.segmentId = segment.id;card.dataset.speaker=segment.speaker;
  const avatar=node('span',undefined,'speaker-avatar');avatar.setAttribute('aria-hidden','true');updateSpeakerAvatar(avatar,segment.speaker);
  const body=node('div',undefined,'segment-body');
  const top = node('div', undefined, 'segment-top');
  const seek = node('button', undefined, 'seek'); seek.append(icon('play'),node('span',time(segment.start))); seek.setAttribute('aria-label', `Play passage from ${passageTime(segment.start)} to ${passageTime(segment.end)}`);seek.title=`Play passage ${passageTime(segment.start)}–${passageTime(segment.end)}`;
  seek.addEventListener('click', () => {
    if(segment.separation)playSeparatedPassage(card,segment);
    else playOriginalPassage(segment);
  });
  const speaker = node('select'); speakerOptions(speaker, segment.speaker); speaker.setAttribute('aria-label', 'Segment speaker');
  speaker.disabled=!!isLive();
  seek.disabled=!!isLive();
  speaker.title='Change who spoke this passage';
  speaker.addEventListener('change', () => { segment.speaker = speaker.value; segment.review = true; changed(); renderSegments(); });
  body.append(seek);top.append(speaker);
  if(segment.separation) {
    card.dataset.separationEpisode=segment.separation.episode_id;
    seek.setAttribute('aria-label',`Play separated passage from ${passageTime(segment.start)} to ${passageTime(segment.end)}`);
  }
  const actions=node('div',undefined,'segment-actions');
  const repairToggle=node('button',undefined,'segment-action repair-toggle');repairToggle.type='button';
  repairToggle.append(icon('wrench'),node('span','Retry transcription'));repairToggle.title='Retry this passage in another language';
  repairToggle.setAttribute('aria-label',`Retry transcription at ${passageTime(segment.start)}`);repairToggle.setAttribute('aria-expanded','false');
  const remove = node('button', 'Delete passage', 'segment-action remove-segment'); remove.type='button'; remove.setAttribute('aria-label', `Delete passage at ${passageTime(segment.start)}`); remove.title='Delete this full passage; original audio is kept';
  remove.addEventListener('click', () => removePassage(segment));
  const details=node('button',undefined,'segment-action passage-details-toggle');details.type='button';details.append(icon('sliders-horizontal'),node('span','Details'));
  details.setAttribute('aria-label',`Timing and details for passage at ${passageTime(segment.start)}`);details.title='Timing & details';
  details.addEventListener('click',()=>{showInspector(segment,card,true);$('segment-details').querySelector('input')?.focus();});
  actions.append(details,repairToggle,remove);actions.hidden=!!isLive();top.append(actions);
  const text = isLive()?node('p',segment.text,'segment-text'):node('textarea');
  if(!isLive()){text.value=segment.text;text.rows=2;text.setAttribute('aria-label',`Transcript at ${time(segment.start)}`);text.placeholder='Enter transcript words';
    text.addEventListener('input',()=>{segment.text=text.value;fitPassageText(text);changed();});}

  body.append(top,text);
  const reason=passageReviewReason(segment);
  appendPassageRepair(body,segment,reason);
  appendPassageSeparation(body,segment);
  const repair=body.querySelector(':scope > .passage-repair');
  if(repair){repair.classList.add('inline-repair');
    repairToggle.addEventListener('click',()=>{repair.open=!repair.open;if(repair.open)repair.querySelector('select')?.focus();});
    repair.addEventListener('toggle',()=>repairToggle.setAttribute('aria-expanded',String(repair.open)));
  }else repairToggle.hidden=true;
  card.append(avatar,body);refreshSpeakerRecognition(card,segment);refreshReadingTurnView(card,segment);
  card.dataset.meetingId=selected.id;savedPassageBindings.set(card,segment);
  // Defer a changed focused row until focus leaves, rather than losing its caret.
  card.addEventListener('focusout',()=>queueMicrotask(()=>{
    if(card.isConnected && !card.contains(document.activeElement) && card.dataset.stale==='true')renderSegments();
  }));
  return card;
}
function renderSegments() {
  renderSpeakerList();
  if(isLive() || isProgressiveImport() || hasPassageDrafts() || hasRecoverablePassageDrafts() || hasPendingPassageSaves() || doc.segments.some(segment=>segment.refinement_state==='provisional') || ['waiting','refining'].includes(selected.refinement_status)){renderLiveSegments();return;}
  $('pending-phrases').hidden=true;
  const pane=$('transcript-pane'),host=$('segments'),previousScroll=pane.scrollTop;
  const inspected=$('inspector').hidden?null:host.querySelector('.segment.active')?.dataset.segmentId;
  const existing=new Map(Array.from(host.children).filter(card=>savedPassageBindings.has(card) && card.dataset.meetingId===selected.id).map(card=>[card.dataset.segmentId,card]));
  const retained=new Set();let cursor=host.firstChild;
  const query=$('search').value.toLowerCase();let visible=0;
  for(let index=0;index<doc.segments.length;index++) {
    const segment=doc.segments[index];
    if(!showTranscriptPassage(segment))continue;
    if(query && !passageSearchText(segment).toLowerCase().includes(query))continue;
    visible++;const signature=savedPassageSignature(segment);let card=existing.get(segment.id);
    const focused=card?.contains(document.activeElement);
    if(!card || (card.dataset.signature!==signature && !focused)) {
      const next=savedCard(segment);next.dataset.signature=signature;
      if(card){if(cursor===card)cursor=next;card.replaceWith(next);}card=next;
    }else {
      // Existing controls close over this stable binding. Move fresh fields into
      // it when a poll/save clones the document, so later edits reach that copy.
      const binding=savedPassageBindings.get(card);
      if(binding!==segment){
        for(const key of Object.keys(binding))if(!Object.hasOwn(segment,key))delete binding[key];
        Object.assign(binding,segment);doc.segments[index]=binding;
      }
      const stale=String(card.dataset.signature!==signature);if(card.dataset.stale!==stale)card.dataset.stale=stale;
    }
    retained.add(card);if(card!==cursor)host.insertBefore(card,cursor);cursor=card.nextSibling;
    refreshAcknowledgedPassageRepair(card,segment);
    refreshReadingTurnView(card,segment);
    refreshPassageSeparation(card,segment);
    const speakerSignature=JSON.stringify([segment.speaker,doc.speakers,selected.speaker_assignments]);
    if(card.dataset.speakerSignature!==speakerSignature) {
      card.dataset.speakerSignature=speakerSignature;card.dataset.speaker=segment.speaker;
      speakerOptions(card.querySelector('.segment-top select'),segment.speaker);
      updateSpeakerAvatar(card.querySelector('.speaker-avatar'),segment.speaker);
      refreshSpeakerRecognition(card,segment);
    }
    setPassageClass(card,'active',segment.id===inspected);
  }
  for(const child of Array.from(host.children))if(!retained.has(child))child.remove();
  groupConsecutivePassages(host);
  fitPassageTexts(host.querySelectorAll('textarea'));
  renderRetainedAudioReview();
  if(inspected && !host.querySelector('.segment.active'))$('inspector').hidden=true;
  renderSearchResults(visible,query);pane.scrollTop=previousScroll;
  renderIcons();refreshPlaybackCards();
}
function refreshPassageLanguageIssue(body,segment) {
  const blocked=languageBlockedPassage(segment),repair=body.querySelector('.passage-repair');
  let message=body.querySelector('.passage-issue');
  if(blocked) {
    const issue=segment.refinement_issue?.kind==='language'?segment.refinement_issue:null;
    const cause=issue?.reason || segment.language_detection?.reason;
    const explanation=cause==='unsupported'?'The detected language is not supported. If it was detected incorrectly, choose the language spoken here.':cause==='detector_error'?'Language detection failed for this passage. Choose the language spoken here to retry.':cause==='mixed'?'This passage contains language changes that the accuracy pass could not resolve. Choose the language for the part you want to retry.':'The language could not be confirmed for this passage. Choose it below to transcribe the saved audio.';
    if(!message) {
      message=node('p',undefined,'passage-review passage-issue');body.insertBefore(message,repair);
      if(repair)repair.open=true;
    }
    message.textContent=explanation+(segment.text.trim()?' Your current words are kept.':'');
  }else if(segment.refinement_issue?.kind==='accuracy') {
    if(!message){message=node('p',undefined,'passage-review passage-issue');body.insertBefore(message,repair);}
    message.textContent=`Accuracy exception: ${segment.refinement_issue.message} Current words and original audio are kept.`;
  }else message?.remove();
  const summary=repair?.querySelector('summary');
  if(summary)summary.textContent=blocked?'Choose passage language':'Retry transcription';
  return blocked;
}
function appendPassageRepair(body,segment,reason,visibleSource=()=>true) {
  if(!isLive()) {
    const blocked=refreshPassageLanguageIssue(body,segment);
    const repair=node('details',undefined,'passage-repair');repair.open=blocked;repair.append(node('summary',blocked?'Choose passage language':'Retry transcription'));
    const tools=node('div',undefined,'passage-retry'), language=node('select');
    language.setAttribute('aria-label',`Choose language to retry passage at ${time(segment.start)}`);
    const choose=node('option','Choose passage language');choose.value='';language.append(choose);
    for(const [code,name] of Object.entries(config.languages)) {
      if(code==='auto')continue;
      const option=node('option',name);option.value=code;language.append(option);
    }
    language.value=segment.language || '';
    const retry=node('button','Retry transcription');
    const busy=['preparing','processing','queued'].includes(selected.status),retryPending=!!selected.passage_retry || !!selected.passage_separation;
    const drafts=hasPassageDrafts() || hasPendingPassageSaves();
    const tooLong=segment.end-segment.start>24.5;
    retry.disabled=busy || retryPending || tooLong || drafts;language.disabled=busy || retryPending || drafts;
    if(drafts)retry.title='Save or discard passage corrections before retrying.';
    if(tooLong)retry.title='Choose a passage of at most 24.5 seconds before retrying.';
    retry.addEventListener('click',async()=>{
      try {
        if(hasPassageDrafts() || hasPendingPassageSaves())throw new Error('Save or discard passage corrections before retrying.');
        await flushSave();
        if(hasPassageDrafts() || hasPendingPassageSaves())throw new Error('Save or discard passage corrections before retrying.');
        if(!language.value)throw new Error('Choose the language spoken in this passage.');
        const jid=selected.id,generation=++selectionGeneration;
        retrying=true;retry.disabled=true;setStatus();
        const next=await api(`/api/jobs/${jid}/retry`,{method:'POST',body:JSON.stringify({revision:selected.revision,segment_id:segment.id,language:language.value})});
        if(selected?.id!==jid || generation!==selectionGeneration)return;
        selected=next;doc=structuredClone(next.document);setStatus();renderEditor();
        notice('Retrying this passage locally. Existing text is retained.');
      } catch(error){retry.disabled=false;notice(error.message,true);}
      finally{retrying=false;if(selected)setStatus();}
    });
    tools.append(language,retry);repair.append(tools);body.append(repair);
    if(blocked && drafts)repair.append(node('p','Save or discard your passage corrections before choosing a language.','passage-review'));
    if(blocked && tooLong) {
      repair.append(node('p','A passage retry can cover up to 24.5 seconds. Adjust this passage’s timing to the words you want to retry, or enter a correction directly.','passage-review'));
      const timing=node('button','Adjust passage timing','text-button');timing.disabled=busy || retryPending || drafts || !visibleSource();
      timing.addEventListener('click',()=>{
        const canAdjust=()=>!isLive() && !hasPassageDrafts() && !hasPendingPassageSaves() && !selected.passage_retry && !selected.passage_separation && !['preparing','processing','queued'].includes(selected.status) && doc.segments.find(row=>row.id===segment.id)===segment && visibleSource();
        if(!canAdjust())return;
        const card=body.closest('.segment');if(!card)return;
        showInspector(segment,card,true,canAdjust);$('segment-details').querySelector('input')?.focus();
      });repair.append(timing);
    }
    if(selected.passage_retry?.segment_id===segment.id) {
      const progress=node('p','Retrying this passage locally…','passage-review');progress.setAttribute('role','status');tools.append(progress);
      const cancel=node('button','Cancel passage retry');
      cancel.addEventListener('click',async()=>{
        try {
          await flushSave();
          const jid=selected.id,generation=++selectionGeneration;retrying=true;cancel.disabled=true;setStatus();
          const next=await api(`/api/jobs/${jid}/cancel`,{method:'POST'});
          if(selected?.id!==jid || generation!==selectionGeneration)return;
          selected=next;doc=structuredClone(next.document);setStatus();renderEditor();
        } catch(error){cancel.disabled=false;notice(error.message,true);}
        finally{retrying=false;if(selected)setStatus();}
      });
      tools.append(cancel);
    }
    if(reason && segment.text.trim()) {
      const reviewed=node('button','Mark words reviewed');reviewed.disabled=busy || drafts || !visibleSource();
      reviewed.addEventListener('click',()=>{if(hasPassageDrafts() || hasPendingPassageSaves() || !visibleSource())return;segment.review_resolution='words_reviewed';segment.review=hasOverlappingSpeakers(segment);changed();renderSegments();});
      tools.append(reviewed);
    }
    const candidate=segment.retry_candidate;
    if(candidate && !savedRetryAccepted(segment)) {
      const result=node('div',undefined,'retry-result');
      result.append(node('p',candidate.error || `Retry result (${config.languages[candidate.language]}). Listen before choosing this text.`));
      if(candidate.text) {
        result.append(node('p',candidate.text,'segment-text'));
        if(candidate.transcription_review)result.append(node('p',reviewReasons[candidate.transcription_review.reason] || 'Retry needs review','passage-review'));
        const currentCandidate=()=>candidate.start===segment.start && candidate.end===segment.end && candidate.speaker===segment.speaker;
        const use=node('button','Use this text');use.disabled=busy || drafts || !currentCandidate() || !visibleSource();
        use.addEventListener('click',()=>{
          if(hasPassageDrafts() || hasPendingPassageSaves() || !visibleSource())return;
          if(!currentCandidate()){notice('This retry belongs to earlier passage timing or a different speaker. Save and retry the current passage.',true);return;}
          candidate.prior_language_detection=segment.language_detection;
          segment.text=candidate.text;segment.language=candidate.language;
          delete segment.review_resolution;
          segment.language_detection={mode:'manual',reason:'passage_retry'};
          segment.transcription_review=candidate.transcription_review || null;
          segment.review=true;candidate.accepted_at=Date.now()/1000;
          changed();renderSegments();notice('Retry text selected.');
        });
        result.append(use);
      } else if(!candidate.error)result.append(node('p','No transcript text returned. Original audio and existing words are retained.','passage-review'));
      body.append(result);
    }
  }
}

function appendPassageHistory(host,segment) {
  if(!segment.refinement_window)return;
  const jid=selected.id,prior=node('button','Compare prior words','text-button');prior.type='button';
  prior.addEventListener('click',async()=>{
    try {
      const history=await api(`/api/jobs/${jid}/refinement/revisions/${encodeURIComponent(segment.refinement_window)}`);
      if(selected?.id!==jid || !host.isConnected || !host.contains(prior))return;
      const previous=history.segments.filter(row=>row.start<segment.end && row.end>segment.start).map(row=>row.text).filter(Boolean).join('\n');
      const compared=structuredClone(doc.segments.find(row=>row.id===segment.id));
      const original=history.segments.find(row=>row.id===segment.id);
      const sameRange=compared && !compared.deleted && original?.start===compared.start && original?.end===compared.end;
      host.append(node('p',previous || 'No previous legible words.','prior-words'));
      if(previous && sameRange) {
        const restore=node('button','Restore prior words','text-button');restore.type='button';
        restore.addEventListener('click',async()=>{
          const current=doc.segments.find(row=>row.id===segment.id);if(!current || selected?.id!==jid)return;
          if(current.deleted || JSON.stringify(current)!==JSON.stringify(compared)){notice('This passage changed since comparison. Compare its current words again before restoring.',true);return;}
          restore.disabled=true;
          try {await savePassageDetails(current,{text:original.text});notice('Prior words restored and protected from accuracy changes.');}
          catch(error){notice(error.message,true);restore.disabled=false;}
        });host.append(restore);
      }else if(previous)host.append(node('p','The passage timing changed. Copy prior words to correct the current range.','passage-review'));
      prior.remove();
    }catch(error){notice(error.message,true);}
  });host.append(prior);
}

const reviewReasons={insufficient_acoustic_context:'Too little acoustic context to decode; audio retained',short_acoustic_context:'Unconfirmed words from a short audio fragment; review the recording',possible_non_speech:'Audio may contain no speech; model words need review',uncertain:'Language uncertain; audio retained',change_pending:'Language change uncertain; audio retained',unsupported:'Detected language unsupported; audio retained',insufficient_speech:'Too little usable speech; audio retained',overlapping_speech:'Overlapping speakers; voices are not separated',refinement_incomplete:'Larger-context result was incomplete; previous words and audio retained',empty_result:'No transcript text returned; audio retained',transcription_failed:'Transcription failed for this passage; audio retained',token_limit:'Transcript may be incomplete; audio retained',unassigned_audio:'Audio outside detected speech; may be silence or missed speech'};
function voiceEstimateLabel(channel) { return channel===0?'Voice A':'Voice B'; }
function separationSample(seconds) {
  const value=seconds*16000,lower=Math.floor(value);
  return value-lower===.5?lower+lower%2:Math.round(value);
}
function separationUnavailable(segment) {
  if(selected.status!=='ready' || segment.separation)return 'Separate an original passage after the recording is ready.';
  if(!Number.isFinite(segment.start) || !Number.isFinite(segment.end) || segment.start<0 || segment.end<=segment.start || segment.end-segment.start>24.5)return 'Choose a passage of at most 24.5 seconds.';
  const start=separationSample(segment.start),end=separationSample(segment.end),activity=segment.speaker_activity;
  if(activity?.audio_revision!==segment.audio_revision || !Number.isSafeInteger(activity?.observed_end_sample) || activity.observed_end_sample<end || !Array.isArray(activity.regions))return 'This passage has no current two-voice overlap observation.';
  const tracks=new Set();let overlap=false;
  for(const region of activity.regions) {
    if(!Number.isSafeInteger(region?.start_sample) || !Number.isSafeInteger(region?.end_sample) || region.start_sample<0 || region.end_sample<=region.start_sample || !Array.isArray(region.speakers) || region.speakers.some(s=>typeof s!=='string'))return 'This passage has invalid audio observations.';
    if(region.end_sample<=start || region.start_sample>=end)continue;
    const voices=new Set(region.speakers);overlap ||= voices.size===2;for(const voice of voices)tracks.add(voice);
  }
  if(tracks.size!==2 || !overlap)return 'Separation needs an observed overlap between exactly two voices.';
  if(Array.isArray(selected.language_history) && selected.language_history.some(epoch=>epoch.start_sample>start && epoch.start_sample<end))return 'Choose a passage inside one language setting.';
  return '';
}
async function separationRequest(path,body) {
  const jid=selected.id,generation=selectionGeneration,cancelling=path==='/cancel';
  if(!cancelling) {
    if(hasPassageDrafts() || hasPendingPassageSaves())throw new Error('Save or discard passage drafts before separating or replacing a passage.');
    await flushSave();
  }
  if(selected?.id!==jid || generation!==selectionGeneration)throw new Error('The selected recording changed. Try again on its current passage.');
  separationRequestMessage={separation:'Starting voice separation…',cancel:'Stopping voice separation…',accept:'Saving separated voices…',discard:'Keeping the original passage…',restore:'Restoring the original passage…'}[path.split('/').pop()] || 'Starting voice separation…';
  retrying=true;const requestGeneration=++selectionGeneration;setStatus();
  try {
    const next=await api(`/api/jobs/${jid}${path}`,{method:'POST',...(body?{body:JSON.stringify(body())}:{})});
    if(selected?.id!==jid || requestGeneration!==selectionGeneration || next.revision<selected.revision)return;
    selected=next;
    if(!cancelling || (!dirty && !hasPassageDrafts() && !hasPendingPassageSaves()))doc=structuredClone(next.document);
    setStatus();renderEditor();
  }finally{retrying=false;separationRequestMessage='';if(selected)setStatus();}
}
function playOriginalPassage(segment,episode=null) {
  const audio=$('player');
  for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause();
  passageStart=segment.start;passageEnd=segment.end;
  const path=episode?`/api/jobs/${selected.id}/separation/${episode}/original/audio`:`/api/jobs/${selected.id}/passages/${encodeURIComponent(segment.id)}/audio`;
  if(audio.getAttribute('src')!==path)audio.setAttribute('src',path);
  audio.currentTime=segment.start;audio.play().catch(e=>notice(e.message,true));
}
function playSeparatedPassage(card,segment) {
  const audio=card.querySelector('.separated-player');
  audio.currentTime=Math.max(0,segment.start-segment.separation.origin_sample/16000);
  audio.play().catch(error=>notice(error.message,true));
}
function separationAudio(path,start,end,label) {
  const audio=node('audio');audio.controls=true;audio.preload='none';audio.src=path;
  audio.setAttribute('aria-label',label);
  audio.addEventListener('loadedmetadata',()=>{audio.currentTime=start;});
  audio.addEventListener('timeupdate',()=>{if(audio.currentTime>=end)audio.pause();});
  audio.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause();});
  audio.addEventListener('error',()=>notice(`${label} unavailable. The original passage is retained.`,true));
  return audio;
}
function separationPlayback(path,start,end,label) {
  const group=node('div',undefined,'separation-playback'),audio=separationAudio(path,start,end,label);
  audio.controls=false;audio.hidden=true;
  group.append(separationPlayButton(audio,start,end,label),audio);return group;
}
function separationPlayButton(audio,start,end,label) {
  const play=node('button',`Play ${label}`,'quiet');
  play.addEventListener('click',()=>{
    if(!audio.paused){audio.pause();return;}
    if(audio.currentTime<start || audio.currentTime>=end || audio.ended)audio.currentTime=start;
    audio.play().catch(error=>notice(error.message,true));
  });
  for(const event of ['play','pause','ended'])audio.addEventListener(event,()=>{play.textContent=`${audio.paused?'Play':'Pause'} ${label}`;});
  return play;
}
function appendPassageSeparation(body,segment) {
  const panel=node('div',undefined,'passage-separation');body.append(panel);
  renderPassageSeparation(panel,segment);
}
function refreshPassageSeparation(card,segment) {
  const panel=card.querySelector('.passage-separation');
  if(panel && !panel.contains(document.activeElement))renderPassageSeparation(panel,segment);
}
function renderPassageSeparation(panel,segment) {
  const signature=JSON.stringify([segment,selected.status,selected.passage_separation,selected.passage_retry,selected.separation_error,setupState?.separation]);
  if(panel.dataset.signature===signature)return;
  panel.dataset.signature=signature;panel.replaceChildren();
  if(isLive() && !segment.separation_candidate) {
    if(segment.separation)panel.append(node('p',`${voiceEstimateLabel(segment.separation.channel)} · Check separated voices`,'separation-hint'));
    return;
  }
  if(segment.separation) {
    const voice=segment.separation,origin=voice.origin_sample/16000;
    const audio=separationAudio(`/api/jobs/${selected.id}/separation/${voice.episode_id}/${voice.channel}/audio`,Math.max(0,segment.start-origin),segment.end-origin,`${voiceEstimateLabel(voice.channel)} separated audio`);
    audio.className='separated-player';audio.controls=false;audio.hidden=true;panel.append(audio);
    panel.append(separationPlayButton(audio,Math.max(0,segment.start-origin),segment.end-origin,voiceEstimateLabel(voice.channel)));
    if(segment.review_resolution!=='words_reviewed')panel.append(node('p','Check separated voices','separation-hint'));
    const historyPanel=node('details',undefined,'separation-history');historyPanel.append(node('summary','Compare with original'));
    const original=node('button','Play original recording','text-button');
    original.addEventListener('click',()=>playOriginalPassage({start:origin,end:origin+voice.length_samples/16000},voice.episode_id));historyPanel.append(original);panel.append(historyPanel);
    if(voice.channel===0) {
      const prior=node('button','Show original passage','text-button');
      prior.addEventListener('click',async()=>{
        prior.disabled=true;
        try {
          const jid=selected.id,generation=selectionGeneration;
          const history=await api(`/api/jobs/${jid}/separation/${voice.episode_id}/history`);
          if(selected?.id!==jid || generation!==selectionGeneration || !panel.isConnected)return;
          const saved=node('div',undefined,'separation-original'),text=node('textarea');text.readOnly=true;text.value=history.original.text;text.setAttribute('aria-label','Original passage for copying');
          saved.append(node('strong','Original passage'),text);
          const unchanged=history.accepted.every(prior=>{
            const current=doc.segments.find(row=>row.id===prior.id);
            return current && ['text','speaker','start','end','machine_revision','protected_fields','language','language_detection'].every(key=>JSON.stringify(current[key]??(key==='protected_fields'?[]:null))===JSON.stringify(prior[key]??(key==='protected_fields'?[]:null)));
          });
          if(unchanged) {
            const restore=node('button','Use original transcript and audio');restore.addEventListener('click',async()=>{try{await separationRequest(`/separation/${voice.episode_id}/restore`);}catch(error){notice(error.message,true);}});saved.append(restore);
          }else saved.append(node('p','A separated passage was edited. Copy the original words above to keep those edits.','passage-review'));
          historyPanel.append(saved);prior.remove();fitPassageText(text);
        }catch(error){prior.disabled=false;notice(error.message,true);}
      });historyPanel.append(prior);
    }
    return;
  }
  const candidate=segment.separation_candidate;
  if(candidate) {
    const result=node('details',undefined,'separation-comparison');result.append(node('summary','Compare voice estimates'));panel.append(result);
    result.append(node('strong','Separated voice estimates · check both'));
    result.append(node('p','A saved separation result is available. The original passage is retained until you choose this result.','passage-review'));
    const original=node('div',undefined,'separation-original'),text=node('textarea');text.readOnly=true;text.value=candidate.original.text;text.setAttribute('aria-label','Original passage words');
    original.append(node('strong','Original passage'),text,separationPlayback(`/api/jobs/${selected.id}/passages/${encodeURIComponent(segment.id)}/audio`,candidate.original.start,candidate.original.end,'Original recording'));result.append(original);
    candidate.voices.forEach((voice,channel)=>{
      const row=node('div',undefined,'separation-estimate');row.append(node('strong',voiceEstimateLabel(channel)));
      row.append(node('p',voice.text || 'No supported words returned. Keep the original and review this audio.','segment-text'));
      row.append(node('small',config.languages[voice.language] || 'Language needs review'));
      if(voice.review || voice.transcription_review)row.append(node('p',voice.transcription_review?.reason==='overlapping_speech'?'Speech may still overlap in this estimate.':reviewReasons[voice.transcription_review?.reason] || 'Words need review.','passage-review'));
      row.append(separationPlayback(`/api/jobs/${selected.id}/separation/${candidate.id}/${channel}/audio`,0,candidate.original.end-candidate.original.start,voiceEstimateLabel(channel)));result.append(row);
    });
    const actions=node('div',undefined,'separation-actions'),keep=node('button','Keep original'),use=node('button','Use both voices','primary');
    const current=['text','speaker','start','end','language'].every(key=>JSON.stringify(segment[key]??null)===JSON.stringify(candidate.original[key]??null));
    use.disabled=isLive() || selected.status!=='ready' || !current || candidate.voices.some(voice=>!voice.text?.trim() || !Object.hasOwn(config.languages,voice.language) || voice.language==='auto');
    if(!current)result.append(node('p','This passage changed. Keep the original and separate its current words and audio again.','passage-review'));
    for(const [button,action] of [[keep,'discard'],[use,'accept']])button.addEventListener('click',async()=>{try{await separationRequest(`/separation/${candidate.id}/${action}`);}catch(error){notice(error.message,true);}});
    if(isLive()){keep.disabled=true;result.append(node('p','Review these estimates after recording stops. The original speaker turns stay in your transcript.','passage-review'));}
    actions.append(keep,use);result.append(actions);return;
  }
  if(selected.passage_separation?.segment_id===segment.id) {
    const status=node('p',selected.passage_separation.phase || 'Separating voices…','passage-review');status.setAttribute('role','status');
    const cancel=node('button','Cancel separation');cancel.addEventListener('click',async()=>{try{await separationRequest('/cancel');}catch(error){notice(error.message,true);}});panel.append(status,cancel);return;
  }
  const start=separationSample(segment.start),end=separationSample(segment.end);
  const regions=segment.speaker_activity?.regions;
  if(!Array.isArray(regions) || !regions.some(region=>region?.end_sample>start && region.start_sample<end && Array.isArray(region.speakers) && new Set(region.speakers).size>1))return;
  const reason=separationUnavailable(segment),action=node('button','Retry voice separation');
  action.disabled=!!reason || !!selected.passage_retry || !!selected.passage_separation || setupState?.separation?.active===true;
  action.title=reason;action.addEventListener('click',async()=>{
    try {
      if(!setupState?.separation?.installed){openSettings('models');throw new Error('Download the optional voice separation model in Settings → Models first.');}
      await separationRequest('/separation',()=>({revision:selected.revision,segment_id:segment.id}));
    }catch(error){notice(error.message,true);}
  });
  const recovery=node('details');recovery.append(node('summary','Voice separation'));
  if(reason || selected.separation_error)recovery.append(node('p',reason || selected.separation_error,'passage-review'));
  recovery.append(action);panel.append(recovery);
}
function reviewSampleTime(sample) {
  const units=Math.round(sample*10000/16000),minutes=Math.floor(units/600000),rest=units%600000;
  return `${String(minutes).padStart(2,'0')}:${String(Math.floor(rest/10000)).padStart(2,'0')}.${String(rest%10000).padStart(4,'0')}`;
}
function appendUncoveredAudio(body,segment) {
  const gaps=segment.transcription_review?.uncovered_audio;
  if(!gaps?.length)return;
  const ranges=gaps.map(g=>`${reviewSampleTime(g.start_sample)}–${reviewSampleTime(g.end_sample)} (${Number(((g.end_sample-g.start_sample)/16).toFixed(4))} ms)`);
  body.append(node('p',`Audio to review: ${ranges.join('; ')}. Current words and original audio retained.`,'review-audio-ranges'));
}
function passageReviewReason(segment) {
  if(segment.separation)return segment.review_resolution==='words_reviewed'?'':'Check separated voices';
  if(segment.review_resolution==='words_reviewed' && segment.text.trim())return hasOverlappingSpeakers(segment) ? reviewReasons.overlapping_speech : segment.speaker==='unassigned' ? 'Speaker uncertain' : '';
  if(segment.transcription_review?.reason==='refinement_conflict')return 'Previous words retained because the new passage also covers another passage needing review.';
  if(['transcription_failed','empty_result','token_limit'].includes(segment.transcription_review?.reason))return reviewReasons[segment.transcription_review.reason];
  if(hasOverlappingSpeakers(segment))return reviewReasons.overlapping_speech;
  if(segment.language_detection?.reason==='mixed')return 'Multiple languages detected; review the passage';
  if(segment.transcription_review?.reason)return reviewReasons[segment.transcription_review.reason] || 'Passage needs review; audio retained';
  if(segment.language_detection?.reason==='recent_context')return 'Language inferred from recent meeting context';
  if(segment.language_detection?.reason==='best_effort')return 'Best-effort words; language needs review';
  if(segment.language_detection?.reason==='needs_language')return 'Choose the meeting language; audio retained';
  if(segment.language_detection?.mode==='auto' && !segment.language)return reviewReasons[segment.language_detection.reason] || reviewReasons.uncertain;
  if(segment.speaker==='unassigned' && segment.text.trim())return 'Speaker uncertain';
  return segment.text.trim() ? '' : reviewReasons.empty_result;
}
function appendPassageEvidence(host,segment) {
  const reason=passageReviewReason(segment);
  if(reason)host.append(node('p',reason,'passage-review'));
  appendUncoveredAudio(host,segment);
  const evidence=node('details',undefined,'passage-evidence');evidence.append(node('summary','Technical diagnostics'));
  const metadata={};
  for(const key of ['review','review_resolution','refinement_state','language','language_detection','confidence','speaker_candidates','speaker_track_mapping','voice_eligible','audio_state','transcription_review','audio_anchor','timing','provenance','refinement_window','machine_revision']) {
    if(segment[key]!==undefined)metadata[key]=segment[key];
  }
  if(doc.provenance)metadata.recording_provenance=doc.provenance;
  const text=node('pre',JSON.stringify(metadata,null,2));text.style.whiteSpace='pre-wrap';text.style.overflowWrap='anywhere';
  const copy=node('button','Copy diagnostics','text-button');copy.addEventListener('click',()=>navigator.clipboard.writeText(text.textContent).catch(error=>notice(error.message,true)));
  evidence.append(copy,text);host.append(evidence);
}
function showTranscriptPassage(segment) {
  if(segment.deleted)return false;
  if(segment.refinement_issue?.kind==='accuracy')return true;
  return !!segment.text.trim() || languageBlockedPassage(segment) || !!segment.separation_candidate || selected?.passage_separation?.segment_id===segment.id || passageDrafts.has(segment.id) || !!recoverablePassageDrafts.get(segment.id)?.text.trim() || segment.protected_fields?.includes('text') || segment.refinement_state==='edited';
}
function languageBlockedPassage(segment) {
  if(isLive() || segment.protected_fields?.includes('text') || segment.refinement_state==='edited')return false;
  if(segment.refinement_issue?.kind==='language')return true;
  return !segment.text.trim() && !isEmptyNonSpeechPlaceholder(segment) &&
    ['uncertain','change_pending','needs_language','unsupported','mixed'].includes(segment.language_detection?.reason);
}
function unresolvedLanguagePassages() {
  return selected?.status==='ready' && ['unresolved','complete_with_exceptions'].includes(selected.refinement_status) ? (doc?.segments || []).filter(languageBlockedPassage) : [];
}
function isEmptyNonSpeechPlaceholder(segment) {
  // Presentation only: legacy no-word hints do not establish verified silence
  // or manufacture acoustic receipts. Keep words, candidates and failures.
  const transcription=segment.transcription_review;
  if(segment.text.trim() || transcription?.candidate_text?.trim() || transcription?.partial_text)return false;
  if(transcription?.reason && transcription.reason!=='insufficient_speech')return false;
  if(['speech','uncertain','pending'].includes(segment.acoustic_evidence?.decision) ||
     ['speech_evidence_pending','insufficient_acoustic_context'].includes(segment.audio_state))return false;
  return ['digital_silence','model_non_speech','insufficient_speech'].includes(segment.audio_state) ||
    segment.language_detection?.reason==='insufficient_speech' || transcription?.reason==='insufficient_speech';
}
function renderRetainedAudioReview() {
  const host=$('retained-audio-review'),open=host.open;
  if(host.contains(document.activeElement))return;
  const rows=doc.segments.filter(segment=>!segment.deleted && !showTranscriptPassage(segment) && !isEmptyNonSpeechPlaceholder(segment) &&
    (segment.transcription_review?.reason || ['unsupported','uncertain','change_pending','needs_language','insufficient_speech','mixed'].includes(segment.language_detection?.reason) ||
     (segment.refinement_state==='unresolved' && segment.language_detection?.reason!=='silence')));
  host.hidden=!rows.length;host.replaceChildren();if(!rows.length)return;
  host.append(node('summary','Audio details'));
  const list=node('div',undefined,'retained-audio-list');
  for(const segment of rows) {
    const row=node('div',undefined,'retained-audio-range');row.dataset.segmentId=segment.id;
    row.append(node('p',`${passageTime(segment.start)}–${passageTime(segment.end)} · ${passageReviewReason(segment)}`));
    if(segment.transcription_review?.candidate_text) {
      const candidate=node('details'),label=node('summary','Unconfirmed model words'),text=node('textarea');
      text.readOnly=true;text.value=segment.transcription_review.candidate_text;text.setAttribute('aria-label','Unconfirmed model words');
      candidate.append(label,text);row.append(candidate);
    }
    if(!isLive()) {
      const play=node('button','Play retained audio','text-button');
      play.addEventListener('click',()=>{passageEnd=segment.end;$('player').currentTime=segment.start;$('player').play().catch(e=>notice(e.message,true));});
      row.append(play);appendPassageRepair(row,segment,passageReviewReason(segment));appendPassageSeparation(row,segment);
    }
    list.append(row);
  }
  host.append(list);host.open=open;
}
// Group presentation only: each passage keeps its own edit, timing, retry and
// playback targets. Recompute after filtering/reconciliation without merging IDs.
function groupConsecutivePassages(host) {
  let previous,previousSection;
  for(const card of host.children) {
    const section=card.dataset.transcriptSection || 'transcript',start=section!==previousSection;
    const continuation=!start && previous && !card.dataset.separationEpisode && !previous.dataset.separationEpisode && !['unassigned','multiple_speakers'].includes(card.dataset.speaker) && speakerIdentity(previous.dataset.speaker)===speakerIdentity(card.dataset.speaker);
    setPassageClass(card,'speaker-continuation',!!continuation);
    setPassageClass(card,'passage-exception',!!card.querySelector('.review-tag:not(.routine-state),.refinement-badge:not(.routine-state)'));
    setPassageClass(card,'retained-correction',section==='corrections');
    const correctionStart=section==='corrections' && start;
    setPassageClass(card,'transcript-section-start',correctionStart);
    let label=card.querySelector('.corrections-section-label');
    if(label && !correctionStart)label.remove();
    if(correctionStart && !label) {
      label=node('span','Your corrections · replaced passages','transcript-section-label corrections-section-label');
      label.setAttribute('role','heading');label.setAttribute('aria-level','2');
      card.insertBefore(label,card.firstChild);
    }
    previous=card;previousSection=section;
  }
}
function setPassageClass(card,name,value) {
  if(card.classList.contains(name)!==!!value)card.classList.toggle(name,!!value);
}
function fitPassageTexts(texts) {
  const changed=[];
  // Finish all width/style reads before resetting heights, and all height reads
  // before applying results. Cached rows do not enter the write phases.
  for(const text of texts){
    const width=text.clientWidth;if(!width)continue;
    const key=JSON.stringify([width,getComputedStyle(text).paddingRight,text.value]);
    if(text.dataset.fit!==key)changed.push({text,key});
  }
  for(const {text} of changed)text.style.height='auto';
  const heights=changed.map(({text})=>text.scrollHeight);
  changed.forEach(({text,key},index)=>{text.style.height=`${heights[index]}px`;text.dataset.fit=key;});
}
function fitPassageText(text) { fitPassageTexts([text]); }
function resetPlayback() {
  for(const card of playingCards)setPassageClass(card,'playing',false);
  playingCards.clear();playbackCards.clear();playbackSegments.clear();playbackRows=[];playbackEnds=[];
  playbackDocument=null;playbackIndexDirty=true;
}
function refreshPlaybackCards() {
  playbackCards=new Map(Array.from($('segments').children).map(card=>[card.dataset.segmentId,card]));
  playbackIndexDirty=true;
  if(playingCards.size)updatePlaybackHighlight();
}
function updatePlaybackHighlight() {
  if(!doc){resetPlayback();return;}
  if(playbackIndexDirty || playbackDocument!==doc){
    playbackDocument=doc;playbackIndexDirty=false;
    playbackSegments=new Map(doc.segments.map(segment=>[segment.id,segment]));
    playbackRows=Array.from(playbackSegments.values()).sort((a,b)=>a.start-b.start);
    let end=-Infinity;playbackEnds=playbackRows.map(segment=>(end=Math.max(end,segment.end)));
  }
  const now=$('player').currentTime;let low=0,high=playbackRows.length;
  while(low<high){const middle=(low+high)>>>1;if(playbackRows[middle].start<=now)low=middle+1;else high=middle;}
  const active=new Set();
  // Prefix maximum ends retain overlapping/nested passages on backward seeks.
  for(let index=low-1;index>=0 && playbackEnds[index]>now;index--){
    const segment=playbackSegments.get(playbackRows[index].id),card=playbackCards.get(segment.id);
    if(card && segment.end>now)active.add(card);
  }
  for(const card of playingCards)if(!active.has(card))setPassageClass(card,'playing',false);
  for(const card of active)if(!playingCards.has(card))setPassageClass(card,'playing',true);
  playingCards=active;
}
function icon(name) { const element=node('i');element.dataset.lucide=name;return element; }
function showInspector(segment,card,force=false,visibleSource=()=>true,commit=null) {
  document.querySelectorAll('.segment.active').forEach(item => item.classList.remove('active'));
  card.classList.add('active');
  if(narrowLayout() && !force && $('inspector').hidden)return;
  $('inspector').hidden=false;
  const details=$('segment-details');details.replaceChildren();
  if(segment.speaker!=='unassigned' && segment.speaker!=='multiple_speakers') {
    const name=node('button','Name / remember voice','text-button passage-name-control');name.type='button';
    name.addEventListener('click',()=>nameSpeaker(card.dataset.speaker));details.append(name);
  }
  for(const [key,label] of [['start','Start time (seconds)'],['end','End time (seconds)']]) {
    const field=node('label',label,'segment-detail-field'), input=node('input');
    input.type='number';input.min=0;input.max=selected.duration;input.step='.01';input.value=segment[key].toFixed(2);
    const hint=node('small',passageTime(segment[key]),'time-hint');
    const draftKey=`${selected.id}:${segment.id}:${key}`;
    if(timingDrafts.has(draftKey)){input.value=timingDrafts.get(draftKey);input.setAttribute('aria-invalid','true');hint.textContent='Enter ordered times within this recording.';}
    input.addEventListener('change',() => {
      if(!visibleSource()){notice('Finish pending passage edits or retries, then open the current passage’s timing again.',true);return;}
      const value=input.value.trim()===''?NaN:Number(input.value),start=key==='start'?value:segment.start,end=key==='end'?value:segment.end;
      if(!Number.isFinite(value) || start<0 || end>selected.duration || start>=end){timingDrafts.set(draftKey,input.value);input.setAttribute('aria-invalid','true');hint.textContent='Start must be before end, within this recording.';renderSaveState();return;}
      timingDrafts.delete(draftKey);input.removeAttribute('aria-invalid');
      if(commit){commit({[key]:value}).catch(error=>{hint.textContent=error.message;input.value=segment[key].toFixed(2);});return;}
      segment[key]=value;segment.timing='user_edited';hint.textContent=passageTime(segment[key]);changed();renderSegments();
    });
    const reset=node('button','Reset','text-button');reset.type='button';reset.addEventListener('click',()=>{timingDrafts.delete(draftKey);input.value=segment[key].toFixed(2);input.removeAttribute('aria-invalid');hint.textContent=passageTime(segment[key]);renderSaveState();});
    field.append(input,hint,reset);details.append(field);
  }
  appendPassageHistory(details,segment);
  appendPassageEvidence(details,segment);
  if(segment.review && markReviewable(segment)) {
    const reviewed=node('button','Mark words reviewed','text-button review-action');
    reviewed.addEventListener('click',()=>{
      segment.review_resolution='words_reviewed';segment.review=false;changed();renderSegments();setStatus();
      const currentCard=document.querySelector(`#segments [data-segment-id="${CSS.escape(segment.id)}"]`);
      if(currentCard)showInspector(segment,currentCard,true);
    });details.append(reviewed);
  }
}
async function save({quiet = false} = {}) {
  if (!dirty) return;
  if (saving) throw new Error('A save is in progress. Try again when it finishes.');
  clearTimeout(autosaveTimer); autosaveTimer = null;
  if (undoRemoval) { clearTimeout(undoRemoval.timer); undoRemoval = null; if (document.querySelector('#notice .notice-action')) notice(''); }
  const generation = editGeneration, jid = selected.id,base=selected,submitted=structuredClone(doc);
  saving = true; renderSaveState();
  try {
    const put=revision=>api(`/api/jobs/${jid}/transcript`, {method: 'PUT', updateSave: true, body: JSON.stringify({revision, document: submitted})});
    let result;
    try {result=await put(base.revision);}
    catch(error) {
      if(error.status!==409)throw error;
      const latest=await api(`/api/jobs/${jid}`);
      // Server diagnostics and candidates may advance revision; authoritative edits still conflict.
      const withoutServerResults=document=>{
        const copy=structuredClone(document);for(const segment of copy?.segments || []){delete segment.retry_candidate;delete segment.separation_candidate;delete segment.refinement_issue;}
        return JSON.stringify(copy);
      };
      if(selected?.id!==jid || latest.status!==base.status || latest.duration!==base.duration ||
        withoutServerResults(latest.document)!==withoutServerResults(base.document))throw error;
      result=await put(latest.revision);
    }
    if (selected?.id !== jid) return;
    selected = result;
    const savedSegments=new Map(result.document.segments.map(segment=>[segment.id,segment]));
    for(const segment of doc.segments) {
      const saved=savedSegments.get(segment.id);
      if(!saved)continue;
      if(saved.refinement_issue)segment.refinement_issue=structuredClone(saved.refinement_issue);
      else delete segment.refinement_issue;
      if(saved.start===segment.start && saved.end===segment.end && saved.speaker===segment.speaker) {
        if(saved.retry_candidate)segment.retry_candidate=structuredClone(saved.retry_candidate);
        else delete segment.retry_candidate;
        if(saved.separation_candidate)segment.separation_candidate=structuredClone(saved.separation_candidate);
        else delete segment.separation_candidate;
      }
      refreshAcknowledgedPassageRepair(playbackCards.get(segment.id),segment,saved);
    }
    if (generation === editGeneration) {
      dirty = false;
      // Keep the live document while someone is typing so the caret and the
      // passage handlers stay attached; adopt the saved copy otherwise.
      if (!$('editor').contains(document.activeElement)) { doc = structuredClone(result.document); renderEditor(); }
    }
    if($('editor').contains(document.activeElement))renderSegments();
    setStatus();
    if (!quiet) notice('Saved on this Mac.');
  } finally { saving = false; renderSaveState(); if (dirty) scheduleAutosave(); }
}
async function upload(files) {
  if (!files.length) return;
  if (!(await saveBeforeLeaving())) return;
  const generation=selectionGeneration;
  if(changingLiveLanguage || changingDefaultLanguage)throw new Error('Wait for the language change to finish before importing.');
  const data = new FormData(); Array.from(files).forEach(file => data.append('files', file)); data.append('language', config.default_language || 'auto');
  $('files').disabled = true; notice('Uploading to the local app…');
  try {
    const created = await api('/api/jobs', {method: 'POST', body: data});
    jobs = await api('/api/jobs');renderJobs();if(selectionGeneration===generation)await select(created[0].id);
    notice(`${created.length} recording${created.length === 1 ? '' : 's'} stored locally. Preparing playback audio…`);
  } finally { $('files').disabled = false; $('files').value = ''; }
}
function manual() {
  doc = {schema_version: 1, speakers: {'speaker_0': 'Speaker 1'}, segments: [], provenance: {kind: 'manual', timing: 'User supplied recording times'}, warnings: ['Manual transcript. Enter text and review timing against the recording.']};
  changed(); setStatus(); renderEditor();
}
function usableActionControl(id) {
  const control = $(id);
  for(let element=control;element;element=element.parentElement) {
    if(element.hidden || element.inert || element.disabled)return false;
  }
  return true;
}
async function startNewMeeting() {
  const blocked=()=>updateFrozen || !usableActionControl('new-meeting') || newMeetingOpening ||
    liveStatuses.includes(meeting?.status) || isLive() || identityBusy || retrying || hasPendingPassageSaves() || !!undoRemoval;
  if(blocked())return false;
  newMeetingOpening=true;
  try {
    if(!(await saveBeforeLeaving()))return false;
    // Recording and focus state may have changed while an autosave was pending.
    if(updateFrozen || !$('setup').hidden || document.querySelector('dialog[open]') ||
      liveStatuses.includes(meeting?.status) || isLive() || identityBusy || retrying || hasPendingPassageSaves() || undoRemoval)return false;
    if((hasPassageDrafts() || hasRecoverablePassageDrafts()) && !confirm('Discard unsaved passage drafts?'))return false;
    passageDrafts.clear();recoverablePassageDrafts.clear();selected=doc=null;resetPlayback();dirty=false;selectionGeneration++;$('player').pause();$('player').removeAttribute('src');
    $('workspace').hidden=true;$('empty').hidden=false;renderJobs();renderStartState();renderShortcutHints();$('meeting-title').focus();
    return true;
  } finally { newMeetingOpening=false; }
}
// Native menus, keyboard shortcuts and matching buttons use one guarded path.
async function dispatchAction(action) {
  if(updateFrozen || document.querySelector('dialog[open]'))return false;
  try {
    if(action==='settings') { openSettings();return true; }
    if(action==='check-updates') {
      if($('update-check').disabled || updateRequesting || updateState.state==='unavailable')return false;
      openSettings('updates');await requestUpdate('check');return true;
    }
    if(action==='download-update') {
      if(updateState.state!=='available' || updateRequesting || liveStatuses.includes(meeting?.status) || isLive())return false;
      openSettings('updates');await requestUpdate('download');return true;
    }
    if(!$('setup').hidden)return false;
    if(action==='new-meeting')return await startNewMeeting();
    if(action==='import-recording' && usableActionControl('files') && !newMeetingOpening && !identityBusy && !retrying) { $('files').click();return true; }
    if(action==='find') {
      const id=$('workspace').hidden?'meeting-search':'search';
      if(!usableActionControl(id))return false;
      $(id).focus();$(id).select();return true;
    }
    if(action==='save' && usableActionControl('save') && !isLive()) { await flushSave();notice('Saved on this Mac.');return true; }
  } catch(error) { notice(error.message,true); }
  return false;
}
window.speakerdeskAction=dispatchAction;
function shortcutDisplay(action) {
  const mac=window.speakerdeskNativeMenu===true || typeof navigator==='undefined' || /Mac|iPhone|iPad/.test(navigator.platform);
  return {text:`${mac?'⌘':'Ctrl+'}${actionShortcuts[action].toUpperCase()}`,aria:`${mac?'Meta':'Control'}+${actionShortcuts[action].toUpperCase()}`};
}
function renderShortcutControl(control) {
  const action=control.dataset.shortcutAction;
  if(!actionShortcuts[action])return;
  const host=control.tagName==='INPUT'?control.closest('label'):control;
  if(!host)return;
  const active=action!=='find' || control.id===($('workspace').hidden?'meeting-search':'search');
  const shortcut=shortcutDisplay(action);
  let hint=host.querySelector('.shortcut-hint');
  if(!hint){hint=node('kbd',undefined,'shortcut-hint');hint.setAttribute('aria-hidden','true');host.append(hint);}
  hint.textContent=shortcut.text;hint.hidden=!active;
  const title=control.dataset.shortcutTitle || control.getAttribute('aria-label') || action;
  control.title=host.title=active?`${title} (${shortcut.text})`:title;
  if(active)control.setAttribute('aria-keyshortcuts',shortcut.aria);
  else control.removeAttribute('aria-keyshortcuts');
}
function renderShortcutHints() {
  document.querySelectorAll('[data-shortcut-action]').forEach(renderShortcutControl);
  document.querySelectorAll('[data-shortcut-key]').forEach(hint=>{
    const shortcut=shortcutDisplay(hint.dataset.shortcutKey);
    hint.textContent=shortcut.text;
    hint.setAttribute('aria-label',shortcut.aria.replace('Meta','Command'));
  });
}
function handleActionKey(event) {
  if(event.defaultPrevented || event.isComposing || !(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey)return;
  const action=Object.keys(actionShortcuts).find(action=>actionShortcuts[action]===event.key.toLowerCase());
  if(!action)return;
  // Native menu accelerators own Command keys. Leave WebKit's event uncancelled
  // so it can reach the menu; do not dispatch a second browser action here.
  if(window.speakerdeskNativeMenu===true && event.metaKey)return;
  event.preventDefault(); // Suppress browser New Window / Open / Find / Save even when blocked.
  if(event.repeat)return;
  void dispatchAction(action);
}
function wire() {
  wirePeople();
  renderShortcutHints();
  $('new-meeting').addEventListener('click', () => void dispatchAction('new-meeting'));
  $('refinement-toggle').addEventListener('click',toggleRefinement);
  $('meeting-search').addEventListener('input',renderJobs);
  $('language').addEventListener('change',()=>changeDefaultLanguage().catch(error=>notice(error.message,true)));
  $('setup-card-install').addEventListener('click',()=>installModels().catch(error=>{$('setup-card-status').textContent=error.message;}));
  $('setup-card-details').addEventListener('click',()=>openSettings(setupState?.ready?'general':'models'));
  document.querySelectorAll('.settings-tabs [role=tab]').forEach(tab=>{
    tab.addEventListener('click',()=>showSettingsTab(tab.dataset.tab));
    tab.addEventListener('keydown',event=>{
      const tabs=[...document.querySelectorAll('.settings-tabs [role=tab]')],index=tabs.indexOf(tab);
      const next={ArrowRight:index+1,ArrowLeft:index-1,Home:0,End:tabs.length-1}[event.key];
      if(next===undefined)return;event.preventDefault();
      const target=tabs[(next+tabs.length)%tabs.length];showSettingsTab(target.dataset.tab);target.focus();
    });
  });
  $('live-language').addEventListener('change',()=>changeLiveLanguage().catch(error=>notice(error.message,true)));
  $('theme').value=speakerdeskTheme.get();
  $('theme').addEventListener('change',async()=>{
    const control=$('theme'),prior=config.theme || 'system';control.disabled=true;
    changingTheme=true;themeGeneration++;
    try { const result=await api('/api/preferences',{method:'PATCH',body:JSON.stringify({theme:control.value})});config.theme=result.theme;speakerdeskTheme.set(result.theme); }
    catch(error){control.value=prior;notice(error.message,true);}finally{changingTheme=false;themeGeneration++;control.disabled=false;}
  });
  $('cancel-job').addEventListener('click',async()=>{
    if(!selected || cancelingJobs.has(selected.id))return;
    const jid=selected.id;cancelingJobs.add(jid);$('cancel-job').disabled=true;$('cancel-job').textContent='Stopping processing…';
    try { await api(`/api/jobs/${jid}/cancel`,{method:'POST'});await poll(); }
    catch(error){cancelingJobs.delete(jid);notice(error.message,true);}
    finally{if(selected)setStatus();}
  });
  $('meeting-form').addEventListener('submit',async event=>{
    event.preventDefault();$('start-meeting').disabled=true;
    try {
      if(changingLiveLanguage || changingDefaultLanguage)throw new Error('Wait for the language change to finish before starting.');
      if(!languageReady($('language').value)){openSettings('models');throw new Error('Finish local model setup before starting a meeting.');}
      const sources=[];if($('capture-microphone').checked)sources.push('microphone');if($('capture-system').checked)sources.push('system');
      if(!sources.length)throw new Error('Choose an audio source before starting.');
      const job=await api('/api/meetings',{method:'POST',body:JSON.stringify({name:$('meeting-title').value,language:$('language').value,sources})});
      followingLive=true;jobs=await api('/api/jobs');await select(job.id);await refreshMeeting();
    }catch(error){notice(error.message,true);}finally{$('start-meeting').disabled=false;renderStartState();}
  });
  $('pause-meeting').addEventListener('click',()=>meetingAction(meeting?.status==='paused'?'resume':'pause').catch(e=>notice(e.message,true)));
  $('stop-meeting').addEventListener('click',()=>meetingAction('stop').catch(e=>notice(e.message,true)));
  $('active-meeting').addEventListener('click',()=>select(meeting.id).catch(e=>notice(e.message,true)));
  $('follow-live').addEventListener('click',()=>{followingLive=true;document.activeElement?.blur?.();$('transcript-pane').scrollTop=$('transcript-pane').scrollHeight;renderFollowLive();});
  $('transcript-pane').addEventListener('scroll',()=>{if(!isLive())return;const p=$('transcript-pane');followingLive=p.scrollHeight-p.scrollTop-p.clientHeight<60;renderFollowLive();});
  $('files').addEventListener('change', () => upload($('files').files).catch(e => notice(e.message, true)));
  $('upload-form').addEventListener('submit', e => e.preventDefault());
  const drop = document.querySelector('main');
  ['dragenter', 'dragover'].forEach(event => drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(event => drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('dragging'); }));
  drop.addEventListener('drop', e => upload(e.dataTransfer.files).catch(err => notice(err.message, true)));
  $('setup-toggle').addEventListener('click', () => void dispatchAction('settings'));
  $('inspector-close').addEventListener('click', () => {
    const active=document.querySelector('.segment.active');$('inspector').hidden=true;
    document.querySelectorAll('.segment.active').forEach(item => item.classList.remove('active'));
    active?.querySelector('textarea')?.focus({preventScroll:true});$('inspector').hidden=true;
  });
  $('clear-search').addEventListener('click',()=>{$('search').value='';renderSegments();$('search').focus();});
  $('search').addEventListener('keydown',event=>{if(event.key==='Escape' && $('search').value){event.preventDefault();$('search').value='';renderSegments();}});
  document.addEventListener('keydown', handleActionKey);
  $('settings-close').addEventListener('click',closeSettings);
  $('update-check').addEventListener('click',()=>void dispatchAction('check-updates'));
  $('update-download').addEventListener('click',()=>void dispatchAction('download-update'));
  $('update-notice-action').addEventListener('click',()=>void dispatchAction('download-update'));
  $('update-notice-dismiss').addEventListener('click',()=>void dismissUpdateNotice());
  $('update-install').addEventListener('click',()=>requestUpdate('install').catch(error=>notice(error.message,true)));
  $('update-cancel').addEventListener('click',()=>requestUpdate('cancel').catch(error=>notice(error.message,true)));
  document.addEventListener('keydown', event => {
    if($('setup').hidden) return;
    if(event.key==='Escape') closeSettings();
    if(event.key==='Tab') {
      const controls=Array.from($('setup').querySelectorAll('button:not([disabled]):not([hidden]),summary,select:not([disabled]),input:not([disabled]),a[href]')).filter(item => item.offsetParent!==null);
      const first=controls[0],last=controls.at(-1);
      if(event.shiftKey && document.activeElement===first){event.preventDefault();last.focus();}
      else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first.focus();}
    }
  });
  $('install-alignment').addEventListener('click',()=>alignmentAction('POST').catch(error=>{ $('alignment-status').textContent=error.message; }));
  $('cancel-model-setup').addEventListener('click', async () => {
    try { await api('/api/setup/cancel',{method:'POST'}); await refreshSetup(); }
    catch(error) { $('model-status').textContent=error.message; }
  });
  $('install-models').addEventListener('click',() => installModels().catch(error => { $('model-status').textContent=error.message; }));
  $('recognize-voices').addEventListener('change',async event => {
    const control=event.target;control.disabled=true;
    try { await api('/api/recognition',{method:'PATCH',body:JSON.stringify({enabled:control.checked})}); }
    catch(error) { control.checked=!control.checked;notice(error.message,true); }
    finally { control.disabled=false;await refreshSetup(); }
  });
  $('run').addEventListener('click', async () => {
    if(startingJob || !selected || $('run').disabled)return;
    const jid=selected.id,generation=selectionGeneration;
    startingJob=jid;setStatus();
    try {
      const next=await api(`/api/jobs/${jid}/run`,{method:'POST'});
      if(selected?.id===jid && generation===selectionGeneration && !(next.revision<selected.revision))selected=next;
    } catch(error) { notice(error.message,true); }
    finally { startingJob=null;if(selected)setStatus(); }
  });
  $('manual').addEventListener('click', manual);
  $('save').addEventListener('click', () => void dispatchAction('save'));
  $('speed').addEventListener('change', () => { $('player').playbackRate = Number($('speed').value); });
  $('player').addEventListener('timeupdate', () => {
    if (!doc) return;
    const now = $('player').currentTime;
    if(passageEnd!==null && now>=passageEnd){$('player').pause();passageEnd=null;}
    updatePlaybackHighlight();
  });
  $('search').addEventListener('input', () => { if (doc) renderSegments(); });
  $('add').addEventListener('click', () => {
    const start = Math.min($('player').currentTime || 0, selected.duration - .01);
    doc.segments.push({id: crypto.randomUUID(), start, end: Math.min(start + 5, selected.duration), speaker: Object.keys(doc.speakers)[0], text: '', timing: 'manual', confidence: null});
    changed(); $('search').value = ''; renderEditor();
    document.querySelector('#segments .segment:last-child textarea')?.focus();
  });
  $('add-speaker').addEventListener('click', () => {
    const names=Object.entries(doc.speakers).filter(([id])=>id!=='unassigned' && id!=='multiple_speakers').map(([,name])=>name);
    let number=names.length+1;while(names.includes(`Speaker ${number}`))number++;
    const id=`manual_${crypto.randomUUID()}`;
    doc.speakers[id]=`Speaker ${number}`;changed();renderEditor();
  });
  $('import').addEventListener('change', async () => {
    try {
      if(hasPendingPassageSaves())throw new Error('Wait for the passage save to finish before importing.');
      if(isProgressiveImport())throw new Error('Finish transcription before replacing the document. Passage corrections remain available.');
      const file = $('import').files[0]; if (!file) return;
      document.querySelector('.export-menu').open = false;
      if (!(await saveBeforeLeaving())) return;
      if ((hasPassageDrafts() || hasRecoverablePassageDrafts()) && !confirm('Discard unsaved or recoverable passage corrections and replace this transcript?')) return;
      if (doc?.segments?.length && !confirm(`Replace the transcript of ${selected.name} with ${file.name}?`)) return;
      const owner={id:selected.id,selection:selectionGeneration,edits:editGeneration,revision:selected.revision};
      saving = true; $('editor').inert = true;
      if (file.size > 5 * 1024 * 1024) throw new Error('Transcript JSON must be at most 5 MB.');
      const incoming = JSON.parse(await file.text());
      if (!incoming.speakers || !Array.isArray(incoming.segments)) throw new Error('JSON needs speakers and segments. See fixtures/conversation.json.');
      // Validate and persist atomically before showing imported data.
      if(selected?.id!==owner.id || selectionGeneration!==owner.selection || editGeneration!==owner.edits)throw new Error('The current edits changed before import. Try again.');
      const result = await api(`/api/jobs/${owner.id}/transcript`, {method: 'PUT', body: JSON.stringify({revision: owner.revision, document: incoming, imported: true})});
      if(selected?.id!==owner.id || selectionGeneration!==owner.selection || editGeneration!==owner.edits)return;
      passageDrafts.clear();recoverablePassageDrafts.clear();
      selected = result; doc = structuredClone(result.document); dirty = false; setStatus(); renderEditor(); notice('Imported and saved locally.');
    } catch (e) { notice(e.message, true); } finally { saving = false; $('editor').inert = false; $('import').value = ''; }
  });
  document.querySelectorAll('[data-export]').forEach(button => button.addEventListener('click', () => downloadExport(button.dataset.export).catch(e=>notice(e.message,true))));
  document.querySelector('a[href="/api/notices"]').addEventListener('click',event=>{
    event.preventDefault();downloadExport().catch(e=>notice(e.message,true));
  });
  $('delete').addEventListener('click', async () => {
    if (saving) { notice('Wait for this save to finish before deleting.'); return; }
    if (!confirm(`Delete ${selected.name} and its local audio and transcript?`)) return;
    const jid=selected.id,generation=selectionGeneration;
    clearTimeout(autosaveTimer); autosaveTimer = null;
    if (undoRemoval) { clearTimeout(undoRemoval.timer); undoRemoval = null; }
    try {
      await api(`/api/jobs/${jid}`, {method: 'DELETE'});
      if(selected?.id===jid && selectionGeneration===generation) {
        passageDrafts.clear();recoverablePassageDrafts.clear();timingDrafts.clear();selected = doc = null;resetPlayback();dirty = false;selectionGeneration++;
        $('player').pause(); $('player').removeAttribute('src'); $('workspace').hidden = true; $('empty').hidden = false;
      }
      jobs = await api('/api/jobs'); renderJobs(); notice('Recording deleted from local storage.');
    } catch (e) { notice(e.message, true); if (dirty) scheduleAutosave(); }
  });
  window.addEventListener('beforeunload', e => { if (dirty || saving || timingDrafts.size || hasPassageDrafts() || hasRecoverablePassageDrafts() || hasPendingPassageSaves()) { e.preventDefault(); e.returnValue = ''; } });
  $('player').addEventListener('loadedmetadata',()=>{if(passageStart!==null)$('player').currentTime=passageStart;});
  $('player').addEventListener('error',()=>notice('Passage audio unavailable. Your words are retained.',true));
}
async function poll() {
  if(updateFrozen)return;
  if (polling) return; polling = true;
  try {
    jobs = await api('/api/jobs'); renderJobs();
    if (selected && !dirty && !saving && !identityBusy && !retrying) {
      const generation = selectionGeneration, jid = selected.id;
      const revision=selected.revision;
      const next = await api(`/api/jobs/${jid}`+(Number.isInteger(revision)?`?known_revision=${revision}`:''));
      if (generation !== selectionGeneration || selected?.id !== jid || dirty || saving || identityBusy || retrying || next.revision < selected.revision) return;
      const changedAudio = !selected.duration && next.duration;
      const finished = next.status !== selected.status && !['preparing','queued','processing',...liveStatuses].includes(next.status);
      // Conditional replies always carry fresh metadata; retain only the server
      // document. Never merge old duration, language, busy or Pause receipts.
      if(next.unchanged===true && next.revision===selected.revision)next.document=selected.document;
      if (next.revision !== selected.revision) { doc = next.document ? structuredClone(next.document) : null; selected = next; if(isLive() || hasPassageDrafts()){renderSegments();refreshIdentitySuggestions();updateNameControls();}else renderEditor(); }
      selected = next; setStatus();
      if (finished) notice('');
      if ((changedAudio || finished) && !isLive() && selected.duration) $('player').src = `/api/jobs/${selected.id}/audio`;
      if(finished)renderEditor();
    }
  } catch (e) { notice(e.message, true); } finally { polling = false; }
}
async function init() {
  if(initializing)return;initializing=true;
  renderShortcutHints();
  try {
    await refreshConfig();
    if(!wired)Object.entries(config.languages).forEach(([code, name]) => {
      for(const id of ['language','live-language']){const option=node('option',id==='live-language' && code==='auto'?'Auto':name);option.value=code;$(id).append(option);}
    });
    renderLanguageDefault();
    if(!wired){wire();wired=true;}
    if(!pollingStarted){setInterval(poll,1000);setInterval(refreshMeeting,500);setInterval(refreshSetup,1500);setInterval(refreshUpdates,1000);pollingStarted=true;}
    jobs = await api('/api/jobs');renderJobs();
    await refreshMeeting();
    const requested=new URLSearchParams(location.search).get('meeting');
    if(requested && jobs.some(j=>j.id===requested))await select(requested);
    else if(meeting?.id && liveStatuses.includes(meeting.status)) await select(meeting.id);
    await refreshSetup();
    await initializeUpdates();initialized=true;
    if(typeof loadRecoveryCopies==='function')await loadRecoveryCopies();
    renderIcons();
  } catch (e) { markServiceUnavailable(`Could not open the local workspace: ${e.message}. Choose Retry.`); }
  finally{initializing=false;}
}
function showSettingsTab(name) {
  document.querySelectorAll('.settings-tabs [role=tab]').forEach(tab=>{const on=tab.dataset.tab===name;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;});
  document.querySelectorAll('.settings-panel').forEach(panel=>{panel.hidden=panel.id!==`panel-${name}`;});
}
function openSettings(tab) { if(tab)showSettingsTab(tab);$('setup').hidden=false;$('setup-toggle').setAttribute('aria-expanded','true');document.querySelector('main').inert=true;document.querySelector('.sidebar').inert=true;$('settings-close').focus();if(tab!=='updates')loadPeople().catch(error=>notice(error.message,true)); }
function closeSettings() { if(updateFrozen)return; $('setup').hidden=true;document.querySelector('main').inert=false;document.querySelector('.sidebar').inert=false;$('setup-toggle').setAttribute('aria-expanded','false'); $('setup-toggle').focus(); }
function formatBytes(bytes) {
  return bytes<1e8?`${Math.round(bytes/1e6)} MB`:`${(bytes/1e9).toFixed(1)} GB`;
}
async function installModels() {
  await api('/api/setup',{method:'POST'}); await refreshSetup();
}
function modelSetupMessage(state) {
  if(!state.supported)return 'This Mac can’t run the local models. Speakerdesk needs Apple Silicon and macOS 15 or later.';
  if(state.status==='failed')return 'Model setup did not finish. Try again, or open Details for more information.';
  return 'Download the local models once. Meetings are transcribed offline on this Mac.';
}
function renderStartState() {
  if(!config)return;
  const modelsReady=languageReady($('language').value), capture=meeting?.capture_available!==false;
  const state=setupState, downloading=state?.status==='downloading', loading=state?.status==='loading', busy=downloading || loading;
  $('setup-card').hidden=modelsReady || !state || loading;
  $('setup-card-install').hidden=loading;
  if(state && !modelsReady && !loading) {
    const percent=state.total_bytes?Math.round(100*state.downloaded_bytes/state.total_bytes):0;
    $('setup-card-text').textContent=downloading?'Downloading local models. Meetings are transcribed offline on this Mac.':modelSetupMessage(state);
    $('setup-card-progress').hidden=!downloading;$('setup-card-progress').value=percent;
    $('setup-card-status').textContent=downloading?`${state.phase} · ${percent}%`:'';
    $('setup-card-install').disabled=busy || !state.supported || state.can_download===false || state.voice?.status==='downloading';
    $('setup-card-install').querySelector('span').textContent=downloading?'Downloading…':state.status==='failed'?'Retry setup':`Download models · ${formatBytes(state.total_bytes)}`;
  }
  renderSeparationReadiness();
  $('start-meeting').disabled=serviceUnavailable || busy || !modelsReady || !capture || changingLiveLanguage || changingDefaultLanguage;
  $('start-hint').textContent=!capture?'Live recording isn’t available in this build. Use Import recording in the sidebar to transcribe a file.'
    :loading?'Loading models…':downloading?'Finish model setup before starting a meeting.':!modelsReady?'Start meeting is available once the models are ready.':'';
}
function renderSeparationReadiness() {
  const separation=setupState?.separation;if(!separation)return;
  const failed=separation.ready!==true && setupState.operation?.target==='separation' && ['failed','interrupted'].includes(setupState.operation.status);
  const message=failed?'Automatic separation needs attention':separation.active?'Automatic separation is setting up':(separation.ready??separation.installed)?'Automatic separation is ready':separation.installed?'Automatic separation needs setup':'Automatic separation needs a model';
  for(const id of ['start-separation-readiness','meeting-separation-readiness']) {
    const host=$(id);host.replaceChildren();host.append(node('span',message));
    if(!(separation.ready??separation.installed) || failed){const action=node('button',failed?'Retry download':separation.installed?'Finish setup':'Download','text-button');action.addEventListener('click',()=>openSettings('models'));host.append(action);}
  }
}
let alignmentChanging=false,alignmentGeneration=0;
async function alignmentAction(method,body) {
  alignmentChanging=true;alignmentGeneration++;
  $('install-alignment').disabled=true;
  try { await api('/api/setup/alignment',{method,...(body?{body:JSON.stringify(body)}:{})}); }
  finally { alignmentChanging=false;alignmentGeneration++;await refreshSetup(); }
}
function renderAlignment(state) {
  const alignment=state.alignment;$('alignment-controls').hidden=!alignment;
  if(!alignment)return;
  const busy=alignment.status==='downloading', loading=alignment.status==='loading';
  const languages=alignment.supported_languages.map(code=>config?.languages[code]||code).join(', ');
  const calibrated=(alignment.timing_accuracy_calibrated_languages||[]).map(code=>config?.languages[code]||code).join(', ');
  $('alignment-coverage').textContent=`Supported languages: ${languages||'None'}. Timing accuracy measured against references: ${calibrated||'None'}.`;
  if(alignment.supported_languages.some(code=>!(alignment.timing_accuracy_calibrated_languages||[]).includes(code)))
    $('alignment-coverage').textContent+=' Accuracy in other supported languages has not been independently measured.';
  $('alignment-cost').textContent=`${formatBytes(alignment.total_bytes)} download · ${formatBytes(alignment.stored_bytes)} stored · up to ${formatBytes(alignment.required_free_bytes)} free space needed.`;
  $('alignment-cost').hidden=loading || (state.status==='failed' && !state.download_required);
  $('alignment-message').textContent=loading?'Preparing transcript timing on this Mac.':state.status==='failed' && !state.download_required?'Retry model setup to check transcript timing.':alignment.message;
  $('alignment-status').textContent=alignment.error || (loading?(alignment.phase || 'Loading models…'):busy?`${alignment.phase} · ${alignment.total_bytes?Math.round(100*alignment.downloaded_bytes/alignment.total_bytes):0}%`:alignment.ready?'Ready':'Not ready');
  $('alignment-progress').hidden=!busy;
  $('alignment-progress').value=alignment.total_bytes?100*alignment.downloaded_bytes/alignment.total_bytes:0;
  $('install-alignment').hidden=loading || (alignment.ready && alignment.status!=='failed') || alignment.can_download===false;
  $('install-alignment').disabled=alignmentChanging || !alignment.can_download;
  $('install-alignment').textContent=busy?'Setting up…':['failed','interrupted'].includes(alignment.status)?'Retry setup': 'Finish setup';
}
async function refreshSetup() {
  const generation=alignmentGeneration;
  try {
    const state=await api('/api/setup');
    if(generation!==alignmentGeneration || alignmentChanging)return;
    const priorSeparation=setupState?.separation;
    setupState=state;
    const downloading=state.status==='downloading', loading=state.status==='loading', busy=downloading || loading;
    const list=$('model-list'),existing=new Map(Array.from(list.children).map(row=>[row.dataset.modelId,row])),retained=new Set();
    const models=[...state.models].sort((a,b)=>Number(b.id==='mossformer2-16k')-Number(a.id==='mossformer2-16k'));
    models.forEach(model => {
      const size=formatBytes(model.bytes);
      let item=existing.get(model.id);
      if(!item){item=node('div',undefined,'model-item');item.dataset.modelId=model.id;item.append(node('strong',model.name),node('span'));list.append(item);}
      retained.add(item);
      const alignment=model.id===state.alignment?.id?state.alignment:null;
      const label=loading?'Loading…':model.name==='Voice recognition' && state.voice?.released===false && !state.voice?.available?'Unavailable':alignment?(alignment.ready?'Ready':alignment.status==='downloading'?'Setting up…':'Needs setup'):model.installed?'Installed':model.optional?`Optional · ${size}`:size;
      if(item.querySelector('span').textContent!==label)item.querySelector('span').textContent=label;
      if(model.id==='mossformer2-16k' && state.separation) {
        item.classList.add('separation-model');
        let install=item.querySelector('button');
        if(!install){
          item.append(node('p','Separates saved passages where two voices overlap.','separation-model-note'));
          install=node('button',undefined,'quiet');item.insertBefore(install,item.querySelector('.separation-model-note'));
          install.addEventListener('click',async()=>{install.dataset.requesting='true';install.disabled=true;try{await api('/api/setup/separation',{method:'POST'});await refreshSetup();}catch(error){notice(error.message,true);}finally{install.dataset.requesting='false';await refreshSetup();}});
        }
        const failed=state.separation.ready!==true && state.operation?.target==='separation' && ['failed','interrupted'].includes(state.operation.status);
        if(failed)item.querySelector('span').textContent='Needs attention';
        let failure=item.querySelector('.model-failure');if(!failure){failure=node('p',undefined,'model-failure');item.insertBefore(failure,install);}
        failure.hidden=!failed;failure.textContent=failed?(state.operation.error || 'Separation setup did not finish. Retry the download.') : '';
        install.textContent=failed?'Retry separation download':state.separation.active?'Setting up separation…':state.separation.installed?'Check separation model':`Download separation · ${formatBytes(state.separation.bytes)}`;
        install.disabled=install.dataset.requesting==='true' || state.separation.active || state.separation.can_download===false || busy;
      }
    });
    for(const item of Array.from(list.children))if(!retained.has(item))item.remove();
    renderAlignment(state);
    if(state.required_free_bytes)$('setup-disk-note').textContent=`First setup needs an internet connection and about ${formatBytes(state.required_free_bytes)} of free disk space including prepared timing weights.`;
    $('setup-disk-note').hidden=loading || state.ready || (state.status==='failed' && !state.download_required);
    $('model-progress').hidden=!downloading;
    $('model-progress').value=state.total_bytes?100*state.downloaded_bytes/state.total_bytes:0;
    const operation=state.operation,failedModels=!state.ready && operation?.target==='models' && ['failed','interrupted'].includes(operation.status);
    const failedSeparation=state.separation?.ready!==true && operation?.target==='separation' && ['failed','interrupted'].includes(operation.status);
    $('model-status').textContent=failedModels?'Transcription setup did not finish. Retry setup.':loading?(state.phase || 'Loading models…'):downloading?`${state.phase} · ${state.total_bytes?Math.round(100*state.downloaded_bytes/state.total_bytes):0}%`:state.ready?'Ready to transcribe on this Mac.':modelSetupMessage(state);
    $('model-error').textContent=[...new Set([state.error,state.message,operation?.error && operation.error!==state.error?`Previous setup attempt: ${operation.error}`:null].filter(Boolean))].join('\n');
    $('model-error-details').hidden=!$('model-error').textContent;
    $('install-models').hidden=loading || (state.ready && !failedModels) || failedSeparation;
    $('cancel-model-setup').hidden=!state.can_cancel;
    const voice=state.voice,voiceBusy=voice.status==='downloading';
    $('install-models').disabled=busy || voiceBusy || !state.supported || state.can_download===false;
    $('install-models').textContent=downloading?'Downloading…':(failedModels || state.status==='failed')?'Retry setup':`Download models · ${formatBytes(state.total_bytes)}`;
    if(voiceAvailable!==voice.available) await loadPeople();
    $('settings-voice-note').textContent=voice.enabled?voice.message:'Recognition is off. Saved names and voices stay on this Mac.';
    if(document.activeElement!==$('recognize-voices')) $('recognize-voices').checked=voice.enabled;
    if(typeof state.core_ready==='boolean' && (state.core_ready !== config.readiness.configured || (state.automatic_language??true)!==config.readiness.automatic_language || (state.alignment?.ready??config.readiness.word_alignment)!==config.readiness.word_alignment)) {
      await refreshConfig();renderLanguageDefault();if(selected)setStatus();
    }
    renderStartState();
    if(doc && JSON.stringify(priorSeparation)!==JSON.stringify(state.separation))renderSegments();
  } catch(error) {
    $('model-error').textContent=error.message;$('model-error-details').hidden=false;
    $('model-status').textContent='Could not check model setup. Open Details and try again.';
  }
}
async function meetingAction(action) {
  if(!meeting?.id)return;
  $('pause-meeting').disabled=true;$('stop-meeting').disabled=true;
  try{await api(`/api/meetings/${meeting.id}/${action}`,{method:'POST'});await refreshMeeting();await poll();}
  finally{await refreshMeeting();}
}
let meetingPoll=false;
function liveLanguageTime(samples) {
  const hundredths=Math.floor(samples/160);
  return `${time(Math.floor(hundredths/100))}.${String(hundredths%100).padStart(2,'0')}`;
}
async function refreshConfig() {
  const generation=defaultLanguageGeneration,appearance=themeGeneration,next=await api('/api/config');
  // A config read started before a preference commit must not undo it.
  if(config && (generation!==defaultLanguageGeneration || changingLiveLanguage || changingDefaultLanguage))next.default_language=config.default_language;
  if(config && (appearance!==themeGeneration || changingTheme))next.theme=config.theme;
  config=next;
  if(['system','light','dark'].includes(next.theme)){speakerdeskTheme.set(next.theme);$('theme').value=next.theme;}
}
function renderLanguageDefault() {
  const control=$('language');
  if(!changingLiveLanguage && !changingDefaultLanguage)control.value=config.languages[config.default_language]?config.default_language:'auto';
  control.disabled=changingLiveLanguage || changingDefaultLanguage;
  renderStartState();
}
function acceptLanguageDefault(language) {
  config.default_language=config.languages[language]?language:'auto';
  defaultLanguageGeneration++;
  $('language').value=config.default_language;
}
async function changeDefaultLanguage() {
  if(changingLiveLanguage || changingDefaultLanguage)return;
  const language=$('language').value;
  changingDefaultLanguage=true;defaultLanguageGeneration++;renderLanguageDefault();renderLiveLanguage();
  try {
    const next=await api('/api/preferences/language',{method:'PATCH',body:JSON.stringify({language})});
    acceptLanguageDefault(next.default_language);
  } finally {
    changingDefaultLanguage=false;renderLanguageDefault();renderLiveLanguage();
  }
}
function renderLiveLanguage() {
  const control=$('live-language');
  const current=meeting?.id===selected?.id && liveStatuses.includes(meeting?.status);
  $('meeting-language-control').hidden=!current;
  if(!changingLiveLanguage)control.value=meeting?.language || 'auto';
  control.disabled=changingLiveLanguage || changingDefaultLanguage || !current || !['recording','paused'].includes(meeting?.status);
  const pending=(meeting?.language_acknowledged_revision || 0)<(meeting?.language_revision || 0);
  $('live-language-status').textContent=meeting?.language_revision?
    `From ${liveLanguageTime(meeting.language_from_sample || 0)}${pending?' · Queued':''} · Changes also set the default`:'Changes also set the default';
}
async function changeLiveLanguage(language=$('live-language').value) {
  if(changingLiveLanguage || changingDefaultLanguage || !meeting?.id || !['recording','paused'].includes(meeting.status))return;
  const jid=meeting.id,revision=meeting.language_revision || 0;
  changingLiveLanguage=true;defaultLanguageGeneration++;$('live-language').value=language;$('language').value=language;renderLanguageDefault();renderLiveLanguage();
  try {
    const next=await api(`/api/meetings/${jid}/language`,{method:'PATCH',body:JSON.stringify({language,language_revision:revision})});
    acceptLanguageDefault(next.default_language);
    if(meeting?.id===jid && next.language_revision >= (meeting.language_revision || 0)) {
      meeting={...meeting,language:next.language,language_revision:next.language_revision,
        language_acknowledged_revision:Math.max(meeting.language_acknowledged_revision || 0,next.language_acknowledged_revision),
        language_from_sample:next.language_history.at(-1).start_sample};
      notice(`${config.languages[next.language]} applies from ${liveLanguageTime(meeting.language_from_sample)}. Earlier words keep their language.`);
    }
  } finally {
    changingLiveLanguage=false;renderLanguageDefault();renderLiveLanguage();await refreshMeeting();
  }
}
async function refreshMeeting() {
  if(meetingPoll)return;meetingPoll=true;
  try{
    const next=await api('/api/meeting');
    if(next.id && next.id===meeting?.id) {
      // A poll begun before PATCH must not restore an older language selection.
      if((next.language_revision || 0)<(meeting.language_revision || 0)) {
        for(const key of ['language','language_revision','language_from_sample'])next[key]=meeting[key];
      }
      next.language_acknowledged_revision=Math.max(next.language_acknowledged_revision || 0,meeting.language_acknowledged_revision || 0);
    }
    meeting=next;
    markServiceAvailable();
    const active=liveStatuses.includes(meeting.status);
    $('capture-footer').hidden=!active;
    renderLiveLanguage();
    renderLanguageDefault();
    $('new-meeting').disabled=active;
    renderStartState();
    $('elapsed').textContent=time(meeting.duration);
    const sources=meeting.sources || (selected?.id===meeting.id ? selected.sources : []) || [];
    for(const [source,id,label] of [['microphone','microphone-level','Microphone'],['system','system-level','Mac audio']]){
      const enabled=sources.includes(source),meter=$(id).closest('.audio-meter');
      meter.classList.toggle('off',!enabled);meter.title=enabled?'':`${label} isn’t recorded in this meeting`;
      meter.querySelector('label').textContent=enabled?label:`${label} · off`;
    }
    $('active-meeting').hidden=selected?.id===meeting.id;
    $('active-meeting').textContent=meeting.status==='paused'?'Paused':meeting.status==='starting'?'Starting…':meeting.status==='finishing'?'Finishing…':'Recording';
    $('microphone-level').value=meeting.status==='paused'?0:Math.min(1,Math.sqrt(meeting.levels.microphone||0));
    $('system-level').value=meeting.status==='paused'?0:Math.min(1,Math.sqrt(meeting.levels.system||0));
    $('live-delay').textContent=meeting.pending_seconds>3?`${Math.round(meeting.pending_seconds)}s behind`:'';
    const pauseLabel=meeting.status==='paused'?'Resume':'Pause';
    const pauseButton=$('pause-meeting');
    if(pauseButton.querySelector('span').textContent!==pauseLabel){pauseButton.replaceChildren(icon(meeting.status==='paused'?'play':'pause'),node('span',pauseLabel));renderIcons();}
    $('pause-meeting').disabled=!['recording','paused'].includes(meeting.status);
    $('stop-meeting').disabled=meeting.status==='finishing';
  }catch(error){notice(error.message,true);}finally{meetingPoll=false;}
}
function freezeForUpdate(frozen) {
  updateFrozen = frozen;
  if (frozen) openSettings('updates');
  $('settings-close').disabled = frozen;
  document.querySelector('main').inert = frozen || !$('setup').hidden;
  document.querySelector('.sidebar').inert = frozen || !$('setup').hidden;
  document.querySelectorAll('.settings-panel').forEach(panel => { panel.inert = frozen && panel.id !== 'panel-updates'; });
  const tabs = document.querySelector('.settings-tabs'); if (tabs) tabs.inert = frozen;
  document.querySelectorAll('dialog').forEach(dialog => { dialog.inert = frozen; });
  $('editor').inert = frozen || retrying || (!!selected && !isProgressiveImport() && ['preparing','processing','queued'].includes(selected.status));
  renderSaveState();renderExportState();
}
function renderUpdates() {
  renderUpdateNotice();
  const state = updateState.state, version = updateState.version || '';
  const messages = {idle:'Check for a newer Speakerdesk version.',checking:'Checking for updates…',current:'Speakerdesk is up to date.',
    available:`Speakerdesk ${version} is available.`,downloading:'Downloading the update…',verifying:'Verifying the update…',
    ready:updateState.error || `Speakerdesk ${version} is ready to install.`,preparing:'Saving your edits before updating…',preparing_install:'Saving your edits before updating…',
    stopping:'Closing the local processing service…',shutting_down:'Closing the local processing service…',installing:'Installing the update. Speakerdesk will restart.',
    error:updateState.error || 'The update could not finish. Try checking again.',unavailable:updateState.error || 'Updates are available in the installed desktop app.'};
  $('update-status').textContent = messages[state] || 'Waiting for the update service…';
  $('update-notes').textContent = typeof updateState.notes === 'string' ? updateState.notes : '';
  $('update-notes').hidden = !$('update-notes').textContent;
  const active = ['checking','downloading','verifying','preparing','preparing_install','stopping','shutting_down','installing'].includes(state);
  const capture = liveStatuses.includes(meeting?.status) || liveStatuses.includes(selected?.status);
  $('update-check').disabled = updateRequesting || active || updateFrozen || state === 'unavailable';
  $('update-download').hidden = state !== 'available';
  $('update-download').disabled = updateRequesting || capture;
  $('update-install').hidden = state !== 'ready';
  $('update-install').disabled = updateRequesting || capture;
  $('update-cancel').hidden = !active && state !== 'ready' && !updateFrozen;
  const cancellable = typeof updateState.cancellable === 'boolean' ? updateState.cancellable :
    !['stopping','shutting_down','installing'].includes(state) && !updateState.reserved;
  $('update-cancel').disabled = updateRequesting || !cancellable;
  $('update-capture-note').hidden = !capture;
  const progress = $('update-progress');progress.hidden = state !== 'downloading';
  if (Number.isFinite(updateState.total_bytes) && updateState.total_bytes > 0) {
    progress.max = updateState.total_bytes;progress.value = Math.min(updateState.downloaded_bytes || 0, progress.max);
  } else progress.removeAttribute('value');
}
function renderUpdateNotice() {
  const version = updateState.version;
  $('update-notice').hidden = updateState.state !== 'available' || !version || version === dismissedUpdateVersion;
  $('update-notice-text').textContent = version ? `Speakerdesk ${version} is available.` : '';
  $('update-notice-action').disabled = updateFrozen || updateRequesting || liveStatuses.includes(meeting?.status) || isLive();
  $('update-notice-action').title = $('update-notice-action').disabled ? 'Finish the recording before updating.' : 'Download this update, then choose when to install and restart.';
}
async function dismissUpdateNotice() {
  if(updateFrozen || updateState.state !== 'available')return;
  const version = updateState.version;
  try {
    const result = await api('/api/preferences/update-notice',{method:'PATCH',body:JSON.stringify({version})});
    dismissedUpdateVersion = result.dismissed_version;renderUpdateNotice();
  } catch(error) { notice(error.message,true); }
}
async function initializeUpdates() {
  await refreshUpdates();
  if(!startupUpdateChecked && window.speakerdeskNativeUpdater===true && updateState.state==='idle') {
    startupUpdateChecked=true;
    try { await requestUpdate('check'); } catch { /* Manual Check for Updates remains available. */ }
  }
}
async function requestUpdate(op) {
  if (updateRequesting) return;
  if (['download','install'].includes(op) && (liveStatuses.includes(meeting?.status) || isLive())) throw new Error('Finish the recording before updating. Paused recordings are still active.');
  updateRequesting = true;renderUpdates();
  try { await api('/api/updates',{method:'POST',body:JSON.stringify({op})});await refreshUpdates(); }
  finally { updateRequesting = false;renderUpdates(); }
}
async function boundedUpdateSave(action, milliseconds = 15000) {
  let timer;
  try {
    await Promise.race([action(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Saving took too long. Your edits remain open; try again when saving finishes.')),milliseconds);})]);
  } finally { clearTimeout(timer); }
}
async function prepareEditorUpdate(id, preparation) {
  if (!Number.isSafeInteger(preparation) || preparation <= 0) return;
  if (updatePreparation?.id === id && updatePreparation.preparation === preparation) return updatePreparation.promise;
  const attempt = {id, preparation, promise:null};updatePreparation = attempt;
  const current=()=>!attempt.cancelled && updatePreparation===attempt && updateState.id===id && updateState.preparation===preparation && ['preparing','preparing_install'].includes(updateState.state);
  attempt.promise = (async()=>{
    const selection = selectionGeneration, generation = editGeneration, jid = selected?.id;
    try {
      if (liveStatuses.includes(meeting?.status) || isLive()) throw new Error('Finish the recording before installing an update. Paused recordings are still active.');
      if (hasPassageDrafts() || hasRecoverablePassageDrafts()) throw new Error('Save or recover your passage drafts before installing an update.');
      if (undoRemoval) throw new Error('Finish or undo the passage removal before installing an update.');
      if (pendingExport) throw new Error('Finish or cancel the export before installing an update.');
      if (hasPendingPassageSaves() || retrying || changingLiveLanguage || changingDefaultLanguage ||
          (typeof identityBusy !== 'undefined' && identityBusy) || document.querySelector('dialog[open]')) throw new Error('Finish the open edit or settings change before installing an update.');
      freezeForUpdate(true);
      await boundedUpdateSave(async()=>{
        while (localMutations || saving) {
          if(!current())return;
          await new Promise(resolve=>setTimeout(resolve,50));
        }
        if(!current())return;
        if(undoRemoval || hasPassageDrafts() || hasRecoverablePassageDrafts() || hasPendingPassageSaves()) throw new Error('Finish or undo your local edits before installing an update.');
        await flushSave({canContinue:current});
      });
      if (updatePreparation !== attempt || updateState.id !== id || updateState.preparation !== preparation || !['preparing','preparing_install'].includes(updateState.state)) return;
      if (dirty || saving || localMutations || hasPassageDrafts() || hasRecoverablePassageDrafts() || hasPendingPassageSaves() ||
          selectionGeneration !== selection || selected?.id !== jid || editGeneration !== generation) throw new Error('Your edits changed while saving. Review them before trying the update again.');
      await api('/api/updates',{method:'POST',body:JSON.stringify({op:'editor_ready',id,preparation})});
    } catch (error) {
      attempt.cancelled = true;
      if (updatePreparation !== attempt || updateState.id !== id || updateState.preparation !== preparation) return;
      $('update-status').textContent = error.message;
      if(!undoRemoval)notice(error.message,true);
      try { await api('/api/updates',{method:'POST',body:JSON.stringify({op:'editor_error',id,preparation,error:error.message})}); }
      catch { $('update-status').textContent = 'The update could not be cancelled. Keep Speakerdesk open and try Cancel again.'; }
    }
  })();
  return attempt.promise;
}
async function refreshUpdates() {
  if (updatePolling) return;
  updatePolling = true;
  try {
    const next = await api('/api/updates');
    if (!Number.isSafeInteger(next.id) || next.id < updateState.id || typeof next.state !== 'string') return;
    updateState = next;
    // A poll begun before dismissal must not restore its stale preference.
    if(!dismissedUpdateVersion && typeof next.dismissed_version==='string')dismissedUpdateVersion=next.dismissed_version;
    const protectedState = ['preparing','preparing_install','stopping','shutting_down','installing'].includes(next.state) || !!next.reserved;
    if (updateFrozen && !protectedState) { updatePreparation = null;freezeForUpdate(false); }
    renderUpdates();
    if (['preparing','preparing_install'].includes(next.state)) void prepareEditorUpdate(next.id,next.preparation);
  } catch (error) {
    if (updateFrozen) $('update-status').textContent = 'Waiting for the local update service. Keep Speakerdesk open.';
  } finally { updatePolling = false; }
}
$('connection-retry').addEventListener('click',()=>initialized?retryConnection():init());
init();
