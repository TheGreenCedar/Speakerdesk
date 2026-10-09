'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="speakerdesk-token"]').content;
let config, jobs = [], selected = null, doc = null, dirty = false, polling = false, saving = false, editGeneration = 0, selectionGeneration = 0;
let meeting = null, followingLive = true, passageEnd = null, retrying = false;
let changingLiveLanguage = false, changingDefaultLanguage = false, defaultLanguageGeneration = 0;
let pendingExport = null, autosaveTimer = null, setupState = null, undoRemoval = null;
let updateState = {id: 0, state: 'unavailable'}, updateFrozen = false, updatePolling = false;
let updatePreparation = null, updateRequesting = false, localMutations = 0;
let startupUpdateChecked = false, newMeetingOpening = false;
let dismissedUpdateVersion = '';
const savedPassageBindings = new WeakMap();
const readingTurnBindings = new WeakMap();
let playbackDocument = null, playbackIndexDirty = true, playbackRows = [], playbackEnds = [];
let playbackSegments = new Map(), playbackCards = new Map(), playingCards = new Set();
const narrowLayout = () => matchMedia('(max-width: 900px)').matches;
const liveStatuses = ['starting','recording','paused','finishing'];
const isLive = () => selected && liveStatuses.includes(selected.status);
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
  if (updateFrozen && mutation && !updater && !updateSave) throw new Error('Speakerdesk is preparing an update. Cancel the update to continue.');
  const headers = {'X-Speakerdesk-Token': token, ...options.headers};
  if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  if (mutation && !updater) localMutations++;
  let updateTimer, controller;
  try {
    if(updater)controller=new AbortController();
    const operation=(async()=>{
      const response = await fetch(path, {...requestOptions, headers,...(controller?{signal:controller.signal}:{})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Local request failed.');
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
      if(kind && (dirty || hasPassageDrafts()))throw new Error('Save or discard your latest edits before exporting.');
    };
    if(kind){if(hasPassageDrafts())throw new Error('Save or discard passage drafts before exporting.');await save();checkMeeting();}
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
  button.disabled = updateFrozen || saving || !dirty;
  (button.querySelector('.shortcut-label') || button).textContent = saving ? 'Saving…' : dirty ? 'Save now' : 'Saved';
  button.classList.toggle('is-dirty', dirty && !saving);
  button.dataset.shortcutTitle = dirty ? 'Edits save automatically. Save now' : 'All changes saved on this Mac';
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
      const status=liveStatuses.includes(job.status)?` · ${job.status==='starting'?'Starting':job.status==='finishing'?'Finishing':job.status==='paused'?'Paused':'Recording'}`:job.status==='failed'?' · Interrupted':'';
      button.title=job.name;button.querySelector('strong').textContent=job.name;
      button.querySelector('small').textContent=date.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'})+status;
      button.dataset.signature=signature;
    }
    setPassageClass(button,'active',selected?.id===job.id);place(button);
  }
  for(const child of Array.from(host.children))if(!retained.has(child))child.remove();
}
function languageReady(mode) {
  return config.readiness.configured && config.readiness.automatic_language
    && setupState?.core_ready !== false && setupState?.alignment?.ready !== false;
}
function setStatus() {
  $('recording-name').textContent = selected.name; $('recording-name').title = selected.name;
  $('meeting-date').textContent = new Date(selected.created*1000).toLocaleString(undefined,{weekday:'long',month:'long',day:'numeric',hour:'numeric',minute:'2-digit'});
  $('job-message').textContent = selected.message;
  const live=isLive(), busy=live || ['preparing','processing','queued'].includes(selected.status);
  $('editor').inert=updateFrozen || retrying || (busy && !live);
  $('recording-dot').hidden=!live;
  $('recording-dot').classList.toggle('paused',selected.status==='paused');
  $('run').hidden = !!doc?.segments?.length || live; $('run').disabled = busy || !languageReady(selected.language);
  $('run').title = languageReady(selected.language) ? '' : 'Download the local models in Settings → Models first.';
  $('manual').hidden = !!doc || busy || !selected.duration;
  $('delete').disabled = busy || !!selected.inference_owned;
  $('duration').textContent = selected.duration ? time(selected.duration) : 'Preparing…';
  $('segment-count').textContent = doc && !live ? passageCountLabel() : '';
  $('save').hidden = !doc || live; $('search').disabled = !doc;
  $('player').hidden = !selected.duration || live;
  document.querySelector('.player-panel').hidden=live;
  document.querySelector('.export-menu').inert=live || !doc;
  document.querySelector('.export-menu').hidden=live;
  $('speaker-tools').hidden=live;$('add').hidden=live;$('editor-hints').hidden=live || !doc;
  renderFollowLive();
  $('listening').hidden=!live || !!doc?.segments?.length;
  renderLiveLanguage();
  renderExportState();
  renderRefinementStatus();
  renderShortcutHints();
}
async function select(jid) {
  if(hasPendingPassageSaves()){notice('Wait for the passage save to finish before switching meetings.');return;}
  if(retrying){notice('Wait for the passage retry to start before switching recordings.');return;}
  if (identityBusy) { notice('Wait for the name change to finish before switching meetings.'); return; }
  $('editor').inert = true;
  let result, generation;
  try {
    if (!(await saveBeforeLeaving())) return;
    if (hasPassageDrafts() && !confirm('Discard unsaved passage drafts and open another recording?')) return;
    passageDrafts.clear();recoverablePassageDrafts.clear();resetPlayback();$('segments').replaceChildren();
    generation = ++selectionGeneration;
    result = await api(`/api/jobs/${jid}`);
  } finally { if (!identityBusy) $('editor').inert = false; }
  if (generation !== selectionGeneration) return;
  selected = result; doc = selected.document ? structuredClone(selected.document) : null; dirty = false;
  passageEnd=null;
  $('empty').hidden = true; $('workspace').hidden = false;
  if(!isLive() && selected.duration) $('player').src = `/api/jobs/${jid}/audio`;
  else {$('player').pause();$('player').removeAttribute('src');}
  $('player').playbackRate = Number($('speed').value);
  $('search').value = ''; notice(''); setStatus(); renderJobs(); renderEditor();
}
async function saveBeforeLeaving() {
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
function nameSpeaker(id) {
  if(updateFrozen){notice('Cancel the update before changing a speaker name.',true);return;}
  flushSave().then(() => openNamePicker(id)).catch(error => notice(error.message, true));
}
function speakerLabel(id) {
  if(id==='unassigned')return 'Speaker unassigned';
  return doc.speakers[id] || (/^speaker_\d+$/.test(id)?`Speaker ${Number(id.slice(8))+1}`:'Unknown speaker');
}
function speakerNameControl(id,className='name-speaker') {
  const assigned=id!=='unassigned',label=speakerLabel(id);
  const control=node(assigned?'button':'span',label,className);control.dataset.speaker=id;
  if(assigned) {
    control.type='button';control.title='Name this speaker or remember their voice';
    control.setAttribute('aria-label',`Name or remember voice for ${label}`);
    control.addEventListener('click',()=>nameSpeaker(id));
  }else {
    control.title='These words have no assigned speaker.';
    control.setAttribute('aria-label',label);
  }
  return control;
}
function removePassage(segment) {
  const index = doc.segments.indexOf(segment), jid = selected.id;
  if (index < 0) return;
  clearTimeout(undoRemoval?.timer);
  const removal = {jid, timer: setTimeout(() => { if (undoRemoval === removal) { undoRemoval = null; scheduleAutosave(); } }, 8000)};
  undoRemoval = removal;
  doc.segments.splice(index, 1); changed(); renderEditor();
  notice('Passage removed.', false, {label: 'Undo', run: () => {
    if (undoRemoval !== removal || selected?.id !== jid || !doc) { notice('That removal is already saved and can’t be undone.'); return; }
    clearTimeout(removal.timer); undoRemoval = null;
    doc.segments.splice(Math.min(index, doc.segments.length), 0, segment); changed(); renderEditor();
    document.querySelector(`[data-segment-id="${CSS.escape(segment.id)}"] textarea`)?.focus();
  }});
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
  for (const id of Object.keys(doc.speakers)) {
    const option = node('option', speakerLabel(id)); option.value = id; option.selected = id === current; selectEl.append(option);
  }
}
function renderEditor() {
  $('inspector').hidden = true;
  $('editor').hidden = !doc; if (!doc) return;
  renderSaveState();
  $('segment-count').textContent = isLive() ? '' : passageCountLabel();
  $('speaker-list').replaceChildren();
  Object.keys(doc.speakers).forEach((id, i) => {
    const row = node('div', undefined, 'speaker-name');
    const dot = node('span', undefined, 'speaker-dot'); dot.dataset.color = i % 8;
    const choose = speakerNameControl(id,'speaker-name-action name-speaker');
    choose.replaceChildren(dot, node('span', speakerLabel(id)));
    if(id!=='unassigned')choose.append(icon('pencil'));
    row.append(choose); $('speaker-list').append(row);
  });
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
  const signature=JSON.stringify([turns,doc.speakers]);
  if(signature!==binding.signature) {
    binding.signature=signature;host.replaceChildren();
    const groups=[];
    for(const turn of turns || []) {
      const previous=groups.at(-1);
      if(previous && previous.speaker===turn.speaker && previous.attribution===turn.attribution) {
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
            groups[index+2]?.attribution==='single' && groups[index+2].speaker===turn.speaker) {
        const unknown=groups[index+1],following=groups[index+2];
        turn.text+=unknown.text+following.text;turn.slices.push(...unknown.slices,...following.slices);
        turn.includesUnassigned=true;
        if(turn.start!==null && unknown.start!==null && following.start!==null)turn.end=following.end;
        else turn.start=turn.end=null;
        index+=2;
      }
      paragraphs.push(turn);
    }
    for(const turn of paragraphs) {
      const row=node('section',undefined,'reading-turn');
      row.dataset.attribution=turn.includesUnassigned?'single_with_unassigned':turn.attribution;
      const label=node('div',undefined,'reading-turn-meta');label.append(node('span',readingTurnLabel(turn),'reading-turn-speaker'));
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
  button.disabled=blocked;button.textContent=reading?'Edit passage':'View speaker turns';
  setPassageClass(card,'has-reading-turns',!!turns);setPassageClass(card,'editing-reading-turns',!reading);
}
function savedPassageSignature(segment) {
  return JSON.stringify([segment,doc.speakers,doc.provenance,selected.speaker_assignments?.[segment.speaker],
    ['preparing','processing','queued'].includes(selected.status)]);
}
function savedCard(segment) {
  const card = node('article', undefined, 'segment'); card.dataset.segmentId = segment.id;card.dataset.speaker=segment.speaker;
  const index = Object.keys(doc.speakers).indexOf(segment.speaker);
  const avatar=node('span',segment.speaker==='unassigned'?'?':`S${index+1}`,'speaker-avatar');avatar.setAttribute('aria-hidden','true');avatar.dataset.color=segment.speaker==='unassigned'?-1:index%8;
  const body=node('div',undefined,'segment-body');
  const top = node('div', undefined, 'segment-top');
  const seek = node('button', undefined, 'seek'); seek.append(icon('play'),node('span',time(segment.start))); seek.setAttribute('aria-label', `Play passage from ${passageTime(segment.start)} to ${passageTime(segment.end)}`);seek.title=`Play passage ${passageTime(segment.start)}–${passageTime(segment.end)}`;
  seek.addEventListener('click', () => { passageEnd=segment.end; $('player').currentTime = segment.start; $('player').play().catch(e => notice(e.message, true)); });
  const speaker = node('select'); speakerOptions(speaker, segment.speaker); speaker.setAttribute('aria-label', 'Segment speaker');
  speaker.disabled=!!isLive();
  seek.disabled=!!isLive();
  speaker.title='Change who spoke this passage';
  speaker.addEventListener('change', () => { segment.speaker = speaker.value; segment.review = true; changed(); renderSegments(); });
  body.append(seek);top.append(speaker);
  if(selected.speaker_assignments?.[segment.speaker]?.source==='automatic_voice') {
    const recognized=node('span','Recognized','review-tag routine-state');recognized.title='Matched a saved voice. Choose the speaker name to correct it.';top.append(recognized);
  }
  const actions=node('div',undefined,'segment-actions');
  const repairToggle=node('button',undefined,'segment-action repair-toggle');repairToggle.type='button';
  repairToggle.append(icon('wrench'),node('span','Repair'));repairToggle.title='Retry this passage in another language';
  repairToggle.setAttribute('aria-label',`Repair passage at ${passageTime(segment.start)}`);repairToggle.setAttribute('aria-expanded','false');
  const remove = node('button', undefined, 'segment-action remove-segment'); remove.type='button'; remove.append(icon('trash-2')); remove.setAttribute('aria-label', `Remove passage at ${passageTime(segment.start)}`); remove.title='Remove passage (you can undo)';
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
  const repair=body.querySelector(':scope > .passage-repair');
  if(repair){repair.classList.add('inline-repair');
    repairToggle.addEventListener('click',()=>{repair.open=!repair.open;if(repair.open)repair.querySelector('select')?.focus();});
    repair.addEventListener('toggle',()=>repairToggle.setAttribute('aria-expanded',String(repair.open)));
  }else repairToggle.hidden=true;
  card.append(avatar,body);refreshReadingTurnView(card,segment);
  card.dataset.meetingId=selected.id;savedPassageBindings.set(card,segment);
  // Defer a changed focused row until focus leaves, rather than losing its caret.
  card.addEventListener('focusout',()=>queueMicrotask(()=>{
    if(card.isConnected && !card.contains(document.activeElement) && card.dataset.stale==='true')renderSegments();
  }));
  return card;
}
function renderSegments() {
  if(isLive() || hasPassageDrafts() || hasRecoverablePassageDrafts() || hasPendingPassageSaves() || doc.segments.some(segment=>segment.refinement_state==='provisional') || ['waiting','refining'].includes(selected.refinement_status)){renderLiveSegments();return;}
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
    refreshReadingTurnView(card,segment);
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
function appendPassageRepair(body,segment,reason) {
  if(!isLive()) {
    const repair=node('details',undefined,'passage-repair');repair.append(node('summary','Repair passage'));
    const tools=node('div',undefined,'passage-retry'), language=node('select');
    language.setAttribute('aria-label',`Choose language to retry passage at ${time(segment.start)}`);
    const choose=node('option','Choose passage language');choose.value='';language.append(choose);
    for(const [code,name] of Object.entries(config.languages)) {
      if(code==='auto')continue;
      const option=node('option',name);option.value=code;language.append(option);
    }
    language.value=segment.language || '';
    const retry=node('button','Retry passage locally');
    const busy=['preparing','processing','queued'].includes(selected.status);
    const tooLong=segment.end-segment.start>24.5;
    retry.disabled=busy || tooLong;language.disabled=busy;
    if(tooLong)retry.title='Choose a passage of at most 24.5 seconds before retrying.';
    retry.addEventListener('click',async()=>{
      try {
        await flushSave();
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
    if(reason && segment.text.trim()) {
      const reviewed=node('button','Mark words reviewed');reviewed.disabled=busy;
      reviewed.addEventListener('click',()=>{segment.review_resolution='words_reviewed';segment.review=hasOverlappingSpeakers(segment);changed();renderSegments();});
      tools.append(reviewed);
    }
    const candidate=segment.retry_candidate;
    if(candidate) {
      const result=node('div',undefined,'retry-result');
      result.append(node('p',candidate.error || `Retry result (${config.languages[candidate.language]}). Listen before choosing this text.`));
      if(candidate.text) {
        result.append(node('p',candidate.text,'segment-text'));
        if(candidate.transcription_review)result.append(node('p',reviewReasons[candidate.transcription_review.reason] || 'Retry needs review','passage-review'));
        const currentCandidate=()=>candidate.start===segment.start && candidate.end===segment.end && candidate.speaker===segment.speaker;
        const use=node('button','Use this text');use.disabled=busy || !currentCandidate();
        use.addEventListener('click',()=>{
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

const reviewReasons={insufficient_acoustic_context:'Too little acoustic context to decode; audio retained',short_acoustic_context:'Unconfirmed words from a short audio fragment; review the recording',possible_non_speech:'Audio may contain no speech; model words need review',uncertain:'Language uncertain; audio retained',change_pending:'Language change uncertain; audio retained',unsupported:'Detected language unsupported; audio retained',insufficient_speech:'Too little usable speech; audio retained',overlapping_speech:'Overlapping speakers; voices are not separated',refinement_incomplete:'Larger-context result was incomplete; previous words and audio retained',empty_result:'No transcript text returned; audio retained',transcription_failed:'Transcription failed for this passage; audio retained',token_limit:'Transcript may be incomplete; audio retained',unassigned_audio:'Audio outside detected speech; may be silence or missed speech'};
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
  if(segment.review_resolution==='words_reviewed' && segment.text.trim())return hasOverlappingSpeakers(segment) ? reviewReasons.overlapping_speech : segment.speaker==='unassigned' ? 'Speaker uncertain' : '';
  if(segment.transcription_review?.reason==='refinement_conflict')return 'Previous words retained because the new passage also covers another passage needing review.';
  if(['transcription_failed','empty_result','token_limit'].includes(segment.transcription_review?.reason))return reviewReasons[segment.transcription_review.reason];
  if(hasOverlappingSpeakers(segment))return reviewReasons.overlapping_speech;
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
  const evidence=node('details',undefined,'passage-evidence');evidence.append(node('summary','Inference evidence'));
  const metadata={};
  for(const key of ['review','review_resolution','refinement_state','language','language_detection','confidence','speaker_candidates','speaker_track_mapping','voice_eligible','audio_state','transcription_review','audio_anchor','timing','provenance','refinement_window','machine_revision']) {
    if(segment[key]!==undefined)metadata[key]=segment[key];
  }
  if(doc.provenance)metadata.recording_provenance=doc.provenance;
  const text=node('pre',JSON.stringify(metadata,null,2));text.style.whiteSpace='pre-wrap';text.style.overflowWrap='anywhere';
  evidence.append(text);host.append(evidence);
}
function showTranscriptPassage(segment) {
  return !!segment.text.trim() || passageDrafts.has(segment.id) || !!recoverablePassageDrafts.get(segment.id)?.text.trim() || segment.protected_fields?.includes('text') || segment.refinement_state==='edited';
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
  const rows=doc.segments.filter(segment=>!showTranscriptPassage(segment) && !isEmptyNonSpeechPlaceholder(segment) &&
    (segment.transcription_review?.reason || ['unsupported','uncertain','change_pending','needs_language','insufficient_speech'].includes(segment.language_detection?.reason) ||
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
      row.append(play);appendPassageRepair(row,segment,passageReviewReason(segment));
    }
    list.append(row);
  }
  host.append(list);host.open=open;
}
// Group presentation only: each passage keeps its own edit, timing, retry and
// playback targets. Recompute after filtering/reconciliation without merging IDs.
function groupConsecutivePassages(host,pending=0) {
  let previous,previousSection;
  const divided=pending>0 || !!host.querySelector('.rolling-segment[data-transcript-section="live"],.rolling-segment[data-transcript-section="corrections"]');
  for(const card of host.children) {
    const section=card.dataset.transcriptSection || 'processed',start=section!==previousSection;
    const continuation=!start && previous && card.dataset.speaker!=='unassigned' && previous.dataset.speaker===card.dataset.speaker;
    setPassageClass(card,'speaker-continuation',!!continuation);
    setPassageClass(card,'passage-exception',!!card.querySelector('.review-tag:not(.routine-state),.refinement-badge:not(.routine-state)'));
    const live=section==='live';
    setPassageClass(card,'live-provisional',live);
    setPassageClass(card,'live-section-start',live && start);
    setPassageClass(card,'retained-correction',section==='corrections');
    setPassageClass(card,'transcript-section-start',divided && start);
    const className=`${section}-section-label`;
    for(const label of card.querySelectorAll('.live-section-label,.processed-section-label,.corrections-section-label'))if(!start || !divided || !label.classList.contains(className))label.remove();
    if(start && divided && !card.querySelector(`.${className}`)) {
      const label=node('span',({processed:'Processed',corrections:'Your corrections · replaced passages',live:'Live'})[section],`transcript-section-label ${className}`);
      label.setAttribute('role','heading');label.setAttribute('aria-level','2');
      if(live){label.title='These words are awaiting refinement and may change with more context.';label.setAttribute('aria-label','Live words awaiting refinement');}
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
function showInspector(segment,card,force=false) {
  document.querySelectorAll('.segment.active').forEach(item => item.classList.remove('active'));
  card.classList.add('active');
  if(narrowLayout() && !force && $('inspector').hidden)return;
  $('inspector').hidden=false;
  const details=$('segment-details');details.replaceChildren();
  for(const [key,label] of [['start','Start time (seconds)'],['end','End time (seconds)']]) {
    const field=node('label',label,'segment-detail-field'), input=node('input');
    input.type='number';input.min=0;input.max=selected.duration;input.step='.01';input.value=segment[key].toFixed(2);
    const hint=node('small',passageTime(segment[key]),'time-hint');
    input.addEventListener('change',() => { segment[key]=Number(input.value);segment.timing='user_edited';hint.textContent=passageTime(segment[key]);changed(); });
    field.append(input,hint);details.append(field);
  }
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
  const generation = editGeneration, jid = selected.id;
  saving = true; renderSaveState();
  try {
    const result = await api(`/api/jobs/${jid}/transcript`, {method: 'PUT', updateSave: true, body: JSON.stringify({revision: selected.revision, document: doc})});
    if (selected?.id !== jid) return;
    selected = result;
    if (generation === editGeneration) {
      dirty = false;
      // Keep the live document while someone is typing so the caret and the
      // passage handlers stay attached; adopt the saved copy otherwise.
      if (!$('editor').contains(document.activeElement)) { doc = structuredClone(result.document); renderEditor(); }
    }
    setStatus();
    if (!quiet) notice('Saved on this Mac.');
  } finally { saving = false; renderSaveState(); if (dirty) scheduleAutosave(); }
}
async function upload(files) {
  if (!files.length) return;
  if (!(await saveBeforeLeaving())) return;
  if(changingLiveLanguage || changingDefaultLanguage)throw new Error('Wait for the language change to finish before importing.');
  const data = new FormData(); Array.from(files).forEach(file => data.append('files', file)); data.append('language', config.default_language || 'auto');
  $('files').disabled = true; notice('Uploading to the local app…');
  try {
    const created = await api('/api/jobs', {method: 'POST', body: data});
    jobs = await api('/api/jobs'); dirty = false; await select(created[0].id);
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
  $('theme').addEventListener('change',()=>speakerdeskTheme.set($('theme').value));
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
  $('install-models').addEventListener('click',() => installModels().catch(error => { $('model-status').textContent=error.message; }));
  $('recognize-voices').addEventListener('change',async event => {
    const control=event.target;control.disabled=true;
    try { await api('/api/recognition',{method:'PATCH',body:JSON.stringify({enabled:control.checked})}); }
    catch(error) { control.checked=!control.checked;notice(error.message,true); }
    finally { control.disabled=false;await refreshSetup(); }
  });
  $('run').addEventListener('click', async () => { try { selected = await api(`/api/jobs/${selected.id}/run`, {method: 'POST'}); setStatus(); } catch (e) { notice(e.message, true); } });
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
  $('add-speaker').addEventListener('click', () => { const id = `speaker_${crypto.randomUUID().slice(0,8)}`; doc.speakers[id] = `Speaker ${Object.keys(doc.speakers).length + 1}`; changed(); renderEditor(); });
  $('import').addEventListener('change', async () => {
    try {
      if(hasPendingPassageSaves())throw new Error('Wait for the passage save to finish before importing.');
      const file = $('import').files[0]; if (!file) return;
      document.querySelector('.export-menu').open = false;
      if (!(await saveBeforeLeaving())) return;
      if (doc?.segments?.length && !confirm(`Replace the transcript of ${selected.name} with ${file.name}?`)) return;
      saving = true; $('editor').inert = true;
      if (file.size > 5 * 1024 * 1024) throw new Error('Transcript JSON must be at most 5 MB.');
      const incoming = JSON.parse(await file.text());
      if (!incoming.speakers || !Array.isArray(incoming.segments)) throw new Error('JSON needs speakers and segments. See fixtures/conversation.json.');
      // Validate and persist atomically before showing imported data.
      const result = await api(`/api/jobs/${selected.id}/transcript`, {method: 'PUT', body: JSON.stringify({revision: selected.revision, document: incoming, imported: true})});
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
    clearTimeout(autosaveTimer); autosaveTimer = null;
    if (undoRemoval) { clearTimeout(undoRemoval.timer); undoRemoval = null; }
    try {
      await api(`/api/jobs/${selected.id}`, {method: 'DELETE'}); passageDrafts.clear();selected = doc = null;resetPlayback(); dirty = false;
      $('player').pause(); $('player').removeAttribute('src'); $('workspace').hidden = true; $('empty').hidden = false;
      jobs = await api('/api/jobs'); renderJobs(); notice('Recording deleted from local storage.');
    } catch (e) { notice(e.message, true); if (dirty) scheduleAutosave(); }
  });
  window.addEventListener('beforeunload', e => { if (dirty || hasPassageDrafts()) { e.preventDefault(); e.returnValue = ''; } });
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
  renderShortcutHints();
  try {
    await refreshConfig();
    Object.entries(config.languages).forEach(([code, name]) => {
      for(const id of ['language','live-language']){const option=node('option',id==='live-language' && code==='auto'?'Auto':name);option.value=code;$(id).append(option);}
    });
    renderLanguageDefault();
    wire(); jobs = await api('/api/jobs'); renderJobs(); setInterval(poll, 1000);
    await refreshMeeting();setInterval(refreshMeeting,500);
    const requested=new URLSearchParams(location.search).get('meeting');
    if(requested && jobs.some(j=>j.id===requested))await select(requested);
    else if(meeting?.id && liveStatuses.includes(meeting.status)) await select(meeting.id);
    await refreshSetup();
    setInterval(refreshSetup,1500);
    await initializeUpdates();setInterval(refreshUpdates,1000);
    renderIcons();
  } catch (e) { notice(e.message, true); }
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
function renderStartState() {
  if(!config)return;
  const modelsReady=languageReady($('language').value), capture=meeting?.capture_available!==false;
  const setupBusy=setupState?.status==='downloading';
  const state=setupState, busy=state?.status==='downloading';
  $('setup-card').hidden=modelsReady || !state;
  if(state && !modelsReady) {
    const percent=state.total_bytes?Math.round(100*state.downloaded_bytes/state.total_bytes):0;
    $('setup-card-text').textContent=!state.supported?'This Mac can’t run the local models. Speakerdesk needs Apple Silicon and macOS 15 or later.'
      :`Download the local models once (${formatBytes(state.total_bytes)}). After that, meetings are transcribed offline on this Mac.`;
    $('setup-card-progress').hidden=!busy;$('setup-card-progress').value=percent;
    $('setup-card-status').textContent=state.error || (busy?`${state.phase} · ${percent}%`:'');
    $('setup-card-install').disabled=busy || !state.supported || state.voice?.status==='downloading';
    $('setup-card-install').querySelector('span').textContent=busy?'Downloading…':state.status==='failed'?'Resume download':`Download models · ${formatBytes(state.total_bytes)}`;
  }
  $('start-meeting').disabled=setupBusy || !modelsReady || !capture || changingLiveLanguage || changingDefaultLanguage;
  $('start-hint').textContent=!capture?'Live recording isn’t available in this build. Use Import recording in the sidebar to transcribe a file.'
    :setupBusy?'Finish model setup before starting a meeting.':!modelsReady?'Start meeting is available once the models are downloaded.':'';
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
  const busy=alignment.status==='downloading';
  const languages=alignment.supported_languages.map(code=>config?.languages[code]||code).join(', ');
  const calibrated=(alignment.timing_accuracy_calibrated_languages||[]).map(code=>config?.languages[code]||code).join(', ');
  $('alignment-coverage').textContent=`Supported languages: ${languages||'None'}. Timing accuracy measured against references: ${calibrated||'None'}.`;
  if(alignment.supported_languages.some(code=>!(alignment.timing_accuracy_calibrated_languages||[]).includes(code)))
    $('alignment-coverage').textContent+=' Accuracy in other supported languages has not been independently measured.';
  $('alignment-cost').textContent=`${formatBytes(alignment.total_bytes)} download · ${formatBytes(alignment.stored_bytes)} stored · up to ${formatBytes(alignment.required_free_bytes)} free space needed.`;
  $('alignment-message').textContent=alignment.message;
  $('alignment-status').textContent=alignment.error || (busy?`${alignment.phase} · ${Math.round(100*alignment.downloaded_bytes/alignment.total_bytes)}%`:alignment.ready?'Ready':'Not ready');
  $('alignment-progress').hidden=!busy;
  $('alignment-progress').value=alignment.total_bytes?100*alignment.downloaded_bytes/alignment.total_bytes:0;
  $('install-alignment').hidden=alignment.ready && alignment.status!=='failed';
  $('install-alignment').disabled=alignmentChanging || !alignment.can_download;
  $('install-alignment').textContent=busy?'Setting up…':['failed','interrupted'].includes(alignment.status)?'Retry setup': 'Finish setup';
}
async function refreshSetup() {
  const generation=alignmentGeneration;
  try {
    const state=await api('/api/setup');
    if(generation!==alignmentGeneration || alignmentChanging)return;
    setupState=state;
    const list=$('model-list');list.replaceChildren();
    state.models.forEach(model => {
      const size=formatBytes(model.bytes);
      const item=node('div',undefined,'model-item');
      const alignment=model.id===state.alignment?.id?state.alignment:null;
      const label=model.name==='Voice recognition' && state.voice?.released===false?'Unavailable':alignment?(alignment.ready?'Ready':alignment.status==='downloading'?'Setting up…':'Needs setup'):model.installed?'Installed':model.optional?`Optional · ${size}`:size;
      item.append(node('strong',model.name),node('span',label));list.append(item);
    });
    renderAlignment(state);
    if(state.required_free_bytes)$('setup-disk-note').textContent=`First setup needs an internet connection and about ${formatBytes(state.required_free_bytes)} of free disk space including prepared timing weights.`;
    const busy=state.status==='downloading';
    $('model-progress').hidden=!busy;
    $('model-progress').value=100*state.downloaded_bytes/state.total_bytes;
    $('model-status').textContent=state.error || (busy?`${state.phase} · ${Math.round(100*state.downloaded_bytes/state.total_bytes)}%`:state.ready?'Ready to transcribe on this Mac.':'Finish required model setup before transcribing.');
    $('install-models').hidden=state.ready;
    const voice=state.voice,voiceBusy=voice.status==='downloading';
    $('install-models').disabled=busy || voiceBusy || !state.supported;
    $('install-models').textContent=busy?'Downloading…':state.status==='failed'?'Resume download':`Download models · ${formatBytes(state.total_bytes)}`;
    if(voiceAvailable!==voice.available) await loadPeople();
    $('settings-voice-note').textContent=voice.enabled?voice.message:'Recognition is off. Saved names and voices stay on this Mac.';
    if(document.activeElement!==$('recognize-voices')) $('recognize-voices').checked=voice.enabled;
    if(state.core_ready && (!config.readiness.configured || !config.readiness.automatic_language)) {
      await refreshConfig();renderLanguageDefault();if(selected)setStatus();
    }
    renderStartState();
  } catch(error) { $('model-status').textContent=error.message; }
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
  const generation=defaultLanguageGeneration, next=await api('/api/config');
  // A config read started before a preference commit must not undo it.
  if(config && (generation!==defaultLanguageGeneration || changingLiveLanguage || changingDefaultLanguage))next.default_language=config.default_language;
  config=next;
}
function renderLanguageDefault() {
  const control=$('language');
  if(!changingLiveLanguage && !changingDefaultLanguage)control.value=config.languages[config.default_language]?config.default_language:'auto';
  control.disabled=changingLiveLanguage || changingDefaultLanguage || ['starting','finishing'].includes(meeting?.status);
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
  if(meeting?.id && ['recording','paused'].includes(meeting.status))return changeLiveLanguage(language);
  if(['starting','finishing'].includes(meeting?.status)){renderLanguageDefault();return;}
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
  $('editor').inert = frozen || retrying || (!!selected && ['preparing','processing','queued'].includes(selected.status));
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
init();
