'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="speakerdesk-token"]').content;
let config, jobs = [], selected = null, doc = null, dirty = false, polling = false, saving = false, editGeneration = 0, selectionGeneration = 0;
let meeting = null, followingLive = true, passageEnd = null, retrying = false;
let changingLiveLanguage = false;
let pendingExport = null, autosaveTimer = null, setupState = null, undoRemoval = null;
const narrowLayout = () => matchMedia('(max-width: 900px)').matches;
const liveStatuses = ['starting','recording','paused','finishing'];
const isLive = () => selected && liveStatuses.includes(selected.status);

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
async function api(path, options = {}) {
  const headers = {'X-Speakerdesk-Token': token, ...options.headers};
  if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {...options, headers});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Local request failed.');
  return data;
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
  document.querySelectorAll('[data-export]').forEach(button=>{button.disabled=!!pendingExport || !doc?.segments?.length;});
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
  button.disabled = saving || !dirty;
  button.textContent = saving ? 'Saving…' : dirty ? 'Save now' : 'Saved';
  button.classList.toggle('is-dirty', dirty && !saving);
  button.title = dirty ? 'Edits save automatically. Press ⌘S to save now.' : 'All changes saved on this Mac';
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
async function flushSave() {
  clearTimeout(autosaveTimer); autosaveTimer = null;
  while (dirty || saving) {
    if (saving) await new Promise(resolve => setTimeout(resolve, 50));
    else await save({quiet: true});
  }
}
function changed() { editGeneration++; dirty = true; renderSaveState(); scheduleAutosave(); }
function time(seconds) {
  const minutes = Math.floor(seconds / 60), rest = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2,'0')}`;
}
function passageTime(seconds) {
  const ms=Math.round(seconds*1000),minutes=Math.floor(ms/60000),rest=ms%60000;
  return `${String(minutes).padStart(2,'0')}:${String(Math.floor(rest/1000)).padStart(2,'0')}.${String(rest%1000).padStart(3,'0')}`;
}
function renderJobs() {
  $('job-count').textContent = jobs.length; $('jobs').replaceChildren();
  const query=$('meeting-search').value.toLowerCase();let lastDate='';
  for (const job of jobs) {
    if(query && !job.name.toLowerCase().includes(query))continue;
    const date=new Date(job.created*1000), today=new Date(), yesterday=new Date();yesterday.setDate(today.getDate()-1);
    const group=date.toDateString()===today.toDateString()?'Today':date.toDateString()===yesterday.toDateString()?'Yesterday':date.toLocaleDateString(undefined,{month:'short',day:'numeric'});
    if(group!==lastDate){$('jobs').append(node('div',group,'date-group'));lastDate=group;}
    const button = node('button', undefined, 'job' + (selected?.id === job.id ? ' active' : ''));
    const status=liveStatuses.includes(job.status)?` · ${job.status==='starting'?'Starting':job.status==='finishing'?'Finishing':job.status==='paused'?'Paused':'Recording'}`:job.status==='failed'?' · Interrupted':'';
    button.title = job.name;
    button.append(node('strong', job.name), node('small', date.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'})+status));
    button.addEventListener('click', () => select(job.id).catch(e => notice(e.message, true)));
    $('jobs').append(button);
  }
}
function languageReady(mode) {
  return config.readiness.configured && (mode !== 'auto' || config.readiness.automatic_language);
}
function setStatus() {
  $('recording-name').textContent = selected.name; $('recording-name').title = selected.name;
  $('meeting-date').textContent = new Date(selected.created*1000).toLocaleString(undefined,{weekday:'long',month:'long',day:'numeric',hour:'numeric',minute:'2-digit'});
  $('job-message').textContent = selected.message;
  const live=isLive(), busy=live || ['preparing','processing','queued'].includes(selected.status);
  $('editor').inert=retrying || (busy && !live);
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
    passageDrafts.clear();recoverablePassageDrafts.clear();$('segments').replaceChildren();
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
  const shown = doc.segments.filter(showTranscriptPassage), flagged = shown.filter(segment => segment.review).length;
  return `${shown.length} passage${shown.length === 1 ? '' : 's'}${flagged ? ` · ${flagged} to review` : ''}`;
}
function renderFollowLive() {
  $('follow-live').hidden = !isLive() || followingLive;
}
function nameSpeaker(id) {
  flushSave().then(() => openNamePicker(id)).catch(error => notice(error.message, true));
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
  return !!segment.text.trim() && !(segment.speaker.startsWith('overlap') || segment.speaker_candidates?.length > 1 || segment.speaker === 'unassigned');
}
function renderSearchResults(visible, query) {
  $('no-results').hidden = visible > 0 || !query;
  $('no-results-text').textContent = `No passages match “${$('search').value.trim()}”.`;
  $('search-count').textContent = query ? `${visible} ${visible === 1 ? 'match' : 'matches'}` : '';
  document.querySelector('.transcript-search').classList.toggle('has-query', !!query);
}
function speakerOptions(selectEl, current) {
  selectEl.replaceChildren();
  for (const [id, name] of Object.entries(doc.speakers)) {
    const option = node('option', name); option.value = id; option.selected = id === current; selectEl.append(option);
  }
}
function renderEditor() {
  $('inspector').hidden = true;
  $('editor').hidden = !doc; if (!doc) return;
  renderSaveState();
  $('segment-count').textContent = isLive() ? '' : passageCountLabel();
  $('speaker-list').replaceChildren();
  Object.entries(doc.speakers).forEach(([id, name], i) => {
    const row = node('div', undefined, 'speaker-name');
    const dot = node('span', undefined, 'speaker-dot'); dot.dataset.color = i % 8;
    const choose = node('button', undefined, 'speaker-name-action name-speaker'); choose.type = 'button';
    choose.append(dot, node('span', name), icon('pencil'));
    choose.setAttribute('aria-label', `Name or remember voice for ${name}`);
    choose.title = 'Name this speaker or remember their voice';
    choose.addEventListener('click', () => nameSpeaker(id));
    row.append(choose); $('speaker-list').append(row);
  });
  $('timing-note').textContent = doc.provenance?.timing || 'User supplied segment boundaries.';
  $('warnings').textContent = (doc.warnings || []).join(' '); $('warnings').hidden = !doc.warnings?.length;
  renderSegments();
  refreshIdentitySuggestions();
  updateNameControls();
}
function renderSegments() {
  if(isLive() || hasPassageDrafts() || hasRecoverablePassageDrafts() || ['waiting','refining'].includes(selected.refinement_status)){renderLiveSegments();return;}
  $('pending-phrases').hidden=true;
  const pane=$('transcript-pane'), previousScroll=pane.scrollTop;
  const inspected=$('inspector').hidden?null:document.querySelector('#segments .segment.active')?.dataset.segmentId;
  $('segments').replaceChildren();
  const query = $('search').value.toLowerCase();
  let visible = 0;
  for (const segment of doc.segments) {
    if(!showTranscriptPassage(segment))continue;
    if (query && !(segment.text + ' ' + doc.speakers[segment.speaker]).toLowerCase().includes(query)) continue;
    visible++;
    const card = node('article', undefined, 'segment'); card.dataset.segmentId = segment.id;card.dataset.speaker=segment.speaker;
    const index = Object.keys(doc.speakers).indexOf(segment.speaker);
    const avatar=node('span',`S${index+1}`,'speaker-avatar');avatar.setAttribute('aria-hidden','true');avatar.dataset.color=index%8;
    const body=node('div',undefined,'segment-body');
    const top = node('div', undefined, 'segment-top');
    const seek = node('button', undefined, 'seek'); seek.append(icon('play'),node('span',passageTime(segment.start))); seek.setAttribute('aria-label', `Play passage from ${passageTime(segment.start)} to ${passageTime(segment.end)}`);seek.title='Play this passage';
    seek.addEventListener('click', () => { passageEnd=segment.end; $('player').currentTime = segment.start; $('player').play().catch(e => notice(e.message, true)); });
    const speaker = node('select'); speakerOptions(speaker, segment.speaker); speaker.setAttribute('aria-label', 'Segment speaker');
    speaker.disabled=!!isLive();
    seek.disabled=!!isLive();
    speaker.title='Change who spoke this passage';
    speaker.addEventListener('change', () => { segment.speaker = speaker.value; segment.review = true; changed(); renderSegments(); });
    top.append(seek, speaker);
    if(selected.speaker_assignments?.[segment.speaker]?.source==='automatic_voice') {
      const recognized=node('span','Recognized','review-tag');recognized.title='Matched a saved voice. Choose the speaker name to correct it.';top.append(recognized);
    }
    if (segment.review && !markReviewable(segment)) {
      const review=node('span','Needs review','review-tag');
      review.title=passageReviewReason(segment) || 'Speaker uncertain. Choose the speaker for this passage.';
      top.append(review);
    } else if (segment.review) {
      const review=node('button','Needs review','review-tag review-action');review.type='button';
      review.title='Play the passage to check the words, then click to mark them reviewed.';
      review.setAttribute('aria-label',`Mark passage at ${passageTime(segment.start)} as reviewed`);
      review.addEventListener('click',()=>{segment.review_resolution='words_reviewed';segment.review=false;changed();renderSegments();setStatus();});
      top.append(review);
    }
    if (!segment.language && segment.language_detection?.mode === 'auto') top.append(node('span', 'Language needs review', 'review-tag'));
    const actions=node('div',undefined,'segment-actions');
    const repairToggle=node('button',undefined,'segment-action repair-toggle');repairToggle.type='button';
    repairToggle.append(icon('wrench'),node('span','Repair'));repairToggle.title='Retry this passage in another language';
    repairToggle.setAttribute('aria-label',`Repair passage at ${passageTime(segment.start)}`);repairToggle.setAttribute('aria-expanded','false');
    const remove = node('button', undefined, 'segment-action remove-segment'); remove.type='button'; remove.append(icon('trash-2')); remove.setAttribute('aria-label', `Remove passage at ${passageTime(segment.start)}`); remove.title='Remove passage (you can undo)';
    remove.addEventListener('click', () => removePassage(segment));
    const details=node('button',undefined,'segment-action details-toggle');details.type='button';details.append(icon('sliders-horizontal'));
    details.setAttribute('aria-label',`Timing and details for passage at ${passageTime(segment.start)}`);details.title='Timing & details';
    details.addEventListener('click',()=>{showInspector(segment,card,true);$('segment-details').querySelector('input')?.focus();});
    actions.append(details,repairToggle,remove);actions.hidden=!!isLive();top.append(actions);
    const text = isLive()?node('p',segment.text,'segment-text'):node('textarea');
    if(!isLive()){text.value=segment.text;text.rows=2;text.setAttribute('aria-label',`Transcript at ${time(segment.start)}`);text.placeholder=segment.language_detection?.mode==='auto' && !segment.language?'Language uncertain or unsupported. Play this passage and enter its original words.':'';
      text.addEventListener('input',()=>{segment.text=text.value;text.style.height='auto';text.style.height=`${text.scrollHeight}px`;changed();});}

    card.addEventListener('focusin', event => {if(!isLive() && !event.target.closest('.segment-actions, .review-action'))showInspector(segment,card);});
    body.append(top,text);
    const reason=passageReviewReason(segment);
    if(reason)body.append(node('p',`${passageTime(segment.start)}–${passageTime(segment.end)} · ${reason}. Play this passage to review it.`, 'passage-review'));
    appendUncoveredAudio(body,segment);
    appendPassageRepair(body,segment,reason);
    const repair=body.querySelector(':scope > .passage-repair');
    if(repair){repair.classList.add('inline-repair');
      repairToggle.addEventListener('click',()=>{repair.open=!repair.open;if(repair.open)repair.querySelector('select')?.focus();});
      repair.addEventListener('toggle',()=>repairToggle.setAttribute('aria-expanded',String(repair.open)));
    }else repairToggle.hidden=true;
    card.append(avatar,body);$('segments').append(card);
    if(segment.id===inspected)card.classList.add('active');
  }
  groupConsecutivePassages($('segments'));
  $('segments').querySelectorAll('textarea').forEach(fitPassageText);
  renderRetainedAudioReview();
  if(inspected && !$('segments').querySelector('.segment.active'))$('inspector').hidden=true;
  renderSearchResults(visible, query);
  pane.scrollTop=isLive() && followingLive ? pane.scrollHeight : previousScroll;
  lucide.createIcons();
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
    retry.disabled=busy || segment.end-segment.start>30;language.disabled=busy;
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
      reviewed.addEventListener('click',()=>{segment.review_resolution='words_reviewed';segment.review=segment.speaker.startsWith('overlap') || segment.speaker_candidates?.length>1;changed();renderSegments();});
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
  if(segment.review_resolution==='words_reviewed' && segment.text.trim())return segment.speaker.startsWith('overlap') || segment.speaker_candidates?.length>1 ? reviewReasons.overlapping_speech : '';
  if(segment.transcription_review?.reason==='refinement_conflict')return 'Previous words retained because the new passage also covers another passage needing review.';
  if(['transcription_failed','empty_result','token_limit'].includes(segment.transcription_review?.reason))return reviewReasons[segment.transcription_review.reason];
  if(segment.speaker.startsWith('overlap') || segment.speaker_candidates?.length>1)return reviewReasons.overlapping_speech;
  if(segment.transcription_review?.reason)return reviewReasons[segment.transcription_review.reason] || 'Passage needs review; audio retained';
  if(segment.language_detection?.reason==='recent_context')return 'Language inferred from recent meeting context';
  if(segment.language_detection?.reason==='best_effort')return 'Best-effort words; language needs review';
  if(segment.language_detection?.reason==='needs_language')return 'Choose the meeting language; audio retained';
  if(segment.language_detection?.mode==='auto' && !segment.language)return reviewReasons[segment.language_detection.reason] || reviewReasons.uncertain;
  return segment.text.trim() ? '' : reviewReasons.empty_result;
}
function showTranscriptPassage(segment) {
  return !!segment.text.trim() || passageDrafts.has(segment.id) || !!recoverablePassageDrafts.get(segment.id)?.text.trim() || segment.protected_fields?.includes('text') || segment.refinement_state==='edited';
}
function renderRetainedAudioReview() {
  const host=$('retained-audio-review'),open=host.open;
  if(host.contains(document.activeElement))return;
  const rows=doc.segments.filter(segment=>!showTranscriptPassage(segment) && segment.audio_state!=='digital_silence' &&
    (segment.transcription_review?.reason || ['unsupported','uncertain','change_pending','needs_language','insufficient_speech'].includes(segment.language_detection?.reason) ||
     (segment.refinement_state==='unresolved' && segment.language_detection?.reason!=='silence')));
  host.hidden=!rows.length;host.replaceChildren();if(!rows.length)return;
  host.append(node('summary',`${rows.length} retained audio ${rows.length===1?'range needs':'ranges need'} review`));
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
function groupConsecutivePassages(host) {
  let previous;
  for(const card of host.children) {
    const continuation=previous && previous.dataset.speaker===card.dataset.speaker;
    card.classList.toggle('speaker-continuation',!!continuation);
    previous=card;
  }
}
function fitPassageText(text) {
  const key=JSON.stringify([text.clientWidth,getComputedStyle(text).paddingRight,text.value]);
  if(text.dataset.fit===key)return;
  text.style.height='auto';text.style.height=`${text.scrollHeight}px`;text.dataset.fit=key;
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
  if(segment.review) details.append(node('p','Check this passage against the recording.','inspector-note'));
}
async function save({quiet = false} = {}) {
  if (!dirty) return;
  if (saving) throw new Error('A save is in progress. Try again when it finishes.');
  clearTimeout(autosaveTimer); autosaveTimer = null;
  if (undoRemoval) { clearTimeout(undoRemoval.timer); undoRemoval = null; if (document.querySelector('#notice .notice-action')) notice(''); }
  const generation = editGeneration, jid = selected.id;
  saving = true; renderSaveState();
  try {
    const result = await api(`/api/jobs/${jid}/transcript`, {method: 'PUT', body: JSON.stringify({revision: selected.revision, document: doc})});
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
  const data = new FormData(); Array.from(files).forEach(file => data.append('files', file)); data.append('language', $('language').value);
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
function wire() {
  wirePeople();
  $('new-meeting').addEventListener('click', () => {
    if(hasPendingPassageSaves()){notice('Wait for the passage save to finish before starting another meeting.');return;}
    if(meeting && liveStatuses.includes(meeting.status)){select(meeting.id).catch(e=>notice(e.message,true));return;}
    startNewMeeting().catch(e=>notice(e.message,true));
  });
  async function startNewMeeting() {
    if(!(await saveBeforeLeaving()))return;
    if(hasPassageDrafts() && !confirm('Discard unsaved passage drafts?'))return;
    passageDrafts.clear();recoverablePassageDrafts.clear();selected=doc=null;dirty=false;selectionGeneration++;$('player').pause();$('player').removeAttribute('src');
    $('workspace').hidden=true;$('empty').hidden=false;renderJobs();renderStartState();$('meeting-title').focus();
  }
  $('refinement-toggle').addEventListener('click',toggleRefinement);
  $('meeting-search').addEventListener('input',renderJobs);
  $('language').addEventListener('change',()=>{try{localStorage.setItem('speakerdesk.language_mode',$('language').value);}catch{}renderStartState();});
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
      if(!languageReady($('language').value)){openSettings('models');throw new Error('Download the local models first, or choose a fixed meeting language in Settings.');}
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
  $('setup-toggle').addEventListener('click', () => $('setup').hidden ? openSettings() : closeSettings());
  $('inspector-close').addEventListener('click', () => {
    const active=document.querySelector('.segment.active');$('inspector').hidden=true;
    document.querySelectorAll('.segment.active').forEach(item => item.classList.remove('active'));
    active?.querySelector('textarea')?.focus({preventScroll:true});$('inspector').hidden=true;
  });
  $('clear-search').addEventListener('click',()=>{$('search').value='';renderSegments();$('search').focus();});
  $('search').addEventListener('keydown',event=>{if(event.key==='Escape' && $('search').value){event.preventDefault();$('search').value='';renderSegments();}});
  document.addEventListener('keydown', event => {
    if(!(event.metaKey || event.ctrlKey) || event.altKey || !$('setup').hidden || document.querySelector('dialog[open]'))return;
    const key=event.key.toLowerCase();
    if(key==='f' && !$('workspace').hidden && !$('search').disabled){event.preventDefault();$('search').focus();$('search').select();}
    if(key==='s' && !$('workspace').hidden){event.preventDefault();flushSave().then(()=>{if(doc && !isLive())notice('Saved on this Mac.');}).catch(e=>notice(e.message,true));}
  });
  $('settings-close').addEventListener('click',closeSettings);
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
  $('install-models').addEventListener('click',() => installModels().catch(error => { $('model-status').textContent=error.message; }));
  $('recognize-voices').addEventListener('change',async event => {
    const control=event.target;control.disabled=true;
    try { await api('/api/recognition',{method:'PATCH',body:JSON.stringify({enabled:control.checked})}); }
    catch(error) { control.checked=!control.checked;notice(error.message,true); }
    finally { control.disabled=false;await refreshSetup(); }
  });
  $('run').addEventListener('click', async () => { try { selected = await api(`/api/jobs/${selected.id}/run`, {method: 'POST'}); setStatus(); } catch (e) { notice(e.message, true); } });
  $('manual').addEventListener('click', manual);
  $('save').addEventListener('click', () => flushSave().catch(e => notice(e.message, true)));
  $('speed').addEventListener('change', () => { $('player').playbackRate = Number($('speed').value); });
  $('player').addEventListener('timeupdate', () => {
    if (!doc) return;
    const now = $('player').currentTime;
    if(passageEnd!==null && now>=passageEnd){$('player').pause();passageEnd=null;}
    document.querySelectorAll('.segment').forEach(card => {
      const s = doc.segments.find(item => item.id === card.dataset.segmentId);
      card.classList.toggle('playing', s.start <= now && now < s.end);
    });
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
      await api(`/api/jobs/${selected.id}`, {method: 'DELETE'}); passageDrafts.clear();selected = doc = null; dirty = false;
      $('player').pause(); $('player').removeAttribute('src'); $('workspace').hidden = true; $('empty').hidden = false;
      jobs = await api('/api/jobs'); renderJobs(); notice('Recording deleted from local storage.');
    } catch (e) { notice(e.message, true); if (dirty) scheduleAutosave(); }
  });
  window.addEventListener('beforeunload', e => { if (dirty || hasPassageDrafts()) { e.preventDefault(); e.returnValue = ''; } });
}
async function poll() {
  if (polling) return; polling = true;
  try {
    jobs = await api('/api/jobs'); renderJobs();
    if (selected && !dirty && !saving && !identityBusy && !retrying) {
      const generation = selectionGeneration, jid = selected.id;
      const next = await api(`/api/jobs/${jid}`);
      if (generation !== selectionGeneration || selected?.id !== jid || dirty || saving || identityBusy || retrying || next.revision < selected.revision) return;
      const changedAudio = !selected.duration && next.duration;
      const finished = next.status !== selected.status && !['preparing','queued','processing',...liveStatuses].includes(next.status);
      if (next.revision !== selected.revision) { doc = next.document ? structuredClone(next.document) : null; selected = next; if(isLive() || hasPassageDrafts()){renderSegments();refreshIdentitySuggestions();updateNameControls();}else renderEditor(); }
      selected = next; setStatus();
      if (finished) notice('');
      if ((changedAudio || finished) && !isLive() && selected.duration) $('player').src = `/api/jobs/${selected.id}/audio`;
      if(finished)renderEditor();
    }
  } catch (e) { notice(e.message, true); } finally { polling = false; }
}
async function init() {
  try {
    config = await api('/api/config');
    Object.entries(config.languages).forEach(([code, name]) => {
      for(const id of ['language','live-language']){const option=node('option',id==='live-language' && code==='auto'?'Auto':name);option.value=code;$(id).append(option);}
    });
    let previousLanguage='auto';try{previousLanguage=localStorage.getItem('speakerdesk.language_mode')||'auto';}catch{}
    $('language').value=config.languages[previousLanguage]?previousLanguage:'auto';
    wire(); jobs = await api('/api/jobs'); renderJobs(); setInterval(poll, 1000);
    await refreshMeeting();setInterval(refreshMeeting,500);
    const requested=new URLSearchParams(location.search).get('meeting');
    if(requested && jobs.some(j=>j.id===requested))await select(requested);
    else if(meeting?.id && liveStatuses.includes(meeting.status)) await select(meeting.id);
    await refreshSetup();
    setInterval(refreshSetup,1500);
    lucide.createIcons();
  } catch (e) { notice(e.message, true); }
}
function showSettingsTab(name) {
  document.querySelectorAll('.settings-tabs [role=tab]').forEach(tab=>{const on=tab.dataset.tab===name;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;});
  document.querySelectorAll('.settings-panel').forEach(panel=>{panel.hidden=panel.id!==`panel-${name}`;});
}
function openSettings(tab) { if(tab)showSettingsTab(tab);$('setup').hidden=false;$('setup-toggle').setAttribute('aria-expanded','true');document.querySelector('main').inert=true;document.querySelector('.sidebar').inert=true;$('settings-close').focus();loadPeople().catch(error=>notice(error.message,true)); }
function closeSettings() { $('setup').hidden=true;document.querySelector('main').inert=false;document.querySelector('.sidebar').inert=false;$('setup-toggle').setAttribute('aria-expanded','false'); $('setup-toggle').focus(); }
function formatBytes(bytes) {
  return bytes<1e8?`${Math.round(bytes/1e6)} MB`:`${(bytes/1e9).toFixed(1)} GB`;
}
async function installModels() {
  await api('/api/setup',{method:'POST'}); await refreshSetup();
}
function renderStartState() {
  if(!config)return;
  const modelsReady=languageReady($('language').value), capture=meeting?.capture_available!==false;
  const state=setupState, busy=state?.status==='downloading';
  $('setup-card').hidden=modelsReady || !state;
  if(state && !modelsReady) {
    const percent=state.total_bytes?Math.round(100*state.downloaded_bytes/state.total_bytes):0;
    const autoOnly=config.readiness.configured && !config.readiness.automatic_language;
    $('setup-card-text').textContent=!state.supported?'This Mac can’t run the local models. Speakerdesk needs Apple Silicon and macOS 15 or later.'
      :autoOnly?'Automatic language detection needs one more download. Download it, or choose a fixed meeting language.'
      :`Download the local models once (${formatBytes(state.total_bytes)}). After that, meetings are transcribed offline on this Mac.`;
    $('setup-card-progress').hidden=!busy;$('setup-card-progress').value=percent;
    $('setup-card-status').textContent=state.error || (busy?`${state.phase} · ${percent}%`:'');
    $('setup-card-install').disabled=busy || !state.supported || state.voice?.status==='downloading';
    $('setup-card-install').querySelector('span').textContent=busy?'Downloading…':state.status==='failed'?'Resume download':`Download models · ${formatBytes(state.total_bytes)}`;
  }
  $('start-meeting').disabled=!modelsReady || !capture;
  $('start-hint').textContent=!capture?'Live recording isn’t available in this build. Use Import recording in the sidebar to transcribe a file.'
    :!modelsReady?'Start meeting is available once the models are downloaded.':'';
}
async function refreshSetup() {
  try {
    const state=await api('/api/setup');setupState=state;
    const list=$('model-list');list.replaceChildren();
    state.models.forEach(model => {
      const size=formatBytes(model.bytes);
      const item=node('div',undefined,'model-item'); item.append(node('strong',model.name),node('span',model.installed?'Installed':size));list.append(item);
    });
    const busy=state.status==='downloading';
    $('model-progress').hidden=!busy;
    $('model-progress').value=100*state.downloaded_bytes/state.total_bytes;
    $('model-status').textContent=state.error || (busy?`${state.phase} · ${Math.round(100*state.downloaded_bytes/state.total_bytes)}%`:state.ready?'Ready to transcribe on this Mac.':'');
    $('install-models').hidden=state.ready;
    const voice=state.voice,voiceBusy=voice.status==='downloading';
    $('install-models').disabled=busy || voiceBusy || !state.supported;
    $('install-models').textContent=busy?'Downloading…':state.status==='failed'?'Resume download':`Download models · ${formatBytes(state.total_bytes)}`;
    if(voiceAvailable!==voice.available) await loadPeople();
    $('settings-voice-note').textContent=voice.enabled?voice.message:'Recognition is off. Saved names and voices stay on this Mac.';
    if(document.activeElement!==$('recognize-voices')) $('recognize-voices').checked=voice.enabled;
    if(state.core_ready && (!config.readiness.configured || !config.readiness.automatic_language)) {
      config=await api('/api/config');if(selected)setStatus();
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
function renderLiveLanguage() {
  const control=$('live-language');
  const current=meeting?.id===selected?.id && liveStatuses.includes(meeting?.status);
  $('meeting-language-control').hidden=!current;
  if(!changingLiveLanguage)control.value=meeting?.language || 'auto';
  control.disabled=changingLiveLanguage || !current || !['recording','paused'].includes(meeting?.status);
  const pending=(meeting?.language_acknowledged_revision || 0)<(meeting?.language_revision || 0);
  $('live-language-status').textContent=meeting?.language_revision?
    `From ${liveLanguageTime(meeting.language_from_sample || 0)}${pending?' · Queued':''}`:'This meeting only';
}
async function changeLiveLanguage() {
  if(changingLiveLanguage || !meeting?.id)return;
  const language=$('live-language').value,jid=meeting.id,revision=meeting.language_revision || 0;
  changingLiveLanguage=true;renderLiveLanguage();
  try {
    const next=await api(`/api/meetings/${jid}/language`,{method:'PATCH',body:JSON.stringify({language,language_revision:revision})});
    if(meeting?.id===jid && next.language_revision >= (meeting.language_revision || 0)) {
      meeting={...meeting,language:next.language,language_revision:next.language_revision,
        language_acknowledged_revision:Math.max(meeting.language_acknowledged_revision || 0,next.language_acknowledged_revision),
        language_from_sample:next.language_history.at(-1).start_sample};
      notice(`${config.languages[next.language]} applies from ${liveLanguageTime(meeting.language_from_sample)}. Earlier words keep their language.`);
    }
  } finally {
    changingLiveLanguage=false;renderLiveLanguage();await refreshMeeting();
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
    if(pauseButton.querySelector('span').textContent!==pauseLabel){pauseButton.replaceChildren(icon(meeting.status==='paused'?'play':'pause'),node('span',pauseLabel));lucide.createIcons();}
    $('pause-meeting').disabled=!['recording','paused'].includes(meeting.status);
    $('stop-meeting').disabled=meeting.status==='finishing';
  }catch(error){notice(error.message,true);}finally{meetingPoll=false;}
}
init();
