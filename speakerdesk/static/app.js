'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="speakerdesk-token"]').content;
let config, jobs = [], selected = null, doc = null, dirty = false, polling = false, saving = false, editGeneration = 0, selectionGeneration = 0;
let meeting = null, followingLive = true;
const liveStatuses = ['starting','recording','paused','finishing'];
const isLive = () => selected && liveStatuses.includes(selected.status);
const colors = ['#729e87','#7796b4','#b49877','#a184ad','#799da2','#ba8590','#99a36e','#867eae'];

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
function notice(text, error = false) {
  $('notice').textContent = text; $('notice').hidden = !text;
  $('notice').className = error ? 'error' : '';
  if (text && !error) setTimeout(() => { if ($('notice').textContent === text) $('notice').hidden = true; }, 4500);
}
function changed() { editGeneration++; dirty = true; $('save').disabled = saving; $('save').textContent = saving ? 'Saving…' : 'Save changes'; }
function time(seconds) {
  const minutes = Math.floor(seconds / 60), rest = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2,'0')}`;
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
    button.append(node('strong', job.name), node('small', date.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'})+status));
    button.addEventListener('click', () => select(job.id).catch(e => notice(e.message, true)));
    $('jobs').append(button);
  }
}
function setStatus() {
  $('recording-name').textContent = selected.name;
  $('meeting-date').textContent = new Date(selected.created*1000).toLocaleString(undefined,{weekday:'long',month:'long',day:'numeric',hour:'numeric',minute:'2-digit'});
  $('job-message').textContent = selected.message;
  const live=isLive(), busy=live || ['preparing','processing','queued'].includes(selected.status);
  $('recording-dot').hidden=!live;
  $('recording-dot').style.fill=selected.status==='paused'?'var(--muted)':'var(--red)';
  $('recording-dot').style.stroke=$('recording-dot').style.fill;
  $('run').hidden = !!doc?.segments?.length || live; $('run').disabled = busy || !config.readiness.configured;
  $('manual').hidden = !!doc || busy || !selected.duration;
  $('delete').disabled = busy;
  $('duration').textContent = selected.duration ? time(selected.duration) : 'Preparing…';
  $('segment-count').textContent = doc && !live ? `${doc.segments.length} passages` : '';
  $('save').hidden = !doc || live; $('search').disabled = !doc;
  $('player').hidden = !selected.duration || live;
  document.querySelector('.player-panel').hidden=live;
  document.querySelector('.export-menu').inert=live || !doc?.segments?.length;
  document.querySelector('.export-menu').hidden=live;
  $('speaker-tools').hidden=live;$('add').hidden=live;
  $('follow-live').hidden=!live;
  $('listening').hidden=!live || !!doc?.segments?.length;
}
async function select(jid) {
  if (identityBusy) { notice('Wait for the name change to finish before switching meetings.'); return; }
  if (saving) { notice('Wait for this save to finish before switching recordings.'); return; }
  if (dirty && !confirm('Discard unsaved edits and open another recording?')) return;
  const generation = ++selectionGeneration;
  const result = await api(`/api/jobs/${jid}`);
  if (generation !== selectionGeneration) return;
  selected = result; doc = selected.document ? structuredClone(selected.document) : null; dirty = false;
  $('empty').hidden = true; $('workspace').hidden = false;
  if(!isLive() && selected.duration) $('player').src = `/api/jobs/${jid}/audio`;
  else {$('player').pause();$('player').removeAttribute('src');}
  $('player').playbackRate = Number($('speed').value);
  $('search').value = ''; notice(''); setStatus(); renderJobs(); renderEditor();
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
  $('save').disabled = !dirty; $('save').textContent = dirty ? 'Save changes' : 'Saved';
  $('segment-count').textContent = isLive() ? '' : `${doc.segments.length} passages`;
  $('speaker-list').replaceChildren();
  Object.entries(doc.speakers).forEach(([id, name], i) => {
    const row = node('div', undefined, 'speaker-name');
    const dot = node('span', undefined, 'speaker-dot'); dot.style.backgroundColor = colors[i % colors.length];
    const input = node('input'); input.value = name; input.maxLength = 100; input.setAttribute('aria-label', `Rename ${name}`);
    input.addEventListener('input', () => { doc.speakers[id] = input.value; changed();
      document.querySelectorAll('.segment-top select').forEach(el => speakerOptions(el, el.value));
    });
    row.append(dot, input); $('speaker-list').append(row);
  });
  $('timing-note').textContent = doc.provenance?.timing || 'User supplied segment boundaries.';
  $('warnings').textContent = (doc.warnings || []).join(' '); $('warnings').hidden = !doc.warnings?.length;
  renderSegments();
  refreshIdentitySuggestions();
}
function renderSegments() {
  const pane=$('transcript-pane'), previousScroll=pane.scrollTop;
  $('segments').replaceChildren();
  const query = $('search').value.toLowerCase();
  let visible = 0;
  for (const segment of doc.segments) {
    if (query && !(segment.text + ' ' + doc.speakers[segment.speaker]).toLowerCase().includes(query)) continue;
    visible++;
    const card = node('article', undefined, 'segment'); card.dataset.segmentId = segment.id;
    const index = Object.keys(doc.speakers).indexOf(segment.speaker);
    const avatar=node('span',`S${index+1}`,'speaker-avatar');avatar.setAttribute('aria-hidden','true');avatar.dataset.color=index%8;
    const body=node('div',undefined,'segment-body');
    const top = node('div', undefined, 'segment-top');
    const seek = node('button', undefined, 'seek'); seek.append(icon('play'),node('span',time(segment.start))); seek.setAttribute('aria-label', `Play segment at ${time(segment.start)}`);
    seek.addEventListener('click', () => { $('player').currentTime = segment.start; $('player').play().catch(e => notice(e.message, true)); });
    const speaker = node('select'); speakerOptions(speaker, segment.speaker); speaker.setAttribute('aria-label', 'Segment speaker');
    speaker.disabled=!!isLive();
    seek.disabled=!!isLive();
    speaker.addEventListener('change', () => { segment.speaker = speaker.value; segment.review = true; changed(); });
    top.append(speaker, seek);
    const nameButton = node('button', undefined, 'name-speaker'); nameButton.append(icon('user-round-pen'));
    nameButton.setAttribute('aria-label', `Name ${doc.speakers[segment.speaker]}`); nameButton.title = 'Choose speaker name';
    nameButton.addEventListener('click', () => openNamePicker(segment.speaker).catch(error => notice(error.message, true)));
    top.append(nameButton);
    if (segment.review) top.append(node('span', 'Review', 'review-tag'));
    const remove = node('button', undefined, 'remove-segment'); remove.append(icon('x')); remove.setAttribute('aria-label', 'Remove segment');
    remove.addEventListener('click', () => { doc.segments = doc.segments.filter(s => s.id !== segment.id); changed(); renderEditor(); });
    remove.hidden=!!isLive();top.append(remove);
    const text = isLive()?node('p',segment.text,'segment-text'):node('textarea');
    if(!isLive()){text.value=segment.text;text.rows=2;text.setAttribute('aria-label',`Transcript at ${time(segment.start)}`);
      text.addEventListener('input',()=>{segment.text=text.value;text.style.height='auto';text.style.height=`${text.scrollHeight}px`;changed();});}

    card.addEventListener('focusin', () => {if(!isLive())showInspector(segment,card);});
    body.append(top,text);card.append(avatar,body);$('segments').append(card);
    if(!isLive()){text.style.height='auto';text.style.height=`${text.scrollHeight}px`;}
  }
  $('no-results').hidden = visible > 0 || (!!isLive() && !query);
  pane.scrollTop=isLive() && followingLive ? pane.scrollHeight : previousScroll;
  lucide.createIcons();
}
function icon(name) { const element=node('i');element.dataset.lucide=name;return element; }
function showInspector(segment,card) {
  document.querySelectorAll('.segment.active').forEach(item => item.classList.remove('active'));
  card.classList.add('active'); $('inspector').hidden=false;
  const details=$('segment-details');details.replaceChildren();
  for(const [key,label] of [['start','Start time (seconds)'],['end','End time (seconds)']]) {
    const field=node('label',label,'segment-detail-field'), input=node('input');
    input.type='number';input.min=0;input.max=selected.duration;input.step='.01';input.value=segment[key].toFixed(2);
    input.addEventListener('change',() => { segment[key]=Number(input.value);segment.timing='user_edited';changed(); });
    field.append(input);details.append(field);
  }
  if(segment.review) details.append(node('p','Check this passage against the recording.','review-tag'));
}
async function save() {
  if (!dirty) return;
  if (saving) throw new Error('A save is in progress. Try again when it finishes.');
  const generation = editGeneration;
  saving = true; $('save').disabled = true; $('save').textContent = 'Saving…';
  try {
    const result = await api(`/api/jobs/${selected.id}/transcript`, {method: 'PUT', body: JSON.stringify({revision: selected.revision, document: doc})});
    selected = result;
    if (generation === editGeneration) { doc = structuredClone(result.document); dirty = false; renderEditor(); }
    setStatus(); notice(dirty ? 'Earlier edits saved. New changes still need saving.' : 'Saved on this computer.');
  } finally { saving = false; $('save').disabled = !dirty; $('save').textContent = dirty ? 'Save changes' : 'Saved'; }
}
async function upload(files) {
  if (saving) throw new Error('Wait for this save to finish before uploading.');
  if (!files.length) return;
  if (dirty && !confirm('Discard unsaved edits and upload a new recording?')) return;
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
    if(meeting && liveStatuses.includes(meeting.status)){select(meeting.id).catch(e=>notice(e.message,true));return;}
    if(dirty && !confirm('Discard unsaved transcript edits?'))return;
    selected=doc=null;dirty=false;selectionGeneration++;$('player').pause();$('player').removeAttribute('src');
    $('workspace').hidden=true;$('empty').hidden=false;renderJobs();$('meeting-title').focus();
  });
  $('meeting-search').addEventListener('input',renderJobs);
  $('language').addEventListener('change',()=>{try{localStorage.setItem('speakerdesk.language',$('language').value);}catch{}});
  $('theme').value=speakerdeskTheme.get();
  $('theme').addEventListener('change',()=>speakerdeskTheme.set($('theme').value));
  $('meeting-form').addEventListener('submit',async event=>{
    event.preventDefault();$('start-meeting').disabled=true;
    try {
      if(!config.readiness.configured){openSettings();throw new Error('Finish setup in Settings before starting your meeting.');}
      const sources=[];if($('capture-microphone').checked)sources.push('microphone');if($('capture-system').checked)sources.push('system');
      if(!sources.length)throw new Error('Choose an audio source before starting.');
      const job=await api('/api/meetings',{method:'POST',body:JSON.stringify({name:$('meeting-title').value,language:$('language').value,sources})});
      followingLive=true;jobs=await api('/api/jobs');await select(job.id);await refreshMeeting();
    }catch(error){notice(error.message,true);}finally{$('start-meeting').disabled=false;}
  });
  $('pause-meeting').addEventListener('click',()=>meetingAction(meeting?.status==='paused'?'resume':'pause').catch(e=>notice(e.message,true)));
  $('stop-meeting').addEventListener('click',()=>meetingAction('stop').catch(e=>notice(e.message,true)));
  $('active-meeting').addEventListener('click',()=>select(meeting.id).catch(e=>notice(e.message,true)));
  $('follow-live').addEventListener('click',()=>{followingLive=true;$('transcript-pane').scrollTop=$('transcript-pane').scrollHeight;$('follow-live').textContent='Following live';});
  $('transcript-pane').addEventListener('scroll',()=>{if(!isLive())return;const p=$('transcript-pane');followingLive=p.scrollHeight-p.scrollTop-p.clientHeight<60;$('follow-live').textContent=followingLive?'Following live':'Return to live';});
  $('files').addEventListener('change', () => upload($('files').files).catch(e => notice(e.message, true)));
  $('upload-form').addEventListener('submit', e => e.preventDefault());
  const drop = document.querySelector('main');
  ['dragenter', 'dragover'].forEach(event => drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(event => drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('dragging'); }));
  drop.addEventListener('drop', e => upload(e.dataTransfer.files).catch(err => notice(err.message, true)));
  $('setup-toggle').addEventListener('click', () => $('setup').hidden ? openSettings() : closeSettings());
  $('inspector-close').addEventListener('click', () => { $('inspector').hidden=true;document.querySelectorAll('.segment.active').forEach(item => item.classList.remove('active')); });
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
  $('install-models').addEventListener('click',async () => {
    try { await api('/api/setup',{method:'POST'}); await refreshSetup(); }
    catch(error) { $('model-status').textContent=error.message; }
  });
  $('run').addEventListener('click', async () => { try { selected = await api(`/api/jobs/${selected.id}/run`, {method: 'POST'}); setStatus(); } catch (e) { notice(e.message, true); } });
  $('manual').addEventListener('click', manual);
  $('save').addEventListener('click', () => save().catch(e => notice(e.message, true)));
  $('speed').addEventListener('change', () => { $('player').playbackRate = Number($('speed').value); });
  $('player').addEventListener('timeupdate', () => {
    if (!doc) return;
    const now = $('player').currentTime;
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
    document.querySelector('.segment:last-child textarea')?.focus();
  });
  $('add-speaker').addEventListener('click', () => { const id = `speaker_${crypto.randomUUID().slice(0,8)}`; doc.speakers[id] = `Speaker ${Object.keys(doc.speakers).length + 1}`; changed(); renderEditor(); });
  $('import').addEventListener('change', async () => {
    try {
      if (saving) throw new Error('Wait for this save to finish before importing.');
      const file = $('import').files[0]; if (!file) return;
      if (dirty && !confirm('Replace the current transcript, including unsaved edits, with this JSON file?')) return;
      saving = true; $('editor').inert = true;
      if (file.size > 5 * 1024 * 1024) throw new Error('Transcript JSON must be at most 5 MB.');
      const incoming = JSON.parse(await file.text());
      if (!incoming.speakers || !Array.isArray(incoming.segments)) throw new Error('JSON needs speakers and segments. See fixtures/conversation.json.');
      // Validate and persist atomically before showing imported data.
      const result = await api(`/api/jobs/${selected.id}/transcript`, {method: 'PUT', body: JSON.stringify({revision: selected.revision, document: incoming, imported: true})});
      selected = result; doc = structuredClone(result.document); dirty = false; setStatus(); renderEditor(); notice('Imported and saved locally.');
    } catch (e) { notice(e.message, true); } finally { saving = false; $('editor').inert = false; $('import').value = ''; }
  });
  document.querySelectorAll('[data-export]').forEach(button => button.addEventListener('click', async () => {
    try {
      await save(); if (dirty) throw new Error('New edits were made during saving. Save them before exporting.'); const kind = button.dataset.export;
      const link = node('a');
      link.href = `/api/jobs/${selected.id}/export/${kind}`;
      link.download = `${selected.name.replace(/\.[^.]+$/, '')}.${kind}`;
      document.body.append(link); link.click(); link.remove();
    } catch (e) { notice(e.message, true); }
  }));
  $('delete').addEventListener('click', async () => {
    if (saving) { notice('Wait for this save to finish before deleting.'); return; }
    if (!confirm(`Delete ${selected.name} and its local audio and transcript?`)) return;
    try {
      await api(`/api/jobs/${selected.id}`, {method: 'DELETE'}); selected = doc = null; dirty = false;
      $('player').pause(); $('player').removeAttribute('src'); $('workspace').hidden = true; $('empty').hidden = false;
      jobs = await api('/api/jobs'); renderJobs(); notice('Recording deleted from local storage.');
    } catch (e) { notice(e.message, true); }
  });
  window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });
}
async function poll() {
  if (polling) return; polling = true;
  try {
    jobs = await api('/api/jobs'); renderJobs();
    if (selected && !dirty && !saving && !identityBusy) {
      const generation = selectionGeneration, jid = selected.id;
      const next = await api(`/api/jobs/${jid}`);
      if (generation !== selectionGeneration || selected?.id !== jid || dirty || saving || identityBusy || next.revision < selected.revision) return;
      const changedAudio = !selected.duration && next.duration;
      const finished = next.status !== selected.status && !['preparing','queued','processing',...liveStatuses].includes(next.status);
      if (next.revision !== selected.revision) { doc = next.document ? structuredClone(next.document) : null; selected = next; renderEditor(); }
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
    Object.entries(config.languages).forEach(([code, name]) => { const option = node('option', name); option.value = code; $('language').append(option); });
    let previousLanguage='en';try{previousLanguage=localStorage.getItem('speakerdesk.language')||'en';}catch{}
    $('language').value=config.languages[previousLanguage]?previousLanguage:'en';
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
function openSettings() { $('setup').hidden=false;$('setup-toggle').setAttribute('aria-expanded','true');document.querySelector('main').inert=true;document.querySelector('.sidebar').inert=true;$('settings-close').focus();loadPeople().catch(error=>notice(error.message,true)); }
function closeSettings() { $('setup').hidden=true;document.querySelector('main').inert=false;document.querySelector('.sidebar').inert=false;$('setup-toggle').setAttribute('aria-expanded','false'); $('setup-toggle').focus(); }
async function refreshSetup() {
  try {
    const state=await api('/api/setup');
    const list=$('model-list');list.replaceChildren();
    state.models.forEach(model => {
      const item=node('div',undefined,'model-item'); item.append(node('strong',model.name),node('span',model.installed?'Installed':`${(model.bytes/1e9).toFixed(2)} GB`));list.append(item);
    });
    const busy=state.status==='downloading';
    $('model-progress').hidden=!busy;
    $('model-progress').value=100*state.downloaded_bytes/state.total_bytes;
    $('model-status').textContent=state.error || (busy?`${state.phase} · ${Math.round(100*state.downloaded_bytes/state.total_bytes)}%`:state.ready?'Ready to transcribe on this Mac.':'Download once, then use offline.');
    $('install-models').hidden=state.ready;
    $('install-models').disabled=busy || !state.supported;
    $('install-models').textContent=busy?'Downloading…':state.status==='failed'?'Retry download':'Download models · 1.7 GB';
    if(state.ready && !config.readiness.configured) {
      config=await api('/api/config');if(selected)setStatus();
    }
  } catch(error) { $('model-status').textContent=error.message; }
}
async function meetingAction(action) {
  if(!meeting?.id)return;
  $('pause-meeting').disabled=true;$('stop-meeting').disabled=true;
  try{await api(`/api/meetings/${meeting.id}/${action}`,{method:'POST'});await refreshMeeting();await poll();}
  finally{await refreshMeeting();}
}
let meetingPoll=false;
async function refreshMeeting() {
  if(meetingPoll)return;meetingPoll=true;
  try{
    meeting=await api('/api/meeting');
    const active=liveStatuses.includes(meeting.status);
    $('capture-footer').hidden=!active;
    $('new-meeting').disabled=active;
    $('start-hint').textContent=meeting.capture_available?'':'Live recording is not included in this build yet.';
    $('elapsed').textContent=time(meeting.duration);
    const sources=meeting.sources || (selected?.id===meeting.id ? selected.sources : []) || [];
    for(const [source,id,label] of [['microphone','microphone-level','Microphone'],['system','system-level','Mac audio']]){
      const enabled=sources.includes(source);$(id).closest('.audio-meter').querySelector('label').textContent=label+(enabled?'':' (off)');
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
