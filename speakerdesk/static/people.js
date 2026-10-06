'use strict';
let people = [], voiceAvailable = false, nameContext = null, identityBusy = false;
let nameClipState = null, nameLoadGeneration = 0, nameOpenGeneration = 0, nameLoading = false;
let voicePreviewStart = null, voicePreviewEnd = null, voicePreviewGeneration = 0;

async function loadPeople() {
  const state = await api('/api/people');
  people = state.people; voiceAvailable = state.voice_available;
}
function acceptIdentityJob(result) {
  if (selected?.id !== result.id) return;
  selected = result; doc = structuredClone(result.document); dirty = false;
  setStatus(); renderEditor();
}
async function identityAction(action) {
  if (identityBusy) throw new Error('Wait for the name change to finish.');
  if (dirty || saving) await flushSave();
  identityBusy = true; $('editor').inert = true; updateNameControls();
  try { return await action(); }
  finally { identityBusy = false; $('editor').inert = false; updateNameControls(); }
}
function renderPeople() {
  $('people-list').replaceChildren();
  if (!people.length) $('people-list').append(node('p', 'No saved people yet.', 'settings-note'));
  for (const person of people) {
    const form = node('form', undefined, 'person-row');
    const input = node('input'); input.value = person.name; input.maxLength = 100; input.required = true;
    input.setAttribute('aria-label', `Saved name for ${person.name}`);
    const saveName = node('button', 'Save name', 'quiet'); saveName.type = 'submit';
    form.append(input, saveName);
    form.append(node('span', person.voice_saved ? voiceAvailable && !person.voice_compatible
      ? 'Save this voice again to use recognition' : 'Voice saved on this Mac'
      : 'Name saved. Open a meeting and choose Name / remember voice beside this person’s speech.', 'settings-note'));
    form.addEventListener('submit', async event => {
      event.preventDefault(); saveName.disabled = true;
      try {
        await api(`/api/people/${person.id}`, {method: 'PATCH', body: JSON.stringify({name: input.value})});
        await loadPeople(); renderPeople(); $('people-message').textContent = 'Name saved.';
      } catch (error) { $('people-message').textContent = error.message; }
      finally { saveName.disabled = false; }
    });
    if (person.voice_saved) {
      const forget = node('button', 'Forget voice', 'text-button'); forget.type = 'button';
      forget.addEventListener('click', async () => {
        forget.disabled = true;
        try {
          await api(`/api/people/${person.id}/voice`, {method: 'DELETE'});
          await loadPeople(); renderPeople(); refreshIdentitySuggestions();
          $('people-message').textContent = 'Voice forgotten. Meeting transcripts and saved names are kept.';
        } catch (error) { $('people-message').textContent = error.message; }
        finally { forget.disabled = false; }
      });
      form.append(forget);
    }
    $('people-list').append(form);
  }
}
function personOptions(current = '') {
  $('name-person').replaceChildren();
  const local = node('option', 'Not linked · this meeting only'); local.value = ''; $('name-person').append(local);
  people.forEach(person => { const option = node('option', person.name); option.value = person.id; $('name-person').append(option); });
  $('name-person').value = current;
}
async function openNamePicker(track) {
  if (identityBusy) throw new Error('Wait for the name change to finish.');
  if (dirty || saving) await flushSave();
  const jid = selected.id, generation = ++nameOpenGeneration;
  await loadPeople();
  if (generation !== nameOpenGeneration || selected?.id !== jid || !doc?.speakers[track] || dirty || identityBusy) return;
  nameContext = {jid, track, revision: selected.revision};
  const assignment=selected.speaker_assignments?.[track];
  const assigned = assignment?.person_id || '';
  personOptions(assigned); $('name-value').value = doc.speakers[track];
  $('name-title').textContent = 'Name speaker';
  $('name-message').textContent = ''; $('voice-consent').checked = false;
  nameClipState = null; $('voice-clip-list').replaceChildren(); $('voice-excluded-list').replaceChildren();
  $('voice-preview').pause(); $('voice-preview').removeAttribute('src'); $('voice-preview').hidden = true;
  $('name-dialog').showModal(); $('name-value').focus(); $('name-value').select();
  await refreshVoiceClips();
}

function selectedVoiceClips() {
  return Array.from($('voice-clip-list').querySelectorAll('input:checked'), input => input.value);
}
function updateNameControls() {
  if (!nameContext || !$('name-dialog').open) return;
  const assignment = selected?.speaker_assignments?.[nameContext.track];
  const confirmed = assignment?.person_id && assignment.source !== 'automatic_voice' && assignment.confirmed !== false
    && $('name-person').value === assignment.person_id;
  const current = selected?.id === nameContext.jid && selected.revision === nameContext.revision && !dirty;
  const ids = selectedVoiceClips(), clips = (nameClipState?.clips || []).filter(c => ids.includes(c.id)).sort((a,b)=>a.start-b.start);
  const enough = ids.length >= (nameClipState?.minimum_clips || 2) && ids.length <= 12
    && clips.every((c,i)=>!i || clips[i-1].end <= c.start);
  const busy = identityBusy || nameLoading || saving;
  const usable = nameClipState?.status === 'ready' && current && !busy;
  $('name-apply').disabled = busy || !current;
  $('name-add-person').disabled = busy;
  $('refresh-voice-clips').disabled = busy;
  $('remember-voice').disabled = !usable || !confirmed || !enough || !$('voice-consent').checked;
  $('check-voice').hidden = !!assignment;
  $('check-voice').disabled = !usable || !enough;
  $('voice-consent').disabled = !usable || !confirmed || !enough;
  $('voice-note').textContent = !current ? 'The transcript changed. Refresh passages before continuing.'
    : confirmed ? 'Name confirmed for this meeting. Remembering the voice is optional.'
    : assignment?.source === 'automatic_voice' ? 'Apply the name to confirm this recognized person before remembering new voice passages.'
    : 'Save this name in People, then apply it to remember the voice. You can also keep this name in the meeting only.';
  $('name-voice-section').hidden = false;
}
async function refreshVoiceClips() {
  const context = nameContext, generation = ++nameLoadGeneration;
  if (!context) return;
  const previousIds = nameClipState ? selectedVoiceClips() : null;
  $('voice-preview').pause(); voicePreviewStart = voicePreviewEnd = null; ++voicePreviewGeneration;
  nameLoading = true; nameClipState = null; $('voice-consent').checked = false;
  $('voice-availability').textContent = 'Checking saved passages…'; updateNameControls();
  try {
    const state = await api(`/api/jobs/${context.jid}/speakers/${encodeURIComponent(context.track)}/voice-clips`);
    if (generation !== nameLoadGeneration || nameContext !== context || !$('name-dialog').open
        || selected?.id !== context.jid || selected.revision !== context.revision || state.revision !== context.revision || dirty) {
      if (nameContext === context && $('name-dialog').open) $('voice-availability').textContent = 'The transcript changed. Refresh passages to use the current audio.';
      return;
    }
    nameClipState = state;
    $('voice-availability').textContent = state.message;
    $('voice-clips-legend').textContent = `Use ${state.minimum_clips}–12 separate passages, 2–10 seconds each`;
    $('voice-clip-list').replaceChildren();
    $('voice-clips').hidden = !state.clips.length;
    for (const clip of state.clips) {
      const row = node('div', undefined, 'voice-clip'), label = node('label'), check = node('input');
      check.type = 'checkbox'; check.value = clip.id;
      check.checked = (previousIds || state.recommended_ids).includes(clip.id); check.disabled = state.status !== 'ready';
      check.addEventListener('change', () => { $('voice-consent').checked = false; updateNameControls(); });
      const description = clip.id === clip.segment_id ? clip.text || 'Audio from this speaker; transcript needs review'
        : 'Audio clip from this speaker’s longer passage';
      label.append(check, node('span', `${passageTime(clip.start)}–${passageTime(clip.end)} · ${description}`));
      const preview = node('button', 'Preview', 'text-button'); preview.type = 'button';
      preview.disabled = !state.audio_available || state.status === 'busy';
      preview.setAttribute('aria-label', `Preview passage from ${passageTime(clip.start)} to ${passageTime(clip.end)}`);
      preview.addEventListener('click', () => previewVoiceClip(clip, context).catch(error => { $('name-message').textContent = error.message; }));
      row.append(label, preview); $('voice-clip-list').append(row);
    }
    $('voice-excluded-list').replaceChildren(); $('voice-excluded').hidden = !state.excluded.length;
    $('voice-excluded-summary').textContent = `${state.excluded.length} unavailable passage${state.excluded.length === 1 ? '' : 's'} — see why`;
    for (const clip of state.excluded) $('voice-excluded-list').append(node('li', `${passageTime(clip.start)}–${passageTime(clip.end)} · ${clip.message}`));
    $('voice-excluded').open = !state.clips.length;
  } catch (error) { if (nameContext === context) $('voice-availability').textContent = `${error.message} Refresh passages to try again.`; }
  finally { if (generation === nameLoadGeneration) { nameLoading = false; updateNameControls(); } }
}
async function previewVoiceClip(clip, context) {
  if (nameContext !== context || selected?.id !== context.jid || selected.revision !== context.revision)
    throw new Error('Refresh passages before previewing this audio.');
  const player = $('voice-preview'); $('player').pause(); player.pause();
  player.hidden = false; voicePreviewEnd = clip.end;
  const src = `/api/jobs/${context.jid}/audio`;
  const generation = ++voicePreviewGeneration;
  if (player.getAttribute('src') !== src) player.src = src;
  if (player.readyState < 1) {
    await new Promise((resolve,reject) => {
      const cleanup = () => { player.removeEventListener('loadedmetadata', ready); player.removeEventListener('error', failed); };
      const ready = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error('The original recording is unavailable. Refresh passages or choose another meeting.')); };
      player.addEventListener('loadedmetadata', ready); player.addEventListener('error', failed);
    });
  }
  if (!$('name-dialog').open || nameContext !== context || generation !== voicePreviewGeneration
      || selected?.id !== context.jid || selected.revision !== context.revision) return;
  voicePreviewStart = clip.start; voicePreviewEnd = clip.end; player.currentTime = clip.start; await player.play();
}
async function refreshIdentitySuggestions() {
  const host = $('identity-suggestions');
  if (!host || !selected || !doc || dirty) { host?.replaceChildren(); return; }
  const jid = selected.id, revision = selected.revision;
  try {
    const state = await api(`/api/jobs/${jid}/identity-suggestions`);
    if (selected?.id !== jid || selected.revision !== revision || state.revision !== revision || dirty) return;
    host.replaceChildren();
    for (const suggestion of state.suggestions) {
      const card = node('div', undefined, 'identity-suggestion');
      const evidence = suggestion.kind === 'explicit_introduction'
        ? `${time(suggestion.evidence.start)} · “${suggestion.evidence.quote}”`
        : suggestion.evidence.clips.map(clip => `${time(clip.start)}–${time(clip.end)}`).join(', ');
      card.append(node('strong', `${suggestion.kind === 'voice_match' ? 'Possible' : 'Suggested name:'} ${suggestion.name}`), node('p', evidence));
      const use = node('button', `Use ${suggestion.name}`, 'quiet');
      use.addEventListener('click', () => identityAction(async () => {
        const result = await api(`/api/jobs/${jid}/speakers/${encodeURIComponent(suggestion.track_id)}/identity`, {
          method: 'POST', body: JSON.stringify({revision, suggestion_id: suggestion.id, person_id: suggestion.person_id || null})});
        acceptIdentityJob(result); notice('Name confirmed for this meeting.');
      }).catch(error => notice(error.message, true)));
      const choose = node('button', 'Choose name', 'text-button');
      choose.addEventListener('click', () => openNamePicker(suggestion.track_id).catch(error => notice(error.message, true)));
      const dismiss = node('button', 'Dismiss', 'text-button');
      dismiss.addEventListener('click', () => identityAction(async () => {
        acceptIdentityJob(await api(`/api/jobs/${jid}/identity-suggestions/${suggestion.id}/dismiss`, {method: 'POST', body: JSON.stringify({revision})}));
      }).catch(error => notice(error.message, true)));
      card.append(use, choose, dismiss); host.append(card);
    }
  } catch (error) { notice(error.message, true); }
}
function wirePeople() {
  const host = node('div'); host.id = 'identity-suggestions'; host.setAttribute('aria-live', 'polite');
  $('transcript-pane').insertBefore(host, $('segments'));
  $('people-toggle').addEventListener('click', async () => {
    try { await loadPeople(); renderPeople(); $('people-message').textContent = ''; $('people-dialog').showModal(); $('person-name').focus(); }
    catch (error) { notice(error.message, true); }
  });
  $('people-close').addEventListener('click', () => $('people-dialog').close());
  $('name-close').addEventListener('click', () => $('name-dialog').close());
  $('name-done').addEventListener('click', () => $('name-dialog').close());
  $('name-dialog').addEventListener('close', () => {
    nameContext = null; nameClipState = null; ++nameLoadGeneration; ++nameOpenGeneration; nameLoading = false;
    $('voice-preview').pause(); voicePreviewStart = voicePreviewEnd = null; ++voicePreviewGeneration;
    $('voice-preview').removeAttribute('src');
  });
  $('voice-preview').addEventListener('timeupdate', () => {
    if (voicePreviewEnd !== null && !$('voice-preview').paused && $('voice-preview').currentTime >= voicePreviewEnd) {
      $('voice-preview').pause(); $('voice-preview').currentTime = voicePreviewEnd;
    }
  });
  $('voice-preview').addEventListener('seeking', () => {
    const player = $('voice-preview');
    if (voicePreviewStart !== null && (player.currentTime < voicePreviewStart || player.currentTime > voicePreviewEnd))
      player.currentTime = Math.max(voicePreviewStart, Math.min(voicePreviewEnd, player.currentTime));
  });
  $('refresh-voice-clips').addEventListener('click', () => identityAction(async () => {
    const context = nameContext;
    const result = await api(`/api/jobs/${context.jid}`);
    if (nameContext !== context || selected?.id !== context.jid) return;
    acceptIdentityJob(result); context.revision = result.revision;
    await refreshVoiceClips();
  }).catch(error => { $('name-message').textContent = error.message; }));
  $('person-form').addEventListener('submit', async event => {
    event.preventDefault(); const button = event.submitter; button.disabled = true;
    try {
      await api('/api/people', {method: 'POST', body: JSON.stringify({name: $('person-name').value})});
      $('person-name').value = ''; await loadPeople(); renderPeople(); $('people-message').textContent = 'Name saved.';
    } catch (error) { $('people-message').textContent = error.message; }
    finally { button.disabled = false; }
  });
  $('name-person').addEventListener('change', () => {
    const person = people.find(p => p.id === $('name-person').value);
    if (person) $('name-value').value = person.name;
    $('voice-consent').checked = false; updateNameControls();
  });
  $('name-value').addEventListener('input', () => { $('name-person').value = ''; $('voice-consent').checked = false; updateNameControls(); });
  $('voice-consent').addEventListener('change', updateNameControls);
  $('name-add-person').addEventListener('click', () => {
    const context = nameContext, name = $('name-value').value;
    identityAction(async () => {
      const person = await api('/api/people', {method: 'POST', body: JSON.stringify({name})});
      await loadPeople();
      if (nameContext !== context || !$('name-dialog').open || selected?.id !== context.jid) return;
      personOptions(person.id); $('voice-consent').checked = false;
      $('name-message').textContent = 'Name saved. Apply it to this speaker to continue with voice setup.';
    }).catch(error => { if (nameContext === context) $('name-message').textContent = error.message; });
  });
  $('name-form').addEventListener('submit', event => {
    event.preventDefault(); const context = nameContext;
    identityAction(async () => {
      if (selected?.id !== context.jid) throw new Error('Open this meeting again to name the speaker.');
      const result = await api(`/api/jobs/${context.jid}/speakers/${encodeURIComponent(context.track)}/identity`, {
        method: 'POST', body: JSON.stringify({revision: context.revision, person_id: $('name-person').value || null, name: $('name-value').value})});
      if (nameContext !== context || !$('name-dialog').open) return;
      acceptIdentityJob(result); context.revision = result.revision;
      $('voice-consent').checked = false;
      $('name-message').textContent = 'Name applied to this meeting.';
      await refreshVoiceClips();
    }).catch(error => { $('name-message').textContent = error.message; });
  });
  for (const kind of ['remember', 'check']) $(`${kind}-voice`).addEventListener('click', () => {
    const context = nameContext;
    identityAction(async () => {
      if (selected?.id !== context.jid) throw new Error('Open this meeting again to choose passages.');
      const segment_ids = selectedVoiceClips();
      const body = {revision: context.revision, segment_ids};
      if (kind === 'remember') {
        const pid = selected.speaker_assignments?.[context.track]?.person_id;
        if (!pid || $('name-person').value !== pid) throw new Error('Apply the saved person before remembering their voice.');
        if (!$('voice-consent').checked) throw new Error('Choose Remember this person’s voice to give consent.');
        await api(`/api/people/${pid}/voice`, {method: 'POST', body: JSON.stringify({...body, meeting_id: context.jid, track_id: context.track, consent: true})});
        $('name-dialog').close(); notice('Voice saved on this Mac for future meetings.');
      } else {
        const result = await api(`/api/jobs/${context.jid}/speakers/${encodeURIComponent(context.track)}/voice-suggestion`, {method: 'POST', body: JSON.stringify(body)});
        $('name-dialog').close(); acceptIdentityJob(result.job);
        notice(result.matched ? 'Review the suggested person before using their name.' : 'No confident match. Choose a name or leave this speaker unknown.');
      }
    }).catch(error => { $('name-message').textContent = error.message; });
  });
}
