'use strict';
let people = [], voiceAvailable = false, nameContext = null, identityBusy = false;

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
  if (saving || identityBusy || dirty) throw new Error('Save transcript edits before naming a speaker.');
  identityBusy = true; $('editor').inert = true;
  try { return await action(); }
  finally { identityBusy = false; $('editor').inert = false; }
}
function renderPeople() {
  $('people-list').replaceChildren();
  if (!people.length) $('people-list').append(node('p', 'No saved people yet.', 'settings-note'));
  for (const person of people) {
    const form = node('form', undefined, 'person-row');
    const input = node('input'); input.value = person.name; input.maxLength = 100; input.required = true;
    input.setAttribute('aria-label', `Saved name for ${person.name}`);
    const saveName = node('button', 'Save name', 'quiet'); saveName.type = 'submit';
    const status = node('span', person.voice_saved
      ? voiceAvailable && !person.voice_compatible ? 'Save this voice again to use recognition' : 'Voice saved on this Mac'
      : 'Name only', 'settings-note');
    form.append(input, saveName, status);
    form.addEventListener('submit', async event => {
      event.preventDefault(); saveName.disabled = true;
      try {
        await api(`/api/people/${person.id}`, {method: 'PATCH', body: JSON.stringify({name: input.value})});
        await loadPeople(); renderPeople(); $('people-message').textContent = 'Saved name updated. Existing meeting names are unchanged.';
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
  const local = node('option', 'This meeting only'); local.value = ''; $('name-person').append(local);
  people.forEach(person => { const option = node('option', person.name); option.value = person.id; $('name-person').append(option); });
  $('name-person').value = current;
}
async function openNamePicker(track) {
  if (saving || identityBusy || dirty) throw new Error('Save transcript edits before naming a speaker.');
  const jid = selected.id;
  await loadPeople();
  if (selected?.id !== jid || !doc?.speakers[track]) return;
  nameContext = {jid, track, revision: selected.revision};
  const assigned = selected.speaker_assignments?.[track]?.person_id || '';
  personOptions(assigned); $('name-value').value = doc.speakers[track];
  $('name-title').textContent = `Name ${doc.speakers[track]}`;
  $('name-message').textContent = ''; $('voice-consent').checked = false;
  $('voice-clip-list').replaceChildren();
  for (const segment of doc.segments.filter(s => s.speaker === track && !s.review && s.finalized !== false
      && !track.startsWith('overlap') && (s.speaker_candidates || [track]).length === 1
      && s.end-s.start >= 2 && s.end-s.start <= 10)) {
    const label = node('label', undefined, 'voice-clip');
    const check = node('input'); check.type = 'checkbox'; check.value = segment.id;
    label.append(check, node('span', `${time(segment.start)}–${time(segment.end)} · ${segment.text}`));
    $('voice-clip-list').append(label);
  }
  const usable = voiceAvailable && doc.provenance?.kind === 'local_inference';
  $('voice-clips').hidden = !usable;
  $('voice-consent-label').hidden = !usable || !assigned;
  $('remember-voice').hidden = !usable || !assigned;
  $('check-voice').hidden = !usable || !!selected.speaker_assignments?.[track];
  $('voice-note').textContent = !voiceAvailable ? 'Voice recognition is not available in this build.'
    : !usable ? 'Voice recognition needs finalized local transcription passages.'
    : assigned ? 'Only the passages you select are used. Saving a voice requires your consent.'
    : 'Apply a saved person first to remember their voice. A suggestion always needs your confirmation.';
  $('name-dialog').showModal(); $('name-value').focus(); $('name-value').select();
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
  $('person-form').addEventListener('submit', async event => {
    event.preventDefault(); const button = event.submitter; button.disabled = true;
    try {
      await api('/api/people', {method: 'POST', body: JSON.stringify({name: $('person-name').value})});
      $('person-name').value = ''; await loadPeople(); renderPeople(); $('people-message').textContent = 'Name saved. No voice was saved.';
    } catch (error) { $('people-message').textContent = error.message; }
    finally { button.disabled = false; }
  });
  $('name-person').addEventListener('change', () => {
    const person = people.find(p => p.id === $('name-person').value);
    if (person) $('name-value').value = person.name;
  });
  $('name-value').addEventListener('input', () => { $('name-person').value = ''; });
  $('name-add-person').addEventListener('click', async () => {
    $('name-add-person').disabled = true;
    try {
      const person = await api('/api/people', {method: 'POST', body: JSON.stringify({name: $('name-value').value})});
      await loadPeople(); personOptions(person.id); $('name-message').textContent = 'Name added to People. Apply it to this meeting when ready. No voice was saved.';
    } catch (error) { $('name-message').textContent = error.message; }
    finally { $('name-add-person').disabled = false; }
  });
  $('name-form').addEventListener('submit', event => {
    event.preventDefault(); const context = nameContext;
    identityAction(async () => {
      if (selected?.id !== context.jid) throw new Error('Open this meeting again to name the speaker.');
      const result = await api(`/api/jobs/${context.jid}/speakers/${encodeURIComponent(context.track)}/identity`, {
        method: 'POST', body: JSON.stringify({revision: context.revision, person_id: $('name-person').value || null, name: $('name-value').value})});
      $('name-dialog').close(); acceptIdentityJob(result); notice('Name confirmed for this meeting.');
    }).catch(error => { $('name-message').textContent = error.message; });
  });
  for (const kind of ['remember', 'check']) $(`${kind}-voice`).addEventListener('click', () => {
    const context = nameContext;
    identityAction(async () => {
      if (selected?.id !== context.jid) throw new Error('Open this meeting again to choose passages.');
      const segment_ids = Array.from($('voice-clip-list').querySelectorAll('input:checked'), input => input.value);
      const body = {revision: context.revision, segment_ids};
      if (kind === 'remember') {
        const pid = selected.speaker_assignments?.[context.track]?.person_id;
        if (!pid || $('name-person').value !== pid) throw new Error('Apply the saved person before remembering their voice.');
        if (!$('voice-consent').checked) throw new Error('Choose Remember this person’s voice to give consent.');
        await api(`/api/people/${pid}/voice`, {method: 'POST', body: JSON.stringify({...body, meeting_id: context.jid, track_id: context.track, consent: true})});
        $('name-dialog').close(); notice('Voice saved on this Mac. Future suggestions need confirmation.');
      } else {
        const result = await api(`/api/jobs/${context.jid}/speakers/${encodeURIComponent(context.track)}/voice-suggestion`, {method: 'POST', body: JSON.stringify(body)});
        $('name-dialog').close(); acceptIdentityJob(result.job);
        notice(result.matched ? 'Review the suggested person before using their name.' : 'No confident match. Choose a name or leave this speaker unknown.');
      }
    }).catch(error => { $('name-message').textContent = error.message; });
  });
}
