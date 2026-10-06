'use strict';
// Drafts are local until a passage revision is explicitly saved. Polling keeps
// arriving words visible without replacing the focused textarea or its caret.
const passageDrafts = new Map();
const recoverablePassageDrafts = new Map();
const passageSaveOperations = new Map();
// Reflow long passages when the window or details pane changes width. Cache
// unchanged text sizes so an hour-long transcript does not relayout on each poll.
const passageWidthObserver=new ResizeObserver(()=>{
  const pane=$('transcript-pane'),host=$('segments');
  const anchor=Array.from(host.children).find(card=>card.getBoundingClientRect().bottom>pane.getBoundingClientRect().top);
  const before=anchor?.getBoundingClientRect().top;
  $('segments').querySelectorAll('textarea').forEach(fitPassageText);
  if(isLive() && followingLive && !host.contains(document.activeElement))pane.scrollTop=pane.scrollHeight;
  else if(anchor?.isConnected)pane.scrollTop+=anchor.getBoundingClientRect().top-before;
});
passageWidthObserver.observe(document.getElementById('transcript-pane'));
function hasPassageDrafts() { return passageDrafts.size > 0; }
function hasRecoverablePassageDrafts() { return !!doc?.segments.some(segment=>recoverablePassageDrafts.has(segment.id)); }
function isPassageSaving(id,jid=selected?.id) { return passageSaveOperations.has(`${jid}:${id}`); }
function hasPendingPassageSaves() { return Array.from(passageSaveOperations.values()).some(operation=>operation.jid===selected?.id); }
function renderPassageSaveState(card) {
  const busy=isPassageSaving(card.dataset.segmentId,card.dataset.meetingId);
  card.classList.toggle('passage-saving',busy);
  card.querySelector('.save-passage').disabled=busy || !passageDrafts.has(card.dataset.segmentId);
  for(const selector of ['.keep-correction','.use-latest','.recover-correction'])card.querySelector(selector).disabled=busy;
}
function renderRefinementStatus() {
  const status=selected?.refinement_status;
  $('refinement-controls').hidden=!selected?.rolling_refinement;
  $('refinement-status').textContent=({waiting:'Accuracy pass queued · recent words may still change',refining:'Improving earlier phrases with more context',paused:'Accuracy pass paused · audio kept for later',complete:'Accuracy pass complete',unresolved:'Some phrases need your review'})[status] || '';
  $('refinement-status').title='Speakerdesk re-checks recent audio with more context and may correct early words. Your saved corrections are never overwritten.';
  $('refinement-toggle').hidden=status==='complete';
  $('refinement-toggle').textContent=status==='paused'?'Resume accuracy pass':status==='unresolved'?'Retry unresolved phrases':'Pause accuracy pass';
}
async function toggleRefinement() {
  const jid=selected.id,button=$('refinement-toggle');button.disabled=true;
  try { await api(`/api/jobs/${jid}/refinement/${['paused','unresolved'].includes(selected.refinement_status)?'resume':'pause'}`,{method:'POST'});await poll(); }
  catch(error){notice(error.message,true);}finally{button.disabled=false;}
}
function liveCard(segment) {
  const card=node('article',undefined,'segment rolling-segment');card.dataset.segmentId=segment.id;
  card.dataset.start=segment.start;card.dataset.end=segment.end;
  card.dataset.speaker=segment.speaker;
  const jid=selected.id,key=`${jid}:${segment.id}`;card.dataset.meetingId=jid;
  const avatar=node('span',`S${Object.keys(doc.speakers).indexOf(segment.speaker)+1}`,'speaker-avatar');
  avatar.dataset.color=Object.keys(doc.speakers).indexOf(segment.speaker)%8;avatar.setAttribute('aria-hidden','true');
  const body=node('div',undefined,'segment-body'),top=node('div',undefined,'segment-top');
  const identity=node('button',doc.speakers[segment.speaker],'name-speaker');
  identity.title='Name this speaker or remember their voice';identity.setAttribute('aria-label',`Name or remember voice for ${doc.speakers[segment.speaker]}`);
  identity.addEventListener('click',()=>openNamePicker(segment.speaker).catch(e=>notice(e.message,true)));
  const timing=node('span',`${passageTime(segment.start)}–${passageTime(segment.end)}`,'rolling-time');
  const state=segment.refinement_state || 'provisional';
  const reviewed=segment.review_resolution==='words_reviewed' && segment.text.trim();
  const badge=node('span',reviewed?'Words reviewed':({provisional:'Provisional',refined:'Refined',unresolved:'Needs review',edited:'Edited'})[state] || state,'refinement-badge');
  badge.title=state==='provisional'?'Words may change with more context.':passageReviewReason(segment);
  top.append(timing,identity);if(state!=='provisional' || reviewed)top.append(badge);body.append(top);
  const draft=passageDrafts.get(segment.id),text=node('textarea');
  text.rows=1;text.value=draft?.text ?? segment.text;
  if(!text.value.trim() && recoverablePassageDrafts.has(segment.id))text.placeholder='Latest words are empty. Your correction is available to recover.';
  text.setAttribute('aria-label',`Transcript at ${time(segment.start)}`);
  const actions=node('div',undefined,'rolling-actions'),save=node('button','Save passage','quiet');
  save.classList.add('save-passage');
  const message=node('span','','passage-save-status');message.setAttribute('role','status');
  save.disabled=!draft;
  card.classList.toggle('has-draft',!!draft);
  text.addEventListener('input',()=>{
    const prior=passageDrafts.get(segment.id);
    passageDrafts.set(segment.id,{text:text.value,revision:prior?.revision ?? segment.machine_revision ?? 0,segment:structuredClone(segment)});
    card.classList.add('has-draft');
    const comparisonCopy=card.querySelector('.local-correction-copy');
    comparisonCopy.value=text.value;fitPassageText(comparisonCopy);
    renderPassageSaveState(card);text.style.height='auto';text.style.height=`${text.scrollHeight}px`;
  });
  text.addEventListener('focus',()=>{followingLive=false;renderFollowLive();});
  const keep=node('button','Keep my correction','quiet'),recover=node('button','Recover my correction','quiet');
  keep.classList.add('keep-correction');recover.classList.add('recover-correction');
  const saveDraft=async revision=>{
    const pending=passageDrafts.get(segment.id);if(!pending || selected?.id!==jid || passageSaveOperations.has(key))return;
    const operation={jid,segment:structuredClone(doc.segments.find(row=>row.id===segment.id) || pending.segment),recovery:recoverablePassageDrafts.get(segment.id)};
    passageSaveOperations.set(key,operation);renderPassageSaveState(card);
    try {
      const result=await api(`/api/jobs/${jid}/segments/${encodeURIComponent(segment.id)}`,{method:'PATCH',body:JSON.stringify({segment_revision:revision,changes:{text:pending.text}})});
      if(selected?.id!==jid || passageSaveOperations.get(key)!==operation)return;
      // Typing during a save creates a new draft against the just-saved revision.
      const current=passageDrafts.get(segment.id);
      if(current?.text===pending.text)passageDrafts.delete(segment.id);
      else if(current)current.revision=result.segment.machine_revision;
      if(recoverablePassageDrafts.get(segment.id)===operation.recovery)recoverablePassageDrafts.delete(segment.id);
      const index=doc.segments.findIndex(s=>s.id===segment.id);if(index>=0)doc.segments[index]=result.segment;
      message.textContent=passageDrafts.has(segment.id)?'Earlier correction saved. Your newer draft is retained.':'Saved · protected from refinement';
      card.dataset.signature='';if(!passageDrafts.has(segment.id)){save.blur();keep.blur();renderLiveSegments();}await poll();
    } catch(error){message.textContent=error.message;await poll();}
    finally{
      if(passageSaveOperations.get(key)===operation)passageSaveOperations.delete(key);
      if(selected?.id===jid)renderLiveSegments();
    }
  };
  save.addEventListener('click',()=>saveDraft(passageDrafts.get(segment.id)?.revision));
  keep.addEventListener('click',()=>saveDraft(Number(keep.dataset.revision)));
  const latest=node('button','Use latest words','text-button');
  latest.classList.add('use-latest');
  latest.addEventListener('click',()=>{
    if(selected?.id!==jid || passageSaveOperations.has(key))return;
    const current=doc.segments.find(s=>s.id===segment.id);
    if(!current){message.textContent='Passage was rearranged. Your draft is retained here for copying.';return;}
    const pending=passageDrafts.get(segment.id);
    if(pending)recoverablePassageDrafts.set(segment.id,structuredClone(pending));
    passageDrafts.delete(segment.id);latest.blur();text.blur();card.dataset.signature='';renderLiveSegments();
  });
  recover.addEventListener('click',()=>{
    if(selected?.id!==jid || passageSaveOperations.has(key))return;
    const previous=recoverablePassageDrafts.get(segment.id);if(!previous)return;
    passageDrafts.set(segment.id,structuredClone(previous));text.value=previous.text;save.disabled=false;fitPassageText(text);
    recover.blur();card.dataset.signature='';renderLiveSegments();
  });
  const comparison=node('div',undefined,'passage-comparison');
  const copy=node('textarea');copy.readOnly=true;copy.rows=1;copy.className='local-correction-copy';copy.setAttribute('aria-label','Your local correction for comparison or copying');
  comparison.append(node('strong','Your correction'),copy,node('strong','Latest machine words'),node('p','','latest-machine-words'));
  actions.append(save,keep,latest,recover,message);body.append(text,actions,comparison);
  const reason=passageReviewReason(segment);
  if(reason && state!=='unresolved')body.append(node('span',reason,'passage-context-note'));
  if(state==='unresolved' && (!reviewed || reason)) {
    const details=node('details',undefined,'rolling-review');details.append(node('summary','Review details'));
    details.append(node('p',passageReviewReason(segment) || 'Previous words and original audio retained.','passage-review'));
    appendUncoveredAudio(details,segment);
    if(segment.refinement_window) {
      const prior=node('button','Show prior words','text-button');
      prior.addEventListener('click',async()=>{
        try {const history=await api(`/api/jobs/${selected.id}/refinement/revisions/${encodeURIComponent(segment.refinement_window)}`);
          const previous=history.segments.filter(s=>s.start<segment.end && s.end>segment.start).map(s=>s.text).filter(Boolean).join('\n');
          details.append(node('p',previous || 'No previous legible words.','prior-words'));prior.remove();}
        catch(error){notice(error.message,true);}
      });details.append(prior);
    }body.append(details);
  }
  card.append(avatar,body);renderPassageSaveState(card);return card;
}
function renderLiveSegments() {
  const pane=$('transcript-pane'),host=$('segments'),scroll=pane.scrollTop;
  const focused=host.contains(document.activeElement);
  const anchor=Array.from(host.children).find(card=>card.dataset.start && card.getBoundingClientRect().bottom>pane.getBoundingClientRect().top);
  const anchorTime=anchor?Number(anchor.dataset.start):null,offset=anchor?anchor.getBoundingClientRect().top-pane.getBoundingClientRect().top:0;
  const existing=new Map(Array.from(host.children).map(card=>[card.dataset.segmentId,card]));
  const query=$('search').value.toLowerCase();let cursor=host.firstChild,pending=0,visible=0;
  const rows=doc.segments.slice();
  for(const [id,draft] of passageDrafts)if(!rows.some(s=>s.id===id))rows.push(draft.segment);
  for(const operation of passageSaveOperations.values())if(operation.jid===selected.id && !rows.some(s=>s.id===operation.segment.id))rows.push(operation.segment);
  rows.sort((a,b)=>a.start-b.start || a.end-b.end);
  const retained=new Set();
  for(const segment of rows) {
    if(!showTranscriptPassage(segment) && !isPassageSaving(segment.id)){if(segment.refinement_state==='provisional' && segment.audio_state!=='digital_silence')pending++;continue;}
    if(query && !(segment.text+' '+doc.speakers[segment.speaker]).toLowerCase().includes(query) && !passageDrafts.has(segment.id))continue;
    const signature=JSON.stringify([segment,doc.speakers[segment.speaker]]);let card=existing.get(segment.id);
    if(!card || (card.dataset.signature!==signature && !card.contains(document.activeElement) && !passageDrafts.has(segment.id) && !isPassageSaving(segment.id))) {
      const next=liveCard(segment);next.dataset.signature=signature;
      if(card){if(cursor===card)cursor=next;card.replaceWith(next);}card=next;
    }
    retained.add(card);visible++;
    card.dataset.speaker=segment.speaker;
    const name=card.querySelector('.name-speaker');name.textContent=doc.speakers[segment.speaker];
    name.setAttribute('aria-label',`Name or remember voice for ${doc.speakers[segment.speaker]}`);
    if(card!==cursor)host.insertBefore(card,cursor);cursor=card.nextSibling;
    const draft=passageDrafts.get(segment.id);
    card.classList.toggle('has-draft',!!draft);
    card.classList.toggle('has-new-words',!!draft && draft.revision!==segment.machine_revision);
    card.classList.toggle('has-recoverable-draft',recoverablePassageDrafts.has(segment.id));
    renderPassageSaveState(card);
    if(draft && draft.revision!==segment.machine_revision) {
      card.querySelector('.passage-save-status').textContent='New words arrived. Compare both versions, then keep your correction or use the latest words.';
      card.querySelector('.local-correction-copy').value=draft.text;
      card.querySelector('.latest-machine-words').textContent=segment.text || 'No machine words returned; audio retained.';
      // This is the exact revision shown alongside the user's correction. A
      // newer unseen result must still cause a CAS conflict on explicit keep.
      card.querySelector('.keep-correction').dataset.revision=segment.machine_revision ?? 0;
    }
  }
  for(const child of Array.from(host.children))if(!retained.has(child))child.remove();
  renderRetainedAudioReview();
  groupConsecutivePassages(host);
  for(const card of host.children) {
    for(const text of card.querySelectorAll('textarea'))if(text.readOnly || !card.contains(document.activeElement))fitPassageText(text);
  }
  $('pending-phrases').hidden=!pending;$('pending-phrases').textContent=pending?'Listening · uncertain phrases are waiting for more context. Original audio retained.':'';
  $('listening').hidden=visible>0 || pending>0 || !isLive();
  renderSearchResults(visible,query);
  $('no-results').hidden=visible>0 || !query;
  if(isLive() && followingLive && !focused)pane.scrollTop=pane.scrollHeight;
  else if(anchorTime!==null) {
    const next=Array.from(host.children).find(card=>Number(card.dataset.start)<=anchorTime && Number(card.dataset.end)>anchorTime) || host.querySelector(`[data-segment-id="${anchor.dataset.segmentId}"]`);
    pane.scrollTop=next?scroll+next.getBoundingClientRect().top-pane.getBoundingClientRect().top-offset:scroll;
  }else pane.scrollTop=scroll;
  renderRefinementStatus();
}
