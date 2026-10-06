'use strict';
// Drafts are local until a passage revision is explicitly saved. Polling keeps
// arriving words visible without replacing the focused textarea or its caret.
const passageDrafts = new Map();
function hasPassageDrafts() { return passageDrafts.size > 0; }
function renderRefinementStatus() {
  const status=selected?.refinement_status;
  $('refinement-controls').hidden=!selected?.rolling_refinement;
  $('refinement-status').textContent=({waiting:'Larger context queued',refining:'Refining earlier phrases',paused:'Refinement paused · audio retained',complete:'Refinement complete',unresolved:'Some phrases need review'})[status] || '';
  $('refinement-toggle').hidden=status==='complete';
  $('refinement-toggle').textContent=status==='paused'?'Resume refinement':status==='unresolved'?'Retry unresolved refinement':'Pause refinement';
}
async function toggleRefinement() {
  const jid=selected.id,button=$('refinement-toggle');button.disabled=true;
  try { await api(`/api/jobs/${jid}/refinement/${['paused','unresolved'].includes(selected.refinement_status)?'resume':'pause'}`,{method:'POST'});await poll(); }
  catch(error){notice(error.message,true);}finally{button.disabled=false;}
}
function liveCard(segment) {
  const card=node('article',undefined,'segment rolling-segment');card.dataset.segmentId=segment.id;
  card.dataset.start=segment.start;card.dataset.end=segment.end;
  const avatar=node('span',`S${Object.keys(doc.speakers).indexOf(segment.speaker)+1}`,'speaker-avatar');
  avatar.dataset.color=Object.keys(doc.speakers).indexOf(segment.speaker)%8;avatar.setAttribute('aria-hidden','true');
  const body=node('div',undefined,'segment-body'),top=node('div',undefined,'segment-top');
  const identity=node('button',doc.speakers[segment.speaker],'name-speaker');
  identity.title='Name / remember voice';identity.setAttribute('aria-label',`Name or remember voice for ${doc.speakers[segment.speaker]}`);
  identity.addEventListener('click',()=>openNamePicker(segment.speaker).catch(e=>notice(e.message,true)));
  const timing=node('span',`${passageTime(segment.start)}–${passageTime(segment.end)}`,'rolling-time');
  const state=segment.refinement_state || 'provisional';
  const badge=node('span',({provisional:'Provisional',refined:'Refined',unresolved:'Needs review',edited:'Edited'})[state] || state,'refinement-badge');
  badge.title=state==='provisional'?'Words may change with more context.':passageReviewReason(segment);
  top.append(identity,timing,badge);body.append(top);
  const draft=passageDrafts.get(segment.id),text=node('textarea');
  text.rows=2;text.value=draft?.text ?? segment.text;
  text.setAttribute('aria-label',`Transcript at ${time(segment.start)}`);
  const actions=node('div',undefined,'rolling-actions'),save=node('button','Save passage','quiet');
  const message=node('span','','passage-save-status');message.setAttribute('role','status');
  save.disabled=!draft;
  text.addEventListener('input',()=>{
    const prior=passageDrafts.get(segment.id);
    passageDrafts.set(segment.id,{text:text.value,revision:prior?.revision ?? segment.machine_revision ?? 0,segment:structuredClone(segment)});
    save.disabled=false;text.style.height='auto';text.style.height=`${text.scrollHeight}px`;
  });
  text.addEventListener('focus',()=>{followingLive=false;$('follow-live').textContent='Return to live';});
  save.addEventListener('click',async()=>{
    const pending=passageDrafts.get(segment.id),jid=selected.id;if(!pending)return;
    save.disabled=true;
    try {
      const result=await api(`/api/jobs/${jid}/segments/${encodeURIComponent(segment.id)}`,{method:'PATCH',body:JSON.stringify({segment_revision:pending.revision,changes:{text:pending.text}})});
      if(selected?.id!==jid)return;
      // Typing during a save creates a new draft against the just-saved revision.
      const current=passageDrafts.get(segment.id);
      if(current?.text===pending.text)passageDrafts.delete(segment.id);
      else if(current)current.revision=result.segment.machine_revision;
      const index=doc.segments.findIndex(s=>s.id===segment.id);if(index>=0)doc.segments[index]=result.segment;
      message.textContent='Saved · protected from refinement';
      card.dataset.signature='';if(!passageDrafts.has(segment.id)){save.blur();renderLiveSegments();}await poll();
    } catch(error){message.textContent=error.message;save.disabled=false;await poll();}
  });
  const latest=node('button','Use latest words','text-button');
  latest.addEventListener('click',()=>{
    const current=doc.segments.find(s=>s.id===segment.id);
    if(!current){message.textContent='Passage was rearranged. Your draft is retained here for copying.';return;}
    passageDrafts.delete(segment.id);latest.blur();text.blur();card.dataset.signature='';renderLiveSegments();
  });
  actions.append(save,latest,message);body.append(text,actions);
  if(state==='unresolved') {
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
  card.append(avatar,body);return card;
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
  rows.sort((a,b)=>a.start-b.start || a.end-b.end);
  const retained=new Set();
  for(const segment of rows) {
    if(!segment.text.trim() && segment.refinement_state==='provisional' && !passageDrafts.has(segment.id)){pending++;continue;}
    if(query && !(segment.text+' '+doc.speakers[segment.speaker]).toLowerCase().includes(query) && !passageDrafts.has(segment.id))continue;
    const signature=JSON.stringify([segment,doc.speakers[segment.speaker]]);let card=existing.get(segment.id);
    if(!card || (card.dataset.signature!==signature && !card.contains(document.activeElement) && !passageDrafts.has(segment.id))) {
      const next=liveCard(segment);next.dataset.signature=signature;
      if(card){if(cursor===card)cursor=next;card.replaceWith(next);}card=next;
    }
    retained.add(card);visible++;
    const name=card.querySelector('.name-speaker');name.textContent=doc.speakers[segment.speaker];
    name.setAttribute('aria-label',`Name or remember voice for ${doc.speakers[segment.speaker]}`);
    if(card!==cursor)host.insertBefore(card,cursor);cursor=card.nextSibling;
    const draft=passageDrafts.get(segment.id);
    if(draft && draft.revision!==segment.machine_revision) {
      card.querySelector('.passage-save-status').textContent=`New words available: ${segment.text}. Your draft is retained; use latest words before editing the new version.`;
    }
  }
  for(const child of Array.from(host.children))if(!retained.has(child))child.remove();
  $('pending-phrases').hidden=!pending;$('pending-phrases').textContent=pending?'Listening · uncertain phrases are waiting for more context. Original audio retained.':'';
  $('listening').hidden=visible>0 || pending>0 || !isLive();
  $('no-results').hidden=visible>0 || !query;
  if(isLive() && followingLive && !focused)pane.scrollTop=pane.scrollHeight;
  else if(anchorTime!==null) {
    const next=Array.from(host.children).find(card=>Number(card.dataset.start)<=anchorTime && Number(card.dataset.end)>anchorTime) || host.querySelector(`[data-segment-id="${anchor.dataset.segmentId}"]`);
    pane.scrollTop=next?scroll+next.getBoundingClientRect().top-pane.getBoundingClientRect().top-offset:scroll;
  }else pane.scrollTop=scroll;
  renderRefinementStatus();
}
