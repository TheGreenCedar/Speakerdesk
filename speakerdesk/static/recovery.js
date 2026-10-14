'use strict';
const recoveryPageId=crypto.randomUUID().replaceAll('-','');
// Recovery is a separate copy for review. It never writes canonical passages.
window.speakerdeskRecoverySnapshot=(closing=false)=>{
  if(closing){
    recoveryClosing=true;
    for(const surface of document.querySelectorAll('main,.sidebar,.settings-overlay,dialog'))surface.inert=true;
  }
  const hasUnsaved=dirty || saving || hasPassageDrafts() || hasRecoverablePassageDrafts() || hasPendingPassageSaves() || timingDrafts.size>0;
  return {schema_version:1,receiver_ready:true,page_id:recoveryPageId,job_id:selected?.id || null,name:selected?.name || '',revision:selected?.revision ?? null,
    has_unsaved:!!hasUnsaved,document:dirty || saving?structuredClone(doc):null,
    passage_drafts:structuredClone(Array.from(passageDrafts)),recoverable_drafts:structuredClone(Array.from(recoverablePassageDrafts)),
    pending_passage_saves:structuredClone(Array.from(passageSaveOperations)),timing_drafts:Array.from(timingDrafts)};
};
window.speakerdeskRecoveryAcknowledged=()=>{
  $('recovery-save-status').textContent='Draft recovery is up to date.';
};
window.speakerdeskRecoveryFailed=(message,closing=false)=>{
  if(closing && !recoveryClosing)return;
  if(closing){
    recoveryClosing=false;
    for(const surface of document.querySelectorAll('main,.sidebar,.settings-overlay,dialog'))surface.inert=false;
    if(!$('setup').hidden){document.querySelector('main').inert=true;document.querySelector('.sidebar').inert=true;}
    if(selected)setStatus();
  }
  $('recovery-copies').hidden=false;$('recovery-copies').open=true;
  if($('recovery-save-status').textContent!==message)$('recovery-save-status').textContent=message;
  $('recovery-copy-open').hidden=false;
};
function recoveryText(receipt){
  const snapshot=receipt.snapshot || receipt,parts=[`Recovery copy: ${snapshot.name || 'Meeting'}`,`Recording: ${snapshot.job_id || ''}`,`Saved revision: ${snapshot.revision ?? 'unknown'}`,'Review these words against the current transcript. This copy has not been applied.'];
  for(const segment of snapshot.document?.segments || [])parts.push(`${passageTime(segment.start)} · ${segment.speaker || ''}\n${segment.text || ''}`);
  for(const [label,entries] of [['Passage corrections',snapshot.passage_drafts],['Recoverable corrections',snapshot.recoverable_drafts]])
    for(const [id,draft] of entries || [])parts.push(`${label} · ${id}\n${draft.text || ''}`);
  for(const [id,operation] of snapshot.pending_passage_saves || [])if(operation.changes)parts.push(`Pending passage changes · ${id}\n${JSON.stringify(operation.changes,null,2)}`);
  if(snapshot.timing_drafts?.length)parts.push('Uncommitted timing fields:\n'+snapshot.timing_drafts.map(([key,value])=>`${key}: ${value}`).join('\n'));
  return parts.join('\n\n');
}
function showRecoveryText(receipt){
  const text=$('recovery-text');text.value=recoveryText(receipt);text.hidden=false;text.focus();text.select();
  $('recovery-save-status').textContent='Recovery words selected. Copy them, then review them beside the current transcript. Nothing is applied automatically.';
}
async function loadRecoveryCopies(){
  try{
    const result=await api('/api/recovery'),host=$('recovery-list');host.replaceChildren();
    for(const receipt of result.receipts || []){
      const row=node('div',undefined,'recovery-copy-row'),title=node('span',receipt.name || 'Meeting');
      const review=node('button','Review / copy','quiet');review.type='button';review.addEventListener('click',async()=>{
        try{showRecoveryText(await api(`/api/recovery/${receipt.id}`));}catch(error){$('recovery-save-status').textContent=error.message;}
      });
      const discard=node('button','Discard copy','text-button');discard.type='button';discard.addEventListener('click',async()=>{
        if(!confirm('Discard this recovery copy? Saved transcripts and open edits stay unchanged.'))return;
        try{await api(`/api/recovery/${receipt.id}`,{method:'DELETE'});await loadRecoveryCopies();}catch(error){$('recovery-save-status').textContent=error.message;}
      });row.append(title,review,discard);host.append(row);
    }
    $('recovery-copies').hidden=!result.receipts?.length && !serviceUnavailable;
  }catch(error){if(serviceUnavailable)$('recovery-copy-open').hidden=false;}
}
$('recovery-copy-open').addEventListener('click',()=>showRecoveryText(window.speakerdeskRecoverySnapshot()));
$('recovery-refresh').addEventListener('click',()=>loadRecoveryCopies());
