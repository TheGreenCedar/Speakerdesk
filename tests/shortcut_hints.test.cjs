// Real frontend hint behavior in the CPU DOM fixture; no native or model work.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');
function fixture(platform='MacIntel') {
  const f=frontend(root);f.seed();f.context.navigator={platform};
  f.Element.prototype.select=function(){this.setSelectionRange(0,this.value.length);};
  for(const [id,action,title,input] of [
    ['new-meeting','new-meeting','New meeting'],['setup-toggle','settings','Settings'],
    ['files','import-recording','Import recording',true],['meeting-search','find','Search saved meetings',true],
    ['search','find','Find in transcript',true],['save','save','All changes saved on this Mac']
  ]) {
    const control=f.document.getElementById(id);control.tagName=input?'INPUT':'BUTTON';
    control.dataset.shortcutAction=action;control.dataset.shortcutTitle=title;
    if(input){const label=f.document.createElement('label');control.parentElement.append(label);label.append(control);}
    else {const label=f.document.createElement('span');label.className='shortcut-label';label.textContent=title;control.append(label);}
  }
  f.run("$('setup').hidden=true;$('workspace').hidden=false;renderShortcutHints();");return f;
}

test('native and browser bindings, template targets, and hints use the same five actions',()=>{
  const f=fixture();const mapping=JSON.parse(f.run('JSON.stringify(actionShortcuts)'));
  assert.deepEqual(mapping,{settings:',','new-meeting':'n','import-recording':'o',find:'f',save:'s'});
  const menus=fs.readFileSync(path.join(root,'desktop/src-tauri/src/menus.rs'),'utf8');
  for(const [action,key] of Object.entries(mapping)) {
    const call=[...menus.matchAll(/MenuItem::with_id\(([^;]+)\)\?/g)].find(m=>m[1].includes(`"${action}"`));
    assert.ok(call,action);assert.ok(call[1].includes(`"Cmd+${key.toUpperCase()}"`),action);
  }
  const html=fs.readFileSync(path.join(root,'speakerdesk/templates/index.html'),'utf8');
  const bound=[...html.matchAll(/data-shortcut-action="([^"]+)"/g)].map(m=>m[1]);
  assert.deepEqual(bound,['new-meeting','find','import-recording','settings','find','save']);
  assert.ok(bound.every(action=>mapping[action]));
  assert.equal(html.includes('aria-keyshortcuts="Meta+'),false,'Platform metadata must come from the binding map.');
});

test('keycaps retain labels and focus targets, use accessible metadata, and never duplicate on rerender',()=>{
  const f=fixture();f.run('renderShortcutHints();renderShortcutHints();');
  assert.equal(f.document.querySelectorAll('.shortcut-hint').length,6);
  for(const control of f.document.querySelectorAll('[data-shortcut-action]')) {
    const host=control.tagName==='INPUT'?control.closest('label'):control;
    const hint=host.querySelector('.shortcut-hint');
    assert.equal(hint.tagName,'KBD');assert.equal(hint.getAttribute('aria-hidden'),'true');
    assert.equal(hint.getAttribute('tabindex'),null);
    if(control.id!=='meeting-search')assert.match(control.getAttribute('aria-keyshortcuts'),/^Meta\+/);
    assert.match(host.title,new RegExp(control.dataset.shortcutTitle));
  }
  assert.equal(f.document.getElementById('new-meeting').querySelector('.shortcut-label').textContent,'New meeting');
});

test('Find hints and accessibility follow the same contextual target as Find dispatch',async()=>{
  const f=fixture();const library=f.document.getElementById('meeting-search'),search=f.document.getElementById('search');
  assert.equal(library.closest('label').querySelector('kbd').hidden,true);assert.equal(library.getAttribute('aria-keyshortcuts'),null);
  assert.equal(await f.run("dispatchAction('find')"),true);assert.equal(f.document.activeElement,search);
  f.run("$('workspace').hidden=true;renderShortcutHints();");
  assert.equal(library.closest('label').querySelector('kbd').hidden,false);assert.equal(library.getAttribute('aria-keyshortcuts'),'Meta+F');
  assert.equal(search.getAttribute('aria-keyshortcuts'),null);
  assert.equal(await f.run("dispatchAction('find')"),true);assert.equal(f.document.activeElement,library);
});

test('Save hint survives saved, dirty, saving and frozen states without changing action availability',()=>{
  const f=fixture(),save=f.document.getElementById('save'),hint=save.querySelector('kbd');
  for(const [setup,label,disabled] of [['dirty=false;saving=false;','Saved',true],['dirty=true;','Save now',false],['saving=true;','Saving…',true],['saving=false;updateFrozen=true;','Save now',true]]) {
    f.run(setup+'renderSaveState();');assert.equal(save.querySelector('.shortcut-label').textContent,label);
    assert.equal(save.querySelector('kbd'),hint);assert.equal(hint.textContent,'⌘S');assert.equal(save.disabled,disabled);
    assert.equal(save.getAttribute('aria-keyshortcuts'),'Meta+S');assert.match(save.title,/⌘S/);
  }
});

test('Windows browser hints use Control while native Mac retains Command',()=>{
  const f=fixture('Win32'),control=f.document.getElementById('new-meeting');
  assert.equal(control.querySelector('kbd').textContent,'Ctrl+N');assert.equal(control.getAttribute('aria-keyshortcuts'),'Control+N');
  f.context.speakerdeskNativeMenu=true;f.run('renderShortcutHints();');
  assert.equal(control.querySelector('kbd').textContent,'⌘N');assert.equal(control.getAttribute('aria-keyshortcuts'),'Meta+N');
});
