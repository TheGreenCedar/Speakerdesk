// Offline headless Chrome layout/a11y checks and synthetic screenshots.
// Uses file fixtures, no server, capture, audio source, models or app profile.
const assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process');
const {readFileSync,writeFileSync,mkdirSync,mkdtempSync,rmSync}=require('node:fs');
const {resolve,join}=require('node:path');
const {pathToFileURL}=require('node:url');
const [chromium,output]=process.argv.slice(2);
assert(chromium && output,'Pass an existing Chrome executable and private evidence directory.');
const root=resolve(__dirname,'..'),out=resolve(output);mkdirSync(out,{recursive:true,mode:0o700});
const read=file=>readFileSync(join(root,file),'utf8');
const page=read('speakerdesk/templates/index.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,'').replace(/<link\b[^>]*>/g,'').replace('{{token}}','synthetic-keycaps-only');
const script=[read('speakerdesk/static/lucide.js'),read('speakerdesk/static/people.js'),read('speakerdesk/static/live_editor.js'),read('speakerdesk/static/app.js').replace(/\ninit\(\);\s*$/,'\n')].join('\n');
const reports=[];
for(const [name,width,theme,platform] of [['desktop',1280,'light','MacIntel'],['narrow',700,'light','MacIntel'],['dark',1000,'dark','MacIntel'],['control',900,'light','Win32']]){
  const test=`(async()=>{const checks=[];const check=(ok,message)=>{if(!ok)throw Error(message);checks.push(message);};
    Object.defineProperty(navigator,'platform',{value:${JSON.stringify(platform)}});
    document.documentElement.dataset.theme=${JSON.stringify(theme)};
    window.fetch=async()=>{throw Error('No network allowed in shortcut visual fixture');};window.loadPeople=async()=>{};
    config={languages:{auto:'Automatic',en:'English'},readiness:{configured:true,automatic_language:true},default_language:'en',capture:{available:false}};
    const fixture={id:'${'c'.repeat(32)}',name:'Design sync',created:1791468000,status:'ready',language:'en',duration:32,revision:1,refinement_status:'complete',message:'',
      document:{schema_version:1,speakers:{s0:'Maya',s1:'Lee'},segments:[
        {id:'r0',start:0,end:8,speaker:'s0',text:'Let’s keep the meeting tools easy to find.',language:'en',machine_revision:1,refinement_state:'refined'},
        {id:'r1',start:8,end:17,speaker:'s1',text:'A small keycap beside each action makes the shortcuts easy to learn.',language:'en',machine_revision:1,refinement_state:'refined'},
        {id:'r2',start:17,end:32,speaker:'s0',text:'The hints should feel quiet and stay readable in a smaller window.',language:'en',machine_revision:1,refinement_state:'refined'}]}};
    function restore(){selected=structuredClone(fixture);doc=structuredClone(fixture.document);dirty=false;jobs=[structuredClone(fixture)];$('empty').hidden=true;$('workspace').hidden=false;setStatus();renderJobs();renderEditor();renderIcons();renderShortcutHints();}
    wire();restore();
    const prefix=${JSON.stringify(platform==='MacIntel'?'⌘':'Ctrl+')},ariaPrefix=${JSON.stringify(platform==='MacIntel'?'Meta':'Control')};
    const controls=[...document.querySelectorAll('[data-shortcut-action]')];check(controls.length===6,'All six bound UI controls are annotated');
    renderShortcutHints();renderShortcutHints();check(document.querySelectorAll('.shortcut-hint').length===6,'Rerenders never duplicate keycaps');
    for(const control of controls){const host=control.tagName==='INPUT'?control.closest('label'):control,cap=host.querySelector('kbd');
      check(cap.getAttribute('aria-hidden')==='true' && !cap.hasAttribute('tabindex'),'Decorative keycaps do not change accessibility or tab order: '+control.id);
      if(control.id!=='meeting-search')check(control.getAttribute('aria-keyshortcuts').startsWith(ariaPrefix+'+'),'Platform-correct accessible shortcut: '+control.id);
      check(cap.textContent.startsWith(prefix),'Platform-correct visible shortcut: '+control.id);
      check(host.title.length>0 && control.title.length>0,'Icon and input hover targets both expose a tooltip: '+control.id);}
    check(!$('people-toggle').hasAttribute('aria-keyshortcuts') && !$('start-meeting').querySelector('kbd') && !$('voice-settings').querySelector('kbd'),'Unbound actions acquire no shortcut');
    check($('meeting-search').closest('label').querySelector('kbd').hidden,'Meeting Find hint is absent when Command F targets transcript');
    const existing=$('save').querySelector('kbd');dirty=true;renderSaveState();check($('save').querySelector('kbd')===existing && $('save').querySelector('.shortcut-label').textContent==='Save now','Save label rerender preserves keycap and exact label');
    dirty=false;renderSaveState();check($('save').disabled,'Saved action stays disabled');
    await dispatchAction('new-meeting');check($('workspace').hidden && !$('meeting-search').closest('label').querySelector('kbd').hidden,'New screen exposes contextual meeting Find hint');restore();
    const dialog=$('name-dialog');dialog.showModal();check(await dispatchAction('new-meeting')===false,'Hints do not bypass modal action guards');dialog.close();
    window.speakerdeskNativeMenu=true;const command=new KeyboardEvent('keydown',{key:'n',metaKey:true,bubbles:true,cancelable:true});document.dispatchEvent(command);check(!command.defaultPrevented && !!selected,'Native Command event stays uncancelled with no browser duplicate');delete window.speakerdeskNativeMenu;
    renderShortcutHints();const editing=new KeyboardEvent('keydown',{key:'a',metaKey:true,bubbles:true,cancelable:true});document.dispatchEvent(editing);check(!editing.defaultPrevented,'Standard editing remains untouched');
    if(innerWidth<=900){check($('search').getBoundingClientRect().width===0,'Narrow Find control has its existing icon-only state');
      check($('search').getAttribute('aria-keyshortcuts')===ariaPrefix+'+F' && $('search').closest('label').title.includes(prefix+'F'),'Icon-only Find retains tooltip and accessible shortcut');
      $('search').focus();check($('search').getBoundingClientRect().width>0,'Icon-only Find expands with keyboard focus');$('search').blur();}
    const layout=[];for(const control of controls){if(control.id==='meeting-search')continue;const host=control.tagName==='INPUT'?control.closest('label'):control,cap=host.querySelector('kbd');
      const a=host.getBoundingClientRect(),b=cap.getBoundingClientRect();layout.push({id:control.id,host:{x:a.x,width:a.width},cap:{x:b.x,width:b.width}});
      check(host.scrollWidth<=host.clientWidth+1,'Control content is not clipped: '+control.id);
      check(b.left>=a.left-1 && b.right<=a.right+1,'Keycap fits its target: '+control.id);}
    check(document.documentElement.scrollWidth<=innerWidth,'No page horizontal overflow');
    dirty=true;renderSaveState();document.activeElement?.blur();
    const result={pass:true,width:innerWidth,height:innerHeight,theme:${JSON.stringify(theme)},platform:${JSON.stringify(platform)},checks,layout,synthetic:true,nativeWebview:false,network:false};
    const proof=document.createElement('pre');proof.id='shortcut-browser-result';proof.hidden=true;proof.textContent=JSON.stringify(result);document.body.append(proof);
  })().catch(error=>{const proof=document.createElement('pre');proof.id='shortcut-browser-result';proof.hidden=true;proof.textContent=JSON.stringify({pass:false,error:error.stack});document.body.append(proof);});`;
  const html=page.replace('</head>',`<style>${read('speakerdesk/static/style.css')}</style></head>`).replace('</body>',`<script>${script}</script><script>${test}</script></body>`);
  const fixture=join(out,`${name}-fixture.html`);writeFileSync(fixture,html);const profile=mkdtempSync(join(out,'chrome-profile-'));
  try{
    const proc=spawnSync(chromium,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync',
      '--host-resolver-rules=MAP * ~NOTFOUND',`--user-data-dir=${profile}`,`--window-size=${width},800`,'--virtual-time-budget=2000',`--screenshot=${join(out,`${name}.png`)}`,'--dump-dom',pathToFileURL(fixture).href],{encoding:'utf8',timeout:45000,maxBuffer:12*1024*1024});
    writeFileSync(join(out,`${name}-chrome.stderr.txt`),proc.stderr||'');
    writeFileSync(join(out,`${name}-process.json`),JSON.stringify({status:proc.status,signal:proc.signal,error:proc.error?.message},null,2));
    assert.equal(proc.status,0,proc.error?.message||`Browser exited with ${proc.signal}: ${proc.stderr}`);
    const match=proc.stdout.match(/<pre id="shortcut-browser-result" hidden="">([\s\S]*?)<\/pre>/);assert(match,'Browser proof missing');
    const result=JSON.parse(match[1].replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>'));writeFileSync(join(out,`${name}-result.json`),JSON.stringify(result,null,2));assert(result.pass,result.error);reports.push(result);
  }finally{rmSync(profile,{recursive:true,force:true});}
}
writeFileSync(join(out,'browser-summary.json'),JSON.stringify({reports:reports.map(r=>({width:r.width,theme:r.theme,platform:r.platform,checks:r.checks.length,pass:r.pass})),synthetic:true,nativeWebview:false,noServer:true},null,2));
console.log(JSON.stringify(reports.map(r=>({width:r.width,theme:r.theme,platform:r.platform,checks:r.checks.length,pass:r.pass}))));
