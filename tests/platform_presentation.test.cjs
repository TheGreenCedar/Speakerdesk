// Real template copy and frontend controller; no browser layout or native capture.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {frontend}=require('./support/frontend_dom.cjs');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'speakerdesk/templates/index.html'),'utf8');

function fixture(platform) {
  const f=frontend(root,{platform});f.seed(0,'uploaded');
  // Supply the real template's leaf copy to the existing lightweight DOM.
  for(const id of ['recovery-location-note','start-description','setup-card-text','capture-system-label',
    'capture-note','privacy-copy','system-meter-label','setup-description','about-platform','about-models',
    'recording-access-macos','recording-access-windows']) {
    const match=html.match(new RegExp(`<([\\w-]+)\\b([^>]*\\bid="${id}"[^>]*)>([\\s\\S]*?)<\\/\\1>`));
    assert.ok(match,`Template copy target ${id} exists`);
    const el=f.document.getElementById(id);
    el.textContent=match[3].replace(/<[^>]*>/g,' ').replace(/&amp;/g,'&').replace(/\s+/g,' ').trim();
    el.hidden=/\bhidden\b/.test(match[2]);
  }
  f.run("if(typeof renderPlatformCopy==='function')renderPlatformCopy();");
  return f;
}

test('host platform selects Windows copy and recording guidance while Mac copy stays unchanged',async()=>{
  for(const [platform,device,audio,about] of [
    ['windows','PC','System audio','Windows x64.'],
    ['macos','Mac','Mac audio','Apple Silicon · macOS 15 or later · 16 GB memory recommended.'],
  ]) {
    const f=fixture(platform),el=id=>f.document.getElementById(id);
    // Browser identity is not the native host platform.
    f.context.navigator={platform:platform==='windows'?'MacIntel':'Win32'};
    assert.equal(el('privacy-copy').textContent,`Audio and transcripts stay on this ${device}`);
    assert.equal(el('capture-system-label').textContent,audio);
    assert.equal(el('system-meter-label').textContent,audio);
    assert.equal(el('about-platform').textContent,about);
    assert.match(el('start-description').textContent,new RegExp(`save on this ${device}\\.`));
    assert.match(el('recovery-location-note').textContent,new RegExp(`kept on this ${device} for review`));
    f.run('renderSaveState()');
    assert.equal(el('save').dataset.shortcutTitle,`All changes saved on this ${device}`);
    assert.equal(el('recording-access-macos').hidden,platform==='windows');
    assert.equal(el('recording-access-windows').hidden,platform!=='windows');
    const permissions=el(`recording-access-${platform}`).textContent;
    if(platform==='windows') {
      assert.match(permissions,/Microphone privacy settings/);
      assert.match(permissions,/Let desktop apps access your microphone/);
      assert.match(permissions,/default microphone and playback device/);
      assert.doesNotMatch(permissions,/macOS|Screen & System Audio Recording/);
      assert.match(el('capture-note').textContent,/default playback device, including other apps/);
      assert.doesNotMatch(el('setup-description').textContent,/disabled|not available|Mac/);
      assert.match(el('about-models').textContent,/exported third-party notices/);
      assert.doesNotMatch(el('about-models').textContent,/MLX|disabled|not available/);
    } else {
      assert.match(permissions,/two macOS permissions/);
      assert.match(permissions,/Screen & System Audio Recording/);
      assert.equal(el('capture-note').textContent,'Mac audio includes voices from your meeting app. Your Mac may ask for recording access when you start.');
      assert.equal(el('setup-description').textContent,'Models run locally. Audio never leaves your Mac.');
      assert.match(el('about-models').textContent,/MLX Audio runtime/);
    }
    for(const id of ['microphone-level','system-level']) {
      const meter=f.document.createElement('div');meter.className='audio-meter';
      meter.append(f.document.createElement('label'),el(id));f.document.body.append(meter);
    }
    el('pause-meeting').append(f.document.createElement('span'));
    f.run("api=async()=>({id:selected.id,status:'paused',duration:30,levels:{},sources:['microphone']})");
    await f.run('refreshMeeting()');
    assert.equal(el('system-level').closest('.audio-meter').querySelector('label').textContent,`${audio} · off`);
  }
});

test('Windows unavailable setup stays disabled and its explanation follows supported state',async()=>{
  const f=fixture('windows'),el=id=>f.document.getElementById(id);
  el('setup-card-install').append(f.document.createElement('span'));
  f.context.setupFixture={status:'failed',supported:false,can_download:false,core_ready:false,ready:false,
    automatic_language:false,total_bytes:2.2e9,downloaded_bytes:0,models:[],
    operation:{target:'models',status:'failed',error:'Local model initialization is unavailable.'},
    voice:{status:'unavailable',available:false,enabled:false}};
  f.run("config.readiness={configured:false,automatic_language:false,word_alignment:false};$('language').value='en';meeting={capture_available:true};api=async()=>structuredClone(setupFixture)");
  await f.run('refreshSetup()');
  const unavailable='Local transcription is not available in this Windows build. Model setup is disabled.';
  assert.equal(el('setup-card-text').textContent,unavailable);
  assert.equal(el('model-status').textContent,unavailable);
  assert.equal(el('setup-card-install').disabled,true);
  assert.equal(el('install-models').disabled,true);
  assert.equal(el('start-meeting').disabled,true);
  assert.equal(f.run('config.readiness.configured'),false);
  assert.equal(el('settings-voice-note').textContent,'Recognition is off. Saved names and voices stay on this PC.');
  assert.equal(f.run("modelSetupMessage({supported:true,status:'missing'})"),
    'Download the local models once. Meetings are transcribed offline on this PC.');
});

test('voice location copy follows the host platform and saving still requires Remember voice',async()=>{
  for(const [platform,device] of [['windows','PC'],['macos','Mac']]) {
    const f=fixture(platform),el=id=>f.document.getElementById(id);
    f.run(fs.readFileSync(path.join(root,'speakerdesk/static/people.js'),'utf8'));
    f.Element.prototype.close=function(){this.open=false;};
    f.run(`people=[{id:'person-a',name:'Ada',voice_saved:true,voice_compatible:true}];voiceAvailable=true;
      selected.speaker_assignments={s0:{person_id:'person-a',name:'Ada',source:'manual',confirmed:true}};
      nameContext={jid:selected.id,track:'s0',revision:selected.revision};$('name-dialog').open=true;$('name-person').value='person-a';
      nameClipState={status:'ready',minimum_clips:2,clips:[{id:'a',start:0,end:3},{id:'b',start:4,end:7}]};
      window.requests=[];api=async(path,options)=>{requests.push({path,body:JSON.parse(options.body)});return {};};renderPeople();`);
    for(const value of ['a','b']) {
      const input=f.document.createElement('input');input.value=value;input.checked=true;el('voice-clip-list').append(input);
    }
    assert.match(el('people-list').textContent,new RegExp(`Voice saved on this ${device}`));
    assert.equal(f.run('voiceEnrollmentState().ready'),true);
    assert.match(f.run('voiceEnrollmentState().reason'),new RegExp(`voice on this ${device} for future meetings`));
    assert.equal(f.run('requests.length'),0);
    f.run('wirePeople()');el('remember-voice').click();
    await new Promise(setImmediate);
    assert.equal(f.run('requests.length'),1);
    assert.equal(f.run('requests[0].body.consent'),true);
    assert.equal(el('notice').textContent,`Voice saved on this ${device} for future meetings.`);
  }
});
