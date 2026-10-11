// Existing editor controller with a synthetic DOM, not desktop acceptance.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {frontend} = require('./support/frontend_dom.cjs');

test('explicit-language import can run while native automatic language and alignment are unavailable', () => {
  const f = frontend(path.resolve(__dirname, '..'));f.seed(0, 'uploaded');
  f.run("config.readiness={configured:true,automatic_language:false,word_alignment:false};setupState={core_ready:true,alignment:{ready:false}};setStatus();");
  assert.equal(f.document.getElementById('run').disabled, false);
  f.run("selected.language='auto';setStatus();");
  assert.equal(f.document.getElementById('run').disabled, true);
  f.run("selected.language='en';config.readiness.configured=false;setStatus();");
  assert.equal(f.document.getElementById('run').disabled, true);
});

test('required alignment on the existing runtime still blocks import until ready', () => {
  const f = frontend(path.resolve(__dirname, '..'));f.seed(0, 'uploaded');
  f.run("setupState={core_ready:true,alignment:{ready:false}};setStatus();");
  assert.equal(f.document.getElementById('run').disabled, true);
  f.run("setupState.alignment.ready=true;setStatus();");
  assert.equal(f.document.getElementById('run').disabled, false);
});
