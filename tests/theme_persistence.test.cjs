const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../speakerdesk/static/theme.js'),'utf8');
test('host preference paints before startup and follows system changes only in System mode',()=>{
  let systemChange;const media={matches:true,addEventListener(type,handler){systemChange=handler;}};
  const context={document:{documentElement:{dataset:{hostTheme:'light'},style:{}}},matchMedia:()=>media};context.window=context;
  vm.runInNewContext(source,context);
  assert.equal(context.document.documentElement.dataset.theme,'light');
  systemChange();assert.equal(context.document.documentElement.dataset.theme,'light');
  context.speakerdeskTheme.set('system');assert.equal(context.document.documentElement.dataset.theme,'dark');
  media.matches=false;systemChange();assert.equal(context.document.documentElement.dataset.theme,'light');
  assert.equal(context.speakerdeskTheme.get(),'system');
});
test('invalid host preference uses System without trusting per-port browser storage',()=>{
  const context={document:{documentElement:{dataset:{hostTheme:'invalid'},style:{}}},matchMedia:()=>({matches:true,addEventListener(){}}),localStorage:{getItem(){throw Error('Must not depend on port-scoped storage');}}};context.window=context;
  vm.runInNewContext(source,context);assert.equal(context.speakerdeskTheme.get(),'system');assert.equal(context.document.documentElement.dataset.theme,'dark');
});
