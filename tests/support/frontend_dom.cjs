// CPU DOM contract fixture, not a layout engine or browser timing substitute.
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
function frontend(root) {
  const trace=[],ids=new Map(),stats={elements:0,listeners:0};let document;
  const dash=key=>key.replace(/[A-Z]/g,c=>'-'+c.toLowerCase());
  class Element {
    constructor(tag='div') {
      stats.elements++;
      this.tagName=tag.toUpperCase();this.children=[];this.parentElement=null;this.attributes={};this._text='';this.value='';this.events=new Map();this.textAssignments=0;
      this.dataset=new Proxy({}, {set:(obj,key,value)=>{obj[key]=String(value);this.attributes['data-'+dash(key)]=String(value);return true;}});
      this.style=new Proxy({}, {set:(obj,key,value)=>{obj[key]=value;if(key==='height')trace.push({kind:value==='auto'?'auto':'height',element:this,value});return true;}});
      this.classList={contains:c=>this.className.split(/\s+/).includes(c),add:(...cs)=>{this.className=[...new Set([...this.className.split(/\s+/).filter(Boolean),...cs])].join(' ');},remove:(...cs)=>{this.className=this.className.split(/\s+/).filter(c=>!cs.includes(c)).join(' ');},toggle:(c,force)=>{const on=force??!this.classList.contains(c);on?this.classList.add(c):this.classList.remove(c);return on;}};
      this.scrollTop=0;this.scrollHeightValue=80;this.currentTime=0;this.hidden=false;this.selectionStart=this.selectionEnd=0;
    }
    get className(){return this.attributes.class||'';}set className(v){this.attributes.class=String(v);}
    get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
    set textContent(v){this.textAssignments++;this.replaceChildren();this._text=String(v);}
    setAttribute(k,v){this.attributes[k]=String(v);if(k==='id'){this.id=String(v);ids.set(this.id,this);}}
    getAttribute(k){return this.attributes[k]??null;}
    removeAttribute(k){delete this.attributes[k];}
    append(...nodes){for(let n of nodes){if(typeof n==='string'){const text=new Element('span');text.textContent=n;n=text;}n.remove();n.parentElement=this;this.children.push(n);}}
    replaceChildren(...nodes){for(const child of this.children)child.parentElement=null;this.children=[];this._text='';this.append(...nodes);}
    insertBefore(n,before){if(n===before)return;n.remove();n.parentElement=this;const i=this.children.indexOf(before);this.children.splice(i<0?this.children.length:i,0,n);}
    replaceWith(n){const p=this.parentElement;if(!p)return;const i=p.children.indexOf(this);n.remove();p.children[i]=n;n.parentElement=p;this.parentElement=null;}
    remove(){if(this.parentElement){const p=this.parentElement;p.children.splice(p.children.indexOf(this),1);this.parentElement=null;}}
    get firstChild(){return this.children[0]||null;}
    get nextSibling(){if(!this.parentElement)return null;return this.parentElement.children[this.parentElement.children.indexOf(this)+1]||null;}
    get isConnected(){return document.body.contains(this);}
    contains(n){return this===n||this.children.some(c=>c.contains(n));}
    matches(selector){
      const not=[...selector.matchAll(/:not\(([^)]+)\)/g)].map(x=>x[1]);if(not.some(x=>this.matches(x)))return false;
      selector=selector.replace(/:not\([^)]+\)/g,'');
      const attrs=[...selector.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)];
      for(const [,k,v] of attrs){if(this.getAttribute(k)===null || (v!==undefined && this.getAttribute(k)!==v))return false;}
      selector=selector.replace(/\[[^\]]+\]/g,'');
      const id=selector.match(/#([\w-]+)/)?.[1];if(id && this.id!==id)return false;
      if([...selector.matchAll(/\.([\w-]+)/g)].some(x=>!this.classList.contains(x[1])))return false;
      const tag=selector.match(/^[\w-]+/)?.[0];return !tag||this.tagName===tag.toUpperCase();
    }
    querySelectorAll(selector){
      const result=[];
      for(const part of selector.split(',')){
        const query=part.trim();
        if(query.startsWith(':scope > ')){for(const child of this.children)if(child.matches(query.slice(9)))result.push(child);continue;}
        const tokens=query.split(/\s*>\s*|\s+/);const direct=query.includes('>');
        const walk=host=>{for(const child of host.children){if(child.matches(tokens.at(-1))){
          let parent=child.parentElement,valid=true;
          for(let i=tokens.length-2;i>=0;i--){while(parent && !parent.matches(tokens[i]) && !direct)parent=parent.parentElement;if(!parent || !parent.matches(tokens[i])){valid=false;break;}parent=parent.parentElement;}
          if(valid)result.push(child);
        }walk(child);}};walk(this);
      }return [...new Set(result)];
    }
    querySelector(s){return this.querySelectorAll(s)[0]||null;}
    closest(s){let e=this;while(e && !e.matches(s))e=e.parentElement;return e;}
    addEventListener(type,fn){stats.listeners++;const handlers=this.events.get(type)||[];handlers.push(fn);this.events.set(type,handlers);}
    dispatchEvent(event){event.target=this;this.lastEventResults=(this.events.get(event.type)||[]).map(fn=>fn(event));return true;}
    click(){this.dispatchEvent({type:'click',preventDefault(){}});}
    focus(){const previous=document.activeElement;document.activeElement=this;if(previous!==this){previous?.dispatchEvent({type:'focusout'});this.dispatchEvent({type:'focus'});}}
    blur(){if(document.activeElement===this){document.activeElement=document.body;this.dispatchEvent({type:'focusout'});}}
    setSelectionRange(a,b){this.selectionStart=a;this.selectionEnd=b;}
    get clientWidth(){trace.push({kind:'width',element:this});return 240;}
    get clientHeight(){return 400;}
    get scrollHeight(){trace.push({kind:'scroll',element:this});return this.scrollHeightValue;}
    getBoundingClientRect(){return {top:0,bottom:80,left:0,right:240,width:240,height:80};}
    pause(){}play(){return Promise.resolve();}
  }
  document={body:new Element('body'),activeElement:null,createElement:tag=>new Element(tag),
    getElementById:id=>{if(!ids.has(id)){const el=new Element();el.setAttribute('id',id);document.body.append(el);}return ids.get(id);},
    querySelector:s=>s.startsWith('meta')?{content:'cpu-fixture-token'}:document.body.querySelector(s),querySelectorAll:s=>document.body.querySelectorAll(s),addEventListener(){}};
  document.activeElement=document.body;
  document.body.append(new Element('main'));
  const editor=document.getElementById('editor'),pane=document.getElementById('transcript-pane'),host=document.getElementById('segments');editor.append(pane);pane.append(host);document.getElementById('workspace').append(editor);
  for(const c of ['sidebar','player-panel','export-menu','transcript-search']){const el=new Element();el.className=c;document.body.append(el);}
  const notices=new Element('a');notices.setAttribute('href','/api/notices');document.body.append(notices);
  const context=vm.createContext({document,structuredClone,console,queueMicrotask,URL,URLSearchParams,
    matchMedia:()=>({matches:false,addEventListener(){}}),ResizeObserver:class{constructor(callback){this.callback=callback;}observe(){}},
    getComputedStyle:()=>({paddingRight:'0px'}),setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,
    localStorage:{getItem:()=>null,setItem(){}},crypto:{randomUUID:()=> 'cpu-fixture'},CSS:{escape:s=>s},
    Event:class{constructor(type){this.type=type;}preventDefault(){}},FormData:class{},fetch:async()=>{throw Error('No HTTP in CPU fixture');},
    identityBusy:false,voiceAvailable:false,wirePeople(){},refreshIdentitySuggestions(){},updateNameControls(){},
    lucide:{createIcons(){}},speakerdeskTheme:{set(){},get:()=> 'light'},confirm:()=>true});
  context.window=context;context.addEventListener=()=>{};
  for(const file of ['live_editor.js','app.js']){
    let source=fs.readFileSync(path.join(root,'speakerdesk/static',file),'utf8');if(file==='app.js')source=source.replace(/\ninit\(\);\s*$/,'\n');
    vm.runInContext(source,context,{filename:file});
  }
  const run=expression=>vm.runInContext(expression,context);
  function seed(count=3,status='ready'){
    context.fixture={id:'d'.repeat(32),name:'Synthetic meeting',created:1,status,language:'en',duration:count*10,revision:1,refinement_status:'complete',
      document:{schema_version:1,speakers:{s0:'Maya',s1:'Albert'},segments:Array.from({length:count},(_,n)=>({id:'r'+n,start:n*10,end:n*10+10,speaker:'s'+(Math.floor(n/3)%2),text:'Words '+n,language:'en',machine_revision:1,refinement_state:'refined'}))}};
    run("config={languages:{auto:'Automatic',en:'English',fr:'French'},readiness:{configured:true,automatic_language:true}};selected=structuredClone(fixture);doc=structuredClone(selected.document);dirty=false;$('search').value='';followingLive=false;");
  }
  return {run,seed,document,trace,stats,context,Element};
}
module.exports={frontend};
