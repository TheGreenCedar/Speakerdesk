'use strict';
(() => {
  const media = matchMedia('(prefers-color-scheme: dark)');
  const stored = () => { try { return localStorage.getItem('speakerdesk.theme') || 'system'; } catch { return 'system'; } };
  const apply = value => {
    const mode = ['light','dark','system'].includes(value) ? value : 'system';
    document.documentElement.dataset.theme = mode === 'system' ? (media.matches ? 'dark' : 'light') : mode;
    document.documentElement.style.colorScheme = document.documentElement.dataset.theme;
    return mode;
  };
  window.speakerdeskTheme = {get:stored, set(value) {const mode=apply(value);try{localStorage.setItem('speakerdesk.theme',mode);}catch{}return mode;}};
  apply(stored());
  media.addEventListener('change',()=>apply(stored()));
  window.addEventListener('storage',event=>{if(event.key==='speakerdesk.theme')apply(stored());});
})();
