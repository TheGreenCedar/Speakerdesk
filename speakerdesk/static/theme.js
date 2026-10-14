'use strict';
(() => {
  const media = matchMedia('(prefers-color-scheme: dark)');
  let preference=document.documentElement.dataset.hostTheme;
  if(!['light','dark','system'].includes(preference))preference='system';
  const stored = () => preference;
  const apply = value => {
    const mode = ['light','dark','system'].includes(value) ? value : 'system';
    document.documentElement.dataset.theme = mode === 'system' ? (media.matches ? 'dark' : 'light') : mode;
    document.documentElement.style.colorScheme = document.documentElement.dataset.theme;
    return mode;
  };
  window.speakerdeskTheme = {get:stored, set(value) {preference=apply(value);return preference;}};
  apply(stored());
  media.addEventListener('change',()=>apply(stored()));
})();
