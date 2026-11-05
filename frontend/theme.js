/* Load synchronously in head, before stylesheets, to initialize theme early. */
(() => {
  const root = document.documentElement;
  const preference = window.matchMedia('(prefers-color-scheme: dark)');
  let explicit = null;
  try { explicit = localStorage.getItem('bough-theme'); } catch {}
  if (explicit !== 'light' && explicit !== 'dark') explicit = null;
  const systemTheme = () => preference.matches ? 'dark' : 'light';
  root.dataset.theme = explicit || systemTheme();
  let button;
  function updateButton() {
    if (!button) return;
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    button.textContent = next === 'dark' ? 'Dark mode' : 'Light mode';
    button.setAttribute('aria-label', `Switch to ${next} mode`);
  }
  function mount() {
    button = document.getElementById('theme-toggle');
    if (!button) return;
    button.addEventListener('click', () => {
      explicit = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = explicit;
      try { localStorage.setItem('bough-theme', explicit); } catch {}
      updateButton();
    });
    updateButton();
  }
  preference.addEventListener('change', () => {
    if (!explicit) { root.dataset.theme = systemTheme(); updateButton(); }
  });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount, { once: true });
  } else mount();
})();
