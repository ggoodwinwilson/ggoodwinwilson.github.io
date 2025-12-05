document.addEventListener('DOMContentLoaded', async () => {
  const target = document.querySelector('#site-sidebar');
  if (!target) return;
  try {
    const res = await fetch('/partials/sidebar.html?v=2', { cache: 'no-store' });
    if (!res.ok) throw new Error(`Sidebar fetch failed: ${res.status}`);
    target.innerHTML = await res.text();
  } catch (err) {
    console.error('Sidebar load error', err);
  }
});
