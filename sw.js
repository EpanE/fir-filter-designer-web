// sw.js — simple offline-first service worker
const CACHE = 'firfd-v1';
const ASSETS = [
  './',
  './index.html',
  './app.py',
  './manifest.webmanifest',
  // icons (add these if you include them)
  './icons/icon-192.png',
  './icons/icon-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Cache-first for same-origin assets; runtime cache for others (PyScript/pyodide)
// First-ever run still needs network to fetch big Pyodide files. After that, offline works.
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;

  const url = new URL(e.request.url);
  const sameOrigin = url.origin === location.origin;

  if (sameOrigin) {
    e.respondWith(
      caches.match(e.request).then(hit =>
        hit || fetch(e.request).then(resp => {
          const copy = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, copy));
          return resp;
        })
      )
    );
  } else {
    // runtime caching for cross-origin (pyscript/pyodide)
    e.respondWith(
      fetch(e.request)
        .then(resp => {
          const copy = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, copy)).catch(()=>{});
          return resp;
        })
        .catch(() => caches.match(e.request))
    );
  }
});
