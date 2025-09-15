// sw.js — FIR Filter Designer (Web)
// Cache name: bump this on every deploy to invalidate old caches.
const CACHE_NAME = 'fir-app-v10';

// List the core files your app needs to run offline.
// Keep the app.py query (?v=10) in sync with index.html.
const ASSETS = [
  './',
  './index.html',
  './app.py?v=10',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/favicon-32.png',
  './manifest.webmanifest'
];

// ----- Install: precache the app shell -----
self.addEventListener('install', (evt) => {
  evt.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

// ----- Activate: clean old caches & take control -----
self.addEventListener('activate', (evt) => {
  evt.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// ----- Fetch strategy -----
// - app.py => network-first (then cache)
// - HTML navigations => network, else offline fallback to cached index.html
// - everything else => cache-first (then network)
self.addEventListener('fetch', (evt) => {
  const req = evt.request;
  const url = new URL(req.url);

  // Only handle same-origin requests
  if (url.origin !== location.origin) return;

  // Non-GET: let it pass through
  if (req.method !== 'GET') return;

  // 1) HTML navigations: try network, fall back to cached shell
  if (req.mode === 'navigate') {
    evt.respondWith(
      fetch(req).catch(() => caches.match('./index.html'))
    );
    return;
  }

  // 2) Python app: network-first so code updates are immediate
  if (url.pathname.endsWith('/app.py') || url.pathname.includes('/app.py?')) {
    evt.respondWith(
      fetch(req)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE_NAME).then((c) => c.put(req, copy));
          return resp;
        })
        .catch(() => caches.match(req))
    );
    return;
  }

  // 3) Everything else (icons, manifest, CSS you host, etc.): cache-first
  evt.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE_NAME).then((c) => c.put(req, copy));
        return resp;
      });
    })
  );
});
