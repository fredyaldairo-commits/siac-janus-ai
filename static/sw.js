/* JNUS AI · Service Worker
   App-shell cache + network-first para la API (mantiene sincronía con el backend). */
const CACHE = 'janus-ai-v13';
const SHELL = [
  '/app',
  '/static/manifest.webmanifest',
  '/static/logo.png',
  '/static/hero.png',
  '/static/consumo.png',
  '/static/microcredito.png',
  '/static/inmobiliario.png',
  '/static/vendor/three.min.js',
  '/static/vendor/gsap.min.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // API → siempre red (datos frescos del backend); fallback a error JSON offline
  if (url.pathname.startsWith('/api/')) {
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(JSON.stringify({ error: 'Sin conexión. Conéctate para usar la IA en vivo.' }),
          { status: 503, headers: { 'Content-Type': 'application/json' } })
      )
    );
    return;
  }
  // HTML / navegación → network-first: siempre la última UI cuando hay red,
  // cache como respaldo offline. Evita que se quede pegada una versión vieja.
  const isHTML = e.request.mode === 'navigate' ||
    (e.request.headers.get('accept') || '').includes('text/html');
  if (isHTML) {
    e.respondWith(
      fetch(e.request).then((res) => {
        if (res && res.status === 200) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      }).catch(() => caches.match(e.request).then((c) => c || caches.match('/app')))
    );
    return;
  }
  // Estáticos (imágenes, vendor JS) → cache-first con actualización en segundo plano
  e.respondWith(
    caches.match(e.request).then((cached) => {
      const network = fetch(e.request).then((res) => {
        if (res && res.status === 200 && e.request.method === 'GET') {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      }).catch(() => cached);
      return cached || network;
    })
  );
});
