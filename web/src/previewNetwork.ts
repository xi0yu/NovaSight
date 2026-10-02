// Imported only by preview.html. No device reads, writes or event streams may leave this page.
const assetFetch = window.fetch.bind(window);
window.fetch = async (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input), location.href);
  const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  const isApi = url.pathname.startsWith("/api/") || url.pathname === "/healthz";
  if (!isApi && url.origin === location.origin && method === "GET") return assetFetch(input, init);
  const reads: Record<string, unknown> = {
    "/healthz": { ok: true },
    "/api/activity": { events: [] },
    "/api/auth/session": { authenticated: false, principal: null, role: null, permissions: [], csrf_token: null, expires_at: null, session_lifetime_seconds: 3600 },
    "/api/models/catalog": { root: { type: "directory", name: "models", relative_path: "", children: [] }, directory_count: 1, model_count: 0, discovered_files: 0, updated_files: 0, cache_hits: 0, force: false },
  };
  const body = method === "GET" ? reads[url.pathname] : undefined;
  return new Response(JSON.stringify(body ?? { code: "PREVIEW_ONLY", message: "这是界面预览，未连接设备；此操作不会写入配置或控制硬件。" }), {
    status: body === undefined ? 503 : 200,
    headers: { "content-type": "application/json" },
  });
};

// Keep Vite's own development channel; all product event channels remain disconnected.
window.WebSocket = new Proxy(window.WebSocket, {
  construct(target, args) {
    if (args[1] === "vite-hmr") return Reflect.construct(target, args);
    return { readyState: WebSocket.CLOSED, close() {}, addEventListener() {}, removeEventListener() {} };
  },
});
