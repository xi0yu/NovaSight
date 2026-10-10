import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const now = Math.floor(Date.now() / 1000);
const active = { authenticated: true, auth_state: "authorized", principal: "operator", role: "operator", permissions: [], csrf_token: "csrf", expires_at: now + 3600, session_lifetime_seconds: 3600, server_time: now, license_expires_at: now + 86400 };
const checking = { ...active, authenticated: false, auth_state: "checking", principal: null, role: null, csrf_token: null, expires_at: null };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

beforeEach(() => {
  vi.resetModules();
  window.localStorage.removeItem("novasight.signed-out");
  vi.stubGlobal("BroadcastChannel", class { onmessage = null; postMessage() {} });
});
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); window.localStorage.removeItem("novasight.signed-out"); });

describe("auth-session-contract A06/A07", () => {
  it("coalesces simultaneous 401 responses, including late responses, and retries each read once", async () => {
    let release!: () => void;
    const wait = new Promise<void>((resolve) => { release = resolve; });
    const attempts = new Map<string, number>();
    let refreshes = 0;
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url === "/api/auth/session") return json(checking);
      if (url === "/api/auth/refresh") { refreshes++; await wait; return json(active); }
      const n = (attempts.get(url) ?? 0) + 1; attempts.set(url, n);
      return n <= 20 ? json({ code: "AUTHENTICATION_REQUIRED" }, 401) : json([]);
    }));
    const api = await import("../api");
    const calls = Array.from({ length: 20 }, () => api.getModelProjects());
    await vi.waitFor(() => expect(refreshes).toBe(1));
    release();
    await Promise.all(calls);
    expect(refreshes).toBe(1);
    expect([...attempts.values()]).toEqual([40]);
  });

  it("does not recursively refresh or replay a write whose response timed out", async () => {
    let writes = 0;
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/auth/session" && init?.method !== "POST") return json(checking);
      if (url === "/api/auth/refresh") return json({ code: "AUTHENTICATION_REQUIRED" }, 401);
      writes++; throw new TypeError("network disconnected after sending body");
    }));
    const api = await import("../api");
    await expect(api.recoverAuthSession()).rejects.toThrow();
    await expect(api.saveLicenseKey("valid-code")).rejects.toThrow();
    expect(writes).toBe(1);
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/auth/refresh")).toHaveLength(1);
  });

  it("retains identity on transport failure and accepts only a confirmed terminal response", async () => {
    const sessions: unknown[] = [];
    const onSession = (event: Event) => sessions.push((event as CustomEvent).detail);
    window.addEventListener("novasight:auth-session", onSession);
    try {
      vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("offline"); }));
      const api = await import("../api");
      await expect(api.recoverAuthSession()).rejects.toThrow();
      expect(sessions).toEqual([]);
      vi.stubGlobal("fetch", vi.fn(async (url: string) => json(url === "/api/auth/session" ? checking : { ...checking, auth_state: "expired" })));
      expect((await api.recoverAuthSession()).auth_state).toBe("expired");
      expect(sessions).toHaveLength(1);
    } finally { window.removeEventListener("novasight:auth-session", onSession); }
  });
  it("does not replay a write after the actual request deadline fires", async () => {
    vi.useFakeTimers();
    let writes = 0;
    vi.stubGlobal("fetch", vi.fn((_url: string, init?: RequestInit) => {
      writes++;
      return new Promise<Response>((_resolve, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("timeout", "AbortError")), { once: true }));
    }));
    const api = await import("../api");
    const result = api.saveLicenseKey("valid-code").catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(30_000);
    const error = await result;
    expect(error).toBeInstanceOf(api.ApiError);
    expect((error as InstanceType<typeof api.ApiError>).status).toBe(408);
    expect(writes).toBe(1);
  });

  it.each(["recovery", "authorization"] as const)("ignores late successful %s after explicit user logout", async (operation) => {
    let finish!: (response: Response) => void;
    vi.stubGlobal("fetch", vi.fn((_url: string, init?: RequestInit) => {
      if (init?.method === "DELETE") return Promise.resolve(json({ ...checking, auth_state: "signed_out" }));
      return new Promise<Response>((resolve) => { finish = resolve; });
    }));
    const api = await import("../api");
    const pending = (operation === "recovery" ? api.recoverAuthSession() : api.establishAuthSession("valid-code")).catch((error: unknown) => error);
    await api.logoutAuthSession();
    finish(json(active));
    const error = await pending;
    expect(error).toBeInstanceOf(DOMException);
    expect((error as DOMException).name).toBe("AbortError");
    expect((await api.recoverAuthSession()).auth_state).toBe("signed_out");
  });

  it("a new authorization in another tab supersedes an in-flight old identity result", async () => {
    let channel!: { onmessage: ((event: MessageEvent) => void) | null };
    vi.stubGlobal("BroadcastChannel", class {
      onmessage: ((event: MessageEvent) => void) | null = null;
      constructor() { channel = this; }
      postMessage() {}
    });
    let finish!: (response: Response) => void;
    let reads = 0;
    vi.stubGlobal("fetch", vi.fn(() => ++reads === 1
      ? new Promise<Response>((resolve) => { finish = resolve; })
      : Promise.resolve(json(active))));
    const api = await import("../api");
    const pending = api.recoverAuthSession().catch((error: unknown) => error);
    channel.onmessage!(new MessageEvent("message", { data: { kind: "signed_in" } }));
    finish(json({ ...checking, auth_state: "revoked" }));
    expect((await pending as DOMException).name).toBe("AbortError");
    await vi.waitFor(() => expect(reads).toBe(2));
    expect((await api.recoverAuthSession()).authenticated).toBe(true);
  });

});
