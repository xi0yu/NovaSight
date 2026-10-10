import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import stoppedRuntime from "./fixtures/stopped-runtime.json" with { type: "json" };
import { activeLicense, authenticatedSession, unauthenticatedSession } from "./fixtures/studio";

async function broker(context: BrowserContext) {
  const state = { access: true, online: true, expired: false, refreshes: 0, writes: 0, timeOutWrites: false };
  const config = { revision: 1, control: { output_enabled: false }, hardware: { backend: "null" }, pipeline: { p_response_scale: 0.2 } };
  await context.routeWebSocket("**/ws/**", () => undefined);
  await context.route("**/healthz", (route) => state.online ? route.fulfill({ json: { ok: true } }) : route.abort("failed"));
  await context.route("**/api/**", async (route) => {
    if (!state.online) return route.abort("failed");
    const path = new URL(route.request().url()).pathname;
    const now = Math.floor(Date.now() / 1000);
    const session = () => ({ ...authenticatedSession, auth_state: "authorized", server_time: now, expires_at: now + 3600, license_expires_at: now + 86400 });
    if (path === "/api/auth/session") return route.fulfill({ json: state.access ? session() : { ...unauthenticatedSession, auth_state: "checking", server_time: now } });
    if (path === "/api/auth/refresh") {
      state.refreshes++;
      if (state.expired) return route.fulfill({ json: { ...unauthenticatedSession, auth_state: "expired", server_time: now } });
      state.access = true;
      return route.fulfill({ json: session() });
    }
    if (!state.access) return route.fulfill({ status: 401, json: { code: "AUTHENTICATION_REQUIRED" } });
    if (path === "/api/license") return route.fulfill({ json: { ...activeLicense, device_model: "Browser contract fixture", valid: !state.expired, expires_at: now + 86400 } });
    if (path === "/api/config") return route.fulfill({ json: config });
    if (path === "/api/studio/v1/status") return route.fulfill({ json: { ...stoppedRuntime, semantic: { ...stoppedRuntime.semantic, snapshot_updated_at_ms: Date.now() } } });
    if (path === "/api/v1/config/parameters") { state.writes++; return state.timeOutWrites ? route.abort("timedout") : route.fulfill({ status: 409, json: { code: "TEST_DRAFT_RETAINED", message: "测试保留草稿" } }); }
    if (path === "/api/models/projects") return route.fulfill({ json: [] });
    return route.fulfill({ status: 503, json: { code: "TEST_UNUSED_ENDPOINT", message: "未使用接口" } });
  });
  return state;
}
async function openDraft(page: Page) {
  await page.goto("/?page=params");
  await page.getByRole("tab", { name: "移动与输出" }).click();
  await expect(page.locator(".auth-recovery-notice").filter({ hasText: "正在核对设备状态" })).toHaveCount(0);
  const input = page.getByRole("textbox", { name: "比例增益 Kp 数值" });
  await input.fill("0.7");
  return input;
}

for (const elapsed of [3_700_000, 7_200_000]) {
  test(`S1/S2 ${elapsed / 1000}s elapsed keeps the page and never exposes short-session expiry`, async ({ context, page }) => {
    const state = await broker(context);
    await page.clock.install();
    const input = await openDraft(page);
    state.access = false;
    await page.clock.fastForward(elapsed);
    await page.evaluate(() => window.dispatchEvent(new Event("focus")));
    await expect.poll(() => state.refreshes).toBeGreaterThan(0);
    await expect(input).toHaveValue(/^0\.7(?:00)?$/);
    await expect(page.getByRole("heading", { name: "输入授权码" })).toHaveCount(0);
    await expect(page.locator("body")).not.toContainText("会话过期");
    expect(page.url()).toContain("page=params");
  });
}
for (const scenario of ["network", "restart"] as const) {
  test(`S3/S4 ${scenario} restores identity separately and never starts output`, async ({ context, page }) => {
    const state = await broker(context);
    const starts: string[] = [];
    page.on("request", (request) => { if (request.method() === "POST" && /\/start$|\/connect$/.test(new URL(request.url()).pathname)) starts.push(request.url()); });
    const input = await openDraft(page);
    state.online = false;
    await page.evaluate(() => window.dispatchEvent(new Event("novasight:auth-required")));
    await expect(page.getByText("正在恢复连接，页面内容已保留")).toBeVisible();
    await expect(input).toBeVisible();
    await expect(page.getByRole("heading", { name: "输入授权码" })).toHaveCount(0);
    state.online = true; state.access = false;
    await page.evaluate(() => window.dispatchEvent(new Event("online")));
    await expect.poll(() => state.refreshes).toBe(1);
    await expect(page.locator(".authenticated-workspace")).toHaveAttribute("data-auth-state", "authorized");
    await expect(input).toHaveValue(/^0\.7(?:00)?$/);
    expect(starts).toEqual([]);
  });
}

test("S5 only a server-confirmed expiration opens authorization", async ({ context, page }) => {
  const state = await broker(context);
  await openDraft(page);
  state.expired = true; state.access = false;
  await page.evaluate(() => window.dispatchEvent(new Event("novasight:auth-required")));
  await expect(page.getByRole("heading", { name: "输入授权码" })).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("授权已到期");
});

test("S7 two tabs share recovery and neither is sent to authorization", async ({ context, page }) => {
  const state = await broker(context);
  state.access = false;
  const other = await context.newPage();
  await Promise.all([page.goto("/?page=params"), other.goto("/?page=fire")]);
  for (const tab of [page, other]) {
    await expect(tab.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
    await expect(tab.getByRole("heading", { name: "输入授权码" })).toHaveCount(0);
  }
  expect(state.refreshes).toBe(1);
  await other.close();
});

test("S8 a timed-out write is not replayed after identity and business recover", async ({ context, page }) => {
  const state = await broker(context); state.timeOutWrites = true;
  const input = await openDraft(page);
  await input.press("Tab");
  await expect.poll(() => state.writes).toBe(1);
  await expect(page.getByText("自动保存失败，修改已保留", { exact: true })).toBeVisible();
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await expect(page.locator(".authenticated-workspace")).toHaveAttribute("data-auth-state", "authorized");
  await page.waitForTimeout(700);
  expect(state.writes).toBe(1);
  await expect(input).toHaveValue(/^0\.7(?:00)?$/);
});

test("P1 new versions never reload automatically and protect unfinished input", async ({ context, page, baseURL }) => {
  test.skip(baseURL?.endsWith(":7351") ?? false, "deployment version checks run in the static-build project");
  const state = await broker(context);
  let changed = false;
  await page.route("**/version.json", async (route) => changed ? route.fulfill({ json: { version: "new-version" } }) : route.continue());
  const input = await openDraft(page);
  const original = page.url();
  changed = true;
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  const update = page.getByRole("button", { name: "更新页面" });
  // Dev runs intentionally have no deployment version polling.
  await expect(update).toBeVisible();
  await update.click();
  await expect(page.getByRole("alertdialog", { name: "还有未保存内容" })).toBeVisible();
  await page.getByRole("button", { name: "返回保存" }).click();
  await expect(input).toHaveValue(/^0\.7(?:00)?$/);
  expect(page.url()).toBe(original);
  expect(state.refreshes).toBe(0);
});
