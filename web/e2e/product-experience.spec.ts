import { expect, test, type Download } from "@playwright/test";
import { activeLicense, clearedLicense, authenticatedSession, mockStudioApi, openUtility } from "./fixtures/studio";

async function downloadedJson(download: Download) {
  const stream = await download.createReadStream();
  if (!stream) throw new Error("Download stream unavailable");
  const decoder = new TextDecoder();
  let text = "";
  for await (const chunk of stream) text += decoder.decode(chunk, { stream: true });
  return JSON.parse(text + decoder.decode());
}

// Match AppConfig's numeric metadata, object groups and nullable device groups.
const savedConfig = {
  schema_version: 20, revision: 12, server: { port: 7351 }, replay: {},
  pipeline: { p_response_scale: 0.7 }, control: { output_enabled: false },
  paths: {}, consumers: {}, limits: {}, capture: null, inference: null, hardware: null,
};

test("a downloaded AppConfig restores through preview, persisted write and readback", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, savedConfig);
  let canonical = structuredClone(savedConfig);
  const writes: unknown[] = [];
  await page.route("**/api/config", async (route) => {
    if (route.request().method() === "POST") {
      const value = route.request().postDataJSON();
      writes.push(value);
      canonical = { ...value, revision: canonical.revision + 1 };
      return route.fulfill({ json: { config: canonical, apply_mode: "hot_update", restart_required: false, applied: true, rolled_back: false, message: "saved" } });
    }
    return route.fulfill({ json: canonical });
  });
  await page.goto("/?page=overview");
  await openUtility(page, "设置");
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "下载备份" }).click();
  const download = await downloadEvent;
  expect(await downloadedJson(download)).toEqual(savedConfig);
  const backupPath = (await download.path())!;
  canonical.pipeline.p_response_scale = 0.2;
  canonical.revision += 1;
  await page.reload();
  await page.getByRole("navigation", { name: "NovaSight Studio 导航" }).waitFor();
  await openUtility(page, "设置");
  await page.locator('input[type="file"]').setInputFiles(backupPath);
  const dialog = page.getByRole("alertdialog", { name: new RegExp("^应用 .+？$") });
  await expect(dialog).toContainText("0.2");
  await expect(dialog).toContainText("0.7");
  expect(writes).toHaveLength(0);
  await dialog.getByRole("button", { name: "确认导入配置" }).click();
  await expect(page.locator(".toast-card").filter({ hasText: "配置导入成功" })).toBeVisible();
  expect(writes).toHaveLength(1);
  expect(await page.evaluate(async () => (await fetch("/api/config")).json())).toEqual({ ...savedConfig, revision: 14 });
  expect(canonical.control.output_enabled).toBe(false);
});

test("rejected backup produces no write and remains readable and exportable after reload", async ({ page }, testInfo) => {
  await mockStudioApi(page, authenticatedSession, savedConfig);
  const mutations: string[] = [];
  page.on("request", (request) => { if (request.method() === "POST") mutations.push(request.url()); });
  await page.goto("/?page=overview");
  await openUtility(page, "设置");
  await page.locator('input[type="file"]').setInputFiles(`${testInfo.config.rootDir}/fixtures/invalid-backup.json`);
  await expect(page.locator(".toast-card").filter({ hasText: "配置导入失败" })).toBeVisible();
  expect(mutations).toEqual([]);
  await page.goto("/?page=activity");
  const feed = page.getByRole("region", { name: "实时日志列表" });
  await expect(feed).toContainText("配置导入失败");
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出日志" }).click();
  const logs = await downloadedJson(await downloadEvent);
  expect(logs.version).toBe("1.10.535");
  expect(logs.records.some((record: { title: string }) => record.title === "配置导入失败")).toBe(true);
});

test("valid refresh shares one startup surface and never exposes the authorization form", async ({ page }) => {
  await mockStudioApi(page);
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/auth/session", async (route) => { await waiting; return route.fulfill({ json: authenticatedSession }); });
  await page.addInitScript(() => {
    (window as Window & { exposedAuthorization?: boolean }).exposedAuthorization = false;
    new MutationObserver(() => {
      if (document.querySelector("#auth-code")) (window as Window & { exposedAuthorization?: boolean }).exposedAuthorization = true;
    }).observe(document, { childList: true, subtree: true });
  });
  try {
    await page.goto("/?page=fire", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("status", { name: "正在打开 NovaSight" })).toBeVisible();
    await expect(page.locator(".route-loading-shell, .license-state-card")).toHaveCount(0);
    release();
    await expect(page.getByRole("heading", { name: "开火稳定", level: 1 })).toBeVisible();
    expect(await page.evaluate(() => (window as Window & { exposedAuthorization?: boolean }).exposedAuthorization)).toBe(false);
  } finally { release(); }
});


test("a late expiry check cannot replace a newly confirmed machine authorization", async ({ page }) => {
  await mockStudioApi(page);
  let reads = 0;
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/license", async (route) => {
    reads += 1;
    if (reads === 1) return route.fulfill({ json: { ...activeLicense, expires_at: Math.floor(Date.now() / 1000) - 1 } });
    if (reads === 2) { await waiting; return route.fulfill({ json: clearedLicense }); }
    return route.fulfill({ json: { ...activeLicense, device_model: "newly-confirmed-device", expires_at: Math.floor(Date.now() / 1000) + 86400 } });
  });
  try {
    await page.goto("/?page=fire");
    await expect.poll(() => reads).toBe(2);
    await openUtility(page, "设置");
    await page.getByText("更换授权码", { exact: true }).click();
    await page.getByRole("textbox", { name: "授权码", exact: true }).fill("replacement-code");
    await page.getByRole("button", { name: "验证并更换", exact: true }).click();
    await expect(page.locator(".toast-card").filter({ hasText: "设备授权已更新" })).toBeVisible();
    release();
    await page.getByRole("button", { name: "关闭设置" }).click();
    await openUtility(page, "关于");
    await expect(page.getByLabel("设备授权状态")).toContainText("newly-confirmed-device");
    await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "开火稳定", level: 1 })).toBeVisible();
  } finally { release(); }
});
