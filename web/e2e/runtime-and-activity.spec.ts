import { expect, test } from "@playwright/test";
import { mockStudioApi, openUtility, authenticatedSession, unauthenticatedSession, activeLicense } from "./fixtures/studio";

test("activity source filters and pause controls keep diagnostic details readable", { tag: "@mobile" }, async ({ page }) => {
  await page.goto("/preview.html?page=activity&showcase=activity");
  await page.getByRole("combobox", { name: "日志来源" }).selectOption("模型");
  const list = page.getByRole("region", { name: "实时日志列表" });
  await expect(list.getByText("模型读取失败", { exact: true })).toBeVisible();
  await expect(list.getByText("画面规格已保存", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "暂停查看", exact: true }).click();
  await expect(page.getByRole("button", { name: "恢复实时", exact: true })).toHaveAttribute("aria-pressed", "true");
  await list.getByText("原始错误与开发者详情", { exact: true }).click();
  await expect(list.getByText("ENOENT: models/arena-fp16.engine", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "恢复实时", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "日志来源" })).toHaveValue("模型");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("toast shows its countdown and disappears while the failed input remains", async ({ page }) => {
  await page.goto("/preview.html?page=control-test");
  await page.getByRole("button", { name: "新增协议" }).click();
  const dialog = page.getByRole("dialog", { name: "新增协议" });
  expect(await page.locator(".mouse-protocol-toolbar").evaluate((element) => !!element.closest("[inert]"))).toBe(true);
  expect(await page.locator(".console-emergency-stop").evaluate((element) => !!element.closest("[inert]"))).toBe(false);
  await dialog.getByRole("textbox", { name: "IP", exact: true }).fill("bad-ip");
  await dialog.getByRole("button", { name: "添加协议" }).click();
  const toast = page.locator(".toast-card.tone-error").filter({ hasText: "协议参数无效" });
  await expect(toast.getByText("3 秒后关闭")).toBeVisible();
  const bar = toast.locator(".toast-card-timer > span");
  await expect(bar).toHaveCSS("animation-timing-function", "linear");
  await expect(bar).toHaveCSS("animation-duration", "3s");
  for (let sample = 0; sample < 3; sample += 1) {
    await page.waitForTimeout(300);
    const progress = await bar.evaluate((element) => ({ fraction: element.getBoundingClientRect().width / element.parentElement!.getBoundingClientRect().width, elapsed: Number(element.getAnimations()[0].currentTime) }));
    expect(progress.fraction).toBeCloseTo(1 - progress.elapsed / 3000, 1);
  }
  await expect(toast.getByText("1 秒后关闭")).toBeVisible();
  await expect(toast).toHaveCount(0, { timeout: 8000 });
  await expect(dialog.getByRole("textbox", { name: "IP", exact: true })).toHaveValue("bad-ip");
  await expect(dialog.getByRole("alert")).toContainText("IPv4");
});

test("repeated invalid protocol input stays in one toast and correction clears the field error", async ({ page }) => {
  await page.goto("/preview.html?page=control-test");
  await page.getByRole("button", { name: "新增协议" }).click();
  const dialog = page.getByRole("dialog", { name: "新增协议" });
  const ip = dialog.getByRole("textbox", { name: "IP", exact: true });
  await ip.fill("bad-ip");
  await dialog.getByRole("button", { name: "添加协议" }).click();
  await expect(ip).toBeFocused();
  await expect(ip).toHaveAttribute("aria-invalid", "true");
  await dialog.getByRole("button", { name: "添加协议" }).click();
  const toast = page.locator(".toast-card").filter({ hasText: "协议参数无效" });
  await expect(toast).toHaveCount(1);
  await expect(toast).toContainText("2 次");
  await expect(dialog.getByRole("alert")).toContainText("IPv4");
  await ip.fill("192.168.2.188");
  await expect(dialog.getByRole("alert")).toHaveCount(0);
  await expect(ip).not.toHaveAttribute("aria-invalid", "true");
});

test("desktop log copying offers selected text when the browser clipboard is unavailable", async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined }));
  await page.goto("/preview.html?page=activity&showcase=activity");
  await page.getByRole("button", { name: "复制日志：模型读取失败" }).click();
  const text = page.getByRole("textbox", { name: "手动复制日志：模型读取失败" });
  await expect(text).toBeFocused();
  await expect(text).toHaveValue(/preview-request-001/);
  await expect(text).toHaveValue(/ENOENT/);
  expect(await text.evaluate((element: HTMLTextAreaElement) => element.selectionEnd - element.selectionStart)).toBe((await text.inputValue()).length);
  await expect(page.getByRole("button", { name: "恢复实时", exact: true })).toBeVisible();
});

test("backup download carries time and revision with accurate toast feedback", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 12, pipeline: { p_response_scale: 0.7 } });
  await page.goto("/?page=overview");
  await openUtility(page, "设置");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "下载备份" }).click();
  expect((await download).suggestedFilename()).toMatch(/^novasight-config-.*-r12\.json$/);
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "已发起备份下载" })).toBeVisible();
});

test("restore dialog shows exact values and scope before any write", async ({ page }, testInfo) => {
  const config = { revision: 4, pipeline: { p_response_scale: 0.5 }, server: { port: 7351 }, hardware: { backend: "null", password: "hidden-before" } };
  await mockStudioApi(page, authenticatedSession, config);
  const writes: string[] = [];
  page.on("request", (request) => { if (request.method() === "POST") writes.push(request.url()); });
  await page.goto("/?page=overview");
  await openUtility(page, "设置");
  await page.locator('input[type="file"]').setInputFiles(`${testInfo.config.rootDir}/fixtures/review-backup.json`);
  const dialog = page.getByRole("alertdialog", { name: "应用 review-backup.json？" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("table")).toContainText("pipeline.p_response_scale");
  await expect(dialog.getByRole("table")).toContainText("0.5");
  await expect(dialog.getByRole("table")).toContainText("0.8");
  await expect(dialog.getByRole("table")).toContainText("重启后生效");
  await expect(dialog).not.toContainText("hidden-before");
  await expect(dialog).not.toContainText("hidden-after");
  await page.screenshot({ path: "../out/desktop-restore-diff.png", animations: "disabled" });
  await dialog.getByRole("button", { name: "取消", exact: true }).click();
  expect(writes).toEqual([]);
  await expect(page.getByRole("dialog", { name: "设置", exact: true })).toBeVisible();
});

test("one license code establishes an authenticated and activated Studio session", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.addInitScript(() => localStorage.setItem("novasight.theme", "rose-white"));
  await page.goto("/");

  await expect(page.locator("html")).toHaveAttribute("data-theme", "nova-daylight");
  await expect(page.locator("#auth-code-help")).toHaveText(
    "支持正式授权码或本次启动的临时授权码；不另设接入码。",
  );
  await page.getByLabel("授权码", { exact: true }).fill("one-license-code");
  const login = page.waitForRequest((request) => request.url().endsWith("/api/auth/session") && request.method() === "POST");
  await page.getByRole("button", { name: "进入控制台" }).click();
  expect((await login).postDataJSON()).toEqual({ key: "one-license-code" });

  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
  await expect(page.locator(".theme-toggle")).toHaveCount(0);
  await expect(page.locator("body")).not.toContainText("one-license-code");
});

test("obsolete access links are cleared and cannot bypass license login", async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.goto("/#access=one-time-access-code");

  await expect(page.getByRole("heading", { name: "输入授权码" })).toBeVisible();
  expect(new URL(page.url()).hash).toBe("");
  await expect(page.locator("body")).not.toContainText("one-time-access-code");
});

test("unavailable runtime leads to the error details", async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/studio/v1/status", (route) => route.fulfill({ status: 503, json: { code: "RUNTIME_UNAVAILABLE", message: "运行状态不可达" } }));
  await page.goto("/?page=overview");

  await page.getByRole("button", { name: "查看异常信息", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "异常信息" })).toBeVisible();
});

test("activity page presents backend faults with domain and developer detail", async ({ page }) => {
  await mockStudioApi(page);
  await page.route(/\/api\/activity(?:\?.*)?$/, async (route) => route.fulfill({ json: { events: [{
    daemon_instance_id: "daemon-test",
    id: 1,
    occurred_at_ms: 1_700_000_000_000,
    level: "error",
    tag: "参数",
    title: "参数保存失败",
    message: "整组参数未生效，修改仍保留。",
    technical_detail: "request_id=req-test: validation failed",
    request_id: "req-test",
    count: 1,
  }] } }));
  await page.goto("/?page=activity");

  const event = page.getByRole("listitem").filter({ hasText: "参数保存失败" });
  await expect(event).toContainText("参数");
  await expect(event).toContainText("整组参数未生效，修改仍保留。");
  await event.getByText("原始错误与开发者详情").click();
  await expect(event).toContainText("request_id=req-test: validation failed");
  await page.getByRole("button", { name: "查看完整详情" }).click();
  await expect(page.getByRole("dialog", { name: "异常信息" })).toContainText("参数保存失败");
  await expect(page.getByRole("dialog", { name: "异常信息" })).toContainText("request_id=req-test: validation failed");
});

test("capture page keeps saved and running specifications visible without horizontal overflow", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 25,
    capture: { device: "/dev/video0", pixel_format: "MJPG", width: 1920, height: 1080, fps: 240 },
  });
  await page.goto("/?page=capture");

  const check = page.locator(".capture-command-header");
  await expect(check).toContainText("运行规格未确认");
  await expect(check.getByText("MJPEG (MJPG) / 1920x1080 / 240 FPS", { exact: true })).toBeVisible();
  await expect(page.getByText("点击一个 FPS 即自动验证并保存整组设备配置。", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "应用画面设置" })).toHaveCount(0);
  if ((page.viewportSize()?.width ?? 0) >= 1200) {
    await expect(page.getByRole("heading", { name: "选择设备和画质" })).toBeInViewport();
  }
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(
    await page.evaluate(() => document.documentElement.clientWidth),
  );
});

test("one backend outage does not repeat friendly and channel errors", async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/studio/v1/status", (route) => route.fulfill({ status: 503, json: { code: "RUNTIME_UNAVAILABLE", message: "运行状态不可达" } }));
  await page.goto("/?page=models");
  await page.getByRole("heading", { level: 1, name: "模型" }).waitFor();
  await page.locator(".error-center-trigger").click();

  await expect(page.getByRole("dialog", { name: "异常信息" })).toContainText("运行态失败");
  await expect(page.locator(".error-center-item").filter({ hasText: "runtime 通道异常" })).toHaveCount(0);
  await expect(page.locator(".error-center-item").filter({ hasText: "config 通道异常" })).toHaveCount(0);
  await expect(page.locator(".error-center-item").filter({ hasText: "projects 通道异常" })).toHaveCount(0);
  await expect(page.locator(".error-center-item").filter({ hasText: "当前操作未完成" })).toHaveCount(0);
});

test("unavailable runtime offers concise recovery without a new popup flow", async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/studio/v1/status", (route) => route.fulfill({ status: 503, json: { code: "RUNTIME_UNAVAILABLE", message: "运行状态不可达" } }));
  await page.goto("/?page=control");

  await expect(page.getByRole("heading", { level: 1, name: "目标与控制" })).toBeVisible();
  await expect(page.getByRole("button", { name: "查看异常", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "重试" })).toBeVisible();

  await page.getByRole("button", { name: "查看异常", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "异常信息" })).toBeVisible();
});

test("legacy license links open About with a visible machine model and deadline", async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/license", (route) => route.fulfill({ json: {
    ...activeLicense, device_model: "Mac16,11", tier: "temporary", credential_format: "ephemeral_code",
    expires_at: Date.now() / 1000 + 86400, duration_value: 1, duration_unit: "day",
  } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/?page=license");
  const dialog = page.getByRole("dialog", { name: "关于 NovaSight" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText("Mac16,11")).toBeVisible();
  await expect(dialog.locator(".license-device-expiry")).toContainText("到期时间");
  await expect(dialog.locator(".license-device-expiry")).toContainText("设备模型 ID");
  await expect(dialog.locator(".license-device-expiry")).toContainText("授权类别");
  await expect(dialog.locator(".license-device-expiry").getByText("测试", { exact: true })).toBeVisible();
  await expect(dialog.getByText("临时测试授权", { exact: true })).toHaveCount(0);
  await expect(dialog.locator(".license-device-expiry")).toContainText(/(?:24:00:00|23:59:\d\d)/);
  await expect(page).toHaveURL(/page=overview/);
  await dialog.getByRole("button", { name: "关闭关于" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "首页" })).toBeVisible();
});

test("license expiry keeps the frontend available for reauthorization", async ({ page }) => {
  await mockStudioApi(page);
  const deadline = Date.now() / 1000 + 4;
  await page.route("**/api/license", (route) => route.fulfill({ json: {
    ...activeLicense, device_model: "Expiry test device", expires_at: deadline, valid: Date.now() / 1000 < deadline,
  } }));
  await page.goto("/?page=overview");
  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "授权码", exact: true })).toBeVisible({ timeout: 8000 });
  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "验证并继续" })).toBeEnabled();
});

test("authenticated operator can manage and safely exit the current license", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=capture");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation).toBeVisible();
  await openUtility(page, "关于");
  await expect(page.getByRole("dialog", { name: "关于 NovaSight" })).toBeVisible();
  await expect(page.getByRole("region", { name: "授权与设备" })).toBeVisible();
  await page.getByRole("button", { name: "退出当前授权" }).click();
  await page.getByRole("button", { name: "确认：先紧急停止设备，再退出授权" }).click();

  await expect(page.getByRole("heading", { level: 1, name: "NovaSight" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "授权码" })).toBeVisible();
  await expect(navigation).toHaveCount(0);
});

test("management exposes onboarding and real object pages without fake cloud controls", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=management");

  await expect(page.getByRole("heading", { level: 1, name: "资源与授权" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "设备与资源" })).toBeVisible();
  await expect(page.getByRole("button", { name: /账单|团队|云端/ })).toHaveCount(0);
  await page.getByRole("button", { name: "查看设备状态" }).click();
  await expect(page).toHaveURL(/\?page=device$/);
  await page.goto("/?page=onboarding");
  await expect(page).toHaveURL(/\?page=onboarding$/);
  await expect(page.getByRole("heading", { level: 1, name: "新手引导" })).toBeVisible();
  await expect(page.getByRole("status", { name: "设置进度待核实" })).toBeVisible();
  await expect(page.getByRole("button", { name: "查看连接问题" })).toBeVisible();
});
