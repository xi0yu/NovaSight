import { expect, test, type Page } from "@playwright/test";

const unauthenticatedSession = {
  authenticated: false,
  principal: null,
  role: null,
  permissions: [],
  csrf_token: null,
  expires_at: null,
  session_lifetime_seconds: 3_600,
};

const authenticatedSession = {
  authenticated: true,
  principal: "operator",
  role: "operator",
  permissions: ["read", "mutate"],
  csrf_token: "csrf-test-token",
  expires_at: Math.floor(Date.now() / 1000) + 3_600,
  session_lifetime_seconds: 3_600,
};

const activeLicense = {
  configured: true,
  valid: true,
  temporary_access_supported: true,
  fingerprint: "credential-fingerprint",
  tier: "pro",
  features: ["capture", "runtime", "models", "config_read", "config_write"],
  license_id: "license-a",
  credential_format: "jwt_rs256",
  token_id: "token-a",
  key_id: "key-a",
  created_at: 1,
  not_before: 1,
  activated_at: 1,
  expires_at: null,
  duration_value: null,
  duration_unit: "",
  updated_at: 1,
  message: "",
};

const clearedLicense = {
  ...activeLicense,
  configured: false,
  valid: false,
  fingerprint: "",
  tier: "",
  features: [],
  license_id: "",
  credential_format: "",
  token_id: "",
  key_id: "",
  created_at: null,
  not_before: null,
  activated_at: null,
  updated_at: null,
};

async function mockStudioApi(
  page: Page,
  initialSession = authenticatedSession,
  runtimeConfig?: Record<string, unknown>,
) {
  await page.route("**/healthz", async (route) => {
    await route.fulfill({ json: { ok: true } });
  });
  await page.routeWebSocket("**/ws/**", () => undefined);
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/auth/session" && route.request().method() === "GET") {
      await route.fulfill({ json: initialSession });
      return;
    }
    if (path === "/api/auth/session" && route.request().method() === "POST") {
      await route.fulfill({ json: authenticatedSession });
      return;
    }
    if (path === "/api/license" && route.request().method() === "GET") {
      await route.fulfill({ json: activeLicense });
      return;
    }
    if (path === "/api/license" && route.request().method() === "DELETE") {
      await route.fulfill({ json: clearedLicense });
      return;
    }
    if (path === "/api/config" && route.request().method() === "GET" && runtimeConfig) {
      await route.fulfill({ json: runtimeConfig });
      return;
    }
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ code: "TEST_UNAVAILABLE", message: "unavailable", detail: "unavailable" }),
    });
  });
}

test("one license code establishes an authenticated and activated Studio session", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.addInitScript(() => localStorage.setItem("novasight.theme", "rose-white"));
  await page.goto("/");

  await expect(page.locator("html")).toHaveAttribute("data-theme", "arena-signal");
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

test("conditional UI preview stays isolated and exposes recoverable states", { tag: "@mobile" }, async ({ page }, testInfo) => {
  const deviceRequests: string[] = [];
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("request", (request) => { if (/\/(api\/|ws\/|healthz)/.test(request.url())) deviceRequests.push(request.url()); });
  await page.goto("/preview.html?page=overview");
  await expect(page.getByRole("heading", { level: 1, name: "首页" })).toBeVisible();
  await expect(page.getByRole("switch", { name: /运行总开关/ })).toBeVisible();
  await expect(page.getByText("设备未确认", { exact: true })).toBeVisible();
  await expect(page.locator("body")).not.toContainText("/dev/video0");
  const choose = page.getByRole("combobox", { name: "选择预览状态" });
  await choose.selectOption("activity");
  await page.getByRole("searchbox", { name: "搜索日志" }).fill("preview-request-001");
  await expect(page.getByText("模型读取失败", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "清理历史" }).click();
  await expect(page.getByRole("alert")).toContainText("记录已保留");
  await page.getByRole("searchbox").fill("");
  await page.screenshot({ path: testInfo.outputPath("activity-states.png"), animations: "disabled" });
  await choose.selectOption("settings-unavailable");
  await expect(page.getByRole("status")).toContainText("设置未读取");
  await choose.selectOption("models-stale");
  await expect(page.getByRole("alert")).toContainText("显示上次读取的文件");
  await expect(page.locator(".model-promotion-track")).toHaveCSS("list-style-type", "none");
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeDisabled();
  await page.screenshot({ path: testInfo.outputPath("model-library-state.png"), animations: "disabled" });
  await page.getByRole("textbox", { name: "新增模型标签" }).fill("测试草稿");
  await page.getByRole("button", { name: "关闭模型管理" }).click();
  await expect(page.getByRole("alertdialog", { name: "放弃未保存的模型标签？" })).toBeVisible();
  await page.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "新增模型标签" })).toHaveValue("测试草稿");
  await choose.selectOption("switch-failed");
  await expect(page.getByRole("dialog")).toContainText("不能假定切换未生效");
  await expect(page.getByRole("dialog")).not.toContainText("%");
  await page.screenshot({ path: testInfo.outputPath("model-switch-state.png"), animations: "disabled" });
  await choose.selectOption("confirmation-error");
  const dialog = page.getByRole("alertdialog");
  await expect(dialog.getByRole("alert")).toContainText("修改仍保留");
  expect((await dialog.boundingBox())!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await choose.selectOption("safety");
  await page.getByRole("button", { name: "模拟未确认" }).click();
  await expect(page.getByText("紧急停止尚未确认", { exact: true })).toBeVisible();
  await choose.selectOption("auth");
  await expect(page.getByRole("heading", { name: "输入授权码" })).toBeVisible();
  await choose.selectOption("license");
  await expect(page.getByText("授权有效", { exact: true })).toBeVisible();
  for (const state of ["offline", "stale", "models-empty", "models", "settings", "switch-running", "switch-success", "confirmation", "toast"]) {
    await choose.selectOption(state);
    await expect(choose).toHaveValue(state);
    await expect(page.getByText("这个页面暂时无法显示", { exact: true })).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  expect(deviceRequests).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("Studio navigation uses a desktop rail and returns to a top strip on narrow screens", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await mockStudioApi(page);
  await page.goto("/?page=overview");

  const navigation = page.locator(".console-sidebar");
  const content = page.locator(".console-main");
  const desktopNavigation = await navigation.boundingBox();
  const desktopContent = await content.boundingBox();
  expect(desktopNavigation).not.toBeNull();
  expect(desktopContent).not.toBeNull();
  expect(desktopNavigation!.x).toBeLessThan(desktopContent!.x);
  expect(desktopNavigation!.height).toBeGreaterThan(desktopContent!.height * 0.9);

  await page.setViewportSize({ width: 760, height: 812 });
  const narrowNavigation = await navigation.boundingBox();
  const narrowContent = await content.boundingBox();
  expect(narrowNavigation).not.toBeNull();
  expect(narrowContent).not.toBeNull();
  expect(narrowNavigation!.y).toBeLessThan(narrowContent!.y);
  expect(narrowNavigation!.width).toBeGreaterThan(narrowContent!.width * 0.9);
});

test("service failure is not shown as first-time setup or an empty model library", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=overview");
  await expect(page.getByRole("status").filter({ hasText: "当前运行结论" })).toContainText("无法确认运行状态");
  await expect(page.locator(".console-safety-deck")).toBeHidden();
  await expect(page.locator(".runtime-overview")).not.toContainText("服务未连接");
  await expect(page.getByText("无法确认运行状态", { exact: true })).toHaveCount(1);
  await expect(page.getByText(/首次设置/)).toHaveCount(0);

  await page.goto("/?page=models");
  await expect(page.getByText("设备模型目录尚未读取成功")).toBeVisible();
  await expect(page.locator(".console-safety-deck .console-live")).toBeHidden();
  await expect(page.locator(".error-center-trigger > span")).toBeHidden();
  expect((await page.locator(".error-center-trigger").boundingBox())?.width ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(44);
  await expect(page.locator(".console-page-heading p")).toBeVisible();
  await expect(page.getByRole("button", { name: "新建文件夹" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "重试读取" })).toBeVisible();
  await expect(page.getByText(/暂无可选模型/)).toHaveCount(0);
});

test("unavailable runtime leads to the error details", async ({ page }) => {
  await mockStudioApi(page);
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

test("Studio navigation restores each page scroll position", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=params");

  const main = page.locator("main.console-main");
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  const paramsScrollTop = await main.evaluate((element) => {
    element.scrollTop = Math.min(720, element.scrollHeight - element.clientHeight);
    return element.scrollTop;
  });
  expect(paramsScrollTop).toBeGreaterThan(0);

  await navigation.getByRole("button", { name: "首页" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "首页" })).toBeFocused();
  await navigation.getByRole("button", { name: "算法参数" }).click();

  await expect.poll(() => main.evaluate((element) => element.scrollTop)).toBe(paramsScrollTop);
});

test("parameter draft survives in-app navigation without a confirmation popup", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 1,
    control: {
      trigger_mode: "always",
    },
    pipeline: {},
  });
  let dialogCount = 0;
  page.on("dialog", async (dialog) => {
    dialogCount += 1;
    await dialog.dismiss();
  });
  await page.goto("/?page=params");

  await page.getByRole("textbox", { name: "触发延迟" }).fill("25");
  await page.getByRole("textbox", { name: "触发延迟" }).press("Tab");
  await expect(page.getByRole("button", { name: "保存并应用" })).toBeEnabled();
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await navigation.getByRole("button", { name: "首页" }).click();
  await navigation.getByRole("button", { name: "算法参数" }).click();

  await expect(page.getByRole("textbox", { name: "触发延迟" })).toHaveValue("25");
  await expect(page.getByRole("button", { name: "保存并应用" })).toBeEnabled();
  expect(dialogCount).toBe(0);
});

test("algorithm parameters separate daily tuning from advanced tools", { tag: "@mobile" }, async ({ page }, testInfo) => {
  let config = {
    revision: 1,
    control: {
      trigger_mode: "always",
    },
    pipeline: { target_fov_radius_px: 180 },
  };
  await mockStudioApi(page, authenticatedSession, config);
  const writes: typeof config[] = [];
  await page.route("**/api/config", async (route) => {
    if (route.request().method() === "POST") {
      const payload = route.request().postDataJSON() as typeof config;
      writes.push(payload);
      config = { ...payload, revision: config.revision + 1 };
      await route.fulfill({ json: {
        config, apply_mode: "epoch_reload", applied: true, rolled_back: false,
        restart_required: false, message: "applied",
      } });
    } else {
      await route.fulfill({ json: config });
    }
  });
  await page.goto("/?page=params");

  await expect(page.getByRole("switch", { name: /运行总开关/ })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "搜索范围", exact: true })).toBeVisible();
  const rangeTab = page.getByRole("tab", { name: "范围与触发", exact: true });
  await rangeTab.press("End");
  await expect(page.getByRole("tab", { name: "进阶调校", exact: true })).toBeFocused();
  await page.getByRole("tab", { name: "进阶调校", exact: true }).press("Home");
  await expect(rangeTab).toBeFocused();
  const radiusInput = page.getByRole("textbox", { name: "搜索半径 数值", exact: true });
  await expect(radiusInput).toHaveValue("180");
  await expect(page.getByRole("textbox", { name: "范围比例 数值", exact: true })).toHaveCount(0);
  await radiusInput.fill("240");
  await radiusInput.press("Tab");
  expect(writes).toHaveLength(0);
  await page.getByRole("button", { name: "保存并应用", exact: true }).click();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0].pipeline.target_fov_radius_px).toBe(240);
  expect(writes[0].pipeline).not.toHaveProperty("target_range_scale");
  await expect(page.locator(".parameter-save-bar")).toHaveCount(0);
  await page.reload();
  await expect(radiusInput).toHaveValue("240");
  await page.screenshot({ path: testInfo.outputPath("parameter-workspace.png"), animations: "disabled" });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.locator(".search-range-scene").screenshot({ path: testInfo.outputPath("search-range.png") });
  await expect(page.getByRole("heading", { name: "移动响应", exact: true })).toHaveCount(0);
  await page.getByRole("tab", { name: "移动与输出", exact: true }).click();
  await expect(page.getByRole("heading", { name: "移动响应", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "单次 X 轴输出上限", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "单次 Y 轴输出上限", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "预测位移上限 数值", exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const outputLimits = page.locator(".control-chain-settings > li").last();
  await outputLimits.scrollIntoViewIfNeeded();
  const outputX = page.getByRole("textbox", { name: "单次 X 轴输出上限", exact: true });
  await expect(outputX).toBeInViewport();
  await outputX.fill("96");
  await outputX.press("Tab");
  await page.getByRole("button", { name: "保存并应用", exact: true }).click();
  await expect.poll(() => writes.length).toBe(2);
  expect(writes[1]).toMatchObject({ pipeline: { max_output_x_counts: 96 } });
  await page.reload();
  await page.getByRole("tab", { name: "移动与输出", exact: true }).click();
  await expect(outputX).toHaveValue("96");
  await outputLimits.scrollIntoViewIfNeeded();
  await outputLimits.screenshot({ path: testInfo.outputPath("motion-output.png") });
  await expect(page.getByRole("region", { name: "响应试算", exact: true })).toHaveCount(0);
  await page.getByRole("tab", { name: "进阶调校", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "搜索半径上限 数值", exact: true })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "响应试算", exact: true })).not.toBeVisible();
  await page.locator("summary").filter({ hasText: "响应试算" }).click();
  await expect(page.getByRole("region", { name: "响应试算", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "目标与瞄点", exact: true }).click();
  await expect(page.getByRole("heading", { name: "目标锁定", exact: true })).toBeVisible();
  await expect(page.getByRole("list", { name: "当前类别配置" }).locator("li")).toHaveCount(16);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  for (const label of ["距离权重", "类别偏好权重", "置信度权重", "控制目标最低置信度", "目标切换确认延迟", "目标丢失保持", "大小权重", "连续性权重", "运动趋势权重", "运动趋势观察窗口"]) {
    await expect(page.getByRole("textbox", { name: `${label} 数值`, exact: true })).toHaveCount(0);
  }
  await expect(page.locator(".parameter-save-bar")).toHaveCount(0);
  await expect(page.getByText("完整配置文件", { exact: true })).toHaveCount(0);
  await expect(page.locator("main.console-main").getByRole("button", { name: "算法参数", exact: true })).toHaveCount(0);
  await expect(page.getByRole("dialog", { name: "控制参数" })).toHaveCount(0);

  await page.getByRole("navigation", { name: "NovaSight Studio 导航" }).getByRole("button", { name: "设置" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "设置" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "备份与恢复" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "备份设置" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "恢复设置" })).toBeVisible();
});

test("model search keeps selection and verification together", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      { type: "model", name: "stable.engine", relative_path: "stable.engine", kind: "engine", size_bytes: 1_048_576, scan_status: "ready", scan_reason: "", recommendation: "recommended", tags: ["稳定"] },
      { type: "model", name: "other.engine", relative_path: "other.engine", kind: "engine", size_bytes: 1_048_576, scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [] },
    ] },
    directory_count: 1, model_count: 2, discovered_files: 0, updated_files: 0, cache_hits: 2, force: false,
  } }));
  await page.goto("/?page=models");
  await page.getByRole("button", { name: /stable.engine，路径 stable.engine/ }).click();
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeEnabled();
  await page.getByRole("searchbox", { name: "查找模型文件" }).fill("other.engine");
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeDisabled();
  await page.getByRole("button", { name: "清除筛选" }).click();
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeEnabled();
  await page.getByRole("navigation", { name: "NovaSight Studio 导航" }).getByRole("button", { name: "首页", exact: true }).click();
  await page.goBack();
  await page.getByRole("textbox", { name: "新增模型标签" }).fill("未保存的标签");
  await page.goForward();
  const discard = page.getByRole("alertdialog", { name: "放弃未保存的模型标签？" });
  await expect(discard).toBeVisible();
  await discard.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page).toHaveURL(/page=models/);
  await expect(page.getByRole("textbox", { name: "新增模型标签" })).toHaveValue("未保存的标签");
  await page.goForward();
  await discard.getByRole("button", { name: "放弃修改", exact: true }).click();
  await expect(page).toHaveURL(/page=overview/);
  await page.goBack();
  await expect(page.getByRole("heading", { name: "模型", exact: true })).toBeVisible();
});

test("stopped mainline asks before model registration and deployment", async ({ page }) => {
  await mockStudioApi(page);
  const mutations: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && (/\/api\/models\/catalog\/register$/.test(request.url()) || /\/publish$/.test(request.url()))) {
      mutations.push(request.url());
    }
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      { type: "model", name: "candidate.engine", relative_path: "candidate.engine", kind: "engine",
        size_bytes: 1_048_576, scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [] },
    ] },
    directory_count: 0, model_count: 1, discovered_files: 1, updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.goto("/?page=models");
  await page.getByRole("button", { name: /candidate.engine，路径 candidate.engine/ }).click();
  await page.getByRole("button", { name: "验证并切换到所选模型" }).click();
  const confirmation = page.getByRole("alertdialog", { name: "验证并部署所选模型？" });
  await expect(confirmation).toContainText("登记后不能直接移动或改名文件");
  await confirmation.getByRole("button", { name: "取消" }).click();
  await expect(confirmation).not.toBeVisible();
  expect(mutations).toEqual([]);
});

test("model library separates registered, unverified, and view-only files", async ({ page }) => {
  await mockStudioApi(page);
  const model = (name: string, kind: "engine" | "onnx", artifact_id?: number) => ({
    type: "model", name, relative_path: name, kind, size_bytes: 1_048_576,
    scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [], artifact_id,
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      model("active.engine", "engine", 12),
      model("candidate.engine", "engine"),
      model("source.onnx", "onnx"),
    ] },
    directory_count: 0, model_count: 3, discovered_files: 0, updated_files: 0, cache_hits: 3, force: false,
  } }));
  await page.goto("/?page=models");
  await expect(page.getByText(/1 个已登记 Engine · 1 个待验证 Engine · 1 个仅供查看/)).toBeVisible();
  const usage = page.getByRole("combobox", { name: "文件使用状态" });
  await usage.selectOption("unregistered_engine");
  await expect(page.locator(".model-catalog-row.model")).toHaveCount(1);
  await expect(page.getByRole("button", { name: /candidate.engine，路径 candidate.engine/ })).toBeVisible();
  await usage.selectOption("view_only");
  await expect(page.locator(".model-catalog-row.model")).toHaveCount(1);
  await expect(page.getByRole("button", { name: /source.onnx，路径 source.onnx/ })).toBeVisible();
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(
    await page.evaluate(() => document.documentElement.clientWidth),
  );
});

test("model library browses nested folders and respects name and size sorting", async ({ page }) => {
  await mockStudioApi(page);
  const engine = (name: string, relative_path: string, size_bytes: number) => ({
    type: "model", name, relative_path, kind: "engine", size_bytes,
    scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [],
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      engine("root.engine", "root.engine", 100),
      { type: "directory", name: "Arena", relative_path: "Arena", children: [
        engine("engine-10.engine", "Arena/engine-10.engine", 100),
        engine("engine-2.engine", "Arena/engine-2.engine", 200),
      ] },
    ] },
    directory_count: 1, model_count: 3, discovered_files: 3, updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.goto("/?page=models");
  await page.getByRole("button", { name: "打开文件夹 Arena，包含 2 个模型" }).click();
  await expect(page.getByRole("button", { name: "Arena", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".model-catalog-row.model").first()).toContainText("engine-2.engine");
  await page.getByRole("combobox", { name: "模型排序" }).selectOption("name_desc");
  await expect(page.locator(".model-catalog-row.model").first()).toContainText("engine-10.engine");
  await page.getByRole("combobox", { name: "模型排序" }).selectOption("size_desc");
  await expect(page.locator(".model-catalog-row.model").first()).toContainText("engine-2.engine");
  await page.getByRole("searchbox", { name: "查找模型文件" }).fill("root.engine");
  await expect(page.getByText("全部文件夹的筛选结果")).toBeVisible();
  await expect(page.getByRole("button", { name: /root.engine，路径 root.engine/ })).toBeVisible();
  await page.getByRole("button", { name: "清除筛选" }).click();
  await expect(page.getByRole("button", { name: /engine-2.engine，路径 Arena\/engine-2.engine/ })).toBeVisible();
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.body.clientWidth));
});

test("an empty model library creates a folder and reads it back without switching models", async ({ page }) => {
  await mockStudioApi(page);
  let created = false;
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: created
      ? [{ type: "directory", name: "Arena", relative_path: "Arena", children: [] }] : [] },
    directory_count: created ? 1 : 0, model_count: 0, discovered_files: 0,
    updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.route("**/api/models/catalog/folders", async (route) => {
    expect(route.request().method()).toBe("POST");
    expect(route.request().postDataJSON()).toEqual({ relative_path: "Arena" });
    created = true;
    await route.fulfill({ json: { relative_path: "Arena" } });
  });
  await page.goto("/?page=models");
  await page.getByRole("button", { name: "新建文件夹" }).click();
  await page.getByLabel("在 全部文件夹 中新建文件夹").fill("Arena");
  await page.getByRole("button", { name: "创建", exact: true }).click();
  await expect(page.getByRole("button", { name: "Arena", exact: true })).toHaveAttribute("aria-current", "page");
  await page.getByRole("button", { name: "全部文件夹", exact: true }).click();
  await expect(page.getByRole("button", { name: "打开文件夹 Arena，包含 0 个模型" })).toBeVisible();
  await expect(page.getByText("文件夹 Arena 已创建；模型部署未改变。")).toBeVisible();
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.body.clientWidth));
});

test("model file move requires confirmation and reads back the new path", async ({ page }) => {
  await mockStudioApi(page);
  let moved = false;
  let publishRequests = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/publish")) publishRequests += 1;
  });
  const engine = (name: string, relative_path: string) => ({
    type: "model", name, relative_path, kind: "engine", size_bytes: 100,
    scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [],
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      ...(moved ? [] : [engine("model.engine", "model.engine")]),
      { type: "directory", name: "Arena", relative_path: "Arena", children: moved ? [engine("renamed.engine", "Arena/renamed.engine")] : [] },
    ] }, directory_count: 1, model_count: 1, discovered_files: 1, updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.route("**/api/models/catalog/move", async (route) => {
    expect(route.request().method()).toBe("POST");
    expect(route.request().postDataJSON()).toEqual({ from_path: "model.engine", to_path: "Arena/renamed.engine" });
    moved = true;
    await route.fulfill({ json: { relative_path: "Arena/renamed.engine" } });
  });
  await page.goto("/?page=models");
  await page.getByRole("button", { name: /model.engine，路径 model.engine/ }).click();
  await page.getByRole("button", { name: "移动或改名文件" }).click();
  await page.getByRole("textbox", { name: "新文件名（.engine）" }).fill("renamed");
  await page.getByRole("combobox", { name: "移到文件夹" }).selectOption("Arena");
  await page.getByRole("button", { name: "检查并确认" }).click();
  await expect(page.getByRole("alertdialog", { name: "确认移动或改名模型文件？" })).toBeVisible();
  expect(moved).toBe(false);
  await page.getByRole("button", { name: "确认更改文件路径" }).click();
  await expect(page.getByText("模型文件已移至 Arena/renamed.engine；当前部署未改变。")).toBeVisible();
  await expect(page.getByRole("button", { name: /renamed.engine，路径 Arena\/renamed.engine/ })).toBeVisible();
  expect(publishRequests).toBe(0);
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.body.clientWidth));
});

test("rejected model file move keeps the draft, explains the reason, and retains backend detail", async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      { type: "model", name: "model.engine", relative_path: "model.engine", kind: "engine", size_bytes: 100,
        scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [] },
    ] }, directory_count: 0, model_count: 1, discovered_files: 1, updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.route("**/api/models/catalog/move", async (route) => route.fulfill({ status: 409, json: {
    code: "MODEL_FILE_IN_USE", message: "model artifact 17 is registered", detail: "model artifact 17 is registered",
  } }));
  await page.goto("/?page=models");
  await page.getByRole("button", { name: /model.engine，路径 model.engine/ }).click();
  await page.getByRole("button", { name: "移动或改名文件" }).click();
  await page.getByRole("textbox", { name: "新文件名（.engine）" }).fill("new");
  await page.getByRole("button", { name: "检查并确认" }).click();
  await page.getByRole("button", { name: "确认更改文件路径" }).click();
  await expect(page.getByRole("alertdialog", { name: "确认移动或改名模型文件？" })).toContainText("这个模型已登记或带有验证文件，不能直接移动");
  await expect(page.getByRole("button", { name: /model.engine，路径 model.engine/ })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "新文件名（.engine）" })).toHaveValue("new");
  await page.getByRole("alertdialog", { name: "确认移动或改名模型文件？" }).getByRole("button", { name: "取消" }).click();
  await page.getByRole("button", { name: /^查看异常信息/ }).click();
  const errorCenter = page.getByRole("dialog", { name: "异常信息" });
  const moveError = errorCenter.getByRole("article").filter({ hasText: "模型文件移动失败" });
  await moveError.getByText("原始错误与开发者详情").click();
  await expect(moveError.locator("pre")).toContainText("model artifact 17 is registered");
});

test("model organization saves catalog metadata without switching the runtime model", async ({ page }) => {
  await mockStudioApi(page);
  let recommendation = "unrated";
  let tags: string[] = [];
  let publishRequests = 0;
  page.on("request", (request) => {
    if (request.url().includes("/api/models/projects/") && request.url().endsWith("/publish")) publishRequests += 1;
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      { type: "model", name: "stable.engine", relative_path: "stable.engine", kind: "engine", size_bytes: 1_048_576,
        scan_status: "need_confirm", scan_reason: "", project_id: 3, project_name: "stable", version_id: 4,
        version_name: "external-1", artifact_id: 12, artifact_status: "pending", recommendation, tags },
    ] },
    directory_count: 1, model_count: 1, discovered_files: 1, updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.route("**/api/models/artifacts/12/metadata", async (route) => {
    const body = route.request().postDataJSON() as { recommendation: string; tags: string[] };
    recommendation = body.recommendation;
    tags = body.tags;
    await route.fulfill({ json: { artifact_id: 12, recommendation, tags } });
  });
  await page.goto("/?page=models");
  await page.getByRole("button", { name: /stable.engine，路径 stable.engine/ }).click();
  await page.getByRole("button", { name: "推荐", exact: true }).last().click();
  await page.getByRole("textbox", { name: "新增模型标签" }).fill("低延迟");
  await page.getByRole("button", { name: "保存整理结果" }).click();
  await expect(page.getByText(/已保存 stable.engine 的推荐状态与 1 个标签/)).toBeVisible();
  expect(recommendation).toBe("recommended");
  expect(tags).toEqual(["低延迟"]);
  expect(publishRequests).toBe(0);
});

test("discarding parameter edits asks first, including on narrow screens", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { trigger_mode: "always" }, pipeline: {} });
  await page.goto("/?page=params");
  await page.getByRole("textbox", { name: "触发延迟" }).fill("25");
  await page.getByRole("textbox", { name: "触发延迟" }).press("Tab");
  await page.getByRole("button", { name: "放弃修改" }).click();
  const confirmation = page.getByRole("alertdialog", { name: "放弃未保存的修改？" });
  await expect(confirmation).toBeVisible();
  expect((await confirmation.boundingBox())?.width ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(page.viewportSize()?.width ?? 0);
  await confirmation.getByRole("button", { name: "取消" }).click();
  await expect(page.getByRole("button", { name: "放弃修改" })).toBeVisible();
});

test("saving parameters while runtime is stopped does not show an output warning", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 1,
    control: { trigger_mode: "hardware", output_enabled: true },
    pipeline: {},
  });
  const savedConfigs: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/api/config" && request.method() === "POST") {
      savedConfigs.push(request.postDataJSON() as Record<string, unknown>);
    }
  });
  await page.goto("/?page=params");
  await page.getByRole("textbox", { name: "触发延迟" }).fill("35");
  await page.getByRole("textbox", { name: "触发延迟" }).press("Tab");
  await page.getByRole("button", { name: "保存并应用" }).click();
  await expect.poll(() => savedConfigs.length).toBe(1);
  await expect(page.getByRole("alertdialog", { name: "物理输出仍开启，确认保存参数？" })).toHaveCount(0);
  expect((savedConfigs[0].control as Record<string, unknown>).output_enabled).toBe(true);
});

test("narrow Studio keeps Chinese navigation and save action reachable", async ({ page }) => {
  await page.setViewportSize({ width: 620, height: 812 });
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { trigger_mode: "always" }, pipeline: {} });
  await page.goto("/?page=params");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation.getByText("算法参数", { exact: true })).toBeVisible();
  await expect(navigation.getByRole("button", { name: "算法参数" })).toBeInViewport();

  const shellHeader = page.locator(".console-top");
  expect((await shellHeader.boundingBox())?.height ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(76);

  await page.getByRole("textbox", { name: "触发延迟" }).fill("25");
  await page.getByRole("textbox", { name: "触发延迟" }).press("Tab");
  const saveBar = page.locator(".parameter-save-bar");
  await expect(saveBar).toHaveCSS("position", "sticky");
  await page.locator("main.console-main").evaluate((main) => { main.scrollTop = 900; });
  await expect(page.getByRole("button", { name: "保存并应用" })).toBeInViewport();
  await expect(page.getByRole("button", { name: "Choose File" })).toHaveCount(0);
  await expect(page.locator('input[type="file"]')).toHaveCount(0);

  for (const control of [
    page.locator(".error-center-trigger"),
    page.getByRole("button", { name: "保存并应用" }),
    navigation.getByRole("button", { name: "算法参数" }),
  ]) {
    expect((await control.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
  }
});


test("1024px Studio switches layout before Windows scrollbars cause overflow", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await mockStudioApi(page);
  await page.goto("/?page=params");

  const gridColumns = await page.locator(".console-app").evaluate(
    (element) => getComputedStyle(element).gridTemplateColumns,
  );
  expect(gridColumns.trim().split(/\s+/)).toHaveLength(1);
  const viewportWidths = await page.locator("body").evaluate((body) => ({
    client: document.documentElement.clientWidth,
    scroll: body.scrollWidth,
  }));
  expect(viewportWidths.scroll).toBeLessThanOrEqual(viewportWidths.client);
});

test("375px Studio keeps shell actions and horizontal navigation accessible", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await mockStudioApi(page);
  await page.goto("/?page=params");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  const activePageButton = navigation.getByRole("button", { name: "算法参数" });
  await activePageButton.scrollIntoViewIfNeeded();

  await expect(activePageButton).toBeInViewport();
  await expect(page.locator(".error-center-trigger")).toBeInViewport();
  await expect(navigation.getByRole("button", { name: "设置" })).toBeVisible();
  await expect(navigation.getByRole("button", { name: "关于" })).toBeVisible();
  expect(await page.locator("body").evaluate((body) => body.scrollWidth)).toBeLessThanOrEqual(375);
});

test("Studio action feedback respects reduced motion, transparency and higher contrast", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=overview");

  const action = page.locator(".console-button:visible:not(:disabled)").first();
  await action.scrollIntoViewIfNeeded();
  const session = await page.context().newCDPSession(page);
  await session.send("Emulation.setEmulatedMedia", {
    features: [
      { name: "prefers-reduced-motion", value: "reduce" },
      { name: "prefers-reduced-transparency", value: "reduce" },
      { name: "prefers-contrast", value: "more" },
    ],
  });
  expect(await page.evaluate(() => matchMedia("(prefers-reduced-transparency: reduce)").matches)).toBe(true);
  await action.hover();
  await page.mouse.down();
  await expect(action).toHaveCSS("transform", "none");
  await page.mouse.up();

  await expect(page.locator(".console-app")).toHaveCSS("backdrop-filter", "none");
  // Check readable rendered text on an opaque surface, not equality of tokens.
  await expect(page.locator(".console-app")).toHaveCSS("background-color", /^rgb\(/);
  const contrast = await page.locator(".console-page-heading p").evaluate((element) => {
    const luminance = (color: string) => {
      const [r, g, b] = color.match(/[\d.]+/g)!.slice(0, 3).map((channel) => {
        const value = Number(channel) / 255;
        return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const text = luminance(getComputedStyle(element).color);
    const surface = luminance(getComputedStyle(element.closest(".console-app")!).backgroundColor);
    return (Math.max(text, surface) + 0.05) / (Math.min(text, surface) + 0.05);
  });
  expect(contrast).toBeGreaterThanOrEqual(4.5);
});

test("760px Studio recovery actions keep touch-safe targets", async ({ page }) => {
  await page.setViewportSize({ width: 760, height: 812 });
  await mockStudioApi(page);
  await page.goto("/?page=control");

  for (const control of [
    page.getByRole("button", { name: "查看异常", exact: true }),
    page.getByRole("button", { name: "重试" }),
    page.locator(".error-center-trigger"),
  ]) {
    expect((await control.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
  }
});

test("one backend outage does not repeat friendly and channel errors", async ({ page }) => {
  await mockStudioApi(page);
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
  await page.goto("/?page=control");

  await expect(page.getByRole("heading", { level: 1, name: "目标与控制" })).toBeVisible();
  await expect(page.getByRole("button", { name: "查看异常", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "重试" })).toBeVisible();

  await page.getByRole("button", { name: "查看异常", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "异常信息" })).toBeVisible();
});

test("authenticated operator can manage and safely exit the current license", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=capture");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation).toBeVisible();
  await navigation.getByRole("button", { name: "关于" }).click();
  await page.getByRole("button", { name: "查看授权" }).click();

  await expect(page).toHaveURL(/\?page=license$/);
  await expect(page.getByRole("heading", { level: 1, name: "授权" })).toBeVisible();
  await page.getByRole("button", { name: "退出当前授权" }).click();
  await page.getByRole("button", { name: "确认：先紧急停止设备，再退出授权" }).click();

  await expect(page.getByRole("heading", { level: 1, name: "NovaSight" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "授权码" })).toBeVisible();
  await expect(navigation).toHaveCount(0);
});

test("configuration pages keep their heading and content within the viewport", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const route of ["models", "license", "params"]) {
    await page.goto(`/?page=${route}`);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(
      await page.evaluate(() => document.documentElement.clientWidth),
    );
  }
});

test("Studio exposes six visible task spaces without a hidden secondary navigation", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=overview");
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  for (const label of ["首页", "设备管理", "算法参数", "实时日志", "设置", "关于"]) {
    await expect(navigation.getByRole("button", { name: label })).toBeVisible();
  }
  await expect(navigation.getByRole("region", { name: "当前设备的深入功能" })).toHaveCount(0);
  await navigation.getByRole("button", { name: "设备管理" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "设备管理" })).toBeFocused();
  await navigation.getByRole("button", { name: "实时日志" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "实时日志" })).toBeFocused();
  await expect(page.getByText(/条故障与警告记录/)).toBeVisible();
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
