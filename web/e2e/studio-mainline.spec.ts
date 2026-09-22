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

test("one license code establishes an authenticated and activated Studio session", async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.goto("/");

  await expect(page.locator("#auth-code-help")).toHaveText(
    "支持正式授权码或本次启动的临时授权码；不另设接入码。",
  );
  await page.getByLabel("授权码", { exact: true }).fill("one-license-code");
  const login = page.waitForRequest((request) => request.url().endsWith("/api/auth/session") && request.method() === "POST");
  await page.getByRole("button", { name: "进入控制台" }).click();
  expect((await login).postDataJSON()).toEqual({ key: "one-license-code" });

  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
  await expect(page.locator("body")).not.toContainText("one-license-code");
});

test("obsolete access links are cleared and cannot bypass license login", async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.goto("/#access=one-time-access-code");

  await expect(page.getByRole("heading", { name: "授权后进入" })).toBeVisible();
  expect(new URL(page.url()).hash).toBe("");
  await expect(page.locator("body")).not.toContainText("one-time-access-code");
});

test("Studio navigation moves keyboard focus to the new page heading", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=capture");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await navigation.getByRole("button", { name: "运行总览" }).click();

  await expect(page.getByRole("heading", { level: 1, name: "运行总览" })).toBeFocused();
});

test("unavailable runtime leads to the error details", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=overview");

  await page.getByRole("button", { name: "查看异常信息" }).click();
  await expect(page.getByRole("dialog", { name: "异常信息" })).toBeVisible();
});

test("capture page keeps saved and running specifications visible without horizontal overflow", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 25,
    capture: { device: "/dev/video0", pixel_format: "MJPG", width: 1920, height: 1080, fps: 240 },
  });
  await page.goto("/?page=capture");

  const check = page.locator(".capture-profile-check");
  await expect(check).toContainText("等待运行验证");
  await expect(check.getByText("MJPEG (MJPG) / 1920x1080 / 240 FPS", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "保存采集配置" })).toBeVisible();
  if ((page.viewportSize()?.width ?? 0) >= 1200) {
    await expect(page.getByRole("heading", { name: "采集设备" })).toBeInViewport();
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

  await navigation.getByRole("button", { name: "运行总览" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "运行总览" })).toBeFocused();
  await navigation.getByRole("button", { name: "参数设置" }).click();

  await expect.poll(() => main.evaluate((element) => element.scrollTop)).toBe(paramsScrollTop);
});

test("parameter draft survives in-app navigation without a confirmation popup", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 1,
    control: {
      trigger_mode: "always",
      recoil: { enabled: false, require_target: true, interval_ms: 16, y_counts: 1 },
    },
    pipeline: {},
  });
  let dialogCount = 0;
  page.on("dialog", async (dialog) => {
    dialogCount += 1;
    await dialog.dismiss();
  });
  await page.goto("/?page=params");

  await page.getByRole("button", { name: "按键触发" }).click();
  await expect(page.getByRole("button", { name: "保存修改" })).toBeEnabled();
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await navigation.getByRole("button", { name: "运行总览" }).click();
  await navigation.getByRole("button", { name: "参数设置" }).click();

  await expect(page.getByRole("button", { name: "按键触发" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "保存修改" })).toBeEnabled();
  expect(dialogCount).toBe(0);
});

test("algorithm parameters explain the two-stage save flow", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 1,
    control: {
      trigger_mode: "always",
      recoil: { enabled: false, require_target: true, interval_ms: 16, y_counts: 1 },
    },
    pipeline: {},
  });
  await page.goto("/?page=params");

  await page.getByRole("button", { name: "算法参数" }).click();

  const dialog = page.getByRole("dialog", { name: "控制参数" });
  await expect(dialog).toContainText("加入草稿并关闭");
  await expect(dialog).toContainText("返回参数页后点击“保存修改”才会写入设备");
});

test("model search keeps selection and verification together", async ({ page }) => {
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

test("rejected model file move keeps the original selection and shows the backend reason", async ({ page }) => {
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
  await expect(page.getByRole("alertdialog", { name: "确认移动或改名模型文件？" })).toContainText("model artifact 17 is registered");
  await expect(page.getByRole("button", { name: /model.engine，路径 model.engine/ })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "新文件名（.engine）" })).toHaveValue("new");
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

test("parameter sections navigate without changing physical output", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { trigger_mode: "hardware", output_enabled: false }, pipeline: {} });
  await page.goto("/?page=params");
  const outputShortcut = page.getByRole("navigation", { name: "参数分区" }).getByRole("button", { name: /物理输出/ });
  await outputShortcut.click();
  await expect(page.getByRole("heading", { name: "物理输出", exact: true })).toBeInViewport();
  expect(page.url()).not.toContain("#parameter-stage-output");
});

test("discarding parameter edits asks first, including on narrow screens", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { trigger_mode: "always" }, pipeline: {} });
  await page.goto("/?page=params");
  await page.getByRole("button", { name: "按键触发" }).click();
  await page.getByRole("button", { name: "放弃修改" }).click();
  const confirmation = page.getByRole("alertdialog", { name: "放弃未保存的修改？" });
  await expect(confirmation).toBeVisible();
  expect((await confirmation.boundingBox())?.width ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(page.viewportSize()?.width ?? 0);
  await confirmation.getByRole("button", { name: "取消" }).click();
  await expect(page.getByRole("button", { name: "放弃修改" })).toBeVisible();
});

test("narrow Studio keeps Chinese navigation and save action reachable", async ({ page }) => {
  await page.setViewportSize({ width: 620, height: 812 });
  await mockStudioApi(page);
  await page.goto("/?page=params");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation.getByText("参数设置", { exact: true })).toBeVisible();
  await expect(navigation.getByRole("button", { name: "参数设置" })).toBeInViewport();

  const shellHeader = page.locator(".console-top");
  expect((await shellHeader.boundingBox())?.height ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(76);

  const saveBar = page.locator(".parameter-save-bar");
  await expect(saveBar).toHaveCSS("position", "sticky");
  await page.locator("main.console-main").evaluate((main) => { main.scrollTop = 900; });
  await expect(page.getByRole("button", { name: "保存修改" })).toBeInViewport();
  await expect(page.getByRole("button", { name: "Choose File" })).toHaveCount(0);
  await expect(page.locator('input[type="file"]')).toHaveAttribute("tabindex", "-1");

  for (const control of [
    page.locator(".error-center-trigger"),
    page.getByRole("button", { name: "保存修改" }),
    navigation.getByRole("button", { name: "参数设置" }),
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
  const activePageButton = navigation.getByRole("button", { name: "参数设置" });
  await activePageButton.scrollIntoViewIfNeeded();

  await expect(activePageButton).toBeInViewport();
  await expect(page.locator(".error-center-trigger")).toBeInViewport();
  await expect(page.locator(".theme-toggle")).toBeInViewport();
  expect(await page.locator("body").evaluate((body) => body.scrollWidth)).toBeLessThanOrEqual(375);
});

test("theme chooser exposes visible controls that a user can actually click", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=overview");

  const trigger = page.locator(".theme-toggle");
  expect.soft(await trigger.evaluate((element) => element.tagName)).toBe("BUTTON");
  expect.soft(await trigger.innerText()).toContain("主题");
  await trigger.click();
  await expect(page.locator(".theme-option")).toHaveCount(3);

  const graphiteRed = page.getByRole("button", { name: /黑灰红/ });
  await expect(graphiteRed).toBeVisible();
  await graphiteRed.scrollIntoViewIfNeeded();
  const optionReceivesPointer = await graphiteRed.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return hit !== null && element.contains(hit);
  });
  expect(optionReceivesPointer).toBe(true);

  await graphiteRed.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "graphite-red");
  await expect(trigger).toContainText("黑灰红");
  expect(await page.evaluate(() => window.localStorage.getItem("novasight.theme"))).toBe("graphite-red");

  await trigger.click();
  const viewport = page.viewportSize();
  await page.mouse.click(12, (viewport?.height ?? 720) - 12);
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByRole("group", { name: "网站主题" })).toHaveCount(0);
});

test("Studio action feedback respects reduced motion, transparency and higher contrast", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=overview");

  const action = page.locator(".console-button:visible:not(:disabled)").first();
  await action.scrollIntoViewIfNeeded();
  await action.hover();
  await page.mouse.down();
  await expect(action).toHaveCSS("transform", "matrix(0.98, 0, 0, 0.98, 0, 0)");
  await page.mouse.move(0, 0);
  await page.mouse.up();

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
  await expect(action).toHaveCSS("box-shadow", /inset/);
  await page.mouse.up();

  await expect(page.locator(".console-app")).toHaveCSS("backdrop-filter", "none");
  await expect(page.locator(".console-app")).toHaveCSS("background-color", "rgb(255, 255, 255)");
  expect(await page.locator("html").evaluate((element) => {
    const style = getComputedStyle(element);
    return style.getPropertyValue("--border-default").trim() === style.getPropertyValue("--border-strong").trim();
  })).toBe(true);
});

test("760px Studio recovery actions keep touch-safe targets", async ({ page }) => {
  await page.setViewportSize({ width: 760, height: 812 });
  await mockStudioApi(page);
  await page.goto("/?page=control");

  for (const control of [
    page.getByRole("button", { name: "查看异常" }),
    page.getByRole("button", { name: "重试" }),
    page.locator(".error-center-trigger"),
    page.locator(".theme-toggle"),
  ]) {
    expect((await control.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
  }
});

test("one backend outage does not repeat friendly and channel errors", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=models");
  await page.getByRole("heading", { level: 1, name: "模型管理" }).waitFor();
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

  await expect(page.getByRole("heading", { level: 1, name: "控制" })).toBeVisible();
  await expect(page.getByRole("button", { name: "查看异常" })).toBeVisible();
  await expect(page.getByRole("button", { name: "重试" })).toBeVisible();

  await page.getByRole("button", { name: "查看异常" }).click();
  await expect(page.getByRole("dialog", { name: "异常信息" })).toBeVisible();
});

test("authenticated operator can manage and safely exit the current license", async ({ page }) => {
  await mockStudioApi(page);
  await page.goto("/?page=capture");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation).toBeVisible();
  await navigation.getByRole("button", { name: "授权管理" }).click();

  await expect(page).toHaveURL(/\?page=license$/);
  await expect(page.getByRole("heading", { level: 1, name: "授权管理" })).toBeVisible();
  await page.getByRole("button", { name: "退出当前授权" }).click();
  await page.getByRole("button", { name: "确认：先紧急停止设备，再退出授权" }).click();

  await expect(page.getByRole("heading", { level: 1, name: "NovaSight" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "授权码" })).toBeVisible();
  await expect(navigation).toHaveCount(0);
});

test("configuration pages explain the next action without horizontal overflow", async ({ page }) => {
  await mockStudioApi(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const [route, text] of [
    ["models", "模型库"],
    ["license", "需要更换授权？"],
    ["params", "先调好控制，再决定是否输出"],
  ] as const) {
    await page.goto(`/?page=${route}`);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expect(page.getByText(text, { exact: route !== "license" })).toBeVisible();
    expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(
      await page.evaluate(() => document.documentElement.clientWidth),
    );
  }
});
