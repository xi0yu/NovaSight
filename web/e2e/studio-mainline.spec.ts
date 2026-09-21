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
