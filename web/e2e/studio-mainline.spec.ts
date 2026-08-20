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

test("manual access code establishes an authenticated Studio session", async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.goto("/");

  await page.locator("#web-access-code").fill("one-time-access-code");
  await page.getByRole("button", { name: "进入控制台" }).click();

  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
  await expect(page.locator("body")).not.toContainText("one-time-access-code");
});

test("startup access link survives StrictMode cleanup and establishes a session", async ({ page }) => {
  await mockStudioApi(page, unauthenticatedSession);
  await page.goto("/#access=one-time-access-code");

  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toBeVisible();
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

test("narrow Studio keeps Chinese navigation and save action reachable", async ({ page }) => {
  await page.setViewportSize({ width: 620, height: 812 });
  await mockStudioApi(page);
  await page.goto("/?page=params");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation.getByText("参数设置", { exact: true })).toBeVisible();

  const shellHeader = page.locator(".console-top");
  expect((await shellHeader.boundingBox())?.height ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(76);

  const saveBar = page.locator(".parameter-save-bar");
  await expect(saveBar).toHaveCSS("position", "sticky");
  await page.locator("main.console-main").evaluate((main) => { main.scrollTop = 900; });
  await expect(page.getByRole("button", { name: "保存修改" })).toBeInViewport();

  for (const control of [
    page.locator(".error-center-trigger"),
    page.getByRole("button", { name: "保存修改" }),
    navigation.getByRole("button", { name: "参数设置" }),
  ]) {
    expect((await control.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
  }
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
