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

async function mockStudioApi(page: Page, initialSession = authenticatedSession) {
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
