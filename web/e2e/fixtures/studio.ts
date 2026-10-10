import type { Page } from "@playwright/test";

export async function openUtility(page: Page, name: "设置" | "关于") {
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  const more = navigation.getByRole("button", { name: "更多", exact: true });
  if (await more.isVisible()) await more.click();
  await navigation.getByRole("button", { name, exact: true }).click();
}

export const unauthenticatedSession = {
  authenticated: false,
  principal: null,
  role: null,
  permissions: [],
  csrf_token: null,
  expires_at: null,
  session_lifetime_seconds: 3_600,
};

export const authenticatedSession = {
  authenticated: true,
  principal: "operator",
  role: "operator",
  permissions: ["read", "mutate"],
  csrf_token: "csrf-test-token",
  expires_at: Math.floor(Date.now() / 1000) + 3_600,
  session_lifetime_seconds: 3_600,
};

export const activeLicense = {
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

export const clearedLicense = {
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

export async function mockStudioApi(
  page: Page,
  initialSession: typeof authenticatedSession | typeof unauthenticatedSession = authenticatedSession,
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
