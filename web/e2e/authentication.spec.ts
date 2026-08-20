import { expect, test } from "@playwright/test";

const unauthenticatedSession = {
  authenticated: false,
  principal: null,
  role: null,
  permissions: [],
  csrf_token: null,
  expires_at: null,
  session_lifetime_seconds: 900,
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/auth/session", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: unauthenticatedSession });
      return;
    }
    await route.fulfill({
      status: 429,
      contentType: "application/json",
      body: JSON.stringify({ code: "AUTH_RATE_LIMITED", message: "rate limited", detail: "rate limited" }),
    });
  });
});

test("anonymous LAN browser only sees the compact authentication entry", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "验证后继续" })).toBeVisible();
  await expect(page.locator("#web-access-code")).toHaveAttribute("type", "password");
  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toHaveCount(0);
  await expect(page.getByText("验证不会改变设备运行状态")).toBeVisible();
});

test("rate-limited authentication gives one bounded recovery message", async ({ page }) => {
  await page.goto("/");
  await page.locator("#web-access-code").fill("not-a-real-code");
  await page.getByRole("button", { name: "进入控制台" }).click();
  await expect(page.getByRole("alert")).toContainText("请等待 60 秒");
  await expect(page.locator("body")).not.toContainText("not-a-real-code");
});

for (const theme of ["rose-white", "graphite-red"] as const) {
  test(`${theme} applies before the authentication screen paints`, async ({ page }) => {
    await page.addInitScript((selectedTheme) => localStorage.setItem("novasight.theme", selectedTheme), theme);
    await page.goto("/");
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await expect(page.getByRole("heading", { name: "验证后继续" })).toBeVisible();
  });
}
