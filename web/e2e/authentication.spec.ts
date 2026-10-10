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
      body: JSON.stringify({ code: "LICENSE_ACTIVATION_RATE_LIMITED", message: "rate limited", detail: "rate limited" }),
    });
  });
});

test("anonymous LAN browser sees the welcome journey and single-code entry", { tag: "@mobile" }, async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".welcome-story").getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByRole("heading", { name: "输入授权码" })).toBeVisible();
  await expect(page.getByRole("list", { name: "开始使用的三个步骤" })).toBeVisible();
  await expect(page.locator("#web-access-code")).toHaveAttribute("type", "password");
  await expect(page.getByRole("navigation", { name: "NovaSight Studio 导航" })).toHaveCount(0);
  await expect(page.getByText("授权不会自动启动主链或开启物理输出")).toBeVisible();
});

test("rate-limited authentication gives one bounded recovery message", async ({ page }) => {
  await page.goto("/");
  await page.locator("#web-access-code").fill("not-a-real-code");
  await page.getByRole("button", { name: "进入控制台" }).click();
  await expect(page.getByRole("alert")).toContainText("请等待 60 秒");
  await expect(page.locator("body")).not.toContainText("not-a-real-code");
});

for (const outcome of ["loaded", "failed"] as const) {
  test(`slow welcome artwork stays hidden until ${outcome} without moving the layout`, async ({ page }) => {
    let release!: () => void;
    const waiting = new Promise<void>((resolve) => { release = resolve; });
    await page.route(/nova-daylight\.webp(?:\?.*)?$/, async (route) => {
      if (route.request().resourceType() !== "image") return route.continue();
      await waiting;
      if (outcome === "failed") return route.abort("failed");
      return route.fulfill({ response: await route.fetch() });
    });
    try {
      await page.goto("/", { waitUntil: "domcontentloaded" });
      const frame = page.locator(".welcome-character");
      const image = frame.locator("img");
      await expect(frame).toHaveAttribute("aria-busy", "true");
      await expect(frame.locator(".image-placeholder")).toBeVisible();
      await expect(image).toBeHidden();
      const bounds = await frame.boundingBox();
      release();
      await expect(frame).toHaveAttribute("aria-busy", "false");
      expect(await frame.boundingBox()).toEqual(bounds);
      if (outcome === "loaded") {
        await expect(image).toBeVisible();
        await expect(frame.locator(".image-placeholder")).toHaveCount(0);
        await expect.poll(() => image.evaluate((element: HTMLImageElement) => element.complete && element.naturalWidth > 0)).toBe(true);
      } else {
        await expect(image).toBeHidden();
        await expect(frame.locator(".image-placeholder")).toBeVisible();
        await expect(frame).toHaveAttribute("title", "图片加载失败");
      }
    } finally { release(); }
  });
}
