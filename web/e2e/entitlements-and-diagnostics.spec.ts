import { expect, test } from "@playwright/test";
import { authenticatedSession, mockStudioApi, openUtility } from "./fixtures/studio";

test("diagnostics is workspace 07 and exposes live evidence and all useful deep routes", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { schema_version: 20, revision: 1, capture: { device: "/dev/video0", pixel_format: "MJPG", width: 1920, height: 1080, fps: 60 }, control: { output_enabled: false } });
  const writes: string[] = [];
  page.on("request", (request) => { if (request.method() !== "GET" && !request.url().endsWith("/api/capture/capabilities")) writes.push(request.url()); });
  await page.goto("/?page=capture");
  await expect(page.locator(".console-main")).not.toContainText("采集排查信息");
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  const entry = navigation.getByRole("button", { name: "链路诊断", exact: true });
  await expect(entry.locator(".console-nav-index")).toHaveText("07");
  await entry.click();
  await expect(page.getByRole("region", { name: "采集排查信息", exact: true })).toContainText("/dev/video0");
  await expect(page.getByRole("region", { name: "推理排查信息", exact: true })).toContainText("推理调度");
  await expect(page.getByRole("button", { name: "刷新状态", exact: true })).toBeVisible();
  await expect(page.getByLabel("链路状态")).toContainText("控制");
  await expect(page.getByRole("region", { name: "采集排查信息", exact: true })).toBeVisible();
  for (const [label, route] of [["设备状态", "device"], ["识别画面", "infer"], ["控制轨迹", "control"], ["性能测量", "latency"]]) {
    await page.getByRole("navigation", { name: "深入检查" }).getByRole("button", { name: label, exact: true }).click();
    await expect(page.locator(".console-main")).toHaveAttribute("data-page", route);
    await expect(entry).toHaveAttribute("aria-current", "page");
    await entry.click();
  }
  expect(writes).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("fire stability has decoded scene art, a honest diagram and preserved save behavior", { tag: "@mobile" }, async ({ page }, testInfo) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { output_enabled: false, fire_stabilization: { enabled: true, strength: 0.5 } } });
  const writes: unknown[] = [];
  await page.route("**/api/v1/config/parameters", async (route) => {
    writes.push(route.request().postDataJSON());
    return route.fulfill({ status: 503, json: { code: "CONFIG_UNAVAILABLE", message: "设置未能保存" } });
  });
  await page.goto("/?page=fire");
  await expect(page.getByRole("region", { name: "稳定辅助的工作方式" })).toBeVisible();
  await expect(page.getByText(/机制示意，非设备实测曲线/)).toBeVisible();
  await expect(page.locator(".fire-stability-scene")).toHaveAttribute("data-image-state", "ready");
  const heroTitle = page.getByRole("heading", { name: "让每次开火，更从容" });
  expect(parseFloat(await heroTitle.evaluate((element) => getComputedStyle(element).fontSize))).toBeLessThanOrEqual(22);
  await page.screenshot({ path: testInfo.outputPath("fire-stability.png"), animations: "disabled" });
  const strength = page.getByRole("textbox", { name: "抑制强度 数值", exact: true });
  const oldCurve = await page.locator(".fire-preview-adjusted").getAttribute("d");
  await strength.fill("0.8"); await strength.press("Tab");
  await expect.poll(() => writes.length).toBe(1);
  await expect(page.locator(".fire-preview-adjusted")).not.toHaveAttribute("d", oldCurve!);
  await expect(page.getByText("自动保存失败，修改已保留", { exact: true })).toBeVisible();
  await expect(strength).toHaveValue("0.80");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("authorization has entitlement seals and only advertises granted abilities", async ({ page }, testInfo) => {
  await mockStudioApi(page); await page.goto("/?page=overview"); await openUtility(page, "关于");
  const about = page.getByRole("dialog", { name: "关于 NovaSight", exact: true });
  await expect(about.getByText("专属权益 · 已解锁")).toBeVisible();
  await expect(about.locator(".license-benefit-card")).toHaveCount(3);
  await expect(about.getByText("硬件控制", { exact: true })).toHaveCount(0);
  await expect(about.getByText("画面采集", { exact: true })).toBeVisible();
  await expect(about.getByRole("button")).toHaveCount(1);
  await expect(about.getByRole("link")).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath("entitlements.png"), animations: "disabled" });
});
