import { expect, test } from "@playwright/test";
import { mockStudioApi, openUtility, authenticatedSession } from "./fixtures/studio";

test("device preview scans a simulated input and saves quality locally without network mutations", { tag: "@mobile" }, async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (request) => { if (/\/(api\/|ws\/|healthz)/.test(request.url())) requests.push(request.url()); });
  await page.goto("/preview.html?page=capture");
  await page.getByRole("button", { name: "重新扫描", exact: true }).click();
  await page.getByRole("option", { name: /\/dev\/preview0/ }).click();
  await page.getByRole("button", { name: "120 FPS", exact: true }).click();
  await expect(page.locator(".capture-command-header")).toContainText("1920x1080");
  await expect(page.locator(".capture-command-header")).toContainText("/dev/preview0");
  await expect(page.locator(".toast-card").filter({ hasText: "画质已更新（预览）" })).toBeVisible();
  await expect(page.locator(".capture-command-header")).toContainText("等待运行验证");
  const scene = (await page.locator(".capture-roi-visual").boundingBox())!;
  const region = (await page.locator(".capture-roi-visual > div").boundingBox())!;
  expect(scene.width / scene.height).toBeCloseTo(1920 / 1080, 1);
  expect(region.width).toBeLessThan(scene.width);
  expect(region.height).toBeLessThan(scene.height);
  await expect(page.locator(".capture-roi-visual")).toContainText("待读取 · 比例示意");
  await page.getByRole("button", { name: "实时日志", exact: true }).click();
  await page.getByRole("combobox", { name: "日志来源" }).selectOption("采集");
  await expect(page.getByRole("region", { name: "实时日志列表" })).toContainText("画质已更新（预览）");
  await page.getByRole("navigation", { name: "NovaSight Studio 导航" }).getByRole("button", { name: "算法参数", exact: true }).click();
  await page.getByRole("textbox", { name: "触发后等待时长 数值", exact: true }).fill("200");
  await page.getByRole("textbox", { name: "触发后等待时长 数值", exact: true }).press("Tab");
  await expect(page.getByRole("status").filter({ hasText: "预览修改已保留" })).toContainText("不写入设备文件");
  expect(requests).toEqual([]);
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

test("narrow Studio keeps Chinese navigation and save action reachable", async ({ page }) => {
  await page.setViewportSize({ width: 620, height: 812 });
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { trigger_mode: "always" }, pipeline: {} });
  await page.goto("/?page=params");

  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation.getByText("算法参数", { exact: true })).toBeVisible();
  await expect(navigation.getByRole("button", { name: "算法参数" })).toBeInViewport();

  const shellHeader = page.locator(".console-top");
  expect((await shellHeader.boundingBox())?.height ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(76);

  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).fill("25");
  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).press("Tab");
  const saveBar = page.locator(".parameter-save-bar");
  await expect(saveBar).toHaveCSS("position", "sticky");
  await page.locator("main.console-main").evaluate((main) => { main.scrollTop = 900; });
  await expect(page.getByRole("button", { name: "重试保存" })).toBeInViewport();
  await expect(page.getByRole("button", { name: "Choose File" })).toHaveCount(0);
  await expect(page.locator('input[type="file"]')).toHaveCount(0);

  for (const control of [
    page.locator(".error-center-trigger"),
    page.getByRole("button", { name: "重试保存" }),
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
  await expect(navigation.getByRole("button", { name: "更多" })).toBeInViewport();
  await navigation.getByRole("button", { name: "更多" }).click();
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

test("audit visual regressions keep ROI readable, mobile controls visible and Chinese filters on one line", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto("/preview.html?page=overview");
  const master = page.getByRole("switch", { name: /运行总开关/ });
  await expect(master).toBeInViewport();
  expect((await master.boundingBox())!.y).toBeLessThan(600);
  await page.screenshot({ path: testInfo.outputPath("overview-first-screen.png"), animations: "disabled" });
  await page.goto("/preview.html?page=capture");
  const roi = page.locator(".capture-roi-visual");
  await roi.scrollIntoViewIfNeeded();
  const contrast = await roi.evaluate((element) => {
    const luminance = (color: string) => {
      const channels = color.match(/[\d.]+/g)!.slice(0, 3).map((channel) => { const value = Number(channel) / 255; return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4; });
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    };
    const bg = luminance(getComputedStyle(element).backgroundColor);
    return [...element.querySelectorAll("span,b,small")].map((label) => { const fg = luminance(getComputedStyle(label).color); return (Math.max(bg, fg) + 0.05) / (Math.min(bg, fg) + 0.05); });
  });
  expect(Math.min(...contrast)).toBeGreaterThanOrEqual(4.5);
  await expect(roi).toContainText("比例示意");
  await roi.screenshot({ path: testInfo.outputPath("roi-contrast.png") });
  await page.goto("/preview.html?page=activity&showcase=activity");
  const filters = page.locator(".activity-filters button");
  await expect(filters).toHaveCount(5);
  for (const filter of await filters.all()) {
    const lineCount = await filter.locator("span").first().evaluate((label) => { const range = document.createRange(); range.selectNodeContents(label); return range.getClientRects().length; });
    expect(lineCount).toBe(1);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
  await page.screenshot({ path: testInfo.outputPath("activity-filters.png"), animations: "disabled" });
});

test("utility dialogs preserve the workspace and keep output last in the business menu", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { output_enabled: false }, pipeline: { p_response_scale: 0.2 } });
  await page.goto("/?page=params");
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(navigation).toBeVisible();
  await expect(page.locator(".console-primary-navigation .console-nav")).toHaveCount(6);
  expect(await page.locator(".console-primary-navigation .console-nav").evaluateAll((buttons) => buttons.map((button) => button.getAttribute("data-page")))).toEqual(["overview", "capture", "params", "fire", "activity", "control-test"]);
  expect(await page.locator(".console-utility-navigation .console-nav").evaluateAll((buttons) => buttons.map((button) => button.getAttribute("data-page")))).toEqual(["settings", "about"]);
  const main = page.locator(".console-main"), beforeUrl = page.url();
  const settingsButton = navigation.getByRole("button", { name: "设置", exact: true });
  await openUtility(page, "设置");
  const settings = page.getByRole("dialog", { name: "设置", exact: true });
  await expect(settings).toBeFocused();
  await expect(main).toHaveAttribute("data-page", "params");
  expect(await main.evaluate((element) => (element as HTMLElement).inert)).toBe(true);
  expect(page.url()).toBe(beforeUrl);
  await page.keyboard.press("Tab");
  await expect(settings.getByRole("button", { name: "关闭设置" })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(settings.locator("summary")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(settings).toHaveCount(0);
  await expect(page.viewportSize()!.width <= 1024 ? navigation.getByRole("button", { name: "更多" }) : settingsButton).toBeFocused();
  await expect(navigation.getByRole("button", { name: "算法参数", exact: true })).toHaveAttribute("aria-current", "page");
  expect(await main.evaluate((element) => (element as HTMLElement).inert)).toBe(false);
  await openUtility(page, "关于");
  const about = page.getByRole("dialog", { name: "关于 NovaSight", exact: true });
  await expect(about).toBeVisible();
  const box = (await about.boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await about.getByRole("button", { name: "打开实时日志" }).click();
  await expect(about).toHaveCount(0);
  await expect(main).toHaveAttribute("data-page", "activity");
  await page.goBack();
  await expect(main).toHaveAttribute("data-page", "params");
  expect(page.url()).toBe(beforeUrl);
});
