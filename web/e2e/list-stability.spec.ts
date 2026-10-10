import { expect, test } from "@playwright/test";
import { authenticatedSession, mockStudioApi } from "./fixtures/studio";

test("model directory keeps its frame, search and selection during delayed reads and failed refresh", { tag: "@mobile" }, async ({ page }, testInfo) => {
  await mockStudioApi(page);
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/models/catalog?*", async (route) => {
    await waiting;
    if (new URL(route.request().url()).searchParams.get("force") === "true") return route.fulfill({ status: 503, json: { message: "目录暂时不可达" } });
    return route.fulfill({ json: {
      root: { type: "directory", name: "models", relative_path: "", children: [{ type: "model", name: "stable.engine", relative_path: "stable.engine", kind: "engine", size_bytes: 1024, scan_status: "ready", scan_reason: "", recommendation: "recommended", tags: [] }] },
      directory_count: 1, model_count: 1, discovered_files: 0, updated_files: 0, cache_hits: 1, force: false,
    } });
  });
  try {
    await page.goto("/?page=models", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("status", { name: "正在读取模型文件" })).toBeVisible();
    const search = page.getByRole("searchbox", { name: "查找模型文件" });
    const before = (await search.boundingBox())!;
    const columnsBefore = (await page.locator(".model-catalog-columns").boundingBox())!;
    await expect(search).toBeDisabled();
    await page.screenshot({ path: testInfo.outputPath("model-reading.png"), animations: "disabled" });
    release();
    const row = page.getByRole("button", { name: /stable.engine，路径/ });
    await expect(row).toBeVisible();
    await expect(search).toBeEnabled();
    const after = (await search.boundingBox())!;
    expect(Math.abs(after.y - before.y)).toBeLessThan(2);
    expect(Math.abs(after.width - before.width)).toBeLessThan(2);
    expect(Math.abs((await page.locator(".model-catalog-columns").boundingBox())!.y - columnsBefore.y)).toBeLessThan(2);
    await row.click();
    await page.getByRole("button", { name: "刷新模型" }).click();
    await expect(page.getByRole("alert").filter({ hasText: "显示上次读取的文件" })).toBeVisible();
    await expect(row).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeDisabled();
    await page.screenshot({ path: testInfo.outputPath("model-ready.png"), animations: "disabled" });
  } finally { release(); }
});

test("capture capabilities stay visible during refresh and cannot apply a failed or different-device result", async ({ page }, testInfo) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, capture: { device: "/dev/video0", pixel_format: "MJPG", width: 1920, height: 1080, fps: 60 } });
  await page.route("**/api/capture/devices", async (route) => route.fulfill({ json: { devices: ["/dev/video0", "/dev/video1"] } }));
  let reads = 0;
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/capture/capabilities", async (route) => {
    reads += 1;
    if (reads > 1) { await waiting; return route.fulfill({ status: 503, json: { message: "设备不可达" } }); }
    return route.fulfill({ json: { available: true, device: "/dev/video0", reason: "", capabilities: [{ pixel_format: "MJPG", width: 1920, height: 1080, fps_list: [60] }] } });
  });
  try {
    await page.goto("/?page=capture");
    const row = page.locator(".capture-resolution-row");
    await expect(row).toContainText("1920 × 1080");
    const before = (await row.boundingBox())!;
    await page.getByRole("option", { name: /\/dev\/video0/ }).click();
    await expect(page.getByText("正在刷新，保留上次读取的规格")).toBeVisible();
    await expect(row).toHaveCount(1);
    expect(Math.abs((await row.boundingBox())!.y - before.y)).toBeLessThan(2);
    await page.screenshot({ path: testInfo.outputPath("capture-refreshing.png"), animations: "disabled" });
    release();
    await expect(page.getByText("读取失败，显示上次规格；重新读取后可选择")).toBeVisible();
    await expect(row.getByRole("button", { name: "60 FPS" })).toBeDisabled();
    await page.getByRole("option", { name: /\/dev\/video1/ }).click();
    await expect(row).toHaveCount(0);
  } finally { release(); }
});

test("activity history uses a stable local placeholder and reveals records together", async ({ page }, testInfo) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, pipeline: {}, control: {} });
  await page.route("**/healthz", async (route) => route.fulfill({ json: { ok: true } }));
  let releaseRuntime!: () => void;
  const runtimeWaiting = new Promise<void>((resolve) => { releaseRuntime = resolve; });
  await page.route("**/api/studio/v1/status", async (route) => { await runtimeWaiting; await route.fulfill({ status: 503, json: { message: "runtime unavailable" } }); });
  await page.route("**/api/config/schema", async (route) => { await runtimeWaiting; await route.fulfill({ status: 503, json: { message: "schema unavailable" } }); });
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => { release = resolve; });
  await page.route(/\/api\/activity(?:\?.*)?$/, async (route) => {
    await waiting;
    await route.fulfill({ json: { events: Array.from({ length: 4 }, (_, index) => ({ daemon_instance_id: "test", id: index, occurred_at_ms: 1700000000000 + index, level: "info", tag: "参数", title: `配置记录 ${index}`, message: "已保存配置", technical_detail: "", request_id: null, count: 1 })) } });
  });
  try {
    await page.goto("/?page=activity", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("status", { name: "正在读取日志记录" })).toBeVisible();
    await expect(page.getByText("暂无运行记录", { exact: true })).toHaveCount(0);
    const list = page.getByRole("region", { name: "实时日志列表" });
    const before = (await list.boundingBox())!;
    release();
    await expect(list.locator(".activity-feed-item")).toHaveCount(4);
    expect(Math.abs((await list.boundingBox())!.y - before.y)).toBeLessThan(2);
    await page.screenshot({ path: testInfo.outputPath("activity-ready.png"), animations: "disabled" });
  } finally { release(); releaseRuntime(); }
});
