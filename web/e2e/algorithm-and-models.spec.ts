import { expect, test } from "@playwright/test";
import { mockStudioApi, openUtility, authenticatedSession } from "./fixtures/studio";

test("service failure is not shown as first-time setup or an empty model library", async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/studio/v1/status", (route) => route.fulfill({ status: 503, json: { code: "RUNTIME_UNAVAILABLE", message: "运行状态不可达" } }));
  await page.goto("/?page=overview");
  await expect(page.getByRole("status").filter({ hasText: "当前运行结论" })).toContainText("无法确认运行状态");
  await expect(page.locator(".console-safety-deck")).toBeHidden();
  await expect(page.locator(".runtime-overview")).not.toContainText("服务未连接");
  await expect(page.getByText("无法确认运行状态", { exact: true })).toHaveCount(1);
  await expect(page.getByText(/首次设置/)).toHaveCount(0);

  await page.goto("/?page=models");
  await expect(page.getByText("设备模型目录尚未读取成功")).toBeVisible();
  await expect(page.locator(".console-safety-deck .console-live")).toBeHidden();
  await expect(page.locator(".error-center-trigger > span")).toBeHidden();
  expect((await page.locator(".error-center-trigger").boundingBox())?.width ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(44);
  await expect(page.locator(".console-page-heading p")).toBeVisible();
  await expect(page.getByRole("button", { name: "新建文件夹" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "重试读取" })).toBeVisible();
  await expect(page.getByText(/暂无可选模型/)).toHaveCount(0);
});

test("parameter draft survives in-app navigation without a confirmation popup", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 1,
    control: {
      trigger_mode: "always",
    },
    pipeline: {},
  });
  let dialogCount = 0;
  page.on("dialog", async (dialog) => {
    dialogCount += 1;
    await dialog.dismiss();
  });
  await page.goto("/?page=params");

  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).fill("25");
  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).press("Tab");
  await expect(page.getByRole("button", { name: "重试保存" })).toBeEnabled();
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await navigation.getByRole("button", { name: "首页" }).click();
  await navigation.getByRole("button", { name: "算法参数" }).click();

  await expect(page.getByRole("textbox", { name: "触发后等待时长 数值" })).toHaveValue("25");
  await expect(page.getByRole("button", { name: "重试保存" })).toBeEnabled();
  expect(dialogCount).toBe(0);
});

test("algorithm parameters separate daily tuning from advanced tools", { tag: "@mobile" }, async ({ page }, testInfo) => {
  let config = {
    revision: 1,
    control: {
      trigger_mode: "always",
    },
    pipeline: { target_fov_radius_px: 180 },
    inference: { detection_class_profile: "default", detection_class_profiles: { default: Array.from({ length: 16 }, (_, index) => `class-${index}`) } },
  };
  await mockStudioApi(page, authenticatedSession, config);
  const writes: typeof config[] = [];
  await page.route("**/api/v1/config/parameters", async (route) => {
    if (route.request().method() === "POST") {
      const payload = route.request().postDataJSON() as { changes: Array<{ section: string; key: string; value: number }> };
      for (const change of payload.changes) {
        if (change.section === "pipeline") config.pipeline = { ...config.pipeline, [change.key]: change.value };
      }
      config = { ...config, revision: config.revision + 1 };
      writes.push(config);
      await route.fulfill({ json: {
        config, apply_mode: "hot_update", applied: true, rolled_back: false,
        restart_required: false, message: "applied",
      } });
    } else {
      await route.fulfill({ json: config });
    }
  });
  await page.goto("/?page=params");

  await expect(page.getByRole("switch", { name: /运行总开关/ })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "搜索范围", exact: true })).toBeVisible();
  const rangeTab = page.getByRole("tab", { name: "范围与触发", exact: true });
  await rangeTab.press("End");
  await expect(page.getByRole("tab", { name: "进阶调校", exact: true })).toBeFocused();
  await page.getByRole("tab", { name: "进阶调校", exact: true }).press("Home");
  await expect(rangeTab).toBeFocused();
  const radiusInput = page.getByRole("textbox", { name: "搜索半径 数值", exact: true });
  await expect(radiusInput).toHaveValue("180");
  await expect(page.getByRole("textbox", { name: "范围比例 数值", exact: true })).toHaveCount(0);
  await radiusInput.fill("240");
  await radiusInput.press("Tab");

  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0].pipeline.target_fov_radius_px).toBe(240);
  expect(writes[0].pipeline).not.toHaveProperty("target_range_scale");
  await expect(page.locator(".parameter-save-bar")).toHaveCount(0);
  await page.route("**/api/config", async (route) => route.fulfill({ json: config }));
  await page.reload();
  await expect(radiusInput).toHaveValue("240");
  await page.screenshot({ path: testInfo.outputPath("parameter-workspace.png"), animations: "disabled" });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await expect(page.getByText("预览搜索范围", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("slider", { name: "搜索半径 滑块" })).toHaveAttribute("max", "1000");
  await expect(page.getByRole("img", { name: /启动响应曲线/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: "移动响应", exact: true })).toHaveCount(0);
  await page.getByRole("tab", { name: "移动与输出", exact: true }).click();
  await expect(page.getByRole("heading", { name: "移动响应", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "单次 X 轴输出上限 数值", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "单次 Y 轴输出上限 数值", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "预测位移上限 数值", exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const outputLimits = page.locator(".control-chain-settings > li").last();
  await outputLimits.scrollIntoViewIfNeeded();
  const outputX = page.getByRole("textbox", { name: "单次 X 轴输出上限 数值", exact: true });
  await expect(outputX).toBeInViewport();
  await outputX.fill("96");
  await outputX.press("Tab");
  await expect.poll(() => writes.length).toBe(2);
  expect(writes[1]).toMatchObject({ pipeline: { max_output_x_counts: 96 } });
  await page.reload();
  await page.getByRole("tab", { name: "移动与输出", exact: true }).click();
  await expect(outputX).toHaveValue("96");
  await outputLimits.scrollIntoViewIfNeeded();
  await outputLimits.screenshot({ path: testInfo.outputPath("motion-output.png") });
  await expect(page.getByRole("region", { name: "响应试算", exact: true })).toHaveCount(0);
  await page.getByRole("tab", { name: "进阶调校", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "搜索半径上限 数值", exact: true })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "响应试算", exact: true })).not.toBeVisible();
  await page.locator("summary").filter({ hasText: "响应试算" }).click();
  await expect(page.getByRole("region", { name: "响应试算", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "目标与瞄点", exact: true }).click();
  await expect(page.getByRole("heading", { name: "目标锁定", exact: true })).toBeVisible();
  await expect(page.getByRole("list", { name: "当前类别配置" }).locator("li")).toHaveCount(16);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  for (const label of ["距离权重", "类别偏好权重", "置信度权重", "控制目标最低置信度", "目标切换确认延迟", "目标丢失保持", "大小权重", "连续性权重", "运动趋势权重", "运动趋势观察窗口"]) {
    await expect(page.getByRole("textbox", { name: `${label} 数值`, exact: true })).toHaveCount(0);
  }
  await expect(page.locator(".parameter-save-bar")).toHaveCount(0);
  await expect(page.getByText("完整配置文件", { exact: true })).toHaveCount(0);
  await expect(page.locator("main.console-main").getByRole("button", { name: "算法参数", exact: true })).toHaveCount(0);
  await expect(page.getByRole("dialog", { name: "控制参数" })).toHaveCount(0);

  await openUtility(page, "设置");
  await expect(page.getByRole("dialog", { name: "设置", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "备份与恢复" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "备份设置" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "恢复设置" })).toBeVisible();
});

test("model search keeps selection and verification together", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page);
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      { type: "model", name: "stable.engine", relative_path: "stable.engine", kind: "engine", size_bytes: 1_048_576, scan_status: "ready", scan_reason: "", recommendation: "recommended", tags: ["稳定"] },
      { type: "model", name: "other.engine", relative_path: "other.engine", kind: "engine", size_bytes: 1_048_576, scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [] },
    ] },
    directory_count: 1, model_count: 2, discovered_files: 0, updated_files: 0, cache_hits: 2, force: false,
  } }));
  await page.goto("/?page=params");
  await page.getByRole("navigation", { name: "算法使用的模型" }).getByRole("button", { name: "管理模型" }).click();
  await page.getByRole("button", { name: /stable.engine，路径 stable.engine/ }).click();
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeEnabled();
  await page.getByRole("searchbox", { name: "查找模型文件" }).fill("other.engine");
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeDisabled();
  await page.getByRole("button", { name: "清除筛选" }).click();
  await expect(page.getByRole("button", { name: "验证并切换到所选模型" })).toBeEnabled();
  await page.getByRole("navigation", { name: "NovaSight Studio 导航" }).getByRole("button", { name: "首页", exact: true }).click();
  await page.goBack();
  await page.getByRole("textbox", { name: "新增模型标签" }).fill("未保存的标签");
  await page.goForward();
  const discard = page.getByRole("alertdialog", { name: "放弃未保存的模型标签？" });
  await expect(discard).toBeVisible();
  await discard.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page).toHaveURL(/page=models/);
  await expect(page.getByRole("textbox", { name: "新增模型标签" })).toHaveValue("未保存的标签");
  await page.goForward();
  await discard.getByRole("button", { name: "放弃修改", exact: true }).click();
  await expect(page).toHaveURL(/page=overview/);
  await page.goBack();
  await expect(page.getByRole("heading", { name: "模型", exact: true })).toBeVisible();
});

test("stopped mainline asks before model registration and deployment", async ({ page }) => {
  await mockStudioApi(page);
  const mutations: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && (/\/api\/models\/catalog\/register$/.test(request.url()) || /\/publish$/.test(request.url()))) {
      mutations.push(request.url());
    }
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      { type: "model", name: "candidate.engine", relative_path: "candidate.engine", kind: "engine",
        size_bytes: 1_048_576, scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [] },
    ] },
    directory_count: 0, model_count: 1, discovered_files: 1, updated_files: 0, cache_hits: 0, force: false,
  } }));
  await page.goto("/?page=models");
  await page.getByRole("button", { name: /candidate.engine，路径 candidate.engine/ }).click();
  await page.getByRole("button", { name: "验证并切换到所选模型" }).click();
  const confirmation = page.getByRole("alertdialog", { name: "验证并部署所选模型？" });
  await expect(confirmation).toContainText("登记后不能直接移动或改名文件");
  await confirmation.getByRole("button", { name: "取消" }).click();
  await expect(confirmation).not.toBeVisible();
  expect(mutations).toEqual([]);
});

test("model library separates registered, unverified, and view-only files", async ({ page }) => {
  await mockStudioApi(page);
  const model = (name: string, kind: "engine" | "onnx", artifact_id?: number) => ({
    type: "model", name, relative_path: name, kind, size_bytes: 1_048_576,
    scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [], artifact_id,
  });
  await page.route("**/api/models/catalog?*", async (route) => route.fulfill({ json: {
    root: { type: "directory", name: "models", relative_path: "", children: [
      model("active.engine", "engine", 12),
      model("candidate.engine", "engine"),
      model("source.onnx", "onnx"),
    ] },
    directory_count: 0, model_count: 3, discovered_files: 0, updated_files: 0, cache_hits: 3, force: false,
  } }));
  await page.goto("/?page=models");
  const usage = page.getByRole("combobox", { name: "文件使用状态" });
  await expect(usage).toHaveText("全部文件（3）已登记 Engine（1）待验证 Engine（1）仅供查看（1）");
  await usage.selectOption("unregistered_engine");
  await expect(page.locator(".model-catalog-row.model")).toHaveCount(1);
  await expect(page.getByRole("button", { name: /candidate.engine，路径 candidate.engine/ })).toBeVisible();
  await usage.selectOption("view_only");
  await expect(page.locator(".model-catalog-row.model")).toHaveCount(1);
  await expect(page.getByRole("button", { name: /source.onnx，路径 source.onnx/ })).toBeVisible();
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(
    await page.evaluate(() => document.documentElement.clientWidth),
  );
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

test("model file move saves directly and reads back the new path", async ({ page }) => {
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
  expect(moved).toBe(false);
  await page.getByRole("button", { name: "保存文件位置" }).click();
  await expect(page.getByRole("alertdialog")).toHaveCount(0);
  await expect(page.getByText("模型文件已移至 Arena/renamed.engine；当前部署未改变。")).toBeVisible();
  await expect(page.getByRole("button", { name: /renamed.engine，路径 Arena\/renamed.engine/ })).toBeVisible();
  expect(publishRequests).toBe(0);
  expect(await page.evaluate(() => document.body.scrollWidth)).toBeLessThanOrEqual(await page.evaluate(() => document.body.clientWidth));
});

test("rejected model file move keeps the draft, explains the reason, and retains backend detail", async ({ page }) => {
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
  await page.getByRole("button", { name: "保存文件位置" }).click();
  await expect(page.locator(".model-file-organize")).toContainText("这个模型已登记或带有验证文件，不能直接移动");
  await expect(page.getByRole("button", { name: /model.engine，路径 model.engine/ })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "新文件名（.engine）" })).toHaveValue("new");
  await page.locator(".model-file-organize").getByRole("button", { name: "取消" }).click();
  await page.getByRole("button", { name: /^查看异常信息/ }).click();
  const errorCenter = page.getByRole("dialog", { name: "异常信息" });
  const moveError = errorCenter.getByRole("article").filter({ hasText: "模型文件移动失败" });
  await moveError.getByText("原始错误与开发者详情").click();
  await expect(moveError.locator("pre")).toContainText("model artifact 17 is registered");
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

test("withdraws rejected edits without a confirmation, including on narrow screens", { tag: "@mobile" }, async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { trigger_mode: "always" }, pipeline: {} });
  await page.goto("/?page=params");
  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).fill("25");
  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).press("Tab");
  await expect(page.getByRole("button", { name: "重试保存" })).toBeEnabled();
  await page.getByRole("button", { name: "撤销未保存修改" }).click();
  await expect(page.getByRole("alertdialog")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "撤销未保存修改" })).toHaveCount(0);
});

test("saving parameters while runtime is stopped does not show an output warning", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, {
    revision: 1,
    control: { trigger_mode: "hardware", output_enabled: true },
    pipeline: {},
  });
  const savedConfigs: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/api/v1/config/parameters" && request.method() === "POST") {
      savedConfigs.push(request.postDataJSON() as Record<string, unknown>);
    }
  });
  await page.goto("/?page=params");
  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).fill("35");
  await page.getByRole("textbox", { name: "触发后等待时长 数值" }).press("Tab");
  await expect.poll(() => savedConfigs.length).toBe(1);
  await expect(page.getByRole("alertdialog", { name: "物理输出仍开启，确认保存参数？" })).toHaveCount(0);
  expect(savedConfigs[0].changes).not.toEqual(expect.arrayContaining([expect.objectContaining({ section: "control", key: "output_enabled" })]));
});

test("silent checking retains the real parameter input without showing authorization", async ({ page }) => {
  await mockStudioApi(page, authenticatedSession, { revision: 1, control: { output_enabled: false }, pipeline: { p_response_scale: 0.2 } });
  await page.goto("/?page=params");
  await page.getByRole("tab", { name: "移动与输出" }).click();
  const gain = page.getByRole("textbox", { name: "比例增益 Kp 数值" });
  await gain.fill("0.7");
  await page.evaluate(() => window.dispatchEvent(new CustomEvent("novasight:auth-required")));
  await expect(page.getByRole("heading", { name: "设备授权" })).toHaveCount(0);
  await expect(gain).toBeVisible();
  await expect(page.getByRole("heading", { level: 1, name: "算法参数" })).toBeVisible();
  await expect(gain).toHaveValue(/^0\.7(?:00)?$/);
});
