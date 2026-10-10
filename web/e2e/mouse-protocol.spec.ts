import { expect, test } from "@playwright/test";
import { mockStudioApi, openUtility, authenticatedSession } from "./fixtures/studio";

test("cards enable a consumer and the separate configuration dialog saves its endpoint", async ({ page }) => {
  const protocol = { id: "selected", model: "net", host: "10.0.0.2", port: 8888, uuid: "A1B2C3D4" };
  const another = { ...protocol, id: "another", model: "nb", host: "10.0.0.3" };
  let config: Record<string, any> = { revision: 1, control: { output_enabled: false }, hardware: { backend: "null", active_mouse_protocol_id: "empty", mouse_protocols: [protocol, another] } };
  await mockStudioApi(page, authenticatedSession, config);
  let writes = 0;
  await page.route("**/api/config", async (route) => {
    if (route.request().method() === "GET") return route.fulfill({ json: config });
    const requested = route.request().postDataJSON();
    expect(requested.revision).toBe(config.revision);
    expect(route.request().headers()["x-novasight-physical-output-ack"]).toBe("confirmed");
    writes += 1;
    config = { ...requested, revision: config.revision + 1 };
    await route.fulfill({ json: { config, applied: true, rolled_back: false, apply_mode: "epoch_reload", restart_required: false, message: "ok" } });
  });
  await page.goto("/?page=control-test");
  await expect(page.locator(".mouse-protocol-toolbar").getByRole("button")).toHaveCount(2);
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".toast-card").filter({ hasText: "空协议无需配置" })).toBeVisible();
  expect(writes).toBe(0);
  await page.getByRole("button", { name: /kmBox Net ·/ }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByRole("article", { name: /kmBox Net ·/ })).toHaveAttribute("data-active", "true");
  expect(config.hardware.active_mouse_protocol_id).toBe(protocol.id);
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "配置协议", exact: true });
  await expect(dialog.getByRole("combobox", { name: "配置的协议" })).toHaveValue(protocol.id);
  await dialog.getByRole("textbox", { name: "IP", exact: true }).fill("10.0.0.99");
  await dialog.getByRole("spinbutton", { name: "Port" }).fill("8899");
  await dialog.getByRole("button", { name: "保存并写入文件" }).click();
  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已保存" })).toContainText("已应用并写入文件");
  expect(writes).toBe(2);
  expect(config.hardware).toMatchObject({ host: "10.0.0.99", port: 8899, active_mouse_protocol_id: protocol.id });
  expect(config.hardware.mouse_protocols[0]).toMatchObject({ host: "10.0.0.99", port: 8899 });
  await page.reload();
  await expect(page.getByRole("button", { name: "kmBox Net · 10.0.0.99:8899" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: /kmBox NB ·/ }).click();
  await expect(page.getByRole("button", { name: /kmBox NB ·/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: /kmBox Net ·/ })).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(config.hardware.active_mouse_protocol_id).toBe(another.id);
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "配置的协议" })).toHaveValue(another.id);
  await page.getByRole("dialog").getByRole("button", { name: "取消", exact: true }).click();
  await page.getByRole("button", { name: "空", exact: true }).click();
  await expect(page.getByRole("button", { name: "空", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: /kmBox NB ·/ })).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".toast-card").filter({ hasText: "空协议无需配置" })).toBeVisible();
  expect(writes).toBe(4);
});

test("candidate tests keep the active protocol unchanged and retain failure input", async ({ page }) => {
  const protocol = { id: "kept", model: "net", host: "10.0.0.2", port: 8888, uuid: "A1B2C3D4" };
  const config = { revision: 1, control: { output_enabled: false }, hardware: { backend: "native_udp", active_mouse_protocol_id: protocol.id, host: protocol.host, port: protocol.port, uuid: protocol.uuid, mouse_protocols: [protocol] } };
  await mockStudioApi(page, authenticatedSession, config);
  let pass = false;
  const requests: string[] = [];
  page.on("request", (request) => { if (request.method() === "POST") requests.push(new URL(request.url()).pathname); });
  await page.route("**/api/executors/kmnet/probe-candidate", async (route) => {
    expect(route.request().postDataJSON()).toMatchObject({ host: "10.0.0.3", port: 8888, uuid: "A1B2C3D4" });
    await route.fulfill(pass ? { json: { reachable: true, elapsed_ms: 8, checked_at_ms: Date.now() } } : { status: 409, json: { code: "DEVICE_UNAVAILABLE", message: "设备握手失败，请核对 IP。" } });
  });
  await page.goto("/?page=control-test");
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  const editor = page.getByRole("dialog", { name: "配置协议" });
  await editor.getByRole("textbox", { name: "IP" }).fill("10.0.0.3");
  await editor.getByRole("button", { name: "测试连通性" }).click();
  await expect(editor.getByRole("alert")).toContainText("握手失败");
  await expect(editor.getByRole("textbox", { name: "IP" })).toHaveValue("10.0.0.3");
  await expect(page.locator(".toast-card.tone-error").filter({ hasText: "连接测试失败" })).toBeVisible();
  await expect(page.locator(".toast-card.tone-error").filter({ hasText: "连接测试失败" })).toHaveCount(0, { timeout: 8000 });
  await expect(editor.getByRole("alert")).toContainText("握手失败");
  pass = true;
  await editor.getByRole("button", { name: "测试连通性" }).click();
  await expect(editor.getByText(/设备握手成功/)).toBeVisible();
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "连接测试成功" })).toBeVisible();
  await editor.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page.getByRole("article", { name: /kmBox Net ·/ })).toHaveAttribute("data-active", "true");
  expect(requests).toEqual(["/api/executors/kmnet/probe-candidate", "/api/executors/kmnet/probe-candidate"]);
  expect(config.hardware.mouse_protocols).toEqual([protocol]);
});

test("silent running preview shows output honesty and does not open empty protocol configuration", async ({ page }) => {
  await page.goto("/preview.html?page=overview");
  await page.getByRole("combobox").selectOption("running-target");
  await expect(page.getByRole("heading", { level: 2, name: /运行中.*空协议 · 静默输出/ })).toBeVisible();
  await expect(page.getByText("系统已停止并确认硬件输出安全", { exact: true })).toHaveCount(0);
  await page.getByRole("navigation", { name: "NovaSight Studio 导航" }).getByRole("button", { name: "鼠标协议", exact: true }).click();
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".toast-card").filter({ hasText: "空协议无需配置" })).toBeVisible();
});

test("duplicate protocols reuse a card and inactive removal is confirmed and saved", async ({ page }) => {
  const protocol = { id: "duplicate", model: "net_pro8k", host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4" };
  let config = { revision: 1, control: { output_enabled: false }, hardware: { backend: "null", active_mouse_protocol_id: "empty", mouse_protocols: [protocol] } };
  await mockStudioApi(page, authenticatedSession, config);
  let writes = 0;
  await page.route("**/api/config", async (route) => {
    if (route.request().method() === "GET") return route.fulfill({ json: config });
    writes += 1;
    const requested = route.request().postDataJSON();
    expect(requested.revision).toBe(config.revision);
    config = { ...requested, revision: config.revision + 1 };
    await route.fulfill({ json: { config, applied: true, rolled_back: false, apply_mode: "epoch_reload", restart_required: false, message: "ok" } });
  });
  await page.goto("/?page=control-test");
  await page.getByRole("button", { name: "新增协议" }).click();
  await page.getByRole("dialog", { name: "新增协议" }).getByRole("button", { name: "添加协议" }).click();
  const editor = page.getByRole("dialog", { name: "配置协议" });
  await expect(editor).toBeVisible();
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已存在" })).toBeVisible();
  expect(writes).toBe(0);
  await editor.getByRole("button", { name: "删除协议" }).click();
  expect(writes).toBe(0);
  await editor.getByRole("button", { name: "保留协议" }).click();
  await editor.getByRole("button", { name: "删除协议" }).click();
  await editor.getByRole("button", { name: "确认删除" }).click();
  await expect(editor).toHaveCount(0);
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已删除" })).toBeVisible();
  expect(writes).toBe(1);
  expect(config.hardware.mouse_protocols).toEqual([]);
  expect(config.hardware.active_mouse_protocol_id).toBe("empty");
});

test("mouse protocols are discoverable, probe without movement, and autosave uses red green top-center feedback", { tag: "@mobile" }, async ({ page }, testInfo) => {
  let config: Record<string, any> = {
    revision: 1, control: { trigger_mode: "always", output_enabled: false },
    pipeline: { p_response_scale: 0.2 },
    hardware: { host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4", monitor_port: 5001, auto_connect: true },
  };
  const mutations: string[] = [];
  page.on("request", (request) => { if (request.method() === "POST") mutations.push(new URL(request.url()).pathname); });
  await mockStudioApi(page, authenticatedSession, config);
  await page.route("**/api/v1/config/parameters", async (route) => {
    const { expected_revision, changes } = route.request().postDataJSON();
    expect(expected_revision).toBe(config.revision);
    for (const { section, key, value } of changes) config[section] = { ...config[section], [key]: value };
    config = { ...config, revision: config.revision + 1 };
    await route.fulfill({ json: { config, apply_mode: "hot_update", restart_required: false, applied: true, rolled_back: false, message: "ok" } });
  });
  await page.route("**/api/executors/kmnet/probe-candidate", async (route) => route.fulfill({ json: { reachable: true, elapsed_ms: 12, checked_at_ms: Date.now() } }));
  await page.goto("/?page=overview");
  const navigation = page.getByRole("navigation", { name: "NovaSight Studio 导航" });
  await expect(page.getByRole("button", { name: "紧急停止", exact: true })).toBeVisible();
  await navigation.getByRole("button", { name: "鼠标协议", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "鼠标协议" })).toBeFocused();
  await expect(page.getByRole("button", { name: /kmBox Net · 192.168.2.188:8888/ })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  await page.getByRole("dialog", { name: "配置协议" }).getByRole("button", { name: "测试连通性", exact: true }).click();
  await expect(page.getByText(/设备握手成功 ·/)).toBeVisible();
  expect(mutations).toEqual(["/api/executors/kmnet/probe-candidate"]);
  await page.screenshot({ path: testInfo.outputPath("mouse-protocol-settings.png"), animations: "disabled" });
  await page.getByRole("dialog", { name: "配置协议" }).getByRole("button", { name: "取消" }).click();
  await navigation.getByRole("button", { name: "算法参数", exact: true }).click();
  await page.getByRole("tab", { name: "移动与输出" }).click();
  const gain = page.getByRole("textbox", { name: "比例增益 Kp 数值" });
  await gain.fill("0.7"); await gain.press("Tab");
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "参数已自动保存并应用" })).toBeVisible();
  const box = await page.locator(".toast-host").boundingBox();
  expect(box!.y).toBeLessThan(40);
  expect(Math.abs(box!.x + box!.width / 2 - page.viewportSize()!.width / 2)).toBeLessThan(2);
  await expect(page.locator(".parameter-save-bar")).toHaveCount(0);
  await page.route("**/api/v1/config/parameters", async (route) => route.fulfill({ status: 422, json: { code: "CONFIG_REJECTED", message: "参数组合无效" } }));
  await gain.fill("0.8"); await gain.press("Tab");
  await expect(page.locator(".toast-card.tone-error").filter({ hasText: "参数自动保存失败" })).toBeVisible();
  await expect(gain).toHaveValue("0.800");
  await expect(page.getByRole("button", { name: "重试保存" })).toBeEnabled();
  await expect(page.getByRole("alertdialog")).toHaveCount(0);
  expect(mutations.filter((path) => /start|stop|diagnostic-move/.test(path))).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.screenshot({ path: testInfo.outputPath("autosave-feedback.png"), animations: "disabled" });
  await openUtility(page, "关于");
  await expect(page.getByRole("dialog", { name: "关于 NovaSight", exact: true })).toBeFocused();
});

test("mouse protocol cards add official product photos, persist selection and keep the 95/5 layout", { tag: "@mobile" }, async ({ page }, testInfo) => {
  let config: Record<string, any> = { revision: 1, control: { output_enabled: false }, hardware: { backend: "null", active_mouse_protocol_id: "empty", mouse_protocols: [], auto_connect: true } };
  let rejectActivation = false;
  const mutations: string[] = [];
  page.on("request", (request) => { if (request.method() === "POST") mutations.push(new URL(request.url()).pathname); });
  await mockStudioApi(page, authenticatedSession, config);
  await page.route("**/api/config", async (route) => {
    if (route.request().method() === "GET") { await route.fulfill({ json: config }); return; }
    if (rejectActivation) { await route.fulfill({ status: 503, json: { code: "CONFIG_UNAVAILABLE", message: "设备配置写入失败" } }); return; }
    const next = route.request().postDataJSON();
    expect(next.revision).toBe(config.revision);
    config = { ...next, revision: config.revision + 1 };
    await route.fulfill({ json: { config, apply_mode: "epoch_reload", restart_required: false, applied: true, rolled_back: false, message: "ok" } });
  });
  await page.goto("/?page=control-test");
  await expect(page.locator(".mouse-protocol-workspace > :not(.mouse-protocol-toolbar):not(.mouse-protocol-cards)")).toHaveCount(0);
  await expect(page.locator(".console-page-header")).toHaveClass(/visually-hidden/);
  await expect(page.getByRole("button", { name: "空", exact: true })).toHaveAttribute("aria-pressed", "true");
  for (const [index, model] of ["net_pro8k", "net", "nb", "ai"].entries()) {
    await page.getByRole("button", { name: "新增协议", exact: true }).click();
    const editor = page.getByRole("dialog", { name: "新增协议" });
    await editor.getByRole("combobox", { name: "型号" }).selectOption(model);
    await editor.getByRole("textbox", { name: "IP", exact: true }).fill(`10.0.0.${index + 2}`);
    await editor.getByRole("spinbutton", { name: "Port", exact: true }).fill("8888");
    await editor.getByRole("textbox", { name: "UUID", exact: true }).fill("A1B2C3D4");
    if (index === 0) {
      for (const button of [page.getByRole("button", { name: "新增协议", exact: true }), editor.getByRole("button", { name: "添加协议" })]) {
        const contrast = await button.evaluate((element) => {
          const luminance = (color: string) => {
            const channels = color.match(/[\d.]+/g)!.slice(0, 3).map(Number).map((value) => {
              const normalized = value / 255;
              return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
            });
            return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
          };
          const style = getComputedStyle(element), foreground = luminance(style.color), background = luminance(style.backgroundColor);
          return (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05);
        });
        expect(contrast).toBeGreaterThanOrEqual(4.5);
      }
      await page.screenshot({ path: testInfo.outputPath("mouse-protocol-editor-cute.png"), animations: "disabled" });
    }
    await editor.getByRole("button", { name: "添加协议" }).click();
    await expect(editor).toHaveCount(0);
    await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已保存" }).last()).toBeVisible();
    await expect(page.locator(".mouse-protocol-card")).toHaveCount(index + 2);
    expect(config.hardware.backend).toBe("null");
  }
  for (const card of await page.locator(".mouse-protocol-card").all()) {
    const measurements = await card.evaluate((element) => {
      const picture = element.querySelector<HTMLElement>(".mouse-protocol-picture")!;
      const caption = element.querySelector<HTMLElement>(".mouse-protocol-caption")!;
      const image = element.querySelector<HTMLImageElement>("img")!;
      return { width: (element as HTMLElement).offsetWidth, height: (element as HTMLElement).offsetHeight, ratio: picture.clientHeight / (picture.clientHeight + caption.clientHeight), centered: getComputedStyle(picture).alignItems === "center" && getComputedStyle(caption).alignItems === "center", loaded: image.complete && image.naturalWidth > 0, source: image.src };
    });
    expect(measurements.width).toBe(testInfo.project.name === "mobile-safety" ? 256 : 320);
    expect(measurements.height).toBe(measurements.width);
    expect(measurements.ratio).toBeCloseTo(0.95, 2);
    expect(measurements.centered).toBe(true);
    expect(measurements.loaded).toBe(true);
    expect(measurements.source).toContain("/src/assets/output-consumers/");
  }
  rejectActivation = true;
  await page.getByRole("button", { name: /kmBox NB ·/ }).click();
  await expect(page.locator(".toast-card.tone-error").filter({ hasText: "协议启用失败" })).toBeVisible();
  await expect(page.getByRole("article", { name: /kmBox NB ·/ })).toHaveAttribute("data-active", "false");
  expect(config.hardware.backend).toBe("null");
  rejectActivation = false;
  await page.getByRole("button", { name: /kmBox NB ·/ }).click();
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已启用" })).toBeVisible();
  const activeCard = page.getByRole("article", { name: /kmBox NB ·/ });
  await expect(activeCard.locator(".mouse-protocol-active-badge")).toHaveText("已启用");
  await expect(activeCard).toHaveCSS("border-color", "rgb(36, 105, 64)");
  await expect(page.getByRole("button", { name: /kmBox NB ·/ })).toHaveAttribute("aria-pressed", "true");
  expect(config.hardware.backend).toBe("native_udp");
  expect(config.hardware.host).toBe("10.0.0.4");
  await page.reload();
  await expect(page.locator(".mouse-protocol-card")).toHaveCount(5);
  await expect(page.getByRole("button", { name: /kmBox NB ·/ })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "空", exact: true }).click();
  await expect(page.getByRole("button", { name: "空", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已取消" })).toBeVisible();
  expect(config.hardware.backend).toBe("null");
  expect(config.hardware.mouse_protocols).toHaveLength(4);
  expect(mutations.every((path) => path === "/api/config")).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.screenshot({ path: testInfo.outputPath("mouse-protocol-cards.png"), animations: "disabled" });
});

test("mouse protocol preview adds cards locally without device API traffic", { tag: "@mobile" }, async ({ page }) => {
  const deviceRequests: string[] = [];
  page.on("request", (request) => { if (/\/(api\/|ws\/|healthz)/.test(request.url())) deviceRequests.push(request.url()); });
  await page.goto("/preview.html?page=control-test");
  await page.getByRole("button", { name: "新增协议" }).click();
  const editor = page.getByRole("dialog", { name: "新增协议" });
  await expect(editor.locator(".." )).toHaveCSS("backdrop-filter", "none");
  await expect(editor.locator("..")).toHaveCSS("background-color", "rgb(248, 243, 246)");
  await editor.getByRole("button", { name: "添加协议" }).click();
  await expect(page.locator(".mouse-protocol-card")).toHaveCount(2);
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已保存" })).toBeVisible();
  const card = page.getByRole("article", { name: "kmBox NetPro8K · 192.168.2.188:8888" });
  await card.getByRole("button", { name: /kmBox NetPro8K ·/ }).click();
  await expect(page.getByRole("button", { name: /kmBox NetPro8K ·/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已启用" })).toBeVisible();
  await page.getByRole("button", { name: "空", exact: true }).click();
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已取消" })).toBeVisible();
  await card.getByRole("button", { name: /kmBox NetPro8K ·/ }).click();
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已启用" })).toHaveCount(2);
  await page.getByRole("button", { name: "配置协议", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "配置协议" }).getByRole("button", { name: "测试连通性" })).toBeDisabled();
  const editDialog = page.getByRole("dialog", { name: "配置协议" });
  await editDialog.getByRole("textbox", { name: "IP", exact: true }).fill("bad-ip");
  await editDialog.getByRole("button", { name: "保存协议（预览）" }).click();
  const failure = page.locator(".toast-card.tone-error").filter({ hasText: "协议参数无效" });
  await expect(failure).toBeVisible();
  expect(await failure.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const layer = document.querySelector(".mouse-protocol-editor-layer")!;
    return Number(getComputedStyle(element.closest(".toast-host")!).zIndex) > Number(getComputedStyle(layer).zIndex)
      && rect.top >= 48 && rect.left >= 0 && rect.right <= innerWidth;
  })).toBe(true);
  await editDialog.getByRole("textbox", { name: "IP", exact: true }).fill("192.168.2.188");
  await editDialog.getByRole("button", { name: "保存协议（预览）" }).click();
  await expect(editDialog).toHaveCount(0);
  await expect(page.locator(".toast-card.tone-success").filter({ hasText: "协议已保存" })).toBeVisible();
  expect(deviceRequests).toEqual([]);
});
