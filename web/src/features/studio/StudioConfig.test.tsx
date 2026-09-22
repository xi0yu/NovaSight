import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { LicenseStatus, RuntimeState } from "../../api";
import { SafetyOperationProvider } from "../runtime/SafetyOperationContext";
import { StudioConsoleView } from "./StudioConsoleView";

const profile = { pixel_format: "MJPG", width: 1920, height: 1080, fps: 240 };
const runtime = {
  semantic: { daemon_instance_id: "test", phase: "running", perception_phase: "running", epoch: 1, snapshot_sequence: 1 },
  running: true, capture: { device: "/dev/video0", running: true, profile },
  statistics: { detection_data_age_ms: 1, detection_freshness_threshold_ms: 55 }, config: { version: 25, effective_version: 25 },
  executor: { executors: { kmnet: { available: true, connected: true, runtime_connected: true, connection_state: "connected" } } },
  pipeline: { running: true, state: "running", deepstream: {} }, inference: {},
  vision: { control: {}, output_trace: {}, target_pipeline: { rejection_reasons: [], counts: {} }, detections: [],
    inference: { roi_offset_x: 0, roi_offset_y: 0, roi_width: 256, roi_height: 256 } },
} as unknown as RuntimeState;
const props = {
  license: { valid: true, features: ["hardware_control"] } as LicenseStatus,
  health: null, runtime,
  runtimeConfig: { revision: 25, capture: { ...profile, device: "/dev/video0", roi_left: 0, roi_top: 0, roi_width: 256, roi_height: 256 },
    pipeline: { projection_fov_x_deg: 90 }, control: { output_enabled: false }, hardware: { auto_connect: true } },
  projects: [], errors: {}, lastUpdated: null, realtimeStatus: "connected" as const,
  onEnsureProjects: vi.fn(async () => []), onLicenseChange: vi.fn(), onRefresh: vi.fn(async () => {}),
  onRuntimeConfigChange: vi.fn(), onRuntimeStateChange: vi.fn(() => true), onStatusTopicChange: vi.fn(),
};
beforeEach(() => {
  history.replaceState(null, "", "/?page=params");
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
});
afterEach(() => { vi.unstubAllGlobals(); history.replaceState(null, "", "/"); });
async function openProfile() {
  await userEvent.click(screen.getByText("配置生效状态", { selector: "b" }));
  await userEvent.click(screen.getByText("查看配置来源与生效状态"));
}

it("opens the existing physical-output confirmation instead of navigating to the same page", async () => {
  vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/config/commands")
    ? Promise.resolve(new Response(JSON.stringify({ code: "TEST_OUTPUT_REJECTED", message: "output rejected by daemon" }), { status: 409, headers: { "content-type": "application/json", "x-request-id": "0123456789abcdef0123456789abcdef" } }))
    : new Promise(() => {})));
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  expect(screen.getByRole("heading", { name: "先调好控制，再决定是否输出" })).toBeVisible();
  expect(screen.getByRole("navigation", { name: "参数分区" })).toHaveTextContent("触发与算法");
  expect(screen.getByText(/保存上面的参数不会改变它/)).toBeVisible();
  await openProfile();
  await userEvent.click(screen.getByRole("button", { name: "打开输出" }));
  expect(screen.getByRole("alertdialog", { name: "允许发送鼠标偏移？" })).toBeVisible();
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/config/commands"))).toBe(false);
  await userEvent.click(screen.getByRole("button", { name: "确认开启输出" }));
  await waitFor(() => expect(screen.getByRole("alertdialog")).toHaveTextContent("output rejected by daemon"));
  const command = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes("/config/commands"))!;
  expect(JSON.parse(String(command[1]?.body))).toMatchObject({ command: "set_output_gate", enabled: true });
  expect(new Headers(command[1]?.headers).get("x-request-id")).toMatch(/^[0-9a-f]{32}$/);
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  await userEvent.click(screen.getByRole("button", { name: /异常信息/ }));
  expect(screen.getByRole("dialog", { name: "异常信息" })).toHaveTextContent("排查编号 0123456789abcdef0123456789abcdef");
});

it("explains missing hardware authorization instead of offering a no-op output action", async () => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} license={{ ...props.license, features: [] }} /></SafetyOperationProvider>);
  await openProfile();
  const card = screen.getByText("物理输出", { selector: ".product-config-profile-item strong" }).closest("li")!;
  expect(card).toHaveTextContent("当前授权不包含硬件控制");
  expect(within(card).queryByRole("button", { name: "打开输出" })).not.toBeInTheDocument();
  expect(within(card).getByRole("button", { name: "查看授权" })).toBeVisible();
});

it.each([{ pixel_format: "NV12" }, { width: 1280 }, { fps: 120 }, { device: "/dev/video1" }])("does not claim capture is effective just because ROI matches (%j)", (changed) => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, capture: { ...props.runtimeConfig.capture, ...changed } }} /></SafetyOperationProvider>);
  const card = screen.getByText("采集规格", { selector: ".product-config-profile-item strong" }).closest("li")!;
  expect(card).not.toHaveTextContent("已生效");
  expect(card).toHaveTextContent("尚未与当前运行规格一致");
});

it("treats MJPEG and MJPG as the same format and shows the exact selected tuple", () => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, capture: { ...props.runtimeConfig.capture, pixel_format: "MJPEG" } }} /></SafetyOperationProvider>);
  const card = screen.getByText("采集规格", { selector: ".product-config-profile-item strong" }).closest("li")!;
  expect(card).toHaveTextContent("已生效");
  expect(card).toHaveTextContent("MJPEG (MJPG) / 1920x1080 / 240 FPS");
});

it("compares saved and running capture tuples on the capture page", () => {
  history.replaceState(null, "", "/?page=capture");
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, capture: { ...props.runtimeConfig.capture, pixel_format: "MJPEG", fps: 120 } }} /></SafetyOperationProvider>);
  const check = screen.getByRole("heading", { name: "保存配置与运行状态" }).closest("section")!;
  expect(check).toHaveTextContent("保存与运行不一致");
  expect(check).toHaveTextContent("MJPEG (MJPG) / 1920x1080 / 120 FPS");
  expect(check).toHaveTextContent("MJPEG (MJPG) / 1920x1080 / 240 FPS");
  expect(check).toHaveTextContent("有效输入 FPS 不是采集卡原始帧率");
});

it("does not claim capture is applied before runtime verification", () => {
  history.replaceState(null, "", "/?page=capture");
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={{ ...runtime, running: false, capture: { ...runtime.capture, running: false } }} /></SafetyOperationProvider>);
  const check = screen.getByRole("heading", { name: "保存配置与运行状态" }).closest("section")!;
  expect(check).toHaveTextContent("等待运行验证");
  expect(check).toHaveTextContent("主链未运行");
  expect(check).not.toHaveTextContent("运行规格一致");
});

it("opens control parameters from the live control chain", async () => {
  history.replaceState(null, "", "/?page=control");
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "前往控制参数" }));
  expect(screen.getByRole("heading", { level: 1, name: "参数设置" })).toBeVisible();
});

it("asks before discarding unsaved parameter edits", async () => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: /目标速度预测/ }));
  await userEvent.click(screen.getByRole("button", { name: "放弃修改" }));
  expect(screen.getByRole("alertdialog", { name: "放弃未保存的修改？" })).toBeVisible();
  expect(screen.getByText("修改尚未保存")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  expect(screen.getByRole("button", { name: "放弃修改" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "放弃修改" }));
  await userEvent.click(screen.getByRole("button", { name: "确认放弃修改" }));
  expect(screen.queryByRole("button", { name: "放弃修改" })).not.toBeInTheDocument();
});

it("keeps unsaved dialog edits when the user cancels closing it", async () => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "权重" }));
  const value = screen.getByRole("slider", { name: "距离权重 滑块" });
  fireEvent.change(value, { target: { value: "0.7" } });
  fireEvent.blur(value);
  await waitFor(() => expect(screen.getByRole("button", { name: "关闭权重调整" })).toHaveAttribute("title", "关闭并放弃本弹窗修改"));
  await userEvent.click(screen.getByRole("button", { name: "关闭权重调整" }));
  expect(screen.getByRole("alertdialog", { name: "关闭并放弃本次修改？" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  expect(screen.getByRole("dialog", { name: "选择权重" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "关闭权重调整" }));
  await userEvent.click(screen.getByRole("button", { name: "放弃弹窗修改" }));
  expect(screen.queryByRole("dialog", { name: "选择权重" })).not.toBeInTheDocument();
});

it("stages a typed selection weight before closing the dialog", async () => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "权重" }));
  const value = screen.getByRole("textbox", { name: "距离权重 数值" });
  fireEvent.change(value, { target: { value: "0.7" } });
  expect(value).toHaveValue("0.7");
  fireEvent.blur(value);
  await waitFor(() => expect(screen.getByRole("button", { name: "关闭权重调整" })).toHaveAttribute("title", "关闭并放弃本弹窗修改"));
});

it("asks before deleting the learned crosshair template", async () => {
  const withTemplate = { ...runtime, vision: { ...runtime.vision, crosshair: { template: { id: "template-1" } } } } as RuntimeState;
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={withTemplate} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "清除模板" }));
  expect(screen.getByRole("alertdialog", { name: "清除已学习的准星模板？" })).toHaveTextContent("template-1");
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("crosshair"))).toBe(false);
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
});

it("asks before sending a physical kmNet diagnostic move", async () => {
  history.replaceState(null, "", "/?page=control-test");
  const stopped = { ...runtime, running: false, semantic: { ...runtime.semantic, phase: "stopped" } } as RuntimeState;
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={stopped} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "发送" }));
  expect(screen.getByRole("alertdialog", { name: "发送一次物理位移？" })).toHaveTextContent("dx=");
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("diagnostic"))).toBe(false);
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  await userEvent.click(screen.getByRole("button", { name: "发送" }));
  await userEvent.click(screen.getByRole("button", { name: "确认发送一次" }));
  await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes("diagnostic-move"))).toHaveLength(1));
  const request = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes("diagnostic-move"))!;
  expect(JSON.parse(String(request[1]?.body))).toMatchObject({ repeat: 1, move_kind: "raw" });
});

it("asks before starting a stopped mainline with physical output enabled", async () => {
  history.replaceState(null, "", "/?page=capture");
  const stopped = { ...runtime, running: false, semantic: { ...runtime.semantic, phase: "stopped" } } as RuntimeState;
  render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} runtimeConfig={{ ...props.runtimeConfig, control: { output_enabled: true } }} /></SafetyOperationProvider>);
  const starts = () => vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes("/api/runtime/start"));
  await userEvent.click(screen.getByRole("button", { name: /^运行$/ }));
  expect(screen.getByRole("alertdialog", { name: "物理输出仍开启，确认启动主链？" })).toHaveTextContent("可能自动连接 kmNet");
  expect(starts()).toHaveLength(0);
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  expect(starts()).toHaveLength(0);
  await userEvent.click(screen.getByRole("button", { name: /^运行$/ }));
  await userEvent.click(screen.getByRole("button", { name: "确认启动并允许物理输出" }));
  expect(starts()).toHaveLength(1);
  expect(new Headers(starts()[0][1]?.headers).get("x-novasight-physical-output-ack")).toBe("confirmed");
});

it("asks before reconnecting kmNet while physical output is enabled", async () => {
  history.replaceState(null, "", "/?page=control-test");
  const disconnected = { ...runtime, executor: { executors: { kmnet: { available: true, connected: false, runtime_connected: false, connection_state: "disconnected", configuration_ready: true, can_connect: true } } } } as unknown as RuntimeState;
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={disconnected} runtimeConfig={{ ...props.runtimeConfig, control: { output_enabled: true } }} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByText("设备连接配置", { selector: "b" }));
  const connects = () => vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes("/api/executors/kmnet/connect"));
  await userEvent.click(screen.getByRole("button", { name: "连接实时会话" }));
  expect(screen.getByRole("alertdialog", { name: "物理输出仍开启，确认连接 kmNet？" })).toHaveTextContent("可能立即发送");
  expect(connects()).toHaveLength(0);
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  expect(connects()).toHaveLength(0);
  await userEvent.click(screen.getByRole("button", { name: "连接实时会话" }));
  await userEvent.click(screen.getByRole("button", { name: "确认连接并允许物理输出" }));
  expect(connects()).toHaveLength(1);
  expect(new Headers(connects()[0][1]?.headers).get("x-novasight-physical-output-ack")).toBe("confirmed");
});

it("shows tracking-budget losses in control diagnostics", async () => {
  history.replaceState(null, "", "/?page=control");
  const counts = { ...runtime.vision.target_pipeline.counts, admitted_to_tracking: 16, dropped_by_budget: 1 };
  const withCounts = { ...runtime, vision: { ...runtime.vision, target_pipeline: { ...runtime.vision.target_pipeline, counts } } };
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={withCounts} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByText("控制排查信息"));
  expect(screen.getByText("进入跟踪 / 预算舍弃").nextElementSibling).toHaveTextContent("16 / 1");
});

it("shows capture selection rejection beside the save control", async () => {
  history.replaceState(null, "", "/?page=capture");
  vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/capture/select")
    ? Promise.resolve(new Response(JSON.stringify({ code: "CAPTURE_PROFILE_UNSUPPORTED", message: "device rejected 240 FPS" }), { status: 422, headers: { "content-type": "application/json" } }))
    : new Promise(() => {})));
  render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={{ ...runtime, running: false, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } }} /></SafetyOperationProvider>);
  const save = screen.getByRole("button", { name: "保存采集配置" });
  expect(save).toBeEnabled();
  await userEvent.click(save);
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/capture/select"))).toBe(true);
  expect(await screen.findByRole("alert")).toHaveTextContent("device rejected 240 FPS");
});

it("requires an explicit capture choice when the saved format is absent from detected capabilities", async () => {
  history.replaceState(null, "", "/?page=capture");
  const stopped = { ...runtime, running: false, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
  vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/capture/capabilities")
    ? Promise.resolve(new Response(JSON.stringify({ available: true, device: "/dev/video0", capabilities: [{ pixel_format: "NV12", width: 1280, height: 720, fps_list: [120] }], reason: "" }), { status: 200, headers: { "content-type": "application/json" } }))
    : new Promise(() => {})));
  render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "检测设备能力" }));
  const format = await screen.findByRole("combobox", { name: "采集格式" });
  expect(format).toHaveValue("");
  expect(screen.getByRole("button", { name: "保存采集配置" })).toBeDisabled();
  await userEvent.selectOptions(format, "NV12:1280x720@120");
  await waitFor(() => expect(screen.getByRole("button", { name: "保存采集配置" })).toBeEnabled());
});

it("does not silently save an auto-high-fps capture profile with no chosen format", () => {
  history.replaceState(null, "", "/?page=capture");
  const stopped = { ...runtime, running: false, capture: { ...runtime.capture, running: false, profile: null }, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
  const unconfigured = { ...props.runtimeConfig, capture: { device: "/dev/video0" } };
  render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} runtimeConfig={unconfigured} /></SafetyOperationProvider>);
  expect(screen.getByRole("combobox", { name: "采集格式" })).toHaveValue("");
  expect(screen.getByRole("button", { name: "保存采集配置" })).toBeDisabled();
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/capture/select"))).toBe(false);
});

it("does not reuse a previous device format when the operator changes capture device", async () => {
  history.replaceState(null, "", "/?page=capture");
  const stopped = { ...runtime, running: false, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
  render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} /></SafetyOperationProvider>);
  const device = screen.getByRole("textbox", { name: "视频设备" });
  await userEvent.clear(device);
  await userEvent.type(device, "/dev/video1");
  await userEvent.tab();
  expect(screen.getByRole("button", { name: "保存采集配置" })).toBeDisabled();
});

it("keeps failed capability detection visible beside capture controls", async () => {
  history.replaceState(null, "", "/?page=capture");
  const stopped = { ...runtime, running: false, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
  vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/capture/capabilities")
    ? Promise.reject(new Error("camera unplugged")) : new Promise(() => {})));
  render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "检测设备能力" }));
  expect(await screen.findByText(/设备能力检测失败：camera unplugged/)).toBeVisible();
});

it("does not let a pending parameter draft get overwritten by config import", async () => {
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: "按键触发" }));
  expect(screen.getByRole("button", { name: "导入" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "导出已保存" })).toBeEnabled();
  expect(screen.getByText(/导出不包含草稿/)).toBeVisible();
});

it("keeps physical output off when an imported file requests it on", async () => {
  const imported = { ...props.runtimeConfig, control: { ...props.runtimeConfig.control, output_enabled: true }, pipeline: { ...props.runtimeConfig.pipeline, target_lost_grace_ms: 99 } };
  let submitted: Record<string, unknown> | null = null;
  vi.stubGlobal("fetch", vi.fn((url, init) => {
    if (String(url).endsWith("/api/config") && (init?.method ?? "GET") === "GET") {
      return Promise.resolve(new Response(JSON.stringify(props.runtimeConfig), { status: 200, headers: { "content-type": "application/json" } }));
    }
    if (String(url).endsWith("/api/config") && init?.method === "POST") {
      submitted = JSON.parse(String(init.body));
      return Promise.resolve(new Response(JSON.stringify({ code: "TEST_REJECTED", message: "probe complete" }), { status: 409, headers: { "content-type": "application/json" } }));
    }
    return new Promise(() => {});
  }));
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  const file = new File([JSON.stringify(imported)], "dangerous.json", { type: "application/json" });
  fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [file] } });
  const confirmation = await screen.findByRole("alertdialog", { name: "应用 dangerous.json？" });
  expect(confirmation).toHaveTextContent("物理输出开关不会随导入文件改变");
  await userEvent.click(within(confirmation).getByRole("button", { name: "确认导入配置" }));
  await waitFor(() => expect(submitted).not.toBeNull());
  expect((submitted as unknown as { control: { output_enabled: boolean } }).control.output_enabled).toBe(false);
});

it("confirms class-profile deletion each time instead of retaining an armed delete", async () => {
  const configured = { ...props.runtimeConfig, inference: {
    detection_class_profile: "default",
    detection_class_profiles: { default: ["enemy"], secondary: ["enemy"] },
  } };
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByText("目标与识别设置"));
  await userEvent.click(screen.getByRole("button", { name: "管理类别配置" }));
  await userEvent.click(screen.getByRole("button", { name: "删除类别配置 default" }));
  expect(screen.getByRole("alertdialog", { name: "删除类别配置“default”？" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  await userEvent.click(screen.getByRole("button", { name: "删除类别配置 default" }));
  expect(screen.getByRole("alertdialog", { name: "删除类别配置“default”？" })).toBeVisible();
});

it("saves a class aim-point edit without reporting unsupported control.aim", async () => {
  const configured = { ...props.runtimeConfig,
    inference: { detection_class_profile: "default", detection_class_profiles: { default: ["enemy"] } },
    control: { ...props.runtimeConfig.control, aim: { class_roles: { default: {} } } },
  };
  let current = structuredClone(configured) as Record<string, unknown>;
  const writes: Array<Record<string, unknown>> = [];
  vi.stubGlobal("fetch", vi.fn((url, init) => {
    if (!String(url).endsWith("/api/config")) return new Promise(() => {});
    if ((init?.method ?? "GET") === "GET") {
      return Promise.resolve(new Response(JSON.stringify(current), { status: 200, headers: { "content-type": "application/json" } }));
    }
    const payload = JSON.parse(String(init?.body)) as Record<string, unknown>;
    writes.push(payload);
    if (typeof payload.section === "string" && typeof payload.key === "string") {
      current = { ...current, [payload.section]: { ...(current[payload.section] as Record<string, unknown>), [payload.key]: payload.value }, revision: Number(current.revision) + 1 };
    } else {
      current = { ...payload, revision: Number(current.revision) + 1 };
    }
    return Promise.resolve(new Response(JSON.stringify({ config: current, apply_mode: payload.section ? "hot_update" : "epoch_reload", restart_required: false, applied: true, rolled_back: false, message: "ok" }), { status: 200, headers: { "content-type": "application/json" } }));
  }));
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByText("目标与识别设置"));
  await userEvent.click(screen.getByRole("button", { name: "管理类别配置" }));
  await userEvent.click(within(screen.getByRole("group", { name: "cls 0 瞄点类型" })).getByRole("button", { name: "头部" }));
  await userEvent.click(screen.getByRole("button", { name: "加入草稿并关闭" }));
  await userEvent.click(screen.getByRole("button", { name: "保存修改" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "保存修改" })).toBeDisabled());
  expect(writes).toHaveLength(1);
  expect(writes[0]).toMatchObject({ control: { output_enabled: false, aim: { class_roles: { default: { "0": "head" } } } } });
});

it("saves a new class profile as one runtime configuration update", async () => {
  const configured = { ...props.runtimeConfig,
    inference: { detection_class_profile: "default", detection_class_profiles: { default: ["enemy"] } },
    control: { ...props.runtimeConfig.control, aim: { class_roles: { default: {} } } },
  };
  let submitted: Record<string, unknown> | null = null;
  vi.stubGlobal("fetch", vi.fn((url, init) => {
    if (!String(url).endsWith("/api/config")) return new Promise(() => {});
    if ((init?.method ?? "GET") === "GET") {
      return Promise.resolve(new Response(JSON.stringify(configured), { status: 200, headers: { "content-type": "application/json" } }));
    }
    submitted = JSON.parse(String(init?.body));
    return Promise.resolve(new Response(JSON.stringify({ config: { ...submitted, revision: 26 }, apply_mode: "epoch_reload", restart_required: false, applied: true, rolled_back: false, message: "ok" }), { status: 200, headers: { "content-type": "application/json" } }));
  }));
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByText("目标与识别设置"));
  await userEvent.click(screen.getByRole("button", { name: "管理类别配置" }));
  await userEvent.type(screen.getByRole("textbox", { name: "新建配置" }), "arena");
  await userEvent.click(screen.getByRole("button", { name: "复制当前" }));
  await userEvent.click(screen.getByRole("button", { name: "加入草稿并关闭" }));
  await userEvent.click(screen.getByRole("button", { name: "保存修改" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "保存修改" })).toBeDisabled());
  expect(submitted).toMatchObject({
    revision: 25,
    inference: { detection_class_profile: "arena", detection_class_profiles: { default: ["enemy"], arena: ["enemy"] } },
    control: { output_enabled: false, aim: { class_roles: expect.any(Object) } },
  });
  expect(vi.mocked(fetch).mock.calls.filter(([url, init]) => String(url).endsWith("/api/config") && init?.method === "POST")).toHaveLength(1);
});

it("retains a rejected class edit and does not claim the draft is empty", async () => {
  const configured = { ...props.runtimeConfig,
    inference: { detection_class_profile: "default", detection_class_profiles: { default: ["enemy"] } },
    control: { ...props.runtimeConfig.control, aim: { class_roles: { default: {} } } },
  };
  vi.stubGlobal("fetch", vi.fn((url, init) => {
    if (!String(url).endsWith("/api/config")) return new Promise(() => {});
    return Promise.resolve((init?.method ?? "GET") === "GET"
      ? new Response(JSON.stringify(configured), { status: 200, headers: { "content-type": "application/json" } })
      : new Response(JSON.stringify({ code: "CLASS_CONFIG_REJECTED", message: "invalid class role" }), { status: 422, headers: { "content-type": "application/json" } }));
  }));
  render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByText("目标与识别设置"));
  await userEvent.click(screen.getByRole("button", { name: "管理类别配置" }));
  await userEvent.click(within(screen.getByRole("group", { name: "cls 0 瞄点类型" })).getByRole("button", { name: "头部" }));
  await userEvent.click(screen.getByRole("button", { name: "加入草稿并关闭" }));
  await userEvent.click(screen.getByRole("button", { name: "保存修改" }));
  await waitFor(() => expect(screen.getByText(/invalid class role/)).toBeVisible());
  expect(screen.getByRole("button", { name: "保存修改" })).toBeEnabled();
  expect(screen.queryByText(/没有剩余未保存修改/)).not.toBeInTheDocument();
});
