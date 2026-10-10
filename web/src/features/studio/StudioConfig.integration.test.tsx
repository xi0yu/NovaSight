import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, beforeEach, expect, it, vi } from "vitest";
import type { LicenseStatus, RuntimeState } from "../../api";
import { getRuntimeConfig } from "../../api";
import { SafetyOperationProvider } from "../runtime/SafetyOperationContext";
import { StudioConsoleView } from "./StudioConsoleView";

const profile = { pixel_format: "MJPG", width: 1920, height: 1080, fps: 240 };
const runtime = {
  semantic: { daemon_instance_id: "test", phase: "running", perception_phase: "running", epoch: 1, snapshot_sequence: 1 },
  presentation: { lifecycle: { can_start: false, can_stop: true }, readiness: { code: "ready", recommended_action: null }, perception: { state: "current" }, output: { state: "blocked", reason_code: "output_disabled", daemon_confirmed_safe: false } },
  running: true, capture: { device: "/dev/video0", running: true, profile },
  statistics: { detection_data_age_ms: 1, detection_freshness_threshold_ms: 55 }, config: { version: 25, effective_version: 25 },
  executor: { executors: { kmnet: { available: true, connected: true, runtime_connected: true, connection_state: "connected" } } },
  pipeline: { running: true, state: "running", deepstream: {} }, inference: {},
  vision: { control: {}, output_trace: {}, target_pipeline: { rejection_reasons: [], counts: {} }, detections: [],
    inference: { roi_offset_x: 0, roi_offset_y: 0, roi_width: 256, roi_height: 256 } },
} as unknown as RuntimeState;
const stoppedPresentation: RuntimeState["presentation"] = {
  lifecycle: { can_start: true, can_stop: false },
  readiness: { code: "stopped", recommended_action: null },
  perception: { state: "stopped" },
  output: { state: "safe", reason_code: "runtime_stopped", daemon_confirmed_safe: true },
};
const props = {
  license: { valid: true, features: ["hardware_control"] } as LicenseStatus,
  health: null, runtime,
  runtimeConfig: { revision: 25, capture: { ...profile, device: "/dev/video0", roi_left: 0, roi_top: 0, roi_width: 256, roi_height: 256 },
    pipeline: { projection_fov_x_deg: 90 }, control: { output_enabled: false }, hardware: { auto_connect: true, host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4", monitor_port: 5001 } },
  projects: [], errors: {}, lastUpdated: null, realtimeStatus: "connected" as const,
  onEnsureProjects: vi.fn(async () => []), onLicenseChange: vi.fn(), onRefresh: vi.fn(async () => {}),
  onRuntimeConfigChange: vi.fn(), onRuntimeStateChange: vi.fn(() => true),
};
type ParameterWrite = { expected_revision: number; changes: Array<{ section: string; key: string; value: unknown }> };
function mockParameterWrites(initial: typeof props.runtimeConfig | Record<string, unknown>) {
  let current = structuredClone(initial) as Record<string, unknown>;
  const writes: ParameterWrite[] = [];
  vi.stubGlobal("fetch", vi.fn((url, init) => {
    if (String(url).endsWith("/api/config") && (init?.method ?? "GET") === "GET") return Promise.resolve(new Response(JSON.stringify(current), { headers: { "content-type": "application/json" } }));
    if (!String(url).endsWith("/api/v1/config/parameters")) return new Promise(() => {});
    const payload = JSON.parse(String(init?.body)) as ParameterWrite;
    writes.push(payload);
    expect(payload.expected_revision).toBe(current.revision);
    current = structuredClone(current);
    for (const { section, key, value } of payload.changes) {
      const record = (current[section] ??= {}) as Record<string, unknown>;
      if (value === null) delete record[key]; else record[key] = value;
    }
    current.revision = Number(current.revision) + 1;
    return Promise.resolve(new Response(JSON.stringify({ config: current, apply_mode: "hot_update", restart_required: false, applied: true, rolled_back: false, message: "ok" }), { headers: { "content-type": "application/json" } }));
  }));
  return writes;
}
beforeEach(() => {
  history.replaceState(null, "", "/?page=params");
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
});
afterEach(() => { history.replaceState(null, "", "/"); });
describe("Capture selection and saved versus running evidence", () => {
  it.each([{ pixel_format: "NV12" }, { width: 1280 }, { fps: 120 }, { device: "/dev/video1" }])("does not claim capture is effective just because ROI matches (%j)", (changed) => {
    history.replaceState(null, "", "/?page=capture");
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, capture: { ...props.runtimeConfig.capture, ...changed } }} /></SafetyOperationProvider>);
    const check = screen.getByText("当前画面输入").closest("header")!;
    expect(check).toHaveTextContent("保存与运行不一致");
  });

  it("does not claim capture is applied before runtime verification", () => {
    history.replaceState(null, "", "/?page=capture");
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={{ ...runtime, running: false, capture: { ...runtime.capture, running: false } }} /></SafetyOperationProvider>);
    const check = screen.getByText("当前画面输入").closest("header")!;
    expect(check).toHaveTextContent("等待运行验证");
    expect(check).toHaveTextContent("主链未运行");
    expect(check).not.toHaveTextContent("运行规格一致");
  });

  it("does not invent /dev/video0 when capture has not been configured", () => {
    history.replaceState(null, "", "/?page=capture");
    const { capture: _capture, ...unconfigured } = props.runtimeConfig;
    const unconfiguredRuntime = { ...runtime, capture: { available: false, device: "", running: false, profile: null } } as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={unconfiguredRuntime} runtimeConfig={unconfigured} /></SafetyOperationProvider>);

    expect(screen.queryByText("/dev/video0")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("输入其他设备路径"));
    expect(screen.getByRole("textbox", { name: "设备路径" })).toHaveValue("");
    expect(screen.getByRole("heading", { name: "尚未配置采集设备" })).toBeVisible();
    expect(screen.getByRole("button", { name: "读取设备规格" })).toBeDisabled();
  });

  it("keeps capture runtime unknown when only saved device settings are available", () => {
    history.replaceState(null, "", "/?page=capture");
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={null} /></SafetyOperationProvider>);
    const header = screen.getByText("当前画面输入").closest("header")!;
    expect(header).toHaveTextContent("运行状态未确认");
    expect(header).not.toHaveTextContent("主链未运行");
  });

  it("asks before reloading ROI while physical output is enabled", async () => {
    history.replaceState(null, "", "/?page=capture");
    vi.stubGlobal("fetch", vi.fn((_url, init) => init?.method === "POST"
      ? Promise.resolve(new Response("blocked", { status: 428 }))
      : new Promise(() => {})));
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, control: { output_enabled: true } }} /></SafetyOperationProvider>);
    const roi = screen.getByRole("region", { name: "识别范围" });
    const writes = () => vi.mocked(fetch).mock.calls.filter(([url, init]) => String(url).endsWith("/api/config") && init?.method === "POST");
    await userEvent.click(within(roi).getByRole("button", { name: /320/ }));
    expect(screen.getByRole("alertdialog", { name: "物理输出仍开启，确认调整 ROI？" })).toBeVisible();
    expect(writes()).toHaveLength(0);
    await userEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(writes()).toHaveLength(0);
    await userEvent.click(within(roi).getByRole("button", { name: /320/ }));
    await userEvent.click(screen.getByRole("button", { name: "确认重载并保持输出开启" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(new Headers(writes()[0][1]?.headers).get("x-novasight-physical-output-ack")).toBe("confirmed");
  });

  it("shows capture selection rejection beside the clicked frame rate", async () => {
    history.replaceState(null, "", "/?page=capture");
    vi.stubGlobal("fetch", vi.fn((url) => {
      if (String(url).includes("/capture/capabilities")) return Promise.resolve(new Response(JSON.stringify({ available: true, device: "/dev/video0", capabilities: [{ pixel_format: "MJPG", width: 1920, height: 1080, fps_list: [240] }], reason: "" }), { status: 200, headers: { "content-type": "application/json" } }));
      if (String(url).includes("/capture/select")) return Promise.resolve(new Response(JSON.stringify({ code: "CAPTURE_PROFILE_UNSUPPORTED", message: "device rejected 240 FPS" }), { status: 422, headers: { "content-type": "application/json" } }));
      return new Promise(() => {});
    }));
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={{ ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } }} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByText("输入其他设备路径"));
    await userEvent.click(screen.getByRole("button", { name: "读取设备规格" }));
    await userEvent.click(await screen.findByRole("button", { name: "240 FPS" }));
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/capture/select"))).toBe(true);
    expect(await screen.findByRole("alert")).toHaveTextContent("device rejected 240 FPS");
  });

  it("requires an explicit capture choice when the saved format is absent from detected capabilities", async () => {
    history.replaceState(null, "", "/?page=capture");
    const stopped = { ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
    vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/capture/capabilities")
      ? Promise.resolve(new Response(JSON.stringify({ available: true, device: "/dev/video0", capabilities: [{ pixel_format: "NV12", width: 1280, height: 720, fps_list: [120] }], reason: "" }), { status: 200, headers: { "content-type": "application/json" } }))
      : new Promise(() => {})));
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByText("输入其他设备路径"));
    await userEvent.click(screen.getByRole("button", { name: "读取设备规格" }));
    const fps = await screen.findByRole("button", { name: "120 FPS" });
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/capture/select"))).toBe(false);
    await userEvent.click(fps);
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/capture/select"))).toBe(true);
  });

  it("does not silently save an auto-high-fps capture profile with no chosen format", () => {
    history.replaceState(null, "", "/?page=capture");
    const stopped = { ...runtime, running: false, capture: { ...runtime.capture, running: false, profile: null }, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
    const unconfigured = { ...props.runtimeConfig, capture: { device: "/dev/video0" } };
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} runtimeConfig={unconfigured} /></SafetyOperationProvider>);
    expect(screen.getByLabelText("分辨率与帧率")).toHaveTextContent("尚未查询设备规格");
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/capture/select"))).toBe(false);
  });

  it("does not reuse a previous device format when the operator changes capture device", async () => {
    history.replaceState(null, "", "/?page=capture");
    const stopped = { ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByText("输入其他设备路径"));
    const device = screen.getByRole("textbox", { name: "设备路径" });
    await userEvent.clear(device);
    await userEvent.type(device, "/dev/video1");
    await userEvent.tab();
    expect(screen.getByLabelText("分辨率与帧率")).toHaveTextContent("尚未查询设备规格");
  });

  it.each([false, true])("keeps capability failure visible and blocks old choices when previous data exists: %s", async (previouslyRead) => {
    history.replaceState(null, "", "/?page=capture");
    const stopped = { ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" }, pipeline: { ...runtime.pipeline, state: "stopped" } } as RuntimeState;
    let available = previouslyRead;
    vi.stubGlobal("fetch", vi.fn((url) => {
      if (!String(url).includes("/capture/capabilities")) return new Promise(() => {});
      return available ? Promise.resolve(new Response(JSON.stringify({ available: true, device: "/dev/video0", capabilities: [{ pixel_format: "MJPG", width: 1920, height: 1080, fps_list: [240] }], reason: "" }), { headers: { "content-type": "application/json" } })) : Promise.reject(new Error("camera unplugged"));
    }));
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByText("输入其他设备路径"));
    await userEvent.click(screen.getByRole("button", { name: "读取设备规格" }));
    if (previouslyRead) {
      const fps = await screen.findByRole("button", { name: "240 FPS" });
      expect(fps).toBeEnabled();
      available = false;
      await userEvent.click(screen.getByRole("button", { name: "读取设备规格" }));
      await waitFor(() => expect(fps).toBeDisabled());
      expect(screen.getByRole("button", { name: "240 FPS" })).toBe(fps);
    }
    expect(await screen.findByText(/设备能力检测失败：camera unplugged/)).toBeVisible();
  });
});

describe("Runtime control and physical-output consent", () => {
  it("runs the home-page master switch immediately and reports a rejected output gate", async () => {
    history.replaceState(null, "", "/?page=overview");
    const stoppedRuntime = {
      ...runtime,
      semantic: { ...runtime.semantic, phase: "stopped" },
      presentation: stoppedPresentation,
      running: false,
      capture: { ...runtime.capture, running: false },
      pipeline: { ...runtime.pipeline, running: false, state: "stopped" },
    } as unknown as RuntimeState;
    vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/config/commands")
      ? Promise.resolve(new Response(JSON.stringify({ code: "TEST_OUTPUT_REJECTED", message: "output rejected by daemon" }), { status: 409, headers: { "content-type": "application/json", "x-request-id": "0123456789abcdef0123456789abcdef" } }))
      : new Promise(() => {})));
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stoppedRuntime} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("switch", { name: /运行总开关/ }));
    expect(screen.queryByRole("alertdialog", { name: "开启 NovaSight？" })).not.toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/config/commands"))).toBe(true));
    const command = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes("/config/commands"))!;
    expect(JSON.parse(String(command[1]?.body))).toMatchObject({ command: "set_control_enabled", enabled: true });
    expect(new Headers(command[1]?.headers).get("x-novasight-physical-output-ack")).toBe("confirmed");
    expect(new Headers(command[1]?.headers).get("x-request-id")).toMatch(/^[0-9a-f]{32}$/);
    await userEvent.click(screen.getByRole("button", { name: /异常信息/ }));
    expect(screen.getByRole("dialog", { name: "异常信息" })).toHaveTextContent("排查编号 0123456789abcdef0123456789abcdef");
  });

  it("routes an unlicensed runtime request directly to authorization", async () => {
    history.replaceState(null, "", "/?page=overview");
    const stoppedRuntime = {
      ...runtime,
      semantic: { ...runtime.semantic, phase: "stopped" },
      presentation: stoppedPresentation,
      running: false,
      capture: { ...runtime.capture, running: false },
      pipeline: { ...runtime.pipeline, running: false, state: "stopped" },
    } as unknown as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stoppedRuntime} license={{ ...props.license, features: [] }} /></SafetyOperationProvider>);
    const control = screen.getByRole("switch", { name: /运行总开关/ });
    expect(control).toBeEnabled();
    await userEvent.click(control);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "关于 NovaSight" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "设备授权" })).toBeVisible();
    expect(location.search).toBe("?page=overview");
  });

  it("asks before sending a physical kmNet diagnostic move", async () => {
    history.replaceState(null, "", "/?page=control-test");
    const stopped = { ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" } } as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={stopped} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
    await userEvent.click(screen.getByText("高级：受控移动与运行诊断"));
    await userEvent.click(screen.getByRole("button", { name: "发送一次测试移动" }));
    expect(screen.getByRole("alertdialog", { name: "发送一次物理位移？" })).toHaveTextContent("dx=");
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("diagnostic"))).toBe(false);
    await waitFor(() => expect(screen.getByRole("alertdialog", { name: "发送一次物理位移？" })).toHaveFocus());
    await userEvent.keyboard("{Escape}");
    expect(screen.getByRole("dialog", { name: "配置协议" })).toBeVisible();
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "发送一次测试移动" }));
    await userEvent.click(screen.getByRole("button", { name: "确认发送一次" }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes("diagnostic-move"))).toHaveLength(1));
    const request = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes("diagnostic-move"))!;
    expect(JSON.parse(String(request[1]?.body))).toMatchObject({ repeat: 1, move_kind: "raw" });
    expect(new Headers(request[1]?.headers).get("X-NovaSight-Physical-Output-Ack")).toBe("confirmed");
  });

  it("starts a stopped mainline immediately with explicit physical output acknowledgement", async () => {
    history.replaceState(null, "", "/?page=overview");
    const stopped = { ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" } } as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} health={{ ok: true }} runtime={stopped} runtimeConfig={{ ...props.runtimeConfig, control: { output_enabled: true } }} /></SafetyOperationProvider>);
    const starts = () => vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes("/api/studio/v1/runtime"));
    await userEvent.click(screen.getByRole("switch", { name: /运行总开关/ }));
    expect(starts()).toHaveLength(1);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(JSON.parse(String(starts()[0][1]?.body))).toEqual({
      desired_state: "running",
      acknowledge_physical_output: true,
    });
  });

  it("starts the empty protocol without a physical license, address, or kmBox connection", async () => {
    history.replaceState(null, "", "/?page=overview");
    const stopped = { ...runtime, running: false, presentation: stoppedPresentation, semantic: { ...runtime.semantic, phase: "stopped" } } as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} license={{ valid: true, features: ["runtime"] } as LicenseStatus} health={{ ok: true }} runtime={stopped}
      runtimeConfig={{ ...props.runtimeConfig, hardware: { backend: "null", auto_connect: true, active_mouse_protocol_id: "empty", mouse_protocols: [] }, control: { output_enabled: true } }} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("switch", { name: /运行总开关/ }));
    expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes("/api/studio/v1/runtime"))).toHaveLength(1);
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/kmnet/connect"))).toBe(false);
    expect(screen.queryByText("请先设置控制设备地址和设备编号。")).not.toBeInTheDocument();
  });

  it("keeps the live kmNet session under the home runtime switch", async () => {
    history.replaceState(null, "", "/?page=control-test");
    const disconnected = { ...runtime, executor: { executors: { kmnet: { available: true, connected: false, runtime_connected: false, connection_state: "disconnected", configuration_ready: true, can_connect: true } } } } as unknown as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={disconnected} runtimeConfig={{ ...props.runtimeConfig, control: { output_enabled: true } }} /></SafetyOperationProvider>);
    expect(screen.getByRole("button", { name: /kmBox Net ·/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
    expect(screen.getByRole("dialog", { name: "查看协议" })).toHaveTextContent("主链运行中");
    expect(screen.getByRole("button", { name: "测试连通性" })).toBeDisabled();
    expect(screen.queryByRole("dialog", { name: "配置协议" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "连接实时会话" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "断开实时会话" })).not.toBeInTheDocument();
  });
});

describe("Algorithm editing, autosave and draft preservation", () => {
  it("treats MJPEG and MJPG as the same format and shows the exact selected tuple", () => {
    history.replaceState(null, "", "/?page=capture");
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, capture: { ...props.runtimeConfig.capture, pixel_format: "MJPEG" } }} /></SafetyOperationProvider>);
    const check = screen.getByText("当前画面输入").closest("header")!;
    expect(check).toHaveTextContent("运行规格一致");
    expect(check).toHaveTextContent("MJPEG (MJPG) / 1920x1080 / 240 FPS");
  });

  it("opens control parameters from the live control chain", async () => {
    history.replaceState(null, "", "/?page=control");
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("button", { name: "前往控制参数" }));
    expect(screen.getByRole("heading", { level: 1, name: "算法参数" })).toBeVisible();
  });

  it("withdraws a queued parameter edit without a confirmation dialog", async () => {
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    await userEvent.click(screen.getByRole("button", { name: /启用运动预测/ }));
    await userEvent.click(screen.getByRole("button", { name: "撤销未保存修改" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "撤销未保存修改" })).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("config/parameters"))).toBe(false);
  });

  it("automatically saves a response gain after editing, without starting or stopping runtime", async () => {
    const writes = mockParameterWrites(props.runtimeConfig);
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const value = screen.getByRole("slider", { name: "比例增益 Kp 滑块" });
    fireEvent.change(value, { target: { value: "0.7" } });
    expect(writes).toHaveLength(0);
    fireEvent.blur(value);
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0]).toMatchObject({ expected_revision: 25, changes: [{ section: "pipeline", key: "p_response_scale", value: 0.7 }] });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([url]) => /runtime\/(start|stop)/.test(String(url)))).toBe(false);
  });

  it("shows one parameter workspace at a time and exposes entry ramp tuning", async () => {
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    expect(screen.getByRole("region", { name: "搜索范围调校" })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "力度渐增时长 数值" })).toHaveValue("200");
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    expect(screen.queryByRole("region", { name: "搜索范围调校" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "力度渐增时长 数值" })).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "比例增益 Kp 数值" })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "单次 X 轴输出上限 数值" })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "单次 Y 轴输出上限 数值" })).toBeVisible();
    expect(screen.queryByText("预测位移上限")).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "力度基准频率 数值" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "响应试算" })).not.toBeInTheDocument();
    expect(screen.queryByText("远距离追赶力度")).not.toBeInTheDocument();
    expect(screen.queryByText("加速介入时机")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "目标与瞄点" }));
    expect(screen.getByRole("button", { name: "编辑目标类别" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "力度渐增时长 数值" })).not.toBeInTheDocument();
    for (const label of ["距离权重 数值", "类别偏好权重 数值", "置信度权重 数值", "控制目标最低置信度 数值", "目标切换确认延迟 数值", "目标丢失保持 数值"]) {
      expect(screen.queryByRole("textbox", { name: label })).not.toBeInTheDocument();
    }
  });

  it("shows tracking-budget losses in control diagnostics", async () => {
    history.replaceState(null, "", "/?page=control");
    const counts = { ...runtime.vision.target_pipeline.counts, admitted_to_tracking: 16, dropped_by_budget: 1 };
    const withCounts = { ...runtime, vision: { ...runtime.vision, target_pipeline: { ...runtime.vision.target_pipeline, counts } } };
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={withCounts} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByText("控制排查信息"));
    expect(screen.getByText("进入跟踪 / 预算舍弃").nextElementSibling).toHaveTextContent("16 / 1");
  });

  it("does not let a pending parameter draft get overwritten by config import", async () => {
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    const delay = screen.getByRole("textbox", { name: "触发后等待时长 数值" });
    fireEvent.change(delay, { target: { value: "25" } });
    fireEvent.blur(delay);
    await userEvent.click(screen.getByRole("button", { name: "设置" }));
    expect(screen.getByRole("button", { name: "先处理参数" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "下载备份" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "选择备份" })).not.toBeInTheDocument();
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
    await userEvent.click(screen.getByRole("button", { name: "设置" }));
    const file = new File([JSON.stringify(imported)], "dangerous.json", { type: "application/json" });
    fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [file] } });
    const confirmation = await screen.findByRole("alertdialog", { name: "应用 dangerous.json？" });
    expect(confirmation).toHaveTextContent("导入不会改变首页运行总开关");
    expect(within(confirmation).getByRole("table")).toHaveTextContent("pipeline.target_lost_grace_ms");
    expect(within(confirmation).getByRole("table")).toHaveTextContent("99");
    await waitFor(() => expect(confirmation).toHaveFocus());
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog", { name: "应用 dangerous.json？" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "设置" })).toBeVisible();
    fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [file] } });
    const renewedConfirmation = await screen.findByRole("alertdialog", { name: "应用 dangerous.json？" });
    await userEvent.click(within(renewedConfirmation).getByRole("button", { name: "确认导入配置" }));
    await waitFor(() => expect(submitted).not.toBeNull());
    expect((submitted as unknown as { control: { output_enabled: boolean } }).control.output_enabled).toBe(false);
    const request = vi.mocked(fetch).mock.calls.find(([url, init]) => String(url).endsWith("/api/config") && init?.method === "POST")!;
    expect(new Headers(request[1]?.headers).get("x-novasight-physical-output-ack")).toBe("confirmed");
  });

  it("saves a disabled entry ramp as zero and restores its duration when toggled before saving", async () => {
    const configured = {
      ...props.runtimeConfig,
      control: { ...props.runtimeConfig.control, trigger_mode: "always" },
      pipeline: { ...props.runtimeConfig.pipeline, target_selection_distance_weight: 0.2, entry_ramp_ms: 0 },
    };
    const submitted = mockParameterWrites(configured);
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);

    const triggerDelay = screen.getByRole("textbox", { name: "触发后等待时长 数值" });
    fireEvent.change(triggerDelay, { target: { value: "35" } });
    fireEvent.blur(triggerDelay);
    const ramp = screen.getByRole("region", { name: "力度渐增设置" });
    expect(within(ramp).getByRole("button", { name: /启用力度渐增/ })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("img", { name: /等待 35 ms，渐增 0 ms，35 ms 后/ })).toBeVisible();
    expect(within(ramp).queryByText("超推荐")).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "力度渐增时长 数值" })).not.toBeInTheDocument();
    await userEvent.click(within(ramp).getByRole("button", { name: /启用力度渐增/ }));
    const rampTime = screen.getByRole("textbox", { name: "力度渐增时长 数值" });
    expect(rampTime).toHaveValue("200");
    fireEvent.change(rampTime, { target: { value: "350" } });
    fireEvent.blur(rampTime);
    await waitFor(() => expect(screen.getByRole("img", { name: /等待 35 ms，渐增 350 ms，385 ms 后/ })).toBeVisible());
    await userEvent.click(within(ramp).getByRole("button", { name: /启用力度渐增/ }));
    await userEvent.click(within(ramp).getByRole("button", { name: /启用力度渐增/ }));
    expect(screen.getByRole("textbox", { name: "力度渐增时长 数值" })).toHaveValue("350");
    await userEvent.click(within(ramp).getByRole("button", { name: /启用力度渐增/ }));
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const distanceWeight = screen.getByRole("textbox", { name: "比例增益 Kp 数值" });
    fireEvent.change(distanceWeight, { target: { value: "0.7" } });
    fireEvent.blur(distanceWeight);
    await waitFor(() => expect(submitted.length).toBeGreaterThan(0));
    const changes = submitted.flatMap((write) => write.changes);
    expect(changes).toEqual(expect.arrayContaining([
      { section: "pipeline", key: "fire_delay_enabled", value: true },
      { section: "pipeline", key: "fire_delay_ms", value: 35 },
      { section: "pipeline", key: "p_response_scale", value: 0.7 },
    ]));
    expect(changes.some((change) => change.key === "target_selection_distance_weight")).toBe(false);
    expect(screen.queryByRole("button", { name: "按键触发" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "保存并应用" })).not.toBeInTheDocument();
  });

  it("keeps the whole parameter draft when the atomic save is rejected", async () => {
    const configured = {
      ...props.runtimeConfig,
      pipeline: { ...props.runtimeConfig.pipeline, target_selection_distance_weight: 0.2 },
    };
    vi.stubGlobal("fetch", vi.fn((url, init) => {
      if (!String(url).endsWith("/api/v1/config/parameters")) return new Promise(() => {});
      if ((init?.method ?? "GET") === "GET") {
        return Promise.resolve(new Response(JSON.stringify(configured), {
          status: 200,
          headers: { "content-type": "application/json" },
        }));
      }
      return Promise.resolve(new Response(JSON.stringify({
        code: "CONFIG_REJECTED",
        message: "参数组合无效",
      }), { status: 409, headers: { "content-type": "application/json" } }));
    }));
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);

    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const distanceWeight = screen.getByRole("textbox", { name: "比例增益 Kp 数值" });
    fireEvent.change(distanceWeight, { target: { value: "0.7" } });
    fireEvent.blur(distanceWeight);

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("参数组合无效"));
    expect(distanceWeight).toHaveValue("0.700");
    expect(screen.getByText("自动保存失败，修改已保留")).toBeVisible();
    expect(screen.getByRole("button", { name: "重试保存" })).toBeEnabled();
  });

  it("rebases edits made while autosave is in flight using the returned revision", async () => {
    const writes: ParameterWrite[] = [];
    let completeFirst: (response: Response) => void = () => {};
    vi.stubGlobal("fetch", vi.fn((url, init) => {
      if (!String(url).endsWith("/api/v1/config/parameters")) return new Promise(() => {});
      writes.push(JSON.parse(String(init?.body)) as ParameterWrite);
      if (writes.length === 1) return new Promise<Response>((resolve) => { completeFirst = resolve; });
      return Promise.resolve(new Response(JSON.stringify({ config: { ...props.runtimeConfig, revision: 27, pipeline: { ...props.runtimeConfig.pipeline, p_response_scale: 0.8 } }, apply_mode: "hot_update", restart_required: false, applied: true, rolled_back: false, message: "ok" }), { headers: { "content-type": "application/json" } }));
    }));
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const value = screen.getByRole("textbox", { name: "比例增益 Kp 数值" });
    fireEvent.change(value, { target: { value: "0.7" } }); fireEvent.blur(value);
    await waitFor(() => expect(writes).toHaveLength(1));
    fireEvent.change(value, { target: { value: "0.8" } }); fireEvent.blur(value);
    completeFirst(new Response(JSON.stringify({ config: { ...props.runtimeConfig, revision: 26, pipeline: { ...props.runtimeConfig.pipeline, p_response_scale: 0.7 } }, apply_mode: "hot_update", restart_required: false, applied: true, rolled_back: false, message: "ok" }), { headers: { "content-type": "application/json" } }));
    await waitFor(() => expect(writes).toHaveLength(2));
    expect(writes[1]).toEqual({ expected_revision: 26, changes: [{ section: "pipeline", key: "p_response_scale", value: 0.8 }] });
    expect(value).toHaveValue("0.800");
  });

  it("does not claim a file-only response applied runtime parameters", async () => {
    vi.stubGlobal("fetch", vi.fn((url) => String(url).endsWith("/api/v1/config/parameters")
      ? Promise.resolve(new Response(JSON.stringify({ config: { ...props.runtimeConfig, revision: 26 }, apply_mode: "process_restart", restart_required: true, applied: false, rolled_back: false, message: "restart required" }), { headers: { "content-type": "application/json" } }))
      : new Promise(() => {})));
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const value = screen.getByRole("textbox", { name: "比例增益 Kp 数值" });
    fireEvent.change(value, { target: { value: "0.7" } }); fireEvent.blur(value);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("参数热更新"));
    expect(value).toHaveValue("0.700");
    expect(screen.getByRole("button", { name: "重试保存" })).toBeEnabled();
  });

  it("does not claim fire stabilization enabled before autosave is confirmed", async () => {
    history.replaceState(null, "", "/?page=fire");
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={{ ...props.runtimeConfig, control: { output_enabled: false, fire_stabilization: { enabled: false, strength: 0.5 } } }} /></SafetyOperationProvider>);
    const toggle = screen.getByRole("button", { name: /开火时稳定/ });
    await userEvent.click(toggle);
    expect(toggle).toHaveTextContent("等待自动保存");
    expect(toggle).not.toHaveTextContent("已开启");
  });

  it("restores saved Kp and fire strength when a fresh page reads the configuration", async () => {
    const writes = mockParameterWrites(props.runtimeConfig);
    const view = render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const kp = screen.getByRole("textbox", { name: "比例增益 Kp 数值" });
    fireEvent.change(kp, { target: { value: "0.7" } });
    fireEvent.blur(kp);
    await waitFor(() => expect(writes).toHaveLength(1));
    await userEvent.click(screen.getByRole("button", { name: "开火稳定" }));
    const strength = screen.getByRole("textbox", { name: "抑制强度 数值" });
    fireEvent.change(strength, { target: { value: "0.8" } });
    fireEvent.blur(strength);
    await waitFor(() => expect(writes).toHaveLength(2));
    await waitFor(() => expect(screen.queryByText("正在自动保存并应用")).not.toBeInTheDocument());
    const saved = await getRuntimeConfig();
    view.unmount();
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={saved} /></SafetyOperationProvider>);
    expect(screen.getByRole("textbox", { name: "抑制强度 数值" })).toHaveValue("0.80");
    await userEvent.click(screen.getByRole("button", { name: "算法参数" }));
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    expect(screen.getByRole("textbox", { name: "比例增益 Kp 数值" })).toHaveValue("0.700");
    expect(writes).toHaveLength(2);
  });
});

describe("Model metadata and target-category configuration", () => {
  it("asks before metadata saving registers an unregistered Engine and locks its path", async () => {
    history.replaceState(null, "", "/?page=models");
    const catalog = {
      root: { type: "directory", name: "models", relative_path: "", children: [{
        type: "model", name: "draft.engine", relative_path: "draft.engine", kind: "engine", size_bytes: 1024,
        scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [],
      }] },
      directory_count: 1, model_count: 1, discovered_files: 0, updated_files: 0, cache_hits: 1, force: false,
    };
    vi.stubGlobal("fetch", vi.fn((url) => String(url).includes("/api/models/catalog?force=false")
      ? Promise.resolve(new Response(JSON.stringify(catalog), { headers: { "content-type": "application/json" } }))
      : new Promise(() => {})));
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(await screen.findByRole("button", { name: /draft.engine，路径 draft.engine/ }));
    expect(screen.getByText(/登记后不能直接移动或改名/)).toBeVisible();
    await userEvent.click(within(screen.getByRole("group", { name: "模型推荐状态" })).getByRole("button", { name: "推荐" }));
    await userEvent.click(screen.getByRole("button", { name: "首页" }));
    expect(screen.getByRole("alertdialog", { name: "放弃未保存的模型标签？" })).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(screen.getByRole("group", { name: "模型推荐状态" })).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "保存整理结果" }));
    expect(screen.getByRole("alertdialog", { name: "登记后保存模型标签？" })).toHaveTextContent("draft.engine");
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/api/models/catalog/register"))).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(screen.getByRole("button", { name: "保存整理结果" })).toBeEnabled();
    await userEvent.click(screen.getByRole("button", { name: "保存整理结果" }));
    await userEvent.click(screen.getByRole("button", { name: "登记并保存" }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/api/models/catalog/register"))).toBe(true));
  });

  it("uses registered model class names only as display hints", async () => {
    const runtimeWithModel = { ...runtime, active_model: {
      project: { id: 1, name: "演示模型", description: "" },
      version: { id: 1, project_id: 1, version: "1", source_kind: "engine", source_path: "", classes: ["头部", "身体"], input_shape: "640x640" },
      artifact: { id: 1, version_id: 1, kind: "engine", path: "", checksum: "", status: "ready", size_bytes: null },
      deployment: { id: 1, project_id: 1, artifact_id: 1, previous_artifact_id: null, updated_seq: 1 },
      artifact_path: "",
    } } as RuntimeState;
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtime={runtimeWithModel} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "目标与瞄点" }));
    const overview = screen.getByRole("list", { name: "当前类别配置" });
    expect(overview).toHaveTextContent("头部");
    expect(overview).toHaveTextContent("身体");
    expect(overview).not.toHaveTextContent("未命名类别");
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    const name = within(screen.getByRole("dialog", { name: "编辑目标类别" })).getByRole("textbox", { name: "cls 0 名称" });
    expect(name).toHaveValue("");
    expect(name).toHaveAttribute("placeholder", "模型登记：头部");
  });

  it("shows one target-class configuration without profile management", async () => {
    const configured = { ...props.runtimeConfig, inference: {
      detection_class_profile: "default",
      detection_class_profiles: { default: ["enemy"], secondary: ["enemy"] },
    } };
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "目标与瞄点" }));
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    const dialog = screen.getByRole("dialog", { name: "编辑目标类别" });
    expect(within(dialog).queryByText("配置文件")).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: /删除类别配置/ })).not.toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole("button", { name: "cls 0 中心" }));
    const changedName = within(dialog).getByRole("textbox", { name: "cls 0 名称" });
    fireEvent.change(changedName, { target: { value: "changed" } }); fireEvent.blur(changedName);
    const changedWeight = within(dialog).getByRole("slider", { name: "cls 0 类别偏好 滑块" });
    fireEvent.change(changedWeight, { target: { value: "0.9" } }); fireEvent.blur(changedWeight);
    await userEvent.click(within(dialog).getByRole("checkbox", { name: "cls 0 参与目标选择" }));
    const unload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unload);
    expect(unload.defaultPrevented).toBe(true);
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "撤销未保存修改" })).toBeVisible();
    const navigation = screen.getByRole("navigation", { name: "NovaSight Studio 导航" });
    await userEvent.click(within(navigation).getByRole("button", { name: "首页" }));
    await userEvent.click(within(navigation).getByRole("button", { name: "算法参数" }));
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    expect(screen.getByRole("textbox", { name: "cls 0 垂直位置 数值" })).toHaveValue("50");
    await userEvent.click(screen.getByRole("button", { name: "cls 1 下部" }));
    await userEvent.click(screen.getByRole("button", { name: "撤销 cls 0 修改" }));
    expect(screen.queryByRole("status", { name: "cls 0 已修改" })).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "cls 0 垂直位置 数值" })).toHaveValue("22");
    expect(screen.getByRole("textbox", { name: "cls 0 名称" })).toHaveValue("enemy");
    expect(screen.getByRole("checkbox", { name: "cls 0 参与目标选择" })).toBeChecked();
    expect(screen.getByRole("textbox", { name: "cls 1 垂直位置 数值" })).toHaveValue("75");
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("saves a class aim-point edit without reporting unsupported control.aim", async () => {
    const configured = { ...props.runtimeConfig,
      inference: { detection_class_profile: "default", detection_class_profiles: { default: ["enemy"] } },
      control: { ...props.runtimeConfig.control, aim: { class_roles: { default: {} } } },
    };
    const writes = mockParameterWrites(configured);
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "目标与瞄点" }));
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    const dialog = screen.getByRole("dialog", { name: "编辑目标类别" });
    await userEvent.click(within(dialog).getByRole("button", { name: "cls 0 中心" }));
    await userEvent.click(within(dialog).getByRole("button", { name: "完成编辑" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "编辑目标类别" })).not.toBeInTheDocument());
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0].changes).toEqual(expect.arrayContaining([{ section: "pipeline", key: "target_class_aim_y_ratios", value: "0:0.5" }]));
    expect(writes[0].changes).not.toEqual(expect.arrayContaining([expect.objectContaining({ key: "output_enabled" })]));
  });

  it("collapses legacy class profiles into one configuration when saving", async () => {
    const configured = { ...props.runtimeConfig,
      inference: {
        detection_class_profile: "arena",
        detection_class_profiles: { default: ["enemy"], arena: ["target"] },
        detection_class_priorities: { default: "0", arena: "0" },
        detection_class_filters: { default: "all", arena: "0" },
      },
      control: { ...props.runtimeConfig.control, aim: { class_roles: { default: {}, arena: {} } } },
    };
    const submitted = mockParameterWrites(configured);
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "目标与瞄点" }));
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    const dialog = screen.getByRole("dialog", { name: "编辑目标类别" });
    await userEvent.click(within(dialog).getByRole("button", { name: "cls 0 中心" }));
    await userEvent.click(within(dialog).getByRole("button", { name: "完成编辑" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "编辑目标类别" })).not.toBeInTheDocument());
    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0].changes).toEqual(expect.arrayContaining([
      { section: "inference", key: "detection_class_profile", value: "default" },
      { section: "inference", key: "detection_class_profiles", value: { default: ["target"] } },
      { section: "control", key: "aim", value: { class_roles: { default: {} } } },
    ]));
    const write = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith("/api/v1/config/parameters"))!;
    expect(new Headers(write[1]?.headers).get("x-novasight-physical-output-ack")).toBe("confirmed");
  });

  it("retains a rejected class edit in the shared page draft", async () => {
    const configured = { ...props.runtimeConfig,
      inference: { detection_class_profile: "default", detection_class_profiles: { default: ["enemy"] } },
      control: { ...props.runtimeConfig.control, aim: { class_roles: { default: {} } } },
    };
    vi.stubGlobal("fetch", vi.fn((url, init) => {
      if (!String(url).endsWith("/api/v1/config/parameters")) return new Promise(() => {});
      return Promise.resolve((init?.method ?? "GET") === "GET"
        ? new Response(JSON.stringify(configured), { status: 200, headers: { "content-type": "application/json" } })
        : new Response(JSON.stringify({ code: "CLASS_CONFIG_REJECTED", message: "invalid class role" }), { status: 422, headers: { "content-type": "application/json" } }));
    }));
    render(<SafetyOperationProvider><StudioConsoleView {...props} runtimeConfig={configured} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "目标与瞄点" }));
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    const dialog = screen.getByRole("dialog", { name: "编辑目标类别" });
    await userEvent.click(within(dialog).getByRole("button", { name: "cls 0 中心" }));
    await userEvent.click(within(dialog).getByRole("button", { name: "完成编辑" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/参数保存失败：.*invalid class role/)).toBeVisible());
    expect(screen.getByRole("button", { name: "重试保存" })).toBeEnabled();
    await userEvent.click(screen.getByRole("button", { name: "编辑目标类别" }));
    expect(screen.getByRole("dialog", { name: "编辑目标类别" })).toBeVisible();
  });
});

describe("Settings and authorization navigation", () => {
  it("opens settings and about as dialogs without replacing the page, draft, URL or scroll position", async () => {
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    const input = screen.getByRole("textbox", { name: "触发后等待时长 数值" });
    fireEvent.change(input, { target: { value: "25" } });
    fireEvent.blur(input);
    const main = document.querySelector<HTMLElement>(".console-main")!;
    main.scrollTop = 132;
    await userEvent.click(screen.getByRole("button", { name: "设置" }));
    expect(screen.getByRole("dialog", { name: "设置" })).toHaveFocus();
    expect(main).toHaveAttribute("data-page", "params");
    expect(main.inert).toBe(true);
    expect(location.search).toBe("?page=params");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "设置" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "设置" })).toHaveFocus();
    expect(main.inert).not.toBe(true);
    expect(input).toHaveValue("25");
    expect(main.scrollTop).toBe(132);
    await userEvent.click(screen.getByRole("button", { name: "关于" }));
    const about = screen.getByRole("dialog", { name: "关于 NovaSight" });
    expect(main).toHaveAttribute("data-page", "params");
    await userEvent.click(within(about).getByRole("button", { name: "打开实时日志" }));
    expect(screen.queryByRole("dialog", { name: "关于 NovaSight" })).not.toBeInTheDocument();
    expect(main).toHaveAttribute("data-page", "activity");
  });

  it("turns legacy utility page links into dialogs over the home workspace", async () => {
    history.replaceState(null, "", "/?page=settings");
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    expect(screen.getByRole("dialog", { name: "设置" })).toBeVisible();
    expect(document.querySelector(".console-main")).toHaveAttribute("data-page", "overview");
    expect(location.search).toBe("?page=overview");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "设置" })).not.toBeInTheDocument();
  });
});

// Only committed edits enter the shared draft; focusing the next control must not defer them.
describe("Algorithm range and chart behavior", () => {
  it("edits both endpoints, updates diagrams before saving and persists the accepted values", async () => {
    const writes = mockParameterWrites(props.runtimeConfig);
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    const controls = [
      ["搜索半径", "target_fov_radius_px", 1000, "范围与触发"],
      ["触发后等待时长", "fire_delay_ms", 2000, "范围与触发"],
      ["力度渐增时长", "entry_ramp_ms", 1000, "范围与触发"],
      ["单次 X 轴输出上限", "max_output_x_counts", 1000, "移动与输出"],
      ["单次 Y 轴输出上限", "max_output_y_counts", 1000, "移动与输出"],
    ] as const;
    for (const [label, key, max, section] of controls) {
      await userEvent.click(screen.getByRole("tab", { name: section }));
      expect(screen.getByRole("slider", { name: `${label} 滑块` })).toHaveAttribute("min", "0");
      expect(screen.getByRole("slider", { name: `${label} 滑块` })).toHaveAttribute("max", String(max));
      for (const value of [max, 0]) {
        const input = screen.getByRole("textbox", { name: `${label} 数值` });
        const previous = writes.length;
        fireEvent.change(input, { target: { value: String(value) } });
        expect(writes).toHaveLength(previous);
        if (key === "entry_ramp_ms") expect(screen.getByRole("img", { name: /启动响应曲线/ })).toHaveAccessibleName(new RegExp(`渐增 ${value} ms`));
        if (key === "fire_delay_ms") expect(screen.getByRole("img", { name: /启动响应曲线/ })).toHaveAccessibleName(new RegExp(`等待 ${value} ms`));
        fireEvent.blur(input);
        await waitFor(() => expect(writes).toHaveLength(previous + 1));
        expect(writes[writes.length - 1].changes).toContainEqual({ section: "pipeline", key, value });
      }
    }
    expect(screen.getByRole("img", { name: /单次响应曲线/ })).toHaveAccessibleName(/X 上限 0，Y 上限 0/);
    expect(screen.queryByText("自动保存并应用", { exact: true })).not.toBeInTheDocument();
  }, 15000);


  it("saves a committed axis cap while preserving the next field's unfinished text", async () => {
    const writes = mockParameterWrites(props.runtimeConfig);
    render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("tab", { name: "移动与输出" }));
    const x = screen.getByRole("textbox", { name: "单次 X 轴输出上限 数值" });
    const y = screen.getByRole("textbox", { name: "单次 Y 轴输出上限 数值" });
    await userEvent.clear(x);
    await userEvent.type(x, "96");
    await userEvent.click(y);
    await userEvent.clear(y);
    await userEvent.type(y, "-");
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0].changes).toEqual([{ section: "pipeline", key: "max_output_x_counts", value: 96 }]);
    expect(y).toHaveValue("-");
    expect(y).toHaveFocus();
  });
});
