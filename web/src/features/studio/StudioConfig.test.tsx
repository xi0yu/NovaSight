import { render, screen, waitFor, within } from "@testing-library/react";
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
    ? Promise.resolve(new Response(JSON.stringify({ code: "TEST_OUTPUT_REJECTED", message: "output rejected by daemon" }), { status: 409, headers: { "content-type": "application/json" } }))
    : new Promise(() => {})));
  render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await openProfile();
  await userEvent.click(screen.getByRole("button", { name: "打开输出" }));
  expect(screen.getByRole("alertdialog", { name: "允许发送鼠标偏移？" })).toBeVisible();
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes("/config/commands"))).toBe(false);
  await userEvent.click(screen.getByRole("button", { name: "确认开启输出" }));
  await waitFor(() => expect(screen.getByRole("alertdialog")).toHaveTextContent("output rejected by daemon"));
  const command = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes("/config/commands"))!;
  expect(JSON.parse(String(command[1]?.body))).toMatchObject({ command: "set_output_gate", enabled: true });
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
