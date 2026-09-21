import { act, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import type { RuntimeState } from "../../api";
import { useClearErrorNotices } from "../../lib/toast";
import { SafetyOperationProvider } from "../runtime/SafetyOperationContext";
import { StudioConsoleView } from "./StudioConsoleView";

const message = "Strict GPU frame failed [epoch=RuntimeEpoch(3), generation=Generation(2784), captured_at_ns=730148790084]: TensorRT inspected input is unsupported: GPU result batch: detection 3 box (166.2967071533203, 90.04590606689453)..(255.5125274658203, 256.00000762939453) is outside coordinate space 256x256";
const fault = { code: "PERCEPTION_FAILED", subsystem: "inference", message };
afterEach(() => vi.unstubAllGlobals());
const runtime = {
  semantic: { daemon_instance_id: "daemon-test", phase: "faulted", perception_phase: "faulted", epoch: 3, snapshot_sequence: 10, snapshot_updated_at_ms: 1234 },
  running: false, source: "test", active_model: null, fatal_error: fault,
  capture: {}, statistics: {}, config: {}, executor: { executors: {} },
  pipeline: { state: "faulted", epoch: 3, last_error: fault, deepstream: {} },
  inference: { terminal_error: true, detail: message },
  vision: { control: {}, output_trace: {}, target_pipeline: { rejection_reasons: [], counts: {} }, detections: [] },
} as unknown as RuntimeState;

it.each(["fatal", "pipeline", "inference", "deepstream"])("shows %s faults, retains recovery history, and ignores repeated snapshots", async (source) => {
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
  const { result } = renderHook(() => useClearErrorNotices());
  act(() => result.current());
  const reported = {
    ...runtime,
    fatal_error: source === "fatal" ? fault : null,
    pipeline: { ...runtime.pipeline, last_error: source === "pipeline" ? fault : null, deepstream: { ...runtime.pipeline.deepstream, terminal_error: source === "deepstream", last_error: source === "deepstream" ? message : null } },
    inference: { ...runtime.inference, terminal_error: source === "inference", detail: source === "inference" ? message : null },
  };
  const props = {
    license: null, health: null, runtime: reported, runtimeConfig: null, projects: [], errors: {}, lastUpdated: null,
    realtimeStatus: "connected" as const, onEnsureProjects: vi.fn(async () => []), onLicenseChange: vi.fn(),
    onRefresh: vi.fn(async () => {}), onRuntimeConfigChange: vi.fn(), onRuntimeStateChange: vi.fn(() => true), onStatusTopicChange: vi.fn(),
  };
  const view = render(<SafetyOperationProvider><StudioConsoleView {...props} /></SafetyOperationProvider>);
  await userEvent.click(screen.getByRole("button", { name: /异常信息/ }));
  const dialog = screen.getByRole("dialog", { name: "异常信息" });
  expect(dialog).toHaveTextContent(message);
  await userEvent.click(screen.getByText("原始错误与开发者详情"));
  expect(dialog.querySelector("pre")).toBeVisible();
  if (source === "fatal" || source === "pipeline") expect(dialog).toHaveTextContent("PERCEPTION_FAILED");
  expect(dialog).toHaveTextContent("daemon-test");
  expect(dialog.querySelectorAll(".error-center-item")).toHaveLength(1);
  view.rerender(<SafetyOperationProvider><StudioConsoleView {...props} runtime={{ ...reported, semantic: { ...runtime.semantic, snapshot_sequence: 11 } }} /></SafetyOperationProvider>);
  expect(dialog).not.toHaveTextContent("本会话重复");
  // Clearing history must not hide a still-active fault.
  await userEvent.click(screen.getByRole("button", { name: "清空历史记录" }));
  expect(screen.getByRole("alertdialog", { name: "清空历史错误记录？" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  expect(screen.getByRole("button", { name: "清空历史记录" })).toBeEnabled();
  await userEvent.click(screen.getByRole("button", { name: "清空历史记录" }));
  await userEvent.click(screen.getByRole("button", { name: "确认清空历史" }));
  expect(dialog).toHaveTextContent(message);
  expect(screen.getByRole("button", { name: "没有可清空的记录" })).toBeDisabled();
  view.rerender(<SafetyOperationProvider><StudioConsoleView {...props} runtime={{ ...reported, semantic: { ...runtime.semantic, epoch: 4 } }} /></SafetyOperationProvider>);
  view.rerender(<SafetyOperationProvider><StudioConsoleView {...props} runtime={{ ...runtime, fatal_error: null, pipeline: { ...runtime.pipeline, last_error: null }, inference: { ...runtime.inference, terminal_error: false, detail: null } }} /></SafetyOperationProvider>);
  expect(dialog).toHaveTextContent(message);
  act(() => result.current());
});
