import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeState } from "../../api";
import type { RuntimeProjection } from "../runtime/runtimeProjection";
import { RuntimeOverviewView } from "./RuntimeOverviewView";

const runtime = {
  semantic: { daemon_instance_id: "daemon-abc123", phase: "running", snapshot_sequence: 9 },
  capture: { running: true, available: true },
  executor: { executors: { kmnet: { runtime_connected: true } } },
  statistics: {
    metrics_available: true,
    nvinfer_input_fps: 240,
    detection_batch_fps: 238,
    inference_latency_ms: 8.2,
    detection_data_age_ms: 4.1,
  },
  vision: { target: null, control: { will_emit: true } },
} as unknown as RuntimeState;

const projection: RuntimeProjection = {
  transport: "current",
  lifecycle: { state: "running", label: "运行中", detail: "快照 #9" },
  perception: { state: "current", label: "感知数据新鲜", detail: "帧龄 4 ms" },
  output: { state: "armed", label: "输出已具备条件", detail: "当前样本满足门控" },
  conclusion: "主链正在运行，输出条件已满足",
  daemonConfirmedSafe: false,
  nextAction: "open-latency",
  nextActionLabel: "检查延迟",
};

const controlProps = {
  controlBusy: false,
  emergencyStopping: false,
  launchPending: false,
  runtimeStopping: false,
  runtimeControlUnavailable: false,
  onToggle: vi.fn(),
  onEmergencyStop: vi.fn(),
};

describe("RuntimeOverviewView", () => {
  it("shows one conclusion, three independent axes and no delivery claim", () => {
    render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={new Date("2026-08-20T08:00:00Z")}
        onAction={() => undefined}
        {...controlProps}
      />
    );
    expect(screen.getByRole("heading", { name: projection.conclusion })).toBeInTheDocument();
    expect(screen.getByText("运行生命周期")).toBeInTheDocument();
    expect(screen.getByText("感知数据", { selector: ".runtime-overview-axes span" })).toBeInTheDocument();
    expect(screen.getByText("硬件输出", { selector: ".runtime-overview-axes span" })).toBeInTheDocument();
    expect(screen.queryByText(/已发送|正在交付|delivering/i)).not.toBeInTheDocument();
  });

  it("only exposes the mapped recovery action", async () => {
    const onAction = vi.fn();
    render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={null}
        onAction={onAction}
        {...controlProps}
      />
    );
    await userEvent.click(screen.getByRole("button", { name: "检查延迟" }));
    expect(onAction).toHaveBeenCalledWith("open-latency");
  });

  it("keeps daemon and output proof collapsed until the operator asks for diagnostics", () => {
    render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={new Date("2026-08-20T08:00:00Z")}
        onAction={() => undefined}
        {...controlProps}
      />
    );

    const disclosure = screen.getByText("运行诊断证据").closest("details");
    expect(disclosure).not.toHaveAttribute("open");
    expect(disclosure).toHaveTextContent("Daemon");
    expect(disclosure).toHaveTextContent("当前样本可输出");
  });

  it("puts real core metrics and runtime controls in the first-screen conclusion", async () => {
    const onToggle = vi.fn();
    const onEmergencyStop = vi.fn();
    render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={new Date("2026-08-20T08:00:00Z")}
        onAction={() => undefined}
        controlBusy={false}
        emergencyStopping={false}
        launchPending={false}
        runtimeStopping={false}
        runtimeControlUnavailable={false}
        onToggle={onToggle}
        onEmergencyStop={onEmergencyStop}
      />
    );

    const metrics = screen.getByLabelText("核心运行数据");
    expect(metrics).toHaveTextContent("推理输入 FPS240");
    expect(metrics).toHaveTextContent("检测结果 FPS238");
    expect(metrics).toHaveTextContent("推理耗时8.2 ms");

    await userEvent.click(screen.getByRole("button", { name: "停止运行" }));
    await userEvent.click(screen.getByRole("button", { name: "紧急停止" }));
    expect(onToggle).toHaveBeenCalledOnce();
    expect(onEmergencyStop).toHaveBeenCalledOnce();
  });

  it("does not present missing live samples as zero", () => {
    const runtimeWithoutSamples = {
      ...runtime,
      statistics: {
        ...runtime.statistics,
        metrics_available: false,
        nvinfer_input_fps: null,
        detection_batch_fps: null,
        inference_latency_ms: null,
        detection_data_age_ms: null,
      },
    } as unknown as RuntimeState;

    render(
      <RuntimeOverviewView
        runtime={runtimeWithoutSamples}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={null}
        onAction={() => undefined}
        {...controlProps}
      />
    );

    const metrics = screen.getByLabelText("核心运行数据");
    expect(metrics).not.toHaveTextContent(/FPS0/);
    expect(screen.getAllByText("等待样本")).toHaveLength(3);
  });

  it("does not present stale statistics as current live metrics", () => {
    render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={{ ...projection, transport: "stale" }}
        readiness={{ state: "action", title: "需处理", detail: "请恢复实时连接。" }}
        lastUpdated={null}
        onAction={() => undefined}
        {...controlProps}
      />
    );

    expect(screen.getAllByText("状态已过期")).toHaveLength(3);
    expect(screen.getByLabelText("核心运行数据")).not.toHaveTextContent("240");
  });

  it("shows pending start and stop feedback before the next runtime snapshot", () => {
    const { rerender } = render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={null}
        onAction={() => undefined}
        {...controlProps}
        controlBusy
        runtimeStopping
      />
    );
    expect(screen.getByRole("button", { name: "正在停止" })).toBeDisabled();

    rerender(
      <RuntimeOverviewView
        runtime={{ ...runtime, semantic: { ...runtime.semantic, phase: "stopped" } } as RuntimeState}
        projection={{ ...projection, lifecycle: { state: "stopped", label: "已停止", detail: "" } }}
        readiness={{ state: "action", title: "需处理", detail: "正在启动。" }}
        lastUpdated={null}
        onAction={() => undefined}
        {...controlProps}
        controlBusy
        launchPending
      />
    );
    expect(screen.getByRole("button", { name: "正在启动" })).toBeDisabled();
  });
});
