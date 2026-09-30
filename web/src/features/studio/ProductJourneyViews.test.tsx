import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DeviceStatusView, HomeSetupPrompt, HomeShortcuts, ManagementView, OnboardingView } from "./ProductJourneyViews";
import { SettingsView } from "./SettingsView";

const incomplete = {
  captureReady: true,
  modelReady: false,
  configReady: false,
  runtimeReady: false,
};

it("only calls revisions consistent when both saved and running settings are known", () => {
  const props = { modules: ["backup", "restore", "details"], desiredRevision: 2, effectiveRevision: 1, restartRequired: false, operationPending: false, parameterChangesPending: false, onExport: vi.fn(), onImport: vi.fn(), onNavigate: vi.fn() };
  const view = render(<SettingsView {...props} configAvailable={false} runtimeVerified={false} />);
  expect(screen.getByRole("status")).toHaveTextContent("设置未读取");
  view.rerender(<SettingsView {...props} configAvailable runtimeVerified={false} />);
  expect(screen.getByRole("status")).toHaveTextContent("运行版本待确认");
  view.rerender(<SettingsView {...props} configAvailable runtimeVerified />);
  expect(screen.getByRole("status")).toHaveTextContent("等待生效");
});

describe("guided product journeys", () => {
  it("opens the missing setup task directly from home without another guide step", async () => {
    const onNavigate = vi.fn();
    const { rerender } = render(<HomeSetupPrompt state={incomplete} statusKnown onNavigate={onNavigate} />);
    await userEvent.click(screen.getByRole("button", { name: "选择识别模型" }));
    expect(onNavigate).toHaveBeenCalledWith("models");
    rerender(<HomeSetupPrompt state={incomplete} statusKnown={false} onNavigate={onNavigate} />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("offers real device, tuning and history destinations from home", async () => {
    const onNavigate = vi.fn();
    render(<HomeShortcuts onNavigate={onNavigate} />);
    for (const label of ["设备与画面", "调整使用手感", "查看使用记录"]) {
      await userEvent.click(screen.getByRole("button", { name: new RegExp(label) }));
    }
    expect(onNavigate.mock.calls).toEqual([["capture"], ["params"], ["activity"]]);
  });

  it("takes a new user to the first unfinished task", async () => {
    const onNavigate = vi.fn();
    render(<OnboardingView state={incomplete} statusKnown onNavigate={onNavigate} />);

    expect(screen.getByRole("status", { name: "已完成 1 步，共 4 步" })).toBeInTheDocument();
    await userEvent.click(screen.getAllByRole("button", { name: "选择模型" })[1]!);
    expect(onNavigate).toHaveBeenCalledWith("models");
  });

  it("does not label an unreachable device as a fresh setup", () => {
    render(<OnboardingView state={{ captureReady: false, modelReady: false, configReady: false, runtimeReady: false }} statusKnown={false} onNavigate={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "正在核对当前设备" })).toBeVisible();
    expect(screen.getByRole("status", { name: "设置进度待核实" })).toBeVisible();
    expect(screen.queryByText("已完成 0 步")).not.toBeInTheDocument();
  });

  it("does not invent device health while the runtime snapshot is missing", () => {
    render(
      <DeviceStatusView
        runtime={null}
        projection={null}
        captureProfile=""
        activeModelName="未发布模型"
        modelLoaded={false}
        lastUpdated={null}
        desiredRevision={0}
        effectiveRevision={0}
        errorCount={0}
        onNavigate={vi.fn()}
        onOpenErrors={vi.fn()}
      />
    );

    expect(screen.getByText("等待当前状态")).toBeInTheDocument();
    expect(screen.getAllByText("等待样本")).toHaveLength(3);
    expect(screen.queryByText("版本一致")).not.toBeInTheDocument();
    expect(screen.queryByText(/GPU|温度/)).not.toBeInTheDocument();
  });

  it("marks old device readings as expired after the realtime channel goes stale", () => {
    const staleRuntime = {
      semantic: { phase: "running" },
      capture: { device: "/dev/video0", running: true, available: true },
      inference: { configured: true },
      statistics: { metrics_available: true, nvinfer_input_fps: 240, inference_latency_ms: 4, detection_data_age_ms: 2 },
      executor: { executors: { kmnet: { runtime_connected: true } } },
    } as unknown as import("../../api").RuntimeState;
    render(<DeviceStatusView runtime={staleRuntime} projection={{ transport: "stale", output: { label: "输出状态未知" } } as import("../runtime/runtimeProjection").RuntimeProjection}
      captureProfile="MJPG 1920x1080" activeModelName="test" modelLoaded lastUpdated={null}
      desiredRevision={25} effectiveRevision={25} errorCount={0} onNavigate={vi.fn()} onOpenErrors={vi.fn()} />);
    expect(screen.getAllByText("状态已过期")).toHaveLength(4);
    expect(screen.queryByText("240.0 FPS")).not.toBeInTheDocument();
    expect(screen.getByText("kmNet 连接状态待核实")).toBeVisible();
  });

  it("continues onboarding from management when setup is incomplete", async () => {
    const onNavigate = vi.fn();
    render(
      <ManagementView
        license={null}
        deviceLabel=""
        runtimeAvailable
        projectCount={0}
        errorCount={0}
        setupState={incomplete}
        onNavigate={onNavigate}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "继续新手引导" }));
    expect(onNavigate).toHaveBeenCalledWith("onboarding");
  });

  it("does not present missing service data as an unconfigured device or empty model library", () => {
    render(<ManagementView license={null} deviceLabel="" runtimeAvailable={false} projectCount={0} errorCount={0}
      setupState={{ captureReady: false, modelReady: false, configReady: false, runtimeReady: false }} onNavigate={vi.fn()} />);
    expect(screen.getByText("项目待核实")).toBeVisible();
    expect(screen.getByText("授权待核实")).toBeVisible();
    expect(screen.queryByText("尚未配置")).not.toBeInTheDocument();
    expect(screen.queryByText("尚未部署模型")).not.toBeInTheDocument();
  });
});
