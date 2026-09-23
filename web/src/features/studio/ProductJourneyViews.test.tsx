import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DeviceStatusView, ManagementView, OnboardingView } from "./ProductJourneyViews";

const incomplete = {
  captureReady: true,
  modelReady: false,
  configReady: false,
  runtimeReady: false,
};

describe("guided product journeys", () => {
  it("takes a new user to the first unfinished task", async () => {
    const onNavigate = vi.fn();
    render(<OnboardingView state={incomplete} onNavigate={onNavigate} />);

    expect(screen.getByRole("status", { name: "已完成 1 步，共 4 步" })).toBeInTheDocument();
    await userEvent.click(screen.getAllByRole("button", { name: "选择模型" })[1]!);
    expect(onNavigate).toHaveBeenCalledWith("models");
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

  it("continues onboarding from management when setup is incomplete", async () => {
    const onNavigate = vi.fn();
    render(
      <ManagementView
        license={null}
        deviceLabel=""
        runtimeAvailable={false}
        projectCount={0}
        errorCount={0}
        setupState={incomplete}
        onNavigate={onNavigate}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "继续新手引导" }));
    expect(onNavigate).toHaveBeenCalledWith("onboarding");
  });
});
