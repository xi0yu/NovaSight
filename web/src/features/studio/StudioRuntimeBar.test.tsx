import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { StudioRuntimeBar } from "./StudioRuntimeBar";

describe("StudioRuntimeBar", () => {
  it("leaves the overview as the single owner of runtime status and controls", () => {
    render(
      <StudioRuntimeBar
        page="overview"
        runtimeAvailable
        runtimeLifecycleActive
        runtimeControlRequested
        runtimeStopping={false}
        launchPending={false}
        diagnosticModeReady={false}
        busy={false}
        emergencyStopping={false}
        runtimeControlUnavailable={false}
        captureStatus="运行中"
        inferenceStatus="识别结果已产出"
        onToggle={vi.fn()}
        onEmergencyStop={vi.fn()}
      />
    );

    expect(screen.queryByRole("region", { name: "主链运行控制" })).not.toBeInTheDocument();
  });
});
