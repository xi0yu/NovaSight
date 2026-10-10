import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StudioRuntimeBar } from "./StudioRuntimeBar";

describe("StudioRuntimeBar", () => {

  it("does not repeat the capture page's verified device status", () => {
    render(<StudioRuntimeBar page="capture" runtimeAvailable runtimeLifecycleActive={false} captureStatus="已停止" inferenceStatus="已停止" />);
    expect(screen.queryByRole("region", { name: "主链运行状态" })).not.toBeInTheDocument();
  });

  it("does not call an unconfirmed runtime idle", () => {
    render(
      <StudioRuntimeBar
        page="capture"
        runtimeAvailable={false}
        runtimeLifecycleActive={false}
        captureStatus="等待状态"
        inferenceStatus="等待状态"
      />
    );
    expect(screen.getByText("运行状态未确认")).toBeInTheDocument();
    expect(screen.queryByText("主链待机")).not.toBeInTheDocument();
  });

});
