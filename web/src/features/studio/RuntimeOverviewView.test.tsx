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

describe("RuntimeOverviewView", () => {
  it("shows one conclusion, three independent axes and no delivery claim", () => {
    render(
      <RuntimeOverviewView
        runtime={runtime}
        projection={projection}
        readiness={{ state: "action", title: "需处理", detail: "请检查端到端延迟。" }}
        lastUpdated={new Date("2026-08-20T08:00:00Z")}
        onAction={() => undefined}
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
      />
    );

    const disclosure = screen.getByText("运行诊断证据").closest("details");
    expect(disclosure).not.toHaveAttribute("open");
    expect(disclosure).toHaveTextContent("Daemon");
    expect(disclosure).toHaveTextContent("当前样本可输出");
  });
});
