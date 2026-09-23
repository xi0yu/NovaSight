import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { RuntimeState } from "../../api";
import { SafetyOperationProvider, useSafetyOperation } from "./SafetyOperationContext";

function safeRuntime(daemonInstanceId: string, snapshotSequence: number): RuntimeState {
  return {
    semantic: { daemon_instance_id: daemonInstanceId, snapshot_sequence: snapshotSequence, phase: "stopped" },
    executor: { executors: { kmnet: { runtime_connected: false } } },
    vision: {
      output_trace: { code: "runtime_stopped" },
      control: { will_emit: false },
    },
  } as unknown as RuntimeState;
}

function ProtectedOperation({ onAuthLost }: { onAuthLost: () => void }) {
  const { beginEmergencyStop } = useSafetyOperation();
  return (
    <button type="button" onClick={() => {
      beginEmergencyStop(null);
      onAuthLost();
    }}>
      模拟紧停后认证失效
    </button>
  );
}

function Harness() {
  const [authenticated, setAuthenticated] = useState(true);
  return authenticated
    ? <ProtectedOperation onAuthLost={() => setAuthenticated(false)} />
    : <main>验证后继续</main>;
}

function CausalHarness() {
  const { beginEmergencyStop, reconcileEmergencyStop } = useSafetyOperation();
  return (
    <>
      <button type="button" onClick={() => beginEmergencyStop(safeRuntime("daemon-a", 8))}>开始紧停</button>
      <button type="button" onClick={() => reconcileEmergencyStop(safeRuntime("daemon-a", 9))}>同实例确认</button>
      <button type="button" onClick={() => reconcileEmergencyStop(safeRuntime("daemon-b", 1))}>重启后安全</button>
    </>
  );
}

describe("SafetyOperationProvider", () => {
  it("keeps an unconfirmed emergency operation visible when protected UI unmounts", async () => {
    render(<SafetyOperationProvider><Harness /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("button", { name: "模拟紧停后认证失效" }));
    expect(screen.getByText("验证后继续")).toBeInTheDocument();
    expect(screen.getByText("紧急停止确认中")).toBeInTheDocument();
    expect(screen.getByText(/等待最终安全快照/)).toBeInTheDocument();
  });

  it("only confirms a newer safe snapshot from the requested daemon", async () => {
    render(<SafetyOperationProvider><CausalHarness /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("button", { name: "开始紧停" }));
    await userEvent.click(screen.getByRole("button", { name: "同实例确认" }));
    expect(screen.getByText("紧急停止已确认")).toBeInTheDocument();
    expect(screen.getByText(/同一 novasightd/)).toBeInTheDocument();
  });

  it("reports current safety without claiming causality after daemon restart", async () => {
    render(<SafetyOperationProvider><CausalHarness /></SafetyOperationProvider>);
    await userEvent.click(screen.getByRole("button", { name: "开始紧停" }));
    await userEvent.click(screen.getByRole("button", { name: "重启后安全" }));
    expect(screen.getByText("紧急停止尚未确认")).toBeInTheDocument();
    expect(screen.getByText(/无法证明这是本次紧急停止/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "同实例确认" }));
    expect(screen.getByText("紧急停止尚未确认")).toBeInTheDocument();
  });
});
