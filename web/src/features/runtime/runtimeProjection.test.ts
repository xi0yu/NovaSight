import { describe, expect, it } from "vitest";

import type { RuntimeState } from "../../api";
import { isDaemonConfirmedSafe, projectRuntimeState } from "./runtimeProjection";

function runtime(overrides: {
  phase?: RuntimeState["semantic"]["phase"];
  outputCode?: string;
  runtimeConnected?: boolean;
  willEmit?: boolean | null;
  age?: number | null;
  threshold?: number | null;
  recommendedAction?: RuntimeState["presentation"]["readiness"]["recommended_action"];
} = {}): RuntimeState {
  const phase = overrides.phase ?? "stopped";
  const age = overrides.age ?? 4;
  const threshold = overrides.threshold ?? 20;
  const daemonSafe = phase === "stopped"
    && (overrides.outputCode ?? "runtime_stopped") === "runtime_stopped"
    && overrides.runtimeConnected !== true
    && overrides.willEmit !== true;
  const perception = phase === "running"
    ? age !== null && threshold !== null && age <= threshold ? "current" : "stale"
    : phase === "faulted" ? "faulted" : phase === "starting" ? "starting" : "stopped";
  const output = daemonSafe
    ? "safe"
    : ["starting", "stopping", "faulted"].includes(phase) || phase === "stopped"
      ? "unknown"
      : perception === "stale" ? "blocked" : "armed";
  return {
    running: phase === "running",
    semantic: {
      daemon_instance_id: "daemon-a",
      phase,
      perception_phase: phase === "running" ? "running" : "stopped",
      epoch: null,
      snapshot_sequence: 8,
      snapshot_updated_at_ms: 100,
    },
    presentation: {
      lifecycle: { can_start: daemonSafe, can_stop: phase !== "stopping" && !daemonSafe },
      readiness: {
        code: phase === "stopped" ? "stopped" : phase === "faulted" ? "failed" : "ready",
        recommended_action: overrides.recommendedAction ?? null,
      },
      perception: { state: perception },
      output: { state: output, reason_code: overrides.outputCode ?? "runtime_stopped", daemon_confirmed_safe: daemonSafe },
    },
    executor: { executors: { kmnet: { runtime_connected: overrides.runtimeConnected ?? false } } },
    statistics: {
      detection_data_age_ms: age,
      detection_freshness_threshold_ms: threshold,
    },
    inference: { detail: null, reason: null },
    vision: {
      output_trace: {
        code: overrides.outputCode ?? "runtime_stopped",
        state: "blocked",
        detail: "输出被安全门控阻止",
        next_action: "wait_next_frame",
      },
      control: { will_emit: overrides.willEmit ?? false },
    },
    fatal_error: null,
  } as unknown as RuntimeState;
}

describe("runtime projection", () => {
  it("only calls the strict stopped tuple daemon-confirmed safe", () => {
    expect(isDaemonConfirmedSafe(runtime())).toBe(true);
    expect(isDaemonConfirmedSafe(runtime({ outputCode: "ready" }))).toBe(false);
    expect(isDaemonConfirmedSafe(runtime({ runtimeConnected: true }))).toBe(false);
    expect(isDaemonConfirmedSafe(runtime({ willEmit: true }))).toBe(false);
    expect(isDaemonConfirmedSafe(runtime({ phase: "stopping" }))).toBe(false);
  });

  it("never reports safe when transport evidence is stale", () => {
    const projection = projectRuntimeState(runtime(), "stale");
    expect(projection?.daemonConfirmedSafe).toBe(true);
    expect(projection?.output.state).toBe("unknown");
    expect(projection?.conclusion).toBe("无法确认实时运行状态");
  });

  it("marks fresh running perception current and ready output armed", () => {
    const projection = projectRuntimeState(runtime({ phase: "running", outputCode: "ready", willEmit: true }), "current");
    expect(projection?.perception.state).toBe("current");
    expect(projection?.output.state).toBe("armed");
  });

  it("blocks stale perception and treats faulted or incomplete stopped evidence as unknown", () => {
    expect(projectRuntimeState(runtime({ phase: "running", age: 30, threshold: 20, outputCode: "ready", willEmit: true }), "current")?.output.state).toBe("blocked");
    expect(projectRuntimeState(runtime({ phase: "faulted", outputCode: "ready", willEmit: true }), "current")?.output.state).toBe("unknown");
    expect(projectRuntimeState(runtime({ phase: "stopped", runtimeConnected: true }), "current")?.output.state).toBe("unknown");
    expect(projectRuntimeState(runtime({ phase: "starting", outputCode: "ready", willEmit: true }), "current")?.output.state).toBe("unknown");
    expect(projectRuntimeState(runtime({ phase: "stopping", outputCode: "ready", willEmit: true }), "current")?.output.state).toBe("unknown");
  });

  it("routes known recovery actions and leaves unknown actions inert", () => {
    expect(projectRuntimeState(runtime({ recommendedAction: "inspect_latency" }), "current")?.nextAction).toBe("open-latency");
    expect(projectRuntimeState(runtime(), "current")?.nextAction).toBeUndefined();
  });

  it("prioritizes capture recovery when the running mainline has no video", () => {
    const projection = projectRuntimeState(runtime({ phase: "running", recommendedAction: "configure_capture" }), "current");
    expect(projection?.nextAction).toBe("open-capture");
    expect(projection?.nextActionLabel).toBe("检查采集");
  });
});
