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
  nextAction?: string;
} = {}): RuntimeState {
  return {
    running: overrides.phase === "running",
    semantic: {
      daemon_instance_id: "daemon-a",
      phase: overrides.phase ?? "stopped",
      perception_phase: overrides.phase === "running" ? "running" : "stopped",
      epoch: null,
      snapshot_sequence: 8,
      snapshot_updated_at_ms: 100,
    },
    executor: { executors: { kmnet: { runtime_connected: overrides.runtimeConnected ?? false } } },
    statistics: {
      detection_data_age_ms: overrides.age ?? 4,
      detection_freshness_threshold_ms: overrides.threshold ?? 20,
    },
    inference: { detail: null, reason: null },
    vision: {
      output_trace: {
        code: overrides.outputCode ?? "runtime_stopped",
        state: "blocked",
        detail: "输出被安全门控阻止",
        next_action: overrides.nextAction ?? "wait_next_frame",
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
    expect(projectRuntimeState(runtime({ nextAction: "check_latency" }), "current")?.nextAction).toBe("open-latency");
    expect(projectRuntimeState(runtime({ nextAction: "future_action" }), "current")?.nextAction).toBeUndefined();
  });

  it("prioritizes capture recovery when the running mainline has no video", () => {
    const projection = projectRuntimeState(runtime({ phase: "running", nextAction: "check_model" }), "current");
    expect(projection?.nextAction).toBe("open-capture");
    expect(projection?.nextActionLabel).toBe("检查采集");
  });
});
