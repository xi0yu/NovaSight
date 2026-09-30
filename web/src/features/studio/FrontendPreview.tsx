import type { LicenseStatus, RuntimeState } from "../../api";
import { SafetyOperationProvider } from "../runtime/SafetyOperationContext";
import { StudioConsoleView } from "./StudioConsoleView";

export const previewRuntime = {
  semantic: {
    daemon_instance_id: "frontend-preview",
    phase: "stopped",
    perception_phase: "stopped",
    epoch: 1,
    snapshot_sequence: 1,
  },
  presentation: {
    lifecycle: { can_start: true, can_stop: false },
    readiness: { code: "stopped", recommended_action: null },
    perception: { state: "stopped" },
    output: { state: "safe", reason_code: "runtime_stopped", daemon_confirmed_safe: true },
  },
  running: false,
  capture: { available: false, device: "", running: false, profile: null, state: "unconfigured" },
  statistics: { metrics_available: false, detection_freshness_threshold_ms: 50 },
  config: { version: 1, effective_version: 1 },
  executor: {
    executors: {
      kmnet: { available: true, connected: false, runtime_connected: false, connection_state: "disconnected" },
    },
  },
  pipeline: { running: false, state: "stopped", deepstream: {} },
  inference: {},
  vision: {
    control: {},
    output_trace: {},
    target_pipeline: { rejection_reasons: [], counts: {} },
    detections: [],
    target: null,
    inference: { roi_offset_x: 0, roi_offset_y: 0, roi_width: 256, roi_height: 256 },
  },
} as unknown as RuntimeState;

export function FrontendPreview({ scenario = "pages" }: { scenario?: string }) {
  return (
    <SafetyOperationProvider>
      <StudioConsoleView
        license={{ valid: true, features: ["hardware_control"] } as LicenseStatus}
        health={scenario === "offline" ? null : { ok: true }}
        runtime={scenario === "offline" ? null : previewRuntime}
        runtimeConfig={{
          revision: 1,
          pipeline: { fire_delay_enabled: false, fire_delay_ms: 0, projection_fov_x_deg: 90 },
          control: { output_enabled: false },
          hardware: { auto_connect: true },
        }}
        projects={[]}
        errors={{}}
        lastUpdated={null}
        realtimeStatus={scenario === "offline" ? "disconnected" : scenario === "stale" ? "stale" : "connected"}
        onEnsureProjects={async () => []}
        onLicenseChange={() => undefined}
        onRefresh={async () => undefined}
        onRuntimeConfigChange={() => undefined}
        onRuntimeStateChange={() => true}
      />
    </SafetyOperationProvider>
  );
}
