import type { LicenseStatus, RuntimeState } from "../../api";
import { SafetyOperationProvider } from "../runtime/SafetyOperationContext";
import { StudioConsoleView } from "./StudioConsoleView";

const captureProfile = { pixel_format: "MJPG", width: 1920, height: 1080, fps: 240 };

const runtime = {
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
  capture: { available: true, device: "/dev/video0", running: false, profile: captureProfile },
  statistics: { metrics_available: false, detection_freshness_threshold_ms: 55 },
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

export function FrontendPreview() {
  return (
    <SafetyOperationProvider>
      <StudioConsoleView
        license={{ valid: true, features: ["hardware_control"] } as LicenseStatus}
        health={{ ok: true }}
        runtime={runtime}
        runtimeConfig={{
          revision: 1,
          capture: { ...captureProfile, device: "/dev/video0", roi_left: 0, roi_top: 0, roi_width: 256, roi_height: 256 },
          pipeline: { fire_delay_enabled: false, fire_delay_ms: 0, projection_fov_x_deg: 90 },
          control: { output_enabled: false },
          hardware: { auto_connect: true },
        }}
        projects={[]}
        errors={{}}
        lastUpdated={null}
        realtimeStatus="connected"
        onEnsureProjects={async () => []}
        onLicenseChange={() => undefined}
        onRefresh={async () => undefined}
        onRuntimeConfigChange={() => undefined}
        onRuntimeStateChange={() => true}
      />
    </SafetyOperationProvider>
  );
}
