import { useState } from "react";
import type { RuntimeConfig, LicenseStatus, RuntimeState } from "../../api";
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
    output: { state: "safe", reason_code: "null_mouse_protocol", daemon_confirmed_safe: true },
  },
  running: false,
  capture: { available: false, device: "", running: false, profile: null, state: "unconfigured" },
  statistics: { metrics_available: false, detection_freshness_threshold_ms: 50 },
  config: { version: 1, effective_version: 1 },
  executor: {
    selected: "null",
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

const previewRunningRuntime = {
  ...previewRuntime,
  semantic: { ...previewRuntime.semantic, phase: "running", perception_phase: "running", snapshot_sequence: 24 },
  presentation: {
    lifecycle: { can_start: false, can_stop: true },
    readiness: { code: "ready", recommended_action: null },
    perception: { state: "current" },
    output: { state: "safe", reason_code: "null_mouse_protocol", daemon_confirmed_safe: true },
  },
  running: true,
  active_model: {
    project: { id: 1, name: "演示模型", description: "仅用于界面预览" },
    version: { id: 1, project_id: 1, version: "preview", source_kind: "engine", source_path: "", classes: ["头部", "身体", "队友"], input_shape: "640x640" },
    artifact: { id: 1, version_id: 1, kind: "engine", path: "", checksum: "", status: "ready", size_bytes: null },
    deployment: { id: 1, project_id: 1, artifact_id: 1, previous_artifact_id: null, updated_seq: 1 },
    artifact_path: "",
  },
  capture: { ...previewRuntime.capture, available: true, device: "演示画面", running: true, state: "running", profile: { pixel_format: "MJPG", width: 1920, height: 1080, fps: 60, preference: "preview", source: "configured" } },
  statistics: {
    ...previewRuntime.statistics,
    metrics_available: true,
    nvinfer_input_fps: 59.8,
    detection_batch_fps: 59.4,
    inference_latency_ms: 8.2,
    detection_data_age_ms: 14.1,
    detection_freshness_threshold_ms: 50,
  },
  config: { ...previewRuntime.config, version: 1, effective_version: 1, restart_required: false },
  pipeline: { ...previewRuntime.pipeline, running: true, state: "running" },
  inference: { available: true, configured: true, loaded: true, running: true, state: "running", detail: null, reason: null },
  vision: {
    ...previewRuntime.vision,
    detections: 2,
    detection_items: [
      { object_id: 1, class_id: 0, cls: 0, score: 0.92, x: 270, y: 120, w: 76, h: 72, cx: 308, cy: 156 },
      { object_id: 2, class_id: 1, cls: 1, score: 0.88, x: 238, y: 190, w: 140, h: 230, cx: 308, cy: 305 },
    ],
    target: { target_detection_index: 0, track_id: 7, class_id: 0, cls: 0, score: 0.92, identity_confidence: 0.9, x1: 270, y1: 120, x2: 346, y2: 192, box_cx: 308, box_cy: 156, cx: 308, cy: 145, observed_aim_x: 308, observed_aim_y: 145 },
    target_pipeline: { code: "selected", stage: "selection", message: "演示目标已选中", rejection_reasons: [], counts: { raw_candidates: 2, eligible_candidates: 2, selected_targets: 1 } },
    output_trace: { code: "null_mouse_protocol", state: "ready", detail: "空协议静默消费鼠标指令，不向物理设备发送输出。", next_action: "none" },
    control: { ...previewRuntime.vision.control, will_emit: false, output_enabled: true, trigger_active: false },
    inference: { roi_offset_x: 640, roi_offset_y: 220, roi_width: 640, roi_height: 640 },
  },
} as unknown as RuntimeState;

export function FrontendPreview({ scenario = "pages" }: { scenario?: string }) {
  const running = scenario === "running-target" || scenario === "running-no-target";
  const [previewLicenseDeadline] = useState(() => Date.now() / 1000 + 86400);
  const [config, setConfig] = useState<RuntimeConfig>(() => ({
          revision: 1,
          ...(running ? { capture: { device: "演示画面", pixel_format: "MJPG", width: 1920, height: 1080, fps: 60, roi_left: 640, roi_top: 220, roi_width: 640, roi_height: 640 } } : {}),
          pipeline: { fire_delay_enabled: false, fire_delay_ms: 0, projection_fov_x_deg: 90, ...(running ? { target_class_filter: "0,1", target_class_weights: "0:0.8,1:0.5" } : {}) },
          control: { output_enabled: running },
          inference: { ...(running ? { detection_class_filter: "0,1", detection_class_filters: { default: "0,1" } } : {}), detection_custom_presets: [{
            id: "preview_three_classes", label: "预览示例 · 三类", note: "模拟类别映射，仅供预览，不会写入设备。",
            verified: false, class_names: ["头部", "身体", "队友"], enabled_ids: [0, 1],
            weights: { 0: 0.8, 1: 0.5 }, roles: { 0: "head", 1: "body" }
          }] },
          hardware: { backend: "null", active_mouse_protocol_id: "empty", mouse_protocols: running ? [{ id: "preview-net", model: "net", host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4" }] : [], auto_connect: true },
        }));
  const runtime = running
    ? scenario === "running-no-target"
      ? {
        ...previewRunningRuntime,
        vision: {
          ...previewRunningRuntime.vision,
          detections: 0,
          detection_items: [],
          target: null,
          target_pipeline: { ...previewRunningRuntime.vision.target_pipeline, code: "no_target", counts: { raw_candidates: 0, eligible_candidates: 0, selected_targets: 0 } },
          output_trace: previewRunningRuntime.vision.output_trace,
        },
      } as RuntimeState
      : previewRunningRuntime
    : previewRuntime;
  return (
    <SafetyOperationProvider>
      <StudioConsoleView
        previewOnly
        license={{ valid: true, configured: true, credential_format: "ephemeral_code", temporary_access_supported: true, tier: "temporary", device_model: "桌面预览（模拟设备）", expires_at: previewLicenseDeadline, features: ["hardware_control"], message: "模拟授权，仅用于界面预览。" } as LicenseStatus}
        health={scenario === "offline" ? null : { ok: true }}
        runtime={scenario === "offline" ? null : runtime}
        runtimeConfig={config}
        projects={[]}
        errors={{}}
        lastUpdated={running ? new Date() : null}
        realtimeStatus={scenario === "offline" ? "disconnected" : scenario === "stale" ? "stale" : "connected"}
        onEnsureProjects={async () => []}
        onLicenseChange={() => undefined}
        onRefresh={async () => undefined}
        onRuntimeConfigChange={setConfig}
        onRuntimeStateChange={() => true}
      />
    </SafetyOperationProvider>
  );
}
