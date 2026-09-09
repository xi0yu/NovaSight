import type { ActiveModel } from "./model";

export type RuntimeSubsystemState =
  | "stopped"
  | "starting"
  | "ready"
  | "running"
  | "degraded"
  | "stopping"
  | "failed"
  | "unavailable";

export interface RuntimeErrorSummary {
  code: string;
  message: string;
  subsystem: string | null;
}

export interface ExecutorAvailability {
  available: boolean;
  configuration_state: "not_applicable" | "uncommissioned" | "restart_required" | "ready";
  configuration_ready: boolean;
  restart_required: boolean;
  can_connect: boolean;
  can_disconnect: boolean;
  blocked_reason: string | null;
  connected: boolean;
  runtime_connected: boolean;
  connecting: boolean;
  buttons_available: boolean;
  button_left: boolean;
  button_right: boolean;
  connection_state:
    | "not_applicable"
    | "uncommissioned"
    | "connecting"
    | "connected"
    | "failed"
    | "degraded"
    | "disconnecting"
    | "stopped";
  retryable: boolean;
  last_error: string | null;
  managed_by_runtime: boolean;
  accepted_command_count: number;
  last_accepted_dx: number | null;
  last_accepted_dy: number | null;
  diagnostic_move_count: number;
  last_diagnostic_dx: number | null;
  last_diagnostic_dy: number | null;
  device_error_count: number;
  device_recovery_count: number;
  last_device_error: string | null;
}

export interface ExecutorStatus {
  selected: string;
  executors: Record<string, ExecutorAvailability>;
  state: RuntimeSubsystemState;
  last_error: RuntimeErrorSummary | null;
}

export interface CaptureProfile {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
  preference: string;
  source: "configured";
}

export interface CaptureState {
  available: boolean;
  running: boolean;
  state: RuntimeSubsystemState;
  device: string;
  backend: string | null;
  profile: CaptureProfile | null;
  last_error: string | null;
}

export interface Statistics {
  nvinfer_input_counter: number;
  detection_batch_counter: number;
  detection_batch_consumed_counter: number;
  targeting_batch_counter: number;
  nvinfer_input_fps: number | null;
  nvinfer_output_fps: number | null;
  detection_batch_fps: number | null;
  targeting_batch_fps: number | null;
  detection_data_age_ms: number | null;
  detection_freshness_threshold_ms: number | null;
  inference_latency_ms: number | null;
  inference_latency_samples: number;
  telemetry_window_ms: number | null;
  metrics_available: boolean;
}

export interface RuntimeInferenceState {
  available: boolean;
  configured: boolean;
  loaded: boolean;
  running: boolean;
  terminal_error: boolean;
  state: RuntimeSubsystemState;
  selected: string | null;
  reason: string | null;
  detail: string | null;
  inference_reason: string | null;
  input_frames: number;
  output_buffers: number;
  metadata_extractions: number;
  published_batches: number;
  timestamp_buffer_pts_matches: number;
  timestamp_frame_meta_pts_matches: number;
  timestamp_correlation_misses: number;
  sampled_detection_generation: number | null;
  preview_enabled: boolean;
  preview_active: boolean;
  preview_encoder_active: boolean;
  preview_consumers: number;
  preview_available: boolean;
  preview_sequence: number;
  preview_reason: string;
  preview_transport: string | null;
  postprocess: RuntimePostprocessState | null;
}

export interface RuntimePostprocessState {
  confidence_threshold: number;
  nms_threshold: number;
}

export interface PreviewSnapshotState {
  preview_enabled: boolean;
  preview_running: boolean;
  preview_active: boolean;
  preview_encoder_active: boolean;
  preview_consumers: number;
  preview_available: boolean;
  preview_sequence: number;
  preview_reason: string;
  preview_transport: string;
}

export interface RuntimeDeepStreamState {
  running: boolean;
  terminal_error: boolean;
  last_error: string | null;
  input_frames: number;
  metadata_extractions: number;
  published_batches: number;
  crosshair_active: boolean;
  crosshair_reason: string;
}

export interface RuntimePipelineSummary {
  running: boolean;
  state: "stopped" | "starting" | "running" | "standby" | "stopping" | "faulted";
  epoch: number | null;
  started_at_ms: number | null;
  mode: string;
  last_error: RuntimeErrorSummary | null;
  deepstream: RuntimeDeepStreamState;
}

export interface RuntimeConfigSummary {
  version: number;
  schema_version: number;
  effective_version: number;
  restart_required: boolean;
}

export interface RuntimeOutputTraceState {
  code: string;
  state: "ready" | "blocked" | "waiting" | "idle";
  detail: string;
  next_action: string;
}

export interface CrosshairObservation {
  status: string;
  found: boolean;
  x: number;
  y: number;
  confidence: number;
  sample_ts_ns: number;
  template_id: string;
  point_hits: number;
  point_count: number;
  offset_x: number;
  offset_y: number;
  reason: string;
}

export interface CrosshairTemplateSummary {
  id: string;
  schema_version: number;
  created_ts_ns: number;
  geometry_signature: string;
  search_size: number;
  point_count: number;
}

export interface CrosshairSnapshot {
  enabled: boolean;
  use_for_control: boolean;
  running: boolean;
  state: string;
  observation: CrosshairObservation | null;
  control_reference_ready: boolean;
  control_reference_source: string;
  control_reference_reason: string;
  recent_samples: number;
  required_samples: number;
  processed_frames: number;
  matched_frames: number;
  dropped_frames: number;
  last_error: string;
  template: CrosshairTemplateSummary | null;
}

export interface CrosshairLearnResponse {
  learned: true;
  template: CrosshairTemplateSummary;
  status: CrosshairSnapshot;
}

export interface RuntimeVisionInferenceState {
  input_width: number | null;
  input_height: number | null;
  generation: number | null;
  model_input_width: number | null;
  model_input_height: number | null;
  source_width: number | null;
  source_height: number | null;
  roi_offset_x: number | null;
  roi_offset_y: number | null;
  roi_width: number | null;
  roi_height: number | null;
}

export interface RuntimeVisionDetectionState {
  object_id: number;
  class_id: number;
  cls: number;
  score: number;
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
}

export interface RuntimeVisionTargetState {
  target_detection_index: number | null;
  track_id: number;
  class_id: number;
  cls: number;
  score: number;
  identity_confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  box_cx: number;
  box_cy: number;
  cx: number;
  cy: number;
  observed_aim_x: number;
  observed_aim_y: number;
}

export interface RuntimeTargetPipelineCounts {
  raw_candidates: number;
  eligible_candidates: number;
  selected_targets: number;
}

export interface RuntimeTargetPipelineState {
  code: string;
  stage: string;
  message: string;
  rejection_reasons: string[];
  counts: RuntimeTargetPipelineCounts;
}

export interface RuntimeCandidateFilterState {
  effective_class_filter: string;
  basic: {
    raw_candidates: number;
    filtered_candidates: number;
    rejected_candidates: number;
    rejected_class_ids: number[];
  };
}

export interface RuntimeMouseObservationState {
  control_width_px: number | null;
  control_height_px: number | null;
  observed_x_px: number | null;
  observed_y_px: number | null;
  predicted_x_px: number | null;
  predicted_y_px: number | null;
  measurement_dt_s: number | null;
}

export type RuntimePredictionMotionState = "continuous" | "stationary" | "unstable" | "unavailable";

export type RuntimeOutputDeliveryState =
  | "idle"
  | "gate_closed"
  | "device_disabled"
  | "trigger_inactive"
  | "generation_fenced"
  | "no_movement"
  | "superseded"
  | "sent"
  | "send_failed";

export type RuntimeRecoilState = "IDLE" | "WAITING" | "READY" | "APPLIED";

export type RuntimeRecoilBlockReason =
  | "RECOIL_DISABLED"
  | "FIRING_INACTIVE"
  | "TARGET_REQUIRED"
  | "INTERVAL_PENDING"
  | "OUTPUT_SATURATED"
  | "";

export interface RuntimeControlPipelineState {
  control_mode: "continuous_atan_medoid_v2";
  movement_strategy: "latest_replace";
  output_delivery_state: RuntimeOutputDeliveryState;
  mode: "CONTINUOUS" | null;
  frame_age_ms: number | null;
  history_position_count: number | null;
  velocity_1: number | null;
  velocity_2: number | null;
  velocity_3: number | null;
  medoid_velocity: number | null;
  prediction_velocity: number | null;
  motion_state: RuntimePredictionMotionState | null;
  measurement_dt_s: number | null;
  reference_dt_ms: number | null;
  prediction_actuation_delay_ms: number | null;
  prediction_lead_ms: number | null;
  prediction_horizon_ms: number | null;
  prediction_raw_offset_x: number | null;
  prediction_allowed_cap_x: number | null;
  prediction_safe_offset_x: number | null;
  prediction_allowed: boolean | null;
  velocity_y_1: number | null;
  velocity_y_2: number | null;
  velocity_y_3: number | null;
  medoid_velocity_y: number | null;
  prediction_velocity_y: number | null;
  motion_state_y: RuntimePredictionMotionState | null;
  measurement_dt_y_s: number | null;
  reference_dt_y_ms: number | null;
  prediction_raw_offset_y: number | null;
  prediction_allowed_cap_y: number | null;
  prediction_safe_offset_y: number | null;
  prediction_allowed_y: boolean | null;
  observed_error_x_px: number | null;
  observed_error_y_px: number | null;
  predicted_error_x_px: number | null;
  predicted_error_y_px: number | null;
  full_error_counts_x: number | null;
  full_error_counts_y: number | null;
  float_demand_x: number | null;
  float_demand_y: number | null;
  integer_command_x: number | null;
  integer_command_y: number | null;
  quantizer_residual_x: number | null;
  quantizer_residual_y: number | null;
  block_reason: string | null;
  fire_delay_enabled: boolean;
  fire_delay_configured_ms: number;
  fire_delay_pending: boolean;
  fire_delay_elapsed_ms: number | null;
  fire_delay_remaining_ms: number | null;
  recoil_mode: "interval_additive" | "target_guarded_interval_additive";
  recoil_enabled: boolean;
  recoil_active: boolean;
  recoil_state: RuntimeRecoilState;
  recoil_interval_ms: number;
  recoil_y_counts: number;
  recoil_elapsed_since_output_ms: number | null;
  recoil_remaining_ms: number;
  recoil_requested_counts_y: number;
  recoil_emitted_counts_y: number;
  recoil_source_generation: number | null;
  recoil_block_reason: RuntimeRecoilBlockReason;
}

export interface RuntimeVisionControlState {
  global_state: "WAITING_TRIGGER_DELAY" | "CALCULATED" | "IDLE";
  output_enabled: boolean;
  aim_x: number | null;
  aim_y: number | null;
  dx: number | null;
  dy: number | null;
  will_emit: boolean | null;
  trigger_active: boolean | null;
  reason: string | null;
  no_send_reason: string | null;
  candidates: number;
  selector_state: "LOCKED" | "SEARCHING";
  selection_reason: "PREFERRED_CLASS" | "FALLBACK_CLASS" | null;
  candidate_filter: RuntimeCandidateFilterState;
  mouse_observation: RuntimeMouseObservationState;
  pipeline: RuntimeControlPipelineState;
}

export interface RuntimeVisionState {
  crosshair: CrosshairSnapshot | null;
  inference: RuntimeVisionInferenceState;
  detections: number;
  detection_items: RuntimeVisionDetectionState[];
  detection_items_truncated: number;
  target: RuntimeVisionTargetState | null;
  target_pipeline: RuntimeTargetPipelineState;
  output_trace: RuntimeOutputTraceState;
  control: RuntimeVisionControlState;
}

export interface RuntimeSemanticState {
  daemon_instance_id: string;
  phase: "stopped" | "starting" | "waiting_model" | "running" | "standby" | "stopping" | "faulted";
  perception_phase: "unavailable" | "stopped" | "starting" | "waiting_model" | "running" | "faulted";
  epoch: number | null;
  snapshot_sequence: number;
  snapshot_updated_at_ms: number;
}

export interface RuntimeState {
  semantic: RuntimeSemanticState;
  running: boolean;
  source: string;
  active_model: ActiveModel | null;
  model_catalog_error: string | null;
  executor: ExecutorStatus;
  capture: CaptureState;
  statistics: Statistics;
  inference: RuntimeInferenceState;
  config: RuntimeConfigSummary;
  pipeline: RuntimePipelineSummary;
  vision: RuntimeVisionState;
  fatal_error: RuntimeErrorSummary | null;
}

export type RuntimeStatusTopic = "summary" | "capture" | "infer" | "control" | "latency";

export interface RuntimeStatusFrame {
  kind: "runtime_snapshot";
  topic: RuntimeStatusTopic | "full";
  full: boolean;
  /** Every frame owns a complete top-level runtime state; topic only prunes detection detail. */
  state: RuntimeState;
}

export interface RuntimeStatusHeartbeat {
  kind: "runtime_heartbeat";
  daemon_instance_id: string;
  snapshot_sequence: number;
}

export type RuntimeStatusMessage = RuntimeStatusFrame | RuntimeStatusHeartbeat | RuntimeState;

type JsonRecord = Record<string, unknown>;
type Assertion = (value: unknown, path: string) => void;

const SUBSYSTEM_STATES = new Set<RuntimeSubsystemState>([
  "stopped", "starting", "ready", "running", "degraded", "stopping", "failed", "unavailable"
]);
const PIPELINE_STATES = new Set<RuntimePipelineSummary["state"]>([
  "stopped", "starting", "running", "standby", "stopping", "faulted"
]);
const OUTPUT_TRACE_STATES = new Set<RuntimeOutputTraceState["state"]>([
  "ready", "blocked", "waiting", "idle"
]);
const STATUS_TOPICS = new Set<RuntimeStatusFrame["topic"]>([
  "summary", "capture", "infer", "control", "latency", "full"
]);

export class RuntimeContractError extends Error {
  constructor(path: string, expected: string) {
    super(`运行时数据契约错误：${path} 应为 ${expected}`);
    this.name = "RuntimeContractError";
  }
}

function expectRecord(value: unknown, path: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new RuntimeContractError(path, "object");
  }
  return value as JsonRecord;
}

function expectString(value: unknown, path: string): void {
  if (typeof value !== "string") throw new RuntimeContractError(path, "string");
}

function expectBoolean(value: unknown, path: string): void {
  if (typeof value !== "boolean") throw new RuntimeContractError(path, "boolean");
}

function expectFiniteNumber(value: unknown, path: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RuntimeContractError(path, "finite number");
  }
}

function expectInteger(value: unknown, path: string): void {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new RuntimeContractError(path, "safe integer");
  }
}

function expectUnsignedInteger(value: unknown, path: string): void {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new RuntimeContractError(path, "unsigned safe integer");
  }
}

function expectNullable(value: unknown, path: string, assertion: Assertion): void {
  if (value !== null) assertion(value, path);
}

function expectArray(value: unknown, path: string, assertion: Assertion): void {
  if (!Array.isArray(value)) throw new RuntimeContractError(path, "array");
  value.forEach((item, index) => assertion(item, `${path}[${index}]`));
}

function expectLiteral<T extends string>(value: unknown, path: string, allowed: ReadonlySet<T>): void {
  if (typeof value !== "string" || !allowed.has(value as T)) {
    throw new RuntimeContractError(path, [...allowed].join(" | "));
  }
}

function assertRuntimeError(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectString(record.code, `${path}.code`);
  expectString(record.message, `${path}.message`);
  expectNullable(record.subsystem, `${path}.subsystem`, expectString);
}

function assertModelProject(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectInteger(record.id, `${path}.id`);
  expectString(record.name, `${path}.name`);
  expectString(record.description, `${path}.description`);
}

function assertModelVersion(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectInteger(record.id, `${path}.id`);
  expectInteger(record.project_id, `${path}.project_id`);
  expectString(record.version, `${path}.version`);
  expectString(record.source_kind, `${path}.source_kind`);
  expectString(record.source_path, `${path}.source_path`);
  expectArray(record.classes, `${path}.classes`, expectString);
  expectString(record.input_shape, `${path}.input_shape`);
}

function assertModelArtifact(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectInteger(record.id, `${path}.id`);
  expectInteger(record.version_id, `${path}.version_id`);
  expectString(record.kind, `${path}.kind`);
  expectString(record.path, `${path}.path`);
  expectString(record.checksum, `${path}.checksum`);
  expectString(record.status, `${path}.status`);
  expectNullable(record.size_bytes, `${path}.size_bytes`, expectUnsignedInteger);
}

function assertActiveModel(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  const deployment = expectRecord(record.deployment, `${path}.deployment`);
  expectInteger(deployment.id, `${path}.deployment.id`);
  expectInteger(deployment.project_id, `${path}.deployment.project_id`);
  expectInteger(deployment.artifact_id, `${path}.deployment.artifact_id`);
  expectNullable(deployment.previous_artifact_id, `${path}.deployment.previous_artifact_id`, expectInteger);
  expectInteger(deployment.updated_seq, `${path}.deployment.updated_seq`);
  assertModelProject(record.project, `${path}.project`);
  assertModelVersion(record.version, `${path}.version`);
  assertModelArtifact(record.artifact, `${path}.artifact`);
  expectString(record.artifact_path, `${path}.artifact_path`);
}

function assertExecutorAvailability(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  ["available", "configuration_ready", "restart_required", "can_connect", "can_disconnect",
    "connected", "runtime_connected", "connecting", "buttons_available", "button_left",
    "button_right", "retryable", "managed_by_runtime"].forEach((key) =>
    expectBoolean(record[key], `${path}.${key}`));
  expectLiteral(record.configuration_state, `${path}.configuration_state`, new Set([
    "not_applicable", "uncommissioned", "restart_required", "ready"
  ]));
  expectLiteral(record.connection_state, `${path}.connection_state`, new Set([
    "not_applicable", "uncommissioned", "connecting", "connected", "failed", "degraded",
    "disconnecting", "stopped"
  ]));
  ["blocked_reason", "last_error", "last_device_error"].forEach((key) =>
    expectNullable(record[key], `${path}.${key}`, expectString));
  ["accepted_command_count", "diagnostic_move_count", "device_error_count", "device_recovery_count"]
    .forEach((key) => expectUnsignedInteger(record[key], `${path}.${key}`));
  ["last_accepted_dx", "last_accepted_dy", "last_diagnostic_dx", "last_diagnostic_dy"]
    .forEach((key) => expectNullable(record[key], `${path}.${key}`, expectInteger));
}

function assertExecutor(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectString(record.selected, `${path}.selected`);
  const executors = expectRecord(record.executors, `${path}.executors`);
  Object.entries(executors).forEach(([key, executor]) =>
    assertExecutorAvailability(executor, `${path}.executors.${key}`));
  expectLiteral(record.state, `${path}.state`, SUBSYSTEM_STATES);
  expectNullable(record.last_error, `${path}.last_error`, assertRuntimeError);
}

function assertCapture(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectBoolean(record.available, `${path}.available`);
  expectBoolean(record.running, `${path}.running`);
  expectLiteral(record.state, `${path}.state`, SUBSYSTEM_STATES);
  expectString(record.device, `${path}.device`);
  expectNullable(record.backend, `${path}.backend`, expectString);
  expectNullable(record.last_error, `${path}.last_error`, expectString);
  expectNullable(record.profile, `${path}.profile`, (profile, profilePath) => {
    const item = expectRecord(profile, profilePath);
    expectString(item.pixel_format, `${profilePath}.pixel_format`);
    expectUnsignedInteger(item.width, `${profilePath}.width`);
    expectUnsignedInteger(item.height, `${profilePath}.height`);
    expectUnsignedInteger(item.fps, `${profilePath}.fps`);
    expectString(item.preference, `${profilePath}.preference`);
    if (item.source !== "configured") throw new RuntimeContractError(`${profilePath}.source`, "configured");
  });
}

function assertStatistics(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  ["nvinfer_input_counter", "detection_batch_counter", "detection_batch_consumed_counter",
    "targeting_batch_counter", "inference_latency_samples"].forEach((key) =>
    expectUnsignedInteger(record[key], `${path}.${key}`));
  ["nvinfer_input_fps", "nvinfer_output_fps", "detection_batch_fps", "targeting_batch_fps",
    "detection_data_age_ms", "detection_freshness_threshold_ms", "inference_latency_ms"]
    .forEach((key) => expectNullable(record[key], `${path}.${key}`, expectFiniteNumber));
  expectNullable(record.telemetry_window_ms, `${path}.telemetry_window_ms`, expectUnsignedInteger);
  expectBoolean(record.metrics_available, `${path}.metrics_available`);
}

function assertInference(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  ["available", "configured", "loaded", "running", "terminal_error", "preview_enabled",
    "preview_active", "preview_encoder_active", "preview_available"].forEach((key) =>
    expectBoolean(record[key], `${path}.${key}`));
  expectLiteral(record.state, `${path}.state`, SUBSYSTEM_STATES);
  ["selected", "reason", "detail", "inference_reason", "preview_transport"].forEach((key) =>
    expectNullable(record[key], `${path}.${key}`, expectString));
  ["input_frames", "output_buffers", "metadata_extractions", "published_batches",
    "timestamp_buffer_pts_matches", "timestamp_frame_meta_pts_matches", "timestamp_correlation_misses",
    "preview_consumers", "preview_sequence"].forEach((key) =>
    expectUnsignedInteger(record[key], `${path}.${key}`));
  expectNullable(record.sampled_detection_generation, `${path}.sampled_detection_generation`, expectUnsignedInteger);
  expectString(record.preview_reason, `${path}.preview_reason`);
  expectNullable(record.postprocess, `${path}.postprocess`, (postprocess, postprocessPath) => {
    const item = expectRecord(postprocess, postprocessPath);
    expectFiniteNumber(item.confidence_threshold, `${postprocessPath}.confidence_threshold`);
    expectFiniteNumber(item.nms_threshold, `${postprocessPath}.nms_threshold`);
  });
}

function assertPipeline(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectBoolean(record.running, `${path}.running`);
  expectLiteral(record.state, `${path}.state`, PIPELINE_STATES);
  expectNullable(record.epoch, `${path}.epoch`, expectUnsignedInteger);
  expectNullable(record.started_at_ms, `${path}.started_at_ms`, expectUnsignedInteger);
  expectString(record.mode, `${path}.mode`);
  expectNullable(record.last_error, `${path}.last_error`, assertRuntimeError);
  const deepstream = expectRecord(record.deepstream, `${path}.deepstream`);
  expectBoolean(deepstream.running, `${path}.deepstream.running`);
  expectBoolean(deepstream.terminal_error, `${path}.deepstream.terminal_error`);
  expectNullable(deepstream.last_error, `${path}.deepstream.last_error`, expectString);
  ["input_frames", "metadata_extractions", "published_batches"].forEach((key) =>
    expectUnsignedInteger(deepstream[key], `${path}.deepstream.${key}`));
  expectBoolean(deepstream.crosshair_active, `${path}.deepstream.crosshair_active`);
  expectString(deepstream.crosshair_reason, `${path}.deepstream.crosshair_reason`);
}

function assertCrosshair(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  ["enabled", "use_for_control", "running", "control_reference_ready"].forEach((key) =>
    expectBoolean(record[key], `${path}.${key}`));
  ["state", "control_reference_source", "control_reference_reason", "last_error"].forEach((key) =>
    expectString(record[key], `${path}.${key}`));
  ["recent_samples", "required_samples", "processed_frames", "matched_frames", "dropped_frames"]
    .forEach((key) => expectUnsignedInteger(record[key], `${path}.${key}`));
  expectNullable(record.observation, `${path}.observation`, (observation, observationPath) => {
    const item = expectRecord(observation, observationPath);
    ["status", "template_id", "reason"].forEach((key) => expectString(item[key], `${observationPath}.${key}`));
    expectBoolean(item.found, `${observationPath}.found`);
    ["x", "y", "confidence", "offset_x", "offset_y"].forEach((key) =>
      expectFiniteNumber(item[key], `${observationPath}.${key}`));
    ["sample_ts_ns", "point_hits", "point_count"].forEach((key) =>
      expectUnsignedInteger(item[key], `${observationPath}.${key}`));
  });
  expectNullable(record.template, `${path}.template`, assertCrosshairTemplate);
}

function assertCrosshairTemplate(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  ["id", "geometry_signature"].forEach((key) => expectString(record[key], `${path}.${key}`));
  ["schema_version", "created_ts_ns", "search_size", "point_count"].forEach((key) =>
    expectUnsignedInteger(record[key], `${path}.${key}`));
}

function assertPreviewSnapshot(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  ["preview_enabled", "preview_running", "preview_active", "preview_encoder_active",
    "preview_available"].forEach((key) => expectBoolean(record[key], `${path}.${key}`));
  ["preview_consumers", "preview_sequence"].forEach((key) =>
    expectUnsignedInteger(record[key], `${path}.${key}`));
  expectString(record.preview_reason, `${path}.preview_reason`);
  expectString(record.preview_transport, `${path}.preview_transport`);
}

function assertVision(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectNullable(record.crosshair, `${path}.crosshair`, assertCrosshair);
  const inference = expectRecord(record.inference, `${path}.inference`);
  ["input_width", "input_height", "generation", "model_input_width", "model_input_height",
    "source_width", "source_height", "roi_offset_x", "roi_offset_y", "roi_width", "roi_height"]
    .forEach((key) => expectNullable(inference[key], `${path}.inference.${key}`, expectUnsignedInteger));
  expectUnsignedInteger(record.detections, `${path}.detections`);
  expectArray(record.detection_items, `${path}.detection_items`, (detection, detectionPath) => {
    const item = expectRecord(detection, detectionPath);
    ["object_id", "class_id", "cls"].forEach((key) => expectUnsignedInteger(item[key], `${detectionPath}.${key}`));
    ["score", "x", "y", "w", "h", "cx", "cy"].forEach((key) =>
      expectFiniteNumber(item[key], `${detectionPath}.${key}`));
  });
  expectUnsignedInteger(record.detection_items_truncated, `${path}.detection_items_truncated`);
  expectNullable(record.target, `${path}.target`, (target, targetPath) => {
    const item = expectRecord(target, targetPath);
    expectNullable(item.target_detection_index, `${targetPath}.target_detection_index`, expectUnsignedInteger);
    ["track_id", "class_id", "cls"].forEach((key) => expectUnsignedInteger(item[key], `${targetPath}.${key}`));
    ["score", "identity_confidence", "x1", "y1", "x2", "y2", "box_cx", "box_cy", "cx", "cy",
      "observed_aim_x", "observed_aim_y"].forEach((key) => expectFiniteNumber(item[key], `${targetPath}.${key}`));
  });
  const targetPipeline = expectRecord(record.target_pipeline, `${path}.target_pipeline`);
  ["code", "stage", "message"].forEach((key) => expectString(targetPipeline[key], `${path}.target_pipeline.${key}`));
  expectArray(targetPipeline.rejection_reasons, `${path}.target_pipeline.rejection_reasons`, expectString);
  const counts = expectRecord(targetPipeline.counts, `${path}.target_pipeline.counts`);
  ["raw_candidates", "eligible_candidates", "selected_targets"].forEach((key) =>
    expectUnsignedInteger(counts[key], `${path}.target_pipeline.counts.${key}`));
  const outputTrace = expectRecord(record.output_trace, `${path}.output_trace`);
  expectString(outputTrace.code, `${path}.output_trace.code`);
  expectLiteral(outputTrace.state, `${path}.output_trace.state`, OUTPUT_TRACE_STATES);
  expectString(outputTrace.detail, `${path}.output_trace.detail`);
  expectString(outputTrace.next_action, `${path}.output_trace.next_action`);
  assertVisionControl(record.control, `${path}.control`);
}

function assertVisionControl(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  expectLiteral(record.global_state, `${path}.global_state`, new Set(["WAITING_TRIGGER_DELAY", "CALCULATED", "IDLE"]));
  expectBoolean(record.output_enabled, `${path}.output_enabled`);
  ["aim_x", "aim_y"].forEach((key) => expectNullable(record[key], `${path}.${key}`, expectFiniteNumber));
  ["dx", "dy"].forEach((key) => expectNullable(record[key], `${path}.${key}`, expectInteger));
  ["will_emit", "trigger_active"].forEach((key) => expectNullable(record[key], `${path}.${key}`, expectBoolean));
  ["reason", "no_send_reason"].forEach((key) =>
    expectNullable(record[key], `${path}.${key}`, expectString));
  expectNullable(record.selection_reason, `${path}.selection_reason`, (item, itemPath) =>
    expectLiteral(item, itemPath, new Set(["PREFERRED_CLASS", "FALLBACK_CLASS"])));
  expectUnsignedInteger(record.candidates, `${path}.candidates`);
  expectLiteral(record.selector_state, `${path}.selector_state`, new Set(["LOCKED", "SEARCHING"]));
  const candidateFilter = expectRecord(record.candidate_filter, `${path}.candidate_filter`);
  expectString(candidateFilter.effective_class_filter, `${path}.candidate_filter.effective_class_filter`);
  const basic = expectRecord(candidateFilter.basic, `${path}.candidate_filter.basic`);
  ["raw_candidates", "filtered_candidates", "rejected_candidates"].forEach((key) =>
    expectUnsignedInteger(basic[key], `${path}.candidate_filter.basic.${key}`));
  expectArray(basic.rejected_class_ids, `${path}.candidate_filter.basic.rejected_class_ids`, expectUnsignedInteger);
  const observation = expectRecord(record.mouse_observation, `${path}.mouse_observation`);
  ["control_width_px", "control_height_px"].forEach((key) =>
    expectNullable(observation[key], `${path}.mouse_observation.${key}`, expectUnsignedInteger));
  ["observed_x_px", "observed_y_px", "predicted_x_px", "predicted_y_px", "measurement_dt_s"]
    .forEach((key) => expectNullable(observation[key], `${path}.mouse_observation.${key}`, expectFiniteNumber));
  assertControlPipeline(record.pipeline, `${path}.pipeline`);
}

function assertControlPipeline(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  if (record.control_mode !== "continuous_atan_medoid_v2") {
    throw new RuntimeContractError(`${path}.control_mode`, "continuous_atan_medoid_v2");
  }
  if (record.movement_strategy !== "latest_replace") {
    throw new RuntimeContractError(`${path}.movement_strategy`, "latest_replace");
  }
  expectLiteral(record.output_delivery_state, `${path}.output_delivery_state`, new Set([
    "idle", "gate_closed", "device_disabled", "trigger_inactive", "generation_fenced",
    "no_movement", "superseded", "sent", "send_failed"
  ]));
  expectNullable(record.mode, `${path}.mode`, (item, itemPath) =>
    expectLiteral(item, itemPath, new Set(["CONTINUOUS"])));
  const nullableNumbers = [
    "frame_age_ms", "velocity_1", "velocity_2", "velocity_3", "medoid_velocity",
    "prediction_velocity", "measurement_dt_s", "reference_dt_ms", "prediction_actuation_delay_ms",
    "prediction_lead_ms", "prediction_horizon_ms", "prediction_raw_offset_x", "prediction_allowed_cap_x",
    "prediction_safe_offset_x", "velocity_y_1", "velocity_y_2", "velocity_y_3", "medoid_velocity_y",
    "prediction_velocity_y", "measurement_dt_y_s", "reference_dt_y_ms", "prediction_raw_offset_y",
    "prediction_allowed_cap_y", "prediction_safe_offset_y", "observed_error_x_px", "observed_error_y_px",
    "predicted_error_x_px", "predicted_error_y_px", "full_error_counts_x", "full_error_counts_y",
    "float_demand_x", "float_demand_y", "quantizer_residual_x", "quantizer_residual_y",
    "fire_delay_elapsed_ms", "fire_delay_remaining_ms", "recoil_elapsed_since_output_ms"
  ];
  nullableNumbers.forEach((key) => expectNullable(record[key], `${path}.${key}`, expectFiniteNumber));
  ["history_position_count", "recoil_source_generation"]
    .forEach((key) => expectNullable(record[key], `${path}.${key}`, expectUnsignedInteger));
  ["integer_command_x", "integer_command_y"]
    .forEach((key) => expectNullable(record[key], `${path}.${key}`, expectInteger));
  ["prediction_allowed", "prediction_allowed_y"].forEach((key) =>
    expectNullable(record[key], `${path}.${key}`, expectBoolean));
  ["motion_state", "motion_state_y"].forEach((key) =>
    expectNullable(record[key], `${path}.${key}`, (item, itemPath) =>
      expectLiteral(item, itemPath, new Set(["continuous", "stationary", "unstable", "unavailable"]))));
  expectNullable(record.block_reason, `${path}.block_reason`, expectString);
  ["fire_delay_enabled", "fire_delay_pending", "recoil_enabled", "recoil_active"].forEach((key) =>
    expectBoolean(record[key], `${path}.${key}`));
  ["fire_delay_configured_ms", "recoil_interval_ms"].forEach((key) =>
    expectUnsignedInteger(record[key], `${path}.${key}`));
  ["recoil_y_counts", "recoil_requested_counts_y", "recoil_emitted_counts_y"].forEach((key) =>
    expectInteger(record[key], `${path}.${key}`));
  expectFiniteNumber(record.recoil_remaining_ms, `${path}.recoil_remaining_ms`);
  expectLiteral(record.recoil_mode, `${path}.recoil_mode`, new Set([
    "interval_additive", "target_guarded_interval_additive"
  ]));
  expectLiteral(record.recoil_state, `${path}.recoil_state`, new Set(["IDLE", "WAITING", "READY", "APPLIED"]));
  expectLiteral(record.recoil_block_reason, `${path}.recoil_block_reason`, new Set([
    "RECOIL_DISABLED", "FIRING_INACTIVE", "TARGET_REQUIRED", "INTERVAL_PENDING", "OUTPUT_SATURATED", ""
  ]));
}

function assertRuntimeState(value: unknown, path: string): void {
  const record = expectRecord(value, path);
  const semantic = expectRecord(record.semantic, `${path}.semantic`);
  expectString(semantic.daemon_instance_id, `${path}.semantic.daemon_instance_id`);
  if ((semantic.daemon_instance_id as string).trim() === "") {
    throw new RuntimeContractError(`${path}.semantic.daemon_instance_id`, "non-empty string");
  }
  expectLiteral(semantic.phase, `${path}.semantic.phase`, new Set([
    "stopped", "starting", "waiting_model", "running", "standby", "stopping", "faulted"
  ]));
  expectLiteral(semantic.perception_phase, `${path}.semantic.perception_phase`, new Set([
    "unavailable", "stopped", "starting", "waiting_model", "running", "faulted"
  ]));
  expectNullable(semantic.epoch, `${path}.semantic.epoch`, expectUnsignedInteger);
  expectUnsignedInteger(semantic.snapshot_sequence, `${path}.semantic.snapshot_sequence`);
  expectUnsignedInteger(semantic.snapshot_updated_at_ms, `${path}.semantic.snapshot_updated_at_ms`);
  expectBoolean(record.running, `${path}.running`);
  expectString(record.source, `${path}.source`);
  expectNullable(record.active_model, `${path}.active_model`, assertActiveModel);
  expectNullable(record.model_catalog_error, `${path}.model_catalog_error`, expectString);
  assertExecutor(record.executor, `${path}.executor`);
  assertCapture(record.capture, `${path}.capture`);
  assertStatistics(record.statistics, `${path}.statistics`);
  assertInference(record.inference, `${path}.inference`);
  const config = expectRecord(record.config, `${path}.config`);
  ["version", "schema_version", "effective_version"].forEach((key) =>
    expectUnsignedInteger(config[key], `${path}.config.${key}`));
  expectBoolean(config.restart_required, `${path}.config.restart_required`);
  assertPipeline(record.pipeline, `${path}.pipeline`);
  assertVision(record.vision, `${path}.vision`);
  expectNullable(record.fatal_error, `${path}.fatal_error`, assertRuntimeError);
}

export function decodeRuntimeState(value: unknown): RuntimeState {
  assertRuntimeState(value, "runtime");
  return value as RuntimeState;
}

export function decodeCaptureState(value: unknown): CaptureState {
  assertCapture(value, "capture");
  return value as CaptureState;
}

export function decodePreviewSnapshot(value: unknown): PreviewSnapshotState {
  assertPreviewSnapshot(value, "preview");
  return value as PreviewSnapshotState;
}

export function decodeCrosshairSnapshot(value: unknown): CrosshairSnapshot {
  assertCrosshair(value, "crosshair");
  return value as CrosshairSnapshot;
}

export function decodeCrosshairLearnResponse(value: unknown): CrosshairLearnResponse {
  const record = expectRecord(value, "crosshair_learn");
  if (record.learned !== true) {
    throw new RuntimeContractError("crosshair_learn.learned", "true");
  }
  assertCrosshairTemplate(record.template, "crosshair_learn.template");
  assertCrosshair(record.status, "crosshair_learn.status");
  return value as CrosshairLearnResponse;
}

export function decodeRuntimeStatusMessage(value: unknown): RuntimeStatusMessage {
  const record = expectRecord(value, "status_message");
  if (record.kind === "runtime_heartbeat") {
    expectString(record.daemon_instance_id, "status_message.daemon_instance_id");
    if ((record.daemon_instance_id as string).trim() === "") {
      throw new RuntimeContractError("status_message.daemon_instance_id", "non-empty string");
    }
    expectUnsignedInteger(record.snapshot_sequence, "status_message.snapshot_sequence");
    return {
      kind: "runtime_heartbeat",
      daemon_instance_id: record.daemon_instance_id as string,
      snapshot_sequence: record.snapshot_sequence as number
    };
  }
  if (record.kind !== "runtime_snapshot") {
    return decodeRuntimeState(value);
  }
  expectLiteral(record.topic, "status_message.topic", STATUS_TOPICS);
  expectBoolean(record.full, "status_message.full");
  return {
    kind: "runtime_snapshot",
    topic: record.topic as RuntimeStatusFrame["topic"],
    full: record.full as boolean,
    state: decodeRuntimeState(record.state)
  };
}
