export type HealthResponse = {
  ok: boolean;
};

export type ExecutorStatus = {
  selected: string;
  executors: Record<string, ExecutorAvailability>;
  state: RuntimeSubsystemState;
  last_error: RuntimeErrorSummary | null;
};

export type ExecutorAvailability = {
  available: boolean;
  configuration_state: "not_applicable" | "uncommissioned" | "restart_required" | "ready" | (string & {});
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
  connection_state: string;
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
};

export type RuntimeSubsystemState =
  | "stopped"
  | "starting"
  | "ready"
  | "running"
  | "degraded"
  | "stopping"
  | "failed"
  | "unavailable";

export type RuntimeErrorSummary = {
  code: string;
  message: string;
  subsystem: string | null;
};

export type ModelProject = {
  id: number;
  name: string;
  description: string;
};

export type ModelVersion = {
  id: number;
  project_id: number;
  version: string;
  source_kind: "pt" | "onnx" | (string & {});
  source_path: string;
  classes: string[];
  input_shape: string;
};

export type ModelArtifact = {
  id: number;
  version_id: number;
  kind: "pt" | "onnx" | "engine" | (string & {});
  path: string;
  checksum: string;
  status: "pending" | "running" | "ready" | "failed" | (string & {});
  size_bytes?: number | null;
};

export type ModelRecommendation = "recommended" | "not_recommended" | "unrated";

export type ModelArtifactMetadata = {
  artifact_id: number;
  recommendation: ModelRecommendation;
  tags: string[];
};

export type ModelCatalogModel = {
  type: "model";
  name: string;
  relative_path: string;
  kind: "onnx" | "engine" | (string & {});
  size_bytes: number;
  scan_status: "ready" | "need_confirm" | "invalid" | "unsupported" | (string & {});
  scan_reason: string;
  project_id?: number;
  project_name?: string;
  version_id?: number;
  version_name?: string;
  artifact_id?: number;
  artifact_status?: ModelArtifact["status"];
  recommendation: ModelRecommendation;
  tags: string[];
};

export type ModelCatalogDirectory = {
  type: "directory";
  name: string;
  relative_path: string;
  children: Array<ModelCatalogDirectory | ModelCatalogModel>;
};

export type ModelCatalogResponse = {
  root: ModelCatalogDirectory;
  directory_count: number;
  model_count: number;
  discovered_files: number;
  updated_files: number;
  cache_hits: number;
  force: boolean;
};

export type ModelCatalogRegisterResponse = {
  project: ModelProject;
  version: ModelVersion;
  artifact: ModelArtifact;
  engine_path: string;
  created: boolean;
};

export type Deployment = {
  id: number;
  project_id: number;
  artifact_id: number;
  previous_artifact_id: number | null;
};

export type ModelPrepareResponse = {
  project: ModelProject;
  version: ModelVersion;
  artifact: ModelArtifact;
  downloaded: boolean;
  url: string;
};

export type ModelUploadResponse = {
  project: ModelProject;
  version: ModelVersion;
  artifact: ModelArtifact;
};

export type ConversionJob = {
  id: number;
  version_id: number;
  target_kind: "onnx" | "engine" | (string & {});
  command: string[];
  status: "pending" | "running" | "failed" | "succeeded" | (string & {});
  log: string;
};

export type ActiveModel = {
  project: ModelProject | null;
  version?: ModelVersion | null;
  deployment: Deployment;
  artifact: ModelArtifact | null;
};

export type OperationReportSection = {
  section: string;
  impact: string;
  status: string;
  message: string;
};

export type ModelSwitchReport = {
  action: string;
  applied: boolean;
  rolled_back: boolean;
  message: string;
  runtime_error?: string;
  artifact_id: number;
  previous_artifact_id: number | null;
  artifact_path: string;
  backend: string;
  input_shape: string;
  classes: number;
  sections: OperationReportSection[];
};

export type ModelPublishResponse = {
  deployment: Deployment;
  inference: Record<string, unknown>;
  parser_contract?: ParserContract;
  preparation?: {
    manifest_action: "generated" | "reused" | (string & {});
    reason: string;
    input_shape: string;
    classes: string[];
  };
  report?: ModelSwitchReport;
};

export type ParserPresetId =
  | "auto"
  | "yolov5"
  | "yolov8"
  | "yolo11"
  | "novasight_generic";

export type ParserContract = {
  requested_preset: ParserPresetId;
  compatibility: "yolov5" | "yolov8_yolo11";
  has_objectness: boolean;
  parser_library: "novasight_builtin";
  parser_function: "NvDsInferParseNovaSight";
  nms_owner: "deepstream";
};

export type DeepStreamManifestRecommendation = {
  model_id: string;
  display_name: string;
  runtime_precision?: string;
  input_name: string;
  input_shape: number[];
  input_dtype?: string;
  input_color_format?: string;
  input_scale_factor?: number;
  maintain_aspect_ratio?: boolean;
  symmetric_padding?: boolean;
  output_name: string;
  output_shape: number[];
  output_dtype?: string;
  class_count: number;
  confidence_threshold?: number;
  nms_iou_threshold?: number;
};

export type DeepStreamRecommendationResponse = {
  artifact_id: number;
  artifact_path: string;
  recommendation: DeepStreamManifestRecommendation;
  io_tensors: Array<{
    name: string;
    shape: number[];
    dtype: string;
    mode: "input" | "output";
  }>;
  class_names: string[];
  output_has_objectness: boolean;
  sources: Record<string, string>;
  warnings: string[];
};

export type ModelProfile = {
  schema_version: number;
  model_id: string;
  display_name: string;
  status:
    | "UNINSPECTED"
    | "INSPECTING"
    | "NEEDS_CONFIGURATION"
    | "READY_FOR_PROBE"
    | "PROBING"
    | "VALIDATED"
    | "INVALID"
    | "INCOMPATIBLE"
    | "ACTIVE";
  input: {
    name: string;
    runtime_shape: number[];
    engine_shape: number[];
    dtype: string;
    layout: string;
  };
  outputs: Array<{
    name: string;
    shape: number[];
    engine_shape: number[];
    dtype: string;
  }>;
  preprocess: {
    color_format: string;
    scale: number | null;
    resize_mode: string;
  };
  decoder: {
    parser_type: string;
    class_count: number;
    bbox_format: string;
    has_objectness: boolean | null;
  };
  labels: string[];
  parser_candidates: Array<{
    parser_type: string;
    confidence: string;
    reason: string;
    requires_confirmation: boolean;
  }>;
  validation: {
    status: string;
    engine_execution_ok: boolean;
    decoder_ok: boolean;
    nms_ok: boolean;
    detection_batch_ok: boolean;
    issues: string[];
  };
};

export type ModelProfileResponse = {
  artifact_id: number;
  profile_path: string;
  profile: ModelProfile;
};

export type ModelProfileConfigurePayload = {
  color_format: "RGB" | "BGR";
  scale: number;
  offsets?: number[];
  mean?: number[];
  std?: number[];
  resize_mode: "direct" | "letterbox";
  symmetric_padding?: boolean;
  padding_value?: number;
  parser_type: string;
  class_count: number;
  labels: string[];
  bbox_format: "xywh" | "xyxy";
  has_objectness: boolean;
  confidence_threshold?: number;
  nms_threshold?: number;
  max_detections?: number;
};

export type ModelProbeResponse = ModelProfileResponse & {
  report: {
    status: string;
    engine_execution_ok: boolean;
    output_tensor_ok: boolean;
    decoder_ok: boolean;
    nms_ok: boolean;
    detection_batch_ok: boolean;
    // Null means the probe injected a model-shaped tensor and did not run the
    // image preprocessing stage.
    preprocess_ms: number | null;
    inference_ms: number;
    decode_ms: number;
    // Native probing currently measures decode + NMS as one operation. Null is
    // deliberate: reporting 0 ms would be a fabricated measurement.
    nms_ms: number | null;
    issues: Array<{ code: string; stage: string; message: string }>;
  };
};

export type CaptureState = {
  available: boolean;
  running: boolean;
  state: RuntimeSubsystemState;
  device: string;
  profile: null | {
    pixel_format: string;
    width: number;
    height: number;
    fps: number;
    preference: string;
    source: "configured" | string;
  };
  backend: string | null;
  last_error: string | null;
};

export type Statistics = {
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
};

export type RuntimeInferenceState = {
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
  postprocess: {
    confidence_threshold: number;
    nms_threshold: number;
  } | null;
};

export type PreviewSnapshotState = {
  preview_enabled: boolean;
  preview_running: boolean;
  preview_active: boolean;
  preview_encoder_active: boolean;
  preview_consumers: number;
  preview_available: boolean;
  preview_sequence: number;
  preview_reason: string;
  preview_transport: string;
};

export type RuntimeDeepStreamState = {
  running: boolean;
  terminal_error: boolean;
  last_error: string | null;
  input_frames: number;
  metadata_extractions: number;
  published_batches: number;
  crosshair_active: boolean;
  crosshair_reason: string;
};

export type RuntimePipelineSummary = {
  running: boolean;
  state: "stopped" | "starting" | "running" | "standby" | "stopping" | "faulted";
  epoch: number | null;
  started_at_ms: number | null;
  mode: string;
  last_error: RuntimeErrorSummary | null;
  deepstream: RuntimeDeepStreamState;
};

export type CaptureCapability = {
  pixel_format: string;
  width: number;
  height: number;
  fps_list: number[];
};

export type CaptureCapabilitiesResponse = {
  available: boolean;
  device: string;
  capabilities: CaptureCapability[];
  reason: string;
};

export type CaptureSelectPayload = {
  device: string;
  preference?: "auto_high_fps" | "auto_low_latency" | "auto_balanced" | "manual";
  pixel_format?: string;
  width?: number;
  height?: number;
  fps?: number;
};

export type RuntimeConfigSummary = {
  version: number;
  schema_version: number;
  effective_version: number;
  restart_required: boolean;
};

export type RuntimeState = {
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
  vision: Record<string, unknown>;
  fatal_error: RuntimeErrorSummary | null;
};

export type RuntimeStatusTopic =
  | "summary"
  | "capture"
  | "infer"
  | "control"
  | "latency";

export type RuntimeStatusFrame = {
  kind: "runtime_snapshot";
  topic: RuntimeStatusTopic | "full";
  full: boolean;
  state: Partial<RuntimeState>;
};

export type RuntimeConfigValue =
  | string
  | number
  | boolean
  | null
  | RuntimeConfigValue[]
  | { [key: string]: RuntimeConfigValue };

export type RuntimeConfig = Record<string, RuntimeConfigValue>;

export type ConfigFieldSchema = {
  path: string;
  label: string;
  type: "string" | "string_list" | "int" | "float" | "select" | "bool";
  options?: string[];
  option_labels?: Record<string, string>;
  min?: number;
  max?: number;
  default?: number;
  recommended_min?: number;
  recommended_max?: number;
  step?: number;
  precision?: number;
  unit?: string;
  description?: string;
  restart_required: boolean;
};

export type ConfigSectionSchema = {
  id: string;
  label: string;
  algorithm_scope?: string[];
  fields: ConfigFieldSchema[];
};

export type ConfigSchemaResponse = {
  version: number;
  values: RuntimeConfig;
  sections: ConfigSectionSchema[];
};

export type ConfigUpdateResponse = {
  config: RuntimeConfig;
  schema?: ConfigSchemaResponse;
  restart_required: boolean;
  applied?: boolean;
  rolled_back?: boolean;
  message?: string;
  sections?: OperationReportSection[];
};

export type ConfigCommandPayload =
  | {
      command: "set_output_gate";
      enabled: boolean;
      expected_revision?: number;
    }
  | {
      command: "set_trigger_mode";
      mode: string;
      expected_revision?: number;
    };

export type LicenseFeature =
  | "capture"
  | "runtime"
  | "models"
  | "tensorrt"
  | "hardware_control"
  | "config_read"
  | "config_write";

export type LicenseStatus = {
  configured: boolean;
  valid: boolean;
  temporary_access_supported: boolean;
  fingerprint: string;
  tier: string;
  features: LicenseFeature[];
  license_id: string;
  credential_format: string;
  token_id: string;
  key_id: string;
  created_at: number | null;
  not_before: number | null;
  activated_at: number | null;
  expires_at: number | null;
  duration_value: number | null;
  duration_unit: string;
  updated_at: number | null;
  message: string;
};

export type TemporaryLicenseResponse = {
  supported: boolean;
  granted: boolean;
  status: LicenseStatus;
};

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function getApiErrorCode(error: unknown): string {
  if (!(error instanceof ApiError)) {
    return "";
  }
  const detail = typeof error.detail === "object" && error.detail !== null
    ? error.detail as Record<string, unknown>
    : {};
  return typeof detail.code === "string" ? detail.code : "";
}

export const API_PATHS = {
  health: "/healthz",
  runtimeState: "/api/runtime/state",
  runtimeStart: "/api/runtime/start",
  runtimeStop: "/api/runtime/stop",
  runtimeEmergencyStop: "/api/v1/runtime/emergency-stop",
  config: "/api/config",
  configCommands: "/api/v1/config/commands",
  configSchema: "/api/config/schema",
  captureCapabilities: "/api/capture/capabilities",
  captureSelect: "/api/capture/select",
  captureImage: "/api/capture/image",
  captureStop: "/api/capture/stop",
  capturePreview: "/api/capture/preview",
  captureStream: "/api/capture/stream.mjpg",
  crosshair: "/api/crosshair",
  crosshairLearn: "/api/crosshair/learn",
  crosshairTemplate: "/api/crosshair/template",
  crosshairTemplatePreview: "/api/crosshair/template.png",
  executors: "/api/executors",
  kmnetConnect: "/api/executors/kmnet/connect",
  kmnetDisconnect: "/api/executors/kmnet/disconnect",
  kmnetDiagnosticMove: "/api/executors/kmnet/diagnostic-move",
  modelProjects: "/api/models/projects",
  modelCatalog: "/api/models/catalog",
  modelJobs: "/api/models/jobs",
  modelJobsList: "/api/models/jobs/list",
  license: "/api/license",
  licenseActivate: "/api/license/activate",
  licenseTemporary: "/api/license/temporary",
  statusWs: "/ws/status"
} as const;

const apiBase = (import.meta.env.VITE_NOVASIGHT_API_BASE ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${apiBase}${path}`;
}

export function streamUrl(cacheKey: number, configVersion = 0, previewFps = 30): string {
  return apiUrl(
    `${API_PATHS.captureStream}?ts=${cacheKey}&config=${configVersion}&fps=${previewFps}`
  );
}

export function statusWebSocketUrl(topic?: RuntimeStatusTopic): string {
  const explicit = import.meta.env.VITE_NOVASIGHT_WS_BASE;
  if (explicit) {
    const base = `${explicit.replace(/\/$/, "")}${API_PATHS.statusWs}`;
    return topic ? `${base}?topic=${encodeURIComponent(topic)}` : base;
  }
  if (apiBase.startsWith("http")) {
    const base = `${apiBase.replace(/^http/, "ws")}${API_PATHS.statusWs}`;
    return topic ? `${base}?topic=${encodeURIComponent(topic)}` : base;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const base = `${protocol}//${window.location.host}${API_PATHS.statusWs}`;
  return topic ? `${base}?topic=${encodeURIComponent(topic)}` : base;
}

type RequestOptions = {
  timeoutMs?: number;
};

const STATUS_REQUEST_TIMEOUT_MS = 5000;
const STANDARD_READ_TIMEOUT_MS = 10000;

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  options?: RequestOptions
): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");

  const timeoutController = options?.timeoutMs ? new AbortController() : null;
  const upstreamSignal = init?.signal;
  let timedOut = false;
  let timeoutId: number | null = null;
  const handleUpstreamAbort = () => timeoutController?.abort(upstreamSignal?.reason);

  if (timeoutController && upstreamSignal) {
    if (upstreamSignal.aborted) {
      handleUpstreamAbort();
    } else {
      upstreamSignal.addEventListener("abort", handleUpstreamAbort, { once: true });
    }
  }
  if (timeoutController && options?.timeoutMs) {
    timeoutId = window.setTimeout(() => {
      timedOut = true;
      timeoutController.abort();
    }, options.timeoutMs);
  }

  try {
    let response: Response;
    try {
      response = await fetch(apiUrl(path), {
        ...init,
        credentials: init?.credentials ?? "include",
        signal: timeoutController?.signal ?? upstreamSignal,
        headers: {
          ...Object.fromEntries(headers.entries())
        }
      });
    } catch (error) {
      if (timedOut) {
        throw new ApiError(
          `请求超时：${path} 在 ${Math.round((options?.timeoutMs ?? 0) / 1000)} 秒内未响应`,
          408,
          { path, timeout_ms: options?.timeoutMs ?? 0 }
        );
      }
      throw error;
    }

    const contentType = response.headers.get("content-type") ?? "";
    let body: unknown = null;
    try {
      body = contentType.includes("application/json")
        ? await response.json()
        : await response.text();
    } catch (error) {
      if (timedOut) {
        throw new ApiError(
          `请求超时：${path} 在 ${Math.round((options?.timeoutMs ?? 0) / 1000)} 秒内未完成响应`,
          408,
          { path, timeout_ms: options?.timeoutMs ?? 0 }
        );
      }
      if (typeof error === "object" && error !== null && "name" in error && error.name === "AbortError") {
        throw error;
      }
      body = "";
    }

    if (!response.ok) {
      const objectBody =
        typeof body === "object" && body !== null ? (body as Record<string, unknown>) : null;
      const detail =
        objectBody && typeof objectBody.last_error === "string"
          ? objectBody.last_error
          : objectBody && typeof objectBody.reason === "string"
            ? objectBody.reason
            : objectBody && typeof objectBody.detail === "string"
              ? objectBody.detail
              : objectBody &&
                  typeof objectBody.detail === "object" &&
                  objectBody.detail !== null &&
                  typeof (objectBody.detail as Record<string, unknown>).message === "string"
                ? String((objectBody.detail as Record<string, unknown>).message)
              : response.statusText;
      throw new ApiError(detail || "请求失败", response.status, body);
    }

    return body as T;
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    upstreamSignal?.removeEventListener("abort", handleUpstreamAbort);
  }
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>(
    API_PATHS.health,
    { signal },
    { timeoutMs: STATUS_REQUEST_TIMEOUT_MS }
  );
}

export function getRuntimeState(
  signal?: AbortSignal,
  timeoutMs = STATUS_REQUEST_TIMEOUT_MS
): Promise<RuntimeState> {
  return requestJson<RuntimeState>(
    API_PATHS.runtimeState,
    { signal },
    { timeoutMs }
  );
}

export function startRuntimePipeline(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.runtimeStart, {
    method: "POST"
  });
}

export function stopRuntimePipeline(): Promise<RuntimeState> {
  return requestJson<RuntimeState>(API_PATHS.runtimeStop, {
    method: "POST"
  });
}

export function emergencyStopRuntimePipeline(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.runtimeEmergencyStop, {
    method: "POST"
  });
}

export function getCrosshairStatus(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.crosshair);
}

export function learnCrosshair(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.crosshairLearn, {
    method: "POST"
  });
}

export function clearCrosshairTemplate(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.crosshairTemplate, {
    method: "DELETE"
  });
}

export function crosshairTemplatePreviewUrl(cacheKey: number): string {
  return apiUrl(`${API_PATHS.crosshairTemplatePreview}?ts=${cacheKey}`);
}

export function diagnosticMoveKmNet(
  dx = 1,
  dy = 0
): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.kmnetDiagnosticMove, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      dx,
      dy,
      repeat: 1,
      interval_ms: 0,
      move_kind: "raw"
    })
  });
}

export function connectKmNet(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.kmnetConnect, {
    method: "POST"
  });
}

export function disconnectKmNet(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.kmnetDisconnect, {
    method: "POST"
  });
}

export function getRuntimeConfig(): Promise<RuntimeConfig> {
  return requestJson<RuntimeConfig>(
    API_PATHS.config,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS }
  );
}

export function getConfigSchema(): Promise<ConfigSchemaResponse> {
  return requestJson<ConfigSchemaResponse>(API_PATHS.configSchema);
}

export function sanitizeRuntimeConfigForUpdate(config: RuntimeConfig): RuntimeConfig {
  const next = structuredClone(config) as RuntimeConfig;
  delete next.version;
  delete next.roi_size;
  return next;
}

export function updateRuntimeConfig(config: RuntimeConfig): Promise<ConfigUpdateResponse> {
  const payload = sanitizeRuntimeConfigForUpdate(config);
  return requestJson<ConfigUpdateResponse>(API_PATHS.config, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
}

export function updateRuntimeConfigField(
  section: string,
  key: string,
  value: RuntimeConfigValue
): Promise<ConfigUpdateResponse> {
  return requestJson<ConfigUpdateResponse>(API_PATHS.config, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ section, key, value })
  });
}

export function updateRuntimeConfigCommand(
  command: ConfigCommandPayload
): Promise<ConfigUpdateResponse> {
  return requestJson<ConfigUpdateResponse>(API_PATHS.configCommands, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(command)
  });
}

export function setRuntimeOutputGate(enabled: boolean): Promise<ConfigUpdateResponse> {
  return updateRuntimeConfigCommand({
    command: "set_output_gate",
    enabled
  });
}

export function setRuntimeTriggerMode(mode: string): Promise<ConfigUpdateResponse> {
  return updateRuntimeConfigCommand({
    command: "set_trigger_mode",
    mode
  });
}

export function getCaptureCapabilities(
  device: string
): Promise<CaptureCapabilitiesResponse> {
  return requestJson<CaptureCapabilitiesResponse>(API_PATHS.captureCapabilities, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ device })
  });
}

export function selectCaptureProfile(payload: CaptureSelectPayload): Promise<CaptureState> {
  return requestJson<CaptureState>(API_PATHS.captureSelect, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
}

export function selectImageSource(path: string, fps: number): Promise<CaptureState> {
  return requestJson<CaptureState>(API_PATHS.captureImage, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ path, fps })
  });
}

export function stopCapture(reason = "用户停止采集"): Promise<CaptureState> {
  return requestJson<CaptureState>(API_PATHS.captureStop, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ reason })
  });
}

export function setCapturePreviewEnabled(
  enabled: boolean,
  options?: { keepalive?: boolean }
): Promise<PreviewSnapshotState> {
  return requestJson<PreviewSnapshotState>(API_PATHS.capturePreview, {
    method: "POST",
    keepalive: options?.keepalive,
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ enabled })
  });
}

export function getModelProjects(): Promise<ModelProject[]> {
  return requestJson<ModelProject[]>(
    API_PATHS.modelProjects,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS }
  );
}

export function getModelCatalog(force = false): Promise<ModelCatalogResponse> {
  return requestJson<ModelCatalogResponse>(
    `${API_PATHS.modelCatalog}?force=${force ? "true" : "false"}`
  );
}

export function registerCatalogModel(
  relativePath: string
): Promise<ModelCatalogRegisterResponse> {
  return requestJson<ModelCatalogRegisterResponse>("/api/models/catalog/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ relative_path: relativePath })
  });
}

export function updateModelArtifactMetadata(
  artifactId: number,
  recommendation: ModelRecommendation,
  tags: string[]
): Promise<ModelArtifactMetadata> {
  return requestJson<ModelArtifactMetadata>(`/api/models/artifacts/${artifactId}/metadata`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ recommendation, tags })
  });
}

export function getModelVersions(projectId: number): Promise<ModelVersion[]> {
  return requestJson<ModelVersion[]>(`${API_PATHS.modelProjects}/${projectId}/versions`);
}

export function getModelArtifacts(versionId: number): Promise<ModelArtifact[]> {
  return requestJson<ModelArtifact[]>(`/api/models/versions/${versionId}/artifacts`);
}

export function getConversionJobs(versionId?: number): Promise<ConversionJob[]> {
  if (typeof versionId !== "number") {
    return requestJson<ConversionJob[]>(API_PATHS.modelJobs);
  }
  return requestJson<ConversionJob[]>(API_PATHS.modelJobsList, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ version_id: versionId })
  });
}

export function publishModel(
  projectId: number,
  artifactId: number,
  parserPreset: ParserPresetId = "auto"
): Promise<ModelPublishResponse> {
  return requestJson<ModelPublishResponse>(`${API_PATHS.modelProjects}/${projectId}/publish`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ artifact_id: artifactId, parser_preset: parserPreset })
  });
}

export function getDeepStreamRecommendation(
  artifactId: number
): Promise<DeepStreamRecommendationResponse> {
  return requestJson<DeepStreamRecommendationResponse>(
    `/api/models/artifacts/${artifactId}/deepstream/recommendation`
  );
}

export function inspectModelArtifact(artifactId: number): Promise<ModelProfileResponse> {
  return requestJson<ModelProfileResponse>(`/api/models/artifacts/${artifactId}/inspect`, {
    method: "POST"
  });
}

export function getModelProfile(artifactId: number): Promise<ModelProfileResponse> {
  return requestJson<ModelProfileResponse>(`/api/models/artifacts/${artifactId}/profile`);
}

export function configureModelProfile(
  artifactId: number,
  payload: ModelProfileConfigurePayload
): Promise<ModelProfileResponse> {
  return requestJson<ModelProfileResponse>(`/api/models/artifacts/${artifactId}/profile`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
}

export function probeModelArtifact(
  artifactId: number,
  inputMode: "fixed" | "latest" = "fixed"
): Promise<ModelProbeResponse> {
  return requestJson<ModelProbeResponse>(`/api/models/artifacts/${artifactId}/probe`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ input_mode: inputMode })
  });
}

export function rollbackModel(projectId: number): Promise<ModelPublishResponse> {
  return requestJson<ModelPublishResponse>(`${API_PATHS.modelProjects}/${projectId}/rollback`, {
    method: "POST"
  });
}

export function prepareYolov8nExample(): Promise<ModelPrepareResponse> {
  return requestJson<ModelPrepareResponse>("/api/models/examples/yolov8n/prepare", {
    method: "POST"
  });
}

export function uploadModelFile(payload: {
  projectName: string;
  version: string;
  description: string;
  classes: string;
  inputShape: string;
  file: File;
}): Promise<ModelUploadResponse> {
  const form = new FormData();
  form.set("project_name", payload.projectName);
  form.set("version", payload.version);
  form.set("description", payload.description);
  form.set("classes", payload.classes);
  form.set("input_shape", payload.inputShape);
  form.set("file", payload.file);
  return requestJson<ModelUploadResponse>("/api/models/upload", {
    method: "POST",
    body: form
  });
}

export function getLicenseStatus(): Promise<LicenseStatus> {
  return requestJson<LicenseStatus>(
    API_PATHS.license,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS }
  );
}

export function requestTemporaryLicense(): Promise<TemporaryLicenseResponse> {
  return requestJson<TemporaryLicenseResponse>(API_PATHS.licenseTemporary, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({})
  });
}

export function saveLicenseKey(key: string): Promise<LicenseStatus> {
  return requestJson<LicenseStatus>(API_PATHS.licenseActivate, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ key })
  });
}

export function clearLicenseKey(): Promise<LicenseStatus> {
  return requestJson<LicenseStatus>(API_PATHS.license, {
    method: "DELETE"
  });
}
