export type HealthResponse = {
  ok: boolean;
};

export type ExecutorStatus = {
  selected: string;
  executors: Record<string, { available: boolean }>;
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
  report?: ModelSwitchReport;
};

export type DeepStreamPreparePayload = {
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

export type DeepStreamPrepareResponse = {
  status: string;
  reason: string;
  artifact: ModelArtifact;
  manifest_path: string;
  model_fingerprint: string;
  nvinfer_config_owner: "runtime" | (string & {});
};

export type DeepStreamRecommendationResponse = {
  artifact_id: number;
  artifact_path: string;
  recommendation: DeepStreamPreparePayload;
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
    preprocess_ms: number;
    inference_ms: number;
    decode_ms: number;
    nms_ms: number;
    issues: Array<{ code: string; stage: string; message: string }>;
  };
};

export type CaptureState = {
  available: boolean;
  device: string;
  profile: null | {
    pixel_format: string;
    width: number;
    height: number;
    fps: number;
    preference: string;
    selection_reason: string;
  };
  backend: string | null;
  fps_capture: number;
  frame_period_ms: number;
  capture_wait_ms: number;
  frames_dropped: number;
  preview_target_fps: number;
  preview_fps: number;
  preview_frames: number;
  preview_output_frames: number;
  preview_dropped: number;
  recoveries: number;
  last_error: string | null;
  statistics?: Statistics;
  report?: ConfigUpdateResponse;
};

export type Statistics = {
  capture_counter: number;
  inference_counter: number;
  detection_batch_counter?: number;
  detection_batch_consumed_counter?: number;
  dropped_counter: number;
  skipped_counter: number;
  stale_dropped_batches?: number;
  timestamp_rejected_batches?: number;
  non_monotonic_dropped_batches?: number;
  mailbox_overwritten_batches?: number;
  published_frames?: number;
  overwritten_frames?: number;
  acquired_frames?: number;
  capture_fps: number;
  inference_fps: number;
  queue_latency?: number;
  inference_latency?: number;
  stale_drop_count?: number;
  stage_roi_ms?: number;
  stage_engine_ms?: number;
  stage_engine_execute_ms?: number;
  stage_decode_ms?: number;
  stage_handoff_ms?: number;
  stage_postprocess_ms?: number;
  stage_control_ms?: number;
  stage_total_ms?: number;
  e2e_latency: number;
  detection_batch_fps?: number;
  control_observation_counter?: number;
  control_observation_fps?: number;
  timestamp_source?: string;
  last_raw_pts_ns?: number;
  last_capture_ts_ns?: number;
  last_probe_observed_ts_ns?: number;
  last_pts_to_probe_ms?: number;
  last_frame_age_ms?: number;
  latest_frame_age_ms?: number;
  batch_age_ms?: number;
  appsink_caps?: string;
  actual_pipeline_string?: string;
  last_inference_latency_ms?: number;
  last_detection_count?: number;
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
  source?: Record<string, unknown>;
  capture?: Record<string, unknown>;
  roi?: Record<string, unknown>;
  roi_size?: number;
  calibration?: Record<string, unknown>;
  consumers?: Record<string, unknown>;
};

export type RuntimeState = {
  running: boolean;
  source: string;
  active_model: ActiveModel | null;
  executor: ExecutorStatus;
  capture: CaptureState;
  statistics?: Statistics;
  inference: Record<string, unknown>;
  config: RuntimeConfigSummary;
  pipeline: Record<string, unknown>;
  vision?: Record<string, unknown>;
  fatal_error: Record<string, unknown> | null;
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
  schema: ConfigSchemaResponse;
  restart_required: boolean;
  applied?: boolean;
  rolled_back?: boolean;
  message?: string;
  sections?: OperationReportSection[];
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
  fingerprint: string;
  tier: string;
  features: LicenseFeature[];
  license_id: string;
  created_at: number | null;
  activated_at: number | null;
  expires_at: number | null;
  duration_value: number | null;
  duration_unit: string;
  updated_at: number | null;
  message: string;
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

export const API_PATHS = {
  health: "/healthz",
  runtimeState: "/api/runtime/state",
  runtimeStart: "/api/runtime/start",
  runtimeStop: "/api/runtime/stop",
  config: "/api/config",
  configSchema: "/api/config/schema",
  captureCapabilities: "/api/capture/capabilities",
  captureSelect: "/api/capture/select",
  captureImage: "/api/capture/image",
  captureStop: "/api/capture/stop",
  captureStream: "/api/capture/stream.mjpg",
  executors: "/api/executors",
  kmnetConnect: "/api/executors/kmnet/connect",
  kmnetDisconnect: "/api/executors/kmnet/disconnect",
  kmnetDiagnosticMove: "/api/executors/kmnet/diagnostic-move",
  kmnetDiagnosticCircle: "/api/executors/kmnet/diagnostic-circle",
  modelProjects: "/api/models/projects",
  modelCatalog: "/api/models/catalog",
  modelJobs: "/api/models/jobs",
  modelJobsList: "/api/models/jobs/list",
  license: "/api/license",
  licenseActivate: "/api/license/activate",
  statusWs: "/ws/status"
} as const;

const apiBase = (import.meta.env.VITE_NOVASIGHT_API_BASE ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${apiBase}${path}`;
}

export function streamUrl(cacheKey: number, configVersion = 0): string {
  return apiUrl(`${API_PATHS.captureStream}?ts=${cacheKey}&config=${configVersion}`);
}

export function statusWebSocketUrl(): string {
  const explicit = import.meta.env.VITE_NOVASIGHT_WS_BASE;
  if (explicit) {
    return `${explicit.replace(/\/$/, "")}${API_PATHS.statusWs}`;
  }
  if (apiBase.startsWith("http")) {
    return `${apiBase.replace(/^http/, "ws")}${API_PATHS.statusWs}`;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${API_PATHS.statusWs}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");

  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      ...Object.fromEntries(headers.entries())
    }
  });

  const contentType = response.headers.get("content-type") ?? "";
  let body: unknown = null;
  try {
    body = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
  } catch {
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
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>(API_PATHS.health);
}

export function getRuntimeState(): Promise<RuntimeState> {
  return requestJson<RuntimeState>(API_PATHS.runtimeState);
}

export function startRuntimePipeline(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.runtimeStart, {
    method: "POST"
  });
}

export function stopRuntimePipeline(): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.runtimeStop, {
    method: "POST"
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

export function diagnosticMoveKmNet(
  dx = 1,
  dy = 0,
  repeat = 1,
  intervalMs = 0,
  moveKind?: string,
  moveMs = 12,
  bezierCtrl?: { x1: number; y1: number; x2: number; y2: number }
): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.kmnetDiagnosticMove, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      dx,
      dy,
      repeat,
      interval_ms: intervalMs,
      move_kind: moveKind,
      move_ms: moveMs,
      ctrl_x1: bezierCtrl?.x1,
      ctrl_y1: bezierCtrl?.y1,
      ctrl_x2: bezierCtrl?.x2,
      ctrl_y2: bezierCtrl?.y2
    })
  });
}

export function diagnosticCircleKmNet(
  radius = 8,
  steps = 32,
  intervalMs = 8
): Promise<Record<string, RuntimeConfigValue>> {
  return requestJson<Record<string, RuntimeConfigValue>>(API_PATHS.kmnetDiagnosticCircle, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ radius, steps, interval_ms: intervalMs })
  });
}

export function getRuntimeConfig(): Promise<RuntimeConfig> {
  return requestJson<RuntimeConfig>(API_PATHS.config);
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

export function getModelProjects(): Promise<ModelProject[]> {
  return requestJson<ModelProject[]>(API_PATHS.modelProjects);
}

export function getModelCatalog(force = false): Promise<ModelCatalogResponse> {
  return requestJson<ModelCatalogResponse>(
    `${API_PATHS.modelCatalog}?force=${force ? "true" : "false"}`
  );
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

export function publishModel(projectId: number, artifactId: number): Promise<ModelPublishResponse> {
  return requestJson<ModelPublishResponse>(`${API_PATHS.modelProjects}/${projectId}/publish`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ artifact_id: artifactId })
  });
}

export function prepareDeepStreamArtifact(
  artifactId: number,
  payload: DeepStreamPreparePayload
): Promise<DeepStreamPrepareResponse> {
  return requestJson<DeepStreamPrepareResponse>(`/api/models/artifacts/${artifactId}/deepstream/prepare`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
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

export type ModelScanResponse = {
  project_count: number;
  previous_project_count: number;
  discovered_files: number;
  updated_files: number;
  cache_hits: number;
  force: boolean;
};

export function scanModelDirectory(force = false): Promise<ModelScanResponse> {
  return requestJson<ModelScanResponse>(`/api/models/scan?force=${force ? "true" : "false"}`, {
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
  return requestJson<LicenseStatus>(API_PATHS.license);
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
