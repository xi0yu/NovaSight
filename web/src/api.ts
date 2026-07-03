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
  deployment: Deployment;
  artifact: ModelArtifact | null;
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
};

export type Statistics = {
  capture_counter: number;
  inference_counter: number;
  dropped_counter: number;
  skipped_counter: number;
  capture_fps: number;
  inference_fps: number;
  e2e_latency: number;
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

export type RuntimeState = {
  running: boolean;
  source: string;
  active_model: ActiveModel | null;
  executor: ExecutorStatus;
  capture: CaptureState;
  statistics?: Statistics;
  inference: Record<string, unknown>;
  config: Record<string, unknown>;
  pipeline: Record<string, unknown>;
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
  type: "string" | "int" | "float" | "select" | "bool";
  options?: string[];
  min?: number;
  max?: number;
  restart_required: boolean;
};

export type ConfigSectionSchema = {
  id: string;
  label: string;
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
};

export type LicenseFeature =
  | "capture"
  | "runtime"
  | "models"
  | "plugins"
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

export type PluginKind = "vision" | "control" | string;

export type PluginInfo = {
  plugin_id: string;
  kind: PluginKind;
  enabled: boolean;
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
  config: "/api/config",
  configSchema: "/api/config/schema",
  captureCapabilities: "/api/capture/capabilities",
  captureSelect: "/api/capture/select",
  captureImage: "/api/capture/image",
  captureStop: "/api/capture/stop",
  captureStream: "/api/capture/stream.mjpg",
  plugins: "/api/plugins",
  modelProjects: "/api/models/projects",
  modelJobs: "/api/models/jobs",
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
    method: "PUT",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
}

export function getCaptureCapabilities(
  device: string
): Promise<CaptureCapabilitiesResponse> {
  return requestJson<CaptureCapabilitiesResponse>(
    `${API_PATHS.captureCapabilities}?device=${encodeURIComponent(device)}`
  );
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

export function getPlugins(): Promise<PluginInfo[]> {
  return requestJson<PluginInfo[]>(API_PATHS.plugins);
}

export function getModelProjects(): Promise<ModelProject[]> {
  return requestJson<ModelProject[]>(API_PATHS.modelProjects);
}

export function getModelVersions(projectId: number): Promise<ModelVersion[]> {
  return requestJson<ModelVersion[]>(`${API_PATHS.modelProjects}/${projectId}/versions`);
}

export function getModelArtifacts(versionId: number): Promise<ModelArtifact[]> {
  return requestJson<ModelArtifact[]>(`/api/models/versions/${versionId}/artifacts`);
}

export function getConversionJobs(versionId?: number): Promise<ConversionJob[]> {
  const suffix =
    typeof versionId === "number" ? `?version_id=${encodeURIComponent(versionId)}` : "";
  return requestJson<ConversionJob[]>(`${API_PATHS.modelJobs}${suffix}`);
}

export function publishModel(projectId: number, artifactId: number): Promise<Deployment> {
  return requestJson<Deployment>(`${API_PATHS.modelProjects}/${projectId}/publish`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ artifact_id: artifactId })
  });
}

export function rollbackModel(projectId: number): Promise<Deployment> {
  return requestJson<Deployment>(`${API_PATHS.modelProjects}/${projectId}/rollback`, {
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
