import {
  decodeCaptureState,
  decodeCrosshairLearnResponse,
  decodeCrosshairSnapshot,
  decodePreviewSnapshot,
  decodeRuntimeState
} from "./contracts/runtime";
import {
  decodeConversionJobs,
  decodeModelArtifactMetadata,
  decodeModelArtifacts,
  decodeModelCatalog,
  decodeModelCatalogRegisterResponse,
  decodeModelPublishResponse,
  decodeModelProjects,
  decodeModelVersions
} from "./contracts/model";
import {
  decodeConfigSchema,
  decodeConfigUpdate,
  decodeRuntimeConfig
} from "./contracts/config";
import { decodeLicenseStatus } from "./contracts/license";
import { decodeCaptureCapabilities } from "./contracts/capture";
import { decodeDiagnosticMoveResponse } from "./contracts/hardware";
import {
  decodeDeepStreamRecommendation,
  decodeModelProbeResponse,
  decodeModelProfileResponse
} from "./contracts/modelIngress";
import { decodeAuthSession } from "./contracts/auth";
import type {
  ConfigCommandPayload,
  ConfigSchemaResponse,
  ConfigUpdateResponse,
  RuntimeConfig,
  RuntimeConfigValue
} from "./contracts/config";
import type {
  CaptureState,
  CrosshairLearnResponse,
  CrosshairSnapshot,
  PreviewSnapshotState,
  RuntimeState,
  RuntimeStatusTopic
} from "./contracts/runtime";
import type {
  ConversionJob,
  ModelArtifact,
  ModelArtifactMetadata,
  ModelCatalogRegisterResponse,
  ModelCatalogResponse,
  ModelPublishResponse,
  ModelProject,
  ModelRecommendation,
  ModelVersion,
  ParserPresetId
} from "./contracts/model";
import type { LicenseStatus } from "./contracts/license";
import type {
  CaptureCapabilitiesResponse,
  CaptureSelectPayload
} from "./contracts/capture";
import type { DiagnosticMoveResponse } from "./contracts/hardware";
import type {
  DeepStreamRecommendationResponse,
  ModelProbeResponse,
  ModelProfileConfigurePayload,
  ModelProfileResponse
} from "./contracts/modelIngress";
import type { AuthSession } from "./contracts/auth";

export {
  decodeRuntimeStatusMessage,
  RuntimeContractError
} from "./contracts/runtime";
export type * from "./contracts/runtime";
export type * from "./contracts/model";
export { ConfigContractError } from "./contracts/config";
export type * from "./contracts/config";
export { LicenseContractError } from "./contracts/license";
export type * from "./contracts/license";
export { CaptureContractError } from "./contracts/capture";
export type * from "./contracts/capture";
export { HardwareContractError } from "./contracts/hardware";
export type * from "./contracts/hardware";
export { ModelIngressContractError } from "./contracts/modelIngress";
export type * from "./contracts/modelIngress";
export { AuthContractError } from "./contracts/auth";
export type * from "./contracts/auth";

export type HealthResponse = {
  ok: boolean;
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

function invalidRequest(field: string, expected: string, value: unknown): never {
  throw new ApiError(`请求参数无效：${field} 应为 ${expected}`, 400, {
    code: "frontend_contract_invalid",
    field,
    expected,
    value
  });
}

function requireNonBlank(value: string, field: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    return invalidRequest(field, "非空字符串", value);
  }
  return value;
}

function requirePositiveSafeInteger(value: number, field: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    return invalidRequest(field, "正安全整数", value);
  }
  return value;
}

function requireUnsignedSafeInteger(value: number, field: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    return invalidRequest(field, "无符号安全整数", value);
  }
  return value;
}

function requirePositiveU32(value: number, field: string): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffff_ffff) {
    return invalidRequest(field, "正 u32", value);
  }
  return value;
}

function requireI16(value: number, field: string): number {
  if (!Number.isSafeInteger(value) || value < -0x8000 || value > 0x7fff) {
    return invalidRequest(field, "i16", value);
  }
  return value;
}

function assertJsonValue(
  value: unknown,
  path: string,
  ancestors: Set<object>,
  allowUndefined: boolean
): void {
  if (value === undefined) {
    if (allowUndefined) return;
    invalidRequest(path, "可序列化 JSON 值", value);
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) invalidRequest(path, "有限数字", value);
    return;
  }
  if (typeof value !== "object") {
    invalidRequest(path, "可序列化 JSON 值", value);
  }
  if (ancestors.has(value)) invalidRequest(path, "无循环引用的 JSON 值", value);
  ancestors.add(value);
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertJsonValue(item, `${path}[${index}]`, ancestors, false));
  } else {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      invalidRequest(path, "普通 JSON 对象", value);
    }
    for (const [key, item] of Object.entries(value)) {
      assertJsonValue(item, `${path}.${key}`, ancestors, true);
    }
  }
  ancestors.delete(value);
}

function encodeJsonBody(value: unknown): string {
  assertJsonValue(value, "body", new Set<object>(), false);
  const encoded = JSON.stringify(value);
  if (encoded === undefined) return invalidRequest("body", "JSON 文档", value);
  return encoded;
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
  authSession: "/api/auth/session",
  runtimeState: "/api/runtime/state",
  runtimeStart: "/api/runtime/start",
  runtimeStop: "/api/runtime/stop",
  runtimeEmergencyStop: "/api/runtime/emergency-stop",
  config: "/api/config",
  configCommands: "/api/v1/config/commands",
  configSchema: "/api/config/schema",
  captureCapabilities: "/api/capture/capabilities",
  captureSelect: "/api/capture/select",
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
  statusWs: "/ws/status"
} as const;

const apiBase = (import.meta.env.VITE_NOVASIGHT_API_BASE ?? "").replace(/\/$/, "");
let csrfToken = "";

function synchronizeAuthSession(session: AuthSession): AuthSession {
  csrfToken = session.authenticated ? session.csrf_token ?? "" : "";
  return session;
}

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
const DEFAULT_REQUEST_TIMEOUT_MS = 30000;
const MODEL_OPERATION_TIMEOUT_MS = 180000;

type JsonDecoder<T> = (value: unknown) => T;

async function requestJson<T>(
  path: string,
  init: RequestInit | undefined,
  options: RequestOptions | undefined,
  decode: JsonDecoder<T>
): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  const method = (init?.method ?? "GET").toUpperCase();
  if (!(["GET", "HEAD", "OPTIONS"].includes(method)) && csrfToken) {
    headers.set("X-NovaSight-CSRF", csrfToken);
  }

  const timeoutMs = options?.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const timeoutController = new AbortController();
  const upstreamSignal = init?.signal;
  let timedOut = false;
  let timeoutId: number | null = null;
  const handleUpstreamAbort = () => timeoutController.abort(upstreamSignal?.reason);

  if (upstreamSignal) {
    if (upstreamSignal.aborted) {
      handleUpstreamAbort();
    } else {
      upstreamSignal.addEventListener("abort", handleUpstreamAbort, { once: true });
    }
  }
  timeoutId = window.setTimeout(() => {
    timedOut = true;
    timeoutController.abort();
  }, timeoutMs);

  try {
    let response: Response;
    try {
      response = await fetch(apiUrl(path), {
        ...init,
        credentials: init?.credentials ?? "include",
        signal: timeoutController.signal,
        headers: {
          ...Object.fromEntries(headers.entries())
        }
      });
    } catch (error) {
      if (timedOut) {
        throw new ApiError(
          `请求超时：${path} 在 ${Math.round(timeoutMs / 1000)} 秒内未响应`,
          408,
          { path, timeout_ms: timeoutMs }
        );
      }
      throw error;
    }

    const contentType = response.headers.get("content-type") ?? "";
    let responseText = "";
    try {
      responseText = await response.text();
    } catch (error) {
      if (timedOut) {
        throw new ApiError(
          `请求超时：${path} 在 ${Math.round(timeoutMs / 1000)} 秒内未完成响应`,
          408,
          { path, timeout_ms: timeoutMs }
        );
      }
      if (typeof error === "object" && error !== null && "name" in error && error.name === "AbortError") {
        throw error;
      }
      throw new ApiError(`响应读取失败：${path}`, 502, { path });
    }

    if (response.ok && response.status === 204) {
      return decode(undefined);
    }

    let body: unknown = responseText;
    if (contentType.includes("application/json") || contentType.includes("+json")) {
      try {
        body = JSON.parse(responseText) as unknown;
      } catch {
        if (response.ok) {
          throw new ApiError(`响应 JSON 无法解析：${path}`, 502, { path });
        }
      }
    } else if (response.ok) {
      throw new ApiError(`响应数据不是 JSON：${path}`, 502, {
        path,
        content_type: contentType || null
      });
    }

    if (!response.ok) {
      const objectBody =
        typeof body === "object" && body !== null ? (body as Record<string, unknown>) : null;
      const detail =
        objectBody && typeof objectBody.last_error === "string"
          ? objectBody.last_error
          : objectBody && typeof objectBody.reason === "string"
            ? objectBody.reason
            : objectBody && typeof objectBody.message === "string"
              ? objectBody.message
            : objectBody && typeof objectBody.detail === "string"
              ? objectBody.detail
              : objectBody &&
                  typeof objectBody.detail === "object" &&
                  objectBody.detail !== null &&
                  typeof (objectBody.detail as Record<string, unknown>).message === "string"
                ? (objectBody.detail as Record<string, string>).message
              : response.statusText;
      if (response.status === 401 && objectBody?.code === "AUTHENTICATION_REQUIRED") {
        csrfToken = "";
        window.dispatchEvent(new CustomEvent("novasight:auth-required"));
      }
      if (response.status === 403 && objectBody?.code === "CSRF_REJECTED") {
        window.dispatchEvent(new CustomEvent("novasight:csrf-rejected"));
      }
      throw new ApiError(detail || "请求失败", response.status, body);
    }

    return decode(body);
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    upstreamSignal?.removeEventListener("abort", handleUpstreamAbort);
  }
}

export function getAuthSession(signal?: AbortSignal): Promise<AuthSession> {
  return requestJson<AuthSession>(
    API_PATHS.authSession,
    { signal },
    { timeoutMs: STATUS_REQUEST_TIMEOUT_MS },
    decodeAuthSession
  ).then(synchronizeAuthSession);
}

export function establishAuthSession(
  accessCode: string,
  signal?: AbortSignal
): Promise<AuthSession> {
  return requestJson<AuthSession>(
    API_PATHS.authSession,
    {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: encodeJsonBody({ access_code: requireNonBlank(accessCode, "access_code") })
    },
    { timeoutMs: STATUS_REQUEST_TIMEOUT_MS },
    decodeAuthSession
  ).then(synchronizeAuthSession);
}

export function logoutAuthSession(signal?: AbortSignal): Promise<AuthSession> {
  return requestJson<AuthSession>(
    API_PATHS.authSession,
    { method: "DELETE", signal },
    { timeoutMs: STATUS_REQUEST_TIMEOUT_MS },
    decodeAuthSession
  ).then(synchronizeAuthSession);
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>(
    API_PATHS.health,
    { signal },
    { timeoutMs: STATUS_REQUEST_TIMEOUT_MS },
    (value) => {
      if (typeof value !== "object" || value === null || Array.isArray(value) || !("ok" in value) || typeof value.ok !== "boolean") {
        throw new ApiError("健康检查响应结构无效", 502, value);
      }
      return { ok: value.ok };
    }
  );
}

export function getRuntimeState(
  signal?: AbortSignal,
  timeoutMs = STATUS_REQUEST_TIMEOUT_MS
): Promise<RuntimeState> {
  return requestJson<RuntimeState>(
    API_PATHS.runtimeState,
    { signal },
    { timeoutMs },
    decodeRuntimeState
  );
}

export function startRuntimePipeline(signal?: AbortSignal): Promise<void> {
  return requestJson<void>(
    API_PATHS.runtimeStart,
    { method: "POST", signal },
    undefined,
    (value) => {
      if (value !== undefined) {
        throw new ApiError("启动响应必须为空", 502, value);
      }
    }
  );
}

export function stopRuntimePipeline(signal?: AbortSignal): Promise<RuntimeState> {
  return requestJson<RuntimeState>(
    API_PATHS.runtimeStop,
    { method: "POST", signal },
    undefined,
    decodeRuntimeState
  );
}

function discardJsonCommandResponse(_value: unknown): void {
  // Command callers read the authoritative RuntimeState immediately after the
  // mutation. Do not pretend the internal RuntimeSnapshot receipt is config.
}

export function emergencyStopRuntimePipeline(signal?: AbortSignal): Promise<void> {
  return requestJson<void>(
    API_PATHS.runtimeEmergencyStop,
    { method: "POST", signal },
    undefined,
    (value) => {
      if (value !== undefined) {
        throw new ApiError("紧急停止响应必须为空", 502, value);
      }
    }
  );
}

export function getCrosshairStatus(): Promise<CrosshairSnapshot> {
  return requestJson<CrosshairSnapshot>(
    API_PATHS.crosshair,
    undefined,
    undefined,
    decodeCrosshairSnapshot
  );
}

export function learnCrosshair(): Promise<CrosshairLearnResponse> {
  return requestJson<CrosshairLearnResponse>(
    API_PATHS.crosshairLearn,
    { method: "POST" },
    undefined,
    decodeCrosshairLearnResponse
  );
}

export function clearCrosshairTemplate(): Promise<CrosshairSnapshot> {
  return requestJson<CrosshairSnapshot>(
    API_PATHS.crosshairTemplate,
    { method: "DELETE" },
    undefined,
    decodeCrosshairSnapshot
  );
}

export function crosshairTemplatePreviewUrl(cacheKey: number): string {
  return apiUrl(`${API_PATHS.crosshairTemplatePreview}?ts=${cacheKey}`);
}

export function diagnosticMoveKmNet(
  dx = 1,
  dy = 0
): Promise<DiagnosticMoveResponse> {
  const checkedDx = requireI16(dx, "dx");
  const checkedDy = requireI16(dy, "dy");
  if (checkedDx === 0 && checkedDy === 0) {
    return invalidRequest("dx/dy", "至少一个非零 i16", { dx, dy });
  }
  return requestJson<DiagnosticMoveResponse>(
    API_PATHS.kmnetDiagnosticMove,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({
        dx: checkedDx,
        dy: checkedDy,
        repeat: 1,
        interval_ms: 0,
        move_kind: "raw"
      })
    },
    undefined,
    decodeDiagnosticMoveResponse
  );
}

export function connectKmNet(signal?: AbortSignal): Promise<void> {
  return requestJson<void>(
    API_PATHS.kmnetConnect,
    { method: "POST", signal },
    undefined,
    discardJsonCommandResponse
  );
}

export function disconnectKmNet(signal?: AbortSignal): Promise<void> {
  return requestJson<void>(
    API_PATHS.kmnetDisconnect,
    { method: "POST", signal },
    undefined,
    discardJsonCommandResponse
  );
}

export function getRuntimeConfig(): Promise<RuntimeConfig> {
  return requestJson<RuntimeConfig>(
    API_PATHS.config,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeRuntimeConfig
  );
}

export function getConfigSchema(): Promise<ConfigSchemaResponse> {
  return requestJson<ConfigSchemaResponse>(
    API_PATHS.configSchema,
    undefined,
    undefined,
    decodeConfigSchema
  );
}

export function sanitizeRuntimeConfigForUpdate(config: RuntimeConfig): RuntimeConfig {
  const next = structuredClone(config);
  delete next.version;
  delete next.roi_size;
  return next;
}

export function updateRuntimeConfig(config: RuntimeConfig): Promise<ConfigUpdateResponse> {
  const payload = sanitizeRuntimeConfigForUpdate(config);
  return requestJson<ConfigUpdateResponse>(
    API_PATHS.config,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody(payload)
    },
    undefined,
    decodeConfigUpdate
  );
}

export function updateRuntimeConfigField(
  section: string,
  key: string,
  value: RuntimeConfigValue,
  expectedRevision?: number
): Promise<ConfigUpdateResponse> {
  requireNonBlank(section, "section");
  requireNonBlank(key, "key");
  if (expectedRevision !== undefined) {
    requireUnsignedSafeInteger(expectedRevision, "expected_revision");
  }
  return requestJson<ConfigUpdateResponse>(
    API_PATHS.config,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ section, key, value, expected_revision: expectedRevision })
    },
    undefined,
    decodeConfigUpdate
  );
}

export function updateRuntimeConfigCommand(
  command: ConfigCommandPayload
): Promise<ConfigUpdateResponse> {
  if (command.expected_revision !== undefined) {
    requireUnsignedSafeInteger(command.expected_revision, "expected_revision");
  }
  return requestJson<ConfigUpdateResponse>(
    API_PATHS.configCommands,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody(command)
    },
    undefined,
    decodeConfigUpdate
  );
}

export function setRuntimeOutputGate(
  enabled: boolean,
  expectedRevision?: number
): Promise<ConfigUpdateResponse> {
  return updateRuntimeConfigCommand({
    command: "set_output_gate",
    enabled,
    expected_revision: expectedRevision
  });
}

export function setRuntimeTriggerMode(
  mode: "always" | "hardware",
  expectedRevision?: number
): Promise<ConfigUpdateResponse> {
  return updateRuntimeConfigCommand({
    command: "set_trigger_mode",
    mode,
    expected_revision: expectedRevision
  });
}

export function getCaptureCapabilities(
  device: string
): Promise<CaptureCapabilitiesResponse> {
  const checkedDevice = requireNonBlank(device, "device");
  return requestJson<CaptureCapabilitiesResponse>(
    API_PATHS.captureCapabilities,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ device: checkedDevice })
    },
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeCaptureCapabilities
  );
}

export function selectCaptureProfile(payload: CaptureSelectPayload, signal?: AbortSignal): Promise<CaptureState> {
  requireNonBlank(payload.device, "device");
  if (payload.pixel_format !== undefined) {
    requireNonBlank(payload.pixel_format, "pixel_format");
  }
  if (payload.width !== undefined) requirePositiveU32(payload.width, "width");
  if (payload.height !== undefined) requirePositiveU32(payload.height, "height");
  if (payload.fps !== undefined) requirePositiveU32(payload.fps, "fps");
  if (
    payload.preference === "manual" &&
    (payload.pixel_format === undefined ||
      payload.width === undefined ||
      payload.height === undefined ||
      payload.fps === undefined)
  ) {
    return invalidRequest(
      "capture_profile",
      "manual 模式必须同时提供 pixel_format/width/height/fps",
      payload
    );
  }
  return requestJson<CaptureState>(
    API_PATHS.captureSelect,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody(payload),
      signal
    },
    undefined,
    decodeCaptureState
  );
}

export function stopCapture(reason = "用户停止采集"): Promise<CaptureState> {
  const checkedReason = requireNonBlank(reason, "reason");
  return requestJson<CaptureState>(
    API_PATHS.captureStop,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ reason: checkedReason })
    },
    undefined,
    decodeCaptureState
  );
}

export function setCapturePreviewEnabled(
  enabled: boolean,
  options?: { keepalive?: boolean }
): Promise<PreviewSnapshotState> {
  return requestJson<PreviewSnapshotState>(
    API_PATHS.capturePreview,
    {
      method: "POST",
      keepalive: options?.keepalive,
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ enabled })
    },
    undefined,
    decodePreviewSnapshot
  );
}

export function getModelProjects(): Promise<ModelProject[]> {
  return requestJson<ModelProject[]>(
    API_PATHS.modelProjects,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeModelProjects
  );
}

export function getModelCatalog(force = false): Promise<ModelCatalogResponse> {
  return requestJson<ModelCatalogResponse>(
    `${API_PATHS.modelCatalog}?force=${force ? "true" : "false"}`,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeModelCatalog
  );
}

export function registerCatalogModel(
  relativePath: string
): Promise<ModelCatalogRegisterResponse> {
  const checkedRelativePath = requireNonBlank(relativePath, "relative_path");
  return requestJson<ModelCatalogRegisterResponse>(
    "/api/models/catalog/register",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ relative_path: checkedRelativePath })
    },
    { timeoutMs: MODEL_OPERATION_TIMEOUT_MS },
    decodeModelCatalogRegisterResponse
  );
}

export function updateModelArtifactMetadata(
  artifactId: number,
  recommendation: ModelRecommendation,
  tags: string[]
): Promise<ModelArtifactMetadata> {
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<ModelArtifactMetadata>(
    `/api/models/artifacts/${checkedArtifactId}/metadata`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ recommendation, tags })
    },
    undefined,
    decodeModelArtifactMetadata
  );
}

export function getModelVersions(projectId: number): Promise<ModelVersion[]> {
  const checkedProjectId = requirePositiveSafeInteger(projectId, "project_id");
  return requestJson<ModelVersion[]>(
    `${API_PATHS.modelProjects}/${checkedProjectId}/versions`,
    undefined,
    undefined,
    decodeModelVersions
  );
}

export function getModelArtifacts(versionId: number): Promise<ModelArtifact[]> {
  const checkedVersionId = requirePositiveSafeInteger(versionId, "version_id");
  return requestJson<ModelArtifact[]>(
    `/api/models/versions/${checkedVersionId}/artifacts`,
    undefined,
    undefined,
    decodeModelArtifacts
  );
}

export function getConversionJobs(versionId?: number): Promise<ConversionJob[]> {
  if (typeof versionId !== "number") {
    return requestJson<ConversionJob[]>(
      API_PATHS.modelJobs,
      undefined,
      { timeoutMs: STANDARD_READ_TIMEOUT_MS },
      decodeConversionJobs
    );
  }
  const checkedVersionId = requirePositiveSafeInteger(versionId, "version_id");
  return requestJson<ConversionJob[]>(
    API_PATHS.modelJobsList,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ version_id: checkedVersionId })
    },
    undefined,
    decodeConversionJobs
  );
}

export function publishModel(
  projectId: number,
  artifactId: number,
  parserPreset: ParserPresetId = "auto"
): Promise<ModelPublishResponse> {
  const checkedProjectId = requirePositiveSafeInteger(projectId, "project_id");
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<ModelPublishResponse>(
    `${API_PATHS.modelProjects}/${checkedProjectId}/publish`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ artifact_id: checkedArtifactId, parser_preset: parserPreset })
    },
    { timeoutMs: MODEL_OPERATION_TIMEOUT_MS },
    decodeModelPublishResponse
  );
}

export function getDeepStreamRecommendation(
  artifactId: number
): Promise<DeepStreamRecommendationResponse> {
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<DeepStreamRecommendationResponse>(
    `/api/models/artifacts/${checkedArtifactId}/deepstream/recommendation`,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeDeepStreamRecommendation
  );
}

export function inspectModelArtifact(artifactId: number): Promise<ModelProfileResponse> {
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<ModelProfileResponse>(
    `/api/models/artifacts/${checkedArtifactId}/inspect`,
    { method: "POST" },
    { timeoutMs: MODEL_OPERATION_TIMEOUT_MS },
    decodeModelProfileResponse
  );
}

export function getModelProfile(artifactId: number): Promise<ModelProfileResponse> {
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<ModelProfileResponse>(
    `/api/models/artifacts/${checkedArtifactId}/profile`,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeModelProfileResponse
  );
}

export function configureModelProfile(
  artifactId: number,
  payload: ModelProfileConfigurePayload
): Promise<ModelProfileResponse> {
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<ModelProfileResponse>(
    `/api/models/artifacts/${checkedArtifactId}/profile`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody(payload)
    },
    { timeoutMs: MODEL_OPERATION_TIMEOUT_MS },
    decodeModelProfileResponse
  );
}

export function probeModelArtifact(
  artifactId: number,
  inputMode: "fixed" | "latest" = "fixed"
): Promise<ModelProbeResponse> {
  const checkedArtifactId = requirePositiveSafeInteger(artifactId, "artifact_id");
  return requestJson<ModelProbeResponse>(
    `/api/models/artifacts/${checkedArtifactId}/probe`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ input_mode: inputMode })
    },
    { timeoutMs: MODEL_OPERATION_TIMEOUT_MS },
    decodeModelProbeResponse
  );
}

export function rollbackModel(projectId: number): Promise<ModelPublishResponse> {
  const checkedProjectId = requirePositiveSafeInteger(projectId, "project_id");
  return requestJson<ModelPublishResponse>(
    `${API_PATHS.modelProjects}/${checkedProjectId}/rollback`,
    { method: "POST" },
    { timeoutMs: MODEL_OPERATION_TIMEOUT_MS },
    decodeModelPublishResponse
  );
}

export function getLicenseStatus(): Promise<LicenseStatus> {
  return requestJson<LicenseStatus>(
    API_PATHS.license,
    undefined,
    { timeoutMs: STANDARD_READ_TIMEOUT_MS },
    decodeLicenseStatus
  );
}

export function saveLicenseKey(key: string): Promise<LicenseStatus> {
  const checkedKey = requireNonBlank(key, "key");
  return requestJson<LicenseStatus>(
    API_PATHS.licenseActivate,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: encodeJsonBody({ key: checkedKey })
    },
    undefined,
    decodeLicenseStatus
  );
}

export function clearLicenseKey(): Promise<LicenseStatus> {
  return requestJson<LicenseStatus>(
    API_PATHS.license,
    { method: "DELETE" },
    undefined,
    decodeLicenseStatus
  );
}
