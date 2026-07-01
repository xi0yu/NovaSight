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

export type ModelArtifact = {
  id: number;
  version_id: number;
  kind: string;
  path: string;
  checksum: string;
  status: string;
};

export type Deployment = {
  id: number;
  project_id: number;
  artifact_id: number;
  previous_artifact_id: number | null;
};

export type ActiveModel = {
  project: ModelProject | null;
  deployment: Deployment;
  artifact: ModelArtifact | null;
};

export type RuntimeState = {
  running: boolean;
  source: string;
  active_model: ActiveModel | null;
  executor: ExecutorStatus;
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

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");

  const response = await fetch(path, {
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
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : response.statusText;
    throw new ApiError(detail || "Request failed", response.status, body);
  }

  return body as T;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/healthz");
}

export function getRuntimeState(): Promise<RuntimeState> {
  return requestJson<RuntimeState>("/api/runtime/state");
}

export function getPlugins(): Promise<PluginInfo[]> {
  return requestJson<PluginInfo[]>("/api/plugins");
}

export function getModelProjects(): Promise<ModelProject[]> {
  return requestJson<ModelProject[]>("/api/models/projects");
}
