import { ApiError } from "../api";

export type ErrorSeverity = "info" | "warn" | "error";

export type NormalizedError = {
  message: string;
  status: number | null;
  severity: ErrorSeverity;
  source: string;
};

const NETWORK_FALLBACK = "无法连接 NovaSight 后端";

function severityForStatus(status: number | null): ErrorSeverity {
  if (status === null) {
    return "error";
  }
  if (status === 401 || status === 403) {
    return "warn";
  }
  if (status === 404) {
    return "warn";
  }
  if (status === 408 || status === 429) {
    return "warn";
  }
  if (status >= 400 && status < 500) {
    return "warn";
  }
  return "error";
}

function isAbortError(error: unknown): boolean {
  if (error === null || typeof error !== "object") {
    return false;
  }
  if (!("name" in error)) {
    return false;
  }
  // After `in` narrowing the property is `unknown`; no further cast needed.
  return error.name === "AbortError";
}

function extractStringField(record: object, key: string): string | null {
  if (!(key in record)) {
    return null;
  }
  // record: object has no index signature; the `in` check above narrows the
  // surface but the compiler still requires an assertion to read by key.
  const bag = record as Record<string, unknown>;
  const value = bag[key];
  return typeof value === "string" ? value : null;
}

function extractFromUnknownBody(body: unknown): string | null {
  if (body === null || typeof body !== "object") {
    return null;
  }
  const direct =
    extractStringField(body, "last_error") ??
    extractStringField(body, "reason") ??
    extractStringField(body, "detail") ??
    extractStringField(body, "message");
  if (direct !== null) {
    return direct;
  }
  if ("detail" in body && typeof body.detail === "object" && body.detail !== null) {
    return extractStringField(body.detail, "message");
  }
  return null;
}

export function normalizeError(
  error: unknown,
  options: { source: string; fallback?: string }
): NormalizedError {
  const fallback = options.fallback ?? NETWORK_FALLBACK;
  if (isAbortError(error)) {
    return {
      message: "请求已取消",
      status: null,
      severity: "info",
      source: options.source
    };
  }
  if (error instanceof ApiError) {
    const bodyMessage = extractFromUnknownBody(error.detail);
    const message = bodyMessage || error.message || fallback;
    return {
      message,
      status: error.status,
      severity: severityForStatus(error.status),
      source: options.source
    };
  }
  if (error instanceof Error) {
    return {
      message: error.message || fallback,
      status: null,
      severity: "error",
      source: options.source
    };
  }
  return {
    message: fallback,
    status: null,
    severity: "error",
    source: options.source
  };
}

export function isQuietErrorsEnabled(): boolean {
  try {
    return window.localStorage.getItem("ns-quiet-errors") === "1";
  } catch {
    return false;
  }
}

export function setQuietErrorsEnabled(enabled: boolean): void {
  try {
    if (enabled) {
      window.localStorage.setItem("ns-quiet-errors", "1");
    } else {
      window.localStorage.removeItem("ns-quiet-errors");
    }
  } catch {
    // localStorage unavailable (Safari private mode / quota) — silently ignore
  }
}
