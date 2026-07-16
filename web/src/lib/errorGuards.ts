import { pushToastRaw, reportError, reportSuccess } from "./toast";
import { normalizeError } from "./api-error";

let installed = false;
let lastUnreportedReason: unknown = null;
let lastReasonAt = 0;
const activeWebSocketFailures = new Set<string>();

function isAbortReason(reason: unknown): boolean {
  if (reason === null || typeof reason !== "object") {
    return false;
  }
  if (!("name" in reason)) {
    return false;
  }
  return reason.name === "AbortError";
}

export function reportUnhandledRejection(reason: unknown): void {
  // Coalesce duplicate "same error within 750ms" — common during
  // initial WebSocket connection storms while a backend is offline.
  if (
    lastUnreportedReason === reason ||
    (Date.now() - lastReasonAt < 750 && lastUnreportedReason === reason)
  ) {
    return;
  }
  lastUnreportedReason = reason;
  lastReasonAt = Date.now();

  if (isAbortReason(reason)) {
    return;
  }
  const source = "unhandled-promise";
  const normalized = normalizeError(reason, { source });
  pushToastRaw({
    tone: normalized.severity === "info" ? "info" : normalized.severity,
    title: "未捕获的异常",
    detail: normalized.message,
    source,
    status: normalized.status
  });
}

export function installGlobalErrorGuards(): void {
  if (installed || typeof window === "undefined") {
    return;
  }
  installed = true;

  window.addEventListener("unhandledrejection", (event) => {
    reportUnhandledRejection(event.reason);
    event.preventDefault();
  });

  window.addEventListener("error", (event) => {
    if (event.defaultPrevented) {
      return;
    }
    // Genuine script errors only — unhandledrejection handles promises.
    if (event.error) {
      reportError(event.error, { source: "window-error" });
    } else if (event.message) {
      pushToastRaw({
        tone: "error",
        title: "脚本错误",
        detail: event.message,
        source: "window-error",
        status: null
      });
    }
  });
}

export function reportWebSocketFailure(reason: unknown, path: string): boolean {
  if (isAbortReason(reason)) {
    return false;
  }
  const source = `websocket:${path}`;
  if (activeWebSocketFailures.has(source)) {
    return false;
  }
  activeWebSocketFailures.add(source);
  const normalized = normalizeError(reason, { source });
  pushToastRaw({
    tone: "warn",
    title: "实时通道异常",
    detail: normalized.message,
    source,
    status: normalized.status
  });
  return true;
}

export function reportWebSocketRecovered(path: string): void {
  const source = `websocket:${path}`;
  if (!activeWebSocketFailures.delete(source)) {
    return;
  }
  reportSuccess("实时更新已恢复", "运行状态已重新切换到实时通道。", source);
}

export function clearWebSocketFailure(path: string): void {
  activeWebSocketFailures.delete(`websocket:${path}`);
}

export function reportNetworkFailure(reason: unknown, path: string): void {
  if (isAbortReason(reason)) {
    return;
  }
  const source = `network:${path}`;
  const normalized = normalizeError(reason, { source });
  pushToastRaw({
    tone: normalized.severity,
    title: "网络请求失败",
    detail: normalized.message,
    source,
    status: normalized.status
  });
}
