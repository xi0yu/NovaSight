import { useCallback, useSyncExternalStore } from "react";

import { type ErrorSeverity, isQuietErrorsEnabled, normalizeError } from "./api-error";

export type ToastTone = ErrorSeverity | "success";

export type Toast = {
  id: number;
  tone: ToastTone;
  title: string;
  detail?: string;
  source: string;
  status: number | null;
  createdAt: number;
};

type Listener = () => void;

const DEFAULT_DURATION_MS = 4500;
const MAX_TOASTS = 4;
const DUPLICATE_WINDOW_MS = 1500;

let nextId = 1;
let toasts: Toast[] = [];
let errorNotices: Toast[] = [];
const listeners = new Set<Listener>();
const errorListeners = new Set<Listener>();
const lastToastAtBySignature = new Map<string, number>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

function emitErrors(): void {
  for (const listener of errorListeners) {
    listener();
  }
}

function setToasts(next: Toast[]): void {
  toasts = next;
  emit();
}

function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): Toast[] {
  return toasts;
}

export function useToasts(): Toast[] {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function useErrorNotices(): Toast[] {
  return useSyncExternalStore(
    (listener) => {
      errorListeners.add(listener);
      return () => errorListeners.delete(listener);
    },
    () => errorNotices,
    () => errorNotices
  );
}

export function useClearErrorNotices(): () => void {
  return useCallback(() => {
    errorNotices = [];
    emitErrors();
  }, []);
}

function dismiss(id: number): void {
  setToasts(toasts.filter((toast) => toast.id !== id));
}

export function useDismissToast(): (id: number) => void {
  return useCallback((id: number) => {
    dismiss(id);
  }, []);
}

function buildToast(input: {
  tone: ToastTone;
  title: string;
  detail?: string;
  source: string;
  status: number | null;
}): Toast {
  return {
    id: nextId++,
    tone: input.tone,
    title: input.title,
    detail: input.detail,
    source: input.source,
    status: input.status,
    createdAt: Date.now()
  };
}

function pushToast(toast: Toast): void {
  if (isQuietErrorsEnabled()) {
    // Quiet mode: surface to console but never to UI. Backend logs own the truth.
    // eslint-disable-next-line no-console
    console.warn(`[ns-quiet] ${toast.title}${toast.detail ? " — " + toast.detail : ""}`);
    return;
  }
  for (const [signature, createdAt] of lastToastAtBySignature) {
    if (toast.createdAt - createdAt > DUPLICATE_WINDOW_MS) {
      lastToastAtBySignature.delete(signature);
    }
  }
  const signature = `${toast.source}\u0000${toast.title}\u0000${toast.detail ?? ""}`;
  const previousCreatedAt = lastToastAtBySignature.get(signature);
  if (previousCreatedAt !== undefined && toast.createdAt - previousCreatedAt <= DUPLICATE_WINDOW_MS) {
    return;
  }
  lastToastAtBySignature.set(signature, toast.createdAt);
  if (toast.tone === "warn" || toast.tone === "error") {
    errorNotices = [...errorNotices, toast].slice(-20);
    emitErrors();
    return;
  }
  const next = [...toasts, toast].slice(-MAX_TOASTS);
  setToasts(next);
  if (toast.tone !== "info" || toast.title !== "请求已取消") {
    window.setTimeout(() => {
      dismiss(toast.id);
    }, DEFAULT_DURATION_MS);
  }
}

export function pushToastRaw(toast: Omit<Toast, "id" | "createdAt">): void {
  pushToast(buildToast(toast));
}

export function reportError(
  error: unknown,
  options: {
    source: string;
    title?: string;
    fallback?: string;
    publicDetail?: string;
    exposeStatus?: boolean;
  }
): void {
  const normalized = normalizeError(error, {
    source: options.source,
    fallback: options.fallback
  });
  // eslint-disable-next-line no-console
  console.error(`[${options.source}]`, error);
  pushToast(
    buildToast({
      tone: normalized.severity,
      title: options.title ?? errorTitleForSource(options.source),
      detail: options.publicDetail ?? normalized.message,
      source: normalized.source,
      status: options.exposeStatus === false ? null : normalized.status
    })
  );
}

function errorTitleForSource(source: string): string {
  if (source.includes("runtime")) return "运行态失败";
  if (source.includes("config")) return "配置读取失败";
  if (source.includes("capture")) return "采集失败";
  if (source.includes("model")) return "模型任务失败";
  if (source.includes("license")) return "授权校验失败";
  if (source.includes("kmnet") || source.includes("kmNet")) return "kmNet 失败";
  if (source.includes("websocket") || source.includes("ws")) return "实时通道异常";
  return "请求失败";
}

export function reportInfo(title: string, detail?: string, source = "system"): void {
  pushToast(
    buildToast({ tone: "info", title, detail, source, status: null })
  );
}

export function reportSuccess(title: string, detail?: string, source = "system"): void {
  pushToast(
    buildToast({ tone: "success", title, detail, source, status: null })
  );
}
