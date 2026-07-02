import { ApiError, type CaptureState } from "../../api";

export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "无法连接 NovaSight 后端";
}

export function formatTime(date: Date | null): string {
  if (!date) {
    return "从未更新";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

export function formatEpoch(seconds: number | null | undefined): string {
  if (!seconds) {
    return "无";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(seconds * 1000));
}

export function statusTone(value: boolean | undefined): "good" | "warn" | "bad" {
  if (value === true) {
    return "good";
  }
  if (value === false) {
    return "bad";
  }
  return "warn";
}

export function formatProfile(capture: CaptureState | undefined): string {
  if (!capture?.profile) {
    return "未配置";
  }
  return `${capture.profile.pixel_format} ${capture.profile.width}x${capture.profile.height} @ ${capture.profile.fps}`;
}
