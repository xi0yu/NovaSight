import { ApiError, type CaptureState } from "../../api";

function apiErrorCode(error: ApiError): string {
  if (typeof error.detail !== "object" || error.detail === null) {
    return "";
  }
  const detail = error.detail as Record<string, unknown>;
  return typeof detail.code === "string" ? detail.code : "";
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const code = apiErrorCode(error);
    if (code === "CONFIG_REVISION_CONFLICT") {
      return "配置刚刚被另一项操作更新，已保留较新的版本；请刷新后重试本次修改。";
    }
    if (code === "CONFIG_RESTART_REQUIRED") {
      return "新配置已经保存，但尚未进入当前进程；请重启 novasightd 后继续。";
    }
    if (code === "HARDWARE_OUTPUT_DISABLED") {
      return "当前以 dry-run 模式运行，不能连接或控制物理 kmNet 设备。";
    }
    if (code === "DEVICE_NOT_CONFIGURED" || code === "DEVICE_UNCOMMISSIONED" || code === "device_uncommissioned") {
      return "kmNet 尚未完成设备委任；请先保存有效的地址、端口和 UUID。";
    }
    return error.message;
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
