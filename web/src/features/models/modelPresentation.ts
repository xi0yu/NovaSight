import type { ModelArtifact } from "../../api";

export function formatModelSize(value: unknown): string {
  const sizeBytes = typeof value === "number" && Number.isFinite(value) ? value : null;
  if (sizeBytes === null || sizeBytes < 0) {
    return "大小未知";
  }
  if (sizeBytes >= 1_000_000_000) {
    return `${(sizeBytes / 1_000_000_000).toFixed(2)} GB`;
  }
  return `${(sizeBytes / 1_000_000).toFixed(2)} MB`;
}

export function modelStatusTone(status: string): "good" | "warn" | "bad" | "idle" {
  if (status === "ready") {
    return "good";
  }
  if (status === "invalid" || status === "failed" || status === "unsupported") {
    return "bad";
  }
  if (status === "need_confirm" || status === "pending" || status === "running") {
    return "warn";
  }
  return "idle";
}

export function modelStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    ready: "可用",
    pending: "待验证",
    running: "处理中",
    need_confirm: "待加载验证",
    invalid: "无效",
    failed: "失败",
    unsupported: "不支持"
  };
  return labels[status] ?? status;
}

export function artifactStatus(artifact: ModelArtifact | null): string {
  return artifact?.status ?? "未登记";
}
