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
    ready: "可选择",
    pending: "切换时检查",
    running: "处理中",
    need_confirm: "切换时检查",
    invalid: "模型不可用",
    failed: "处理未完成",
    unsupported: "不支持"
  };
  return labels[status] ?? "待确认";
}

export function artifactStatus(artifact: ModelArtifact | null): string {
  return artifact?.status ?? "pending";
}
