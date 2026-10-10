import type { NovaIconName } from "./iconNames";

export type NovaStatus = "normal" | "running" | "waiting" | "warning" | "error" | "disabled";
export type StatusTone = "good" | "warn" | "bad" | "idle";

export type StatusSpec = {
  icon: NovaIconName;
  className: string;
  label: string;
};

export const statusSpecs: Record<NovaStatus, StatusSpec> = {
  normal: {
    icon: "check-circle",
    className: "normal",
    label: "Normal",
  },
  running: {
    icon: "activity-pulse",
    className: "running",
    label: "Running",
  },
  waiting: {
    icon: "clock",
    className: "waiting",
    label: "Waiting",
  },
  warning: {
    icon: "triangle-alert",
    className: "warning",
    label: "Warning",
  },
  error: {
    icon: "error-circle",
    className: "error",
    label: "Error",
  },
  disabled: {
    icon: "empty-circle",
    className: "disabled",
    label: "Disabled",
  },
};

export function toneToStatus(tone: StatusTone): NovaStatus {
  switch (tone) {
    case "good":
      return "normal";
    case "warn":
      return "warning";
    case "bad":
      return "error";
    case "idle":
    default:
      return "waiting";
  }
}
