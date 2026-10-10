export type RuntimeDeliveryStatus =
  | "connecting"
  | "connected"
  | "fallback"
  | "stale"
  | "disconnected"
  | "offline"
  | "paused";

export type RuntimeDeliveryTone = "good" | "warn" | "bad" | "idle";

const DELIVERY_LABELS: Record<RuntimeDeliveryStatus, string> = {
  connected: "实时更新",
  fallback: "降级更新",
  connecting: "正在连接实时通道",
  stale: "运行数据已陈旧",
  disconnected: "运行状态不可用",
  offline: "浏览器离线",
  paused: "后台已暂停"
};

const DELIVERY_DESCRIPTIONS: Record<RuntimeDeliveryStatus, string> = {
  connected: "运行状态由实时通道持续更新。",
  fallback: "实时通道异常，运行状态正通过低频轮询继续更新。",
  connecting: "正在建立实时通道，当前运行状态可能短暂延迟。",
  stale: "实时通道仍在连接，但运行状态已经超过可信更新时间。",
  disconnected: "实时通道和运行态兜底当前都不可用。",
  offline: "浏览器网络已断开，恢复联网后会自动重新同步。",
  paused: "页面处于后台，实时连接与普通轮询已暂停。"
};

export function runtimeDeliveryLabel(status: RuntimeDeliveryStatus): string {
  return DELIVERY_LABELS[status];
}

export function runtimeDeliveryDescription(status: RuntimeDeliveryStatus): string {
  return DELIVERY_DESCRIPTIONS[status];
}

export function runtimeDeliveryTone(status: RuntimeDeliveryStatus): RuntimeDeliveryTone {
  switch (status) {
    case "connected":
      return "good";
    case "fallback":
    case "connecting":
    case "stale":
      return "warn";
    case "disconnected":
    case "offline":
      return "bad";
    case "paused":
    default:
      return "idle";
  }
}
