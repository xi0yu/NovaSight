export type ServerActivityLevel = "info" | "warn" | "error";

export interface ServerActivityEvent {
  daemon_instance_id: string;
  id: number;
  occurred_at_ms: number;
  level: ServerActivityLevel;
  tag: string;
  title: string;
  message: string;
  technical_detail: string;
  request_id: string | null;
  count: number;
}

export interface ActivityHistoryResponse {
  events: ServerActivityEvent[];
}

export interface ActivityEventFrame {
  kind: "activity_event";
  event: ServerActivityEvent;
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} 应为对象`);
  }
  return value as Record<string, unknown>;
}

function activityEvent(value: unknown, path: string): ServerActivityEvent {
  const item = record(value, path);
  if (!Number.isSafeInteger(item.id) || !Number.isSafeInteger(item.occurred_at_ms)) throw new Error(`${path} 缺少有效时间或编号`);
  if (item.level !== "info" && item.level !== "warn" && item.level !== "error") throw new Error(`${path}.level 无效`);
  for (const key of ["daemon_instance_id", "tag", "title", "message", "technical_detail"] as const) {
    if (typeof item[key] !== "string") throw new Error(`${path}.${key} 应为字符串`);
  }
  if (item.request_id !== null && typeof item.request_id !== "string") throw new Error(`${path}.request_id 应为字符串或 null`);
  if (!Number.isSafeInteger(item.count) || Number(item.count) < 1) throw new Error(`${path}.count 无效`);
  return item as unknown as ServerActivityEvent;
}

export function decodeActivityHistory(value: unknown): ActivityHistoryResponse {
  const root = record(value, "activity");
  if (!Array.isArray(root.events)) throw new Error("activity.events 应为数组");
  return { events: root.events.map((event, index) => activityEvent(event, `activity.events[${index}]`)) };
}

export function decodeActivityEventFrame(value: unknown): ActivityEventFrame {
  const root = record(value, "activity_frame");
  if (root.kind !== "activity_event") throw new Error("activity_frame.kind 无效");
  return { kind: "activity_event", event: activityEvent(root.event, "activity_frame.event") };
}
