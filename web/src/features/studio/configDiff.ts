import type { RuntimeConfig } from "../../api";

export type ConfigDifference = { path: string; before: string; after: string; restart: boolean };
const sensitive = /password|secret|token|credential|private[_-]?key|license[_-]?key/i;

function display(value: unknown, path: string): string {
  if (value === undefined) return "未设置";
  if (sensitive.test(path)) return "••••（已隐藏）";
  return JSON.stringify(value, (key, item) => sensitive.test(key) ? "••••（已隐藏）" : item);
}

export function configDifferences(before: RuntimeConfig, after: RuntimeConfig): ConfigDifference[] {
  const changes: ConfigDifference[] = [];
  const visit = (left: unknown, right: unknown, path: string) => {
    if (JSON.stringify(left) === JSON.stringify(right)) return;
    const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
    if (object(left) && object(right) && !sensitive.test(path)) {
      for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) visit(left[key], right[key], path ? `${path}.${key}` : key);
    } else {
      changes.push({ path, before: display(left, path), after: display(right, path), restart: /^(server|paths)(\.|$)/.test(path) });
    }
  };
  for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
    if (!["revision", "version", "roi_size"].includes(key)) visit(before[key], after[key], key);
  }
  return changes;
}
