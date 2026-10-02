import type { SVGProps, ReactNode } from "react";

import { type NovaIconName } from "../../design/iconNames";

type IconBase =
  | "activity"
  | "alert"
  | "arrow"
  | "batch"
  | "box"
  | "camera"
  | "check"
  | "chip"
  | "clock"
  | "command"
  | "control"
  | "curve"
  | "database"
  | "device"
  | "frame"
  | "gpu"
  | "grid"
  | "help"
  | "link"
  | "lock"
  | "log"
  | "model"
  | "network"
  | "notification"
  | "plug"
  | "roi"
  | "search"
  | "settings"
  | "target"
  | "terminal"
  | "x";

export type { NovaIconName };

export type NovaIconProps = Omit<SVGProps<SVGSVGElement>, "name"> & {
  name: NovaIconName;
  size?: number;
  strokeWidth?: number;
  label?: string;
};

const iconPaths: Record<IconBase, ReactNode> = {
  activity: <path d="M3 12h4l2-5 4 10 2-5h6" />,
  alert: (
    <>
      <path d="M12 4l9 16H3L12 4z" />
      <path d="M12 9v4M12 17h.01" />
    </>
  ),
  arrow: (
    <>
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </>
  ),
  batch: (
    <>
      <rect x="5" y="5" width="9" height="9" rx="2" />
      <rect x="10" y="10" width="9" height="9" rx="2" />
    </>
  ),
  box: (
    <>
      <path d="M4 8l8-4 8 4-8 4-8-4z" />
      <path d="M4 8v8l8 4 8-4V8" />
      <path d="M12 12v8" />
    </>
  ),
  camera: (
    <>
      <path d="M5 8h2l1.5-2h7L17 8h2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2z" />
      <circle cx="12" cy="13" r="3.2" />
    </>
  ),
  check: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M8.5 12.2l2.2 2.2 4.8-5" />
    </>
  ),
  chip: (
    <>
      <rect x="5" y="5" width="14" height="14" rx="3" />
      <path d="M9 9h6M9 12h6M9 15h4" />
      <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7.5V12l3 2" />
    </>
  ),
  command: (
    <>
      <path d="M9 9H7a3 3 0 1 1 3-3v12a3 3 0 1 1-3-3h10a3 3 0 1 1-3 3V6a3 3 0 1 1 3 3H9z" />
    </>
  ),
  control: (
    <>
      <path d="M4 15c2.4-6 5.5-6 8-1s5.6 4.8 8-2" />
      <circle cx="7" cy="14" r="1.5" />
      <circle cx="12" cy="11" r="1.5" />
      <circle cx="17" cy="13" r="1.5" />
    </>
  ),
  curve: (
    <>
      <path d="M4 15c2-5 4-5 6-1s4 4 6-1 3-5 4-4" />
      <path d="M4 18c5-1 9-2 16-9" opacity="0.55" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="6" rx="7" ry="3" />
      <path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6" />
      <path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
    </>
  ),
  device: (
    <>
      <rect x="6" y="3" width="12" height="18" rx="2.5" />
      <path d="M10 17h4M10 7h4" />
    </>
  ),
  frame: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="2.5" />
      <path d="M8 9h8M8 13h5" />
    </>
  ),
  gpu: (
    <>
      <rect x="4" y="7" width="16" height="10" rx="2.5" />
      <path d="M8 10h2v4H8zM12 10h2v4h-2zM16 10h1v4h-1" />
      <path d="M7 4v3M12 4v3M17 4v3M7 17v3M12 17v3M17 17v3" />
    </>
  ),
  grid: (
    <>
      <rect x="4" y="4" width="6" height="6" rx="1.4" />
      <rect x="14" y="4" width="6" height="6" rx="1.4" />
      <rect x="4" y="14" width="6" height="6" rx="1.4" />
      <rect x="14" y="14" width="6" height="6" rx="1.4" />
    </>
  ),
  help: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M9.8 9a2.6 2.6 0 0 1 4.6 1.6c0 2-2.4 2.1-2.4 4" />
      <path d="M12 17h.01" />
    </>
  ),
  link: (
    <>
      <path d="M9.5 14.5l5-5" />
      <path d="M10.5 7.5l1-1a4 4 0 0 1 5.7 5.7l-1 1" />
      <path d="M13.5 16.5l-1 1a4 4 0 0 1-5.7-5.7l1-1" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="10" width="14" height="10" rx="2" />
      <path d="M8 10V7a4 4 0 0 1 8 0v3" />
    </>
  ),
  log: (
    <>
      <path d="M6 4h9l3 3v13H6z" />
      <path d="M15 4v4h4M9 11h6M9 15h6" />
    </>
  ),
  model: (
    <>
      <path d="M12 3l7 4v10l-7 4-7-4V7l7-4z" />
      <circle cx="12" cy="12" r="2" />
      <path d="M12 10V7M10.3 13l-2.8 1.8M13.7 13l2.8 1.8" />
    </>
  ),
  network: (
    <>
      <circle cx="6" cy="12" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="18" cy="18" r="2" />
      <path d="M7.8 11.1l8.4-4.2M7.8 12.9l8.4 4.2" />
    </>
  ),
  notification: (
    <>
      <path d="M7 10a5 5 0 0 1 10 0v4l2 3H5l2-3v-4z" />
      <path d="M10 20h4" />
    </>
  ),
  plug: (
    <>
      <path d="M8 3v6M16 3v6M7 9h10v3a5 5 0 0 1-10 0V9z" />
      <path d="M12 17v4" />
    </>
  ),
  roi: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="2.5" />
      <path d="M8 10V8h2M14 8h2v2M16 14v2h-2M10 16H8v-2" />
      <circle cx="8" cy="8" r="1" />
      <circle cx="16" cy="8" r="1" />
      <circle cx="16" cy="16" r="1" />
      <circle cx="8" cy="16" r="1" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6" />
      <path d="M16 16l4 4" />
    </>
  ),
  settings: (
    <>
      <path d="M12 8.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7z" />
      <path d="M4 13.5v-3l2.1-.5.7-1.7L5.6 6.5l2.1-2.1 1.8 1.2 1.7-.7L11.5 3h3l.5 2.1 1.7.7 1.8-1.2 2.1 2.1-1.2 1.8.7 1.7 2.1.5v3l-2.1.5-.7 1.7 1.2 1.8-2.1 2.1-1.8-1.2-1.7.7-.5 2.1h-3l-.5-2.1-1.7-.7-1.8 1.2-2.1-2.1 1.2-1.8-.7-1.7L4 13.5z" />
    </>
  ),
  target: (
    <>
      <rect x="5" y="5" width="14" height="14" rx="4" />
      <circle cx="12" cy="12" r="1.7" />
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3" opacity="0.55" />
    </>
  ),
  terminal: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path d="M8 10l2 2-2 2M12 15h4" />
    </>
  ),
  x: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M9 9l6 6M15 9l-6 6" />
    </>
  ),
};

const iconAliases: Partial<Record<NovaIconName, IconBase>> = {
  dashboard: "grid",
  studio: "model",
  capture: "camera",
  inference: "chip",
  models: "model",
  target: "target",
  tracking: "target",
  prediction: "curve",
  control: "control",
  devices: "device",
  performance: "activity",
  logs: "log",
  system: "chip",
  settings: "settings",
  help: "help",
  account: "device",
  search: "search",
  command: "command",
  notification: "notification",
  camera: "camera",
  "hdmi-input": "plug",
  "capture-card": "device",
  "video-stream": "frame",
  frame: "frame",
  fps: "activity",
  crop: "roi",
  roi: "roi",
  resolution: "frame",
  "aspect-ratio": "frame",
  rotate: "arrow",
  mirror: "frame",
  exposure: "activity",
  "refresh-frame": "activity",
  fullscreen: "frame",
  preview: "frame",
  "source-switch": "arrow",
  "signal-input": "plug",
  "signal-lost": "x",
  timestamp: "clock",
  "latest-frame": "batch",
  "ai-model": "model",
  "neural-network": "network",
  tensorrt: "chip",
  cuda: "chip",
  gpu: "gpu",
  cpu: "chip",
  "inference-engine": "chip",
  "model-load": "model",
  "model-verify": "check",
  fp16: "box",
  fp32: "box",
  int8: "box",
  onnx: "model",
  engine: "chip",
  "input-tensor": "box",
  "output-tensor": "box",
  batch: "batch",
  "inference-speed": "activity",
  "model-cache": "database",
  "model-error": "alert",
  "model-compile": "chip",
  "model-switch": "arrow",
  "detection-box": "target",
  "target-center": "target",
  "candidate-target": "target",
  "current-target": "target",
  "target-lock": "target",
  "target-lost": "x",
  "target-filter": "target",
  fov: "roi",
  "track-trace": "curve",
  kalman: "curve",
  "velocity-vector": "arrow",
  acceleration: "activity",
  "future-position": "curve",
  confidence: "check",
  "track-id": "target",
  occlusion: "target",
  reidentify: "target",
  "target-switch": "arrow",
  "prediction-line": "curve",
  controller: "control",
  pid: "control",
  pd: "control",
  ema: "curve",
  smoothing: "curve",
  "response-curve": "curve",
  gain: "activity",
  damping: "curve",
  error: "alert",
  angle: "curve",
  "pixel-error": "roi",
  output: "arrow",
  "velocity-limit": "activity",
  "acceleration-limit": "activity",
  deadzone: "target",
  "max-step": "arrow",
  scheduler: "clock",
  "command-queue": "batch",
  "device-send": "plug",
  "pause-output": "clock",
  "emergency-stop": "x",
  "auto-tune": "settings",
  "parameter-reset": "activity",
  "parameter-save": "check",
  jetson: "chip",
  usb: "plug",
  hid: "device",
  kmbox: "device",
  network: "network",
  lan: "network",
  wifi: "network",
  latency: "clock",
  temperature: "activity",
  fan: "activity",
  memory: "chip",
  disk: "database",
  power: "plug",
  service: "device",
  daemon: "terminal",
  plugin: "box",
  driver: "chip",
  database: "database",
  "backend-api": "network",
  websocket: "network",
  log: "log",
  crash: "alert",
  restart: "activity",
  update: "arrow",
  terminal: "terminal",
  connected: "check",
  disconnected: "plug",
  connecting: "activity",
  "connection-error": "alert",
  "permission-denied": "lock",
  "driver-missing": "alert",
  start: "arrow",
  stop: "x",
  pause: "clock",
  resume: "arrow",
  refresh: "activity",
  save: "check",
  delete: "x",
  edit: "settings",
  copy: "batch",
  import: "arrow",
  export: "arrow",
  upload: "arrow",
  download: "arrow",
  expand: "arrow",
  collapse: "arrow",
  back: "arrow",
  forward: "arrow",
  more: "command",
  filter: "settings",
  sort: "activity",
  pin: "target",
  unlock: "lock",
  lock: "lock",
  show: "frame",
  hide: "x",
  "restore-default": "activity",
  undo: "arrow",
  redo: "arrow",
  "check-circle": "check",
  "online-dot": "check",
  "shield-check": "check",
  "play-circle": "arrow",
  "activity-pulse": "activity",
  "rotating-arc": "activity",
  "data-stream": "activity",
  clock: "clock",
  "empty-circle": "clock",
  "triangle-alert": "alert",
  "thermometer-warning": "alert",
  "clock-alert": "alert",
  "error-circle": "x",
  "plug-off": "plug",
  "broken-link": "link",
  "x-circle": "x",
};

export function NovaIcon({
  name,
  size = 20,
  strokeWidth = 1.7,
  label,
  className,
  ...props
}: NovaIconProps) {
  const accessibilityProps = label
    ? { role: "img" as const, "aria-label": label }
    : { "aria-hidden": true };
  const base = iconAliases[name] ?? "box";
  const resolvedClassName = ["nova-icon", className].filter(Boolean).join(" ");

  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={resolvedClassName}
      {...accessibilityProps}
      {...props}
    >
      {iconPaths[base]}
    </svg>
  );
}
