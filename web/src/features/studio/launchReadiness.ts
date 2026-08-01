import type { LicenseStatus, RuntimeConfig, RuntimeState } from "../../api";
import { getRuntimeMainlineStatus, type RuntimeMainlineStatus } from "../shared/runtimeStatus";

export type LaunchReadinessState = "ready" | "action" | "blocked" | "idle";

export type LaunchReadinessAction =
  | "open-license"
  | "open-model-manager"
  | "open-capture"
  | "start-mainline"
  | "open-control"
  | "open-kmnet-test"
  | "open-params";

export type LaunchReadinessItem = {
  id: "license" | "model" | "capture" | "deepstream" | "control" | "kmnet";
  label: string;
  state: LaunchReadinessState;
  detail: string;
  evidence: string;
  blocking: boolean;
  action?: LaunchReadinessAction;
  actionLabel?: string;
};

export type LaunchProductConfigRow = {
  label: string;
  value: string;
  detail: string;
};

export type LaunchReadinessSummary = {
  state: LaunchReadinessState;
  title: string;
  detail: string;
  readyCount: number;
  blockingCount: number;
  score: number;
  primaryAction?: LaunchReadinessAction;
  primaryActionLabel?: string;
  secondaryAction?: LaunchReadinessAction;
  secondaryActionLabel?: string;
  items: LaunchReadinessItem[];
  productConfig: LaunchProductConfigRow[];
};

export type BuildLaunchReadinessInput = {
  license: LicenseStatus | null;
  runtime: RuntimeState | null;
  runtimeConfig: RuntimeConfig | null;
};

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function nestedRecord(source: unknown, key: string): Record<string, unknown> {
  return asRecord(asRecord(source)[key]);
}

function readString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function readNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function formatTraceFragment(value: string | null | undefined): string {
  const normalized = value?.trim();
  if (!normalized) {
    return "未生成";
  }
  if (normalized.length <= 16) {
    return normalized.toUpperCase();
  }
  return `${normalized.slice(0, 8).toUpperCase()}-${normalized.slice(-6).toUpperCase()}`;
}

function formatTier(tier: string): string {
  if (tier === "temporary") return "临时测试授权";
  if (tier === "pro") return "专业版";
  if (tier === "premium" || tier === "ultimate") return "旗舰版";
  return tier || "未授权";
}

function formatBackend(value: string): string {
  return value === "deepstream_nvinfer" ? "DeepStream / nvinfer" : value || "未配置";
}

function formatOptionalInteger(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? String(Math.trunc(value)) : "—";
}

function formatCaptureProfile(config: Record<string, unknown>, runtime: RuntimeState | null): string {
  const pixelFormat = readString(config.pixel_format).toUpperCase();
  const width = readNumber(config.width);
  const height = readNumber(config.height);
  const fps = readNumber(config.fps);
  if (pixelFormat && width > 0 && height > 0 && fps > 0) {
    return `${pixelFormat} ${width}x${height}@${fps}`;
  }
  const profile = runtime?.capture?.profile;
  if (profile) {
    return `${profile.pixel_format.toUpperCase()} ${profile.width}x${profile.height}@${profile.fps}`;
  }
  return "等待采集配置";
}

function hasConfiguredCaptureProfile(captureConfig: Record<string, unknown>, runtime: RuntimeState | null): boolean {
  return (
    readString(captureConfig.device).trim().length > 0 &&
    readString(captureConfig.pixel_format).trim().length > 0 &&
    readNumber(captureConfig.width) > 0 &&
    readNumber(captureConfig.height) > 0 &&
    readNumber(captureConfig.fps) > 0
  ) || runtime?.capture?.profile !== null && runtime?.capture?.profile !== undefined;
}

function buildLicenseItem(license: LicenseStatus | null): LaunchReadinessItem {
  if (license?.valid) {
    const trace = formatTraceFragment(license.token_id || license.license_id || license.fingerprint);
    return {
      id: "license",
      label: "个人授权",
      state: "ready",
      detail: `${formatTier(license.tier)} 已签收，授权可追溯。`,
      evidence: `追踪码 ${trace}`,
      blocking: true
    };
  }
  return {
    id: "license",
    label: "个人授权",
    state: "blocked",
    detail: "尚未通过授权门禁，不能进入生产工作台。",
    evidence: license?.message || "等待授权",
    blocking: true,
    action: "open-license",
    actionLabel: "处理授权"
  };
}

function buildModelItem(runtime: RuntimeState | null, status: RuntimeMainlineStatus): LaunchReadinessItem {
  const artifact = runtime?.active_model?.artifact;
  const modelName = runtime?.active_model?.project?.name || "未发布模型";
  if (runtime?.active_model && artifact) {
    return {
      id: "model",
      label: "模型部署",
      state: "ready",
      detail: `${modelName} 已发布到运行配置。`,
      evidence: `${artifact.kind} · ${artifact.status}`,
      blocking: true
    };
  }
  const failedBecauseModel = status.readinessCode === "model_load_failed";
  return {
    id: "model",
    label: "模型部署",
    state: failedBecauseModel ? "blocked" : "action",
    detail: failedBecauseModel
      ? "当前模型加载失败，需要重新验证或切换 TensorRT Engine。"
      : "启动前需要发布一个可加载的 TensorRT Engine。",
    evidence: runtime?.model_catalog_error || modelName,
    blocking: true,
    action: "open-model-manager",
    actionLabel: failedBecauseModel ? "修复模型" : "配置模型"
  };
}

function buildCaptureItem(
  runtime: RuntimeState | null,
  runtimeConfig: RuntimeConfig | null,
  status: RuntimeMainlineStatus
): LaunchReadinessItem {
  const captureConfig = nestedRecord(runtimeConfig, "capture");
  const device = readString(captureConfig.device, runtime?.capture?.device ?? "/dev/video0");
  const captureProfile = formatCaptureProfile(captureConfig, runtime);
  const configured = hasConfiguredCaptureProfile(captureConfig, runtime);
  if (status.readinessCode === "no_video" || runtime?.capture?.state === "failed") {
    return {
      id: "capture",
      label: "采集输入",
      state: "blocked",
      detail: "主链未收到有效画面，需要检查采集卡信号、格式或分辨率。",
      evidence: runtime?.capture?.last_error || status.readinessDetail,
      blocking: true,
      action: "open-capture",
      actionLabel: "检查采集"
    };
  }
  if (configured) {
    return {
      id: "capture",
      label: "采集输入",
      state: "ready",
      detail: `${device} 已有可提交的采集规格。`,
      evidence: captureProfile,
      blocking: true
    };
  }
  return {
    id: "capture",
    label: "采集输入",
    state: "action",
    detail: "缺少明确采集规格，启动时只能依赖自动探测。",
    evidence: `${device} · ${captureProfile}`,
    blocking: true,
    action: "open-capture",
    actionLabel: "选择采集"
  };
}

function buildDeepStreamItem(runtime: RuntimeState | null, status: RuntimeMainlineStatus): LaunchReadinessItem {
  const backend = readString(runtime?.inference?.selected, "deepstream_nvinfer");
  if (status.failed && status.readinessCode !== "no_video") {
    return {
      id: "deepstream",
      label: "DeepStream 推理",
      state: "blocked",
      detail: status.readinessDetail,
      evidence: status.failureMessage || formatBackend(backend),
      blocking: true,
      action: status.readinessCode === "model_load_failed" ? "open-model-manager" : "open-capture",
      actionLabel: status.readinessCode === "model_load_failed" ? "修复模型" : "检查运行"
    };
  }
  if (status.running && status.hasInferenceSignal) {
    return {
      id: "deepstream",
      label: "DeepStream 推理",
      state: "ready",
      detail: "nvinfer 输入、元数据或 DetectionBatch 已产生真实计数。",
      evidence: status.progressSummary,
      blocking: true
    };
  }
  return {
    id: "deepstream",
    label: "DeepStream 推理",
    state: runtime?.inference?.configured ? "idle" : "action",
    detail: runtime?.inference?.configured
      ? "运行环境已配置，等待启动后产生推理证据。"
      : "推理后端尚未报告可用配置。",
    evidence: formatBackend(backend),
    blocking: true,
    action: "start-mainline",
    actionLabel: "启动验证"
  };
}

function buildControlItem(runtime: RuntimeState | null, status: RuntimeMainlineStatus): LaunchReadinessItem {
  if (status.running && status.hasRuntimeConsumption) {
    return {
      id: "control",
      label: "控制算法",
      state: "ready",
      detail: "目标选择、跟踪或 Atan 控制已经消费 DetectionBatch。",
      evidence: `consumed=${formatOptionalInteger(status.consumedBatches)} · targeting=${formatOptionalInteger(status.targetingBatches)}`,
      blocking: true
    };
  }
  if (status.running && status.hasInferenceSignal) {
    return {
      id: "control",
      label: "控制算法",
      state: "idle",
      detail: "推理链已产生数据，正在等待运行时消费。",
      evidence: status.progressSummary,
      blocking: true,
      action: "open-control",
      actionLabel: "查看控制"
    };
  }
  return {
    id: "control",
    label: "控制算法",
    state: runtime?.running ? "idle" : "action",
    detail: "启动主链后才会验证目标选择、预测与 Atan 输出。",
    evidence: "等待 DetectionBatch 消费",
    blocking: true,
    action: "start-mainline",
    actionLabel: "启动主链"
  };
}

function buildKmNetItem(runtime: RuntimeState | null, runtimeConfig: RuntimeConfig | null): LaunchReadinessItem {
  const controlConfig = nestedRecord(runtimeConfig, "control");
  const hardwareConfig = nestedRecord(runtimeConfig, "hardware");
  const executor = runtime?.executor;
  const kmnet = executor?.executors?.kmnet;
  const outputEnabled = readBoolean(controlConfig.output_enabled, true);
  const autoConnect = readBoolean(hardwareConfig.auto_connect, true);
  const restartRequired = kmnet?.restart_required === true;
  const host = readString(hardwareConfig.host, "192.168.2.188");
  const port = readNumber(hardwareConfig.port, 8888);
  if (!outputEnabled) {
    return {
      id: "kmnet",
      label: "kmNet 输出",
      state: "idle",
      detail: "物理输出已安全暂停，主链仍可采集、推理和计算控制量。",
      evidence: "output_enabled=false",
      blocking: false,
      action: "open-params",
      actionLabel: "打开输出"
    };
  }
  if (restartRequired) {
    return {
      id: "kmnet",
      label: "kmNet 输出",
      state: "action",
      detail: "kmNet 新配置需要重启后端才会接管输出。",
      evidence: kmnet?.blocked_reason || `${host}:${port}`,
      blocking: false,
      action: "open-kmnet-test",
      actionLabel: "查看设备"
    };
  }
  if (kmnet?.runtime_connected) {
    return {
      id: "kmnet",
      label: "kmNet 输出",
      state: "ready",
      detail: "运行时设备通道已连接，可以接收新鲜控制命令。",
      evidence: `${host}:${port} · accepted=${kmnet.accepted_command_count}`,
      blocking: false
    };
  }
  return {
    id: "kmnet",
    label: "kmNet 输出",
    state: autoConnect ? "action" : "blocked",
    detail: autoConnect
      ? "输出已允许，但运行时设备通道尚未连接。"
      : "输出已允许，但 kmNet 自动连接尚未配置。",
    evidence: kmnet?.last_error || kmnet?.blocked_reason || `${host}:${port}`,
    blocking: false,
    action: "open-kmnet-test",
    actionLabel: "配置 kmNet"
  };
}

function choosePrimaryAction(items: LaunchReadinessItem[], status: RuntimeMainlineStatus): Pick<
  LaunchReadinessSummary,
  "primaryAction" | "primaryActionLabel" | "secondaryAction" | "secondaryActionLabel"
> {
  const blockingAction = items.find((item) => item.blocking && item.action && item.state !== "ready");
  if (blockingAction?.action) {
    const secondaryAction = status.running
      ? undefined
      : blockingAction.action === "start-mainline"
        ? "open-capture"
        : "start-mainline";
    return {
      primaryAction: blockingAction.action,
      primaryActionLabel: blockingAction.actionLabel,
      secondaryAction,
      secondaryActionLabel: secondaryAction === "open-capture" ? "检查采集" : secondaryAction ? "启动主链" : undefined
    };
  }
  if (!status.running) {
    return {
      primaryAction: "start-mainline",
      primaryActionLabel: "启动主链",
      secondaryAction: "open-capture",
      secondaryActionLabel: "检查采集"
    };
  }
  const optionalAction = items.find((item) => !item.blocking && item.action && item.state !== "ready");
  if (optionalAction?.action) {
    return {
      primaryAction: "open-control",
      primaryActionLabel: "查看控制链",
      secondaryAction: optionalAction.action,
      secondaryActionLabel: optionalAction.actionLabel
    };
  }
  return {
    primaryAction: "open-control",
    primaryActionLabel: "查看控制链"
  };
}

function buildProductConfig(
  license: LicenseStatus | null,
  runtime: RuntimeState | null,
  runtimeConfig: RuntimeConfig | null,
  status: RuntimeMainlineStatus
): LaunchProductConfigRow[] {
  const captureConfig = nestedRecord(runtimeConfig, "capture");
  const inferenceConfig = nestedRecord(runtimeConfig, "inference");
  const controlConfig = nestedRecord(runtimeConfig, "control");
  const hardwareConfig = nestedRecord(runtimeConfig, "hardware");
  const triggerMode = readString(controlConfig.trigger_mode, "always");
  const outputEnabled = readBoolean(controlConfig.output_enabled, true);
  const host = readString(hardwareConfig.host, "192.168.2.188");
  const port = readNumber(hardwareConfig.port, 8888);
  const uuid = readString(hardwareConfig.uuid, "12345678");

  return [
    {
      label: "授权",
      value: license?.valid ? formatTier(license.tier) : "等待授权",
      detail: formatTraceFragment(license?.token_id || license?.license_id || license?.fingerprint)
    },
    {
      label: "模型",
      value: runtime?.active_model?.project?.name || "未发布",
      detail: runtime?.active_model?.artifact
        ? `${runtime.active_model.artifact.kind} · ${runtime.active_model.artifact.status}`
        : "需要发布 Engine"
    },
    {
      label: "采集",
      value: readString(captureConfig.device, runtime?.capture?.device || "/dev/video0"),
      detail: formatCaptureProfile(captureConfig, runtime)
    },
    {
      label: "推理",
      value: formatBackend(readString(inferenceConfig.backend, runtime?.inference?.selected || "deepstream_nvinfer")),
      detail: status.progressSummary
    },
    {
      label: "控制",
      value: triggerMode === "hardware" ? "硬件触发" : "检测即控制",
      detail: outputEnabled ? "允许输出" : "安全暂停"
    },
    {
      label: "kmNet",
      value: `${host}:${port}`,
      detail: `UUID ${uuid || "未填写"}`
    }
  ];
}

export function buildLaunchReadiness({
  license,
  runtime,
  runtimeConfig
}: BuildLaunchReadinessInput): LaunchReadinessSummary {
  const status = getRuntimeMainlineStatus(runtime);
  const items = [
    buildLicenseItem(license),
    buildModelItem(runtime, status),
    buildCaptureItem(runtime, runtimeConfig, status),
    buildDeepStreamItem(runtime, status),
    buildControlItem(runtime, status),
    buildKmNetItem(runtime, runtimeConfig)
  ];
  const blockingItems = items.filter((item) => item.blocking);
  const readyCount = blockingItems.filter((item) => item.state === "ready").length;
  const blockingCount = blockingItems.length;
  const score = blockingCount === 0 ? 100 : Math.round((readyCount / blockingCount) * 100);
  const hasBlocked = blockingItems.some((item) => item.state === "blocked");
  const hasAction = blockingItems.some((item) => item.state === "action");
  const state: LaunchReadinessState = hasBlocked
    ? "blocked"
    : readyCount === blockingCount
      ? "ready"
      : hasAction
        ? "action"
        : "idle";
  const title = state === "ready"
    ? status.running ? "主链已经具备生产运行证据" : "主链准备完成，等待启动"
    : state === "blocked"
      ? "主链启动前有阻断项"
      : "按顺序补齐主链准备项";
  const detail = state === "ready"
    ? status.running
      ? status.readinessDetail
      : "模型、采集和运行配置已就绪；启动后继续核对 DeepStream 与控制消费计数。"
    : "先处理第一条阻断或待配置项，再继续启动；输出设备独立于采集和推理验证。";

  return {
    state,
    title,
    detail,
    readyCount,
    blockingCount,
    score,
    ...choosePrimaryAction(items, status),
    items,
    productConfig: buildProductConfig(license, runtime, runtimeConfig, status)
  };
}
