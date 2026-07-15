import { ChangeEvent, Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureState,
  CaptureSelectPayload,
  connectKmNet,
  diagnosticCircleKmNet,
  diagnosticMoveKmNet,
  disconnectKmNet,
  getRuntimeState,
  HealthResponse,
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelProject,
  ModelVersion,
  RuntimeConfig,
  RuntimeConfigValue,
  RuntimeState,
  getCaptureCapabilities,
  getModelArtifacts,
  getModelCatalog,
  getModelVersions,
  publishModel,
  registerCatalogModel,
  selectCaptureProfile,
  startRuntimePipeline,
  stopCapture,
  stopRuntimePipeline,
  streamUrl,
  updateRuntimeConfig,
  updateRuntimeConfigField
} from "../../api";
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";
import { NovaIcon, StatusBadge, ThemeToggle, type NovaIconName } from "../../components/visual";
import { ModelCatalogTree } from "../models/ModelCatalogTree";

type ConsolePage = "capture" | "infer" | "control" | "params" | "control-test" | "stats" | "latency";

const DEFAULT_CONSOLE_PAGE: ConsolePage = "capture";
const CONSOLE_PAGES = new Set<ConsolePage>(["capture", "infer", "control", "params", "control-test", "stats", "latency"]);

const CONTROL_ALGORITHM_OPTIONS = [
  {
    id: "universal_saturated",
    label: "通用控制",
    description: "无需精确游戏参数，适合快速适配。"
  },
  {
    id: "calibrated_angular",
    label: "精确角度控制",
    description: "依赖 FOV 和 counts_per_360 标定。"
  },
  {
    id: "dual_phase_atan_robust_predictive_v2",
    label: "稳健预测控制",
    description: "在精确标定上增加同目标短窗受限预测。"
  }
] as const;
const DEFAULT_CONTROL_ALGORITHM = "dual_phase_atan_robust_predictive_v2";

type StudioConsoleViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  runtimeConfig: RuntimeConfig | null;
  projects: ModelProject[];
  errors: Partial<Record<string, string>>;
  lastUpdated: Date | null;
  realtimeStatus: "connecting" | "connected" | "stale" | "disconnected";
  onRefresh: () => Promise<void>;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type LaunchStatus = "idle" | "running" | "success" | "failed" | "cancelled";

type LaunchStage = {
  label: string;
  title: string;
  caption: string;
};

const navItems: { id: ConsolePage; index: string; label: string; icon: NovaIconName }[] = [
  { id: "capture", index: "01", label: "采集", icon: "capture" },
  { id: "infer", index: "02", label: "模型推理", icon: "inference" },
  { id: "control", index: "03", label: "控制", icon: "control" },
  { id: "params", index: "04", label: "参数设置", icon: "settings" },
  { id: "control-test", index: "05", label: "控制测试", icon: "kmbox" },
  { id: "stats", index: "06", label: "统计", icon: "performance" },
  { id: "latency", index: "07", label: "采集延迟", icon: "latency" }
];

const RUNTIME_MAINLINE_BACKENDS = new Set(["deepstream_nvinfer"]);

// Every production backend must prove both DetectionBatch publication and
// runtime consumption before the launch dialog declares the control path ready.
const MAINLINE_LAUNCH_STAGES_CUSTOM_TENSORRT: LaunchStage[] = [
  {
    label: "阶段 1 / 5",
    title: "检查运行环境",
    caption: "确认 Studio 已连接到 Jetson 运行服务。"
  },
  {
    label: "阶段 2 / 5",
    title: "应用采集配置",
    caption: "按当前设备、格式、分辨率与帧率选择采集配置。"
  },
  {
    label: "阶段 3 / 5",
    title: "启动主链运行管线",
    caption: "请求后端启动采集、ROI、推理、DetectionBatch 与控制主链。"
  },
  {
    label: "阶段 4 / 5",
    title: "激活跟踪与控制",
    caption: "runtime 开始消费 DetectionBatch 后，跟踪、预测与控制模块随即激活。"
  },
  {
    label: "阶段 5 / 5",
    title: "确认设备执行器",
    caption: "刷新执行器状态，确认输出链路由后端持有。"
  }
];

function pageFromUrl(): ConsolePage {
  const raw = new URLSearchParams(window.location.search).get("page");
  return CONSOLE_PAGES.has(raw as ConsolePage) ? (raw as ConsolePage) : DEFAULT_CONSOLE_PAGE;
}

function writePageToUrl(page: ConsolePage, mode: "push" | "replace" = "push") {
  const url = new URL(window.location.href);
  url.searchParams.set("page", page);
  if (mode === "replace") {
    window.history.replaceState({ page }, "", url);
  } else {
    window.history.pushState({ page }, "", url);
  }
}

const ROI_SIZE_CHOICES = [256, 320, 480, 640];
type CaptureBackendMode = "deepstream_nvinfer";
const CAPTURE_BACKEND_CHOICES: {
  value: CaptureBackendMode;
  label: string;
  memory: "nvmm";
  inferenceBackend: "deepstream_nvinfer";
  preprocessBackend: "cuda";
}[] = [
  {
    value: "deepstream_nvinfer",
    label: "DeepStream nvinfer",
    memory: "nvmm",
    inferenceBackend: "deepstream_nvinfer",
    preprocessBackend: "cuda"
  }
];
const KMNET_RECOMMENDED = {
  host: "192.168.2.188",
  port: 8888,
  uuid: "12345678",
  monitor_port: 5001
};

const ARTIFACT_KIND_RANK: Record<string, number> = {
  engine: 0,
  onnx: 1
};

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function nestedRecord(value: unknown, key: string): Record<string, unknown> {
  return asRecord(asRecord(value)[key]);
}

function recordList(value: unknown): Record<string, string[]> {
  const raw = asRecord(value);
  return Object.fromEntries(
    Object.entries(raw)
      .filter(([, items]) => Array.isArray(items))
      .map(([key, items]) => [key, (items as unknown[]).map((item) => String(item))])
  );
}

function profileNumberRecords(value: unknown): Record<string, Record<string, number>> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).map(([profileName, rawValues]) => [
      profileName,
      Object.fromEntries(
        Object.entries(asRecord(rawValues)).flatMap(([classId, ratio]) =>
          typeof ratio === "number" && Number.isFinite(ratio)
            ? [[classId, clampNumber(ratio, 0, 1)]]
            : []
        )
      )
    ])
  );
}

function classDisplayName(value: string, classId: number): string {
  const normalized = value.trim().replace(new RegExp(`^${classId}\\s*[-:：]\\s*`), "");
  return normalized || `未知类别（cls ${classId}）`;
}

function parseClassPriority(value: string): number[] {
  const seen = new Set<number>();
  return value.split(",").flatMap((part) => {
    const classId = Number(part.trim());
    if (!Number.isInteger(classId) || classId < 0 || classId > 255 || seen.has(classId)) {
      return [];
    }
    seen.add(classId);
    return [classId];
  });
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0)
    : [];
}

function readNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function readString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function triggerModeLabel(value: string): string {
  if (value === "always") {
    return "检测目标自动控制";
  }
  return "kmNet 硬件触发";
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, value));
}

const NO_SAMPLE = "—";
const NOT_INSTRUMENTED = "未接入";
const UNAVAILABLE = "不可用";

function formatNumber(value: unknown, digits = 1): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? number.toFixed(digits) : NO_SAMPLE;
}

function formatPercent(value: unknown, digits = 1): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : NO_SAMPLE;
}

function formatShape(value: unknown): string {
  if (Array.isArray(value) && value.length > 0) {
    return value.map((item) => String(item)).join("x");
  }
  const text = readString(value, "").trim();
  return text || NO_SAMPLE;
}

function formatOptionalNumber(value: unknown, digits = 1, unit = ""): string {
  const number = readNullableNumber(value);
  if (number === null) {
    return NO_SAMPLE;
  }
  return `${number.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function formatOptionalInteger(value: unknown): string {
  const number = readNullableNumber(value);
  return number === null || number < 0 ? NO_SAMPLE : Math.trunc(number).toString();
}

function formatPoint(x: unknown, y: unknown, digits = 1, unit = ""): string {
  const xNumber = readNullableNumber(x);
  const yNumber = readNullableNumber(y);
  if (xNumber === null || yNumber === null) {
    return NO_SAMPLE;
  }
  const suffix = unit ? ` ${unit}` : "";
  return `${xNumber.toFixed(digits)}, ${yNumber.toFixed(digits)}${suffix}`;
}

function formatDurationFromNs(start: unknown, end: unknown): string {
  const startNs = readNullableNumber(start);
  const endNs = readNullableNumber(end);
  if (startNs === null || endNs === null || startNs <= 0 || endNs < startNs) {
    return NO_SAMPLE;
  }
  return `${((endNs - startNs) / 1e6).toFixed(3)} ms`;
}

function shortTimestampSource(value: string): string {
  if (value === "gst_clock_base_time_pts") {
    return "GstClock";
  }
  if (value === "first_probe_offset_pts") {
    return "Probe校准";
  }
  if (value === "observed_probe_time_invalid_pts") {
    return "Probe时间";
  }
  if (!value || value === "uninitialized") {
    return "-";
  }
  return value.length > 12 ? `${value.slice(0, 12)}...` : value;
}

function formatDate(value: Date | null): string {
  return value ? value.toLocaleTimeString("zh-CN", { hour12: false }) : "--:--:--";
}

function compactDriverSource(value: unknown): string {
  const raw = readString(value, "");
  if (!raw) {
    return "-";
  }
  const normalized = raw.replace(/\\/g, "/");
  if (normalized.includes("/vendor/kmnet/")) {
    return `vendor/${normalized.split("/vendor/kmnet/")[1]}`;
  }
  return normalized.split("/").slice(-2).join("/") || normalized;
}

function groupCapabilities(caps: CaptureCapability[]): CapabilityChoice[] {
  return caps
    .flatMap((cap) =>
      cap.fps_list.map((fps) => ({
        pixel_format: cap.pixel_format.toUpperCase(),
        width: cap.width,
        height: cap.height,
        fps
      }))
    )
    .sort((left, right) => {
      const formatRank = (value: string) => ["MJPG", "NV12", "YUYV", "BGR3"].indexOf(value);
      const leftRank = formatRank(left.pixel_format);
      const rightRank = formatRank(right.pixel_format);
      return (
        (leftRank < 0 ? 99 : leftRank) - (rightRank < 0 ? 99 : rightRank) ||
        right.fps - left.fps ||
        right.width * right.height - left.width * left.height
      );
    });
}

function choiceId(choice: CapabilityChoice): string {
  return `${choice.pixel_format}:${choice.width}x${choice.height}@${choice.fps}`;
}

function choiceLabel(choice: CapabilityChoice): string {
  return `${choice.pixel_format} / ${choice.width}x${choice.height} / ${choice.fps} FPS`;
}

function choiceMatchesConfig(choice: CapabilityChoice, config: Record<string, unknown>): boolean {
  return (
    readString(config.pixel_format, "").toUpperCase() === choice.pixel_format.toUpperCase() &&
    readNumber(config.width, 0) === choice.width &&
    readNumber(config.height, 0) === choice.height &&
    readNumber(config.fps, 0) === choice.fps
  );
}

function nearestRoiSize(value: number): number {
  return ROI_SIZE_CHOICES.reduce((best, current) =>
    Math.abs(current - value) < Math.abs(best - value) ? current : best
  );
}

function cloneRuntimeConfig(config: RuntimeConfig | null): RuntimeConfig | null {
  if (!config) {
    return null;
  }
  const next = structuredClone(config) as RuntimeConfig;
  delete next.version;
  return next;
}

function normalizeRuntimeConfig(config: RuntimeConfig): RuntimeConfig {
  const next = structuredClone(config) as RuntimeConfig;
  delete next.version;
  delete next.roi_size;
  return next;
}

export function StudioConsoleView({
  health,
  runtime,
  runtimeConfig,
  projects,
  errors,
  lastUpdated,
  realtimeStatus,
  onRefresh
}: StudioConsoleViewProps) {
  const [activePage, setActivePage] = useState<ConsolePage>(() => pageFromUrl());
  const [device, setDevice] = useState(
    readString(nestedRecord(runtimeConfig, "capture").device, runtime?.capture?.device ?? "/dev/video0")
  );
  const [caps, setCaps] = useState<CaptureCapabilitiesResponse | null>(null);
  const [selectedChoiceId, setSelectedChoiceId] = useState("");
  const [selectedModelProjectId, setSelectedModelProjectId] = useState<number | "">("");
  const [selectedModelVersionId, setSelectedModelVersionId] = useState<number | "">("");
  const [selectedModelArtifactId, setSelectedModelArtifactId] = useState<number | "">("");
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [modelArtifacts, setModelArtifacts] = useState<ModelArtifact[]>([]);
  const [modelCatalogRefreshKey, setModelCatalogRefreshKey] = useState(0);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogDirectory | null>(null);
  const [modelCatalogModelCount, setModelCatalogModelCount] = useState(0);
  const [modelCatalogDirectoryCount, setModelCatalogDirectoryCount] = useState(0);
  const [modelCatalogLoading, setModelCatalogLoading] = useState(false);
  const [expandedModelDirectories, setExpandedModelDirectories] = useState<Set<string>>(
    () => new Set([""])
  );
  const [selectedModelCatalogPath, setSelectedModelCatalogPath] = useState<string>();
  const [kmnetTestDx, setKmnetTestDx] = useState(10);
  const [kmnetTestDy, setKmnetTestDy] = useState(0);
  const [kmnetTestMs, setKmnetTestMs] = useState(300);
  const [kmnetMoveKind, setKmnetMoveKind] = useState("raw");
  const [kmnetBezierX1, setKmnetBezierX1] = useState(-50);
  const [kmnetBezierY1, setKmnetBezierY1] = useState(-60);
  const [kmnetBezierX2, setKmnetBezierX2] = useState(70);
  const [kmnetBezierY2, setKmnetBezierY2] = useState(80);
  const [kmnetTestMessage, setKmnetTestMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [modelSwitchMessage, setModelSwitchMessage] = useState("");
  const [modelCatalogMessage, setModelCatalogMessage] = useState("");
  const [launchDialogOpen, setLaunchDialogOpen] = useState(false);
  const [launchStatus, setLaunchStatus] = useState<LaunchStatus>("idle");
  const [launchStageIndex, setLaunchStageIndex] = useState(0);
  const [launchCompletedStages, setLaunchCompletedStages] = useState(0);
  const [launchError, setLaunchError] = useState("");
  const [launchProgressDetail, setLaunchProgressDetail] = useState("");
  const [launchToastVisible, setLaunchToastVisible] = useState(false);
  const [mainlineLaunchAccepted, setMainlineLaunchAccepted] = useState(false);
  const [mainlineLaunchMessage, setMainlineLaunchMessage] = useState("");
  const [configDraft, setConfigDraft] = useState<RuntimeConfig | null>(() => cloneRuntimeConfig(runtimeConfig));
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const configDraftRef = useRef<RuntimeConfig | null>(cloneRuntimeConfig(runtimeConfig));
  const loadedModelProjectIdRef = useRef<number | "">("");
  const loadedModelVersionIdRef = useRef<number | "">("");
  const requestedModelSelectionRef = useRef<{
    projectId: number;
    versionId: number;
    artifactId: number;
  } | null>(null);
  const preferLatestModelVersionRef = useRef(false);
  const pendingConfigWritesRef = useRef(0);
  const configWriteSeqRef = useRef(0);
  const launchCancelledRef = useRef(false);
  const launchTimerRef = useRef<number | null>(null);
  const launchTimerResolveRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    writePageToUrl(activePage, "replace");
    const onPopState = () => setActivePage(pageFromUrl());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigatePage = useCallback((page: ConsolePage) => {
    setActivePage(page);
    writePageToUrl(page);
  }, []);

  useEffect(() => {
    if (!launchDialogOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [launchDialogOpen]);

  useEffect(() => {
    if (!launchDialogOpen || launchStatus === "running") {
      return undefined;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setLaunchDialogOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [launchDialogOpen, launchStatus]);

  useEffect(() => () => {
    if (launchTimerRef.current !== null) {
      window.clearTimeout(launchTimerRef.current);
      launchTimerRef.current = null;
    }
    if (launchTimerResolveRef.current) {
      launchTimerResolveRef.current();
      launchTimerResolveRef.current = null;
    }
  }, []);

  const capture = runtime?.capture;
  const statistics = runtime?.statistics ?? capture?.statistics;
  const captureStatistics = asRecord(statistics);
  const config = configDraft ?? runtimeConfig;
  const captureConfig = nestedRecord(config, "capture");
  const configuredCaptureDevice = readString(captureConfig.device, "");
  const configuredCapturePixelFormat = readString(captureConfig.pixel_format, "");
  const configuredCaptureWidth = readNumber(captureConfig.width, 0);
  const configuredCaptureHeight = readNumber(captureConfig.height, 0);
  const configuredCaptureFps = readNumber(captureConfig.fps, 0);
  const roiConfig = nestedRecord(config, "roi");
  const limitsConfig = nestedRecord(config, "limits");
  const inferenceConfig = nestedRecord(config, "inference");
  const preprocessConfig = nestedRecord(config, "preprocess");
  const controlConfig = nestedRecord(config, "control");
  const hardwareConfig = nestedRecord(config, "hardware");
  const consumersConfig = nestedRecord(config, "consumers");
  const vision = asRecord(runtime?.vision);
  const execution = asRecord(vision.execution);
  const executionIntent = asRecord(execution.intent);
  const executionMeta = asRecord(execution.metadata);
  const inferenceTrace = asRecord(vision.inference);
  const runtimeInference = asRecord(runtime?.inference);
  const pipeline = asRecord(runtime?.pipeline);
  const deepstreamStatus = asRecord(pipeline.deepstream);
  const deepstreamMailbox = asRecord(deepstreamStatus.detection_batch_mailbox);
  const deepstreamNvmmOutput = asRecord(deepstreamStatus.nvmm_output);
  const captureBackendMode: CaptureBackendMode = "deepstream_nvinfer";
  const captureBackendChoice = CAPTURE_BACKEND_CHOICES[0];
  const configuredCaptureMemory = readString(captureConfig.memory, captureBackendChoice.memory);
  const configuredPreprocessBackend = readString(preprocessConfig.backend, captureBackendChoice.preprocessBackend);
  const configuredInferenceBackend = readString(inferenceConfig.backend, captureBackendChoice.inferenceBackend);
  const selectedRuntimeBackend = readString(runtimeInference.selected, configuredInferenceBackend);
  const deepstreamNvinferSelected = selectedRuntimeBackend === "deepstream_nvinfer";
  const mainlineRuntimeSelected = RUNTIME_MAINLINE_BACKENDS.has(selectedRuntimeBackend);
  const runtimeMainlineSelected = mainlineRuntimeSelected;
  const runtimeMainlineStatus = getRuntimeMainlineStatus(runtime);
  const mainlineTerminalError = runtimeMainlineStatus.terminalError;
  const runtimeInferenceConfigured = runtimeInference.configured === true;
  const runtimeInferenceReason = readString(runtimeInference.reason, "");
  const runtimeInferenceDetail = readString(runtimeInference.detail, "");
  const runtimeMainlineRunning = runtimeMainlineStatus.running;
  const mainlineLaunchPending =
    mainlineRuntimeSelected &&
    mainlineLaunchAccepted &&
    !runtimeMainlineRunning &&
    !runtimeMainlineStatus.failed &&
    runtime?.fatal_error === null;
  const captureMainRunning = runtimeMainlineSelected
    ? (!runtimeMainlineStatus.failed && runtimeMainlineRunning) || mainlineLaunchPending
    : capture?.available === true;
  const captureMainConfigured = capture?.available === true || runtimeInferenceConfigured;
  const captureStatusText = runtimeMainlineSelected
    ? runtimeMainlineStatus.failed
      ? "主链故障"
      : runtimeMainlineRunning
        ? "采集+TensorRT+控制运行中"
      : mainlineLaunchPending
        ? "启动确认中"
        : runtimeInferenceConfigured
          ? "主链待启动"
          : "未配置"
    : captureMainRunning
      ? "运行中"
      : captureMainConfigured
        ? "已配置"
        : "已停止";
  const inferenceStatusText = runtimeMainlineSelected
    ? runtimeMainlineStatus.failed
      ? "管线故障"
      : runtimeMainlineRunning
        ? runtimeMainlineStatus.hasRuntimeConsumption
          ? "runtime 已消费"
          : runtimeMainlineStatus.hasInferenceSignal
            ? "DetectionBatch 已产出"
            : "等待 DetectionBatch"
      : mainlineLaunchPending
        ? "等待后端反馈"
        : runtimeInferenceConfigured
          ? "待启动"
          : "未配置"
    : runtime?.running
      ? "运行中"
      : "已停止";
  const runtimeModelOutput = asRecord(runtimeInference.model_output);
  const runtimePostprocess = asRecord(runtimeInference.postprocess);
  const runtimeModelOutputClassNames = stringArray(runtimeModelOutput.class_names);
  const executorStatus = asRecord(runtime?.executor);
  const schedulerStatus = asRecord(executorStatus.scheduler);
  const executors = asRecord(executorStatus.executors);
  const kmnetStatus = asRecord(executors.kmnet);
  const selectedProfile = capture?.profile;
  const runningPixelFormat = selectedProfile?.pixel_format ?? "";
  const runningWidth = selectedProfile?.width ?? 0;
  const runningHeight = selectedProfile?.height ?? 0;
  const runningFps = selectedProfile?.fps ?? 0;
  const configuredCaptureProfile =
    configuredCapturePixelFormat && configuredCaptureWidth > 0 && configuredCaptureHeight > 0 && configuredCaptureFps > 0
      ? {
          pixel_format: configuredCapturePixelFormat.toUpperCase(),
          width: configuredCaptureWidth,
          height: configuredCaptureHeight,
          fps: configuredCaptureFps
        }
      : null;
  const configuredChoiceId = configuredCaptureProfile
    ? `${configuredCaptureProfile.pixel_format}:${configuredCaptureProfile.width}x${configuredCaptureProfile.height}@${configuredCaptureProfile.fps}`
    : "";
  const runningChoiceId = selectedProfile
    ? `${selectedProfile.pixel_format.toUpperCase()}:${selectedProfile.width}x${selectedProfile.height}@${selectedProfile.fps}`
    : "";
  const displayCaptureProfile = configuredCaptureProfile ?? selectedProfile ?? null;
  const choices = useMemo(() => groupCapabilities(caps?.capabilities ?? []), [caps]);
  const selectedChoice =
    choices.find((choice) => choiceId(choice) === selectedChoiceId) ??
    choices.find((choice) => choiceId(choice) === configuredChoiceId) ??
    choices.find((choice) => choiceId(choice) === runningChoiceId) ??
    choices[0];
  const roiSize = readNumber(roiConfig.size, 640);
  const previewFps = readNumber(limitsConfig.stream_fps, 30);
  const sourceWidth = selectedProfile?.width ?? readNumber(inferenceTrace.source_width, 0);
  const sourceHeight = selectedProfile?.height ?? readNumber(inferenceTrace.source_height, 0);
  const roiX = sourceWidth > 0 ? Math.max(0, Math.floor((sourceWidth - roiSize) / 2)) : 0;
  const roiY = sourceHeight > 0 ? Math.max(0, Math.floor((sourceHeight - roiSize) / 2)) : 0;
  const confidence = readNumber(inferenceConfig.confidence_threshold, 0.25);
  const nms = readNumber(inferenceConfig.nms_threshold, 0.45);
  const detectionProfiles = recordList(inferenceConfig.detection_class_profiles);
  const activeDetectionProfile = readString(inferenceConfig.detection_class_profile, "default");
  const activeDetectionClass = readString(inferenceConfig.detection_class_filter, "all");
  const detectionProfileNames = Object.keys(detectionProfiles);
  const detectionClasses = detectionProfiles[activeDetectionProfile] ?? detectionProfiles.default ?? [];
  const detectionClassPriority = readString(inferenceConfig.detection_class_priority, "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
  const classPriorityIds = parseClassPriority(detectionClassPriority);
  useEffect(() => {
    if (!runtimeMainlineSelected) {
      setMainlineLaunchAccepted(false);
      setMainlineLaunchMessage("");
      return;
    }
    if (runtimeMainlineRunning) {
      setMainlineLaunchAccepted(false);
      setMainlineLaunchMessage("");
      return;
    }
    if (
      mainlineLaunchAccepted &&
      (runtimeMainlineStatus.failed || runtime?.fatal_error !== null)
    ) {
      setMainlineLaunchAccepted(false);
      setMainlineLaunchMessage("");
      setLocalError(`主链启动未确认：${runtimeMainlineStatus.failureMessage || runtimeInferenceDetail || runtimeInferenceReason || "后端运行态未进入运行状态。"}`);
    }
  }, [
    runtimeMainlineSelected,
    mainlineTerminalError,
    mainlineLaunchAccepted,
    runtimeMainlineStatus.failed,
    runtimeMainlineStatus.failureMessage,
    runtime?.fatal_error,
    runtimeInferenceDetail,
    runtimeInferenceReason,
    runtimeMainlineRunning
  ]);
  const controlMode = readString(controlConfig.active_algorithm ?? controlConfig.mode, DEFAULT_CONTROL_ALGORITHM);
  const algorithmConfigs = nestedRecord(controlConfig, "algorithms");
  const algorithmConfig = (algorithmId: string): Record<string, unknown> => {
    const namespaced = nestedRecord(algorithmConfigs, algorithmId);
    return Object.keys(namespaced).length > 0
      ? namespaced
      : nestedRecord(controlConfig, algorithmId);
  };
  const aimConfig = nestedRecord(controlConfig, "aim");
  const calibratedAngularConfig = algorithmConfig("calibrated_angular");
  const universalSaturatedConfig = algorithmConfig("universal_saturated");
  const dualPhaseConfig = algorithmConfig("dual_phase_atan_robust_predictive_v2");
  const dualPhaseProjectionConfig = nestedRecord(dualPhaseConfig, "projection");
  const dualPhaseModeConfig = nestedRecord(dualPhaseConfig, "mode");
  const dualPhaseAtanConfig = nestedRecord(dualPhaseConfig, "atan");
  const dualPhaseFarConfig = nestedRecord(dualPhaseAtanConfig, "far");
  const dualPhaseNearConfig = nestedRecord(dualPhaseAtanConfig, "near");
  const dualPhaseVelocityConfig = nestedRecord(dualPhaseConfig, "velocity");
  const dualPhasePredictionConfig = nestedRecord(dualPhaseConfig, "prediction");
  const dualPhasePredictionFarConfig = nestedRecord(dualPhasePredictionConfig, "far");
  const dualPhasePredictionNearConfig = nestedRecord(dualPhasePredictionConfig, "near");
  const sharedControlConfig = nestedRecord(controlConfig, "shared");
  const aimYRatio = readNumber(aimConfig.y_ratio, 0.22);
  const classAimRatioProfiles = profileNumberRecords(aimConfig.class_y_ratios);
  const activeClassAimRatios = classAimRatioProfiles[activeDetectionProfile] ?? {};
  const targetFovRadiusPx = readNumber(controlConfig.target_fov_radius_px, 180);
  const candidateRatioMaxAspect = readNumber(controlConfig.candidate_ratio_max_aspect, 6);
  const candidateQualityConfidenceWeight = readNumber(controlConfig.candidate_quality_confidence_weight, 0.7);
  const candidateQualityAreaWeight = readNumber(controlConfig.candidate_quality_area_weight, 0.3);
  const candidateSelectionClassWeight = readNumber(controlConfig.candidate_selection_class_weight, 0.40);
  const candidateSelectionQualityWeight = readNumber(controlConfig.candidate_selection_quality_weight, 0.05);
  const candidateSelectionDistanceWeight = readNumber(controlConfig.candidate_selection_distance_weight, 0.55);
  const trackerMaxMatchDistance = readNumber(controlConfig.tracker_max_match_distance, 1.5);
  const trackerPositionCostWeight = readNumber(controlConfig.tracker_position_cost_weight, 0.75);
  const trackerIouCostWeight = readNumber(controlConfig.tracker_iou_cost_weight, 0.25);
  const trackerMaxMissedFrames = readNumber(controlConfig.tracker_max_missed_frames, 2);
  const targetSwitchPreferenceAdvantage = readNumber(controlConfig.target_switch_min_preference_advantage, 0.08);
  const targetSwitchContinuityScore = readNumber(controlConfig.target_switch_min_continuity_score, 0.7);
  const targetSwitchDelayMs = readNumber(controlConfig.target_switch_delay_ms, 50);
  const kalmanAccelerationNoise = readNumber(controlConfig.kalman_acceleration_noise, 1200);
  const kalmanMeasurementNoiseX = readNumber(controlConfig.kalman_measurement_noise_x, 16);
  const kalmanMeasurementNoiseY = readNumber(controlConfig.kalman_measurement_noise_y, 16);
  const moveKind = kmnetMoveKind;
  const moveMs = kmnetTestMs;
  const calibratedFovX = readNumber(calibratedAngularConfig.fov_x_deg, 105);
  const calibratedCountsPer360X = readNumber(calibratedAngularConfig.counts_per_360_x, 9980);
  const calibratedCountsPer360Y = readNumber(calibratedAngularConfig.counts_per_360_y, 9980);
  const calibratedKpX = readNumber(calibratedAngularConfig.kp_x, 1);
  const calibratedKpY = readNumber(calibratedAngularConfig.kp_y, 1);
  const calibratedKdX = readNumber(calibratedAngularConfig.kd_x, 0);
  const calibratedKdY = readNumber(calibratedAngularConfig.kd_y, 0);
  const calibratedDEmaAlpha = readNumber(calibratedAngularConfig.d_ema_alpha, 0.3);
  const calibratedMaxAngleX = readNumber(calibratedAngularConfig.max_angle_step_x_deg, 2);
  const calibratedMaxAngleY = readNumber(calibratedAngularConfig.max_angle_step_y_deg, 1.5);
  const universalResponseScaleX = readNumber(universalSaturatedConfig.response_scale_x_px, 80);
  const universalResponseScaleY = readNumber(universalSaturatedConfig.response_scale_y_px, 60);
  const universalMaxStepX = readNumber(universalSaturatedConfig.max_step_x_counts, 50);
  const universalMaxStepY = readNumber(universalSaturatedConfig.max_step_y_counts, 40);
  const dualPhaseFovX = readNumber(dualPhaseProjectionConfig.fov_x_deg, 105);
  const dualPhaseCountsPer360 = readNumber(dualPhaseProjectionConfig.counts_per_360, 9980);
  const dualPhaseNearThreshold = readNumber(dualPhaseModeConfig.near_threshold_px, 12);
  const dualPhaseFarKp = readNumber(dualPhaseFarConfig.kp, 0.45);
  const dualPhaseNearKp = readNumber(dualPhaseNearConfig.kp, 0.22);
  const dualPhaseAtanScale = readNumber(dualPhaseAtanConfig.scale_counts, 256);
  const dualPhaseFarMaxCounts = readNumber(dualPhaseFarConfig.max_counts_per_update, 127);
  const dualPhaseNearMaxCounts = readNumber(dualPhaseNearConfig.max_counts_per_update, 72);
  const dualPhaseLeadFrames = readNumber(dualPhasePredictionConfig.lead_frames, 1.0);
  const dualPhaseVelocitySmoothingFrames = readNumber(dualPhaseVelocityConfig.smoothing_frames, 3.0);
  const dualPhaseHistoryResetGapMs = readNumber(dualPhaseVelocityConfig.history_reset_gap_ms, 80.0);
  const dualPhaseFarPredictionCap = readNumber(dualPhasePredictionFarConfig.absolute_cap_px, 10.0);
  const dualPhaseNearPredictionCap = readNumber(dualPhasePredictionNearConfig.absolute_cap_px, 3.0);
  const dualPhaseInvertY = readBoolean(dualPhaseProjectionConfig.invert_y, false);
  const sharedDeadzoneX = readNumber(sharedControlConfig.deadzone_x_px, 4);
  const sharedDeadzoneY = readNumber(sharedControlConfig.deadzone_y_px, 4);
  const sharedMaxSlewX = readNumber(sharedControlConfig.max_count_slew_x, 10);
  const sharedMaxSlewY = readNumber(sharedControlConfig.max_count_slew_y, 8);
  const sharedInvertY = readBoolean(sharedControlConfig.invert_y, false);
  const triggerActivationDelayMs = readNumber(sharedControlConfig.trigger_activation_delay_ms, 0);
  const recoilEnabled = readBoolean(sharedControlConfig.recoil_enabled, false);
  const recoilStartDelayMs = readNumber(sharedControlConfig.recoil_start_delay_ms, 0);
  const recoilYRate = readNumber(sharedControlConfig.recoil_y_rate_counts_s, 0);
  const recoilRampUpMs = readNumber(sharedControlConfig.recoil_ramp_up_ms, 120);
  const recoilMaxCounts = readNumber(sharedControlConfig.recoil_max_counts_per_observation, 8);
  const triggerMode = readString(controlConfig.trigger_mode, "always");
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);
  const kmnetAutoConnect = readBoolean(hardwareConfig.auto_connect, true);
  const schedulerEnabled = readBoolean(controlConfig.scheduler_enabled, true);
  const schedulerStepCountsX = readNumber(controlConfig.scheduler_step_counts_x, 8);
  const schedulerStepCountsY = readNumber(controlConfig.scheduler_step_counts_y, 8);
  const schedulerIntervalMs = readNumber(controlConfig.scheduler_interval_ms, 4);
  const dualPhaseActive = controlMode === "dual_phase_atan_robust_predictive_v2";
  const activeControlAlgorithm = CONTROL_ALGORITHM_OPTIONS.find((item) => item.id === controlMode)
    ?? CONTROL_ALGORITHM_OPTIONS[2];
  const controlModeLabel = activeControlAlgorithm.label;

  useEffect(() => {
    if (!runtimeConfig || pendingConfigWritesRef.current > 0) {
      return;
    }
    const next = normalizeRuntimeConfig(runtimeConfig);
    configDraftRef.current = next;
    setConfigDraft(next);
  }, [runtimeConfig]);
  const kmnetConnected = kmnetStatus.connected === true;
  const kmnetConnecting = kmnetStatus.connecting === true;
  const kmnetDriverAvailable = kmnetStatus.available === true;
  const kmnetButtonLeft = kmnetStatus.button_left === true;
  const kmnetButtonRight = kmnetStatus.button_right === true;
  const previewEnabled = consumersConfig.preview !== false;
  const activeModelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const artifact = runtime?.active_model?.artifact;
  const version = runtime?.active_model?.version;
  const activeArtifactLabel = artifact ? `${artifact.kind.toUpperCase()} · ${artifact.path}` : "未加载产物";
  const lastModelSwitchError = readString(runtime?.inference?.last_switch_error, "");
  const runtimeInputShape = readString(runtime?.inference?.input_shape, "");
  const registeredInputShape = version?.input_shape === "engine-probe-required"
    ? "等待 TensorRT engine 探测"
    : version?.input_shape ?? "";
  const displayedInputShape = runtimeInputShape || registeredInputShape;
  const deepstreamModelOutputSummary = formatShape(runtimeModelOutput.shape);
  const deepstreamClassNamesSummary =
    runtimeModelOutputClassNames.length > 0
      ? runtimeModelOutputClassNames.slice(0, 8).join(", ") +
        (runtimeModelOutputClassNames.length > 8 ? ` +${runtimeModelOutputClassNames.length - 8}` : "")
      : "-";
  const runtimePostprocessParser = readString(runtimePostprocess.parser, "-");
  const runtimePostprocessConfidence = readNumber(runtimePostprocess.confidence_threshold, Number.NaN);
  const runtimePostprocessNms = readNumber(runtimePostprocess.nms_threshold, Number.NaN);
  const detectionBatchFps = readNumber(statistics?.detection_batch_fps, 0);
  const controlObservationFps = readNumber(statistics?.control_observation_fps, 0);
  const lastFrameAgeMs = readNumber(statistics?.last_frame_age_ms, 0);
  const controlLatencyGuardMs = 55;
  const inferenceThroughputHealthy = readNumber(statistics?.inference_fps, 0) > 0 || detectionBatchFps > 0;
  const inferenceStaleRejected =
    inferenceTrace.stale_rejected === true ||
    inferenceTrace.latest_rejected === true;
  const inferenceFreshnessBlocked =
    inferenceStaleRejected ||
    (
      inferenceThroughputHealthy &&
      controlObservationFps <= 0 &&
      lastFrameAgeMs > controlLatencyGuardMs
    );
  const switchableArtifacts = modelArtifacts.filter(
    (item) =>
      item.kind === "engine" &&
      (item.status === "ready" || item.status === "pending" || item.status === "failed") &&
      (selectedModelVersionId === "" || item.version_id === selectedModelVersionId)
  );
  const sortedSwitchableArtifacts = [...switchableArtifacts].sort((left, right) => {
    const leftRank = ARTIFACT_KIND_RANK[left.kind] ?? 99;
    const rightRank = ARTIFACT_KIND_RANK[right.kind] ?? 99;
    return leftRank - rightRank || left.path.localeCompare(right.path);
  });
  const selectedSwitchArtifact =
    typeof selectedModelArtifactId === "number"
      ? sortedSwitchableArtifacts.find((item) => item.id === selectedModelArtifactId) ?? null
      : sortedSwitchableArtifacts[0] ?? null;
  const preferredSwitchArtifact = sortedSwitchableArtifacts[0] ?? null;
  const blockedSwitchArtifacts = modelArtifacts.filter(
    (item) =>
      item.kind === "engine" &&
      !["ready", "pending", "failed"].includes(item.status)
  );
  const detections = readNumber(vision.detections, 0);
  const target = asRecord(vision.target);
  const activeRuntimeClassId = readNullableNumber(target.cls ?? target.class_id);
  const runtimeDetectionClassIds = recordArray(vision.detection_items).flatMap((item) => {
    const classId = readNullableNumber(item.cls ?? item.class_id);
    return classId !== null && Number.isInteger(classId) ? [classId] : [];
  });
  const classEditorIds = Array.from(new Set([
    ...detectionClasses.map((_, classId) => classId),
    ...classPriorityIds,
    ...runtimeDetectionClassIds,
    ...(activeRuntimeClassId !== null && Number.isInteger(activeRuntimeClassId)
      ? [activeRuntimeClassId]
      : [])
  ])).filter((classId) => classId >= 0 && classId <= 255).sort((left, right) => left - right);
  const control = asRecord(vision.control);
  const targetPipeline = asRecord(vision.target_pipeline);
  const targetPipelineCounts = asRecord(targetPipeline.counts);
  const targetPipelineCode = readString(targetPipeline.code, "");
  const targetPipelineStage = readString(targetPipeline.stage, "");
  const targetPipelineMessage = readString(targetPipeline.message, "");
  const targetPipelineRejections = Array.isArray(targetPipeline.rejection_reasons)
    ? targetPipeline.rejection_reasons.map((item) => String(item)).join(", ")
    : "";
  const selectorDebug = asRecord(control.selector_debug);
  const controlCandidateFilter = asRecord(control.candidate_filter);
  const selectionCenter = asRecord(controlCandidateFilter.selection_center_px);
  const rejectedControlCandidates = recordArray(controlCandidateFilter.rejected);
  const firstRejectedControlCandidate = rejectedControlCandidates[0] ?? {};
  const trackerRuntimeDebug = asRecord(selectorDebug.tracker);
  const trackerTiming = asRecord(trackerRuntimeDebug.timing);
  const trackDiagnostics = asRecord(control.track_diagnostics ?? target.track_diagnostics);
  const diagnosticTracks = recordArray(trackDiagnostics.tracks);
  const selectedTrackId = finiteNumber(trackDiagnostics.selected_track_id);
  const selectedTrackDebug =
    selectedTrackId === null
      ? {}
      : diagnosticTracks.find((item) => readNumber(item.track_id, Number.NaN) === selectedTrackId) ?? {};
  const selectedTrackEstimate = asRecord(selectedTrackDebug.estimate);
  const controlPipeline = asRecord(asRecord(vision.control).pipeline);
  const mouseObservation = asRecord(control.mouse_observation ?? target.mouse_observation);
  const rawAimDebug = asRecord(mouseObservation.raw_aim);
  const controlHasSample = Object.keys(control).length > 0;
  const controlHasTarget = Object.keys(target).length > 0;
  const controlCandidateCount = readNullableNumber(
    control.candidates ?? selectorDebug.filtered_candidates
  );
  const controlTrackId = readNullableNumber(
    target.track_id ?? trackDiagnostics.selected_track_id
  );
  const controlWidthPx = readNullableNumber(mouseObservation.control_width_px);
  const controlHeightPx = readNullableNumber(mouseObservation.control_height_px);
  const controlCenterX = controlWidthPx === null ? null : controlWidthPx * 0.5;
  const controlCenterY = controlHeightPx === null ? null : controlHeightPx * 0.5;
  const predictedAimX = readNullableNumber(mouseObservation.predicted_x_px ?? control.aim_x);
  const predictedAimY = readNullableNumber(mouseObservation.predicted_y_px ?? control.aim_y);
  const observedAimX = readNullableNumber(mouseObservation.observed_x_px ?? target.observed_aim_x);
  const observedAimY = readNullableNumber(mouseObservation.observed_y_px ?? target.observed_aim_y);
  const predictedErrorXPx =
    predictedAimX !== null && controlCenterX !== null ? predictedAimX - controlCenterX : null;
  const predictedErrorYPx =
    predictedAimY !== null && controlCenterY !== null ? predictedAimY - controlCenterY : null;
  const predictedErrorDistancePx =
    predictedErrorXPx !== null && predictedErrorYPx !== null
      ? Math.hypot(predictedErrorXPx, predictedErrorYPx)
      : null;
  const controlMeasurementDtS = readNullableNumber(controlPipeline.measurement_dt_s ?? mouseObservation.measurement_dt_s);
  const controlPredictionHorizonS = readNullableNumber(mouseObservation.prediction_horizon_s);
  const controlSendDuration = formatDurationFromNs(
    execution.device_send_start_ts_ns ?? executionMeta.device_send_start_ts_ns,
    execution.device_send_end_ts_ns ?? executionMeta.device_send_end_ts_ns
  );
  const controlActualDx = executionMeta.driver_dx ?? execution.output_dx ?? executionIntent.dx;
  const controlActualDy = executionMeta.driver_dy ?? execution.output_dy ?? executionIntent.dy;
  const controlNoSendReason = execution.sent === true
    ? "已发送"
    : !controlHasTarget
      ? targetPipelineMessage || readString(control.selection_reason, "无目标")
      : control.will_emit !== true
        ? readString(control.no_send_reason, readString(control.trigger_reason, readString(control.reason, "控制门控未通过")))
        : kmnetStatus.connected !== true
          ? "设备未连接"
          : readString(execution.message, readString(control.reason, "控制输出为零"));
  const deepstreamInputFrames = readNumber(runtimeInference.input_frames, 0);
  const deepstreamOutputBuffers = readNumber(runtimeInference.output_buffers, 0);
  const deepstreamBatchMetaBuffers = readNumber(runtimeInference.batch_meta_buffers, 0);
  const deepstreamFrameMetaFrames = readNumber(runtimeInference.frame_meta_frames, 0);
  const deepstreamPublishedBatches = readNumber(runtimeInference.published_batches, 0);
  const deepstreamParserStatus = asRecord(runtimeInference.parser);
  const deepstreamInferenceCompleted =
    deepstreamOutputBuffers > 0 ||
    deepstreamPublishedBatches > 0;
  const inferenceRan =
    inferenceTrace.ran === true || (deepstreamNvinferSelected && deepstreamInferenceCompleted);
  const inferenceAvailable =
    inferenceTrace.available === true ||
    (deepstreamNvinferSelected && runtimeInference.loaded === true && runtimeInference.terminal_error !== true);
  const inferenceReason = inferenceTrace.ran === true
    ? readString(inferenceTrace.reason, readString(vision.inference_reason, "-"))
    : readString(
        runtimeInference.inference_reason,
        readString(inferenceTrace.reason, readString(vision.inference_reason, "-"))
      );
  const deepstreamPreviewStreamReady =
    deepstreamNvinferSelected &&
    previewEnabled &&
    runtimeInference.preview_enabled === true &&
    runtimeInference.loaded === true &&
    runtimeInference.terminal_error !== true;
  const previewImageAvailable = deepstreamNvinferSelected
    ? deepstreamPreviewStreamReady
    : runtime?.capture?.available === true;
  const previewUnavailableReason = deepstreamNvinferSelected
    ? readString(runtimeInference.preview_reason, "等待 DeepStream 硬件预览帧")
    : "预览帧尚不可用";
  const inferenceBatchGeneration = readNumber(
    inferenceTrace.generation ?? inferenceTrace.detection_batch_generation ?? pipeline.last_generation,
    Number.NaN
  );
  const inferenceAcquiredGeneration = readNumber(inferenceTrace.acquired_generation, Number.NaN);
  const inferenceLatestGeneration = readNumber(inferenceTrace.latest_generation, Number.NaN);
  const inferenceBrokerPublishedGeneration = readNumber(inferenceTrace.broker_published_generation, Number.NaN);
  const inferenceGenerationLag = readNumber(inferenceTrace.generation_lag, Number.NaN);
  const inferenceFrameIdLag = readNumber(inferenceTrace.frame_id_lag, Number.NaN);
  const inferencePublishedSinceAcquire = readNumber(inferenceTrace.published_since_acquire, Number.NaN);
  const inferenceResultAgeMs = readNumber(
    inferenceTrace.result_age_ms ?? inferenceTrace.detection_batch_result_age_ms,
    Number.NaN
  );
  const latestFrameBroker = asRecord(pipeline.latest_frame_broker);
  const latestCaptureFrameId = readNullableNumber(
    latestFrameBroker.published_frame_id ?? deepstreamStatus.last_frame_id
  );
  const latestCaptureGeneration = readNullableNumber(
    latestFrameBroker.published_generation ?? deepstreamMailbox.latest_generation
  );
  const latestCaptureTsNs = readNullableNumber(
    latestFrameBroker.published_capture_ts_ns ?? deepstreamStatus.last_capture_ts_ns
  );
  const latestCaptureAgeMs = readNullableNumber(
    latestFrameBroker.published_frame_age_ms ?? deepstreamStatus.latest_frame_age_ms ?? captureStatistics.latest_frame_age_ms
  );
  const latestCaptureOutputWidth = readNullableNumber(
    latestFrameBroker.published_width ?? deepstreamNvmmOutput.width
  );
  const latestCaptureOutputHeight = readNullableNumber(
    latestFrameBroker.published_height ?? deepstreamNvmmOutput.height
  );
  const latestCaptureOutputFormat = readString(
    latestFrameBroker.published_format ?? deepstreamNvmmOutput.pixel_format,
    ""
  );
  const latestCaptureOutputMemory = readString(
    latestFrameBroker.published_resource_memory ?? deepstreamNvmmOutput.memory,
    ""
  );
  const latestCaptureTimestampSource = readString(
    latestFrameBroker.published_capture_ts_source ?? deepstreamStatus.timestamp_source,
    ""
  );
  const capturePublishedFrames = readNullableNumber(
    latestFrameBroker.published_frames ?? deepstreamStatus.input_frames ?? captureStatistics.published_frames
  );
  const captureOverwrittenFrames = readNullableNumber(
    latestFrameBroker.overwritten_frames ?? deepstreamMailbox.overwritten_batches ?? captureStatistics.overwritten_frames
  );
  const captureDroppedFrames = readNullableNumber(
    deepstreamStatus.stale_dropped_batches ?? captureStatistics.dropped_counter ?? capture?.frames_dropped
  );
  const captureFramePeriodMs = readNullableNumber(
    deepstreamStatus.last_capture_interval_ms ?? capture?.frame_period_ms
  );
  const captureArrivalFps = readNullableNumber(
    deepstreamStatus.input_fps ?? capture?.fps_capture ?? captureStatistics.capture_fps
  );
  const captureBackendLabel = deepstreamNvinferSelected
    ? "deepstream_nvinfer"
    : readString(capture?.backend, captureBackendMode);
  const captureReason = deepstreamNvinferSelected
    ? readString(
        deepstreamStatus.last_error,
        runtimeMainlineRunning ? "NVMM 管线持续向 nvinfer 输入帧" : "等待 DeepStream 主链启动"
      )
    : readString(
        capture?.last_error,
        capture?.available === true
          ? readString(capture?.profile?.selection_reason, "采集帧持续到达")
          : "尚无采集样本"
      );
  const inferenceDebug = asRecord(inferenceTrace.debug);
  const decodeDebug = asRecord(inferenceDebug.decode);
  const inferenceTimings = asRecord(inferenceDebug.timings);
  const trtTimings = asRecord(decodeDebug.timings);
  const preprocessDebug = asRecord(inferenceDebug.preprocess);
  const detectionBatchMetadata = asRecord(inferenceTrace.detection_batch_metadata);
  const nativeParserTelemetry = Object.keys(asRecord(detectionBatchMetadata.parser)).length > 0
    ? asRecord(detectionBatchMetadata.parser)
    : deepstreamParserStatus;
  const roiInputWidth = readNumber(inferenceTrace.input_width, readNumber(preprocessDebug.roi_width, roiSize));
  const roiInputHeight = readNumber(inferenceTrace.input_height, readNumber(preprocessDebug.roi_height, roiSize));
  const modelInputWidth = readNumber(inferenceTrace.model_input_width, readNumber(preprocessDebug.model_width, 0));
  const modelInputHeight = readNumber(inferenceTrace.model_input_height, readNumber(preprocessDebug.model_height, 0));
  const inputDownscaleFactor = readNumber(
    inferenceTrace.input_downscale_factor,
    readNumber(preprocessDebug.downscale_factor, 0)
  );
  const inputPixelRatio = readNumber(inferenceTrace.input_pixel_ratio, readNumber(preprocessDebug.pixel_ratio, 0));
  const inputDensityWarning =
    inferenceTrace.input_density_warning === true || preprocessDebug.density_warning === true;
  const inferenceFrameId = readNullableNumber(inferenceTrace.frame_id ?? runtimeInference.last_frame_id);
  const inferenceCaptureTsNs = readNullableNumber(inferenceTrace.capture_ts_ns);
  const inferenceStartTsNs = readNullableNumber(inferenceTrace.inference_start_ts_ns);
  const inferenceEndTsNs = readNullableNumber(inferenceTrace.inference_end_ts_ns);
  const inferenceStartAgeMs =
    inferenceStartTsNs !== null && inferenceCaptureTsNs !== null && inferenceStartTsNs >= inferenceCaptureTsNs
      ? (inferenceStartTsNs - inferenceCaptureTsNs) / 1e6
      : null;
  const inferenceEndAgeMs =
    inferenceEndTsNs !== null && inferenceCaptureTsNs !== null && inferenceEndTsNs >= inferenceCaptureTsNs
      ? (inferenceEndTsNs - inferenceCaptureTsNs) / 1e6
      : readNullableNumber(inferenceResultAgeMs);
  const inferenceInputDtype = readString(runtimeInference.input_dtype, "");
  const inferenceInputLayout = readString(
    inferenceDebug.input_layout,
    readString(runtimeInference.input_layout, "")
  );
  const inferencePreprocessMs = readNullableNumber(
    inferenceTrace.preprocess_ms ?? inferenceTimings.native_preprocess_total_ms ?? inferenceTimings.numpy_tensor_ms
  );
  const inferenceUploadMs = readNullableNumber(
    inferenceTrace.h2d_ms ?? trtTimings.h2d_enqueue_ms
  );
  const inferenceEnqueueMs = readNullableNumber(trtTimings.execute_enqueue_ms);
  const inferenceSyncWaitMs = readNullableNumber(trtTimings.stream_sync_ms);
  const inferenceTotalMs = readNullableNumber(
    inferenceTrace.detection_batch_inference_latency_ms ?? inferenceTimings.execute_total_ms ?? statistics?.stage_engine_ms
  );
  const inferencePostprocessMs = readNullableNumber(
    trtTimings.decode_ms ?? nativeParserTelemetry.decode_ms ?? statistics?.stage_postprocess_ms
  );
  const inferenceRawCandidateCount = readNullableNumber(
    decodeDebug.raw_candidates ?? nativeParserTelemetry.input_candidates
  );
  const inferenceThresholdCandidateCount = readNullableNumber(
    decodeDebug.threshold_candidates ?? nativeParserTelemetry.decoded_candidates
  );
  const inferenceNmsDetectionCount = readNullableNumber(decodeDebug.nms_detections ?? inferenceTrace.mapped_detections);
  const inferenceHighestConfidence = readNullableNumber(decodeDebug.max_score);
  const inferenceOutputName = readString(runtimeInference.output_name, readString(runtimeModelOutput.name, ""));
  const inferenceOutputShape = formatShape(
    decodeDebug.output_shape ?? inferenceDebug.output_shape ?? runtimeInference.output_shape ?? runtimeModelOutput.shape
  );
  const inferenceParser = readString(
    runtimePostprocess.parser,
    readString(decodeDebug.selected_layout, deepstreamNvinferSelected ? "NvDsInferParseNovaSight" : "")
  );
  const inferenceBatchPublished =
    (readNullableNumber(inferenceTrace.publish_ts_ns) ?? 0) > 0 || deepstreamPublishedBatches > 0;
  const inferenceBatchStale =
    inferenceTrace.is_stale === true || inferenceTrace.stale_rejected === true || inferenceTrace.latest_rejected === true;
  const lastError = localError ?? Object.values(errors)[0] ?? capture?.last_error;

  useEffect(() => {
    if (configuredCaptureDevice) {
      setDevice(configuredCaptureDevice);
    } else if (runtime?.capture?.device) {
      setDevice(runtime.capture.device);
    }
  }, [configuredCaptureDevice, runtime?.capture?.device]);

  useEffect(() => {
    if (!selectedChoiceId && configuredChoiceId) {
      setSelectedChoiceId(configuredChoiceId);
      return;
    }
    if (!selectedChoiceId && runningChoiceId) {
      setSelectedChoiceId(runningChoiceId);
    }
  }, [configuredChoiceId, runningChoiceId, selectedChoiceId]);

  useEffect(() => {
    let cancelled = false;
    setModelCatalogLoading(true);
    getModelCatalog()
      .then(async (result) => {
        if (cancelled) {
          return;
        }
        setModelCatalog(result.root);
        setModelCatalogModelCount(result.model_count);
        setModelCatalogDirectoryCount(result.directory_count);
        setExpandedModelDirectories((current) => {
          const next = new Set(current);
          next.add("");
          for (const child of result.root.children) {
            if (child.type === "directory") {
              next.add(child.relative_path);
            }
          }
          return next;
        });
        if (result.updated_files > 0) {
          await onRefresh();
        }
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setModelCatalog(null);
        setModelCatalogModelCount(0);
        setModelCatalogDirectoryCount(0);
        setLocalError(`模型目录读取失败：${getErrorMessage(err)}`);
        reportError(err, { source: "model-catalog", title: "模型目录读取失败" });
      })
      .finally(() => {
        if (!cancelled) {
          setModelCatalogLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modelCatalogRefreshKey, onRefresh]);

  useEffect(() => {
    if (!modelCatalog || typeof artifact?.id !== "number") {
      return;
    }
    setSelectedModelCatalogPath((current) =>
      current ?? findCatalogModelPath(modelCatalog, artifact.id)
    );
  }, [artifact?.id, modelCatalog]);

  useEffect(() => {
    if (projects.length === 0) {
      setSelectedModelProjectId("");
      setModelVersions([]);
      setSelectedModelVersionId("");
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
      return;
    }
    setSelectedModelProjectId((current) =>
      typeof current === "number" && projects.some((project) => project.id === current)
        ? current
        : runtime?.active_model?.project?.id ?? projects[0].id
    );
  }, [projects, runtime?.active_model?.project?.id]);

  useEffect(() => {
    if (selectedModelProjectId === "") {
      loadedModelProjectIdRef.current = "";
      setModelVersions([]);
      setSelectedModelVersionId("");
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
      return;
    }
    let cancelled = false;
    const projectChanged = loadedModelProjectIdRef.current !== selectedModelProjectId;
    loadedModelProjectIdRef.current = selectedModelProjectId;
    if (projectChanged) {
      preferLatestModelVersionRef.current = false;
      setModelVersions([]);
      setSelectedModelVersionId("");
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
    }
    getModelVersions(selectedModelProjectId)
      .then((items) => {
        if (cancelled) {
          return;
        }
        setModelVersions(items);
        const preferLatest = preferLatestModelVersionRef.current;
        preferLatestModelVersionRef.current = false;
        setSelectedModelVersionId((current) => {
          const requested = requestedModelSelectionRef.current;
          if (
            requested?.projectId === selectedModelProjectId &&
            items.some((item) => item.id === requested.versionId)
          ) {
            return requested.versionId;
          }
          if (preferLatest) {
            return items[items.length - 1]?.id ?? "";
          }
          if (typeof current === "number" && items.some((item) => item.id === current)) {
            return current;
          }
          const activeVersionId = runtime?.active_model?.version?.id;
          return typeof activeVersionId === "number" &&
            items.some((item) => item.id === activeVersionId)
            ? activeVersionId
            : items[0]?.id ?? "";
        });
      })
      .catch((err) => {
        if (!cancelled) {
          setModelVersions([]);
          setSelectedModelVersionId("");
          setLocalError(`模型版本读取失败：${getErrorMessage(err)}`);

          reportError(err, { source: 'studio', title: '操作失败' });
          reportError(err, { source: "model-versions", title: "模型版本读取失败" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modelCatalogRefreshKey, runtime?.active_model?.version?.id, selectedModelProjectId]);

  useEffect(() => {
    if (selectedModelVersionId === "") {
      loadedModelVersionIdRef.current = "";
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
      return;
    }
    let cancelled = false;
    const versionChanged = loadedModelVersionIdRef.current !== selectedModelVersionId;
    loadedModelVersionIdRef.current = selectedModelVersionId;
    if (versionChanged) {
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
    }
    getModelArtifacts(selectedModelVersionId)
      .then((items) => {
        if (cancelled) {
          return;
        }
        setModelArtifacts(items);
        const runnable = items
          .filter(
            (item) =>
              (item.status === "ready" || (item.kind === "engine" && item.status === "pending")) &&
              (item.kind === "onnx" || item.kind === "engine")
          )
          .sort((left, right) => {
            const leftRank = ARTIFACT_KIND_RANK[left.kind] ?? 99;
            const rightRank = ARTIFACT_KIND_RANK[right.kind] ?? 99;
            return leftRank - rightRank || left.path.localeCompare(right.path);
          });
        setSelectedModelArtifactId((current) => {
          const requested = requestedModelSelectionRef.current;
          if (
            requested?.versionId === selectedModelVersionId &&
            runnable.some((item) => item.id === requested.artifactId)
          ) {
            requestedModelSelectionRef.current = null;
            return requested.artifactId;
          }
          if (typeof current === "number" && runnable.some((item) => item.id === current)) {
            return current;
          }
          return typeof artifact?.id === "number" &&
            runnable.some((item) => item.id === artifact.id)
            ? artifact.id
            : runnable[0]?.id ?? "";
        });
      })
      .catch((err) => {
        if (!cancelled) {
          setModelArtifacts([]);
          setLocalError(`模型产物读取失败：${getErrorMessage(err)}`);

          reportError(err, { source: 'studio', title: '操作失败' });
          reportError(err, { source: "model-artifacts", title: "模型产物读取失败" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modelCatalogRefreshKey, selectedModelVersionId]);

  const refreshCapabilities = useCallback(async () => {
    setBusy("caps");
    setLocalError(null);
    try {
      const result = await getCaptureCapabilities(device);
      setCaps(result);
      if (!result.available) {
        const reason = result.reason || "设备不可用";
        setLocalError(reason);
        reportError(new Error(reason), { source: "capture-caps", title: "采集设备不可用" });
      }
      const grouped = groupCapabilities(result.capabilities);
      const configured = grouped.find((choice: CapabilityChoice) => choiceMatchesConfig(choice, {
        pixel_format: configuredCapturePixelFormat,
        width: configuredCaptureWidth,
        height: configuredCaptureHeight,
        fps: configuredCaptureFps
      }));
      const running = grouped.find((choice: CapabilityChoice) => runningPixelFormat && (
        choice.pixel_format.toUpperCase() === runningPixelFormat.toUpperCase() &&
        choice.width === runningWidth &&
        choice.height === runningHeight &&
        choice.fps === runningFps
      ));
      const next = configured ?? running ?? grouped[0];
      if (next) {
        setSelectedChoiceId(choiceId(next));
      }
    } catch (err) {
      setLocalError(getErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }, [
    configuredCaptureFps,
    configuredCaptureHeight,
    configuredCapturePixelFormat,
    configuredCaptureWidth,
    device,
    runningFps,
    runningHeight,
    runningPixelFormat,
    runningWidth
  ]);

  useEffect(() => {
    void refreshCapabilities();
  }, [refreshCapabilities]);

  const buildCapturePayload = useCallback((): CaptureSelectPayload => {
    const choice = selectedChoice;
    return choice
      ? {
          device,
          preference: "manual",
          pixel_format: choice.pixel_format,
          width: choice.width,
          height: choice.height,
          fps: choice.fps
        }
      : selectedProfile
        ? {
            device,
            preference: "manual",
            pixel_format: selectedProfile.pixel_format,
            width: selectedProfile.width,
            height: selectedProfile.height,
            fps: selectedProfile.fps
          }
        : {
            device,
            preference: "auto_high_fps"
          };
  }, [device, selectedChoice, selectedProfile]);

  const applyCapture = useCallback(async () => {
    setBusy("capture");
    setLocalError(null);
    const payload = buildCapturePayload();
    try {
      await selectCaptureProfile(payload);
      await onRefresh();
    } catch (err) {
      setLocalError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [buildCapturePayload, onRefresh]);

  const stopCurrentCapture = useCallback(async () => {
    setBusy("stop");
    setLocalError(null);
    setMainlineLaunchAccepted(false);
    setMainlineLaunchMessage("");
    try {
      if (runtimeMainlineSelected) {
        await stopRuntimePipeline();
      } else {
        await stopCapture();
      }
      await onRefresh();
    } catch (err) {
      setLocalError(getErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }, [runtimeMainlineSelected, onRefresh]);

  const startInferenceThread = useCallback(async () => {
    setBusy("runtime.start");
    setLocalError(null);
    try {
      const status = asRecord(await startRuntimePipeline());
      if (readBoolean(status.failed) || !readBoolean(status.running)) {
        throw new Error(readString(status.last_error, "后端未确认推理管线运行。"));
      }
      await onRefresh();
    } catch (err) {
      setLocalError(`启动推理失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  const waitForLaunchFeedback = useCallback((ms: number) => new Promise<void>((resolve) => {
    if (launchTimerRef.current !== null) {
      window.clearTimeout(launchTimerRef.current);
      launchTimerRef.current = null;
    }
    if (launchTimerResolveRef.current) {
      launchTimerResolveRef.current();
      launchTimerResolveRef.current = null;
    }
    launchTimerResolveRef.current = resolve;
    launchTimerRef.current = window.setTimeout(() => {
      launchTimerRef.current = null;
      launchTimerResolveRef.current = null;
      resolve();
    }, ms);
  }), []);

  const resetLaunchDialog = useCallback(() => {
    if (launchTimerRef.current !== null) {
      window.clearTimeout(launchTimerRef.current);
      launchTimerRef.current = null;
    }
    if (launchTimerResolveRef.current) {
      launchTimerResolveRef.current();
      launchTimerResolveRef.current = null;
    }
    launchCancelledRef.current = false;
    setLaunchStatus("idle");
    setLaunchStageIndex(0);
    setLaunchCompletedStages(0);
    setLaunchError("");
    setLaunchProgressDetail("");
  }, []);

  const openMainlineLaunchDialog = useCallback(() => {
    resetLaunchDialog();
    setLaunchDialogOpen(true);
  }, [resetLaunchDialog]);

  const closeLaunchDialog = useCallback(() => {
    if (launchStatus === "running") {
      return;
    }
    setLaunchDialogOpen(false);
  }, [launchStatus]);

  const showLaunchToast = useCallback(() => {
    setLaunchToastVisible(true);
    window.setTimeout(() => setLaunchToastVisible(false), 2600);
  }, []);

  const captureLaunchFailureMessage = useCallback((state: CaptureState): string => {
    const report = asRecord(state.report);
    const sections = Array.isArray(report.sections) ? report.sections.map(asRecord) : [];
    return (
      readString(state.last_error, "") ||
      readString(report.message, "") ||
      sections.map((section) => readString(section.message, "")).find(Boolean) ||
      "采集配置未进入可用状态。"
    );
  }, []);

  const assertCaptureLaunchState = useCallback((state: CaptureState) => {
    if (state.available !== true) {
      throw new Error(`采集阶段失败：${captureLaunchFailureMessage(state)}`);
    }
  }, [captureLaunchFailureMessage]);

  const assertRuntimeLaunchState = useCallback((state: RuntimeState, stageTitle: string) => {
    const status = getRuntimeMainlineStatus(state);
    if (status.failed || !status.running) {
      throw new Error(`${stageTitle}失败：${status.failureMessage || "后端运行态未确认。"}`);
    }
  }, []);

  const waitForRuntimeMainlineReady = useCallback(async (
    stageTitle: string,
    timeoutMs = 6000,
    intervalMs = 600
  ) => {
    const deadline = Date.now() + timeoutMs;
    let lastMessage = "后端运行态未确认。";
    while (Date.now() <= deadline) {
      if (launchCancelledRef.current) {
        throw new Error("launch cancelled");
      }
      const state = await getRuntimeState();
      const status = getRuntimeMainlineStatus(state);
      setLaunchProgressDetail(
        status.progressSummary ? `当前计数：${status.progressSummary}` : "等待后端运行态确认。"
      );
      if (status.failed) {
        throw new Error(`${stageTitle}失败：${status.failureMessage || lastMessage}`);
      }
      if (status.running) {
        setLaunchProgressDetail(
          status.progressSummary ? `运行态已确认：${status.progressSummary}` : "运行态已确认。"
        );
        return state;
      }
      if (status.failureMessage) {
        lastMessage = status.failureMessage;
      }
      await waitForLaunchFeedback(intervalMs);
      if (launchCancelledRef.current) {
        throw new Error("launch cancelled");
      }
    }
    throw new Error(`${stageTitle}失败：等待后端运行态超时，${lastMessage}`);
  }, [waitForLaunchFeedback]);

  const waitForRuntimeEvidence = useCallback(async (
    stageTitle: string,
    hasEvidence: (state: RuntimeState) => boolean,
    missingMessage: string,
    timeoutMs = 6000,
    intervalMs = 600
  ) => {
    const deadline = Date.now() + timeoutMs;
    let lastSummary = "";
    while (Date.now() <= deadline) {
      if (launchCancelledRef.current) {
        throw new Error("launch cancelled");
      }
      const state = await getRuntimeState();
      const status = getRuntimeMainlineStatus(state);
      lastSummary = status.progressSummary;
      setLaunchProgressDetail(
        status.progressSummary ? `当前计数：${status.progressSummary}` : "等待主链输出启动证据。"
      );
      if (status.failed) {
        throw new Error(`${stageTitle}失败：${status.failureMessage || missingMessage}`);
      }
      if (!status.running) {
        await waitForLaunchFeedback(intervalMs);
        continue;
      }
      if (hasEvidence(state)) {
        setLaunchProgressDetail(
          status.progressSummary ? `已收到启动证据：${status.progressSummary}` : "已收到启动证据。"
        );
        return state;
      }
      await waitForLaunchFeedback(intervalMs);
    }
    throw new Error(`${stageTitle}失败：${missingMessage}${lastSummary ? ` 当前计数：${lastSummary}` : ""}`);
  }, [waitForLaunchFeedback]);

  const startMainlineLaunch = useCallback(async () => {
    if (launchStatus === "running") {
      return;
    }
    launchCancelledRef.current = false;
    setBusy("runtime.start");
    setLocalError(null);
    setLaunchStatus("running");
    setLaunchError("");
    setLaunchProgressDetail("正在提交启动请求，等待后端阶段反馈。");
    setLaunchStageIndex(0);
    setLaunchCompletedStages(0);

    const ensureNotCancelled = () => {
      if (launchCancelledRef.current) {
        throw new Error("launch cancelled");
      }
    };
    const runStage = async (index: number, action?: () => Promise<void>) => {
      ensureNotCancelled();
      setLaunchStageIndex(index);
      await waitForLaunchFeedback(160);
      ensureNotCancelled();
      if (action) {
        await action();
      }
      ensureNotCancelled();
      setLaunchCompletedStages(index + 1);
      await waitForLaunchFeedback(180);
    };

    try {
      await runStage(0);
      await runStage(1, async () => {
        const captureState = await selectCaptureProfile(buildCapturePayload());
        assertCaptureLaunchState(captureState);
      });
      await runStage(2, async () => {
        const status = asRecord(await startRuntimePipeline());
        const accepted = readBoolean(status.running, true);
        if (!accepted) {
          const reason = readString(status.last_error, "后端未确认主链运行。");
          throw new Error(reason);
        }
        await waitForRuntimeMainlineReady("启动主链运行管线");
        setMainlineLaunchAccepted(true);
        setMainlineLaunchMessage("主链启动请求已提交，正在等待后端状态确认。");
      });
      await runStage(3, async () => {
        await waitForRuntimeEvidence(
          "激活跟踪与控制",
          (state) => getRuntimeMainlineStatus(state).hasRuntimeConsumption,
          "runtime 尚未消费 DetectionBatch，跟踪与控制没有输入。"
        );
      });
      await runStage(4, async () => {
        const state = await waitForRuntimeMainlineReady("确认设备执行器", 1600);
        assertRuntimeLaunchState(state, "确认设备执行器");
        await onRefresh();
      });
      setLaunchStatus("success");
      setLaunchCompletedStages(MAINLINE_LAUNCH_STAGES_CUSTOM_TENSORRT.length);

      setLaunchProgressDetail((detail) => detail || "主链启动完成，后端运行态已确认。");
      showLaunchToast();
    } catch (err) {
      if (launchCancelledRef.current) {
        setLaunchStatus("cancelled");
        setLaunchError("");
        setLaunchProgressDetail("启动已取消，已停止继续等待后端阶段反馈。");
        return;
      }
      setLaunchStatus("failed");
      setLaunchError(getErrorMessage(err));
      setMainlineLaunchAccepted(false);
      setMainlineLaunchMessage("");
      setLocalError(`启动主链失败：${getErrorMessage(err)}`);

      reportError(err, { source: "mainline-launch", title: "启动主链失败" });
      try {
        await stopRuntimePipeline();
      } catch {
        // Keep the original launch error visible; refresh below exposes stop failures if backend reports them.
      }
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [
    assertCaptureLaunchState,
    assertRuntimeLaunchState,
    buildCapturePayload,
    launchStatus,
    onRefresh,
    showLaunchToast,
    waitForLaunchFeedback,
    waitForRuntimeEvidence,
    waitForRuntimeMainlineReady
  ]);

  const cancelMainlineLaunch = useCallback(async () => {
    if (launchStatus !== "running") {
      closeLaunchDialog();
      return;
    }
    launchCancelledRef.current = true;
    if (launchTimerRef.current !== null) {
      window.clearTimeout(launchTimerRef.current);
      launchTimerRef.current = null;
    }
    if (launchTimerResolveRef.current) {
      launchTimerResolveRef.current();
      launchTimerResolveRef.current = null;
    }
    setLaunchStatus("cancelled");
    setLaunchError("");
    setLaunchProgressDetail("启动已取消，已向后端发送停止主链请求。");
    setMainlineLaunchAccepted(false);
    setMainlineLaunchMessage("");
    setBusy(null);
    try {
      await stopRuntimePipeline();
      await onRefresh();
    } catch (err) {
      setLocalError(`取消启动失败：${getErrorMessage(err)}`);

      reportError(err, { source: "mainline-cancel", title: "取消启动失败" });
    }
  }, [closeLaunchDialog, launchStatus, onRefresh]);

  const toggleCapture = useCallback(async () => {
    if (captureMainRunning) {
      await stopCurrentCapture();
      return;
    }
    if (runtimeMainlineSelected) {
      openMainlineLaunchDialog();
      return;
    }
    await applyCapture();
  }, [applyCapture, captureMainRunning, runtimeMainlineSelected, openMainlineLaunchDialog, stopCurrentCapture]);

  const updateConfigField = useCallback(
    async (section: string, key: string, value: RuntimeConfigValue) => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const next = base ? normalizeRuntimeConfig(base) : null;
      if (!next) {
        return;
      }
      const writeSeq = ++configWriteSeqRef.current;
      pendingConfigWritesRef.current += 1;
      setBusy(`${section}.${key}`);
      setLocalError(null);
      const sectionValue = {
        ...asRecord(next[section])
      };
      sectionValue[key] = value;
      next[section] = sectionValue as RuntimeConfig[string];
      configDraftRef.current = next;
      setConfigDraft(next);
      try {
        const result = await updateRuntimeConfigField(section, key, value);
        if (writeSeq === configWriteSeqRef.current) {
          const applied = normalizeRuntimeConfig(result.config);
          configDraftRef.current = applied;
          setConfigDraft(applied);
        }
      } catch (err) {
        setLocalError(`配置同步失败：${getErrorMessage(err)}`);

        reportError(err, { source: 'studio', title: '操作失败' });
        if (writeSeq === configWriteSeqRef.current) {
          configDraftRef.current = null;
          setConfigDraft(null);
        }
      } finally {
        pendingConfigWritesRef.current = Math.max(0, pendingConfigWritesRef.current - 1);
        if (writeSeq === configWriteSeqRef.current) {
          setBusy(null);
        }
        if (pendingConfigWritesRef.current === 0) {
          await onRefresh();
        }
      }
    },
    [onRefresh, runtimeConfig]
  );

  const updateControlGroupField = useCallback(
    async (group: "aim" | "calibrated_angular" | "universal_saturated" | "shared", key: string, value: RuntimeConfigValue) => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const control = nestedRecord(base, "control");
      if (group !== "aim" && group !== "shared") {
        const algorithms = nestedRecord(control, "algorithms");
        const algorithm = {
          ...nestedRecord(algorithms, group),
          [key]: value
        };
        await updateConfigField("control", "algorithms", {
          ...algorithms,
          [group]: algorithm
        } as RuntimeConfigValue);
        return;
      }
      const groupValue = {
        ...nestedRecord(control, group),
        [key]: value
      };
      await updateConfigField("control", group, groupValue as RuntimeConfigValue);
    },
    [runtimeConfig, updateConfigField]
  );

  const updateDualPhasePath = useCallback(
    async (path: string[], value: RuntimeConfigValue) => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const control = nestedRecord(base, "control");
      const algorithms = nestedRecord(control, "algorithms");
      const algorithmId = readString(
        control.active_algorithm,
        "dual_phase_atan_robust_predictive_v2"
      );
      const algorithm = nestedRecord(algorithms, algorithmId);
      const writeNested = (
        record: Record<string, unknown>,
        remainingPath: string[]
      ): Record<string, unknown> => {
        const [head, ...rest] = remainingPath;
        if (!head) {
          return record;
        }
        return {
          ...record,
          [head]: rest.length > 0
            ? writeNested(nestedRecord(record, head), rest)
            : value
        };
      };
      await updateConfigField("control", "algorithms", {
        ...algorithms,
        [algorithmId]: writeNested(algorithm, path)
      } as RuntimeConfigValue);
    },
    [runtimeConfig, updateConfigField]
  );

  const updateDetectionClassName = useCallback(
    async (classId: number, name: string) => {
      const nextClasses = [...detectionClasses];
      while (nextClasses.length <= classId) {
        nextClasses.push("");
      }
      nextClasses[classId] = name.trim();
      await updateConfigField("inference", "detection_class_profiles", {
        ...detectionProfiles,
        [activeDetectionProfile]: nextClasses
      } as RuntimeConfigValue);
    },
    [activeDetectionProfile, detectionClasses, detectionProfiles, updateConfigField]
  );

  const updateClassAimRatio = useCallback(
    async (classId: number, ratio: number | null) => {
      const nextProfileRatios = { ...activeClassAimRatios };
      if (ratio === null) {
        delete nextProfileRatios[String(classId)];
      } else {
        nextProfileRatios[String(classId)] = clampNumber(Number(ratio.toFixed(2)), 0, 1);
      }
      const nextProfiles = { ...classAimRatioProfiles };
      if (Object.keys(nextProfileRatios).length > 0) {
        nextProfiles[activeDetectionProfile] = nextProfileRatios;
      } else {
        delete nextProfiles[activeDetectionProfile];
      }
      await updateControlGroupField("aim", "class_y_ratios", nextProfiles as RuntimeConfigValue);
    },
    [
      activeClassAimRatios,
      activeDetectionProfile,
      classAimRatioProfiles,
      updateControlGroupField
    ]
  );

  const moveClassPriority = useCallback(
    async (classId: number, direction: -1 | 1) => {
      const current = classPriorityIds.includes(classId)
        ? [...classPriorityIds]
        : [...classPriorityIds, classId];
      const index = current.indexOf(classId);
      const targetIndex = clampNumber(index + direction, 0, current.length - 1);
      if (index === targetIndex) {
        return;
      }
      [current[index], current[targetIndex]] = [current[targetIndex], current[index]];
      await updateConfigField("inference", "detection_class_priority", current.join(","));
    },
    [classPriorityIds, updateConfigField]
  );

  const applyKmNetRecommended = useCallback(async () => {
    const next = cloneRuntimeConfig(runtimeConfig);
    if (!next) {
      return;
    }
    setBusy("kmnet.defaults");
    setLocalError(null);
    next.hardware = {
      ...asRecord(next.hardware),
      ...KMNET_RECOMMENDED
    } as RuntimeConfig[string];
    try {
      await updateRuntimeConfig(next);
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 推荐参数应用失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
    } finally {
      setBusy(null);
    }
  }, [onRefresh, runtimeConfig]);

  const toggleHardwareConnection = useCallback(async () => {
    setBusy("kmnet.toggle");
    setLocalError(null);
    try {
      if (kmnetConnected || kmnetConnecting) {
        await disconnectKmNet();
      } else {
        await connectKmNet();
      }
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet ${kmnetConnected || kmnetConnecting ? "断开" : "连接"}失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetConnected, kmnetConnecting, onRefresh]);

  const diagnosticMoveHardware = useCallback(async (
    dx = kmnetTestDx,
    dy = kmnetTestDy,
    repeat = 1,
    intervalMs = 0,
    moveKindOverride?: string,
    moveMsOverride = kmnetTestMs,
    bezierCtrl?: { x1: number; y1: number; x2: number; y2: number }
  ) => {
    setBusy("kmnet.diagnostic");
    setLocalError(null);
    setKmnetTestMessage("");
    try {
      const effectiveMoveKind = moveKindOverride ?? moveKind;
      const result = await diagnosticMoveKmNet(
        Math.round(dx),
        Math.round(dy),
        repeat,
        intervalMs,
        effectiveMoveKind,
        Math.round(moveMsOverride),
        bezierCtrl
      );
      const status = asRecord(result.status);
      const metadata = asRecord(result.metadata);
      const stepsSent = readNumber(result.steps_sent, result.sent === true ? repeat : 0);
      const queued = result.queued === true;
      const apiName = readString(metadata.api_name, effectiveMoveKind);
      const driverRc = String(metadata.driver_rc ?? "-");
      setKmnetTestMessage(
        queued
          ? `已下发 ${effectiveMoveKind} dx=${Math.round(dx)} dy=${Math.round(dy)} · ${repeat} 步 · ${Math.round(moveMsOverride)}ms · 后端后台执行`
          : result.sent === true
          ? `已发送 ${apiName} rc=${driverRc} dx=${Math.round(dx)} dy=${Math.round(dy)} · ${Math.round(moveMsOverride)}ms · ${stepsSent}/${repeat} 步 · 累计 ${readNumber(status.move_count, 0)} 次`
          : `未发送 ${effectiveMoveKind}：${readString(result.message, "未知原因")} · ${stepsSent}/${repeat} 步`
      );
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 诊断移动失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetTestDx, kmnetTestDy, kmnetTestMs, moveKind, onRefresh]);

  const diagnosticCircleHardware = useCallback(async () => {
    setBusy("kmnet.circle");
    setLocalError(null);
    setKmnetTestMessage("");
    try {
      const result = await diagnosticCircleKmNet(8, 32, 8);
      const failed = asRecord(result.failed);
      if (result.queued === true) {
        setKmnetTestMessage(`画圆测试已下发 · ${readNumber(result.steps_requested, 32)} 步 · 半径 ${readNumber(result.radius, 8)} · 后端后台执行`);
      } else if (result.sent === true) {
        setKmnetTestMessage(`画圆测试已发送 ${readNumber(result.steps_sent, 0)} 步 · 半径 ${readNumber(result.radius, 8)}`);
      } else {
        setKmnetTestMessage(
          `画圆测试中断：第 ${readNumber(failed.step, 0)} 步 · ${readString(failed.message, "未发送")}`
        );
      }
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 画圆测试失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  const exportConfig = () => {
    if (!runtimeConfig) {
      return;
    }
    const blob = new Blob([JSON.stringify(runtimeConfig, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "novasight-config.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const importConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setBusy("import");
    setLocalError(null);
    try {
      const payload = JSON.parse(await file.text()) as RuntimeConfig;
      await updateRuntimeConfig(payload);
      await onRefresh();
    } catch (err) {
      setLocalError(`导入失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
    } finally {
      setBusy(null);
    }
  };

  const toggleModelDirectory = (relativePath: string) => {
    setExpandedModelDirectories((current) => {
      const next = new Set(current);
      if (next.has(relativePath)) {
        next.delete(relativePath);
      } else {
        next.add(relativePath);
      }
      return next;
    });
  };

  const selectModelFromCatalog = async (model: ModelCatalogModel) => {
    setSelectedModelCatalogPath(model.relative_path);
    let projectId = model.project_id;
    let versionId = model.version_id;
    let artifactId = model.artifact_id;
    if (typeof projectId !== "number" || typeof versionId !== "number" || typeof artifactId !== "number") {
      if (model.kind !== "engine") {
        setLocalError("当前运行主线只允许选择 TensorRT .engine 模型。");
        return;
      }
      setBusy("model.register");
      setLocalError(null);
      try {
        const registered = await registerCatalogModel(model.relative_path);
        projectId = registered.project.id;
        versionId = registered.version.id;
        artifactId = registered.artifact.id;
        setModelCatalogMessage(
          `已引用原始 Engine：${model.relative_path}；未复制模型文件。`
        );
        await onRefresh();
        setModelCatalogRefreshKey((current) => current + 1);
      } catch (err) {
        setLocalError(`模型引用登记失败：${getErrorMessage(err)}`);
        reportError(err, { source: "model-register", title: "模型引用登记失败" });
        return;
      } finally {
        setBusy(null);
      }
    }
    setLocalError(null);
    requestedModelSelectionRef.current = {
      projectId,
      versionId,
      artifactId
    };
    setSelectedModelProjectId(projectId);
    if (selectedModelProjectId === projectId) {
      setSelectedModelVersionId(versionId);
      if (selectedModelVersionId === versionId) {
        setSelectedModelArtifactId(artifactId);
        requestedModelSelectionRef.current = null;
      }
    }
  };

  const switchModel = async () => {
    if (selectedModelProjectId === "" || selectedSwitchArtifact === null) {
      setLocalError("请选择 TensorRT engine 产物。");
      return;
    }
    if (
      selectedModelVersionId === "" ||
      selectedSwitchArtifact.version_id !== selectedModelVersionId
    ) {
      setLocalError("模型选择已刷新，请重新选择这个版本下的推理产物。");
      return;
    }
    setBusy("model.switch");
    setLocalError(null);
    setModelSwitchMessage("");
    try {
      if (selectedSwitchArtifact.kind !== "engine") {
        throw new Error("NovaSight DeepStream 主线只支持 TensorRT Engine。");
      }
      setModelSwitchMessage("正在读取 TensorRT Engine 契约并自动生成 DeepStream 配置...");
      const response = await publishModel(selectedModelProjectId, selectedSwitchArtifact.id);
      if (response.report && !response.report.applied) {
        throw new Error(response.report.message);
      }
      setModelSwitchMessage(
        response.report?.message ??
          "Engine 契约读取完成，DeepStream 配置已自动生成并切换。"
      );
      await onRefresh();
      setModelCatalogRefreshKey((current) => current + 1);
    } catch (err) {
      setLocalError(`模型切换未生效：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  };

  const refreshModelCatalog = async () => {
    setBusy("model.refresh");
    setLocalError(null);
    setModelCatalogMessage("");
    preferLatestModelVersionRef.current = true;
    try {
      const result = await getModelCatalog(false);
      await onRefresh();
      setModelCatalogRefreshKey((current) => current + 1);
      setModelCatalogMessage(
        `刷新完成：发现 ${result.model_count} 个模型文件；列表直接读取原文件，未复制模型。`
      );
    } catch (err) {
      preferLatestModelVersionRef.current = false;
      setLocalError(`模型列表刷新失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
    } finally {
      setBusy(null);
    }
  };

  const rescanModelCatalog = async () => {
    setBusy("model.scan");
    setLocalError(null);
    setModelCatalogMessage("");
    preferLatestModelVersionRef.current = true;
    try {
      const result = await getModelCatalog(true);
      await onRefresh();
      setModelCatalogRefreshKey((current) => current + 1);
      setModelCatalogMessage(
        `重新校验完成：发现 ${result.model_count} 个模型文件；未生成模型副本。`
      );
    } catch (err) {
      preferLatestModelVersionRef.current = false;
      setLocalError(`模型目录扫描失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
    } finally {
      setBusy(null);
    }
  };

  const launchStages = MAINLINE_LAUNCH_STAGES_CUSTOM_TENSORRT;
  const activeLaunchStage =
    launchStages[Math.min(launchStageIndex, launchStages.length - 1)];
  const launchVisibleCompletedStages =
    launchStatus === "idle"
      ? launchCompletedStages
      : Math.max(
          launchCompletedStages,
          Math.min(launchStageIndex + 1, launchStages.length)
        );
  const launchProgress =
    launchStatus === "success"
      ? 100
      : Math.round((launchVisibleCompletedStages / launchStages.length) * 100);
  const launchIndicatorClass =
    launchStatus === "running"
      ? "launch-stage-indicator running"
      : launchStatus === "success"
        ? "launch-stage-indicator success"
        : launchStatus === "failed"
          ? "launch-stage-indicator failed"
          : "launch-stage-indicator";
  const launchIndicatorText =
    launchStatus === "success"
      ? "✓"
      : launchStatus === "failed"
        ? "!"
        : launchStatus === "running"
          ? ""
          : `${launchCompletedStages}/${launchStages.length}`;
  const launchTitle =
    launchStatus === "success"
      ? runtimeMainlineRunning
        ? "视觉处理链路已运行"
        : "主链启动请求已提交"
      : launchStatus === "failed"
        ? "启动主链失败"
        : launchStatus === "cancelled"
          ? "启动流程已停止"
          : activeLaunchStage.title;
  const launchCaption =
    launchStatus === "success"
      ? runtimeMainlineRunning
        ? "采集、ROI、推理、跟踪、控制与执行出口已交由后端主链持有。"
        : "正在等待状态流确认 DetectionBatch 与控制输出，顶部会保持启动确认中。"
      : launchStatus === "failed"
        ? launchError || "后端拒绝启动，已保留当前运行态。"
        : launchStatus === "cancelled"
          ? "已向后端发送停止请求，前端不执行额外回滚逻辑。"
          : activeLaunchStage.caption;
  const realtimeStatusText =
    realtimeStatus === "connected"
      ? "实时推送已连接"
      : realtimeStatus === "stale"
        ? "实时数据陈旧"
        : realtimeStatus === "connecting"
          ? "实时推送连接中"
          : "实时推送已断开";
  const realtimeStatusClass =
    realtimeStatus === "connected"
      ? "console-live good"
      : realtimeStatus === "stale" || realtimeStatus === "connecting"
        ? "console-live warn"
        : "console-live bad";

  return (
    <section className="console-app">
      <header className="console-top">
        <div className="console-brand">
          <div className="console-logo">
            <NovaIcon name="prediction-line" size={24} strokeWidth={1.9} />
          </div>
          NovaSight Studio
        </div>
        <div className="console-toolbar">
          <div className="console-group">
            <div className="console-toolbar-item">
              <NovaIcon name="models" size={15} />
              <span>项目：{projects[0]?.name ?? "默认项目"}</span>
              <NovaIcon name="collapse" size={13} />
            </div>
            <div className="console-toolbar-item">
              <NovaIcon name="jetson" size={15} />
              <span>设备：Jetson Orin Nano</span>
              <NovaIcon name="collapse" size={13} />
            </div>
          </div>
          <div className="console-group">
            <StatusBadge status={health?.ok ? "normal" : "error"} icon={health?.ok ? "check-circle" : "plug-off"} label={health?.ok ? "后端在线" : "后端离线"} size="sm" />
            <ThemeToggle />
            <div className="console-pill"><NovaIcon name="account" size={14} />退出</div>
            <div className={realtimeStatusClass}>{realtimeStatusText} · {formatDate(lastUpdated)}</div>
          </div>
        </div>
      </header>

      <aside className="console-sidebar">
        {navItems.map((item) => (
          <button
            className={activePage === item.id ? "console-nav active" : "console-nav"}
            key={item.id}
            onClick={() => navigatePage(item.id)}
            type="button"
          >
            <span className="console-nav-icon">
              <NovaIcon name={item.icon} size={19} />
            </span>
            <b>{item.label}</b>
          </button>
        ))}
      </aside>

      <main className="console-main">
        <section className="console-process">
          <div className="console-process-state">
            <NovaIcon name="activity-pulse" size={16} />
            {runtimeMainlineSelected
              ? `主链：${captureStatusText} · 推理：${inferenceStatusText}`
              : `采集：${captureStatusText} · 推理：${inferenceStatusText}`}
          </div>
          <button
            className={captureMainRunning ? "console-button danger" : "console-button primary"}
            disabled={busy === "capture" || busy === "stop" || busy === "runtime.start"}
            onClick={() => void toggleCapture()}
            type="button"
          >
            <NovaIcon name={captureMainRunning ? "stop" : "start"} size={16} />
            {captureMainRunning ? (runtimeMainlineSelected ? "停止主链" : "停止采集") : runtimeMainlineSelected ? "启动主链" : "启动采集"}
          </button>
          {!runtimeMainlineSelected && !runtime?.running && capture?.available ? (
            <button
              className="console-button"
              disabled={busy === "runtime.start"}
              onClick={() => void startInferenceThread()}
              type="button"
            >
              恢复推理线程
            </button>
          ) : null}
          <button className="console-button" onClick={exportConfig} type="button">
            <NovaIcon name="export" size={16} />
            导出配置
          </button>
          <button className="console-button" onClick={() => fileInputRef.current?.click()} type="button">
            <NovaIcon name="import" size={16} />
            导入配置
          </button>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={importConfig} />
        </section>

        {mainlineLaunchPending ? (
          <div className="console-info">
            {mainlineLaunchMessage || "主链启动请求已提交，正在等待后端状态确认。"}
          </div>
        ) : null}

        {lastError ? <div className="console-error">{lastError}</div> : null}

        <section className={activePage === "capture" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集状态" value={captureMainRunning ? "运行中" : "未运行"} small={captureBackendLabel || NO_SAMPLE} />
            <Metric title="采集 FPS" value={formatOptionalNumber(captureArrivalFps, 1)} small={deepstreamNvinferSelected ? "nvinfer input" : "appsink arrival"} />
            <Metric title="最新帧龄" value={formatOptionalNumber(latestCaptureAgeMs, 1)} small="ms" />
            <Metric title={deepstreamNvinferSelected ? "Batch 覆盖" : "LatestFrame 覆盖"} value={formatOptionalInteger(captureOverwrittenFrames)} small={deepstreamNvinferSelected ? "batches" : "frames"} />
          </div>

          <div className="console-grid1">
            <div>
              <div className="console-card">
                <SectionTitle title="采集设备" />
                <label>视频设备</label>
                <input value={device} onChange={(event) => setDevice(event.target.value)} />
                <label>数据通路</label>
                <select value="deepstream_nvinfer" disabled aria-label="采集数据通路">
                  <option value="deepstream_nvinfer">DeepStream nvinfer</option>
                </select>
                <div className="console-kv compact-kv">
                  <span>backend</span><b>{captureBackendMode}</b>
                  <span>memory</span><b>{configuredCaptureMemory}</b>
                  <span>preprocess</span><b>{configuredPreprocessBackend}</b>
                  <span>inference</span><b>{configuredInferenceBackend}</b>
                </div>
                <label>采集格式</label>
                <select value={selectedChoice ? choiceId(selectedChoice) : ""} onChange={(event) => setSelectedChoiceId(event.target.value)}>
                  {choices.map((choice) => (
                    <option key={choiceId(choice)} value={choiceId(choice)}>{choiceLabel(choice)}</option>
                  ))}
                  {choices.length === 0 ? <option>请先检测设备能力</option> : null}
                </select>
                <label>缓冲策略</label>
                <select value="latest-frame" disabled>
                  <option value="latest-frame">最新帧优先 / 单槽覆盖</option>
                </select>
                <label>推理画面预览</label>
                <div className="mini-segmented" role="group" aria-label="推理画面预览帧率">
                  {[15, 30].map((fps) => (
                    <button
                      className={previewFps === fps ? "active" : ""}
                      disabled={busy === "limits.stream_fps"}
                      key={fps}
                      onClick={() => void updateConfigField("limits", "stream_fps", fps)}
                      type="button"
                    >
                      {fps}fps
                    </button>
                  ))}
                </div>
                <button className="console-button primary" disabled={busy === "caps"} onClick={refreshCapabilities} type="button">
                  {busy === "caps" ? "检测中..." : "检测设备能力"}
                </button>
              </div>

              <div className="console-card">
                <SectionTitle title="ROI 裁剪" />
                <label>ROI 尺寸</label>
                <CommitNumberControl
                  value={roiSize}
                  min={256}
                  max={640}
                  step={16}
                  digits={0}
                  onCommit={(value) => updateConfigField("roi", "size", nearestRoiSize(value))}
                />
                <div className="mini-segmented roi-size-segmented" role="group" aria-label="ROI 尺寸">
                  {ROI_SIZE_CHOICES.map((size) => (
                    <button
                      className={roiSize === size ? "active" : ""}
                      disabled={busy === "roi.size"}
                      key={size}
                      onClick={() => void updateConfigField("roi", "size", size)}
                      type="button"
                    >
                      {size}
                    </button>
                  ))}
                </div>
                <div className="console-kv compact-kv">
                  <span>源画面</span><b>{sourceWidth > 0 ? `${sourceWidth}x${sourceHeight}` : NO_SAMPLE}</b>
                  <span>ROI 区域</span><b>{sourceWidth > 0 ? `x=${roiX}, y=${roiY}` : NO_SAMPLE}</b>
                  <span>坐标系</span><b>推理 / 预览 / 控制统一 ROI</b>
                </div>
              </div>
            </div>
          </div>

          <div className="console-grid2 diagnostic-grid" data-layer="capture">
            <div className="console-card">
              <SectionTitle title="采集基础状态" />
              <div className="console-kv">
                <span>采集状态</span><b>{captureMainRunning ? "运行中" : "未运行"}</b>
                <span>采集原因</span><b>{captureReason || NO_SAMPLE}</b>
                <span>采集设备</span><b>{capture?.device || configuredCaptureDevice || NO_SAMPLE}</b>
                <span>采集后端</span><b>{captureBackendLabel || NO_SAMPLE}</b>
                <span>输入格式</span><b>{displayCaptureProfile?.pixel_format || NO_SAMPLE}</b>
                <span>输入分辨率</span><b>{displayCaptureProfile ? `${displayCaptureProfile.width}x${displayCaptureProfile.height}` : NO_SAMPLE}</b>
                <span>输入帧率</span><b>{displayCaptureProfile ? `${displayCaptureProfile.fps.toFixed(1)} FPS` : NO_SAMPLE}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="ROI 与输出帧" />
              <div className="console-kv">
                <span>ROI 原点</span><b>{sourceWidth > 0 ? `${roiX}, ${roiY}` : NO_SAMPLE}</b>
                <span>ROI 尺寸</span><b>{`${roiSize}x${roiSize}`}</b>
                <span>采集输出尺寸</span><b>{latestCaptureOutputWidth !== null && latestCaptureOutputHeight !== null && latestCaptureOutputWidth > 0 && latestCaptureOutputHeight > 0 ? `${latestCaptureOutputWidth}x${latestCaptureOutputHeight}` : NO_SAMPLE}</b>
                <span>输出像素格式</span><b>{latestCaptureOutputFormat || NO_SAMPLE}</b>
                <span>内存类型</span><b>{latestCaptureOutputMemory || NO_SAMPLE}</b>
                <span>{deepstreamNvinferSelected ? "输出契约" : "appsink caps"}</span><b>{deepstreamNvinferSelected ? "NVMM NV12" : readString(captureStatistics.appsink_caps, "") || NO_SAMPLE}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="最新帧状态" />
              <div className="console-kv">
                <span>最新 frame_id</span><b>{formatOptionalInteger(latestCaptureFrameId)}</b>
                <span>最新 generation</span><b>{formatOptionalInteger(latestCaptureGeneration)}</b>
                <span>最新帧时间戳</span><b>{latestCaptureTsNs !== null && latestCaptureTsNs > 0 ? `${Math.trunc(latestCaptureTsNs)} ns` : NO_SAMPLE}</b>
                <span>时间戳来源</span><b>{latestCaptureTimestampSource || NO_SAMPLE}</b>
                <span>最新帧龄</span><b>{formatOptionalNumber(latestCaptureAgeMs, 2, "ms")}</b>
                <span>帧到达间隔</span><b>{formatOptionalNumber(captureFramePeriodMs, 2, "ms")}</b>
                <span>{deepstreamNvinferSelected ? "DetectionBatch 覆盖次数" : "LatestFrame 覆盖次数"}</span><b>{formatOptionalInteger(captureOverwrittenFrames)}</b>
                <span>{deepstreamNvinferSelected ? "stale 拒绝数" : "采集丢帧数"}</span><b>{formatOptionalInteger(captureDroppedFrames)}</b>
                <span>已发布 / 已取得</span><b>{`${formatOptionalInteger(capturePublishedFrames)} / ${formatOptionalInteger(deepstreamNvinferSelected ? deepstreamMailbox.acquired_batches : latestFrameBroker.acquired_frames)}`}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="采集性能" />
              <div className="console-kv">
                <span>采集 FPS</span><b>{formatOptionalNumber(captureStatistics.capture_fps, 1, "FPS")}</b>
                <span>{deepstreamNvinferSelected ? "nvinfer 输入 FPS" : "appsink 到达 FPS"}</span><b>{formatOptionalNumber(captureArrivalFps, 1, "FPS")}</b>
                <span>采集抖动</span><b>{NOT_INSTRUMENTED}</b>
                <span>采集到应用层延迟</span><b>{latestCaptureTimestampSource === "userspace_monotonic_receive" ? UNAVAILABLE : NOT_INSTRUMENTED}</b>
                <span>采集等待调用</span><b>{formatOptionalNumber(capture?.capture_wait_ms, 2, "ms")}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "infer" ? "console-page active" : "console-page"}>
          <div className="console-tabs">
            <div className="console-tab active"><NovaIcon name="model-verify" size={15} />配置1</div>
            <div className="console-tab"><NovaIcon name="ai-model" size={15} />配置2</div>
            <div className="console-tab"><NovaIcon name="ai-model" size={15} />配置3</div>
          </div>
          <div className="console-metrics">
            <Metric title="推理 FPS" value={formatNumber(statistics?.inference_fps, 1)} small="FPS" />
            <Metric title="推理状态" value={inferenceRan ? (inferenceAvailable ? "已执行" : "执行失败") : "未执行"} small={selectedRuntimeBackend || NO_SAMPLE} />
            <Metric title="推理总耗时" value={formatOptionalNumber(inferenceTotalMs, 2)} small="ms" />
            <Metric title="NMS 后检测" value={formatOptionalInteger(inferenceNmsDetectionCount)} small="detections" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <SectionTitle title="模型设置" />
              <div className="console-action-row">
                <button
                  className="console-button secondary"
                  disabled={busy !== null}
                  onClick={() => void refreshModelCatalog()}
                  type="button"
                >
                  <NovaIcon name="refresh" size={15} />
                  {busy === "model.refresh" ? "刷新中..." : "刷新模型"}
                </button>
                <button
                  className="console-button"
                  disabled={busy !== null}
                  onClick={() => void rescanModelCatalog()}
                  type="button"
                >
                  <NovaIcon name="model-verify" size={15} />
                  {busy === "model.scan" ? "校验中..." : "强制重新校验"}
                </button>
              </div>
              {modelCatalogMessage ? <div className="model-switch-note good">{modelCatalogMessage}</div> : null}
              <div className="model-catalog-heading">
                <label>模型目录</label>
                <span>{modelCatalogDirectoryCount} 个文件夹 · {modelCatalogModelCount} 个模型</span>
              </div>
              {modelCatalogLoading ? (
                <div className="model-catalog-placeholder">正在递归读取 models 目录...</div>
              ) : modelCatalog && modelCatalog.children.length > 0 ? (
                <ModelCatalogTree
                  root={modelCatalog}
                  expandedDirectories={expandedModelDirectories}
                  selectedPath={selectedModelCatalogPath}
                  activeArtifactId={artifact?.id ?? null}
                  onToggleDirectory={toggleModelDirectory}
                  onSelectModel={selectModelFromCatalog}
                />
              ) : (
                <div className="model-catalog-placeholder">
                  models 目录中没有 .onnx 或 .engine 模型。
                </div>
              )}
              <div className="model-primary-summary">
                <span>{readString(runtime?.inference?.selected, "按模型后缀自动选择")}</span>
                <span>{displayedInputShape ? `运行输入 ${displayedInputShape}` : "等待模型输入信息"}</span>
                <span>{selectedSwitchArtifact?.kind ? selectedSwitchArtifact.kind.toUpperCase() : "无可用产物"}</span>
              </div>
              <div className="model-active-summary">
                <span>当前运行模型</span>
                <b>{activeModelName}</b>
                <small>{activeArtifactLabel}</small>
              </div>
              {modelSwitchMessage || lastModelSwitchError ? (
                <div className={lastModelSwitchError && !modelSwitchMessage ? "model-switch-note bad" : "model-switch-note good"}>
                  {modelSwitchMessage || `上次切换失败：${lastModelSwitchError}`}
                </div>
              ) : null}
              <button
                className="console-button primary console-full-button"
                disabled={busy !== null || selectedModelProjectId === "" || selectedSwitchArtifact === null}
                onClick={switchModel}
                type="button"
              >
                {busy === "model.switch"
                  ? "自动配置并切换中..."
                  : selectedSwitchArtifact?.status === "pending" || selectedSwitchArtifact?.status === "failed"
                    ? "自动配置并加载模型"
                    : "切换模型"}
              </button>
              {selectedSwitchArtifact?.status === "pending" || selectedSwitchArtifact?.status === "failed" ? (
                <p className="console-field-hint">
                  后端将直接读取 Engine 的 I/O、Shape 和数据类型，自动推导类别契约并生成唯一 DeepStream manifest。
                </p>
              ) : null}
              {selectedSwitchArtifact === null && blockedSwitchArtifacts.length > 0 ? (
                <div className="model-switch-note bad">
                  模型产物不可切换：{blockedSwitchArtifacts.map((item) => `${item.path} (${item.status})`).join("，")}。请修复模型或 manifest 后强制重新校验。
                </div>
              ) : null}
              <label>置信度阈值</label>
              <CommitNumberControl
                value={confidence}
                min={0}
                max={1}
                step={0.01}
                digits={2}
                onCommit={(value) => updateConfigField("inference", "confidence_threshold", value)}
              />
              <label>NMS 阈值</label>
              <CommitNumberControl
                value={nms}
                min={0}
                max={1}
                step={0.01}
                digits={2}
                onCommit={(value) => updateConfigField("inference", "nms_threshold", value)}
              />
              <div className="class-profile-toolbar">
                <span>
                  <b>类别设置</b>
                  <small>名称、优先顺序与框内瞄点由当前模型类别表共同管理。</small>
                </span>
                <select
                  aria-label="检测类别配置"
                  value={activeDetectionProfile}
                  onChange={(event) => void updateConfigField("inference", "detection_class_profile", event.target.value)}
                >
                  {(detectionProfileNames.length > 0 ? detectionProfileNames : ["default"]).map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
              <div className="class-default-aim">
                <span>
                  <b>默认瞄点高度</b>
                  <small>从 bbox 顶部向下的比例；未单独设置的类别都使用此值。</small>
                </span>
                <CommitNumberControl
                  value={aimYRatio}
                  min={0}
                  max={1}
                  step={0.01}
                  digits={2}
                  onCommit={(value) => updateControlGroupField("aim", "y_ratio", value)}
                />
                <div className="class-aim-preview large" aria-hidden="true">
                  <i style={{ top: `${aimYRatio * 100}%` }} />
                </div>
              </div>
              <div className="class-editor" role="table" aria-label="模型类别与瞄点设置">
                <div className="class-editor-head" role="row">
                  <span>ID</span><span>类别名称</span><span>优先级</span><span>瞄点高度</span><span>目标筛选</span>
                </div>
                {classEditorIds.map((classId) => {
                  const configuredName = detectionClasses[classId] ?? "";
                  const displayName = configuredName ? classDisplayName(configuredName, classId) : "";
                  const priorityIndex = classPriorityIds.indexOf(classId);
                  const enabled = activeDetectionClass === "all" || activeDetectionClass === String(classId);
                  return (
                    <div className="class-editor-row" role="row" key={`class-editor-${classId}`}>
                      <b className="class-id">cls {classId}</b>
                      <InlineTextControl
                        ariaLabel={`cls ${classId} 类别名称`}
                        value={displayName}
                        placeholder={`未知类别（cls ${classId}）`}
                        onCommit={(value) => updateDetectionClassName(classId, value)}
                      />
                      <div className="class-priority-control">
                        <b>{priorityIndex >= 0 ? priorityIndex + 1 : "—"}</b>
                        <button
                          aria-label={`提高 cls ${classId} 优先级`}
                          disabled={priorityIndex === 0}
                          type="button"
                          onClick={() => void moveClassPriority(classId, -1)}
                        >↑</button>
                        <button
                          aria-label={`降低 cls ${classId} 优先级`}
                          disabled={priorityIndex < 0 || priorityIndex === classPriorityIds.length - 1}
                          type="button"
                          onClick={() => void moveClassPriority(classId, 1)}
                        >↓</button>
                      </div>
                      <ClassAimRatioControl
                        classId={classId}
                        defaultRatio={aimYRatio}
                        overrideRatio={activeClassAimRatios[String(classId)]}
                        onCommit={(value) => updateClassAimRatio(classId, value)}
                      />
                      <button
                        className={enabled ? "class-enable active" : "class-enable"}
                        type="button"
                        onClick={() => void updateConfigField(
                          "inference",
                          "detection_class_filter",
                          activeDetectionClass === String(classId) ? "all" : String(classId)
                        )}
                      >
                        {activeDetectionClass === "all"
                          ? "仅选此类"
                          : activeDetectionClass === String(classId)
                            ? "恢复全部"
                            : "切换此类"}
                      </button>
                    </div>
                  );
                })}
              </div>
              <p className="console-field-hint">
                未知 class id 会显示为“未知类别（cls N）”并自动使用默认瞄点；编辑名称后即加入当前类别配置。
              </p>
              <details className="model-debug-details">
                <summary>工程调试详情</summary>
                <label>模型版本</label>
                <select
                  value={selectedModelVersionId}
                  onChange={(event) => {
                    const nextVersionId =
                      event.target.value === "" ? "" : Number(event.target.value);
                    setSelectedModelVersionId(
                      typeof nextVersionId === "number" && Number.isFinite(nextVersionId)
                        ? nextVersionId
                        : ""
                    );
                    setSelectedModelArtifactId("");
                    setModelArtifacts([]);
                  }}
                >
                  {modelVersions.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.version === "default" ? "自动发现版本" : item.version} · 输入 {item.input_shape}
                    </option>
                  ))}
                  {modelVersions.length === 0 ? <option value="">暂无版本</option> : null}
                </select>
                <label>推理产物</label>
                <select
                  value={selectedSwitchArtifact?.id ?? ""}
                  onChange={(event) => setSelectedModelArtifactId(Number(event.target.value))}
                >
                  {sortedSwitchableArtifacts.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.kind} · {item.path} · {formatModelSizeMb(item.size_bytes)}{item.status === "pending" ? " · 待验证" : ""}
                    </option>
                  ))}
                  {sortedSwitchableArtifacts.length === 0 ? <option value="">暂无可验证的 ONNX / engine 产物</option> : null}
                </select>
                <div className="model-debug-grid">
                  <span>推荐产物</span><b>{preferredSwitchArtifact ? `${preferredSwitchArtifact.kind} · ${preferredSwitchArtifact.path}` : "-"}</b>
                  <span>当前产物</span><b>{selectedSwitchArtifact ? `${selectedSwitchArtifact.kind} · ${selectedSwitchArtifact.path}` : "-"}</b>
                  <span>运行输入</span><b>{runtimeInputShape || "-"}</b>
                  <span>登记输入</span><b>{registeredInputShape || "-"}</b>
                  <span>运行输出</span><b>{deepstreamModelOutputSummary}</b>
                  <span>输出层</span><b>{readString(runtimeModelOutput.name, "-")}</b>
                  <span>运行类别</span><b>{formatNumber(runtimeModelOutput.class_count, 0)}</b>
                  <span>解析器</span><b>{runtimePostprocessParser}</b>
                  <span>运行置信度</span><b>{formatNumber(runtimePostprocessConfidence, 2)}</b>
                  <span>运行 NMS</span><b>{formatNumber(runtimePostprocessNms, 2)}</b>
                  <span>ROI</span><b>{roiSize} · 自动缩放</b>
                  <span>后端</span><b>{readString(runtime?.inference?.selected, "auto")}</b>
                  <span>类别数量</span><b>{String(version?.classes.length ?? 0)}</b>
                  <span className="wide">类别名</span><b className="wide">{deepstreamClassNamesSummary}</b>
                </div>
              </details>
            </div>
            <div className="console-card">
              <SectionTitle title="ROI 输入预览" />
              <PreviewFrame
                enabled={activePage === "infer" && previewEnabled}
                imageAvailable={previewImageAvailable}
                unavailableReason={previewUnavailableReason}
                runtime={runtime}
                roiSize={roiSize}
              />
              <div className="console-kv">
                <span>画面阶段</span><b>{deepstreamNvinferSelected ? "nvinfer 前 NVMM ROI" : "推理输入 ROI"}</b>
                <span>ROI 输入</span><b>{`${roiInputWidth || "-"}x${roiInputHeight || "-"}`}</b>
                <span>模型输入</span><b>{modelInputWidth && modelInputHeight ? `${modelInputWidth}x${modelInputHeight}` : "-"}</b>
                <span>压缩倍率</span><b>{inputDownscaleFactor ? `${formatNumber(inputDownscaleFactor, 2)}x` : "-"}</b>
                <span>有效像素</span><b>{inputPixelRatio ? formatPercent(inputPixelRatio, 1) : "-"}</b>
                <span>输入资源</span><b>{readString(inferenceTrace.input_resource_memory, "") || NO_SAMPLE}</b>
                <span>坐标空间</span><b>{readString(inferenceTrace.detection_coordinate_space, "") || NO_SAMPLE}</b>
              </div>
              {inputDensityWarning ? (
                <div className="inference-density-warning">
                  ROI 正在被压缩到模型输入，远距离小目标可能丢失。建议让 ROI 尺寸接近模型输入，或切换到更大输入尺寸的模型。
                </div>
              ) : null}
            </div>
          </div>
          <div className="console-grid2 diagnostic-grid" data-layer="inference">
            <div className="console-card">
              <SectionTitle title="推理调度" />
              <div className="console-kv">
                <span>推理状态</span><b>{inferenceRan ? (inferenceAvailable ? "已执行" : "执行失败") : "未执行"}</b>
                <span>推理原因</span><b>{inferenceReason || NO_SAMPLE}</b>
                <span>nvinfer 输入帧</span><b>{formatOptionalInteger(deepstreamInputFrames)}</b>
                <span>nvinfer 输出 Buffer</span><b>{formatOptionalInteger(deepstreamOutputBuffers)}</b>
                <span>BatchMeta Buffer</span><b>{formatOptionalInteger(deepstreamBatchMetaBuffers)}</b>
                <span>FrameMeta 帧</span><b>{formatOptionalInteger(deepstreamFrameMetaFrames)}</b>
                <span>Parser 调用</span><b>{formatOptionalInteger(deepstreamParserStatus.decode_calls)}</b>
                <span>Parser 失败</span><b>{formatOptionalInteger(deepstreamParserStatus.parse_failures)}</b>
                <span>Parser 错误码</span><b>{formatOptionalInteger(deepstreamParserStatus.last_error_code)}</b>
                <span>Buffer PTS 匹配</span><b>{formatOptionalInteger(runtimeInference.timestamp_buffer_pts_matches)}</b>
                <span>FrameMeta PTS 匹配</span><b>{formatOptionalInteger(runtimeInference.timestamp_frame_meta_pts_matches)}</b>
                <span>顺序回退匹配</span><b>{formatOptionalInteger(runtimeInference.timestamp_ordered_fallback_matches)}</b>
                <span>PTS 关联失败</span><b>{formatOptionalInteger(runtimeInference.timestamp_correlation_misses)}</b>
                <span>当前推理 frame_id</span><b>{formatOptionalInteger(inferenceFrameId)}</b>
                <span>Acquire generation</span><b>{formatOptionalInteger(inferenceAcquiredGeneration)}</b>
                <span>Batch generation</span><b>{formatOptionalInteger(inferenceBatchGeneration)}</b>
                <span>推理开始帧龄</span><b>{formatOptionalNumber(inferenceStartAgeMs, 2, "ms")}</b>
                <span>推理结束帧龄</span><b>{formatOptionalNumber(inferenceEndAgeMs, 2, "ms")}</b>
                <span>推理期间到达新帧数</span><b>{formatOptionalInteger(inferencePublishedSinceAcquire)}</b>
                <span>结束时 generation 差</span><b>{formatOptionalInteger(inferenceGenerationLag)}</b>
                <span>结束时 frame_id 差</span><b>{formatOptionalInteger(inferenceFrameIdLag)}</b>
                <span>过期结果丢弃次数</span><b>{formatOptionalInteger(statistics?.stale_drop_count)}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="模型输入" />
              <div className="console-kv">
                <span>模型名称</span><b>{activeModelName || NO_SAMPLE}</b>
                <span>推理后端</span><b>{selectedRuntimeBackend || NO_SAMPLE}</b>
                <span>ROI 输入尺寸</span><b>{roiInputWidth > 0 && roiInputHeight > 0 ? `${roiInputWidth}x${roiInputHeight}` : NO_SAMPLE}</b>
                <span>模型输入尺寸</span><b>{modelInputWidth > 0 && modelInputHeight > 0 ? `${modelInputWidth}x${modelInputHeight}` : displayedInputShape || NO_SAMPLE}</b>
                <span>输入数据类型</span><b>{inferenceInputDtype || NO_SAMPLE}</b>
                <span>输入布局</span><b>{inferenceInputLayout || NOT_INSTRUMENTED}</b>
                <span>输入准备耗时</span><b>{formatOptionalNumber(inferencePreprocessMs, 3, "ms")}</b>
                <span>CUDA 上传耗时</span><b>{formatOptionalNumber(inferenceUploadMs, 3, "ms")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="TensorRT 执行" />
              <div className="console-kv">
                <span>TensorRT enqueue 耗时</span><b>{formatOptionalNumber(inferenceEnqueueMs, 3, "ms")}</b>
                <span>GPU 核心执行耗时</span><b>{NOT_INSTRUMENTED}</b>
                <span>CUDA stream 同步等待</span><b>{formatOptionalNumber(inferenceSyncWaitMs, 3, "ms")}</b>
                <span>推理核心耗时</span><b>{NOT_INSTRUMENTED}</b>
                <span>推理线程总耗时</span><b>{formatOptionalNumber(inferenceTotalMs, 3, "ms")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="输出与后处理" />
              <div className="console-kv">
                <span>输出 Tensor 名称</span><b>{inferenceOutputName || NO_SAMPLE}</b>
                <span>输出形状</span><b>{inferenceOutputShape}</b>
                <span>输出布局 / 解析器</span><b>{inferenceParser || NO_SAMPLE}</b>
                <span>置信度阈值</span><b>{formatOptionalNumber(runtimePostprocessConfidence, 2)}</b>
                <span>NMS IoU 阈值</span><b>{formatOptionalNumber(runtimePostprocessNms, 2)}</b>
                <span>解析前候选数</span><b>{formatOptionalInteger(inferenceRawCandidateCount)}</b>
                <span>置信度过滤后候选数</span><b>{formatOptionalInteger(inferenceThresholdCandidateCount)}</b>
                <span>NMS 后检测数</span><b>{formatOptionalInteger(inferenceNmsDetectionCount)}</b>
                <span>最高检测置信度</span><b>{formatOptionalNumber(inferenceHighestConfidence, 3)}</b>
                <span>后处理耗时</span><b>{formatOptionalNumber(inferencePostprocessMs, 3, "ms")}</b>
                <span>DetectionBatch 状态</span><b>{!inferenceRan ? NO_SAMPLE : !inferenceBatchPublished ? "未发布" : inferenceBatchStale ? "已过期" : "可消费"}</b>
                <span>DetectionBatch published</span><b>{!inferenceRan ? NO_SAMPLE : inferenceBatchPublished ? "是" : "否"}</b>
                <span>DetectionBatch age</span><b>{formatOptionalNumber(inferenceResultAgeMs, 2, "ms")}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "control" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="控制状态" value={controlHasSample ? readString(control.global_state, "已计算") : "未执行"} small={controlNoSendReason || NO_SAMPLE} />
            <Metric title="目标链路" value={targetPipelineCode || NO_SAMPLE} small={targetPipelineStage || NO_SAMPLE} />
            <Metric title="当前 Track" value={formatOptionalInteger(controlTrackId)} small={readString(target.class_name, "") || "target"} />
            <Metric title="预测误差" value={formatOptionalNumber(predictedErrorDistancePx, 1)} small="px" />
            <Metric title="实际发送" value={execution.sent === true ? formatPoint(controlActualDx, controlActualDy, 0) : NO_SAMPLE} small="counts" />
          </div>
          <div className="console-grid2 diagnostic-grid" data-layer="control">
            <div className="console-card">
              <SectionTitle title="目标选择" />
              <div className="console-kv">
                <span>控制状态</span><b>{controlHasSample ? readString(control.global_state, "已计算") : "未执行"}</b>
                <span>控制原因</span><b>{readString(control.reason, readString(control.selection_reason, "")) || NO_SAMPLE}</b>
                <span>阻断阶段</span><b>{targetPipelineStage || NO_SAMPLE}</b>
                <span>诊断代码</span><b>{targetPipelineCode || NO_SAMPLE}</b>
                <span>诊断信息</span><b>{targetPipelineMessage || NO_SAMPLE}</b>
                <span>检测数量</span><b>{formatOptionalInteger(inferenceTrace.mapped_detections)}</b>
                <span>解码 / 阈值 / NMS</span><b>{`${formatOptionalInteger(targetPipelineCounts.decode_raw_candidates)} / ${formatOptionalInteger(targetPipelineCounts.threshold_candidates)} / ${formatOptionalInteger(targetPipelineCounts.nms_detections)}`}</b>
                <span>基础 / 关联 / CONFIRMED / FOV 内</span><b>{`${formatOptionalInteger(targetPipelineCounts.basic_candidates)} / ${formatOptionalInteger(targetPipelineCounts.association_candidates)} / ${formatOptionalInteger(targetPipelineCounts.tracker_active)} / ${formatOptionalInteger(targetPipelineCounts.inside_fov)}`}</b>
                <span>过滤原因</span><b>{targetPipelineRejections || NO_SAMPLE}</b>
                <span>选择 FOV 中心</span><b>{formatPoint(selectionCenter.x, selectionCenter.y, 1, "px")}</b>
                <span>选择 FOV 半径</span><b>{formatOptionalNumber(controlCandidateFilter.selection_radius_px, 1, "px")}</b>
                <span>被拒绝瞄点</span><b>{formatPoint(firstRejectedControlCandidate.aim_x, firstRejectedControlCandidate.aim_y, 1, "px")}</b>
                <span>瞄点距中心</span><b>{formatOptionalNumber(firstRejectedControlCandidate.distance_px, 1, "px")}</b>
                <span>候选目标数量</span><b>{formatOptionalInteger(controlCandidateCount)}</b>
                <span>最终选择数量</span><b>{controlHasTarget ? "1" : controlHasSample ? "0" : NO_SAMPLE}</b>
                <span>当前 track_id</span><b>{formatOptionalInteger(controlTrackId)}</b>
                <span>目标类别</span><b>{readString(target.class_name, "") || NO_SAMPLE}</b>
                <span>目标置信度</span><b>{formatOptionalNumber(target.score, 3)}</b>
                <span>Track quality</span><b>{formatOptionalNumber(selectedTrackDebug.track_quality, 3)}</b>
                <span>类别 / 质量分</span><b>{formatPoint(control.class_score ?? target.class_score, control.quality_score ?? target.quality_score, 3)}</b>
                <span>距离 / 综合分</span><b>{formatPoint(control.distance_score ?? target.distance_score, control.selection_score ?? target.selection_score, 3)}</b>
                <span>目标选择状态</span><b>{readString(control.selector_state, "") || NO_SAMPLE}</b>
                <span>目标选择原因</span><b>{readString(control.selection_reason, "") || NO_SAMPLE}</b>
                <span>目标框坐标</span><b>{controlHasTarget ? `${formatPoint(target.x1, target.y1, 1)} -> ${formatPoint(target.x2, target.y2, 1)}` : NO_SAMPLE}</b>
                <span>目标框中心</span><b>{formatPoint(target.box_cx ?? target.cx, target.box_cy ?? target.cy, 1, "px")}</b>
                <span>Tracker 关联</span><b>{readString(trackerRuntimeDebug.association_algorithm, "") || NO_SAMPLE}</b>
                <span>关联输入 / 截断</span><b>{`${formatOptionalInteger(trackerRuntimeDebug.input_candidates)} / ${formatOptionalInteger(trackerRuntimeDebug.association_candidates_dropped)}`}</b>
                <span>轨迹 / 检测上限</span><b>{`${formatOptionalInteger(trackerRuntimeDebug.max_active_tracks)} / ${formatOptionalInteger(trackerRuntimeDebug.max_detections_for_association)}`}</b>
                <span>CONFIRMED / TENTATIVE / LOST</span><b>{`${formatOptionalInteger(trackerRuntimeDebug.confirmed_tracks ?? trackerRuntimeDebug.active_tracks)} / ${formatOptionalInteger(trackerRuntimeDebug.tentative_tracks)} / ${formatOptionalInteger(trackerRuntimeDebug.lost_track_count)}`}</b>
                <span>本轮恢复 track</span><b>{Array.isArray(trackerRuntimeDebug.restored_track_ids) && trackerRuntimeDebug.restored_track_ids.length > 0 ? trackerRuntimeDebug.restored_track_ids.join(", ") : NO_SAMPLE}</b>
                <span>Predict / Matrix</span><b>{formatPoint(trackerTiming.tracker_predict_us, trackerTiming.association_matrix_us, 2, "us")}</b>
                <span>Hungarian / Update</span><b>{formatPoint(trackerTiming.hungarian_us, trackerTiming.tracker_update_us, 2, "us")}</b>
                <span>Tracker total</span><b>{formatOptionalNumber(trackerTiming.tracker_total_us, 2, "us")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="瞄准点与预测" />
              <div className="console-kv">
                <span>原始瞄准点</span><b>{formatPoint(observedAimX, observedAimY, 1, "px")}</b>
                <span>aim_y_ratio</span><b>{formatOptionalNumber(control.aim_y_ratio ?? rawAimDebug.y_ratio, 2)}</b>
                {dualPhaseActive ? (
                  <>
                    <span>短窗位置数</span><b>{formatOptionalInteger(controlPipeline.history_position_count)}</b>
                    <span>三段速度 px/ms</span><b>{`${formatOptionalNumber(controlPipeline.velocity_1, 3)} / ${formatOptionalNumber(controlPipeline.velocity_2, 3)} / ${formatOptionalNumber(controlPipeline.velocity_3, 3)}`}</b>
                    <span>中位 / EMA 速度</span><b>{formatPoint(controlPipeline.median_velocity, controlPipeline.filtered_velocity, 3, "px/ms")}</b>
                    <span>速度离散度</span><b>{formatOptionalNumber(controlPipeline.velocity_spread, 3, "px/ms")}</b>
                    <span>运动可信度</span><b>{formatPercent(controlPipeline.motion_confidence, 1)}</b>
                    <span>平均帧间隔</span><b>{formatOptionalNumber(controlPipeline.reference_dt_ms, 2, "ms")}</b>
                    <span>前瞻帧数</span><b>{formatOptionalNumber(controlPipeline.prediction_lead_frames, 2, " 帧")}</b>
                    <span>原始 / 安全预测</span><b>{formatPoint(controlPipeline.prediction_raw_offset_x, controlPipeline.prediction_safe_offset_x, 2, "px")}</b>
                    <span>预测允许上限</span><b>{formatOptionalNumber(controlPipeline.prediction_allowed_cap_x, 2, "px")}</b>
                    <span>预测后瞄准点</span><b>{formatPoint(predictedAimX, predictedAimY, 1, "px")}</b>
                  </>
                ) : (
                  <>
                    <span>控制瞄准点</span><b>{formatPoint(predictedAimX, predictedAimY, 1, "px")}</b>
                    <span>位置预测</span><b>不参与当前控制算法</b>
                  </>
                )}
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="误差与角度" />
              <div className="console-kv">
                <span>屏幕中心</span><b>{formatPoint(controlCenterX, controlCenterY, 1, "px")}</b>
                <span>observed error px</span><b>{formatPoint(controlPipeline.observed_error_x_px, controlPipeline.observed_error_y_px, 2, "px")}</b>
                <span>predicted error px</span><b>{formatPoint(predictedErrorXPx, predictedErrorYPx, 2, "px")}</b>
                <span>observed error rad</span><b>{formatPoint(controlPipeline.observed_error_x_rad, controlPipeline.observed_error_y_rad, 6, "rad")}</b>
                <span>predicted error rad</span><b>{formatPoint(controlPipeline.predicted_error_x_rad, controlPipeline.predicted_error_y_rad, 6, "rad")}</b>
                <span>误差距离</span><b>{formatOptionalNumber(predictedErrorDistancePx, 2, "px")}</b>
                <span>控制 dt</span><b>{controlMeasurementDtS === null ? NO_SAMPLE : `${(controlMeasurementDtS * 1000).toFixed(3)} ms`}</b>
                <span>焦距 X / Y</span><b>{formatPoint(controlPipeline.focal_x_px, controlPipeline.focal_y_px, 2, "px")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="控制器输出" />
              <div className="console-kv">
                <span>控制模式</span><b>{controlModeLabel}</b>
                <span>移动策略</span><b>{readString(controlPipeline.movement_strategy, "") || NO_SAMPLE}</b>
                {dualPhaseActive ? (
                  <>
                    <span>FAR / NEAR</span><b>{readString(controlPipeline.mode, readString(controlPipeline.control_mode, "")) || NO_SAMPLE}</b>
                    <span>完整修正 counts</span><b>{formatPoint(controlPipeline.full_error_counts_x, controlPipeline.full_error_counts_y, 2)}</b>
                    <span>Atan 浮点需求</span><b>{formatPoint(controlPipeline.float_demand_x, controlPipeline.float_demand_y, 2)}</b>
                    <span>整数输出</span><b>{formatPoint(controlPipeline.integer_command_x, controlPipeline.integer_command_y, 0, "counts")}</b>
                    <span>量化余量</span><b>{formatPoint(controlPipeline.quantizer_residual_x, controlPipeline.quantizer_residual_y, 3, "counts")}</b>
                  </>
                ) : (
                  <>
                    <span>Kp X / Y</span><b>{readString(controlPipeline.control_mode, controlMode) === "calibrated_angular" ? formatPoint(calibratedKpX, calibratedKpY, 2) : NO_SAMPLE}</b>
                    <span>Kd X / Y</span><b>{readString(controlPipeline.control_mode, controlMode) === "calibrated_angular" ? formatPoint(calibratedKdX, calibratedKdY, 2) : NO_SAMPLE}</b>
                    <span>D 原始值</span><b>{formatPoint(controlPipeline.d_raw_x_rad_s, controlPipeline.d_raw_y_rad_s, 5, "rad/s")}</b>
                    <span>D EMA 值</span><b>{formatPoint(controlPipeline.d_ema_x_rad_s, controlPipeline.d_ema_y_rad_s, 5, "rad/s")}</b>
                    <span>P 项输出</span><b>{formatPoint(controlPipeline.p_x_rad, controlPipeline.p_y_rad, 6, "rad")}</b>
                    <span>D 项输出</span><b>{formatPoint(controlPipeline.d_x_rad, controlPipeline.d_y_rad, 6, "rad")}</b>
                    <span>角度控制量</span><b>{formatPoint(controlPipeline.requested_output_x_rad, controlPipeline.requested_output_y_rad, 6, "rad")}</b>
                    <span>角度限幅后</span><b>{formatPoint(controlPipeline.limited_output_x_rad, controlPipeline.limited_output_y_rad, 6, "rad")}</b>
                    <span>理论 counts</span><b>{formatPoint(controlPipeline.theoretical_counts_x_float, controlPipeline.theoretical_counts_y_float, 2)}</b>
                    <span>模式限幅后 counts</span><b>{formatPoint(controlPipeline.mode_limited_counts_x_float, controlPipeline.mode_limited_counts_y_float, 2)}</b>
                    <span>压枪状态</span><b>{controlPipeline.recoil_active === true ? "输出中" : recoilEnabled ? "等待左键或延迟" : "关闭"}</b>
                    <span>到位状态</span><b>{readString(controlPipeline.arrival_state, "") || NO_SAMPLE}</b>
                    <span>到位限制后 counts</span><b>{formatPoint(controlPipeline.deadzone_limited_counts_x_float, controlPipeline.deadzone_limited_counts_y_float, 2)}</b>
                    <span>Slew 后 counts</span><b>{formatPoint(controlPipeline.slew_limited_counts_x_float, controlPipeline.slew_limited_counts_y_float, 2)}</b>
                    <span>累计余量 counts</span><b>{formatPoint(controlPipeline.residual_x_counts, controlPipeline.residual_y_counts, 2)}</b>
                  </>
                )}
                <span>触发持续 / 启动延迟</span><b>{`${formatOptionalNumber(control.trigger_hold_ms, 1, "ms")} / ${formatOptionalNumber(control.trigger_activation_delay_ms, 1, "ms")}`}</b>
                <span>控制预算</span><b>{formatPoint(control.dx, control.dy, 0, "counts")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title={dualPhaseActive ? "MouseCommandExecutor 与设备发送" : "Scheduler 与设备发送"} />
              <div className="console-kv">
                <span>触发状态</span><b>{control.trigger_active === true ? "按下" : control.trigger_active === false ? "未按下" : NO_SAMPLE}</b>
                <span>是否允许发包</span><b>{control.will_emit === true ? "是" : control.will_emit === false ? "否" : NO_SAMPLE}</b>
                <span>不发包原因</span><b>{controlNoSendReason || NO_SAMPLE}</b>
                <span>本轮控制意图</span><b>{formatPoint(control.dx, control.dy, 0, "counts")}</b>
                <span>{dualPhaseActive ? "发送语义" : "待执行 counts"}</span><b>{dualPhaseActive ? "每观测至多一次 move" : formatPoint(schedulerStatus.pending_dx, schedulerStatus.pending_dy, 0, "counts")}</b>
                <span>本次发送 counts</span><b>{execution.sent === true ? formatPoint(controlActualDx, controlActualDy, 0, "counts") : NO_SAMPLE}</b>
                <span>{dualPhaseActive ? "跨帧剩余计划" : "剩余 pending steps"}</span><b>{dualPhaseActive ? "无" : formatOptionalInteger(schedulerStatus.pending_steps)}</b>
                <span>inflight 估计</span><b>{NOT_INSTRUMENTED}</b>
                <span>发送频率</span><b>{NOT_INSTRUMENTED}</b>
                <span>设备发送耗时</span><b>{controlSendDuration}</b>
                <span>最后发送时间</span><b>{formatOptionalInteger(execution.device_send_end_ts_ns ?? executionMeta.device_send_end_ts_ns)}</b>
                {!dualPhaseActive ? <><span>旧计划截断次数</span><b>{formatOptionalInteger(schedulerStatus.cancelled_pending)}</b><span>最近取消原因</span><b>{readString(schedulerStatus.last_cancel_reason, "") || NO_SAMPLE}</b></> : null}
                <span>设备连接</span><b>{kmnetConnected ? "已连接" : "未连接"}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "params" || activePage === "control-test" ? "console-page active" : "console-page"}>
          {activePage === "params" ? (
          <>
            <div className="console-metrics">
              <Metric title="控制模式" value={controlModeLabel} small="单选策略" />
              <Metric title="触发方式" value={triggerModeLabel(triggerMode)} small="trigger" />
              <Metric title="默认瞄点 Y" value={aimYRatio.toFixed(2)} small={`${Object.keys(activeClassAimRatios).length} 个类别覆盖`} />
              <Metric title="位置预测" value={dualPhaseActive ? `${dualPhaseLeadFrames.toFixed(2)} 帧` : "不使用"} small={dualPhaseActive ? "平均 dt 前瞻" : "反馈控制"} />
            <Metric title="发送方式" value={dualPhaseActive ? "单观测单命令" : schedulerEnabled ? `${schedulerIntervalMs.toFixed(1)} ms` : "观测直发"} small={dualPhaseActive ? "MouseCommandExecutor" : schedulerEnabled ? `${schedulerStepCountsX}/${schedulerStepCountsY} counts` : "scheduler off"} />
            </div>
            <div className="console-grid2" data-algorithm-page={controlMode}>
              <div className="console-card">
                <SectionTitle title="控制模式" />
                <div className="mini-segmented control-algorithm-segmented" role="group" aria-label="控制模式">
                  {CONTROL_ALGORITHM_OPTIONS.map((algorithm) => (
                    <button
                      className={controlMode === algorithm.id ? "active" : ""}
                      key={algorithm.id}
                      onClick={() => void updateConfigField("control", "active_algorithm", algorithm.id)}
                      type="button"
                    >
                      {algorithm.label}
                    </button>
                  ))}
                </div>
                <p className="console-section-note">{activeControlAlgorithm.description}</p>
                <label>触发方式</label>
                <select value={triggerMode} onChange={(event) => void updateConfigField("control", "trigger_mode", event.target.value)}>
                  <option value="hardware">kmNet 硬件按键触发</option>
                  <option value="always">检测到目标后自动控制</option>
                </select>
                {triggerMode === "hardware" ? (
                  <NumberControl
                    label="按下后启动延迟 ms"
                    detail="kmNet 硬件触发键持续按下达到此时间后，才允许控制输出。自动控制模式不使用该参数。"
                    value={triggerActivationDelayMs}
                    min={0}
                    max={1000}
                    step={1}
                    onCommit={(value) => updateControlGroupField("shared", "trigger_activation_delay_ms", value)}
                  />
                ) : null}
                <div className="control-aim-source-note">
                  <span>
                    <b>瞄点规则由模型类别设置统一提供</b>
                    <small>默认 {aimYRatio.toFixed(2)}；当前配置有 {Object.keys(activeClassAimRatios).length} 个类别使用单独比例。</small>
                  </span>
                  <div className="class-aim-preview" aria-hidden="true">
                    <i style={{ top: `${aimYRatio * 100}%` }} />
                  </div>
                </div>
                <ModuleSwitch
                  label="反转 Y 轴"
                  detail="只改变当前算法输出到设备的 Y 方向。"
                  enabled={dualPhaseActive ? dualPhaseInvertY : sharedInvertY}
                  onToggle={(enabled) => dualPhaseActive
                    ? updateDualPhasePath(["projection", "invert_y"], enabled)
                    : updateControlGroupField("shared", "invert_y", enabled)}
                />
                {dualPhaseActive ? (
                  <div className="console-kv compact-kv">
                    <span>输出执行器</span><b>MouseCommandExecutor</b>
                    <span>每个推理结果</span><b>至多一次 move(dx, dy)</b>
                    <span>轨迹 Scheduler</span><b>不参与</b>
                    <span>误差死区</span><b>不使用</b>
                  </div>
                ) : (
                  <>
                    <NumberControl label="X 到位阈值" value={sharedDeadzoneX} min={0} max={10} step={0.1} onCommit={(value) => updateControlGroupField("shared", "deadzone_x_px", value)} />
                    <NumberControl label="Y 到位阈值" value={sharedDeadzoneY} min={0} max={10} step={0.1} onCommit={(value) => updateControlGroupField("shared", "deadzone_y_px", value)} />
                    <NumberControl label="X counts 增长限制" detail="限制相邻观测中 X 输出增大的速度；接近目标时允许立即减速，反向输出前先归零。" value={sharedMaxSlewX} min={0.1} max={1000} step={0.1} onCommit={(value) => updateControlGroupField("shared", "max_count_slew_x", value)} />
                    <NumberControl label="Y counts 增长限制" detail="限制相邻观测中 Y 输出增大的速度；接近目标时允许立即减速，反向输出前先归零。" value={sharedMaxSlewY} min={0.1} max={1000} step={0.1} onCommit={(value) => updateControlGroupField("shared", "max_count_slew_y", value)} />
                    <ModuleSwitch label="Scheduler 分步发送" detail="关闭后每个新观测直接发送完整 counts" enabled={schedulerEnabled} onToggle={(enabled) => updateConfigField("control", "scheduler_enabled", enabled)} />
                    <NumberControl label="Scheduler X 单步" value={schedulerStepCountsX} min={1} max={20} step={1} onCommit={(value) => updateConfigField("control", "scheduler_step_counts_x", Math.round(value))} />
                    <NumberControl label="Scheduler Y 单步" value={schedulerStepCountsY} min={1} max={20} step={1} onCommit={(value) => updateConfigField("control", "scheduler_step_counts_y", Math.round(value))} />
                    <NumberControl label="Scheduler 间隔 ms" value={schedulerIntervalMs} min={1} max={10} step={0.1} onCommit={(value) => updateConfigField("control", "scheduler_interval_ms", value)} />
                  </>
                )}
              </div>

              <div className="console-card">
                <SectionTitle title={`控制算法 · ${controlModeLabel}`} />
                {dualPhaseActive ? (
                  <>
                    <NumberControl label="水平 FOVX" value={dualPhaseFovX} min={30} max={179} step={0.1} onCommit={(value) => updateDualPhasePath(["projection", "fov_x_deg"], value)} />
                    <NumberControl label="每圈 counts" value={dualPhaseCountsPer360} min={1} max={100000} step={1} onCommit={(value) => updateDualPhasePath(["projection", "counts_per_360"], value)} />
                    <NumberControl label="NEAR 阈值 px" detail="测量误差距离不大于该值时使用 NEAR，否则直接使用 FAR。" value={dualPhaseNearThreshold} min={0} max={1000} step={0.1} onCommit={(value) => updateDualPhasePath(["mode", "near_threshold_px"], value)} />
                    <NumberControl label="FAR Kp" value={dualPhaseFarKp} min={0.001} max={0.999} step={0.001} onCommit={(value) => updateDualPhasePath(["atan", "far", "kp"], value)} />
                    <NumberControl label="NEAR Kp" value={dualPhaseNearKp} min={0.001} max={0.999} step={0.001} onCommit={(value) => updateDualPhasePath(["atan", "near", "kp"], value)} />
                    <NumberControl label="共享 Atan 尺度 counts" detail="FAR 与 NEAR 使用同一个非线性压缩尺度。" value={dualPhaseAtanScale} min={0.1} max={10000} step={0.1} onCommit={(value) => updateDualPhasePath(["atan", "scale_counts"], value)} />
                    <NumberControl label="FAR 单次上限 counts" value={dualPhaseFarMaxCounts} min={1} max={127} step={1} onCommit={(value) => updateDualPhasePath(["atan", "far", "max_counts_per_update"], value)} />
                    <NumberControl label="NEAR 单次上限 counts" value={dualPhaseNearMaxCounts} min={1} max={127} step={1} onCommit={(value) => updateDualPhasePath(["atan", "near", "max_counts_per_update"], value)} />
                    <NumberControl label="前瞻帧数" detail="预测量 = 平滑目标速度 × 最近三段捕获间隔的平均 dt × 前瞻帧数；0 完全关闭位置预测。" value={dualPhaseLeadFrames} min={0} max={10} step={0.01} onCommit={(value) => updateDualPhasePath(["prediction", "lead_frames"], value)} />
                    <NumberControl label="速度平滑帧数" detail="越大越稳但转向越慢；内部仍使用真实 capture timestamp 处理变帧率。" value={dualPhaseVelocitySmoothingFrames} min={0.1} max={20} step={0.1} onCommit={(value) => updateDualPhasePath(["velocity", "smoothing_frames"], value)} />
                    <NumberControl label="历史中断重置 ms" value={dualPhaseHistoryResetGapMs} min={0.1} max={500} step={0.1} onCommit={(value) => updateDualPhasePath(["velocity", "history_reset_gap_ms"], value)} />
                    <NumberControl label="FAR 预测绝对上限 px" value={dualPhaseFarPredictionCap} min={0} max={100} step={0.1} onCommit={(value) => updateDualPhasePath(["prediction", "far", "absolute_cap_px"], value)} />
                    <NumberControl label="NEAR 预测绝对上限 px" value={dualPhaseNearPredictionCap} min={0} max={100} step={0.1} onCommit={(value) => updateDualPhasePath(["prediction", "near", "absolute_cap_px"], value)} />
                  </>
                ) : controlMode === "calibrated_angular" ? (
                  <>
                    <NumberControl label="水平 FOVX" value={calibratedFovX} min={30} max={179} step={0.1} onCommit={(value) => updateControlGroupField("calibrated_angular", "fov_x_deg", value)} />
                    <NumberControl label="X 每圈 counts" value={calibratedCountsPer360X} min={1} max={100000} step={1} onCommit={(value) => updateControlGroupField("calibrated_angular", "counts_per_360_x", value)} />
                    <NumberControl label="Y 每圈 counts" value={calibratedCountsPer360Y} min={1} max={100000} step={1} onCommit={(value) => updateControlGroupField("calibrated_angular", "counts_per_360_y", value)} />
                    <NumberControl label="Kp X" value={calibratedKpX} min={0} max={2} step={0.01} onCommit={(value) => updateControlGroupField("calibrated_angular", "kp_x", value)} />
                    <NumberControl label="Kp Y" value={calibratedKpY} min={0} max={2} step={0.01} onCommit={(value) => updateControlGroupField("calibrated_angular", "kp_y", value)} />
                    <NumberControl label="Kd X" value={calibratedKdX} min={0} max={1} step={0.01} onCommit={(value) => updateControlGroupField("calibrated_angular", "kd_x", value)} />
                    <NumberControl label="Kd Y" value={calibratedKdY} min={0} max={1} step={0.01} onCommit={(value) => updateControlGroupField("calibrated_angular", "kd_y", value)} />
                    <NumberControl label="D 项 EMA" value={calibratedDEmaAlpha} min={0.01} max={1} step={0.01} onCommit={(value) => updateControlGroupField("calibrated_angular", "d_ema_alpha", value)} />
                    <NumberControl label="X 最大角度步长 deg" value={calibratedMaxAngleX} min={0.0001} max={180} step={0.0001} onCommit={(value) => updateControlGroupField("calibrated_angular", "max_angle_step_x_deg", value)} />
                    <NumberControl label="Y 最大角度步长 deg" value={calibratedMaxAngleY} min={0.0001} max={180} step={0.0001} onCommit={(value) => updateControlGroupField("calibrated_angular", "max_angle_step_y_deg", value)} />
                  </>
                ) : (
                  <>
                    <NumberControl label="水平响应尺度 px" value={universalResponseScaleX} min={0.1} max={4000} step={0.1} onCommit={(value) => updateControlGroupField("universal_saturated", "response_scale_x_px", value)} />
                    <NumberControl label="垂直响应尺度 px" value={universalResponseScaleY} min={0.1} max={4000} step={0.1} onCommit={(value) => updateControlGroupField("universal_saturated", "response_scale_y_px", value)} />
                    <NumberControl label="最大水平移动 counts" value={universalMaxStepX} min={0.1} max={1000} step={0.1} onCommit={(value) => updateControlGroupField("universal_saturated", "max_step_x_counts", value)} />
                    <NumberControl label="最大垂直移动 counts" value={universalMaxStepY} min={0.1} max={1000} step={0.1} onCommit={(value) => updateControlGroupField("universal_saturated", "max_step_y_counts", value)} />
                  </>
                )}
              </div>

              <div className="console-card">
                  <SectionTitle title="Y 轴后坐力前馈 · 所有控制算法" />
                  <ModuleSwitch label="启用 Y 轴压枪" detail="只在检测到真实左键持续按下且存在有效目标时，将设备 counts 前馈叠加到视觉 Y 误差反馈；不会开启目标 Y 速度预测。" enabled={recoilEnabled} onToggle={(enabled) => updateControlGroupField("shared", "recoil_enabled", enabled)} />
                  {recoilEnabled ? (
                    <>
                      <NumberControl label="压枪启动延迟 ms" detail="左键持续按下达到此时间后才开始压枪；与硬件触发启动延迟相互独立。" value={recoilStartDelayMs} min={0} max={1000} step={1} onCommit={(value) => updateControlGroupField("shared", "recoil_start_delay_ms", value)} />
                      <NumberControl label="Y 压枪速率 counts/s" detail="持续按压时每秒追加的 Y 轴设备 counts；最终方向仍受反转 Y 轴设置影响。" value={recoilYRate} min={0} max={5000} step={1} onCommit={(value) => updateControlGroupField("shared", "recoil_y_rate_counts_s", value)} />
                      <NumberControl label="压枪渐入 ms" detail="从 0 平滑增长到完整压枪速率所需时间，避免按下瞬间产生突跳。" value={recoilRampUpMs} min={0} max={2000} step={1} onCommit={(value) => updateControlGroupField("shared", "recoil_ramp_up_ms", value)} />
                      <NumberControl label="单观测最大压枪 counts" detail="每个新观测最多允许叠加的压枪量，防止异常观测间隔产生大步输出。" value={recoilMaxCounts} min={0.1} max={20} step={0.1} onCommit={(value) => updateControlGroupField("shared", "recoil_max_counts_per_observation", value)} />
                    </>
                  ) : null}
                </div>

              <div className="console-card">
                <SectionTitle title="目标选择与切换 · 通用参数" />
                <NumberControl label="目标选择半径（640 基准 px）" detail="以 640×640 ROI 为基准；运行时按当前 ROI 尺寸同比缩放，保证 320～640 ROI 使用一致的相对选择范围。" value={targetFovRadiusPx} min={1} max={640} step={1} onCommit={(value) => updateConfigField("control", "target_fov_radius_px", value)} />
                <NumberControl label="候选框最大宽高比" detail="拒绝宽高比或高宽比超过此值的异常细长框。值越大越宽松。" value={candidateRatioMaxAspect} min={1} max={20} step={0.1} onCommit={(value) => updateConfigField("control", "candidate_ratio_max_aspect", value)} />
                <NumberControl label="质量权重：置信度" detail="候选质量分数中检测置信度的相对权重；会与面积权重归一化后使用。" value={candidateQualityConfidenceWeight} min={0} max={2} step={0.05} onCommit={(value) => updateConfigField("control", "candidate_quality_confidence_weight", value)} />
                <NumberControl label="质量权重：可见尺寸" detail="使用 bbox 面积占 ROI 比例的平方根，作为与 ROI 分辨率无关的可见尺寸分；会与置信度权重归一化后使用。" value={candidateQualityAreaWeight} min={0} max={2} step={0.05} onCommit={(value) => updateConfigField("control", "candidate_quality_area_weight", value)} />
                <NumberControl label="综合分权重：类别" detail="类别优先列表第一项得 1.0，第二项得 0.5，其余类别得 0.0；该值控制类别分在最终选择中的占比。" value={candidateSelectionClassWeight} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "candidate_selection_class_weight", value)} />
                <NumberControl label="综合分权重：质量" detail="质量分由检测置信度、ROI 归一化 bbox 面积和 Track 可靠性共同限制。" value={candidateSelectionQualityWeight} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "candidate_selection_quality_weight", value)} />
                <NumberControl label="综合分权重：距离" detail="距离按当前实际选择半径归一化，ROI 从 320 调整到 640 时保持同样的相对含义。" value={candidateSelectionDistanceWeight} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "candidate_selection_distance_weight", value)} />
                <NumberControl label="切换最小优势" detail="新候选的综合分减去当前锁定目标综合分，至少达到此值才允许进入切换确认。" value={targetSwitchPreferenceAdvantage} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "target_switch_min_preference_advantage", value)} />
                <NumberControl label="切换最小连续性" detail="新候选 Track 的身份连续性至少达到此值，才允许进入切换确认；值越高越不易误切换。" value={targetSwitchContinuityScore} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "target_switch_min_continuity_score", value)} />
                <NumberControl label="目标切换确认延迟 ms" detail="新候选持续同时满足优势和连续性阈值达到此时间后，才正式替换当前目标。" value={targetSwitchDelayMs} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "target_switch_delay_ms", value)} />
              </div>

              <div className="console-card">
                <SectionTitle title={dualPhaseActive ? "Tracker · 公共参数" : "Tracker / Kalman · 公共参数"} />
                <div className="console-kv compact-kv"><span>关联算法</span><b>Hungarian</b><span>输出状态</span><b>仅 ACTIVE</b></div>
                <NumberControl label="归一化匹配距离" value={trackerMaxMatchDistance} min={0.1} max={5} step={0.05} onCommit={(value) => updateConfigField("control", "tracker_max_match_distance", value)} />
                <NumberControl label="位置代价权重" value={trackerPositionCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "tracker_position_cost_weight", value)} />
                <NumberControl label="IoU 代价权重" value={trackerIouCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "tracker_iou_cost_weight", value)} />
                <NumberControl label="最大漏检轮数" value={trackerMaxMissedFrames} min={0} max={10} step={1} onCommit={(value) => updateConfigField("control", "tracker_max_missed_frames", Math.round(value))} />
                {!dualPhaseActive ? (
                  <details className="model-debug-details">
                    <summary>Kalman 高级参数</summary>
                    <NumberControl label="加速度噪声" value={kalmanAccelerationNoise} min={0.001} max={10000} step={10} onCommit={(value) => updateConfigField("control", "kalman_acceleration_noise", value)} />
                    <NumberControl label="X 测量噪声" value={kalmanMeasurementNoiseX} min={0.001} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_measurement_noise_x", value)} />
                    <NumberControl label="Y 测量噪声" value={kalmanMeasurementNoiseY} min={0.001} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_measurement_noise_y", value)} />
                  </details>
                ) : null}
              </div>
            </div>
          </>
          ) : (
          <>
          <div className="console-metrics">
            <Metric title="连接状态" value={kmnetConnected ? "已连接" : kmnetConnecting ? "连接中" : "未连接"} small={kmnetConnected ? "online" : kmnetConnecting ? "connecting" : "offline"} />
            <Metric title="驱动状态" value={kmnetDriverAvailable ? "可用" : "不可用"} small="kmNet" />
            <Metric title="按键监听" value={kmnetStatus.monitoring === true ? "监听中" : "未监听"} small="monitor" />
            <Metric title="自动连接" value={kmnetAutoConnect ? "已启用" : "已关闭"} small="startup" />
            <Metric title="发送次数" value={formatNumber(kmnetStatus.move_count, 0)} small="counts" />
            <Metric title="最近移动" value={`${formatNumber(kmnetStatus.last_dx, 0)} / ${formatNumber(kmnetStatus.last_dy, 0)}`} small="dx / dy" />
          </div>
          <div className="console-grid2 control-test-grid">
            <div className="console-card">
              <SectionTitle title="kmNet 控制面板" />
              <div className="kmnet-status-grid">
                <div className={kmnetConnected ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>连接</span>
                  <b>{kmnetConnected ? "已连接" : kmnetConnecting ? "连接中" : "未连接"}</b>
                </div>
                <div className={kmnetStatus.monitoring === true ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>按键</span>
                  <b>{kmnetStatus.monitoring === true ? `${kmnetButtonLeft ? "左键" : "-"} / ${kmnetButtonRight ? "右键" : "-"}` : "未监听"}</b>
                </div>
                <div className={kmnetDriverAvailable ? "kmnet-status-tile good" : "kmnet-status-tile bad"}>
                  <span>驱动</span>
                  <b>{kmnetDriverAvailable ? "可用" : "不可用"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>平台</span>
                  <b>{`${readString(kmnetStatus.driver_platform, "-")}/${readString(kmnetStatus.driver_machine, "-")}`}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>发送次数</span>
                  <b>{formatNumber(kmnetStatus.move_count, 0)}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>最近移动</span>
                  <b>{`${formatNumber(kmnetStatus.last_dx, 0)} / ${formatNumber(kmnetStatus.last_dy, 0)}`}</b>
                </div>
                <div className={execution.sent === true ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>算法发送</span>
                  <b>{execution.sent === true ? "已发送" : execution.sent === false ? "未发送" : "-"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>执行器</span>
                  <b>{readString(execution.executor_id, readString(executorStatus.selected, "-"))}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>连接阶段</span>
                  <b>{readString(kmnetStatus.connection_stage, "-")}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>最近驱动调用</span>
                  <b>{`${readString(kmnetStatus.last_driver_call, "-")} · ${formatNumber(kmnetStatus.last_driver_call_duration_ms, 2)} ms`}</b>
                </div>
                <div className={kmnetStatus.route_available === true ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>网络路由</span>
                  <b>{kmnetStatus.route_available === true ? "已找到" : kmnetStatus.route_available === false ? "无路由" : "未检查"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>源 IP / 目标 IP</span>
                  <b>{`${readString(kmnetStatus.route_local_ip, "-")} / ${readString(kmnetStatus.route_resolved_ip, "-")}`}</b>
                </div>
              </div>
              <div className="kmnet-driver-line">
                <span>{compactDriverSource(kmnetStatus.driver_source)}</span>
                <span>{readString(kmnetStatus.driver_python, "-")}</span>
              </div>
              <TextControl label="kmnetip" value={kmnetHost} onCommit={(value) => updateConfigField("hardware", "host", value)} />
              <NumberControl label="kmnetport" value={kmnetPort} min={0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "port", Math.round(value))} />
              <TextControl label="kmnetuuid" value={kmnetUuid} onCommit={(value) => updateConfigField("hardware", "uuid", value)} />
              <NumberControl label="monitor_port" value={kmnetMonitorPort} min={0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "monitor_port", Math.round(value))} />
              <ModuleSwitch
                label="服务启动自动连接"
                detail="后端启动完成后按当前地址连接 kmNet"
                enabled={kmnetAutoConnect}
                onToggle={(enabled) => updateConfigField("hardware", "auto_connect", enabled)}
              />
              <label>移动 API</label>
              <select
                value={moveKind}
                onChange={(event) => setKmnetMoveKind(event.target.value)}
              >
                <option value="raw">move：最快直移</option>
                <option value="enc_raw">enc_move：加密直移</option>
                <option value="auto">move_auto：模拟移动</option>
                <option value="enc_auto">enc_move_auto：加密模拟移动</option>
                <option value="bezier">move_beizer：贝塞尔曲线</option>
                <option value="enc_bezier">enc_move_beizer：加密贝塞尔曲线</option>
              </select>
              <label>命令调度</label>
              {dualPhaseActive ? (
                <div className="console-kv compact-kv">
                  <span>执行层</span><b>MouseCommandExecutor</b>
                  <span>行为</span><b>当前观测整数 counts 直接发送一次</b>
                  <span>Scheduler</span><b>此算法固定绕过</b>
                </div>
              ) : (
                <>
                  <ModuleSwitch label="Scheduler 分步发送" detail="关闭后每个新观测直接调用一次 kmNet" enabled={schedulerEnabled} onToggle={(enabled) => updateConfigField("control", "scheduler_enabled", enabled)} />
                  <NumberControl label="Scheduler 间隔 ms" value={schedulerIntervalMs} min={1} max={10} step={0.1} onCommit={(value) => updateConfigField("control", "scheduler_interval_ms", value)} />
                  <NumberControl label="X 单步 counts" value={schedulerStepCountsX} min={1} max={20} step={1} onCommit={(value) => updateConfigField("control", "scheduler_step_counts_x", Math.round(value))} />
                  <NumberControl label="Y 单步 counts" value={schedulerStepCountsY} min={1} max={20} step={1} onCommit={(value) => updateConfigField("control", "scheduler_step_counts_y", Math.round(value))} />
                </>
              )}
              <div className="console-action-row">
                <button
                  className={kmnetConnected || kmnetConnecting ? "console-button danger" : "console-button primary"}
                  aria-pressed={kmnetConnected || kmnetConnecting}
                  disabled={busy === "kmnet.toggle"}
                  onClick={() => void toggleHardwareConnection()}
                  type="button"
                >
                  {kmnetConnected ? "断开 kmNet" : kmnetConnecting ? "取消连接 kmNet" : "连接 kmNet"}
                </button>
                <button
                  className="console-button"
                  disabled={busy === "kmnet.defaults"}
                  onClick={() => void applyKmNetRecommended()}
                  type="button"
                >
                  应用推荐参数
                </button>
              </div>
              <div className="kmnet-test-panel">
                <div className="kmnet-test-inputs">
                  <label>
                    <span>dx</span>
                    <input
                      type="number"
                      value={kmnetTestDx}
                      onChange={(event) => setKmnetTestDx(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    <span>dy</span>
                    <input
                      type="number"
                      value={kmnetTestDy}
                      onChange={(event) => setKmnetTestDy(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    <span>ms</span>
                    <input
                      type="number"
                      value={kmnetTestMs}
                      onChange={(event) => setKmnetTestMs(Number(event.target.value))}
                    />
                  </label>
                </div>
                <div className="kmnet-pad">
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(0, -10, 1, 0, "raw", 0)}>↑</button>
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(-10, 0, 1, 0, "raw", 0)}>←</button>
                  <button type="button" onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "raw", 0)} disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}>发送</button>
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(10, 0, 1, 0, "raw", 0)}>→</button>
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(0, 10, 1, 0, "raw", 0)}>↓</button>
                </div>
                <div className="kmnet-test-section">
                  <h3>最快直移</h3>
                  <p>调用 move / enc_move，只传 x、y。适合验证最底层驱动是否能立即移动。</p>
                  <div className="console-action-row">
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "raw", 0)}
                      type="button"
                    >
                      move
                    </button>
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "enc_raw", 0)}
                      type="button"
                    >
                      enc_move
                    </button>
                  </div>
                </div>
                <div className="kmnet-test-section">
                  <h3>自动模拟</h3>
                  <p>调用 move_auto / enc_move_auto，指定 ms，按最小步进逼近目标。</p>
                  <div className="console-action-row">
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "auto", kmnetTestMs)}
                      type="button"
                    >
                      move_auto
                    </button>
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "enc_auto", kmnetTestMs)}
                      type="button"
                    >
                      enc_move_auto
                    </button>
                  </div>
                </div>
                <div className="kmnet-test-section">
                  <h3>贝塞尔曲线</h3>
                  <p>调用 move_beizer / enc_move_beizer，参数为 x、y、ms、x1、y1、x2、y2。</p>
                  <div className="kmnet-test-inputs bezier">
                    <label><span>x1</span><input type="number" value={kmnetBezierX1} onChange={(event) => setKmnetBezierX1(Number(event.target.value))} /></label>
                    <label><span>y1</span><input type="number" value={kmnetBezierY1} onChange={(event) => setKmnetBezierY1(Number(event.target.value))} /></label>
                    <label><span>x2</span><input type="number" value={kmnetBezierX2} onChange={(event) => setKmnetBezierX2(Number(event.target.value))} /></label>
                    <label><span>y2</span><input type="number" value={kmnetBezierY2} onChange={(event) => setKmnetBezierY2(Number(event.target.value))} /></label>
                  </div>
                  <div className="console-action-row">
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "bezier", kmnetTestMs, { x1: kmnetBezierX1, y1: kmnetBezierY1, x2: kmnetBezierX2, y2: kmnetBezierY2 })}
                      type="button"
                    >
                      move_beizer
                    </button>
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "enc_bezier", kmnetTestMs, { x1: kmnetBezierX1, y1: kmnetBezierY1, x2: kmnetBezierX2, y2: kmnetBezierY2 })}
                      type="button"
                    >
                      enc_move_beizer
                    </button>
                  </div>
                </div>
                <div className="kmnet-test-section">
                  <h3>辅助诊断</h3>
                  <p>连续发送用于检查频率响应；右移大步和画圆用于确认肉眼可见移动。</p>
                  <div className="console-action-row">
                    <button
                      className="console-button"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 3, 4, "raw", 0)}
                      type="button"
                    >
                      连续发送
                    </button>
                    <button
                      className="console-button primary"
                      disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                      onClick={() => void diagnosticMoveHardware(600, 0, 1, 0, "raw", 0)}
                      type="button"
                    >
                      右移大步测试
                    </button>
                  </div>
                  <button
                    className="kmnet-circle-button"
                    disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                    onClick={() => void diagnosticCircleHardware()}
                    type="button"
                  >
                    {busy === "kmnet.circle" ? "画圆中" : "画圆测试"}
                  </button>
                </div>
                <div className="kmnet-test-result">
                  <span>最近移动</span>
                  <b>{`${readNumber(kmnetStatus.last_dx, 0)} / ${readNumber(kmnetStatus.last_dy, 0)} · ${readNumber(kmnetStatus.move_count, 0)} 次`}</b>
                </div>
                {kmnetTestMessage ? <div className="kmnet-test-message">{kmnetTestMessage}</div> : null}
              </div>
              {readString(kmnetStatus.last_error, "") ? (
                <div className="kmnet-error">
                  {`[${readString(kmnetStatus.last_connect_error_stage, "unknown")}/${readString(kmnetStatus.last_connect_error_type, "unknown")}] ${readString(kmnetStatus.last_error, "")}`}
                </div>
              ) : null}
            </div>
          </div>
          </>
          )}
        </section>

        <section className={activePage === "stats" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="总 FPS" value={formatNumber(statistics?.capture_fps, 1)} small="FPS" />
            <Metric title="平均延迟" value={formatNumber(statistics?.e2e_latency, 1)} small="ms" />
            <Metric title="P95 延迟" value="待机" small="ms" />
            <Metric title="运行时长" value={runtime?.running ? "运行中" : "00:00"} small="time" />
          </div>
          <div className="console-grid3">
            <KvCard title="采集统计" rows={[["成功帧", String(statistics?.capture_counter ?? 0)], ["丢弃帧", String(statistics?.dropped_counter ?? 0)], ["抖动", formatNumber(capture?.frame_period_ms, 2)]]} />
            <KvCard
              title="推理统计"
              notice={inferenceFreshnessBlocked ? (
                <div className="stats-diagnosis failed">
                  <strong>吞吐正常，但批次新鲜度不合格</strong>
                  <span>
                    latest 推理有输出，但批次未进入控制；
                    结束时帧差 {formatNumber(inferenceGenerationLag, 0)}，
                    最后帧龄 {formatNumber(lastFrameAgeMs, 1)}ms。
                  </span>
                  <em>实时控制不会补完旧帧；过期或非 latest 的 DetectionBatch 会被丢弃。</em>
                </div>
              ) : null}
              rows={[
              ["完成帧", String(statistics?.inference_counter ?? 0)],
              ["推理 FPS", formatNumber(statistics?.inference_fps, 1)],
              ["Batch 已发布", formatNumber(statistics?.detection_batch_counter, 0)],
              ["Batch 已消费", formatNumber(statistics?.detection_batch_consumed_counter, 0)],
              ["Batch 发布 FPS", formatNumber(statistics?.detection_batch_fps, 1)],
              ["控制观察 FPS", formatNumber(statistics?.control_observation_fps, 1)],
              ["跳过帧", formatNumber(statistics?.skipped_counter, 0)],
              ["推理前过期", formatNumber(statistics?.stale_dropped_batches, 0)],
              ["时间戳拒绝", formatNumber(statistics?.timestamp_rejected_batches, 0)],
              ["非单调拒绝", formatNumber(statistics?.non_monotonic_dropped_batches, 0)],
              ["Batch 覆盖", formatNumber(statistics?.mailbox_overwritten_batches, 0)],
              ["旧 batch 丢弃", formatNumber(statistics?.stale_drop_count, 0)],
              ["推理期间发布", formatNumber(inferencePublishedSinceAcquire, 0)],
              ["结束时帧差", formatNumber(inferenceGenerationLag, 0)],
              ["时间戳", shortTimestampSource(readString(statistics?.timestamp_source, "-"))],
              ["最后帧龄", formatNumber(statistics?.last_frame_age_ms, 1)],
              ["Batch age", formatNumber(inferenceResultAgeMs, 1)],
              ["ROI", formatNumber(statistics?.stage_roi_ms, 1)],
              ["推理总耗时", formatNumber(statistics?.stage_engine_ms, 1)],
              ["TRT执行", formatNumber(statistics?.stage_engine_execute_ms, 1)],
              ["解码/NMS", formatNumber(statistics?.stage_decode_ms, 1)],
              ["映射后处理", formatNumber(statistics?.stage_postprocess_ms, 1)],
              ["控制", formatNumber(statistics?.stage_control_ms, 1)]
            ]}
            />
            <KvCard title="系统状态" rows={[["CPU", "待机"], ["GPU", "待机"], ["温度", "-"]]} />
          </div>
          <div className="console-card">
            <SectionTitle title="性能占比" />
            <Bar label="采集" width={35} />
            <Bar label="预处理" width={18} />
            <Bar label="推理" width={52} />
            <Bar label="后处理" width={24} />
          </div>
        </section>

        <section className={activePage === "latency" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集等待" value={formatNumber(capture?.capture_wait_ms, 2)} small="ms" />
            <Metric title="帧间隔" value={formatNumber(capture?.frame_period_ms, 2)} small="ms" />
            <Metric title="端到端" value={formatNumber(statistics?.e2e_latency, 1)} small="ms" />
            <Metric title="队列等待" value={formatNumber(statistics?.queue_latency, 1)} small="ms" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <SectionTitle title="延迟链路" />
              <div className="console-timeline">
                <Event label="Capture" value={formatNumber(capture?.capture_wait_ms, 2)} width={30} />
                <Event label="Queue" value={formatNumber(statistics?.queue_latency, 1)} width={18} />
                <Event label="ROI" value={formatNumber(statistics?.stage_roi_ms, 1)} width={18} />
                <Event label="推理总耗时" value={formatNumber(statistics?.stage_engine_ms, 1)} width={56} />
                <Event label="TRT执行" value={formatNumber(statistics?.stage_engine_execute_ms, 1)} width={18} />
                <Event label="解码/NMS" value={formatNumber(statistics?.stage_decode_ms, 1)} width={34} />
                <Event label="映射后处理" value={formatNumber(statistics?.stage_postprocess_ms, 1)} width={20} />
                <Event label="Control" value={formatNumber(statistics?.stage_control_ms, 1)} width={14} />
              </div>
            </div>
            <KvCard title="采集诊断" rows={[["状态判断", capture?.available ? "采集中" : "等待数据"], ["队列积压", String(readNumber(asRecord(pipeline.queue).size, 0))], ["建议", capture?.available ? "观察队列等待和帧间隔" : "启动后分析"]]} />
          </div>
        </section>
      </main>

      {launchDialogOpen ? (
        <div
          aria-hidden="false"
          className="launch-dialog-layer"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && launchStatus !== "running") {
              closeLaunchDialog();
            }
          }}
        >
          <section
            aria-labelledby="launch-dialog-title"
            aria-modal="true"
            className="launch-dialog"
            role="dialog"
          >
            <header className="launch-dialog-header">
              <div className="launch-dialog-title-wrap">
                <div className="launch-dialog-icon" aria-hidden="true">
                  <NovaIcon name="start" size={22} strokeWidth={1.9} />
                </div>
                <div>
                  <h2 id="launch-dialog-title">启动视觉处理链路</h2>
                  <p>只展示必要启动阶段，不加载额外运行监控。</p>
                </div>
              </div>
              <button
                aria-label="关闭"
                className="launch-dialog-close"
                disabled={launchStatus === "running"}
                onClick={closeLaunchDialog}
                type="button"
              >
                <NovaIcon name="x-circle" size={18} />
              </button>
            </header>

            <div className="launch-dialog-body">
              <div className="launch-stage-visual">
                <div className={launchIndicatorClass}>{launchIndicatorText}</div>
                <div>
                  <div className="launch-stage-label">
                    {launchStatus === "idle" ? "准备启动" : launchStatus === "success" ? "启动完成" : launchStatus === "failed" ? "启动失败" : launchStatus === "cancelled" ? "已取消" : activeLaunchStage.label}
                  </div>
                  <div className="launch-stage-title">{launchTitle}</div>
                  <div className="launch-stage-caption">{launchCaption}</div>
                </div>
              </div>

              <div className="launch-progress-block">
                <div className="launch-progress-meta">
                  <span>总进度</span>
                  <span>{launchProgress}%</span>
                </div>
                <div className="launch-progress-track">
                  <div className="launch-progress-bar" style={{ width: `${launchProgress}%` }} />
                </div>
                <div className="launch-progress-note">
                  {launchProgressDetail ? <strong>{launchProgressDetail}</strong> : null}
                  <span>前端仅维护阶段开始、阶段完成、启动失败、启动完成四类低频反馈；不轮询 FPS、温度、显存或后端日志，不进入采集、推理、跟踪、控制线程。</span>
                </div>
              </div>
            </div>

            <footer className="launch-dialog-footer">
              <button className="console-button" onClick={() => void cancelMainlineLaunch()} type="button">
                <NovaIcon name={launchStatus === "running" ? "pause-output" : "x-circle"} size={16} />
                {launchStatus === "running" ? "取消启动" : "关闭"}
              </button>
              <button
                className="console-button primary"
                disabled={launchStatus === "running"}
                onClick={launchStatus === "success" ? closeLaunchDialog : () => void startMainlineLaunch()}
                type="button"
              >
                <NovaIcon name={launchStatus === "running" ? "activity-pulse" : launchStatus === "success" ? "dashboard" : "start"} size={16} />
                {launchStatus === "running" ? "正在启动" : launchStatus === "success" ? "进入工作台" : launchStatus === "failed" || launchStatus === "cancelled" ? "重新启动" : "开始启动"}
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      <div className={launchToastVisible ? "launch-toast show" : "launch-toast"} role="status">
        主链启动请求已提交
      </div>
    </section>
  );
}

function NumberControl({
  label,
  detail,
  value,
  min,
  max,
  step,
  onCommit
}: {
  label: string;
  detail?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  return (
    <>
      <label title={detail}>{label}</label>
      <CommitNumberControl
        value={value}
        min={min}
        max={max}
        step={step}
        digits={step >= 1 ? 0 : 2}
        onCommit={onCommit}
      />
    </>
  );
}

function CommitNumberControl({
  value,
  min,
  max,
  step,
  digits,
  onCommit
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  digits: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(value);
  const [isEditing, setIsEditing] = useState(false);
  const committingRef = useRef(false);

  useEffect(() => {
    if (!isEditing) {
      setDraft(value);
    }
  }, [isEditing, value]);

  const commit = useCallback(() => {
    if (committingRef.current) {
      return;
    }
    const next = clampNumber(Number(draft.toFixed(digits)), min, max);
    if (Math.abs(next - value) >= step / 2) {
      committingRef.current = true;
      setDraft(next);
      void Promise.resolve(onCommit(next)).finally(() => {
        committingRef.current = false;
        setIsEditing(false);
      });
    } else {
      setDraft(value);
      setIsEditing(false);
    }
  }, [draft, digits, max, min, onCommit, step, value]);

  return (
    <div className="console-row">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={draft}
        onBlur={commit}
        onChange={(event) => {
          setIsEditing(true);
          setDraft(clampNumber(Number(event.target.value), min, max));
        }}
        onFocus={() => setIsEditing(true)}
        onPointerDown={() => setIsEditing(true)}
        onMouseUp={commit}
        onPointerUp={commit}
        onTouchEnd={commit}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number.isInteger(draft) ? String(draft) : draft.toFixed(digits)}
        onBlur={commit}
        onFocus={() => setIsEditing(true)}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
            setIsEditing(true);
            setDraft(clampNumber(next, min, max));
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
    </div>
  );
}

function TextControl({
  label,
  value,
  onCommit
}: {
  label: string;
  value: string;
  onCommit: (value: string) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const commit = useCallback(() => {
    const next = draft.trim();
    if (next !== value) {
      void onCommit(next);
    }
  }, [draft, onCommit, value]);

  return (
    <>
      <label>{label}</label>
      <input
        value={draft}
        onBlur={commit}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
    </>
  );
}

function InlineTextControl({
  value,
  placeholder,
  ariaLabel,
  onCommit
}: {
  value: string;
  placeholder: string;
  ariaLabel: string;
  onCommit: (value: string) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => setDraft(value), [value]);

  const commit = useCallback(() => {
    const next = draft.trim();
    if (next !== value) {
      void onCommit(next);
    }
  }, [draft, onCommit, value]);

  return (
    <input
      aria-label={ariaLabel}
      value={draft}
      placeholder={placeholder}
      onBlur={commit}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function ClassAimRatioControl({
  classId,
  defaultRatio,
  overrideRatio,
  onCommit
}: {
  classId: number;
  defaultRatio: number;
  overrideRatio: number | undefined;
  onCommit: (value: number | null) => Promise<void> | void;
}) {
  const custom = overrideRatio !== undefined;
  const effectiveRatio = custom ? overrideRatio : defaultRatio;
  const [draft, setDraft] = useState(effectiveRatio);

  useEffect(() => setDraft(effectiveRatio), [effectiveRatio]);

  const commit = useCallback(() => {
    if (!custom) {
      return;
    }
    const next = clampNumber(Number(draft.toFixed(2)), 0, 1);
    setDraft(next);
    if (next !== overrideRatio) {
      void onCommit(next);
    }
  }, [custom, draft, onCommit, overrideRatio]);

  return (
    <div className="class-aim-control">
      <button
        className={custom ? "class-aim-mode custom" : "class-aim-mode"}
        type="button"
        onClick={() => void onCommit(custom ? null : defaultRatio)}
      >
        {custom ? "单独设置" : "使用默认"}
      </button>
      <input
        aria-label={`cls ${classId} 瞄点高度`}
        disabled={!custom}
        min={0}
        max={1}
        step={0.01}
        type="number"
        value={draft.toFixed(2)}
        onBlur={commit}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
            setDraft(clampNumber(next, 0, 1));
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
      <div className="class-aim-preview" aria-hidden="true">
        <i style={{ top: `${effectiveRatio * 100}%` }} />
      </div>
    </div>
  );
}

function metricIconForTitle(title: string): NovaIconName {
  if (title.includes("FPS")) {
    return "fps";
  }
  if (title.includes("延迟") || title.includes("等待") || title.includes("帧间隔") || title.includes("端到端") || title.includes("队列")) {
    return "latency";
  }
  if (title.includes("分辨率")) {
    return "resolution";
  }
  if (title.includes("像素") || title.includes("格式")) {
    return "frame";
  }
  if (title.includes("丢帧") || title.includes("跳过")) {
    return "signal-lost";
  }
  if (title.includes("目标")) {
    return "target";
  }
  if (title.includes("引擎")) {
    return "engine";
  }
  if (title.includes("算法")) {
    return "pid";
  }
  if (title.includes("触发")) {
    return "activity-pulse";
  }
  if (title.includes("Kp")) {
    return "gain";
  }
  if (title.includes("角度") || title.includes("FOV")) {
    return "fov";
  }
  if (title.includes("限幅")) {
    return "max-step";
  }
  if (title.includes("运行时长")) {
    return "clock";
  }
  return "performance";
}

function cardIconForTitle(title: string): NovaIconName {
  if (title.includes("采集")) {
    return "capture";
  }
  if (title.includes("推理")) {
    return "inference";
  }
  if (title.includes("系统")) {
    return "system";
  }
  if (title.includes("诊断")) {
    return "triangle-alert";
  }
  return "dashboard";
}

function sectionIconForTitle(title: string): NovaIconName {
  if (title.includes("采集设备")) {
    return "capture-card";
  }
  if (title.includes("ROI")) {
    return "roi";
  }
  if (title.includes("模型")) {
    return "models";
  }
  if (title.includes("推理输出")) {
    return "output-tensor";
  }
  if (title.includes("算法")) {
    return "pid";
  }
  if (title.includes("控制量")) {
    return "output";
  }
  if (title.includes("kmNet")) {
    return "hid";
  }
  if (title.includes("性能")) {
    return "performance";
  }
  if (title.includes("延迟")) {
    return "latency";
  }
  return "dashboard";
}

function sectionDescriptionForTitle(title: string): string {
  if (title.includes("采集设备")) {
    return "视频源、后端通路与 latest-frame 策略";
  }
  if (title.includes("ROI")) {
    return "裁剪区域决定推理、预览和控制坐标基准";
  }
  if (title.includes("模型设置")) {
    return "绑定当前运行模型和 TensorRT 引擎";
  }
  if (title.includes("推理输出")) {
    return "查看 DetectionBatch、目标框和新鲜度";
  }
  if (title.includes("鼠标移动算法")) {
    return "PD/PID、滤波、预测和输出限幅";
  }
  if (title.includes("控制量反馈")) {
    return "控制决策、调度器和执行器状态";
  }
  if (title.includes("kmNet")) {
    return "硬件连接、触发键和移动诊断";
  }
  if (title.includes("性能占比")) {
    return "采集、预处理、推理和后处理耗时";
  }
  if (title.includes("延迟链路")) {
    return "按阶段定位实时链路瓶颈";
  }
  if (title.includes("采集统计")) {
    return "最新帧采集吞吐与丢弃情况";
  }
  if (title.includes("推理统计")) {
    return "批次消费、新鲜度和阶段耗时";
  }
  if (title.includes("系统状态")) {
    return "硬件资源和服务状态摘要";
  }
  if (title.includes("采集诊断")) {
    return "队列积压、帧间隔和采集建议";
  }
  return "";
}

function SectionTitle({ title, icon }: { title: string; icon?: NovaIconName }) {
  const description = sectionDescriptionForTitle(title);

  return (
    <h2 className="console-title">
      <span className="console-title-icon">
        <NovaIcon name={icon ?? sectionIconForTitle(title)} size={18} />
      </span>
      <span className="console-title-copy">
        <span>{title}</span>
        {description ? <small>{description}</small> : null}
      </span>
    </h2>
  );
}

function Metric({
  title,
  value,
  small,
  icon,
}: {
  title: string;
  value: string;
  small: string;
  icon?: NovaIconName;
}) {
  const resolvedIcon = icon ?? metricIconForTitle(title);

  return (
    <div className="console-metric" aria-label={`${title}: ${value} ${small}`}>
      <span className="console-metric-head">
        <span className="console-metric-icon">
          <NovaIcon name={resolvedIcon} size={18} />
        </span>
        <span>{title}</span>
      </span>
      <strong>{value}</strong>
      <small>{small}</small>
    </div>
  );
}

function ModuleSwitch({
  label,
  detail,
  enabled,
  onToggle
}: {
  label: string;
  detail: string;
  enabled: boolean;
  onToggle: (enabled: boolean) => Promise<void> | void;
}) {
  const [visualEnabled, setVisualEnabled] = useState(enabled);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (!pending) {
      setVisualEnabled(enabled);
    }
  }, [enabled, pending]);

  const toggle = useCallback(async () => {
    if (pending) {
      return;
    }
    const next = !visualEnabled;
    setVisualEnabled(next);
    setPending(true);
    try {
      await onToggle(next);
    } finally {
      setPending(false);
    }
  }, [onToggle, pending, visualEnabled]);

  return (
    <button
      className={visualEnabled ? "module-switch on" : "module-switch"}
      onClick={() => void toggle()}
      disabled={pending}
      type="button"
    >
      <span>
        <b>{label}</b>
        <small>{detail}</small>
      </span>
      <i>{visualEnabled ? "开" : "关"}</i>
    </button>
  );
}

type PreviewDetection = {
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
  score: number;
  className: string;
  index: number;
};

const PREVIEW_OVERLAY_HOLD_MS = 100;

type PreviewOverlaySnapshot = {
  detections: PreviewDetection[];
  target: Record<string, unknown>;
  updatedAtMs: number;
};

function useStablePreviewOverlay(
  detections: PreviewDetection[],
  target: Record<string, unknown>
): { detections: PreviewDetection[]; target: Record<string, unknown> } {
  const snapshotRef = useRef<PreviewOverlaySnapshot | null>(null);
  const nowMs = Date.now();
  if (detections.length > 0) {
    snapshotRef.current = { detections, target, updatedAtMs: nowMs };
    return { detections, target };
  }
  const previous = snapshotRef.current;
  if (previous !== null && nowMs - previous.updatedAtMs <= PREVIEW_OVERLAY_HOLD_MS) {
    return { detections: previous.detections, target: previous.target };
  }
  return { detections, target };
}

function readPreviewDetections(value: unknown): PreviewDetection[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map(asRecord).flatMap((item, index) => {
    const displayBox = asRecord(item.display_box);
    const roiBox = asRecord(item.roi_box);
    const box = Object.keys(displayBox).length > 0
      ? displayBox
      : Object.keys(roiBox).length > 0
        ? roiBox
        : item;
    const x = finiteNumber(box.x);
    const y = finiteNumber(box.y);
    const w = finiteNumber(box.w);
    const h = finiteNumber(box.h);
    if (x === null || y === null || w === null || h === null || w <= 0 || h <= 0) {
      return [];
    }
    return [{
      x,
      y,
      w,
      h,
      cx: finiteNumber(box.cx) ?? x + w / 2,
      cy: finiteNumber(box.cy) ?? y + h / 2,
      score: finiteNumber(item.score) ?? 0,
      className: readString(item.class_name, readString(item.className, `类别${readNumber(item.class_id, 0)}`)),
      index
    }];
  });
}

function percent(value: number, total: number): string {
  return `${clampNumber(total > 0 ? (value / total) * 100 : 0, 0, 100)}%`;
}

function formatModelSizeMb(value: unknown): string {
  const sizeBytes = finiteNumber(value);
  if (sizeBytes === null || sizeBytes < 0) {
    return "大小不可用";
  }
  return `${(sizeBytes / 1_000_000).toFixed(2)} MB`;
}

function findCatalogModelPath(
  directory: ModelCatalogDirectory,
  artifactId: number
): string | undefined {
  for (const child of directory.children) {
    if (child.type === "model") {
      if (child.artifact_id === artifactId) {
        return child.relative_path;
      }
      continue;
    }
    const nested = findCatalogModelPath(child, artifactId);
    if (nested) {
      return nested;
    }
  }
  return undefined;
}

function PreviewFrame({
  enabled,
  imageAvailable,
  unavailableReason,
  runtime,
  roiSize
}: {
  enabled: boolean;
  imageAvailable: boolean;
  unavailableReason: string;
  runtime: RuntimeState | null;
  roiSize: number;
}) {
  const configVersion = typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const vision = asRecord(runtime?.vision);
  const inferenceTrace = asRecord(vision.inference);
  const control = asRecord(vision.control);
  const previewWidth = readNumber(inferenceTrace.input_width, roiSize);
  const previewHeight = readNumber(inferenceTrace.input_height, roiSize);
  const displaySize = Math.max(previewWidth, previewHeight, roiSize);
  const liveDetections = readPreviewDetections(vision.detection_items);
  const liveTarget = asRecord(vision.target);
  const overlay = useStablePreviewOverlay(liveDetections, liveTarget);
  const detections = overlay.detections;
  const target = overlay.target;
  const mouseObservation = asRecord(target.mouse_observation ?? control.mouse_observation);
  const rawAim = asRecord(mouseObservation.raw_aim);
  const targetDetectionIndex = readNullableNumber(target.target_detection_index);
  const targetCx = readNullableNumber(target.cx);
  const targetCy = readNullableNumber(target.cy);
  const targetAimX = readNullableNumber(mouseObservation.predicted_aim_x_roi_px)
    ?? readNullableNumber(rawAim.aim_roi_x_px)
    ?? targetCx;
  const targetAimY = readNullableNumber(mouseObservation.predicted_aim_y_roi_px)
    ?? readNullableNumber(rawAim.aim_roi_y_px)
    ?? targetCy;
  const showImage = enabled && imageAvailable;
  const showOverlay = enabled && detections.length > 0;
  const selectedDetection = detections.find((item) => (
    targetDetectionIndex !== null
      ? item.index === targetDetectionIndex
      : targetCx !== null && targetCy !== null && Math.abs(item.cx - targetCx) <= 2 && Math.abs(item.cy - targetCy) <= 2
  ));
  const selectedIndex = selectedDetection?.index ?? null;
  const sourceWidth = readNumber(inferenceTrace.source_width, 0);
  const sourceHeight = readNumber(inferenceTrace.source_height, 0);
  const roiOffsetX = readNumber(inferenceTrace.roi_offset_x, 0);
  const roiOffsetY = readNumber(inferenceTrace.roi_offset_y, 0);
  const centerX = clampNumber(
    sourceWidth > 0 ? sourceWidth / 2 - roiOffsetX : previewWidth / 2,
    0,
    previewWidth
  );
  const centerY = clampNumber(
    sourceHeight > 0 ? sourceHeight / 2 - roiOffsetY : previewHeight / 2,
    0,
    previewHeight
  );
  return (
    <div
      className={showOverlay ? "console-preview has-overlay" : "console-preview"}
      style={{
        "--roi-size": `${displaySize}px`,
        "--preview-aspect": `${previewWidth} / ${previewHeight}`
      } as CSSProperties}
    >
      <div className="console-preview-frame">
        {showImage ? <img alt="实时画面 / ROI" src={streamUrl(configVersion, configVersion)} /> : null}
        {enabled && !showImage ? <div className="console-preview-unavailable">{unavailableReason}</div> : null}
        {showOverlay ? (
          <div className="console-detection-layer" aria-hidden="true">
            <svg className="console-target-lines" viewBox={`0 0 ${previewWidth} ${previewHeight}`} preserveAspectRatio="none">
              {detections.map((item) => {
                const selected = item.index === selectedIndex;
                const endX = selected && targetAimX !== null ? targetAimX : item.cx;
                const endY = selected && targetAimY !== null ? targetAimY : item.cy;
                return (
                  <line
                    className={selected ? "primary" : "secondary"}
                    key={`line-${item.index}-${item.className}`}
                    x1={centerX}
                    y1={centerY}
                    x2={clampNumber(endX, 0, previewWidth)}
                    y2={clampNumber(endY, 0, previewHeight)}
                  />
                );
              })}
            </svg>
            <span
              className="console-roi-center"
              style={{
                left: percent(centerX, previewWidth),
                top: percent(centerY, previewHeight)
              }}
            />
            {detections.map((item) => {
              const selected = item.index === selectedIndex;
              return (
                <span
                  className={selected ? "console-detection-box primary" : "console-detection-box secondary"}
                  key={`box-${item.index}-${item.className}`}
                  style={{
                    left: percent(item.x, previewWidth),
                    top: percent(item.y, previewHeight),
                    width: percent(item.w, previewWidth),
                    height: percent(item.h, previewHeight)
                  }}
                >
                  <b>{selected ? "主要目标" : "其他目标"} · {item.className} {item.score.toFixed(2)}</b>
                </span>
              );
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function KvCard({ title, rows, notice }: { title: string; rows: [string, string][]; notice?: ReactNode }) {
  return (
    <div className="console-card">
      <SectionTitle icon={cardIconForTitle(title)} title={title} />
      {notice}
      <div className="console-kv">
        {rows.map(([key, value]) => (
          <Fragment key={key}>
            <span>{key}</span>
            <b>{value}</b>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function Bar({ label, width }: { label: string; width: number }) {
  return (
    <>
      <label>{label}</label>
      <div className="console-bar"><i style={{ width: `${width}%` }} /></div>
    </>
  );
}

function Event({ label, value, width }: { label: string; value: string; width: number }) {
  return (
    <div className="console-event">
      <span>{label}</span>
      <div className="console-bar"><i style={{ width: `${width}%` }} /></div>
      <b>{value} ms</b>
    </div>
  );
}
