import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureState,
  CaptureSelectPayload,
  connectKmNet,
  diagnosticCircleKmNet,
  diagnosticMoveKmNet,
  disconnectKmNet,
  clearCrosshairTemplate,
  crosshairTemplatePreviewUrl,
  getRuntimeState,
  learnCrosshair,
  HealthResponse,
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelCatalogResponse,
  ModelProject,
  ModelVersion,
  MotionProfile,
  MotionProfileRuntime,
  ParserPresetId,
  RuntimeConfig,
  RuntimeConfigValue,
  RuntimeState,
  activateMotionProfile,
  disableMotionProfile,
  getCaptureCapabilities,
  getModelArtifacts,
  getModelCatalog,
  getModelVersions,
  getMotionProfileRuntime,
  getMotionProfiles,
  publishModel,
  registerCatalogModel,
  selectCaptureProfile,
  setCapturePreviewEnabled,
  startRuntimePipeline,
  stopCapture,
  stopRuntimePipeline,
  streamUrl,
  updateRuntimeConfig,
  updateRuntimeConfigField
} from "../../api";
import { reportError, useClearErrorNotices, useErrorNotices } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";
import { NovaIcon, StatusBadge, ThemeGallery, ThemeToggle } from "../../components/visual";
import { ModelSelectionPanel } from "../models/ModelSelectionPanel";
import { ModelSwitchDialog, type ModelSwitchDialogStatus } from "../models/ModelSwitchDialog";
import { formatModelSize } from "../models/modelPresentation";
import {
  runtimeDeliveryDescription,
  runtimeDeliveryLabel,
  runtimeDeliveryTone,
  type RuntimeDeliveryStatus
} from "../shared/runtimeDelivery";
import { AdvancedSettingsDialog } from "./AdvancedSettingsDialog";
import { AimTargetRange, type AimRole, type AimRoleRatios } from "./AimTargetRange";
import { CommitNumberControl, InlineTextControl, NumberControl, TextControl } from "./StudioControls";
import { CONSOLE_PAGES, DEFAULT_CONSOLE_PAGE, StudioNavigation, type ConsolePage } from "./StudioNavigation";
import { StudioPageHeader } from "./StudioPageHeader";
import { Event, KvCard, Metric, SectionTitle } from "./StudioPresentation";
import { trapDialogTabKey } from "./dialogFocus";
import "./studio-settings.css";

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
  realtimeStatus: RuntimeDeliveryStatus;
  onRefresh: () => Promise<void>;
  onRuntimeConfigChange: (config: RuntimeConfig) => void;
  onRuntimeStateChange: (runtime: RuntimeState) => void;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type LaunchStatus = "idle" | "running" | "success" | "failed" | "cancelled";
type LaunchStepState = "pending" | "running" | "success" | "failed";

type LaunchStage = {
  title: string;
  caption: string;
};

const RUNTIME_MAINLINE_BACKENDS = new Set(["deepstream_nvinfer"]);
const LAUNCH_STATUS_REQUEST_TIMEOUT_MS = 15000;

// Output delivery is configured and connected independently. Mainline launch
// only proves capture, inference and mouse-algorithm consumption are ready.
const MAINLINE_LAUNCH_STAGES_CUSTOM_TENSORRT: LaunchStage[] = [
  {
    title: "检查运行环境",
    caption: "确认 Studio 已连接到 Jetson 运行服务。"
  },
  {
    title: "应用采集配置",
    caption: "按当前设备、格式、分辨率与帧率选择采集配置。"
  },
  {
    title: "启动主链运行管线",
    caption: "请求后端启动采集、ROI、推理与 DetectionBatch 数据通路。"
  },
  {
    title: "激活鼠标算法",
    caption: "确认目标选择、跟踪、预测与鼠标算法已开始消费 DetectionBatch；输出设备不影响本步骤。"
  }
];

function resolveLaunchStepState(
  index: number,
  status: LaunchStatus,
  activeIndex: number,
  completedStages: number
): LaunchStepState {
  if (status === "success" || index < completedStages) {
    return "success";
  }
  if (status === "failed" && index === activeIndex) {
    return "failed";
  }
  if (status === "running" && index === activeIndex) {
    return "running";
  }
  return "pending";
}

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

function profileRoleRecords(value: unknown): Record<string, Record<string, AimRole>> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).map(([profileName, rawValues]) => [
      profileName,
      Object.fromEntries(
        Object.entries(asRecord(rawValues)).flatMap(([classId, role]) =>
          role === "head" || role === "body" || role === "other"
            ? [[classId, role]]
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

function parseDetectionClassFilter(value: string): Set<number> | null {
  const normalized = value.trim().toLowerCase();
  if (normalized === "all") {
    return null;
  }
  if (normalized === "none") {
    return new Set();
  }
  const classIds = parseClassPriority(normalized);
  return classIds.length > 0 ? new Set(classIds) : null;
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

function formatRecoilBlockReason(value: unknown): string {
  const reason = readString(value);
  const labels: Record<string, string> = {
    RECOIL_DISABLED: "关闭",
    LEFT_TRIGGER_INACTIVE: "等待真实左键",
    LEFT_TRIGGER_HOLD_INVALID: "左键状态无效",
    RECOIL_START_DELAY: "等待启动延迟",
    RECOIL_COUNTS_ZERO: "固定强度为零",
  };
  return labels[reason] ?? (reason || "等待真实左键或启动延迟");
}

function crosshairStateLabel(value: string): string {
  return {
    idle: "尚未运行",
    disabled: "已关闭",
    searching: "等待模板或匹配",
    candidate: "正在确认",
    confirmed: "已确认",
    uncertain: "短时丢失，保持已确认中心"
  }[value] ?? value;
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
  onRefresh,
  onRuntimeConfigChange,
  onRuntimeStateChange
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
  const [parserPreset, setParserPreset] = useState<ParserPresetId>("auto");
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [modelArtifacts, setModelArtifacts] = useState<ModelArtifact[]>([]);
  const [modelCatalogRefreshKey, setModelCatalogRefreshKey] = useState(0);
  const [modelDetailsRefreshKey, setModelDetailsRefreshKey] = useState(0);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogDirectory | null>(null);
  const [modelCatalogModelCount, setModelCatalogModelCount] = useState(0);
  const [modelCatalogDirectoryCount, setModelCatalogDirectoryCount] = useState(0);
  const [modelCatalogLoading, setModelCatalogLoading] = useState(false);
  const [motionProfiles, setMotionProfiles] = useState<MotionProfile[]>([]);
  const [motionProfileRuntime, setMotionProfileRuntime] = useState<MotionProfileRuntime | null>(null);
  const [selectedMotionProfileId, setSelectedMotionProfileId] = useState("");
  const [motionProfileBusy, setMotionProfileBusy] = useState(false);
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
  const [errorCenterOpen, setErrorCenterOpen] = useState(false);
  const errorNotices = useErrorNotices();
  const clearErrorNotices = useClearErrorNotices();
  const [modelSwitchMessage, setModelSwitchMessage] = useState("");
  const [modelSwitchDialogOpen, setModelSwitchDialogOpen] = useState(false);
  const [modelSwitchDialogStatus, setModelSwitchDialogStatus] = useState<ModelSwitchDialogStatus>("running");
  const [modelSwitchStageIndex, setModelSwitchStageIndex] = useState(0);
  const [modelSwitchCompletedStages, setModelSwitchCompletedStages] = useState(0);
  const [modelSwitchProgressDetail, setModelSwitchProgressDetail] = useState("");
  const [modelSwitchDialogError, setModelSwitchDialogError] = useState("");
  const [modelCatalogMessage, setModelCatalogMessage] = useState("");
  const [launchDialogOpen, setLaunchDialogOpen] = useState(false);
  const [classConfigDialogOpen, setClassConfigDialogOpen] = useState(false);
  const [targetWeightsDialogOpen, setTargetWeightsDialogOpen] = useState(false);
  const [algorithmSettingsDialogOpen, setAlgorithmSettingsDialogOpen] = useState(false);
  const [targetAdvancedDialogOpen, setTargetAdvancedDialogOpen] = useState(false);
  const [trackerSettingsDialogOpen, setTrackerSettingsDialogOpen] = useState(false);
  const [newClassProfileName, setNewClassProfileName] = useState("");
  const [renamedClassProfileName, setRenamedClassProfileName] = useState("");
  const [classProfileDeleteArmed, setClassProfileDeleteArmed] = useState(false);
  const [launchStatus, setLaunchStatus] = useState<LaunchStatus>("idle");
  const [launchStageIndex, setLaunchStageIndex] = useState(0);
  const [launchCompletedStages, setLaunchCompletedStages] = useState(0);
  const [launchError, setLaunchError] = useState("");
  const [launchProgressDetail, setLaunchProgressDetail] = useState("");
  const [launchToastVisible, setLaunchToastVisible] = useState(false);
  const [mainlineLaunchAccepted, setMainlineLaunchAccepted] = useState(false);
  const [mainlineLaunchMessage, setMainlineLaunchMessage] = useState("");
  const [crosshairMessage, setCrosshairMessage] = useState("");
  const [crosshairPreviewKey, setCrosshairPreviewKey] = useState(0);
  const [previewActiveOverride, setPreviewActiveOverride] = useState<boolean | null>(null);
  const [previewTogglePending, setPreviewTogglePending] = useState(false);
  const [configDraft, setConfigDraft] = useState<RuntimeConfig | null>(() => cloneRuntimeConfig(runtimeConfig));
  const [pendingConfigWriteCount, setPendingConfigWriteCount] = useState(0);
  const [dialogSaveError, setDialogSaveError] = useState<string | null>(null);
  const dialogSaving = busy !== null || pendingConfigWriteCount > 0;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const configDraftRef = useRef<RuntimeConfig | null>(cloneRuntimeConfig(runtimeConfig));
  const loadedModelProjectIdRef = useRef<number | "">("");
  const loadedModelVersionIdRef = useRef<number | "">("");
  const preferLatestModelVersionRef = useRef(false);
  const pendingConfigWritesRef = useRef(0);
  const configWriteSeqRef = useRef(0);
  const launchCancelledRef = useRef(false);
  const launchTimerRef = useRef<number | null>(null);
  const launchTimerResolveRef = useRef<(() => void) | null>(null);
  const classConfigDialogRef = useRef<HTMLElement | null>(null);
  const targetWeightsDialogRef = useRef<HTMLElement | null>(null);
  const errorCenterDialogRef = useRef<HTMLElement | null>(null);
  const modelSwitchDialogRef = useRef<HTMLElement | null>(null);
  const dialogSavingRef = useRef(false);

  useEffect(() => {
    dialogSavingRef.current = dialogSaving;
  }, [dialogSaving]);

  useEffect(() => {
    const anyConfigDialogOpen =
      classConfigDialogOpen ||
      targetWeightsDialogOpen ||
      algorithmSettingsDialogOpen ||
      targetAdvancedDialogOpen ||
      trackerSettingsDialogOpen;
    if (!anyConfigDialogOpen) {
      setDialogSaveError(null);
    }
  }, [
    algorithmSettingsDialogOpen,
    classConfigDialogOpen,
    targetAdvancedDialogOpen,
    targetWeightsDialogOpen,
    trackerSettingsDialogOpen
  ]);

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

  const openMotionProfileStudio = useCallback(() => {
    window.open("?page=motion-profile", "novasight-motion-profile", "popup=yes,width=1280,height=820");
  }, []);

  const refreshMotionProfileRuntime = useCallback(async () => {
    const [profiles, status] = await Promise.all([getMotionProfiles(), getMotionProfileRuntime()]);
    setMotionProfiles(profiles);
    setMotionProfileRuntime(status);
    setSelectedMotionProfileId((current) => {
      if (status.active_profile && profiles.some((profile) => profile.profile_id === status.active_profile)) {
        return status.active_profile;
      }
      if (current && profiles.some((profile) => profile.profile_id === current)) {
        return current;
      }
      return profiles[0]?.profile_id ?? "";
    });
  }, []);

  useEffect(() => {
    if (activePage !== "params") {
      return;
    }
    const refresh = () => {
      void refreshMotionProfileRuntime().catch((error) => {
        reportError(error, { source: "motion-profile-runtime", title: "真人轨迹状态读取失败" });
      });
    };
    refresh();
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, [activePage, refreshMotionProfileRuntime]);

  const setMotionControlMode = useCallback(async (useHumanProfile: boolean) => {
    if (motionProfileBusy) {
      return;
    }
    if (useHumanProfile && !selectedMotionProfileId) {
      openMotionProfileStudio();
      return;
    }
    setMotionProfileBusy(true);
    try {
      if (useHumanProfile) {
        await activateMotionProfile(selectedMotionProfileId);
      } else {
        await disableMotionProfile();
      }
      await refreshMotionProfileRuntime();
    } catch (error) {
      reportError(error, { source: "motion-profile-mode", title: "控制轨迹切换失败" });
    } finally {
      setMotionProfileBusy(false);
    }
  }, [motionProfileBusy, openMotionProfileStudio, refreshMotionProfileRuntime, selectedMotionProfileId]);

  const selectMotionProfile = useCallback(async (profileId: string) => {
    setSelectedMotionProfileId(profileId);
    if (!motionProfileRuntime?.enabled || !profileId) {
      return;
    }
    setMotionProfileBusy(true);
    try {
      await activateMotionProfile(profileId);
      await refreshMotionProfileRuntime();
    } catch (error) {
      reportError(error, { source: "motion-profile-select", title: "真人画像切换失败" });
    } finally {
      setMotionProfileBusy(false);
    }
  }, [motionProfileRuntime?.enabled, refreshMotionProfileRuntime]);

  useEffect(() => {
    if (!modelSwitchDialogOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => modelSwitchDialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && modelSwitchDialogStatus !== "running") {
        setModelSwitchDialogOpen(false);
      } else {
        trapDialogTabKey(event, modelSwitchDialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [modelSwitchDialogOpen, modelSwitchDialogStatus]);

  useEffect(() => {
    if (!errorCenterOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => errorCenterDialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setErrorCenterOpen(false);
      } else {
        trapDialogTabKey(event, errorCenterDialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [errorCenterOpen]);

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
    if (!classConfigDialogOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => classConfigDialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!dialogSavingRef.current) {
          setClassConfigDialogOpen(false);
        }
      } else {
        trapDialogTabKey(event, classConfigDialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [classConfigDialogOpen]);

  useEffect(() => {
    if (!targetWeightsDialogOpen) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => targetWeightsDialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!dialogSavingRef.current) {
          setTargetWeightsDialogOpen(false);
        }
      } else {
        trapDialogTabKey(event, targetWeightsDialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [targetWeightsDialogOpen]);

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
  const crosshairConfig = nestedRecord(config, "crosshair");
  const limitsConfig = nestedRecord(config, "limits");
  const inferenceConfig = nestedRecord(config, "inference");
  const preprocessConfig = nestedRecord(config, "preprocess");
  const controlConfig = nestedRecord(config, "control");
  const hardwareConfig = nestedRecord(config, "hardware");
  const powerSavingConfig = nestedRecord(config, "power_saving");
  const consumersConfig = nestedRecord(config, "consumers");
  const vision = asRecord(runtime?.vision);
  const crosshairStatus = asRecord(vision.crosshair);
  const crosshairObservation = asRecord(crosshairStatus.observation);
  const crosshairTemplate = asRecord(crosshairStatus.template);
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
  const runtimePowerSaving = asRecord(runtime?.power_saving);
  const runtimePowerMode = readString(runtimePowerSaving.mode, "disabled");
  const runtimePowerRunIntent = readBoolean(runtimePowerSaving.run_intent);
  const runtimePowerAutoResume = readBoolean(runtimePowerSaving.auto_resume, true);
  const runtimePowerReason = readString(runtimePowerSaving.reason, "");
  const hostPresenceStandby = runtimePowerMode === "cold_standby" && runtimePowerRunIntent;
  const hostPresenceGrace = runtimePowerMode === "grace";
  const runtimePowerInterrupted = runtimePowerMode === "interrupted";
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
      : runtimePowerInterrupted
        ? "主链意外停止"
      : hostPresenceStandby
        ? "主机离线省流待机"
      : hostPresenceGrace
        ? "主机心跳中断 · 宽限运行"
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
      : runtimePowerInterrupted
        ? "需要人工处理"
      : hostPresenceStandby
        ? "推理已暂停"
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
  const runtimeControlRequested = captureMainRunning || (
    runtimeMainlineSelected && runtimePowerRunIntent
  );
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
  const detectedChoices = useMemo(() => groupCapabilities(caps?.capabilities ?? []), [caps]);
  const choices = useMemo(() => {
    if (detectedChoices.length > 0) {
      return detectedChoices;
    }
    if (configuredCaptureProfile) {
      return [configuredCaptureProfile];
    }
    if (selectedProfile) {
      return [{
        pixel_format: selectedProfile.pixel_format.toUpperCase(),
        width: selectedProfile.width,
        height: selectedProfile.height,
        fps: selectedProfile.fps
      }];
    }
    return [];
  }, [configuredChoiceId, detectedChoices, runningChoiceId]);
  const selectedChoice =
    choices.find((choice) => choiceId(choice) === selectedChoiceId) ??
    choices.find((choice) => choiceId(choice) === configuredChoiceId) ??
    choices.find((choice) => choiceId(choice) === runningChoiceId) ??
    choices[0];
  const roiSize = readNumber(roiConfig.size, 640);
  const crosshairEnabled = readBoolean(crosshairConfig.enabled, false);
  const crosshairUseForControl = readBoolean(crosshairConfig.use_for_control, false);
  const crosshairSearchSize = readNumber(crosshairConfig.search_size, 96);
  const crosshairSampleHz = readNumber(crosshairConfig.sample_hz, 10);
  const crosshairSampleFrames = readNumber(crosshairConfig.sample_frames, 5);
  const crosshairState = readString(crosshairStatus.state, "idle");
  const crosshairTemplateId = readString(crosshairTemplate.id, "");
  const crosshairRecentSamples = readNumber(crosshairStatus.recent_samples, 0);
  const crosshairRequiredSamples = readNumber(crosshairStatus.required_samples, crosshairSampleFrames);
  const crosshairConfidence = readNumber(crosshairObservation.confidence, 0);
  const crosshairOffsetX = readNumber(crosshairObservation.offset_x, 0);
  const crosshairOffsetY = readNumber(crosshairObservation.offset_y, 0);
  const crosshairReferenceReady = readBoolean(crosshairStatus.control_reference_ready, false);
  const crosshairBranchActive = readBoolean(deepstreamStatus.crosshair_active, false);
  const crosshairBranchReason = readString(deepstreamStatus.crosshair_reason, "");
  const crosshairProcessingError = readString(crosshairStatus.last_error, "");
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
    setRenamedClassProfileName(activeDetectionProfile);
    setClassProfileDeleteArmed(false);
  }, [activeDetectionProfile]);
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
  const rawAimRoleRatios = nestedRecord(aimConfig, "role_y_ratios");
  const aimRoleRatios: AimRoleRatios = {
    head: clampNumber(readNumber(rawAimRoleRatios.head, 0.22), 0, 1),
    body: clampNumber(readNumber(rawAimRoleRatios.body, 0.22), 0, 1),
    other: clampNumber(readNumber(rawAimRoleRatios.other, 0.22), 0, 1)
  };
  const classRoleProfiles = profileRoleRecords(aimConfig.class_roles);
  const activeClassRoles = classRoleProfiles[activeDetectionProfile] ?? {};
  const targetFovRadiusPx = readNumber(controlConfig.target_fov_radius_px, 180);
  const candidateRatioMaxAspect = readNumber(controlConfig.candidate_ratio_max_aspect, 6);
  const candidateQualityConfidenceWeight = readNumber(controlConfig.candidate_quality_confidence_weight, 0.7);
  const candidateQualityAreaWeight = readNumber(controlConfig.candidate_quality_area_weight, 0.3);
  const candidateSelectionClassWeight = readNumber(controlConfig.candidate_selection_class_weight, 0.55);
  const candidateSelectionQualityWeight = readNumber(controlConfig.candidate_selection_quality_weight, 0.05);
  const candidateSelectionDistanceWeight = readNumber(controlConfig.candidate_selection_distance_weight, 0.40);
  const candidateQualityWeightTotal = candidateQualityConfidenceWeight + candidateQualityAreaWeight;
  const candidateSelectionWeightTotal = candidateSelectionClassWeight + candidateSelectionQualityWeight + candidateSelectionDistanceWeight;
  const normalizedQualityConfidenceWeight = candidateQualityWeightTotal > 0
    ? candidateQualityConfidenceWeight / candidateQualityWeightTotal
    : 1;
  const normalizedQualityAreaWeight = candidateQualityWeightTotal > 0
    ? candidateQualityAreaWeight / candidateQualityWeightTotal
    : 0;
  const normalizedSelectionClassWeight = candidateSelectionWeightTotal > 0
    ? candidateSelectionClassWeight / candidateSelectionWeightTotal
    : 0;
  const normalizedSelectionQualityWeight = candidateSelectionWeightTotal > 0
    ? candidateSelectionQualityWeight / candidateSelectionWeightTotal
    : 1;
  const normalizedSelectionDistanceWeight = candidateSelectionWeightTotal > 0
    ? candidateSelectionDistanceWeight / candidateSelectionWeightTotal
    : 0;
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
  const dualPhaseFarKp = readNumber(dualPhaseFarConfig.kp, 0.90);
  const dualPhaseNearKp = readNumber(dualPhaseNearConfig.kp, 0.30);
  const dualPhaseAtanScale = readNumber(dualPhaseAtanConfig.scale_counts, 1024);
  const dualPhaseFarMaxCounts = readNumber(dualPhaseFarConfig.max_counts_per_update, 600);
  const dualPhaseNearMaxCounts = readNumber(dualPhaseNearConfig.max_counts_per_update, 120);
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
  const recoilYCountsPerObservation = readNumber(sharedControlConfig.recoil_y_counts_per_observation, 0);
  const triggerMode = readString(controlConfig.trigger_mode, "always");
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);
  const kmnetAutoConnect = readBoolean(hardwareConfig.auto_connect, true);
  const hostPresencePowerSavingEnabled = readBoolean(
    powerSavingConfig.host_presence_enabled,
    false
  );
  const targetHostId = readString(powerSavingConfig.target_host_id, "");
  const hostHeartbeatTimeoutS = readNumber(powerSavingConfig.heartbeat_timeout_s, 6);
  const hostOfflineGraceS = readNumber(powerSavingConfig.offline_grace_s, 15);
  const hostAutoResume = readBoolean(powerSavingConfig.auto_resume, true);
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
  const kmnetConnectionState = readString(
    kmnetStatus.connection_state,
    kmnetConnected ? "connected" : kmnetConnecting ? "connecting" : "disconnected"
  );
  const kmnetConnectionFailed = kmnetConnectionState === "failed";
  const kmnetConnectionDegraded = kmnetConnectionState === "degraded";
  const kmnetRetryable = kmnetStatus.retryable === true;
  const kmnetLastError = readString(kmnetStatus.last_error, "");
  const kmnetConnectionLabel = kmnetConnected
    ? kmnetConnectionDegraded ? "已连接，监听异常" : "已连接"
    : kmnetConnecting
      ? "连接中"
      : kmnetConnectionFailed
        ? "连接失败"
        : "未连接";
  const kmnetConnectionActionLabel = kmnetConnected
    ? "断开 kmNet"
    : kmnetConnecting
      ? "取消连接"
      : kmnetConnectionFailed
        ? kmnetRetryable ? "重新连接" : "驱动不可用"
        : "连接 kmNet";
  const kmnetDiagnosticDisabled = !kmnetConnected || busy === "kmnet.diagnostic" || busy === "kmnet.circle";
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
  const controlObservationFps = readNumber(statistics?.control_observation_fps, 0);
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
  const selectedCatalogModel = useMemo(
    () => modelCatalog && selectedModelCatalogPath
      ? findCatalogModelByPath(modelCatalog, selectedModelCatalogPath)
      : null,
    [modelCatalog, selectedModelCatalogPath]
  );
  const selectedCatalogArtifactMatches = selectedCatalogModel === null ||
    selectedCatalogModel.artifact_id === selectedSwitchArtifact?.id;
  const selectedPreviewArtifact = selectedCatalogArtifactMatches ? selectedSwitchArtifact : null;
  const selectedPreviewVersion =
    typeof selectedModelVersionId === "number" &&
    (selectedCatalogModel === null || selectedCatalogModel.version_id === selectedModelVersionId)
      ? modelVersions.find((item) => item.id === selectedModelVersionId) ?? null
      : null;
  const preferredSwitchArtifact = sortedSwitchableArtifacts[0] ?? null;
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
  const orderedClassEditorIds = [
    ...classPriorityIds.filter((classId) => classEditorIds.includes(classId)),
    ...classEditorIds.filter((classId) => !classPriorityIds.includes(classId))
  ];
  const configuredDetectionClassIds = parseDetectionClassFilter(activeDetectionClass);
  const selectedDetectionClassIds = configuredDetectionClassIds ?? new Set(classEditorIds);
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
  const basicCandidateFilter = asRecord(controlCandidateFilter.basic);
  const effectiveClassFilter = readString(controlCandidateFilter.effective_class_filter, activeDetectionClass);
  const rejectedBasicClassIds = Array.isArray(basicCandidateFilter.rejected_class_ids)
    ? basicCandidateFilter.rejected_class_ids
        .map((item) => Number(item))
        .filter((item) => Number.isInteger(item) && item >= 0 && item <= 255)
    : [];
  const recoveryClassIdSet = new Set([
    ...(configuredDetectionClassIds ?? []),
    ...runtimeDetectionClassIds
  ]);
  const detectedClassFilterValue = orderedClassEditorIds
    .filter((classId) => recoveryClassIdSet.has(classId))
    .join(",");
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
  const runtimePreviewActive = runtimeInference.preview_active === true;
  const previewActive = previewActiveOverride ?? runtimePreviewActive;
  const previewImageAvailable = deepstreamNvinferSelected
    ? deepstreamPreviewStreamReady && previewActive
    : runtime?.capture?.available === true;
  const previewUnavailableReason = deepstreamNvinferSelected
    ? readString(runtimeInference.preview_reason, "等待 DeepStream 硬件预览帧")
    : "预览帧尚不可用";

  useEffect(() => {
    if (previewActiveOverride === runtimePreviewActive) {
      setPreviewActiveOverride(null);
    }
  }, [previewActiveOverride, runtimePreviewActive]);

  const updatePreviewActive = useCallback(async (
    enabled: boolean,
    options?: { quiet?: boolean; keepalive?: boolean }
  ) => {
    if (!deepstreamPreviewStreamReady || previewTogglePending) {
      return;
    }
    setPreviewTogglePending(true);
    setPreviewActiveOverride(enabled);
    try {
      const inference = await setCapturePreviewEnabled(enabled, {
        keepalive: options?.keepalive
      });
      if (runtime) {
        onRuntimeStateChange({
          ...runtime,
          inference: {
            ...runtime.inference,
            ...inference
          }
        });
      }
    } catch (error) {
      setPreviewActiveOverride(null);
      if (!options?.quiet) {
        setLocalError(`预览状态切换失败：${getErrorMessage(error)}`);
        reportError(error, { source: "studio-preview", title: "预览切换失败" });
      }
    } finally {
      setPreviewTogglePending(false);
    }
  }, [
    deepstreamPreviewStreamReady,
    onRuntimeStateChange,
    previewTogglePending,
    runtime
  ]);

  useEffect(() => {
    const releasePreview = () => {
      if (previewActive) {
        void updatePreviewActive(false, { quiet: true, keepalive: true });
      }
    };
    if (activePage !== "infer") {
      releasePreview();
    }
    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") {
        releasePreview();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [activePage, previewActive, updatePreviewActive]);
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
    latestFrameBroker.published_frames ?? deepstreamStatus.capture_frames ?? captureStatistics.published_frames
  );
  const captureDroppedFrames = readNullableNumber(
    deepstreamStatus.stale_dropped_batches ?? captureStatistics.dropped_counter ?? capture?.frames_dropped
  );
  const captureFramePeriodMs = readNullableNumber(
    deepstreamStatus.last_capture_interval_ms ?? capture?.frame_period_ms
  );
  const captureSourceFps = readNullableNumber(
    deepstreamStatus.capture_fps ?? capture?.fps_capture ?? captureStatistics.capture_fps
  );
  const nvinferInputFps = readNullableNumber(
    deepstreamStatus.input_fps ?? captureStatistics.nvinfer_input_fps
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
  const inferenceRuntimePrecision = readString(
    runtimeInference.runtime_precision,
    readString(deepstreamStatus.runtime_precision, "")
  );
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
  const currentErrorDetails = useMemo(() => {
    const items: Array<{ key: string; title: string; detail: string; time?: number }> = [];
    for (const notice of errorNotices) {
      items.push({
        key: `notice-${notice.id}`,
        title: notice.title,
        detail: notice.detail || notice.source,
        time: notice.createdAt
      });
    }
    for (const [source, detail] of Object.entries(errors)) {
      if (detail) items.push({ key: `state-${source}`, title: `${source} 通道异常`, detail });
    }
    if (localError) items.push({ key: "local", title: "当前操作未完成", detail: localError });
    if (capture?.last_error) items.push({ key: "capture", title: "采集链路异常", detail: capture.last_error });
    if (lastModelSwitchError) items.push({ key: "model-switch", title: "模型切换异常", detail: lastModelSwitchError });
    if (runtimePowerInterrupted) {
      items.push({
        key: "runtime-power",
        title: "主链运行被中断",
        detail: runtimePowerReason || "主链并非由省流策略停止，请检查运行管线。"
      });
    }
    return items.filter((item, index, all) => (
      all.findIndex((candidate) => candidate.title === item.title && candidate.detail === item.detail) === index
    ));
  }, [capture?.last_error, errorNotices, errors, lastModelSwitchError, localError, runtimePowerInterrupted, runtimePowerReason]);

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

  const applyModelCatalogResult = useCallback((result: ModelCatalogResponse) => {
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
  }, []);

  useEffect(() => {
    if (activePage !== "infer") {
      return undefined;
    }
    let cancelled = false;
    setModelCatalogLoading(true);
    getModelCatalog()
      .then(async (result) => {
        if (cancelled) {
          return;
        }
        applyModelCatalogResult(result);
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
  }, [activePage, applyModelCatalogResult, modelCatalogRefreshKey, onRefresh]);

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
    if (activePage !== "infer") {
      return undefined;
    }
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

          reportError(err, { source: "model-versions", title: "模型版本读取失败" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activePage, modelDetailsRefreshKey, runtime?.active_model?.version?.id, selectedModelProjectId]);

  useEffect(() => {
    if (activePage !== "infer") {
      return undefined;
    }
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

          reportError(err, { source: "model-artifacts", title: "模型产物读取失败" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activePage, modelDetailsRefreshKey, selectedModelVersionId]);

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
        const stoppedState = await stopRuntimePipeline();
        onRuntimeStateChange(stoppedState);
      } else {
        await stopCapture();
      }
      await onRefresh();
    } catch (err) {
      setLocalError(getErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }, [runtimeMainlineSelected, onRefresh, onRuntimeStateChange]);

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
      const state = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
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
      const state = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
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
    let enteredPowerStandby = false;

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
        if (readBoolean(status.standby)) {
          const powerSaving = asRecord(status.power_saving);
          enteredPowerStandby = true;
          setMainlineLaunchAccepted(false);
          setMainlineLaunchMessage(
            readBoolean(powerSaving.auto_resume, true)
              ? readString(powerSaving.reason, "等待目标主机心跳")
              : "等待目标主机上线；自动恢复已关闭，请上线后再次点击启动"
          );
          return;
        }
        const accepted = readBoolean(status.running, true);
        if (!accepted) {
          const reason = readString(status.last_error, "后端未确认主链运行。");
          throw new Error(reason);
        }
        await waitForRuntimeMainlineReady("启动主链运行管线");
        setMainlineLaunchAccepted(true);
        setMainlineLaunchMessage("主链启动请求已提交，正在等待后端状态确认。");
      });
      if (enteredPowerStandby) {
        setLaunchStatus("success");
        setLaunchCompletedStages(3);
        setLaunchProgressDetail(
          hostAutoResume
            ? "启动意图已保存；等待目标主机心跳后自动启动主链。"
            : "启动意图已保存；目标主机上线后需要再次点击启动。"
        );
        await onRefresh();
        return;
      }
      await runStage(3, async () => {
        await waitForRuntimeEvidence(
          "激活鼠标算法",
          (state) => getRuntimeMainlineStatus(state).hasRuntimeConsumption,
          "runtime 尚未消费 DetectionBatch，目标选择、跟踪、预测与鼠标算法没有输入。"
        );
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
        const stoppedState = await stopRuntimePipeline();
        onRuntimeStateChange(stoppedState);
      } catch {
        // Keep the original launch error visible; refresh below exposes stop failures if backend reports them.
      }
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [
    assertCaptureLaunchState,
    buildCapturePayload,
    hostAutoResume,
    launchStatus,
    onRefresh,
    onRuntimeStateChange,
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
      const stoppedState = await stopRuntimePipeline();
      onRuntimeStateChange(stoppedState);
      await onRefresh();
    } catch (err) {
      setLocalError(`取消启动失败：${getErrorMessage(err)}`);

      reportError(err, { source: "mainline-cancel", title: "取消启动失败" });
    }
  }, [closeLaunchDialog, launchStatus, onRefresh, onRuntimeStateChange]);

  const toggleCapture = useCallback(async () => {
    if (runtimeControlRequested) {
      await stopCurrentCapture();
      return;
    }
    if (runtimeMainlineSelected) {
      openMainlineLaunchDialog();
      return;
    }
    await applyCapture();
  }, [applyCapture, runtimeControlRequested, runtimeMainlineSelected, openMainlineLaunchDialog, stopCurrentCapture]);

  const updateConfigField = useCallback(
    async (section: string, key: string, value: RuntimeConfigValue) => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const next = base ? normalizeRuntimeConfig(base) : null;
      if (!next) {
        return;
      }
      const writeSeq = ++configWriteSeqRef.current;
      pendingConfigWritesRef.current += 1;
      setPendingConfigWriteCount((count) => count + 1);
      setDialogSaveError(null);
      const keepsEditorInteractive = section === "control" && key === "aim";
      if (!keepsEditorInteractive) {
        setBusy(`${section}.${key}`);
      }
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
          onRuntimeConfigChange(applied);
        }
      } catch (err) {
        const message = getErrorMessage(err);
        setLocalError(`配置同步失败：${message}`);
        setDialogSaveError(message);

        reportError(err, { source: 'studio', title: '操作失败' });
        if (writeSeq === configWriteSeqRef.current) {
          configDraftRef.current = null;
          setConfigDraft(null);
        }
      } finally {
        pendingConfigWritesRef.current = Math.max(0, pendingConfigWritesRef.current - 1);
        setPendingConfigWriteCount((count) => Math.max(0, count - 1));
        if (!keepsEditorInteractive && writeSeq === configWriteSeqRef.current) {
          setBusy(null);
        }
      }
    },
    [onRuntimeConfigChange, runtimeConfig]
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

  const handleLearnCrosshair = useCallback(async () => {
    setBusy("crosshair.learn");
    setCrosshairMessage("");
    setLocalError(null);
    try {
      const result = await learnCrosshair();
      const template = asRecord(result.template);
      setCrosshairPreviewKey(Date.now());
      setCrosshairMessage(`准星模板 ${readString(template.id, "")} 已生成，等待连续观测确认。`);
      await onRefresh();
    } catch (error) {
      const message = getErrorMessage(error);
      setLocalError(`准星学习失败：${message}`);
      reportError(error, { source: "crosshair", title: "准星学习失败" });
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  const handleClearCrosshair = useCallback(async () => {
    setBusy("crosshair.clear");
    setCrosshairMessage("");
    setLocalError(null);
    try {
      await clearCrosshairTemplate();
      setCrosshairPreviewKey(Date.now());
      setCrosshairMessage("准星模板已清除，控制基准已回退到几何中心。");
      await onRefresh();
    } catch (error) {
      const message = getErrorMessage(error);
      setLocalError(`清除准星模板失败：${message}`);
      reportError(error, { source: "crosshair", title: "清除准星模板失败" });
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  useEffect(() => {
    const onCrosshairShortcut = (event: KeyboardEvent) => {
      if (
        event.key !== "F8"
        || activePage !== "params"
        || !crosshairEnabled
        || event.repeat
      ) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, select, textarea, button, [role='dialog']")) {
        return;
      }
      event.preventDefault();
      if (!runtimeMainlineRunning) {
        setCrosshairMessage("请先启动主链，再使用 F8 学习当前准星。");
        return;
      }
      if (crosshairRecentSamples < crosshairRequiredSamples) {
        setCrosshairMessage(`正在积累学习帧：${crosshairRecentSamples}/${crosshairRequiredSamples}`);
        return;
      }
      void handleLearnCrosshair();
    };
    document.addEventListener("keydown", onCrosshairShortcut);
    return () => document.removeEventListener("keydown", onCrosshairShortcut);
  }, [
    activePage,
    crosshairEnabled,
    crosshairRecentSamples,
    crosshairRequiredSamples,
    handleLearnCrosshair,
    runtimeMainlineRunning
  ]);

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

  const updateAimRoleRatio = useCallback(
    async (role: AimRole, ratio: number) => {
      await updateControlGroupField("aim", "role_y_ratios", {
        ...aimRoleRatios,
        [role]: clampNumber(Number(ratio.toFixed(2)), 0, 1)
      } as RuntimeConfigValue);
    },
    [aimRoleRatios, updateControlGroupField]
  );

  const updateClassAimRole = useCallback(
    async (classId: number, role: AimRole) => {
      const nextProfiles = {
        ...classRoleProfiles,
        [activeDetectionProfile]: {
          ...activeClassRoles,
          [String(classId)]: role
        }
      };
      await updateControlGroupField("aim", "class_roles", nextProfiles as RuntimeConfigValue);
    },
    [activeClassRoles, activeDetectionProfile, classRoleProfiles, updateControlGroupField]
  );

  const setClassPriorityPosition = useCallback(
    async (classId: number, targetIndex: number) => {
      const current = [...orderedClassEditorIds];
      const index = current.indexOf(classId);
      if (index < 0 || index === targetIndex) {
        return;
      }
      current.splice(index, 1);
      current.splice(clampNumber(targetIndex, 0, current.length), 0, classId);
      await updateConfigField("inference", "detection_class_priority", current.join(","));
    },
    [orderedClassEditorIds, updateConfigField]
  );

  const toggleDetectionClass = useCallback(
    async (classId: number) => {
      const selected = parseDetectionClassFilter(activeDetectionClass) ?? new Set(classEditorIds);
      if (selected.has(classId)) {
        selected.delete(classId);
      } else {
        selected.add(classId);
      }
      const orderedSelection = orderedClassEditorIds.filter((id) => selected.has(id));
      await updateConfigField(
        "inference",
        "detection_class_filter",
        orderedSelection.length === classEditorIds.length
          ? "all"
          : orderedSelection.length === 0
            ? "none"
            : orderedSelection.join(",")
      );
    },
    [activeDetectionClass, classEditorIds, orderedClassEditorIds, updateConfigField]
  );

  const persistClassProfiles = useCallback(
    async (
      profiles: Record<string, string[]>,
      roleProfiles: Record<string, Record<string, AimRole>>,
      nextActiveProfile: string
    ) => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const next = base ? normalizeRuntimeConfig(base) : null;
      if (!next) {
        return;
      }
      next.inference = {
        ...asRecord(next.inference),
        detection_class_profiles: profiles,
        detection_class_profile: nextActiveProfile
      } as RuntimeConfig[string];
      const control = asRecord(next.control);
      next.control = {
        ...control,
        aim: {
          ...nestedRecord(control, "aim"),
          class_roles: roleProfiles
        }
      } as RuntimeConfig[string];
      setBusy("class-profiles.save");
      setLocalError(null);
      setDialogSaveError(null);
      configDraftRef.current = next;
      setConfigDraft(next);
      try {
        await updateRuntimeConfig(next);
        await onRefresh();
      } catch (err) {
        const message = getErrorMessage(err);
        configDraftRef.current = null;
        setConfigDraft(null);
        setLocalError(`类别配置同步失败：${message}`);
        setDialogSaveError(message);
        reportError(err, { source: "class-profiles", title: "类别配置保存失败" });
      } finally {
        setBusy(null);
      }
    },
    [onRefresh, runtimeConfig]
  );

  const createClassProfile = useCallback(async (copyCurrent: boolean) => {
    const profileName = newClassProfileName.trim();
    if (!profileName) {
      setLocalError("请输入新的类别配置名称。");
      return;
    }
    if (Object.prototype.hasOwnProperty.call(detectionProfiles, profileName)) {
      setLocalError(`类别配置“${profileName}”已存在。`);
      return;
    }
    await persistClassProfiles(
      { ...detectionProfiles, [profileName]: copyCurrent ? [...detectionClasses] : [] },
      copyCurrent && Object.keys(activeClassRoles).length > 0
        ? { ...classRoleProfiles, [profileName]: { ...activeClassRoles } }
        : { ...classRoleProfiles },
      profileName
    );
    setNewClassProfileName("");
  }, [
    activeClassRoles,
    classRoleProfiles,
    detectionClasses,
    detectionProfiles,
    newClassProfileName,
    persistClassProfiles
  ]);

  const renameClassProfile = useCallback(async () => {
    const profileName = renamedClassProfileName.trim();
    if (!profileName || profileName === activeDetectionProfile) {
      return;
    }
    if (Object.prototype.hasOwnProperty.call(detectionProfiles, profileName)) {
      setLocalError(`类别配置“${profileName}”已存在。`);
      return;
    }
    const nextProfiles = Object.fromEntries(
      Object.entries(detectionProfiles).map(([name, classes]) => [
        name === activeDetectionProfile ? profileName : name,
        classes
      ])
    );
    const nextRoleProfiles = { ...classRoleProfiles };
    if (Object.prototype.hasOwnProperty.call(nextRoleProfiles, activeDetectionProfile)) {
      nextRoleProfiles[profileName] = nextRoleProfiles[activeDetectionProfile];
      delete nextRoleProfiles[activeDetectionProfile];
    }
    await persistClassProfiles(nextProfiles, nextRoleProfiles, profileName);
  }, [
    activeDetectionProfile,
    classRoleProfiles,
    detectionProfiles,
    persistClassProfiles,
    renamedClassProfileName
  ]);

  const deleteClassProfile = useCallback(async () => {
    if (detectionProfileNames.length <= 1) {
      setLocalError("至少需要保留一个类别配置。");
      return;
    }
    const nextProfiles = { ...detectionProfiles };
    delete nextProfiles[activeDetectionProfile];
    const nextRoleProfiles = { ...classRoleProfiles };
    delete nextRoleProfiles[activeDetectionProfile];
    const nextActiveProfile = Object.keys(nextProfiles)[0];
    await persistClassProfiles(nextProfiles, nextRoleProfiles, nextActiveProfile);
  }, [
    activeDetectionProfile,
    classRoleProfiles,
    detectionProfileNames.length,
    detectionProfiles,
    persistClassProfiles
  ]);

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
      setLocalError(`kmNet ${kmnetConnected ? "断开" : kmnetConnecting ? "取消连接" : "连接"}失败：${getErrorMessage(err)}`);

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

  const toggleModelDirectory = useCallback((relativePath: string) => {
    setExpandedModelDirectories((current) => {
      const next = new Set(current);
      if (next.has(relativePath)) {
        next.delete(relativePath);
      } else {
        next.add(relativePath);
      }
      return next;
    });
  }, []);

  const selectModelFromCatalog = useCallback((model: ModelCatalogModel) => {
    setParserPreset("auto");
    setSelectedModelCatalogPath(model.relative_path);
    setLocalError(null);
  }, []);

  const switchModel = async () => {
    if (!selectedCatalogModel || selectedCatalogModel.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 产物。");
      return;
    }
    setBusy("model.switch");
    setLocalError(null);
    setModelSwitchMessage("");
    setModelSwitchDialogOpen(true);
    setModelSwitchDialogStatus("running");
    setModelSwitchStageIndex(0);
    setModelSwitchCompletedStages(0);
    setModelSwitchDialogError("");
    setModelSwitchProgressDetail("已按 .engine 后缀接受候选，准备登记模型引用。");
    try {
      setModelSwitchCompletedStages(1);
      setModelSwitchStageIndex(1);
      let projectId = selectedCatalogModel.project_id;
      let artifactId = selectedCatalogModel.artifact_id;
      if (typeof projectId !== "number" || typeof artifactId !== "number") {
        setModelSwitchProgressDetail("模型尚未登记，正在建立轻量文件引用；此步骤不会读取 Engine 内容。");
        const registered = await registerCatalogModel(selectedCatalogModel.relative_path);
        projectId = registered.project.id;
        artifactId = registered.artifact.id;
        setModelCatalogMessage(`已引用原始 Engine：${selectedCatalogModel.relative_path}；未复制模型文件。`);
      } else {
        setModelSwitchProgressDetail("已找到现有模型登记，跳过重复登记。");
      }
      setModelSwitchCompletedStages(2);
      setModelSwitchStageIndex(2);
      setModelSwitchProgressDetail("正在后端事务中验证 TensorRT 契约，并复用或生成运行 manifest。");
      const response = await publishModel(
        projectId,
        artifactId,
        parserPreset
      );
      if (response.report && !response.report.applied) {
        throw new Error(response.report.message);
      }
      const parserContract = response.parser_contract;
      const parserLabel = parserContract?.compatibility === "yolov5"
        ? "YOLO v5 兼容"
        : parserContract?.compatibility === "yolov8_yolo11"
          ? "YOLO v8 / v11 兼容"
          : "";
      const switchSummary = response.report?.message ??
        "Engine 契约读取完成，DeepStream 配置已自动生成并切换。";
      const manifestSummary = response.preparation?.manifest_action === "generated"
        ? "已自动生成运行 manifest"
        : response.preparation?.manifest_action === "reused"
          ? "已复用匹配的运行 manifest"
          : "运行 manifest 已准备";
      setModelSwitchMessage(
        parserLabel
          ? `${switchSummary} · 已验证 ${parserLabel} · NovaSight 内置 parser`
          : switchSummary
      );
      setModelSwitchCompletedStages(5);
      setModelSwitchStageIndex(4);
      setModelSwitchDialogStatus("success");
      setModelSwitchProgressDetail(`${manifestSummary}；${switchSummary}`);
      setModelCatalogRefreshKey((current) => current + 1);
      setModelDetailsRefreshKey((current) => current + 1);
      await onRefresh();
    } catch (err) {
      const message = getErrorMessage(err);
      setModelSwitchDialogStatus("failed");
      setModelSwitchDialogError(message);
      setLocalError(`模型切换未生效：${message}`);

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
      applyModelCatalogResult(result);
      setModelDetailsRefreshKey((current) => current + 1);
      await onRefresh();
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

  const launchStages = MAINLINE_LAUNCH_STAGES_CUSTOM_TENSORRT;
  const launchProgress =
    launchStatus === "success"
      ? 100
      : Math.round((launchCompletedStages / launchStages.length) * 100);
  const launchSummary =
    launchStatus === "success"
      ? "主链启动完成"
      : launchStatus === "failed"
        ? "启动在当前步骤中断"
        : launchStatus === "cancelled"
          ? "启动流程已取消"
          : launchStatus === "running"
            ? `正在执行第 ${Math.min(launchStageIndex + 1, launchStages.length)} 项`
            : "等待用户确认启动";
  const realtimeStatusText = runtimeDeliveryLabel(realtimeStatus);
  const realtimeStatusDescription = runtimeDeliveryDescription(realtimeStatus);
  const realtimeStatusClass = `console-live ${runtimeDeliveryTone(realtimeStatus)}`;
  const backendStatus = errors.health
    ? "error"
    : health?.ok
      ? "normal"
      : health === null
        ? "waiting"
        : "error";
  const backendStatusLabel = errors.health
    ? "后端不可达"
    : health?.ok
      ? "后端在线"
      : health === null
        ? "后端检查中"
        : "后端异常";
  const hasInferenceLatencySample = runtime?.running === true && readNumber(statistics?.inference_counter, 0) > 0;
  const latencyStages = [
    { label: "采集 / 解码 / ROI / 排队", value: hasInferenceLatencySample ? readNullableNumber(statistics?.stage_ingress_ms) : null, digits: 1 },
    { label: "nvinfer（含 parser）", value: hasInferenceLatencySample ? readNullableNumber(statistics?.stage_engine_ms) : null, digits: 1 },
    { label: "Batch 构建", value: hasInferenceLatencySample ? readNullableNumber(statistics?.stage_batch_build_ms) : null, digits: 2 },
    { label: "控制等待", value: hasInferenceLatencySample ? readNullableNumber(statistics?.stage_control_wait_ms) : null, digits: 2 },
    { label: "控制计算", value: hasInferenceLatencySample ? readNullableNumber(statistics?.stage_control_ms) : null, digits: 2 }
  ];
  const latencyStageTotal = latencyStages.reduce(
    (total, stage) => total + (stage.value !== null && stage.value > 0 ? stage.value : 0),
    0
  );
  const nvinferTimingScope = readString(statistics?.stage_engine_scope, "");
  const nvinferTimingScopeLabel = nvinferTimingScope === "sink_to_src_including_parser"
    ? "sink → src（包含 parser）"
    : nvinferTimingScope || "sink → src（包含 parser）";
  const latencyUnattributedMs = readNullableNumber(statistics?.stage_unattributed_ms);
  const latencyTimelineClosed = latencyUnattributedMs !== null && Math.abs(latencyUnattributedMs) <= 0.01;

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
            <StatusBadge
              status={backendStatus}
              icon={backendStatus === "normal" ? "check-circle" : backendStatus === "waiting" ? "clock" : "plug-off"}
              label={backendStatusLabel}
              size="sm"
            />
            <button
              className={currentErrorDetails.length > 0 ? "error-center-trigger has-errors" : "error-center-trigger"}
              onClick={() => setErrorCenterOpen(true)}
              type="button"
            >
              <NovaIcon name={currentErrorDetails.length > 0 ? "triangle-alert" : "shield-check"} size={15} />
              <span>异常信息</span>
              {currentErrorDetails.length > 0 ? <b>{currentErrorDetails.length}</b> : null}
            </button>
            <ThemeToggle />
            <div
              aria-label={`${realtimeStatusText}。${realtimeStatusDescription}运行态 ${formatDate(lastUpdated)}`}
              className={realtimeStatusClass}
              role="status"
              title={realtimeStatusDescription}
            >
              {realtimeStatusText} · 运行态 {formatDate(lastUpdated)}
            </div>
          </div>
        </div>
      </header>

      <aside className="console-sidebar">
        <StudioNavigation activePage={activePage} onNavigate={navigatePage} />
      </aside>

      <main className="console-main">
        <StudioPageHeader
          page={activePage}
          runtimeRunning={runtime?.running === true}
          realtimeConnected={realtimeStatus === "connected"}
        />
        <section className="console-process">
          <div className="console-process-state">
            <NovaIcon name="activity-pulse" size={16} />
            {runtimeMainlineSelected
              ? `主链：${captureStatusText} · 推理：${inferenceStatusText}`
              : `采集：${captureStatusText} · 推理：${inferenceStatusText}`}
          </div>
          <button
            className={runtimeControlRequested ? "console-button danger" : "console-button primary"}
            disabled={busy === "capture" || busy === "stop" || busy === "runtime.start"}
            onClick={() => void toggleCapture()}
            type="button"
          >
            <NovaIcon name={runtimeControlRequested ? "stop" : "start"} size={16} />
            {runtimeControlRequested ? (runtimeMainlineSelected ? "停止主链" : "停止采集") : runtimeMainlineSelected ? "启动主链" : "启动采集"}
          </button>
          <button
            className="console-button"
            onClick={openMotionProfileStudio}
            type="button"
          >
            <NovaIcon name="track-trace" size={16} />
            真人轨迹
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
          {activePage === "params" ? (
            <>
              <button className="console-button" onClick={exportConfig} type="button">
                <NovaIcon name="export" size={16} />
                导出全部参数
              </button>
              <button className="console-button" onClick={() => fileInputRef.current?.click()} type="button">
                <NovaIcon name="import" size={16} />
                导入全部参数
              </button>
              <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={importConfig} />
            </>
          ) : null}
        </section>

        {hostPresenceStandby ? (
          <div className="console-info">
            省流待机：{runtimePowerReason || "等待目标主机心跳"}。
            {runtimePowerAutoResume ? "主机恢复后将按运行意图自动启动。" : "主机上线后需要再次点击启动。"}
          </div>
        ) : mainlineLaunchPending ? (
          <div className="console-info">
            {mainlineLaunchMessage || "主链启动请求已提交，正在等待后端状态确认。"}
          </div>
        ) : null}

        <section className={activePage === "capture" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集状态" value={captureMainRunning ? "运行中" : "未运行"} small={captureBackendLabel || NO_SAMPLE} />
            <Metric title="采集 FPS" value={formatOptionalNumber(captureSourceFps, 1)} small={deepstreamNvinferSelected ? "v4l2 source" : "appsink arrival"} />
            <Metric title="数据新鲜度" value={formatOptionalNumber(latestCaptureAgeMs, 1)} small="距当前 ms" />
            <Metric title="采集帧间隔" value={formatOptionalNumber(captureFramePeriodMs, 2)} small="ms" />
          </div>
          <div className="console-card power-saving-card">
            <SectionTitle title="目标主机离线省流" />
            <p className="console-section-note">
              可选功能：游戏电脑离线后暂停 Jetson 的采集与推理，电脑恢复后可自动继续。配置立即生效，无需重启后端。
            </p>
            <ModuleSwitch
              label="游戏电脑离线时自动待机"
              detail={targetHostId ? `监管标识：${targetHostId}` : "请先填写游戏电脑标识，再启用此功能"}
              enabled={hostPresencePowerSavingEnabled}
              disabled={!hostPresencePowerSavingEnabled && !targetHostId.trim()}
              onToggle={(enabled) => updateConfigField("power_saving", "host_presence_enabled", enabled)}
            />
            <div className="power-saving-host-field">
              <TextControl
                label="游戏电脑标识"
                value={targetHostId}
                onCommit={(value) => updateConfigField("power_saving", "target_host_id", value.trim())}
              />
              <p className="console-field-hint">需与游戏电脑上 host_presence_agent.py 的 --host-id 完全一致，例如 gaming-pc。</p>
            </div>
            <details className="compact-settings-details">
              <summary>高级时序设置</summary>
              <NumberControl
                label="掉线判定时间 s"
                detail="超过该时间未收到心跳，开始进入离线宽限。"
                value={hostHeartbeatTimeoutS}
                min={1}
                max={120}
                step={1}
                onCommit={(value) => updateConfigField("power_saving", "heartbeat_timeout_s", value)}
              />
              <NumberControl
                label="停止前宽限 s"
                detail="宽限结束后暂停采集、解码、推理与预览。"
                value={hostOfflineGraceS}
                min={0}
                max={600}
                step={1}
                onCommit={(value) => updateConfigField("power_saving", "offline_grace_s", value)}
              />
              <ModuleSwitch
                label="电脑恢复后自动继续"
                detail="只恢复省流策略暂停的任务；用户主动停止后不会自动启动。"
                enabled={hostAutoResume}
                onToggle={(enabled) => updateConfigField("power_saving", "auto_resume", enabled)}
              />
            </details>
          </div>

          <div className="console-grid2 capture-config-grid compact-content-grid">
              <div className="console-card">
                <SectionTitle title="采集设备" />
                <label>视频设备</label>
                <input
                  value={device}
                  onChange={(event) => {
                    setDevice(event.target.value);
                    setCaps(null);
                    setSelectedChoiceId("");
                  }}
                />
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
                <div className="capture-format-control">
                  <select value={selectedChoice ? choiceId(selectedChoice) : ""} onChange={(event) => setSelectedChoiceId(event.target.value)}>
                    {choices.map((choice) => (
                      <option key={choiceId(choice)} value={choiceId(choice)}>{choiceLabel(choice)}</option>
                    ))}
                    {choices.length === 0 ? <option>请先检测设备能力</option> : null}
                  </select>
                  <button className="console-button secondary" disabled={busy === "caps"} onClick={refreshCapabilities} type="button">
                    <NovaIcon name="refresh" size={15} />
                    {busy === "caps" ? "检测中..." : "检测设备能力"}
                  </button>
                </div>
                <p className="console-section-note">
                  {caps
                    ? `已读取 ${choices.length} 组设备格式；更换采集卡或设备路径后请重新检测。`
                    : "当前使用已保存的采集格式，不会在打开页面时自动探测设备。"}
                </p>
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
              <SectionTitle title="最新画面状态" />
              <div className="console-kv">
                <span>最新 frame_id</span><b>{formatOptionalInteger(latestCaptureFrameId)}</b>
                <span>最新 generation</span><b>{formatOptionalInteger(latestCaptureGeneration)}</b>
                <span>最新帧时间戳</span><b>{latestCaptureTsNs !== null && latestCaptureTsNs > 0 ? `${Math.trunc(latestCaptureTsNs)} ns` : NO_SAMPLE}</b>
                <span>时间戳来源</span><b>{latestCaptureTimestampSource || NO_SAMPLE}</b>
                <span>数据距当前时间</span><b>{formatOptionalNumber(latestCaptureAgeMs, 2, "ms")}</b>
                <span>帧到达间隔</span><b>{formatOptionalNumber(captureFramePeriodMs, 2, "ms")}</b>
                <span>{deepstreamNvinferSelected ? "stale 拒绝数" : "采集丢帧数"}</span><b>{formatOptionalInteger(captureDroppedFrames)}</b>
                <span>已发布 / 已取得</span><b>{`${formatOptionalInteger(capturePublishedFrames)} / ${formatOptionalInteger(deepstreamNvinferSelected ? deepstreamMailbox.acquired_batches : latestFrameBroker.acquired_frames)}`}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="采集性能" />
              <div className="console-kv">
                <span>采集源 FPS</span><b>{formatOptionalNumber(captureSourceFps, 1, "FPS")}</b>
                <span>{deepstreamNvinferSelected ? "nvinfer 输入 FPS" : "appsink 到达 FPS"}</span><b>{formatOptionalNumber(deepstreamNvinferSelected ? nvinferInputFps : captureSourceFps, 1, "FPS")}</b>
                <span>采集等待调用</span><b>{formatOptionalNumber(capture?.capture_wait_ms, 2, "ms")}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "infer" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="推理 FPS" value={formatNumber(statistics?.inference_fps, 1)} small="FPS" />
            <Metric title="推理状态" value={inferenceRan ? (inferenceAvailable ? "已执行" : "执行失败") : "未执行"} small={selectedRuntimeBackend || NO_SAMPLE} />
            <Metric title="推理引擎耗时" value={formatOptionalNumber(inferenceTotalMs, 2)} small="ms" />
            <Metric title="NMS 后检测" value={formatOptionalInteger(inferenceNmsDetectionCount)} small="detections" />
          </div>
          <div className="console-card model-selection-card">
            <SectionTitle title="模型设置" />
            <ModelSelectionPanel
              root={modelCatalog}
              loading={modelCatalogLoading}
              directoryCount={modelCatalogDirectoryCount}
              modelCount={modelCatalogModelCount}
              expandedDirectories={expandedModelDirectories}
              selectedPath={selectedModelCatalogPath}
              selectedModel={selectedCatalogModel}
              selectedArtifact={selectedPreviewArtifact}
              selectedVersion={selectedPreviewVersion}
              activeArtifactId={artifact?.id ?? null}
              activeModelName={activeModelName}
              activeArtifactLabel={activeArtifactLabel}
              runtimeBackend={readString(runtime?.inference?.selected, "")}
              runtimeInputShape={displayedInputShape}
              catalogMessage={modelCatalogMessage}
              switchMessage={modelSwitchMessage}
              busy={busy}
              canSwitch={selectedCatalogModel?.kind === "engine"}
              parserPreset={parserPreset}
              onParserPresetChange={setParserPreset}
              onRefresh={() => void refreshModelCatalog()}
              onToggleDirectory={toggleModelDirectory}
              onSelectModel={selectModelFromCatalog}
              onSwitch={() => void switchModel()}
            />
          </div>
          <div className="console-grid2 inference-config-grid">
            <div className="console-card">
              <SectionTitle title="推理参数" />
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
                      {item.kind} · {item.path} · {formatModelSize(item.size_bytes)}{item.status === "pending" ? " · 待验证" : ""}
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
                supported={activePage === "infer" && previewEnabled && deepstreamPreviewStreamReady}
                active={previewActive}
                togglePending={previewTogglePending}
                onToggle={(enabled) => void updatePreviewActive(enabled)}
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
              <p className="console-section-note">输入 Tensor 类型来自当前 Engine 契约；运行精度来自已应用的模型 manifest。两者可以不同，float32 输入不代表 Engine 内部使用 FP32 计算。</p>
              <div className="console-kv">
                <span>模型名称</span><b>{activeModelName || NO_SAMPLE}</b>
                <span>推理后端</span><b>{selectedRuntimeBackend || NO_SAMPLE}</b>
                <span>ROI 输入尺寸</span><b>{roiInputWidth > 0 && roiInputHeight > 0 ? `${roiInputWidth}x${roiInputHeight}` : NO_SAMPLE}</b>
                <span>模型输入尺寸</span><b>{modelInputWidth > 0 && modelInputHeight > 0 ? `${modelInputWidth}x${modelInputHeight}` : displayedInputShape || NO_SAMPLE}</b>
                <span>输入 Tensor 类型</span><b>{inferenceInputDtype || NO_SAMPLE}</b>
                <span>Engine 精度声明</span><b>{inferenceRuntimePrecision ? `${inferenceRuntimePrecision}（manifest）` : NO_SAMPLE}</b>
                <span>内部逐层精度</span><b>未实时检测</b>
                <span>输入布局</span><b>{inferenceInputLayout || UNAVAILABLE}</b>
                <span>输入准备耗时</span><b>{formatOptionalNumber(inferencePreprocessMs, 3, "ms")}</b>
                <span>CUDA 上传耗时</span><b>{formatOptionalNumber(inferenceUploadMs, 3, "ms")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="推理引擎阶段" />
              <p className="console-section-note">从数据进入 nvinfer 到输出离开：包含 DeepStream 输入预处理、TensorRT 执行和自定义 parser 解析；不包含目标跟踪与鼠标控制。</p>
              <div className="console-kv">
                <span>TensorRT enqueue 耗时</span><b>{formatOptionalNumber(inferenceEnqueueMs, 3, "ms")}</b>
                <span>CUDA stream 同步等待</span><b>{formatOptionalNumber(inferenceSyncWaitMs, 3, "ms")}</b>
                <span>sink → src 总耗时</span><b>{formatOptionalNumber(inferenceTotalMs, 3, "ms")}</b>
                <span>计时范围</span><b>预处理 + TensorRT + parser</b>
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
                <span>生效类别过滤</span><b>{effectiveClassFilter === "all" ? "全部类别" : effectiveClassFilter === "none" ? "未选择任何类别" : `cls ${effectiveClassFilter}`}</b>
                <span>被类别过滤的 cls</span><b>{rejectedBasicClassIds.length > 0 ? rejectedBasicClassIds.join(", ") : NO_SAMPLE}</b>
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
              {targetPipelineCode === "BASIC_CANDIDATE_REJECTED" && targetPipelineRejections.includes("class_filter") && detectedClassFilterValue ? (
                <div className="control-filter-recovery">
                  <button
                    className="console-button primary"
                    disabled={busy !== null}
                    onClick={() => void updateConfigField("inference", "detection_class_filter", detectedClassFilterValue)}
                    type="button"
                  >
                    允许当前检测类别
                  </button>
                  <small>保留已有选择，并加入本帧检测到的 cls；保存后立即热更新。</small>
                </div>
              ) : null}
            </div>
            <div className="console-card">
              <SectionTitle title="瞄准点与预测" />
              <div className="console-kv">
                <span>原始瞄准点</span><b>{formatPoint(observedAimX, observedAimY, 1, "px")}</b>
                <span>类别配置 / 瞄点类型</span><b>{`${readString(controlPipeline.active_class_profile, activeDetectionProfile)} / ${readString(controlPipeline.effective_aim_role, "other")}`}</b>
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
                    <span>到位状态</span><b>{readString(controlPipeline.arrival_state, "") || NO_SAMPLE}</b>
                    <span>到位限制后 counts</span><b>{formatPoint(controlPipeline.deadzone_limited_counts_x_float, controlPipeline.deadzone_limited_counts_y_float, 2)}</b>
                    <span>Slew 后 counts</span><b>{formatPoint(controlPipeline.slew_limited_counts_x_float, controlPipeline.slew_limited_counts_y_float, 2)}</b>
                    <span>累计余量 counts</span><b>{formatPoint(controlPipeline.residual_x_counts, controlPipeline.residual_y_counts, 2)}</b>
                  </>
                )}
                <span>固定压枪状态</span><b>{controlPipeline.recoil_active === true ? "输出中" : recoilEnabled ? formatRecoilBlockReason(controlPipeline.recoil_block_reason) : "关闭"}</b>
                <span>固定压枪 / 观测</span><b>{`${recoilYCountsPerObservation.toFixed(1)} counts · ${(dualPhaseActive ? dualPhaseInvertY : sharedInvertY) ? "-Y" : "+Y"}`}</b>
                <span>预估固定压枪强度</span><b>{`${(recoilYCountsPerObservation * controlObservationFps).toFixed(1)} counts/s @ ${controlObservationFps.toFixed(1)} FPS`}</b>
                <span>视觉 / 固定 / 合成 Y</span><b>{`${formatOptionalNumber(controlPipeline.feedback_demand_y, 2)} / ${formatOptionalNumber(controlPipeline.recoil_y_counts_float, 2)} / ${formatOptionalNumber(controlPipeline.combined_demand_y, 2)} counts`}</b>
                <span>触发持续 / 启动延迟</span><b>{`${formatOptionalNumber(control.trigger_hold_ms, 1, "ms")} / ${formatOptionalNumber(control.trigger_activation_delay_ms, 1, "ms")}`}</b>
                <span>控制预算</span><b>{formatPoint(control.dx, control.dy, 0, "counts")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title={dualPhaseActive ? "Latest Replace 与设备发送" : "Scheduler 与设备发送"} />
              <div className="console-kv">
                <span>触发状态</span><b>{control.trigger_active === true ? "按下" : control.trigger_active === false ? "未按下" : NO_SAMPLE}</b>
                <span>是否允许发包</span><b>{control.will_emit === true ? "是" : control.will_emit === false ? "否" : NO_SAMPLE}</b>
                <span>不发包原因</span><b>{controlNoSendReason || NO_SAMPLE}</b>
                <span>本轮控制意图</span><b>{formatPoint(control.dx, control.dy, 0, "counts")}</b>
                <span>{dualPhaseActive ? "发送语义" : "待执行 counts"}</span><b>{dualPhaseActive ? "仅保留最新观测" : formatPoint(schedulerStatus.pending_dx, schedulerStatus.pending_dy, 0, "counts")}</b>
                <span>本次发送 counts</span><b>{execution.sent === true ? formatPoint(controlActualDx, controlActualDy, 0, "counts") : NO_SAMPLE}</b>
                <span>{dualPhaseActive ? "待发送最新命令" : "剩余 pending steps"}</span><b>{formatOptionalInteger(schedulerStatus.pending_steps)}</b>
                <span>设备发送耗时</span><b>{controlSendDuration}</b>
                <span>最后发送时间</span><b>{formatOptionalInteger(execution.device_send_end_ts_ns ?? executionMeta.device_send_end_ts_ns)}</b>
                {!dualPhaseActive ? <><span>旧计划截断次数</span><b>{formatOptionalInteger(schedulerStatus.cancelled_pending)}</b><span>最近取消原因</span><b>{readString(schedulerStatus.last_cancel_reason, "") || NO_SAMPLE}</b></> : null}
                <span>设备连接</span><b>{kmnetConnectionLabel}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "params" || activePage === "control-test" ? "console-page active" : "console-page"}>
          {activePage === "params" ? (
          <>
            <div className="console-metrics params-summary-metrics">
              <Metric title="控制模式" value={controlModeLabel} small="单选策略" />
              <Metric
                title="轨迹来源"
                value={motionProfileRuntime === null ? "读取中" : motionProfileRuntime.enabled ? "真人轨迹" : "静态参数"}
                small={motionProfileRuntime?.enabled ? motionProfileRuntime.profile_name || "运行内存画像" : motionProfileRuntime === null ? "等待运行态" : "配置文件"}
              />
              <Metric title="触发方式" value={triggerModeLabel(triggerMode)} small="trigger" />
              <Metric title="类型瞄点 Y" value={`${Math.round(aimRoleRatios.head * 100)} / ${Math.round(aimRoleRatios.body * 100)} / ${Math.round(aimRoleRatios.other * 100)}`} small="头部 / 身体 / 其他 %" />
              <Metric title="位置预测" value={dualPhaseActive ? `${dualPhaseLeadFrames.toFixed(2)} 帧` : "不使用"} small={dualPhaseActive ? "平均 dt 前瞻" : "反馈控制"} />
            <Metric title="发送方式" value={dualPhaseActive ? "最新覆盖" : schedulerEnabled ? `${schedulerIntervalMs.toFixed(1)} ms` : "观测直发"} small={dualPhaseActive ? `${schedulerIntervalMs.toFixed(1)} ms 单槽` : schedulerEnabled ? `${schedulerStepCountsX}/${schedulerStepCountsY} counts` : "scheduler off"} />
            </div>
            <div className={motionProfileRuntime?.enabled ? "console-card motion-control-mode-card human" : motionProfileRuntime === null ? "console-card motion-control-mode-card loading" : "console-card motion-control-mode-card static"}>
              <div className="motion-control-mode-copy">
                <span className="class-config-eyebrow">硬件触发后的控制轨迹</span>
                <h3>{motionProfileRuntime === null ? "正在读取控制轨迹" : motionProfileRuntime.enabled ? "真人轨迹算法" : "静态控制算法"}</h3>
                <p>
                  {motionProfileRuntime === null
                    ? "正在从后端确认当前运行内存使用的轨迹来源。"
                    : motionProfileRuntime.enabled
                    ? "当前画像直接覆盖运行内存中的静态节奏参数；关闭后立即恢复配置文件中的控制参数。"
                    : "使用参数页中已经调整好的固定控制参数，不加载真人画像。"}
                </p>
              </div>
              <div className="motion-control-mode-actions">
                <div className="motion-mode-segmented" role="group" aria-label="控制轨迹来源">
                  <button
                    aria-pressed={motionProfileRuntime?.enabled !== true}
                    className={motionProfileRuntime?.enabled ? "" : "active"}
                    disabled={motionProfileBusy || motionProfileRuntime === null}
                    onClick={() => void setMotionControlMode(false)}
                    type="button"
                  >
                    <NovaIcon name="settings" size={16} />
                    静态控制算法
                  </button>
                  <button
                    aria-pressed={motionProfileRuntime?.enabled === true}
                    className={motionProfileRuntime?.enabled ? "active" : ""}
                    disabled={motionProfileBusy || motionProfiles.length === 0}
                    onClick={() => void setMotionControlMode(true)}
                    type="button"
                  >
                    <NovaIcon name="track-trace" size={16} />
                    真人轨迹算法
                  </button>
                </div>
                <div className="motion-profile-picker">
                  <label htmlFor="motion-profile-select">真人画像</label>
                  <select
                    id="motion-profile-select"
                    disabled={motionProfileBusy || motionProfiles.length === 0}
                    onChange={(event) => void selectMotionProfile(event.target.value)}
                    value={selectedMotionProfileId}
                  >
                    {motionProfiles.length === 0 ? <option value="">尚未训练画像</option> : null}
                    {motionProfiles.map((profile) => (
                      <option key={profile.profile_id} value={profile.profile_id}>
                        {profile.name} · {profile.sample_count} 条
                      </option>
                    ))}
                  </select>
                  <button className="console-button" onClick={openMotionProfileStudio} type="button">
                    {motionProfiles.length === 0 ? "去训练画像" : "管理与训练"}
                  </button>
                </div>
                <small className="motion-control-memory-note">
                  {motionProfileBusy
                    ? "正在切换运行内存…"
                    : motionProfileRuntime?.enabled
                      ? `运行中：${motionProfileRuntime.profile_name || motionProfileRuntime.active_profile} · ${motionProfileRuntime.sample_count} 条样本`
                      : "当前未启用真人曲线；文件配置不会被修改。"}
                </small>
              </div>
            </div>
            <div className="console-card class-config-summary-card">
              <div className="class-config-summary-main">
                <div className="class-config-summary-icon" aria-hidden="true">
                  <NovaIcon name="target" size={20} strokeWidth={1.8} />
                </div>
                <div>
                  <span className="class-config-eyebrow">类别配置</span>
                  <h3>{activeDetectionProfile}</h3>
                  <p>类别名称、选择顺序、瞄点类型与三条共享瞄点线在独立靶场统一管理。</p>
                </div>
              </div>
              <dl className="class-config-summary-stats">
                <div><dt>已定义类别</dt><dd>{detectionClasses.filter(Boolean).length}</dd></div>
                <div><dt>瞄点类型</dt><dd>头部 / 身体 / 其他</dd></div>
                <div><dt>已映射</dt><dd>{Object.keys(activeClassRoles).length}</dd></div>
                <div><dt>目标筛选</dt><dd>{selectedDetectionClassIds.size}/{classEditorIds.length} 类</dd></div>
              </dl>
              <button
                className="console-button primary"
                onClick={() => setClassConfigDialogOpen(true)}
                type="button"
              >
                <NovaIcon name="settings" size={16} />
                管理类别配置
              </button>
            </div>
            <div className="console-grid2 params-control-grid compact-content-grid" data-algorithm-page={controlMode}>
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
                    <b>瞄点规则由三种类型统一提供</b>
                    <small>头 {Math.round(aimRoleRatios.head * 100)}%、身体 {Math.round(aimRoleRatios.body * 100)}%、其他 {Math.round(aimRoleRatios.other * 100)}%；未映射类别自动使用“其他”。</small>
                  </span>
                  <div className="role-aim-mini-preview" aria-hidden="true">
                    <i className="head" style={{ top: `${aimRoleRatios.head * 100}%` }} />
                    <i className="body" style={{ top: `${aimRoleRatios.body * 100}%` }} />
                    <i className="other" style={{ top: `${aimRoleRatios.other * 100}%` }} />
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
                    <span>输出交付</span><b>Latest Replace</b>
                    <span>每个推理结果</span><b>覆盖尚未发送的旧命令</b>
                    <span>待发送容量</span><b>1 条完整命令</b>
                    <span>设备发送校验</span><b>MouseCommandExecutor</b>
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
                <p className="console-section-note">日常使用只需选择算法；投影、增益、限幅和预测属于工程调校参数。</p>
                <div className="advanced-settings-summary">
                  {dualPhaseActive ? (
                    <>
                      <div><span>FOVX</span><b>{dualPhaseFovX.toFixed(1)}°</b></div>
                      <div><span>FAR / NEAR Kp</span><b>{dualPhaseFarKp.toFixed(3)} / {dualPhaseNearKp.toFixed(3)}</b></div>
                      <div><span>预测前瞻</span><b>{dualPhaseLeadFrames.toFixed(2)} 帧</b></div>
                    </>
                  ) : controlMode === "calibrated_angular" ? (
                    <>
                      <div><span>FOVX</span><b>{calibratedFovX.toFixed(1)}°</b></div>
                      <div><span>Kp X / Y</span><b>{calibratedKpX.toFixed(2)} / {calibratedKpY.toFixed(2)}</b></div>
                      <div><span>Kd X / Y</span><b>{calibratedKdX.toFixed(2)} / {calibratedKdY.toFixed(2)}</b></div>
                    </>
                  ) : (
                    <>
                      <div><span>响应尺度 X / Y</span><b>{universalResponseScaleX.toFixed(1)} / {universalResponseScaleY.toFixed(1)} px</b></div>
                      <div><span>最大移动 X / Y</span><b>{universalMaxStepX.toFixed(1)} / {universalMaxStepY.toFixed(1)}</b></div>
                    </>
                  )}
                </div>
                <button className="console-button console-full-button" onClick={() => setAlgorithmSettingsDialogOpen(true)} type="button">
                  <NovaIcon name="settings" size={15} />
                  调整算法高级参数
                </button>
              </div>

              <div className="console-card crosshair-reference-card">
                <SectionTitle title="视觉准星基准" />
                <p className="console-section-note">
                  从主链中心独立采样真实 HUD 准星。未学习、未确认或观测过期时，控制会自动使用 ROI 几何中心。
                </p>
                <ModuleSwitch
                  label="启用低频准星观测"
                  detail="新增独立 NVMM 中心小图支路；修改后需要重启主链。"
                  enabled={crosshairEnabled}
                  onToggle={(enabled) => updateConfigField("crosshair", "enabled", enabled)}
                />
                <ModuleSwitch
                  label="用于目标选择与鼠标控制"
                  detail={crosshairTemplateId
                    ? "只有连续确认且未过期的观测会接管控制基准。"
                    : "请先启动主链并学习准星模板；现在始终使用几何中心。"}
                  enabled={crosshairUseForControl}
                  disabled={!crosshairEnabled || !crosshairTemplateId}
                  onToggle={(enabled) => updateConfigField("crosshair", "use_for_control", enabled)}
                />
                <div className="crosshair-reference-panel">
                  <div className="crosshair-template-preview" data-empty={!crosshairTemplateId}>
                    {crosshairTemplateId ? (
                      <img
                        alt="已学习的准星结构模板"
                        src={crosshairTemplatePreviewUrl(crosshairPreviewKey)}
                      />
                    ) : (
                      <span><NovaIcon name="target" size={30} /><b>等待学习</b></span>
                    )}
                  </div>
                  <div className="console-kv compact-kv crosshair-reference-kv">
                    <span>观测状态</span><b data-state={crosshairState}>{crosshairStateLabel(crosshairState)}</b>
                    <span>采样支路</span><b>{crosshairBranchActive ? "运行中" : crosshairEnabled ? "不可用" : "关闭"}</b>
                    <span>控制可用</span><b>{crosshairReferenceReady ? "是" : "否，使用几何中心"}</b>
                    <span>模板</span><b>{crosshairTemplateId || "尚未生成"}</b>
                    <span>中心偏移</span><b>{crosshairReferenceReady ? `${crosshairOffsetX.toFixed(1)}, ${crosshairOffsetY.toFixed(1)} px` : "—"}</b>
                    <span>匹配置信度</span><b>{crosshairConfidence > 0 ? `${(crosshairConfidence * 100).toFixed(1)}%` : "—"}</b>
                    <span>学习帧</span><b>{crosshairRecentSamples}/{crosshairRequiredSamples}</b>
                  </div>
                </div>
                <div className="crosshair-learn-actions">
                  <button
                    className="console-button primary"
                    disabled={!crosshairEnabled || !runtimeMainlineRunning || crosshairRecentSamples < crosshairRequiredSamples || busy === "crosshair.learn"}
                    onClick={() => void handleLearnCrosshair()}
                    type="button"
                  >
                    <NovaIcon name="target" size={15} />
                    {busy === "crosshair.learn" ? "正在学习…" : "学习当前准星 · F8"}
                  </button>
                  <button
                    className="console-button"
                    disabled={!crosshairTemplateId || busy === "crosshair.clear"}
                    onClick={() => void handleClearCrosshair()}
                    type="button"
                  >
                    清除模板
                  </button>
                </div>
                {crosshairProcessingError ? (
                  <p className="crosshair-inline-message warning">准星处理失败：{crosshairProcessingError}</p>
                ) : runtimeMainlineRunning && crosshairEnabled && !crosshairBranchActive ? (
                  <p className="crosshair-inline-message warning">准星采样支路不可用：{crosshairBranchReason || "请查看 DeepStream 状态"}</p>
                ) : !runtimeMainlineRunning && crosshairEnabled ? (
                  <p className="crosshair-inline-message warning">需要先启动主链，独立中心采样支路才会提供学习帧。</p>
                ) : crosshairMessage ? (
                  <p className="crosshair-inline-message">{crosshairMessage}</p>
                ) : null}
                <details className="crosshair-advanced-settings">
                  <summary>采样高级设置</summary>
                  <div className="advanced-settings-grid">
                    <NumberControl label="中心搜索区 px" detail="只截取 ROI 正中心的小区域，不扫描整幅画面。" value={crosshairSearchSize} min={32} max={Math.max(32, roiSize)} step={2} onCommit={(value) => updateConfigField("crosshair", "search_size", Math.round(value / 2) * 2)} />
                    <NumberControl label="观测频率 Hz" detail="已与推理支路隔离；10 Hz 通常足够验证固定 HUD 准星。" value={crosshairSampleHz} min={1} max={30} step={1} onCommit={(value) => updateConfigField("crosshair", "sample_hz", Math.round(value))} />
                    <NumberControl label="学习采样帧数" detail="使用多帧中位图减少动态背景对模板的污染。" value={crosshairSampleFrames} min={3} max={15} step={1} onCommit={(value) => updateConfigField("crosshair", "sample_frames", Math.round(value))} />
                  </div>
                </details>
              </div>

              <div className="console-card">
                  <SectionTitle title="固定 Y 轴压枪 · 所有控制算法" />
                  <ModuleSwitch label="启用固定 Y 压枪" detail="真实左键达到启动延迟后，每个新鲜目标观测固定追加一次反向 Y counts；不使用渐入、时间速率、积分或 Y 预测。" enabled={recoilEnabled} onToggle={(enabled) => updateControlGroupField("shared", "recoil_enabled", enabled)} />
                  {recoilEnabled ? (
                    <>
                      <NumberControl label="开始压枪前等待 ms" detail="从真实左键按下开始计时；未达到该时间时固定压枪保持为零。" value={recoilStartDelayMs} min={0} max={1000} step={1} onCommit={(value) => updateControlGroupField("shared", "recoil_start_delay_ms", value)} />
                      <NumberControl label="每个新观测固定 Y counts" detail={`每个新鲜目标观测追加相同数值；当前约 ${(recoilYCountsPerObservation * controlObservationFps).toFixed(1)} counts/s（${controlObservationFps.toFixed(1)} 控制观测 FPS），小数由独立余量累计。`} value={recoilYCountsPerObservation} min={0} max={20} step={0.1} onCommit={(value) => updateControlGroupField("shared", "recoil_y_counts_per_observation", value)} />
                    </>
                  ) : null}
                </div>

              <div className="console-card">
                <SectionTitle title="目标选择与切换 · 通用参数" />
                <NumberControl label="目标选择半径（640 基准 px）" detail="以 640×640 ROI 为基准；运行时按当前 ROI 尺寸同比缩放，保证 320～640 ROI 使用一致的相对选择范围。" value={targetFovRadiusPx} min={1} max={640} step={1} onCommit={(value) => updateConfigField("control", "target_fov_radius_px", value)} />
                <div className="target-weight-summary">
                  <div>
                    <span>当前综合分权重</span>
                    <strong>
                      类别 {(normalizedSelectionClassWeight * 100).toFixed(0)}%
                      <i>·</i>
                      质量 {(normalizedSelectionQualityWeight * 100).toFixed(0)}%
                      <i>·</i>
                      距离 {(normalizedSelectionDistanceWeight * 100).toFixed(0)}%
                    </strong>
                    <small>类别偏好已提高，距离影响相应降低；原始值会在计算前自动归一化。</small>
                  </div>
                  <button className="console-button" onClick={() => setTargetWeightsDialogOpen(true)} type="button">
                    <NovaIcon name="settings" size={15} />
                    调整权重
                  </button>
                </div>
                <div className="advanced-settings-summary compact">
                  <div><span>异常框宽高比</span><b>≤ {candidateRatioMaxAspect.toFixed(1)}</b></div>
                  <div><span>切换门槛</span><b>{targetSwitchPreferenceAdvantage.toFixed(2)}</b></div>
                  <div><span>确认延迟</span><b>{targetSwitchDelayMs.toFixed(0)} ms</b></div>
                </div>
                <button className="console-button console-full-button" onClick={() => setTargetAdvancedDialogOpen(true)} type="button">
                  <NovaIcon name="settings" size={15} />
                  目标切换高级设置
                </button>
              </div>

              <div className="console-card">
                <SectionTitle title={dualPhaseActive ? "Tracker · 公共参数" : "Tracker / Kalman · 公共参数"} />
                <div className="console-kv compact-kv"><span>关联算法</span><b>Hungarian</b><span>输出状态</span><b>仅 ACTIVE</b></div>
                <div className="advanced-settings-summary compact">
                  <div><span>匹配距离</span><b>{trackerMaxMatchDistance.toFixed(2)}</b></div>
                  <div><span>位置 / IoU</span><b>{trackerPositionCostWeight.toFixed(2)} / {trackerIouCostWeight.toFixed(2)}</b></div>
                  <div><span>最大漏检</span><b>{trackerMaxMissedFrames} 帧</b></div>
                </div>
                <button className="console-button console-full-button" onClick={() => setTrackerSettingsDialogOpen(true)} type="button">
                  <NovaIcon name="settings" size={15} />
                  管理 Tracker / Kalman
                </button>
              </div>
            </div>
          </>
          ) : (
          <>
          <div className="console-metrics">
            <Metric title="连接状态" value={kmnetConnectionLabel} small={kmnetConnected ? "online" : kmnetConnecting ? "connecting" : kmnetRetryable ? "retry available" : "offline"} />
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
                <div className={kmnetConnected ? "kmnet-status-tile good" : kmnetConnectionFailed ? "kmnet-status-tile bad" : "kmnet-status-tile idle"}>
                  <span>连接</span>
                  <b>{kmnetConnectionLabel}</b>
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
              {kmnetLastError || kmnetConnectionFailed || kmnetConnectionDegraded ? (
                <div className={kmnetConnectionFailed ? "kmnet-connection-notice failed" : "kmnet-connection-notice warn"} role="status">
                  <div>
                    <strong>{kmnetConnectionFailed ? "输出设备未连接" : kmnetConnectionDegraded ? "输出已连接，但按键监听不可用" : "最近一次输出失败"}</strong>
                    <span>{kmnetLastError || (kmnetConnectionFailed ? "请检查地址、端口和驱动后重新连接。" : "不依赖硬件按键的控制仍可继续使用。")}</span>
                  </div>
                  {kmnetRetryable ? <small>修改配置后点击“重新连接”，无需重启主链。</small> : null}
                </div>
              ) : null}
              <div className="console-action-row kmnet-connection-actions">
                <button
                  className={kmnetConnected ? "console-button danger" : kmnetConnecting ? "console-button" : "console-button primary"}
                  aria-pressed={kmnetConnected}
                  disabled={busy === "kmnet.toggle" || (!kmnetDriverAvailable && !kmnetConnected && !kmnetConnecting)}
                  onClick={() => void toggleHardwareConnection()}
                  type="button"
                >
                  {busy === "kmnet.toggle" ? "处理中…" : kmnetConnectionActionLabel}
                </button>
                <span>{kmnetConnected ? "输出命令可发送" : kmnetConnecting ? "正在初始化驱动与网络连接" : kmnetConnectionFailed ? "主链可继续运行，输出暂不可用" : "连接后才会发送控制输出"}</span>
              </div>
              <TextControl label="kmnetip" value={kmnetHost} onCommit={(value) => updateConfigField("hardware", "host", value)} />
              <NumberControl label="kmnetport" value={kmnetPort} min={0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "port", Math.round(value))} />
              <TextControl label="kmnetuuid" value={kmnetUuid} onCommit={(value) => updateConfigField("hardware", "uuid", value)} />
              <NumberControl label="monitor_port" value={kmnetMonitorPort} min={0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "monitor_port", Math.round(value))} />
              <ModuleSwitch
                label="后端服务启动时自动连接"
                detail="独立于主链启动；连接失败不会阻止采集、推理和鼠标算法运行"
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
                  <span>执行层</span><b>Latest Replace Scheduler</b>
                  <span>行为</span><b>新观测覆盖未发送的旧命令</b>
                  <span>待发送容量</span><b>1 条完整命令</b>
                  <span>最终校验</span><b>MouseCommandExecutor</b>
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
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(0, -10, 1, 0, "raw", 0)}>↑</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(-10, 0, 1, 0, "raw", 0)}>←</button>
                  <button type="button" onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "raw", 0)} disabled={kmnetDiagnosticDisabled}>发送</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(10, 0, 1, 0, "raw", 0)}>→</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(0, 10, 1, 0, "raw", 0)}>↓</button>
                </div>
                <div className="kmnet-test-section">
                  <h3>最快直移</h3>
                  <p>调用 move / enc_move，只传 x、y。适合验证最底层驱动是否能立即移动。</p>
                  <div className="console-action-row">
                    <button
                      className="console-button"
                      disabled={kmnetDiagnosticDisabled}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "raw", 0)}
                      type="button"
                    >
                      move
                    </button>
                    <button
                      className="console-button"
                      disabled={kmnetDiagnosticDisabled}
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
                      disabled={kmnetDiagnosticDisabled}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "auto", kmnetTestMs)}
                      type="button"
                    >
                      move_auto
                    </button>
                    <button
                      className="console-button"
                      disabled={kmnetDiagnosticDisabled}
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
                      disabled={kmnetDiagnosticDisabled}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 1, 0, "bezier", kmnetTestMs, { x1: kmnetBezierX1, y1: kmnetBezierY1, x2: kmnetBezierX2, y2: kmnetBezierY2 })}
                      type="button"
                    >
                      move_beizer
                    </button>
                    <button
                      className="console-button"
                      disabled={kmnetDiagnosticDisabled}
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
                      disabled={kmnetDiagnosticDisabled}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy, 3, 4, "raw", 0)}
                      type="button"
                    >
                      连续发送
                    </button>
                    <button
                      className="console-button primary"
                      disabled={kmnetDiagnosticDisabled}
                      onClick={() => void diagnosticMoveHardware(600, 0, 1, 0, "raw", 0)}
                      type="button"
                    >
                      右移大步测试
                    </button>
                  </div>
                  <button
                    className="kmnet-circle-button"
                    disabled={kmnetDiagnosticDisabled}
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
            </div>
          </div>
          </>
          )}
        </section>

        <section className={activePage === "latency" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="进入 nvinfer" value={hasInferenceLatencySample ? formatNumber(statistics?.stage_ingress_ms, 1) : NO_SAMPLE} small="ms" />
            <Metric title="完整链路" value={hasInferenceLatencySample ? formatNumber(statistics?.stage_total_ms, 2) : NO_SAMPLE} small="采集→控制 ms" />
            <Metric title="Batch 发布龄" value={hasInferenceLatencySample ? formatNumber(statistics?.e2e_latency, 1) : NO_SAMPLE} small="ms" />
            <Metric title="控制计算" value={hasInferenceLatencySample ? formatNumber(statistics?.stage_control_ms, 2) : NO_SAMPLE} small="ms" />
          </div>
          <div className="console-grid2 latency-analysis-grid">
            <div className="console-card">
              <SectionTitle title="延迟链路" />
              <p className="console-section-note">只累加互不重叠的真实测量区间；没有运行样本时不绘制比例。</p>
              <div className="console-timeline">
                {latencyStages.map((stage) => (
                  <Event
                    key={stage.label}
                    label={stage.label}
                    value={stage.value === null ? NO_SAMPLE : stage.value.toFixed(stage.digits)}
                    width={stage.value !== null && latencyStageTotal > 0 ? (Math.max(0, stage.value) / latencyStageTotal) * 100 : null}
                  />
                ))}
              </div>
            </div>
            <KvCard
              title="测量边界"
              rows={[
                ["采集 / 解码 / ROI / 排队", "合并测量到 nvinfer sink"],
                ["ROI 独立耗时", "当前未单独打点"],
                ["nvinfer 范围", nvinferTimingScopeLabel],
                ["解码 / NMS", "parser 已包含在 nvinfer，不重复累加"],
                ["Batch 发布龄范围", "采集时间戳 → DetectionBatch 发布"],
                ["完整链路范围", "采集时间戳 → 控制计算完成"],
                ["阶段完整性", latencyTimelineClosed ? "已闭合 · 未归因 0.00ms" : `未归因 ${formatOptionalNumber(latencyUnattributedMs, 2, "ms")}`],
                ["时间戳来源", shortTimestampSource(readString(statistics?.timestamp_source, "暂无样本"))]
              ]}
              notice={<p className="latency-boundary-note">DeepStream 当前没有在解码器、ROI 和内部队列之间分别打点，因此不能诚实拆成三个独立数字。</p>}
            />
          </div>
        </section>
      </main>

      <ThemeGallery />

      <AdvancedSettingsDialog
        description="这些参数决定投影、响应曲线、限幅与预测行为。日常使用无需频繁调整。"
        eyebrow="参数设置 / 控制算法"
        footerNote={`当前算法：${controlModeLabel}`}
        onClose={() => setAlgorithmSettingsDialogOpen(false)}
        open={algorithmSettingsDialogOpen}
        saveError={dialogSaveError}
        saving={dialogSaving}
        title={`${controlModeLabel} · 高级参数`}
      >
        <div className="advanced-settings-grid">
          {dualPhaseActive ? (
            <>
              <NumberControl label="水平 FOVX" value={dualPhaseFovX} min={30} max={179} step={0.1} onCommit={(value) => updateDualPhasePath(["projection", "fov_x_deg"], value)} />
              <NumberControl label="每圈 counts" value={dualPhaseCountsPer360} min={1} max={100000} step={1} onCommit={(value) => updateDualPhasePath(["projection", "counts_per_360"], value)} />
              <NumberControl label="NEAR 阈值 px" detail="测量误差距离不大于该值时使用 NEAR，否则直接使用 FAR。" value={dualPhaseNearThreshold} min={0} max={1000} step={0.1} onCommit={(value) => updateDualPhasePath(["mode", "near_threshold_px"], value)} />
              <NumberControl label="FAR Kp" value={dualPhaseFarKp} min={0.001} max={0.999} step={0.001} onCommit={(value) => updateDualPhasePath(["atan", "far", "kp"], value)} />
              <NumberControl label="NEAR Kp" value={dualPhaseNearKp} min={0.001} max={0.999} step={0.001} onCommit={(value) => updateDualPhasePath(["atan", "near", "kp"], value)} />
              <NumberControl label="共享 Atan 尺度 counts" detail="FAR 与 NEAR 使用同一个非线性压缩尺度。" value={dualPhaseAtanScale} min={0.1} max={10000} step={0.1} onCommit={(value) => updateDualPhasePath(["atan", "scale_counts"], value)} />
              <NumberControl label="FAR 单次上限 counts" value={dualPhaseFarMaxCounts} min={1} max={2000} step={1} onCommit={(value) => updateDualPhasePath(["atan", "far", "max_counts_per_update"], value)} />
              <NumberControl label="NEAR 单次上限 counts" value={dualPhaseNearMaxCounts} min={1} max={2000} step={1} onCommit={(value) => updateDualPhasePath(["atan", "near", "max_counts_per_update"], value)} />
              <NumberControl label="前瞻帧数" detail="预测量 = 平滑目标速度 × 平均 capture dt × 前瞻帧数；0 完全关闭位置预测。" value={dualPhaseLeadFrames} min={0} max={10} step={0.01} onCommit={(value) => updateDualPhasePath(["prediction", "lead_frames"], value)} />
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
      </AdvancedSettingsDialog>

      <AdvancedSettingsDialog
        description="控制异常框过滤、候选切换门槛和防抖确认。设置过严会阻止切换，过松会造成目标跳变。"
        eyebrow="参数设置 / 目标选择"
        footerNote="这些设置不会改变框内 aim Y，只影响选择与切换。"
        onClose={() => setTargetAdvancedDialogOpen(false)}
        open={targetAdvancedDialogOpen}
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="目标切换高级设置"
      >
        <div className="advanced-settings-grid two-column">
          <NumberControl label="候选框最大宽高比" detail="拒绝宽高比或高宽比超过此值的异常细长框。值越大越宽松。" value={candidateRatioMaxAspect} min={1} max={20} step={0.1} onCommit={(value) => updateConfigField("control", "candidate_ratio_max_aspect", value)} />
          <NumberControl label="切换最小优势" detail="新候选综合分减去当前锁定目标综合分，至少达到此值才允许切换。" value={targetSwitchPreferenceAdvantage} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "target_switch_min_preference_advantage", value)} />
          <NumberControl label="切换最小连续性" detail="新候选 Track 的身份连续性至少达到此值，才允许进入切换确认。" value={targetSwitchContinuityScore} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "target_switch_min_continuity_score", value)} />
          <NumberControl label="目标切换确认延迟 ms" detail="新候选持续满足优势和连续性阈值达到此时间后，才正式替换当前目标。" value={targetSwitchDelayMs} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "target_switch_delay_ms", value)} />
        </div>
      </AdvancedSettingsDialog>

      <AdvancedSettingsDialog
        description="Tracker 负责跨帧身份关联，Kalman 负责位置估计。错误设置可能造成断轨、误关联或位置滞后。"
        eyebrow="参数设置 / Tracker"
        footerNote="关联算法固定为 Hungarian；仅输出 ACTIVE Track。"
        onClose={() => setTrackerSettingsDialogOpen(false)}
        open={trackerSettingsDialogOpen}
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="Tracker / Kalman 高级设置"
      >
        <div className="advanced-settings-grid two-column">
          <NumberControl label="归一化匹配距离" value={trackerMaxMatchDistance} min={0.1} max={5} step={0.05} onCommit={(value) => updateConfigField("control", "tracker_max_match_distance", value)} />
          <NumberControl label="位置代价权重" value={trackerPositionCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "tracker_position_cost_weight", value)} />
          <NumberControl label="IoU 代价权重" value={trackerIouCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "tracker_iou_cost_weight", value)} />
          <NumberControl label="最大漏检轮数" value={trackerMaxMissedFrames} min={0} max={10} step={1} onCommit={(value) => updateConfigField("control", "tracker_max_missed_frames", Math.round(value))} />
        </div>
        <div className="advanced-settings-divider">
          <span>Kalman 估计器</span>
          <small>{dualPhaseActive ? "当前算法仍使用 Tracker 的 Kalman 位置估计。" : "调整过程噪声与观测噪声。"}</small>
        </div>
        <div className="advanced-settings-grid two-column">
          <NumberControl label="加速度噪声" value={kalmanAccelerationNoise} min={0.001} max={10000} step={10} onCommit={(value) => updateConfigField("control", "kalman_acceleration_noise", value)} />
          <NumberControl label="X 测量噪声" value={kalmanMeasurementNoiseX} min={0.001} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_measurement_noise_x", value)} />
          <NumberControl label="Y 测量噪声" value={kalmanMeasurementNoiseY} min={0.001} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_measurement_noise_y", value)} />
        </div>
      </AdvancedSettingsDialog>

      {targetWeightsDialogOpen ? (
        <div
          className="target-weight-dialog-layer"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !dialogSaving) {
              setTargetWeightsDialogOpen(false);
            }
          }}
        >
          <section
            aria-busy={dialogSaving}
            aria-labelledby="target-weight-dialog-title"
            aria-modal="true"
            className="target-weight-dialog"
            ref={targetWeightsDialogRef}
            role="dialog"
            tabIndex={-1}
          >
            <header className="target-weight-dialog-header">
              <div>
                <span className="class-config-eyebrow">目标选择 / 评分策略</span>
                <h2 id="target-weight-dialog-title">调整目标选择权重</h2>
                <p>权重决定多个候选同时出现时，类别偏好、候选可靠性和准星距离各自占多大影响。</p>
              </div>
              <button
                aria-label="关闭权重调整"
                className="launch-dialog-close"
                disabled={dialogSaving}
                onClick={() => setTargetWeightsDialogOpen(false)}
                type="button"
              >
                <NovaIcon name="x-circle" size={18} />
              </button>
            </header>

            <div
              className="target-weight-dialog-body"
              {...({ inert: dialogSaving ? "" : undefined } as { inert?: string })}
            >
              <section className="target-weight-section">
                <div className="target-weight-section-heading">
                  <div>
                    <span>综合目标分数</span>
                    <small>三项原始值会自动归一化；当前总和为 {candidateSelectionWeightTotal.toFixed(2)}。</small>
                  </div>
                  <b>类别优先</b>
                </div>
                <div className="target-weight-composition" aria-label="综合目标分数权重占比">
                  <i className="class" style={{ flexGrow: normalizedSelectionClassWeight }} />
                  <i className="quality" style={{ flexGrow: normalizedSelectionQualityWeight }} />
                  <i className="distance" style={{ flexGrow: normalizedSelectionDistanceWeight }} />
                </div>
                <div className="target-weight-legend">
                  <span><i className="class" />类别 <b>{(normalizedSelectionClassWeight * 100).toFixed(0)}%</b></span>
                  <span><i className="quality" />质量 <b>{(normalizedSelectionQualityWeight * 100).toFixed(0)}%</b></span>
                  <span><i className="distance" />距离 <b>{(normalizedSelectionDistanceWeight * 100).toFixed(0)}%</b></span>
                </div>
                <div className="target-weight-controls">
                  <NumberControl label="综合分权重：类别" detail="类别顺序第一项得 1.0，第二项得 0.5，其余类别得 0.0。提高后更倾向优先类别。" value={candidateSelectionClassWeight} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "candidate_selection_class_weight", value)} />
                  <NumberControl label="综合分权重：质量" detail="检测置信度、同类别可见尺寸与 Track 可靠性形成的质量分。" value={candidateSelectionQualityWeight} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "candidate_selection_quality_weight", value)} />
                  <NumberControl label="综合分权重：距离" detail="候选瞄点到准星的距离，按当前目标选择半径归一化。降低后允许优先类别位于更远位置。" value={candidateSelectionDistanceWeight} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "candidate_selection_distance_weight", value)} />
                </div>
              </section>

              <section className="target-weight-section secondary">
                <div className="target-weight-section-heading">
                  <div>
                    <span>候选质量内部构成</span>
                    <small>质量分只占上方综合分的一部分，并且最终还会被 Track 可靠性限制。</small>
                  </div>
                  <b>置信度 {(normalizedQualityConfidenceWeight * 100).toFixed(0)}%</b>
                </div>
                <div className="target-weight-composition quality-composition" aria-label="候选质量权重占比">
                  <i className="confidence" style={{ flexGrow: normalizedQualityConfidenceWeight }} />
                  <i className="area" style={{ flexGrow: normalizedQualityAreaWeight }} />
                </div>
                <div className="target-weight-legend">
                  <span><i className="confidence" />置信度 <b>{(normalizedQualityConfidenceWeight * 100).toFixed(0)}%</b></span>
                  <span><i className="area" />同类别可见尺寸 <b>{(normalizedQualityAreaWeight * 100).toFixed(0)}%</b></span>
                </div>
                <div className="target-weight-controls two-column">
                  <NumberControl label="质量权重：置信度" detail="检测器输出的类别置信度。与可见尺寸权重归一化后使用。" value={candidateQualityConfidenceWeight} min={0} max={2} step={0.05} onCommit={(value) => updateConfigField("control", "candidate_quality_confidence_weight", value)} />
                  <NumberControl label="质量权重：可见尺寸" detail="bbox 面积相对于当前画面同类别候选面积中位数的平方根；不是相对于整个 ROI。" value={candidateQualityAreaWeight} min={0} max={2} step={0.05} onCommit={(value) => updateConfigField("control", "candidate_quality_area_weight", value)} />
                </div>
              </section>
            </div>

            <footer className="target-weight-dialog-footer">
              <span className={dialogSaveError ? "dialog-save-status error" : "dialog-save-status"} role="status" aria-live="polite">
                {dialogSaving
                  ? "正在自动保存并同步运行配置…"
                  : dialogSaveError
                    ? `保存失败 · ${dialogSaveError}`
                    : "已自动保存 · 修改后立即生效，无需重启主链。"}
              </span>
              <button className="console-button primary" disabled={dialogSaving} onClick={() => setTargetWeightsDialogOpen(false)} type="button">
                关闭
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {classConfigDialogOpen ? (
        <div
          className="class-config-dialog-layer"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !dialogSaving) {
              setClassConfigDialogOpen(false);
            }
          }}
        >
          <section
            aria-busy={dialogSaving}
            aria-labelledby="class-config-dialog-title"
            aria-modal="true"
            className="class-config-dialog"
            ref={classConfigDialogRef}
            role="dialog"
            tabIndex={-1}
          >
            <header className="class-config-dialog-header">
              <div>
                <span className="class-config-eyebrow">参数设置 / 类别配置</span>
                <h2 id="class-config-dialog-title">管理类别配置</h2>
                <p>把模型类别归入头部、身体或其他瞄点类型，再在人物靶上统一标定三条垂直瞄点线。</p>
              </div>
              <button
                aria-label="关闭类别配置"
                className="launch-dialog-close"
                disabled={dialogSaving}
                onClick={() => setClassConfigDialogOpen(false)}
                type="button"
              >
                <NovaIcon name="x-circle" size={18} />
              </button>
            </header>

            <div
              className="class-config-dialog-layout"
              {...({ inert: dialogSaving ? "" : undefined } as { inert?: string })}
            >
              <aside className="class-profile-rail" aria-label="类别配置文件">
                <div className="class-profile-rail-heading">
                  <span>配置文件</span>
                  <b>{detectionProfileNames.length}</b>
                </div>
                <div className="class-profile-list">
                  {(detectionProfileNames.length > 0 ? detectionProfileNames : ["default"]).map((name) => (
                    <button
                      aria-current={name === activeDetectionProfile ? "page" : undefined}
                      className={name === activeDetectionProfile ? "active" : ""}
                      key={name}
                      onClick={() => void updateConfigField("inference", "detection_class_profile", name)}
                      type="button"
                    >
                      <span>{name}</span>
                      <small>{(detectionProfiles[name] ?? []).filter(Boolean).length} 类</small>
                    </button>
                  ))}
                </div>
                <div className="class-profile-create">
                  <label htmlFor="new-class-profile">新建配置</label>
                  <input
                    id="new-class-profile"
                    onChange={(event) => setNewClassProfileName(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        void createClassProfile(false);
                      }
                    }}
                    placeholder="例如 valorant"
                    value={newClassProfileName}
                  />
                  <div className="class-profile-create-actions">
                    <button
                      className="console-button primary"
                      disabled={busy !== null || newClassProfileName.trim() === ""}
                      onClick={() => void createClassProfile(false)}
                      type="button"
                    >
                      新建空白
                    </button>
                    <button
                      className="console-button"
                      disabled={busy !== null || newClassProfileName.trim() === ""}
                      onClick={() => void createClassProfile(true)}
                      type="button"
                    >
                      <NovaIcon name="copy" size={15} />
                      复制当前
                    </button>
                  </div>
                </div>
              </aside>

              <div className="class-config-workspace">
                <div className="class-config-profile-bar">
                  <div>
                    <span>当前配置名称</span>
                    <small>重命名会同步迁移该配置对应的类别瞄点类型映射。</small>
                  </div>
                  <input
                    aria-label="当前类别配置名称"
                    onChange={(event) => setRenamedClassProfileName(event.target.value)}
                    value={renamedClassProfileName}
                  />
                  <button
                    className="console-button"
                    disabled={busy !== null || renamedClassProfileName.trim() === "" || renamedClassProfileName.trim() === activeDetectionProfile}
                    onClick={() => void renameClassProfile()}
                    type="button"
                  >
                    <NovaIcon name="edit" size={15} />
                    重命名
                  </button>
                  <button
                    aria-label={`删除类别配置 ${activeDetectionProfile}`}
                    className="console-button danger"
                    disabled={busy !== null || detectionProfileNames.length <= 1}
                    onClick={() => {
                      if (classProfileDeleteArmed) {
                        void deleteClassProfile();
                      } else {
                        setClassProfileDeleteArmed(true);
                      }
                    }}
                    type="button"
                  >
                    <NovaIcon name="delete" size={15} />
                    {classProfileDeleteArmed ? "确认删除" : "删除"}
                  </button>
                </div>

                <AimTargetRange
                  disabled={busy !== null}
                  ratios={aimRoleRatios}
                  onCommit={updateAimRoleRatio}
                />

                <section className="class-filter-section" aria-labelledby="class-filter-title">
                  <div className="class-filter-heading">
                    <span>
                      <b id="class-filter-title">参与目标选择的类别</b>
                      <small>可同时选择多个类别；高亮卡片会进入候选目标计算。</small>
                    </span>
                    <div className="class-filter-actions">
                      <span>{selectedDetectionClassIds.size}/{classEditorIds.length} 已选择</span>
                      <button
                        className="console-button secondary"
                        disabled={busy !== null || selectedDetectionClassIds.size === classEditorIds.length}
                        onClick={() => void updateConfigField("inference", "detection_class_filter", "all")}
                        type="button"
                      >
                        全部选择
                      </button>
                      <button
                        className="console-button"
                        disabled={busy !== null || selectedDetectionClassIds.size === 0}
                        onClick={() => void updateConfigField("inference", "detection_class_filter", "none")}
                        type="button"
                      >
                        全部取消
                      </button>
                    </div>
                  </div>
                  <div className="class-filter-options" role="group" aria-label="目标类别多选">
                    {classEditorIds.map((classId) => {
                      const selected = selectedDetectionClassIds.has(classId);
                      const configuredName = detectionClasses[classId] ?? "";
                      return (
                        <button
                          aria-pressed={selected}
                          className={selected ? "selected" : ""}
                          disabled={busy !== null}
                          key={`class-filter-${classId}`}
                          onClick={() => void toggleDetectionClass(classId)}
                          type="button"
                        >
                          <span className="class-filter-check" aria-hidden="true">{selected ? "✓" : ""}</span>
                          <b>cls {classId}</b>
                          <small>{configuredName ? classDisplayName(configuredName, classId) : `未知类别（cls ${classId}）`}</small>
                        </button>
                      );
                    })}
                  </div>
                </section>

                <div className="class-editor" role="table" aria-label="模型类别与瞄点设置">
                  <div className="class-editor-head" role="row">
                    <span>目标</span><span>类别名称</span><span>目标优先级</span><span>瞄点类型</span>
                  </div>
                  {orderedClassEditorIds.map((classId) => {
                    const configuredName = detectionClasses[classId] ?? "";
                    const displayName = configuredName ? classDisplayName(configuredName, classId) : "";
                    const priorityIndex = orderedClassEditorIds.indexOf(classId);
                    return (
                      <div className="class-editor-row" role="row" key={`class-editor-${classId}`}>
                        <button
                          aria-label={`${selectedDetectionClassIds.has(classId) ? "取消" : "允许"} cls ${classId} 参与目标选择`}
                          aria-pressed={selectedDetectionClassIds.has(classId)}
                          className={`class-role-select ${selectedDetectionClassIds.has(classId) ? "selected" : ""}`}
                          disabled={busy !== null}
                          onClick={() => void toggleDetectionClass(classId)}
                          type="button"
                        >
                          <i aria-hidden="true">{selectedDetectionClassIds.has(classId) ? "✓" : ""}</i>
                          <b>cls {classId}</b>
                        </button>
                        <InlineTextControl
                          ariaLabel={`cls ${classId} 类别名称`}
                          value={displayName}
                          placeholder={`未知类别（cls ${classId}）`}
                          onCommit={(value) => updateDetectionClassName(classId, value)}
                        />
                        <label className="class-priority-control">
                          <span className="visually-hidden">cls {classId} 目标优先级</span>
                          <select
                            aria-label={`cls ${classId} 目标优先级`}
                            disabled={busy !== null}
                            value={priorityIndex}
                            onChange={(event) => void setClassPriorityPosition(classId, Number(event.target.value))}
                          >
                            {orderedClassEditorIds.map((_, index) => (
                              <option key={`priority-${classId}-${index}`} value={index}>第 {index + 1} 位</option>
                            ))}
                          </select>
                        </label>
                        <div className="class-role-segmented" role="group" aria-label={`cls ${classId} 瞄点类型`}>
                          {(["head", "body", "other"] as AimRole[]).map((role) => (
                            <button
                              aria-pressed={(activeClassRoles[String(classId)] ?? "other") === role}
                              className={`${role} ${(activeClassRoles[String(classId)] ?? "other") === role ? "active" : ""}`}
                              disabled={busy !== null}
                              key={role}
                              onClick={() => void updateClassAimRole(classId, role)}
                              type="button"
                            >
                              {role === "head" ? "头部" : role === "body" ? "身体" : "其他"}
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="console-field-hint">
                  未知 class id 会显示为“未知类别（cls N）”并使用“其他”瞄点类型；人物靶只负责展示比例，实际值仍相对于各自 bbox。
                </p>
              </div>
            </div>

            <footer className="class-config-dialog-footer">
              <span className={dialogSaveError ? "dialog-save-status error" : "dialog-save-status"} role="status" aria-live="polite">
                {dialogSaving
                  ? "正在自动保存类别配置…"
                  : dialogSaveError
                    ? `保存失败 · ${dialogSaveError}`
                    : `已自动保存 · 当前配置：${activeDetectionProfile}`}
              </span>
              <button className="console-button primary" disabled={dialogSaving} onClick={() => setClassConfigDialogOpen(false)} type="button">
                关闭
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {errorCenterOpen ? (
        <div
          className="error-center-layer"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setErrorCenterOpen(false);
          }}
        >
          <section
            aria-labelledby="error-center-title"
            aria-modal="true"
            className="error-center-dialog"
            ref={errorCenterDialogRef}
            role="dialog"
            tabIndex={-1}
          >
            <header className="error-center-header">
              <div>
                <span>SYSTEM DIAGNOSTICS</span>
                <h2 id="error-center-title">异常信息</h2>
                <p>页面保持安静；网络、后端与操作错误统一收拢在这里。</p>
              </div>
              <button aria-label="关闭异常信息" onClick={() => setErrorCenterOpen(false)} type="button">
                <NovaIcon name="x-circle" size={18} />
              </button>
            </header>
            <div className="error-center-body">
              {currentErrorDetails.length > 0 ? currentErrorDetails.map((item) => (
                <article className="error-center-item" key={item.key}>
                  <NovaIcon name="triangle-alert" size={17} />
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.detail}</p>
                    {item.time ? <time>{formatDate(new Date(item.time))}</time> : null}
                  </div>
                </article>
              )) : (
                <div className="error-center-empty">
                  <NovaIcon name="shield-check" size={28} />
                  <strong>当前没有异常</strong>
                  <span>后端连接和最近操作均未报告错误。</span>
                </div>
              )}
            </div>
            <footer className="error-center-footer">
              <button
                className="console-button"
                disabled={errorNotices.length === 0}
                onClick={clearErrorNotices}
                type="button"
              >
                清除历史记录
              </button>
              <button className="console-button primary" onClick={() => setErrorCenterOpen(false)} type="button">完成</button>
            </footer>
          </section>
        </div>
      ) : null}

      <ModelSwitchDialog
        completedStages={modelSwitchCompletedStages}
        currentStage={modelSwitchStageIndex}
        detail={modelSwitchProgressDetail}
        dialogRef={modelSwitchDialogRef}
        error={modelSwitchDialogError}
        modelName={selectedCatalogModel?.relative_path ?? "TensorRT Engine"}
        onClose={() => setModelSwitchDialogOpen(false)}
        open={modelSwitchDialogOpen}
        status={modelSwitchDialogStatus}
      />

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
                  <p>所有步骤将按顺序执行；任一项失败都会中断后续流程。</p>
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
              <div
                aria-atomic="true"
                aria-live="polite"
                className={`launch-overview ${launchStatus}`}
              >
                <div className="launch-progress-meta">
                  <span>{launchSummary}</span>
                  <strong>{launchCompletedStages} / {launchStages.length}</strong>
                </div>
                <div className="launch-progress-track">
                  <div
                    className="launch-progress-bar"
                    style={{ transform: `scaleX(${launchProgress / 100})` }}
                  />
                </div>
              </div>

              <ol className="launch-stage-list">
                {launchStages.map((stage, index) => {
                  const stepState = resolveLaunchStepState(
                    index,
                    launchStatus,
                    launchStageIndex,
                    launchCompletedStages
                  );
                  const stepDetail = stepState === "failed"
                    ? launchError || "该步骤执行失败。"
                    : stepState === "running" && launchProgressDetail
                      ? launchProgressDetail
                      : "";
                  return (
                    <li className={`launch-stage-row ${stepState}`} key={stage.title}>
                      <div className="launch-stage-marker" aria-hidden="true">
                        <LaunchStepIndicator state={stepState} />
                      </div>
                      <div className="launch-stage-copy">
                        <div className="launch-stage-heading">
                          <span>{String(index + 1).padStart(2, "0")}</span>
                          <strong>{stage.title}</strong>
                          <em>{launchStepStateLabel(stepState)}</em>
                        </div>
                        <p>{stage.caption}</p>
                        {stepDetail ? <small>{stepDetail}</small> : null}
                      </div>
                    </li>
                  );
                })}
              </ol>
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

function launchStepStateLabel(state: LaunchStepState): string {
  switch (state) {
    case "running":
      return "执行中";
    case "success":
      return "已完成";
    case "failed":
      return "失败";
    default:
      return "未执行";
  }
}

function LaunchStepIndicator({ state }: { state: LaunchStepState }) {
  return (
    <span className={`launch-step-indicator ${state}`}>
      <svg viewBox="0 0 24 24" focusable="false">
        {state === "success" ? (
          <>
            <circle cx="12" cy="12" r="9" />
            <path d="m7.8 12.2 2.7 2.7 5.9-6" />
          </>
        ) : state === "failed" ? (
          <>
            <circle cx="12" cy="12" r="9" />
            <path d="m8.7 8.7 6.6 6.6m0-6.6-6.6 6.6" />
          </>
        ) : state === "running" ? (
          <>
            <circle className="launch-step-track" cx="12" cy="12" r="9" />
            <path className="launch-step-arc" d="M12 3a9 9 0 0 1 9 9" />
          </>
        ) : (
          <circle cx="12" cy="12" r="8" />
        )}
      </svg>
    </span>
  );
}

function ModuleSwitch({
  label,
  detail,
  enabled,
  disabled = false,
  onToggle
}: {
  label: string;
  detail: string;
  enabled: boolean;
  disabled?: boolean;
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
    if (pending || disabled) {
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
  }, [disabled, onToggle, pending, visualEnabled]);

  return (
    <button
      className={visualEnabled ? "module-switch on" : "module-switch"}
      onClick={() => void toggle()}
      disabled={pending || disabled}
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

function findCatalogModelByPath(
  directory: ModelCatalogDirectory,
  relativePath: string
): ModelCatalogModel | null {
  for (const child of directory.children) {
    if (child.type === "model") {
      if (child.relative_path === relativePath) {
        return child;
      }
      continue;
    }
    const nested = findCatalogModelByPath(child, relativePath);
    if (nested) {
      return nested;
    }
  }
  return null;
}

function PreviewFrame({
  supported,
  active,
  togglePending,
  onToggle,
  imageAvailable,
  unavailableReason,
  runtime,
  roiSize
}: {
  supported: boolean;
  active: boolean;
  togglePending: boolean;
  onToggle: (enabled: boolean) => void;
  imageAvailable: boolean;
  unavailableReason: string;
  runtime: RuntimeState | null;
  roiSize: number;
}) {
  const previewRef = useRef<HTMLDivElement | null>(null);
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
  const showImage = supported && active && imageAvailable;
  const showOverlay = supported && active && runtime?.running === true && detections.length > 0;
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
      ref={previewRef}
      className={showOverlay ? "console-preview has-overlay" : "console-preview"}
      style={{
        "--roi-size": `${displaySize}px`,
        "--preview-aspect": `${previewWidth} / ${previewHeight}`
      } as CSSProperties}
    >
      {supported && active ? (
        <div className="console-preview-live-control">
          <span>实时预览会占用 Jetson 资源</span>
          <button disabled={togglePending} onClick={() => onToggle(false)} type="button">
            {togglePending ? "正在关闭…" : "关闭预览"}
          </button>
        </div>
      ) : null}
      <div className="console-preview-frame">
        {showImage ? <img alt="实时画面 / ROI" src={streamUrl(configVersion, configVersion)} /> : null}
        {supported && active && !showImage ? <div className="console-preview-unavailable">{unavailableReason}</div> : null}
        {supported && !active ? (
          <div className="console-preview-gate" role="status">
            <span className="console-preview-gate-kicker">性能保护已启用</span>
            <strong>实时预览已暂停</strong>
            <p>推理、跟踪与控制继续运行。开启画面会占用 NVJPEG 与内存带宽。</p>
            <button disabled={togglePending} onClick={() => onToggle(true)} type="button">
              {togglePending ? "正在开启…" : "开启实时预览"}
            </button>
          </div>
        ) : null}
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
