import { ChangeEvent, lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureState,
  CaptureSelectPayload,
  getApiErrorCode,
  connectKmNet,
  diagnosticMoveKmNet,
  disconnectKmNet,
  clearCrosshairTemplate,
  crosshairTemplatePreviewUrl,
  emergencyStopRuntimePipeline,
  getRuntimeState,
  getRuntimeConfig,
  learnCrosshair,
  HealthResponse,
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelCatalogResponse,
  ModelProject,
  ModelRecommendation,
  ModelVersion,
  ParserPresetId,
  RuntimeConfig,
  RuntimeConfigValue,
  RuntimeState,
  RuntimeStatusTopic,
  getCaptureCapabilities,
  getModelArtifacts,
  getModelCatalog,
  getModelVersions,
  publishModel,
  registerCatalogModel,
  selectCaptureProfile,
  setCapturePreviewEnabled,
  startRuntimePipeline,
  stopCapture,
  stopRuntimePipeline,
  streamUrl,
  updateRuntimeConfig,
  updateRuntimeConfigField,
  updateModelArtifactMetadata
} from "../../api";
import { reportError, reportSuccess, useClearErrorNotices, useErrorNotices } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";
import { NovaIcon, StatusBadge, ThemeGallery, ThemeToggle } from "../../components/visual";
import { CurrentModelSummary } from "../models/CurrentModelSummary";
import { ModelSwitchDialog, type ModelSwitchDialogStatus } from "../models/ModelSwitchDialog";
import {
  runtimeDeliveryDescription,
  runtimeDeliveryLabel,
  runtimeDeliveryTone,
  type RuntimeDeliveryStatus
} from "../shared/runtimeDelivery";
import { AdvancedSettingsDialog } from "./AdvancedSettingsDialog";
import {
  ActionConfirmationDialog,
  type ActionConfirmationRequest
} from "./ActionConfirmationDialog";
import { AimTargetRange, type AimRole, type AimRoleRatios } from "./AimTargetRange";
import { CommitNumberControl, InlineTextControl, NumberControl, TextControl } from "./StudioControls";
import { CONSOLE_PAGES, DEFAULT_CONSOLE_PAGE, StudioNavigation, type ConsolePage } from "./StudioNavigation";
import { StudioPageHeader } from "./StudioPageHeader";
import { KvCard, Metric, SectionTitle } from "./StudioPresentation";
import { trapDialogTabKey } from "./dialogFocus";
import "./studio-settings.css";

const DEFAULT_CONTROL_ALGORITHM = "dual_phase_atan_robust_predictive_v2";
const CONTROL_ALGORITHM_LABEL = "双阶段 Atan 控制";
const CONTROL_ALGORITHM_DESCRIPTION = "唯一生产控制器：单目标预测、角度投影、连续双阶段 Atan 响应、限幅与量化。";
const ModelManagerDialog = lazy(() =>
  import("../models/ModelManagerDialog").then((module) => ({
    default: module.ModelManagerDialog
  }))
);

function ModelManagerLoadingDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="model-manager-dialog-layer">
      <section
        aria-labelledby="model-manager-loading-title"
        aria-modal="true"
        className="model-manager-dialog"
        role="dialog"
      >
        <header className="model-manager-dialog-header">
          <div className="model-manager-dialog-title">
            <span className="model-manager-dialog-icon" aria-hidden="true">
              <NovaIcon name="models" size={22} />
            </span>
            <div>
              <span className="class-config-eyebrow">MODEL VAULT</span>
              <h2 id="model-manager-loading-title">模型管理与切换</h2>
              <p>正在读取模型管理界面，运行主链不受影响。</p>
            </div>
          </div>
          <button aria-label="关闭模型管理" className="launch-dialog-close" onClick={onClose} type="button">
            <NovaIcon name="x-circle" size={18} />
          </button>
        </header>
        <div className="model-manager-dialog-body model-manager-loading-body" role="status">
          <span className="route-loading-mark" aria-hidden="true" />
          <strong>正在加载模型目录组件…</strong>
          <small>弱网下可能需要几秒，当前模型和推理不会切换。</small>
        </div>
      </section>
    </div>
  );
}

type StudioConsoleViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  runtimeConfig: RuntimeConfig | null;
  projects: ModelProject[];
  errors: Partial<Record<string, string>>;
  lastUpdated: Date | null;
  realtimeStatus: RuntimeDeliveryStatus;
  onEnsureProjects: (force?: boolean) => Promise<ModelProject[]>;
  onRefresh: () => Promise<void>;
  onRuntimeConfigChange: (config: RuntimeConfig) => void;
  onRuntimeStateChange: (runtime: RuntimeState) => void;
  onStatusTopicChange: (topic: RuntimeStatusTopic) => void;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type LaunchStatus = "idle" | "running" | "success" | "failed" | "cancelled";
type LaunchStepState = "pending" | "running" | "success" | "failed";
type ConfigDialogId = "class-config" | "target-weights" | "algorithm" | "target-advanced" | "tracker";
type DualPhasePipelineField =
  | "freshness_threshold_ms"
  | "projection_fov_x_deg"
  | "projection_counts_per_360"
  | "near_threshold_px"
  | "far_kp"
  | "near_kp"
  | "atan_scale_counts"
  | "far_max_counts_per_update"
  | "near_max_counts_per_update"
  | "prediction_enabled"
  | "velocity_smoothing_frames"
  | "velocity_history_reset_gap_ms"
  | "velocity_spread_base_px_ms"
  | "velocity_spread_relative"
  | "velocity_change_base_px_ms"
  | "velocity_change_relative"
  | "prediction_lead_frames"
  | "prediction_far_absolute_cap_px"
  | "prediction_far_base_cap_px"
  | "prediction_far_relative_cap"
  | "prediction_near_absolute_cap_px"
  | "prediction_near_base_cap_px"
  | "prediction_near_relative_cap"
  | "arrival_radius_counts"
  | "residual_cap"
  | "output_interval_ms"
  | "actuation_feedback_delay_ms";

type TargetingPipelineField =
  | "target_fov_radius_px"
  | "target_min_confidence"
  | "target_track_max_age"
  | "target_track_max_lost_age_ms"
  | "tracker_max_match_distance"
  | "tracker_position_cost_weight"
  | "tracker_iou_cost_weight"
  | "tracker_scale_cost_weight"
  | "tracker_max_size_ratio"
  | "tracker_max_association_dt_ms"
  | "target_selection_class_ratio"
  | "target_switch_min_preference_advantage"
  | "target_switch_min_continuity_score"
  | "target_switch_delay_ms"
  | "candidate_max_aspect_ratio";

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
    caption: "请求后端启动采集、画面识别与控制功能。"
  },
  {
    title: "激活鼠标算法",
    caption: "确认目标选择、跟踪与 Atan 鼠标算法已开始消费 DetectionBatch；输出设备不影响本步骤。"
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
const KMNET_RECOMMENDED = {
  auto_connect: true,
  backend: "native_udp",
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

function serializeRustClassAimRatios(
  roles: Record<string, AimRole>,
  ratios: AimRoleRatios
): string {
  return Object.entries(roles)
    .flatMap(([classId, role]) => {
      const numericClassId = Number(classId);
      return Number.isInteger(numericClassId) && numericClassId >= 0 && numericClassId <= 255
        ? [[numericClassId, ratios[role]] as const]
        : [];
    })
    .sort(([left], [right]) => left - right)
    .map(([classId, ratio]) => `${classId}:${ratio.toFixed(2)}`)
    .join(",");
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
    FIRING_INACTIVE: "等待真实左键",
    DT_INVALID: "控制周期无效",
    OBSERVATION_AGE_INVALID: "观测时间无效",
    TARGET_STALE: "目标观测已过期",
    TARGET_INVALID: "等待有效目标",
    ERROR_INVALID: "垂直误差无效",
    POSITION_BRAKE: "位置保护刹车",
    RECOIL_RATE_ZERO: "压枪速率为零",
  };
  return labels[reason] ?? (reason || "等待真实左键或有效目标");
}

function formatRecoilState(stateValue: unknown, reasonValue: unknown): string {
  const state = readString(stateValue);
  const labels: Record<string, string> = {
    IDLE: "待机",
    STARTUP: "启动渐入",
    ACTIVE: "即时追加",
    HOLD: "稳定保持",
    BRAKE: "位置刹车",
    STALE: "观测过期",
  };
  return labels[state] ?? formatRecoilBlockReason(reasonValue);
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
const STANDARD_DECIMAL_DIGITS = 2;

function formatNumber(value: unknown, digits = STANDARD_DECIMAL_DIGITS): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? number.toFixed(digits) : NO_SAMPLE;
}

function formatPercent(value: unknown, digits = STANDARD_DECIMAL_DIGITS): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : NO_SAMPLE;
}

function formatOptionalNumber(value: unknown, digits = STANDARD_DECIMAL_DIGITS, unit = ""): string {
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

function readNullableBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function formatAxisSettlement(x: unknown, y: unknown): string {
  const settledX = readNullableBoolean(x);
  const settledY = readNullableBoolean(y);
  if (settledX === null || settledY === null) {
    return NO_SAMPLE;
  }
  return `${settledX ? "X 已到位" : "X 调整中"} / ${settledY ? "Y 已到位" : "Y 调整中"}`;
}

function formatFeedbackGate(x: unknown, y: unknown): string {
  const pendingX = readNullableBoolean(x);
  const pendingY = readNullableBoolean(y);
  if (pendingX === null || pendingY === null) {
    return NO_SAMPLE;
  }
  return pendingX || pendingY ? "等待新画面" : "允许闭环更新";
}

function freshnessSummary(ageMs: number | null, thresholdMs: number | null): string {
  if (ageMs === null) {
    return "等待结果样本";
  }
  if (thresholdMs === null || thresholdMs <= 0) {
    return "阈值不可用";
  }
  if (ageMs <= thresholdMs) {
    return "控制可用";
  }
  return ageMs <= thresholdMs * 2 ? "超过控制阈值" : "严重过期";
}

function formatPoint(x: unknown, y: unknown, digits = STANDARD_DECIMAL_DIGITS, unit = ""): string {
  const xNumber = readNullableNumber(x);
  const yNumber = readNullableNumber(y);
  if (xNumber === null || yNumber === null) {
    return NO_SAMPLE;
  }
  const suffix = unit ? ` ${unit}` : "";
  return `${xNumber.toFixed(digits)}, ${yNumber.toFixed(digits)}${suffix}`;
}

function formatDate(value: Date | null): string {
  return value ? value.toLocaleTimeString("zh-CN", { hour12: false }) : "--:--:--";
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

function canonicalCaptureFormat(value: string): string {
  const normalized = value.trim().toUpperCase();
  if (normalized === "MJPEG") return "MJPG";
  if (normalized === "YUY2") return "YUYV";
  return normalized;
}

function choiceId(choice: CapabilityChoice): string {
  return `${choice.pixel_format}:${choice.width}x${choice.height}@${choice.fps}`;
}

function choiceLabel(choice: CapabilityChoice): string {
  return `${choice.pixel_format} / ${choice.width}x${choice.height} / ${choice.fps} FPS`;
}

function choiceMatchesConfig(choice: CapabilityChoice, config: Record<string, unknown>): boolean {
  return (
    canonicalCaptureFormat(readString(config.pixel_format, "")) ===
      canonicalCaptureFormat(choice.pixel_format) &&
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

function runtimeConfigValuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => runtimeConfigValuesEqual(value, right[index]));
  }
  if (
    left === null
    || right === null
    || typeof left !== "object"
    || typeof right !== "object"
  ) {
    return false;
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord);
  const rightKeys = Object.keys(rightRecord);
  return leftKeys.length === rightKeys.length
    && leftKeys.every(
      (key) => Object.prototype.hasOwnProperty.call(rightRecord, key)
        && runtimeConfigValuesEqual(leftRecord[key], rightRecord[key])
    );
}

function runtimeConfigsEqual(left: RuntimeConfig | null, right: RuntimeConfig | null): boolean {
  return runtimeConfigValuesEqual(left, right);
}

const CONFIG_SECTION_LABELS: Record<string, string> = {
  capture: "采集与 ROI",
  inference: "模型推理",
  preprocess: "预处理",
  pipeline: "目标选择与控制算法",
  control: "控制输出",
  hardware: "kmNet 硬件",
  crosshair: "准星学习",
  limits: "安全限制",
  consumers: "数据消费",
  server: "后端服务"
};

function changedRuntimeConfigSections(current: RuntimeConfig, candidate: RuntimeConfig): string[] {
  const ignored = new Set(["revision", "version", "roi_size"]);
  return Array.from(new Set([...Object.keys(current), ...Object.keys(candidate)]))
    .filter((key) => !ignored.has(key) && !runtimeConfigValuesEqual(current[key], candidate[key]))
    .map((key) => CONFIG_SECTION_LABELS[key] ?? key);
}

export function StudioConsoleView({
  health,
  runtime,
  runtimeConfig,
  projects,
  errors,
  lastUpdated,
  realtimeStatus,
  onEnsureProjects,
  onRefresh,
  onRuntimeConfigChange,
  onRuntimeStateChange,
  onStatusTopicChange
}: StudioConsoleViewProps) {
  const [activePage, setActivePage] = useState<ConsolePage>(() => pageFromUrl());

  useEffect(() => {
    if (activePage === "infer" || activePage === "control" || activePage === "latency" || activePage === "capture") {
      onStatusTopicChange(activePage);
      return;
    }
    onStatusTopicChange("summary");
  }, [activePage, onStatusTopicChange]);
  const [wideThemeGallery, setWideThemeGallery] = useState(
    () => window.matchMedia("(min-width: 1280px)").matches
  );
  const [device, setDevice] = useState(
    readString(nestedRecord(runtimeConfig, "capture").device, runtime?.capture?.device ?? "/dev/video0")
  );

  useEffect(() => {
    const query = window.matchMedia("(min-width: 1280px)");
    const update = () => setWideThemeGallery(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
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
  const [selectedModelCatalogPath, setSelectedModelCatalogPath] = useState<string>();
  const [kmnetTestDx, setKmnetTestDx] = useState(10);
  const [kmnetTestDy, setKmnetTestDy] = useState(0);
  const [kmnetTestMessage, setKmnetTestMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [errorCenterOpen, setErrorCenterOpen] = useState(false);
  const errorNotices = useErrorNotices();
  const clearErrorNotices = useClearErrorNotices();
  const [modelSwitchMessage, setModelSwitchMessage] = useState("");
  const [modelManagerDialogOpen, setModelManagerDialogOpen] = useState(false);
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
  const [confirmationRequest, setConfirmationRequest] = useState<ActionConfirmationRequest | null>(null);
  const [confirmationBusy, setConfirmationBusy] = useState(false);
  const [confirmationError, setConfirmationError] = useState<string | null>(null);
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
  const [configDialogDirty, setConfigDialogDirty] = useState(false);
  const [configDialogSaving, setConfigDialogSaving] = useState(false);
  const [dialogSaveError, setDialogSaveError] = useState<string | null>(null);
  const dialogSaving = configDialogSaving;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const configDraftRef = useRef<RuntimeConfig | null>(cloneRuntimeConfig(runtimeConfig));
  const runtimeConfigLatestRef = useRef<RuntimeConfig | null>(runtimeConfig);
  const activeConfigDialogRef = useRef<ConfigDialogId | null>(null);
  const configDialogBaselineRef = useRef<RuntimeConfig | null>(null);
  const loadedModelProjectIdRef = useRef<number | "">("");
  const loadedModelVersionIdRef = useRef<number | "">("");
  const preferLatestModelVersionRef = useRef(false);
  const pendingConfigWritesRef = useRef(0);
  const configWriteSeqRef = useRef(0);
  const configWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const launchCancelledRef = useRef(false);
  const launchTimerRef = useRef<number | null>(null);
  const launchTimerResolveRef = useRef<(() => void) | null>(null);
  const classConfigDialogRef = useRef<HTMLElement | null>(null);
  const targetWeightsDialogRef = useRef<HTMLElement | null>(null);
  const errorCenterDialogRef = useRef<HTMLElement | null>(null);
  const modelSwitchDialogRef = useRef<HTMLElement | null>(null);
  const dialogSavingRef = useRef(false);

  const setConfigDialogVisibility = useCallback((dialog: ConfigDialogId, open: boolean) => {
    if (dialog === "class-config") setClassConfigDialogOpen(open);
    else if (dialog === "target-weights") setTargetWeightsDialogOpen(open);
    else if (dialog === "algorithm") setAlgorithmSettingsDialogOpen(open);
    else if (dialog === "target-advanced") setTargetAdvancedDialogOpen(open);
    else setTrackerSettingsDialogOpen(open);
  }, []);

  const finishConfigDialog = useCallback((dialog: ConfigDialogId) => {
    setConfigDialogVisibility(dialog, false);
    activeConfigDialogRef.current = null;
    configDialogBaselineRef.current = null;
    setConfigDialogDirty(false);
    setDialogSaveError(null);
  }, [setConfigDialogVisibility]);

  const openConfigDialog = useCallback((dialog: ConfigDialogId) => {
    if (dialogSavingRef.current || pendingConfigWritesRef.current > 0) {
      return;
    }
    const source = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfigLatestRef.current);
    if (!source) {
      return;
    }
    const baseline = normalizeRuntimeConfig(source);
    activeConfigDialogRef.current = dialog;
    configDialogBaselineRef.current = structuredClone(baseline) as RuntimeConfig;
    configDraftRef.current = baseline;
    setConfigDraft(baseline);
    setConfigDialogDirty(false);
    setDialogSaveError(null);
    setConfigDialogVisibility(dialog, true);
  }, [setConfigDialogVisibility]);

  const saveConfigDialog = useCallback(async (dialog: ConfigDialogId) => {
    if (dialogSavingRef.current || activeConfigDialogRef.current !== dialog) {
      return;
    }

    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
      await Promise.resolve();
    }

    const draft = configDraftRef.current;
    const baseline = configDialogBaselineRef.current;
    if (!draft || !baseline || runtimeConfigsEqual(draft, baseline)) {
      const latest = cloneRuntimeConfig(runtimeConfigLatestRef.current) ?? draft;
      configDraftRef.current = latest;
      setConfigDraft(latest);
      finishConfigDialog(dialog);
      return;
    }

    dialogSavingRef.current = true;
    setConfigDialogSaving(true);
    setBusy("config-dialog.save");
    setDialogSaveError(null);
    setLocalError(null);
    try {
      const result = await updateRuntimeConfig(draft);
      const applied = normalizeRuntimeConfig(result.config);
      runtimeConfigLatestRef.current = applied;
      configDraftRef.current = applied;
      setConfigDraft(applied);
      onRuntimeConfigChange(applied);
      reportSuccess(
        "配置已保存",
        result.restart_required
          ? "新参数已写入配置；重启 novasightd 后进入运行主链。"
          : "新参数已经应用。",
        "config-dialog"
      );
      finishConfigDialog(dialog);
    } catch (error) {
      const message = getErrorMessage(error);
      setLocalError(`配置保存失败：${message}`);
      setDialogSaveError(message);
      reportError(error, { source: "config-dialog", title: "配置保存失败" });
    } finally {
      dialogSavingRef.current = false;
      setConfigDialogSaving(false);
      setBusy(null);
    }
  }, [finishConfigDialog, onRuntimeConfigChange]);

  const requestDismissConfigDialog = useCallback(async (dialog: ConfigDialogId) => {
    if (dialogSavingRef.current || activeConfigDialogRef.current !== dialog) {
      return;
    }

    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
      await Promise.resolve();
    }

    const draft = configDraftRef.current;
    const baseline = configDialogBaselineRef.current;
    if (!draft || !baseline || runtimeConfigsEqual(draft, baseline)) {
      const latest = cloneRuntimeConfig(runtimeConfigLatestRef.current) ?? draft;
      configDraftRef.current = latest;
      setConfigDraft(latest);
      finishConfigDialog(dialog);
      return;
    }

    setConfirmationRequest({
      eyebrow: "未保存修改",
      title: "放弃这次参数修改？",
      description: "关闭或按 Esc 不再隐式保存。只有点击“保存并关闭”才会把草稿写入后端。",
      details: [
        "放弃后，本次弹窗内的修改会恢复为打开前的配置。",
        "运行中的采集、推理和控制主链不会因此改变。"
      ],
      confirmLabel: "放弃修改",
      cancelLabel: "继续编辑",
      danger: true,
      onConfirm: () => {
        const restored = structuredClone(baseline) as RuntimeConfig;
        configDraftRef.current = restored;
        setConfigDraft(restored);
        finishConfigDialog(dialog);
      }
    });
  }, [finishConfigDialog]);

  const confirmPendingAction = useCallback(async () => {
    const request = confirmationRequest;
    if (!request || confirmationBusy) return;
    setConfirmationError(null);
    setConfirmationBusy(true);
    try {
      if (request.handoffOnConfirm) {
        setConfirmationRequest(null);
      }
      const completed = await request.onConfirm();
      if (!request.handoffOnConfirm && completed !== false) {
        setConfirmationRequest(null);
      }
    } catch (error) {
      const message = getErrorMessage(error);
      setConfirmationError(message);
      setLocalError((current) => current ?? `当前操作未完成：${message}`);
    } finally {
      setConfirmationBusy(false);
    }
  }, [confirmationBusy, confirmationRequest]);

  useEffect(() => {
    setConfirmationError(null);
  }, [confirmationRequest]);

  const stageConfigDialogDraft = useCallback((next: RuntimeConfig) => {
    configDraftRef.current = next;
    setConfigDraft(next);
    setConfigDialogDirty(!runtimeConfigsEqual(configDialogBaselineRef.current, next));
    setDialogSaveError(null);
  }, []);

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
          void requestDismissConfigDialog("class-config");
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
  }, [classConfigDialogOpen, requestDismissConfigDialog]);

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
          void requestDismissConfigDialog("target-weights");
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
  }, [requestDismissConfigDialog, targetWeightsDialogOpen]);

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
  const statistics = runtime?.statistics;
  const config = configDraft ?? runtimeConfig;
  const captureConfig = nestedRecord(config, "capture");
  const configuredCaptureDevice = readString(captureConfig.device, "");
  const configuredCapturePixelFormat = readString(captureConfig.pixel_format, "");
  const configuredCaptureWidth = readNumber(captureConfig.width, 0);
  const configuredCaptureHeight = readNumber(captureConfig.height, 0);
  const configuredCaptureFps = readNumber(captureConfig.fps, 0);
  const configuredRoiLeft = readNumber(captureConfig.roi_left, 0);
  const configuredRoiTop = readNumber(captureConfig.roi_top, 0);
  const configuredRoiWidth = readNumber(captureConfig.roi_width, 0);
  const configuredRoiHeight = readNumber(captureConfig.roi_height, 0);
  const roiConfig = nestedRecord(config, "roi");
  const crosshairConfig = nestedRecord(config, "crosshair");
  const limitsConfig = nestedRecord(config, "limits");
  const inferenceConfig = nestedRecord(config, "inference");
  const preprocessConfig = nestedRecord(config, "preprocess");
  const controlConfig = nestedRecord(config, "control");
  const rustPipelineConfig = nestedRecord(config, "pipeline");
  const rustControlPlane = Object.prototype.hasOwnProperty.call(
    rustPipelineConfig,
    "projection_fov_x_deg"
  );
  const hardwareConfig = nestedRecord(config, "hardware");
  const consumersConfig = nestedRecord(config, "consumers");
  const vision = asRecord(runtime?.vision);
  const crosshairStatus = asRecord(vision.crosshair);
  const crosshairObservation = asRecord(crosshairStatus.observation);
  const crosshairTemplate = asRecord(crosshairStatus.template);
  const inferenceTrace = asRecord(vision.inference);
  const runtimeInference = asRecord(runtime?.inference);
  const pipeline = asRecord(runtime?.pipeline);
  const deepstreamStatus = asRecord(pipeline.deepstream);
  const configuredInferenceBackend = readString(inferenceConfig.backend, "deepstream_nvinfer");
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
  const captureStatusText = capture?.state === "failed" || capture?.state === "unavailable"
    ? "采集异常"
    : capture?.running === true
      ? "运行中"
      : mainlineLaunchPending || capture?.state === "starting"
        ? "启动中"
        : captureMainConfigured
          ? "待启动"
          : "未配置";
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
  const runtimeControlRequested = captureMainRunning;
  const runtimePostprocess = asRecord(runtimeInference.postprocess);
  const executorStatus = asRecord(runtime?.executor);
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
  const displayCaptureProfileSource = configuredCaptureProfile
    ? "已保存配置"
    : selectedProfile?.source === "configured"
      ? "当前生效配置"
      : selectedProfile?.source || NO_SAMPLE;
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
  const roiSize = rustControlPlane
    ? configuredRoiWidth > 0 && configuredRoiHeight > 0
      ? Math.min(configuredRoiWidth, configuredRoiHeight)
      : 640
    : readNumber(roiConfig.size, 640);
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
  const roiX = rustControlPlane
    ? configuredRoiLeft
    : sourceWidth > 0 ? Math.max(0, Math.floor((sourceWidth - roiSize) / 2)) : 0;
  const roiY = rustControlPlane
    ? configuredRoiTop
    : sourceHeight > 0 ? Math.max(0, Math.floor((sourceHeight - roiSize) / 2)) : 0;
  const confidence = readNumber(inferenceConfig.confidence_threshold, 0.25);
  const nms = readNumber(inferenceConfig.nms_threshold, 0.45);
  const detectionProfiles = recordList(inferenceConfig.detection_class_profiles);
  const activeDetectionProfile = readString(inferenceConfig.detection_class_profile, "default");
  const activeDetectionClass = readString(
    rustControlPlane
      ? rustPipelineConfig.target_class_filter
      : inferenceConfig.detection_class_filter,
    "all"
  );
  const detectionProfileNames = Object.keys(detectionProfiles);
  const detectionClasses = detectionProfiles[activeDetectionProfile] ?? detectionProfiles.default ?? [];
  const detectionClassPriority = readString(
    rustControlPlane
      ? rustPipelineConfig.target_class_priority
      : inferenceConfig.detection_class_priority,
    "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  );
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
      setLocalError(`主链启动未确认：${runtimeMainlineStatus.readinessDetail || runtimeInferenceDetail || runtimeInferenceReason || "后端运行态未进入运行状态。"}`);
    }
  }, [
    runtimeMainlineSelected,
    mainlineTerminalError,
    mainlineLaunchAccepted,
    runtimeMainlineStatus.failed,
    runtimeMainlineStatus.failureMessage,
    runtimeMainlineStatus.readinessDetail,
    runtime?.fatal_error,
    runtimeInferenceDetail,
    runtimeInferenceReason,
    runtimeMainlineRunning
  ]);
  const aimConfig = nestedRecord(controlConfig, "aim");
  const rawAimRoleRatios = nestedRecord(aimConfig, "role_y_ratios");
  const aimRoleRatios: AimRoleRatios = {
    head: clampNumber(readNumber(rawAimRoleRatios.head, 0.22), 0, 1),
    body: clampNumber(readNumber(rawAimRoleRatios.body, 0.22), 0, 1),
    other: clampNumber(
      readNumber(
        rustControlPlane
          ? rustPipelineConfig.target_aim_y_ratio
          : rawAimRoleRatios.other,
        0.22
      ),
      0,
      1
    )
  };
  const classRoleProfiles = profileRoleRecords(aimConfig.class_roles);
  const activeClassRoles = classRoleProfiles[activeDetectionProfile] ?? {};
  const targetFovRadiusPx = readNumber(rustPipelineConfig.target_fov_radius_px, 180);
  const candidateRatioMaxAspect = readNumber(rustPipelineConfig.candidate_max_aspect_ratio, 6);
  const normalizedSelectionClassWeight = clampNumber(
    readNumber(rustPipelineConfig.target_selection_class_ratio, 0.35),
    0,
    1
  );
  const normalizedSelectionDistanceWeight = 1 - normalizedSelectionClassWeight;
  const trackerMaxMatchDistance = readNumber(rustPipelineConfig.tracker_max_match_distance, 1.5);
  const trackerPositionCostWeight = readNumber(rustPipelineConfig.tracker_position_cost_weight, 0.75);
  const trackerIouCostWeight = readNumber(rustPipelineConfig.tracker_iou_cost_weight, 0.25);
  const targetTrackMaxAge = readNumber(rustPipelineConfig.target_track_max_age, 2);
  const targetLostGraceMs = readNumber(rustPipelineConfig.target_track_max_lost_age_ms, 120);
  const targetSwitchPreferenceAdvantage = readNumber(rustPipelineConfig.target_switch_min_preference_advantage, 0.08);
  const targetSwitchContinuityScore = readNumber(rustPipelineConfig.target_switch_min_continuity_score, 0.7);
  const targetSwitchDelayMs = readNumber(rustPipelineConfig.target_switch_delay_ms, 50);
  const freshnessThresholdMs = readNumber(rustPipelineConfig.freshness_threshold_ms, 55);
  const dualPhaseFovX = readNumber(rustPipelineConfig.projection_fov_x_deg, 105);
  const dualPhaseCountsPer360 = readNumber(rustPipelineConfig.projection_counts_per_360, 9980);
  const dualPhaseNearThreshold = readNumber(rustPipelineConfig.near_threshold_px, 12);
  const dualPhaseFarKp = readNumber(rustPipelineConfig.far_kp, 0.22);
  const dualPhaseNearKp = readNumber(rustPipelineConfig.near_kp, 0.20);
  const dualPhaseAtanScale = readNumber(rustPipelineConfig.atan_scale_counts, 256);
  const dualPhaseFarMaxCounts = readNumber(rustPipelineConfig.far_max_counts_per_update, 127);
  const dualPhaseNearMaxCounts = readNumber(rustPipelineConfig.near_max_counts_per_update, 72);
  const dualPhaseArrivalRadiusCounts = readNumber(rustPipelineConfig.arrival_radius_counts, 3);
  const dualPhasePredictionEnabled = readBoolean(rustPipelineConfig.prediction_enabled, false);
  const dualPhasePredictionSmoothingFrames = readNumber(rustPipelineConfig.velocity_smoothing_frames, 3);
  const dualPhasePredictionHistoryResetGapMs = readNumber(rustPipelineConfig.velocity_history_reset_gap_ms, 80);
  const velocitySpreadBasePxMs = readNumber(rustPipelineConfig.velocity_spread_base_px_ms, 0.12);
  const velocitySpreadRelative = readNumber(rustPipelineConfig.velocity_spread_relative, 0.50);
  const velocityChangeBasePxMs = readNumber(rustPipelineConfig.velocity_change_base_px_ms, 0.20);
  const velocityChangeRelative = readNumber(rustPipelineConfig.velocity_change_relative, 0.75);
  const dualPhasePredictionLeadFrames = readNumber(rustPipelineConfig.prediction_lead_frames, 1);
  const dualPhasePredictionFarCapPx = readNumber(rustPipelineConfig.prediction_far_absolute_cap_px, 10);
  const dualPhasePredictionFarBaseCapPx = readNumber(rustPipelineConfig.prediction_far_base_cap_px, 1.25);
  const dualPhasePredictionFarRelativeCap = readNumber(rustPipelineConfig.prediction_far_relative_cap, 0.30);
  const dualPhasePredictionNearCapPx = readNumber(rustPipelineConfig.prediction_near_absolute_cap_px, 3);
  const dualPhasePredictionNearBaseCapPx = readNumber(rustPipelineConfig.prediction_near_base_cap_px, 0.75);
  const dualPhasePredictionNearRelativeCap = readNumber(rustPipelineConfig.prediction_near_relative_cap, 0.20);
  const residualCap = readNumber(rustPipelineConfig.residual_cap, 1);
  const actuationFeedbackDelayMs = readNumber(rustPipelineConfig.actuation_feedback_delay_ms, 4);
  const targetMinConfidence = readNumber(rustPipelineConfig.target_min_confidence, 0.5);
  const trackerScaleCostWeight = readNumber(rustPipelineConfig.tracker_scale_cost_weight, 0.15);
  const trackerMaxSizeRatio = readNumber(rustPipelineConfig.tracker_max_size_ratio, 2.5);
  const trackerMaxAssociationDtMs = readNumber(rustPipelineConfig.tracker_max_association_dt_ms, 150);
  const recoilConfig = (controlConfig.recoil ?? {}) as Record<string, unknown>;
  const recoilEnabled = readBoolean(recoilConfig.enabled, false);
  const recoilRequireTarget = readBoolean(recoilConfig.require_target, true);
  const recoilStartupRampMs = readNumber(recoilConfig.startup_ms, 35);
  const recoilBaseRate = readNumber(recoilConfig.base_rate_counts_s, 0);
  const triggerMode = readString(controlConfig.trigger_mode, "always");
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);
  const kmnetAutoConnect = readBoolean(hardwareConfig.auto_connect, true);
  const outputEnabled = readBoolean(controlConfig.output_enabled, true);
  const schedulerIntervalMs = readNumber(rustPipelineConfig.output_interval_ms, 4);
  const controlModeLabel = CONTROL_ALGORITHM_LABEL;

  useEffect(() => {
    runtimeConfigLatestRef.current = runtimeConfig;
    if (!runtimeConfig || pendingConfigWritesRef.current > 0 || activeConfigDialogRef.current !== null) {
      return;
    }
    const next = normalizeRuntimeConfig(runtimeConfig);
    configDraftRef.current = next;
    setConfigDraft(next);
  }, [runtimeConfig]);
  const kmnetConnected = kmnetStatus.connected === true;
  const kmnetRuntimeConnected = kmnetStatus.runtime_connected === true;
  const kmnetConnecting = kmnetStatus.connecting === true;
  const kmnetExecutorAvailable = kmnetStatus.available === true;
  const kmnetConnectionState = readString(
    kmnetStatus.connection_state,
    kmnetConnected ? "connected" : kmnetConnecting ? "connecting" : "disconnected"
  );
  const kmnetConnectionFailed = kmnetConnectionState === "failed";
  const kmnetConnectionDegraded = kmnetConnectionState === "degraded";
  const kmnetRetryable = kmnetStatus.retryable === true;
  const kmnetLastError = readString(kmnetStatus.last_error, "");
  const kmnetLastDeviceError = readString(kmnetStatus.last_device_error, "");
  const desiredConfigRevision = readNumber(runtime?.config?.version, 0);
  const effectiveConfigRevision = readNumber(runtime?.config?.effective_version, desiredConfigRevision);
  const configRestartRequired = runtime?.config?.restart_required === true
    || desiredConfigRevision !== effectiveConfigRevision;
  const kmnetRestartRequired = kmnetStatus.restart_required === true;
  const kmnetConfigurationState = readString(
    kmnetStatus.configuration_state,
    kmnetRestartRequired ? "restart_required" : kmnetAutoConnect ? "ready" : "uncommissioned"
  );
  const kmnetConfigurationReady = kmnetStatus.configuration_ready === true
    || (kmnetConfigurationState === "ready" && !kmnetRestartRequired);
  const kmnetCanConnect = kmnetStatus.can_connect === true
    || (kmnetConfigurationReady
      && runtime?.running === true
      && !kmnetRuntimeConnected
      && !kmnetConnecting);
  const kmnetCanDisconnect = kmnetStatus.can_disconnect === true
    || (runtime?.running === true && kmnetRuntimeConnected);
  const kmnetBlockedReason = readString(kmnetStatus.blocked_reason, "");
  const kmnetConnectionLabel = kmnetRestartRequired
    ? "配置等待重启"
    : kmnetConnected
    ? kmnetConnectionDegraded ? "已连接，监听异常" : "已连接"
    : kmnetConnecting
      ? "连接中"
      : kmnetConnectionFailed
        ? "连接失败"
        : "未连接";
  const kmnetRuntimeConnectionLabel = runtime?.running === true
    ? kmnetRestartRequired
      ? kmnetRuntimeConnected ? "旧配置仍连接" : "等待重启"
      : kmnetRuntimeConnected ? "已连接" : "未连接"
    : "主链未运行";
  const kmnetDiagnosticDisabled = kmnetRestartRequired
    || !kmnetExecutorAvailable
    || runtime?.running === true
    || busy === "kmnet.diagnostic";
  const kmnetButtonLeft = kmnetStatus.button_left === true;
  const kmnetButtonRight = kmnetStatus.button_right === true;
  const previewEnabled = consumersConfig.preview !== false;
  const activeModelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const artifact = runtime?.active_model?.artifact;
  const version = runtime?.active_model?.version;
  const registeredInputShape = version?.input_shape === "engine-probe-required"
    ? "等待 TensorRT engine 探测"
    : version?.input_shape ?? "";
  const displayedInputShape = registeredInputShape;
  const runtimePostprocessConfidence = readNumber(runtimePostprocess.confidence_threshold, Number.NaN);
  const runtimePostprocessNms = readNumber(runtimePostprocess.nms_threshold, Number.NaN);
  const runtimeRoiLeft = readNumber(inferenceTrace.roi_offset_x, Number.NaN);
  const runtimeRoiTop = readNumber(inferenceTrace.roi_offset_y, Number.NaN);
  const runtimeRoiWidth = readNumber(inferenceTrace.roi_width, Number.NaN);
  const runtimeRoiHeight = readNumber(inferenceTrace.roi_height, Number.NaN);
  const runtimeRoiAvailable = runtimeMainlineRunning
    && Number.isFinite(runtimeRoiLeft)
    && Number.isFinite(runtimeRoiTop)
    && Number.isFinite(runtimeRoiWidth)
    && Number.isFinite(runtimeRoiHeight);
  const roiSettingsApplied = runtimeRoiAvailable
    && runtimeRoiLeft === roiX
    && runtimeRoiTop === roiY
    && runtimeRoiWidth === (rustControlPlane ? configuredRoiWidth : roiSize)
    && runtimeRoiHeight === (rustControlPlane ? configuredRoiHeight : roiSize);
  const roiApplyLabel = !runtimeRoiAvailable
    ? "等待真实帧"
    : roiSettingsApplied
      ? "已生效"
      : configRestartRequired
        ? "已保存 · 等待重启"
        : "运行区域不同";
  const runtimePostprocessAvailable = runtimeMainlineRunning
    && runtimeInference.loaded === true
    && Number.isFinite(runtimePostprocessConfidence)
    && Number.isFinite(runtimePostprocessNms);
  const postprocessSettingsApplied = runtimePostprocessAvailable
    && Math.abs(runtimePostprocessConfidence - confidence) < 0.0001
    && Math.abs(runtimePostprocessNms - nms) < 0.0001;
  const postprocessApplyLabel = !runtimePostprocessAvailable
    ? "主链未装载"
    : postprocessSettingsApplied
      ? "已生效"
      : configRestartRequired
        ? "已保存 · 等待重启"
        : "运行值不同";
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
  const detectionCount = readNullableNumber(vision.detections);
  const target = asRecord(vision.target);
  const activeRuntimeClassId = readNullableNumber(target.cls ?? target.class_id);
  const activeRuntimeClassLabel = activeRuntimeClassId !== null
    ? `cls ${Math.round(activeRuntimeClassId)}`
    : "";
  const runtimeDetectionItems = recordArray(vision.detection_items);
  const runtimeDetectionClassIds = runtimeDetectionItems.flatMap((item) => {
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
  const rawCandidateCount = readNullableNumber(
    targetPipelineCounts.raw_candidates ?? targetPipelineCounts.decode_raw_candidates
  );
  const eligibleCandidateCount = readNullableNumber(
    targetPipelineCounts.eligible_candidates ?? targetPipelineCounts.inside_fov
  );
  const selectedTargetCount = readNullableNumber(targetPipelineCounts.selected_targets);
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
  const controlPipeline = asRecord(asRecord(vision.control).pipeline);
  const mouseObservation = asRecord(control.mouse_observation);
  const controlHasSample = readString(control.global_state, "IDLE") !== "IDLE";
  const controlHasTarget = Object.keys(target).length > 0;
  const controlCandidateCount = readNullableNumber(control.candidates);
  const controlTrackId = readNullableNumber(target.track_id);
  const controlWidthPx = readNullableNumber(mouseObservation.control_width_px);
  const controlHeightPx = readNullableNumber(mouseObservation.control_height_px);
  const controlCenterX = controlWidthPx === null ? null : controlWidthPx * 0.5;
  const controlCenterY = controlHeightPx === null ? null : controlHeightPx * 0.5;
  const predictedAimX = readNullableNumber(mouseObservation.predicted_x_px ?? control.aim_x);
  const predictedAimY = readNullableNumber(mouseObservation.predicted_y_px ?? control.aim_y);
  const observedAimX = readNullableNumber(mouseObservation.observed_x_px);
  const observedAimY = readNullableNumber(mouseObservation.observed_y_px);
  const predictedErrorXPx =
    predictedAimX !== null && controlCenterX !== null ? predictedAimX - controlCenterX : null;
  const predictedErrorYPx =
    predictedAimY !== null && controlCenterY !== null ? predictedAimY - controlCenterY : null;
  const predictedErrorDistancePx =
    predictedErrorXPx !== null && predictedErrorYPx !== null
      ? Math.hypot(predictedErrorXPx, predictedErrorYPx)
      : null;
  const controlMeasurementDtS = readNullableNumber(controlPipeline.measurement_dt_s ?? mouseObservation.measurement_dt_s);
  const controlFrameAgeMs = readNullableNumber(controlPipeline.frame_age_ms);
  const acceptedCommandCount = readNullableNumber(kmnetStatus.accepted_command_count);
  const hasAcceptedCommand = (acceptedCommandCount ?? 0) > 0;
  const lastAcceptedCommand = formatPoint(
    kmnetStatus.last_accepted_dx,
    kmnetStatus.last_accepted_dy,
    0,
    "counts"
  );
  const controlNoSendReason = !controlHasTarget
      ? targetPipelineMessage || readString(control.selection_reason, "无目标")
      : control.will_emit !== true
        ? readString(control.no_send_reason, readString(control.reason, "控制门控未通过"))
        : !kmnetRuntimeConnected
          ? "主链设备通道未连接"
          : "命令已获准进入设备通道";
  const deepstreamInputFrames = runtimeMainlineStatus.nvinferInputFrames;
  const deepstreamOutputBuffers = readNullableNumber(runtimeInference.output_buffers);
  const deepstreamMetadataExtractions = runtimeMainlineStatus.metadataExtractions;
  const deepstreamPublishedBatches = runtimeMainlineStatus.publishedBatches;
  const deepstreamInferenceCompleted =
    (deepstreamOutputBuffers ?? 0) > 0 ||
    (deepstreamPublishedBatches ?? 0) > 0;
  const inferenceRan = deepstreamNvinferSelected && deepstreamInferenceCompleted;
  const inferenceReason = readString(
    runtimeInference.inference_reason,
    readString(runtimeInference.reason, "")
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
      const preview = await setCapturePreviewEnabled(enabled, {
        keepalive: options?.keepalive
      });
      if (runtime) {
        onRuntimeStateChange({
          ...runtime,
          inference: {
            ...runtime.inference,
            preview_enabled: preview.preview_enabled,
            preview_active: preview.preview_active,
            preview_encoder_active: preview.preview_encoder_active,
            preview_consumers: preview.preview_consumers,
            preview_available: preview.preview_available,
            preview_sequence: preview.preview_sequence,
            preview_reason: preview.preview_reason,
            preview_transport: preview.preview_transport
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

  const nvinferInputFps = readNullableNumber(statistics?.nvinfer_input_fps);
  const nvinferOutputFps = readNullableNumber(statistics?.nvinfer_output_fps);
  const detectionBatchFps = readNullableNumber(statistics?.detection_batch_fps);
  const targetingBatchFps = readNullableNumber(statistics?.targeting_batch_fps);
  const detectionDataAgeMs = readNullableNumber(statistics?.detection_data_age_ms);
  const detectionFreshnessThresholdMs = readNullableNumber(
    statistics?.detection_freshness_threshold_ms
  );
  const detectionFreshness = freshnessSummary(
    detectionDataAgeMs,
    detectionFreshnessThresholdMs
  );
  const captureReason = capture?.last_error || (
    capture?.running !== true
      ? "采集尚未启动"
      : nvinferInputFps !== null && nvinferInputFps > 0
        ? "nvinfer 正在接收有效输入"
        : (statistics?.nvinfer_input_counter ?? 0) > 0
          ? "已收到输入帧，正在建立速率窗口"
          : "采集已启动，等待第一帧"
  );
  const roiInputWidth = readNumber(inferenceTrace.input_width, 0);
  const roiInputHeight = readNumber(inferenceTrace.input_height, 0);
  const modelInputWidth = readNumber(inferenceTrace.model_input_width, 0);
  const modelInputHeight = readNumber(inferenceTrace.model_input_height, 0);
  const inputDownscaleFactor =
    roiInputWidth > 0 && roiInputHeight > 0 && modelInputWidth > 0 && modelInputHeight > 0
      ? Math.max(roiInputWidth / modelInputWidth, roiInputHeight / modelInputHeight)
      : null;
  const inputAreaRatio =
    roiInputWidth > 0 && roiInputHeight > 0 && modelInputWidth > 0 && modelInputHeight > 0
      ? (modelInputWidth * modelInputHeight) / (roiInputWidth * roiInputHeight)
      : null;
  const inputDensityWarning = inputDownscaleFactor !== null && inputDownscaleFactor > 1;
  const sampledDetectionGeneration = readNullableNumber(
    inferenceTrace.generation ?? runtimeInference.sampled_detection_generation
  );
  const inferenceTotalMs = readNullableNumber(statistics?.inference_latency_ms);
  const inferenceHighestConfidence = runtimeDetectionItems.reduce<number | null>(
    (highest, item) => {
      const score = readNullableNumber(item.score);
      return score === null ? highest : highest === null ? score : Math.max(highest, score);
    },
    null
  );
  const inferenceBatchPublished = (deepstreamPublishedBatches ?? 0) > 0;
  const inferenceBatchState = !inferenceRan
    ? NO_SAMPLE
    : !inferenceBatchPublished
      ? "尚未发布"
      : detectionDataAgeMs === null || detectionFreshnessThresholdMs === null
        ? "等待帧龄样本"
        : detectionDataAgeMs <= detectionFreshnessThresholdMs
          ? "新鲜 · 控制可用"
          : "已超过控制阈值";
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
    return items.filter((item, index, all) => (
      all.findIndex((candidate) => candidate.title === item.title && candidate.detail === item.detail) === index
    ));
  }, [capture?.last_error, errorNotices, errors, localError]);

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
  }, []);

  useEffect(() => {
    if (activePage !== "infer" || !modelManagerDialogOpen) {
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
  }, [activePage, applyModelCatalogResult, modelCatalogRefreshKey, modelManagerDialogOpen, onRefresh]);

  useEffect(() => {
    if (!modelCatalog || typeof artifact?.id !== "number") {
      return;
    }
    const activePath = findCatalogModelPath(modelCatalog, artifact.id);
    if (!activePath) return;
    setSelectedModelCatalogPath((current) => current ?? activePath);
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
    if (activePage !== "infer" || !modelManagerDialogOpen) {
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
  }, [activePage, modelDetailsRefreshKey, modelManagerDialogOpen, runtime?.active_model?.version?.id, selectedModelProjectId]);

  useEffect(() => {
    if (activePage !== "infer" || !modelManagerDialogOpen) {
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
  }, [activePage, modelDetailsRefreshKey, modelManagerDialogOpen, selectedModelVersionId]);

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
    return (
      readString(state.last_error, "") ||
      "采集配置未进入可用状态。"
    );
  }, []);

  const assertCaptureLaunchState = useCallback((state: CaptureState) => {
    if (state.available !== true) {
      throw new Error(`采集阶段失败：${captureLaunchFailureMessage(state)}`);
    }
  }, [captureLaunchFailureMessage]);

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
        throw new Error(`${stageTitle}失败：${status.readinessDetail || missingMessage}`);
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
      if (action) {
        await action();
      }
      ensureNotCancelled();
      setLaunchCompletedStages(index + 1);
    };

    try {
      await runStage(0, async () => {
        const state = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
        const status = getRuntimeMainlineStatus(state);
        if (status.failed) {
          throw new Error(status.readinessDetail);
        }
      });
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
        setMainlineLaunchAccepted(true);
        setMainlineLaunchMessage("后端已确认主链运行，正在核对运行时消费数据。");
      });
      await runStage(3, async () => {
        const state = await waitForRuntimeEvidence(
          "激活鼠标算法",
          (state) => getRuntimeMainlineStatus(state).hasRuntimeConsumption,
          "runtime 尚未消费 DetectionBatch，目标选择、跟踪与 Atan 鼠标算法没有输入。"
        );
        const status = getRuntimeMainlineStatus(state);
        setLaunchProgressDetail(`${status.readinessLabel}：${status.readinessDetail}`);
      });
      setLaunchStatus("success");
      setLaunchCompletedStages(MAINLINE_LAUNCH_STAGES_CUSTOM_TENSORRT.length);

      setLaunchProgressDetail((detail) => detail || "主链启动完成，后端运行态已确认。");
      // Capture selection may persist a newer configuration revision. Refresh
      // after successful launch so subsequent editors start from that exact
      // canonical revision instead of the pre-launch snapshot.
      await onRefresh();
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
    launchStatus,
    onRefresh,
    onRuntimeStateChange,
    showLaunchToast,
    waitForLaunchFeedback,
    waitForRuntimeEvidence
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
    setLaunchProgressDetail("正在请求后端立即停止输出并中止启动，不再等待普通生命周期锁。");
    setMainlineLaunchAccepted(false);
    setMainlineLaunchMessage("");
    setBusy(null);
    try {
      await emergencyStopRuntimePipeline();
      const stoppedState = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
      onRuntimeStateChange(stoppedState);
      setLaunchProgressDetail("后端已确认紧急停止；旧输出已失效，启动流程已取消。");
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
    async (
      section: string,
      key: string,
      value: RuntimeConfigValue,
      options?: { optimistic?: boolean; rethrow?: boolean }
    ): Promise<void> => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const next = base ? normalizeRuntimeConfig(base) : null;
      if (!next) {
        return;
      }
      const sectionValue = {
        ...asRecord(next[section])
      };
      sectionValue[key] = value;
      next[section] = sectionValue as RuntimeConfig[string];
      if (activeConfigDialogRef.current !== null) {
        stageConfigDialogDraft(next);
        return;
      }
      const optimistic = options?.optimistic !== false;
      const writeSeq = ++configWriteSeqRef.current;
      pendingConfigWritesRef.current += 1;
      setDialogSaveError(null);
      const keepsEditorInteractive = section === "control" && key === "aim";
      if (!keepsEditorInteractive) {
        setBusy(`${section}.${key}`);
      }
      setLocalError(null);
      if (optimistic) {
        configDraftRef.current = next;
        setConfigDraft(next);
      }
      try {
        const request = configWriteQueueRef.current.then(() =>
          updateRuntimeConfigField(section, key, value)
        );
        configWriteQueueRef.current = request.then(
          () => undefined,
          () => undefined
        );
        const result = await request;
        const applied = normalizeRuntimeConfig(result.config);
        // Always advance the canonical persisted revision. A newer optimistic
        // edit may still own the visible draft, but the next queued transaction
        // must never be built from an older revision.
        runtimeConfigLatestRef.current = applied;
        if (writeSeq === configWriteSeqRef.current) {
          configDraftRef.current = applied;
          setConfigDraft(applied);
          onRuntimeConfigChange(applied);
        }
      } catch (err) {
        const message = getErrorMessage(err);
        setLocalError(`配置同步失败：${message}`);
        setDialogSaveError(message);

        reportError(err, { source: 'studio', title: '操作失败' });
        if (optimistic && writeSeq === configWriteSeqRef.current) {
          configDraftRef.current = null;
          setConfigDraft(null);
        }
        if (options?.rethrow) {
          throw err;
        }
      } finally {
        pendingConfigWritesRef.current = Math.max(0, pendingConfigWritesRef.current - 1);
        if (!keepsEditorInteractive && writeSeq === configWriteSeqRef.current) {
          setBusy(null);
        }
      }
    },
    [onRuntimeConfigChange, runtimeConfig, stageConfigDialogDraft]
  );

  const requestOutputGateChange = useCallback((enabled: boolean) => {
    if (!enabled) {
      return updateConfigField(
        "control",
        "output_enabled",
        false,
        { optimistic: false, rethrow: true }
      );
      return;
    }
    setConfirmationRequest({
      eyebrow: "物理输出",
      title: "允许发送鼠标偏移？",
      description: "开启后，Rust 控制链产生的新鲜控制量可以通过当前 kmNet 会话发送到物理设备。",
      details: [
        `设备：${kmnetHost || "未填写"}:${kmnetPort || "未填写"} · ${kmnetRuntimeConnected ? "当前已连接" : "当前未连接"}`,
        "旧命令不会补发；暂停输出或断开 kmNet 会立即清空待发送命令。"
      ],
      confirmLabel: "确认开启输出",
      danger: true,
      onConfirm: () => updateConfigField(
        "control",
        "output_enabled",
        true,
        { optimistic: false, rethrow: true }
      )
    });
    return false;
  }, [kmnetHost, kmnetPort, kmnetRuntimeConnected, updateConfigField]);

  const updateConfigSection = useCallback(
    async (
      section: string,
      values: Record<string, RuntimeConfigValue>
    ) => {
      const persistSection = async (source: RuntimeConfig) => {
        const current = cloneRuntimeConfig(source);
        if (!current) {
          throw new Error("尚未读取运行配置。");
        }
        const currentSection = asRecord(current[section]);
        const nextSection = {
          ...currentSection,
          ...values
        };
        if (runtimeConfigValuesEqual(currentSection, nextSection)) {
          return null;
        }
        current[section] = nextSection as RuntimeConfig[string];
        return updateRuntimeConfig(current);
      };
      const request = configWriteQueueRef.current.then(async () => {
        const current = cloneRuntimeConfig(runtimeConfigLatestRef.current);
        if (!current) {
          throw new Error("尚未读取运行配置。");
        }
        try {
          return await persistSection(current);
        } catch (error) {
          if (getApiErrorCode(error) !== "CONFIG_REVISION_CONFLICT") {
            throw error;
          }
          // Capture selection and other backend-owned transactions may have
          // advanced the revision after this view was rendered. Re-read the
          // canonical document and replay only this section's intended keys;
          // never resend an entire stale configuration.
          const latest = normalizeRuntimeConfig(await getRuntimeConfig());
          runtimeConfigLatestRef.current = latest;
          configDraftRef.current = latest;
          setConfigDraft(latest);
          onRuntimeConfigChange(latest);
          return persistSection(latest);
        }
      });
      configWriteQueueRef.current = request.then(
        () => undefined,
        () => undefined
      );
      const result = await request;
      if (!result) {
        return null;
      }
      const applied = normalizeRuntimeConfig(result.config);
      runtimeConfigLatestRef.current = applied;
      configDraftRef.current = applied;
      setConfigDraft(applied);
      onRuntimeConfigChange(applied);
      return result;
    },
    [onRuntimeConfigChange]
  );

  const handleCenteredRoiSizeChange = useCallback(
    async (requestedSize: number) => {
      const size = nearestRoiSize(requestedSize);
      if (!rustControlPlane) {
        await updateConfigField("roi", "size", size);
        return;
      }
      const width = configuredCaptureWidth > 0 ? configuredCaptureWidth : sourceWidth;
      const height = configuredCaptureHeight > 0 ? configuredCaptureHeight : sourceHeight;
      if (width <= 0 || height <= 0) {
        setLocalError("请先选择采集分辨率，再修改 ROI。");
        return;
      }
      if (size > width || size > height) {
        setLocalError(`ROI ${size}x${size} 超出当前采集画面 ${width}x${height}。`);
        return;
      }
      setLocalError(null);
      try {
        await updateConfigSection("capture", {
          roi_left: Math.floor((width - size) / 2),
          roi_top: Math.floor((height - size) / 2),
          roi_width: size,
          roi_height: size
        });
      } catch (error) {
        const message = getErrorMessage(error);
        setLocalError(`ROI 配置同步失败：${message}`);
        reportError(error, { source: "capture-roi", title: "ROI 配置未保存" });
      }
    },
    [
      configuredCaptureHeight,
      configuredCaptureWidth,
      rustControlPlane,
      sourceHeight,
      sourceWidth,
      updateConfigField,
      updateConfigSection
    ]
  );

  const updateDetectionClassFilter = useCallback(
    async (value: string) => {
      if (rustControlPlane) {
        await updateConfigField("pipeline", "target_class_filter", value);
      }
      await updateConfigField("inference", "detection_class_filter", value);
    },
    [rustControlPlane, updateConfigField]
  );

  const updateControlGroupField = useCallback(
    async (group: "aim" | "recoil", key: string, value: RuntimeConfigValue) => {
      const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
      const control = nestedRecord(base, "control");
      const groupValue = {
        ...nestedRecord(control, group),
        [key]: value
      };
      await updateConfigField("control", group, groupValue as RuntimeConfigValue);
    },
    [runtimeConfig, updateConfigField]
  );

  const updatePipelineField = useCallback(
    async (key: TargetingPipelineField, value: RuntimeConfigValue) => {
      await updateConfigField("pipeline", key, value);
    },
    [updateConfigField]
  );

  const updateDualPhaseField = useCallback(
    async (key: DualPhasePipelineField, value: RuntimeConfigValue) => {
      await updateConfigField("pipeline", key, value);
    },
    [updateConfigField]
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

  const performClearCrosshair = useCallback(async () => {
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

  const requestClearCrosshair = useCallback(() => {
    setConfirmationRequest({
      eyebrow: "准星学习",
      title: "清除当前准星模板？",
      description: "清除后，控制基准会立即回退到 ROI 几何中心；以后需要重新采样并学习模板。",
      details: ["采集、推理与目标跟踪不会停止。", "已学习的模板内容无法从后端恢复。"],
      confirmLabel: "确认清除模板",
      danger: true,
      onConfirm: performClearCrosshair
    });
  }, [performClearCrosshair]);

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
      const nextRatios = {
        ...aimRoleRatios,
        [role]: clampNumber(Number(ratio.toFixed(2)), 0, 1)
      };
      if (rustControlPlane) {
        const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
        const next = base ? normalizeRuntimeConfig(base) : null;
        if (!next) {
          return;
        }
        const control = asRecord(next.control);
        const aim = nestedRecord(control, "aim");
        next.control = {
          ...control,
          aim: { ...aim, role_y_ratios: nextRatios }
        } as RuntimeConfig[string];
        next.pipeline = {
          ...asRecord(next.pipeline),
          target_aim_y_ratio: nextRatios.other,
          target_class_aim_y_ratios: serializeRustClassAimRatios(
            activeClassRoles,
            nextRatios
          )
        } as RuntimeConfig[string];
        stageConfigDialogDraft(next);
        return;
      }
      await updateControlGroupField(
        "aim",
        "role_y_ratios",
        nextRatios as RuntimeConfigValue
      );
    },
    [
      activeClassRoles,
      aimRoleRatios,
      runtimeConfig,
      rustControlPlane,
      stageConfigDialogDraft,
      updateControlGroupField
    ]
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
      if (rustControlPlane) {
        const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
        const next = base ? normalizeRuntimeConfig(base) : null;
        if (!next) {
          return;
        }
        const control = asRecord(next.control);
        const aim = nestedRecord(control, "aim");
        next.control = {
          ...control,
          aim: { ...aim, class_roles: nextProfiles }
        } as RuntimeConfig[string];
        next.pipeline = {
          ...asRecord(next.pipeline),
          target_class_aim_y_ratios: serializeRustClassAimRatios(
            nextProfiles[activeDetectionProfile] ?? {},
            aimRoleRatios
          )
        } as RuntimeConfig[string];
        stageConfigDialogDraft(next);
        return;
      }
      await updateControlGroupField("aim", "class_roles", nextProfiles as RuntimeConfigValue);
    },
    [
      activeClassRoles,
      activeDetectionProfile,
      aimRoleRatios,
      classRoleProfiles,
      runtimeConfig,
      rustControlPlane,
      stageConfigDialogDraft,
      updateControlGroupField
    ]
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
      if (rustControlPlane) {
        const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
        const next = base ? normalizeRuntimeConfig(base) : null;
        if (!next) {
          return;
        }
        next.inference = {
          ...asRecord(next.inference),
          detection_class_priority: current.join(",")
        } as RuntimeConfig[string];
        next.pipeline = {
          ...asRecord(next.pipeline),
          target_class_priority: current.join(",")
        } as RuntimeConfig[string];
        stageConfigDialogDraft(next);
        return;
      }
      await updateConfigField("inference", "detection_class_priority", current.join(","));
    },
    [
      orderedClassEditorIds,
      runtimeConfig,
      rustControlPlane,
      stageConfigDialogDraft,
      updateConfigField
    ]
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
      await updateDetectionClassFilter(
        orderedSelection.length === classEditorIds.length
          ? "all"
          : orderedSelection.length === 0
            ? "none"
            : orderedSelection.join(",")
      );
    },
    [activeDetectionClass, classEditorIds, orderedClassEditorIds, updateDetectionClassFilter]
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
      if (rustControlPlane) {
        next.pipeline = {
          ...asRecord(next.pipeline),
          target_aim_y_ratio: aimRoleRatios.other,
          target_class_aim_y_ratios: serializeRustClassAimRatios(
            roleProfiles[nextActiveProfile] ?? {},
            aimRoleRatios
          )
        } as RuntimeConfig[string];
      }
      setLocalError(null);
      stageConfigDialogDraft(next);
    },
    [aimRoleRatios, runtimeConfig, rustControlPlane, stageConfigDialogDraft]
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
    setBusy("kmnet.defaults");
    setLocalError(null);
    try {
      const result = await updateConfigSection("hardware", KMNET_RECOMMENDED);
      setKmnetTestMessage(
        result?.restart_required
          ? "kmNet 参数已保存；请重启 novasightd，使新的物理设备适配器生效。"
          : "kmNet 参数已经是推荐值。"
      );
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 推荐参数应用失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      // An external writer may still win the optimistic revision check.
      // Re-read the canonical revision instead of leaving the editor on its
      // pre-transaction snapshot.
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [onRefresh, updateConfigSection]);

  const requestApplyKmNetRecommended = useCallback(() => {
    setConfirmationRequest({
      eyebrow: "kmNet 配置",
      title: "覆盖当前 kmNet 参数？",
      description: "推荐参数会替换当前地址、端口、UUID、监听端口和自动连接设置。保存后通常需要重启 novasightd。",
      details: [
        `当前：${kmnetHost || "未填写"}:${kmnetPort || "未填写"} · UUID ${kmnetUuid || "未填写"}`,
        `将改为：${KMNET_RECOMMENDED.host}:${KMNET_RECOMMENDED.port} · UUID ${KMNET_RECOMMENDED.uuid}`
      ],
      confirmLabel: "确认覆盖参数",
      danger: true,
      onConfirm: applyKmNetRecommended
    });
  }, [applyKmNetRecommended, kmnetHost, kmnetPort, kmnetUuid]);

  const setKmNetConnection = useCallback(async (connect: boolean) => {
    setBusy(connect ? "kmnet.connect" : "kmnet.disconnect");
    setLocalError(null);
    setKmnetTestMessage("");
    let physicalDisconnectCompleted = false;
    try {
      if (connect && !kmnetAutoConnect) {
        throw new Error("请先检查地址、端口和 UUID，并开启“主链启动时连接设备”；连接操作不会替你覆盖参数。");
      } else if (!connect) {
        // Physical stop comes first. Persisting the fail-closed output gate is
        // still attempted afterwards, but a storage error cannot keep an
        // already requested device session alive.
        await disconnectKmNet();
        physicalDisconnectCompleted = true;
        if (outputEnabled) {
          const gateResult = await updateRuntimeConfigField("control", "output_enabled", false);
          const applied = normalizeRuntimeConfig(gateResult.config);
          runtimeConfigLatestRef.current = applied;
          configDraftRef.current = applied;
          setConfigDraft(applied);
          onRuntimeConfigChange(applied);
        }
        setKmnetTestMessage(
          "kmNet 已断开；采集、推理与目标计算继续运行，物理偏移输出已关闭。"
        );
      } else {
        await connectKmNet();
        setKmnetTestMessage("kmNet 连接成功；若偏移输出已允许，新的实时命令现在可以发送。");
      }
      await onRefresh();
    } catch (err) {
      const action = connect ? "连接" : "断开";
      const detail = getErrorMessage(err);
      setLocalError(
        physicalDisconnectCompleted
          ? `kmNet 已断开，但未能持久化关闭输出：${detail}`
          : `kmNet ${action}失败：${detail}`
      );
      reportError(err, {
        source: "kmnet-lifecycle",
        title: physicalDisconnectCompleted ? "kmNet 已安全断开，配置保存失败" : `kmNet ${action}失败`
      });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetAutoConnect, onRefresh, onRuntimeConfigChange, outputEnabled]);

  const diagnosticMoveHardware = useCallback(async (
    dx = kmnetTestDx,
    dy = kmnetTestDy
  ) => {
    setBusy("kmnet.diagnostic");
    setLocalError(null);
    setKmnetTestMessage("");
    try {
      const result = await diagnosticMoveKmNet(
        Math.round(dx),
        Math.round(dy)
      );
      const status = asRecord(result.status);
      const metadata = asRecord(result.metadata);
      const stepsSent = readNumber(result.steps_sent, result.sent === true ? 1 : 0);
      const apiName = readString(metadata.api_name, "rust_pointer_device_send");
      setKmnetTestMessage(
        result.sent === true
          ? `已通过 ${apiName} 发送 dx=${Math.round(dx)} dy=${Math.round(dy)} · ${stepsSent}/1 步 · 诊断累计 ${readNumber(status.diagnostic_move_count, 0)} 次`
          : `未发送 raw：${readString(result.message, "未知原因")} · ${stepsSent}/1 步`
      );
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 诊断移动失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetTestDx, kmnetTestDy, onRefresh]);

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

  const requestConfigImportConfirmation = (
    fileName: string,
    imported: RuntimeConfig,
    baseline: RuntimeConfig,
    canonicalChanged = false
  ): void => {
    const changedSections = changedRuntimeConfigSections(baseline, imported);
    setConfirmationRequest({
      eyebrow: canonicalChanged ? "配置已在后台更新" : "导入运行配置",
      title: canonicalChanged ? "请按最新配置重新确认" : `应用 ${fileName}？`,
      description: canonicalChanged
        ? "你确认前，后端配置已被其他操作更新。NovaSight 没有覆盖新 revision，下面的差异已按最新配置重新计算。"
        : "导入会在确认时重新读取后端 canonical revision，再以该事务基线替换整份运行配置。",
      details: [
        `将修改：${changedSections.join("、") || "没有差异"}`,
        "保存成功后，后端会明确返回是否需要重启 novasightd。"
      ],
      confirmLabel: canonicalChanged ? "按最新配置导入" : "确认导入配置",
      danger: true,
      onConfirm: async () => {
        setBusy("import");
        const executeImport = async (): Promise<boolean> => {
          try {
            const canonical = normalizeRuntimeConfig(await getRuntimeConfig());
            if (!runtimeConfigValuesEqual(canonical.revision, baseline.revision)) {
              runtimeConfigLatestRef.current = canonical;
              configDraftRef.current = canonical;
              setConfigDraft(canonical);
              onRuntimeConfigChange(canonical);
              if (changedRuntimeConfigSections(canonical, imported).length === 0) {
                reportSuccess("无需导入配置", "后端最新配置已经与导入文件一致。", "config-import");
                return true;
              }
              requestConfigImportConfirmation(fileName, imported, canonical, true);
              return false;
            }
            const payload = normalizeRuntimeConfig(imported);
            payload.revision = canonical.revision;
            const result = await updateRuntimeConfig(payload);
            const applied = normalizeRuntimeConfig(result.config);
            runtimeConfigLatestRef.current = applied;
            configDraftRef.current = applied;
            setConfigDraft(applied);
            onRuntimeConfigChange(applied);
            reportSuccess(
              "配置导入成功",
              result.restart_required ? "配置已保存；重启 novasightd 后全部生效。" : "配置已经进入当前后端。",
              "config-import"
            );
            return true;
          } catch (error) {
            if (getApiErrorCode(error) === "CONFIG_REVISION_CONFLICT") {
              const canonical = normalizeRuntimeConfig(await getRuntimeConfig());
              runtimeConfigLatestRef.current = canonical;
              configDraftRef.current = canonical;
              setConfigDraft(canonical);
              onRuntimeConfigChange(canonical);
              if (changedRuntimeConfigSections(canonical, imported).length === 0) {
                reportSuccess("无需导入配置", "后端最新配置已经与导入文件一致。", "config-import");
                return true;
              }
              requestConfigImportConfirmation(fileName, imported, canonical, true);
              return false;
            }
            throw error;
          }
        };
        const request = configWriteQueueRef.current.then(executeImport);
        configWriteQueueRef.current = request.then(
          () => undefined,
          () => undefined
        );
        try {
          return await request;
        } finally {
          setBusy(null);
        }
      }
    });
  };

  const importConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setLocalError(null);
    try {
      const decoded = JSON.parse(await file.text()) as unknown;
      if (!decoded || typeof decoded !== "object" || Array.isArray(decoded)) {
        throw new Error("配置文件根节点必须是 JSON 对象。");
      }
      const current = cloneRuntimeConfig(runtimeConfigLatestRef.current);
      if (!current) {
        throw new Error("尚未读取后端当前配置，不能安全导入。");
      }
      const payload = normalizeRuntimeConfig(decoded as RuntimeConfig);
      const changedSections = changedRuntimeConfigSections(current, payload);
      if (changedSections.length === 0) {
        setLocalError("导入文件与当前配置一致，没有需要应用的修改。");
        return;
      }
      requestConfigImportConfirmation(file.name, payload, current);
    } catch (err) {
      setLocalError(`导入失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
    }
  };

  const selectModelFromCatalog = useCallback((model: ModelCatalogModel) => {
    setParserPreset("auto");
    setSelectedModelCatalogPath(model.relative_path);
    setLocalError(null);
  }, []);

  const closeModelManager = useCallback(() => {
    setModelManagerDialogOpen(false);
  }, []);

  const openModelManager = useCallback(() => {
    void onEnsureProjects(false).catch(() => undefined);
    const activePath = modelCatalog && typeof artifact?.id === "number"
      ? findCatalogModelPath(modelCatalog, artifact.id)
      : undefined;
    if (activePath) {
      setSelectedModelCatalogPath(activePath);
    }
    setModelManagerDialogOpen(true);
  }, [artifact?.id, modelCatalog, onEnsureProjects]);

  const ensureCatalogModelRegistration = async (model: ModelCatalogModel) => {
    if (typeof model.project_id === "number" && typeof model.artifact_id === "number") {
      return { projectId: model.project_id, artifactId: model.artifact_id };
    }
    const registered = await registerCatalogModel(model.relative_path);
    setSelectedModelProjectId(registered.project.id);
    setSelectedModelVersionId(registered.version.id);
    setSelectedModelArtifactId(registered.artifact.id);
    setModelCatalogMessage(`已登记模型引用：${model.relative_path}；Engine 文件保持原位。`);
    return { projectId: registered.project.id, artifactId: registered.artifact.id };
  };

  const saveModelMetadata = async (
    recommendation: ModelRecommendation,
    tags: string[]
  ) => {
    if (!selectedCatalogModel || selectedCatalogModel.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 模型后再整理标签。");
      return;
    }
    setBusy("model.metadata");
    setLocalError(null);
    try {
      const { artifactId } = await ensureCatalogModelRegistration(selectedCatalogModel);
      await updateModelArtifactMetadata(artifactId, recommendation, tags);
      const updatedCatalog = await getModelCatalog(false);
      applyModelCatalogResult(updatedCatalog);
      setModelCatalogMessage(`已保存 ${selectedCatalogModel.name} 的推荐状态与 ${tags.length} 个标签。`);
      setModelDetailsRefreshKey((current) => current + 1);
      await onRefresh();
    } catch (err) {
      setLocalError(`模型整理结果保存失败：${getErrorMessage(err)}`);
      reportError(err, { source: "model-metadata", title: "模型整理失败" });
    } finally {
      setBusy(null);
    }
  };

  const performModelSwitch = async () => {
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
      if (typeof selectedCatalogModel.project_id !== "number" || typeof selectedCatalogModel.artifact_id !== "number") {
        setModelSwitchProgressDetail("模型尚未登记，正在建立轻量文件引用；此步骤不会读取 Engine 内容。");
      } else {
        setModelSwitchProgressDetail("已找到现有模型登记，跳过重复登记。");
      }
      const { projectId, artifactId } = await ensureCatalogModelRegistration(selectedCatalogModel);
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
      setModelManagerDialogOpen(false);
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

  const switchModel = () => {
    if (!selectedCatalogModel || selectedCatalogModel.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 产物。");
      return;
    }
    if (!runtimeMainlineRunning) {
      void performModelSwitch();
      return;
    }
    const candidatePath = selectedCatalogModel.relative_path;
    setConfirmationRequest({
      eyebrow: "运行中切换模型",
      title: "停止并重启推理主链？",
      description: "当前主链正在运行。模型发布会停止现有管线、验证所选 Engine，并在成功后使用新模型重新启动。",
      details: [
        `所选模型：${candidatePath}`,
        "切换期间 DetectionBatch 与物理输出会短暂停止；失败时后端会回滚原部署。"
      ],
      confirmLabel: "确认切换模型",
      danger: true,
      handoffOnConfirm: true,
      onConfirm: performModelSwitch
    });
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
        `刷新完成：发现 ${result.model_count} 个模型文件；仅刷新目录与元数据，未读取 Engine 内容。`
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
  const telemetryWindowMs = readNullableNumber(statistics?.telemetry_window_ms);
  const runtimeMetricsStatus = statistics?.metrics_available === true
    ? telemetryWindowMs === null
      ? "已有样本 · 正在建立速率窗口"
      : "真实样本可用"
    : "等待首个运行样本";

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

        {mainlineLaunchPending ? (
          <div className="console-info">
            {mainlineLaunchMessage || "主链启动请求已提交，正在等待后端状态确认。"}
          </div>
        ) : runtimeMainlineSelected && runtimeMainlineRunning && runtimeMainlineStatus.readinessCode !== "ready" ? (
          <div className="console-info" role="status">
            {runtimeMainlineStatus.readinessLabel}：{runtimeMainlineStatus.readinessDetail}
          </div>
        ) : null}

        <section className={activePage === "capture" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集状态" value={captureStatusText} small={capture?.device || configuredCaptureDevice || "等待设备"} />
            <Metric title="主链输入 FPS" value={formatOptionalNumber(nvinferInputFps)} small="nvinfer sink 有效输入" />
            <Metric title="配置输入 FPS" value={formatOptionalNumber(configuredCaptureFps, 0)} small="配置值 · 非实时测量" />
            <Metric title="输入累计" value={formatOptionalInteger(statistics?.nvinfer_input_counter)} small="当前运行周期" />
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
                <p className="console-section-note">
                  这是 NVJPEG 预览分支的最高帧率；远程观看时还可以在画面上选择省流档。
                </p>
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
                  onCommit={handleCenteredRoiSizeChange}
                />
                <div className="mini-segmented roi-size-segmented" role="group" aria-label="ROI 尺寸">
                  {ROI_SIZE_CHOICES.map((size) => (
                    <button
                      className={roiSize === size ? "active" : ""}
                      disabled={busy === "roi.size" || busy === "capture.roi"}
                      key={size}
                      onClick={() => void handleCenteredRoiSizeChange(size)}
                      type="button"
                    >
                      {size}
                    </button>
                  ))}
                </div>
                <div className="console-kv compact-kv">
                  <span>源画面</span><b>{sourceWidth > 0 ? `${sourceWidth}x${sourceHeight}` : NO_SAMPLE}</b>
                  <span>配置 ROI</span><b>{sourceWidth > 0 ? `x=${roiX}, y=${roiY}, ${rustControlPlane ? `${configuredRoiWidth}x${configuredRoiHeight}` : `${roiSize}x${roiSize}`}` : NO_SAMPLE}</b>
                  <span>运行 ROI</span><b>{runtimeRoiAvailable ? `x=${runtimeRoiLeft}, y=${runtimeRoiTop}, ${runtimeRoiWidth}x${runtimeRoiHeight}` : NO_SAMPLE}</b>
                  <span>应用状态</span><b>{roiApplyLabel}</b>
                </div>
              </div>
          </div>

          <div className="console-grid2 diagnostic-grid" data-layer="capture">
            <div className="console-card">
              <SectionTitle title="采集基础状态" />
              <div className="console-kv">
                <span>采集状态</span><b>{captureStatusText}</b>
                <span>采集原因</span><b>{captureReason || NO_SAMPLE}</b>
                <span>采集设备</span><b>{capture?.device || configuredCaptureDevice || NO_SAMPLE}</b>
                <span>参数来源</span><b>{displayCaptureProfileSource}</b>
                <span>配置输入格式</span><b>{displayCaptureProfile?.pixel_format || NO_SAMPLE}</b>
                <span>配置输入分辨率</span><b>{displayCaptureProfile ? `${displayCaptureProfile.width}x${displayCaptureProfile.height}` : NO_SAMPLE}</b>
                <span>配置输入帧率</span><b>{displayCaptureProfile ? `${displayCaptureProfile.fps.toFixed(STANDARD_DECIMAL_DIGITS)} FPS` : NO_SAMPLE}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="主链输入健康" />
              <div className="console-kv">
                <span>nvinfer 输入 FPS</span><b>{formatOptionalNumber(nvinferInputFps, STANDARD_DECIMAL_DIGITS, "FPS")}</b>
                <span>nvinfer 输入累计</span><b>{formatOptionalInteger(statistics?.nvinfer_input_counter)}</b>
                <span>统计窗口</span><b>{formatOptionalNumber(telemetryWindowMs, 0, "ms")}</b>
                <span>统计状态</span><b>{runtimeMetricsStatus}</b>
                <span>说明</span><b>当前后端未提供采集卡原始 FPS 与协商 Caps</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "infer" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="推理 FPS" value={formatOptionalNumber(nvinferOutputFps)} small="nvinfer 实际完成" />
            <Metric title="结果 FPS" value={formatOptionalNumber(detectionBatchFps)} small="DetectionBatch 有效产出" />
            <Metric title="结果新鲜度" value={formatOptionalNumber(detectionDataAgeMs)} small={`${detectionFreshness} · ms`} />
            <Metric title="最近检测" value={formatOptionalInteger(detectionCount)} small="最近遥测 · 最多 5Hz" />
          </div>
          <div className="console-card model-selection-card">
            <SectionTitle title="模型设置" />
            <CurrentModelSummary
              artifactKind={artifact?.kind ?? ""}
              deployed={artifact !== null && artifact !== undefined}
              inputShape={displayedInputShape}
              loaded={runtimeInference.loaded === true}
              modelName={activeModelName}
              onOpenManager={openModelManager}
            />
          </div>
          <div className="console-grid2 inference-config-grid">
            <div className="console-card">
              <SectionTitle title="推理参数" />
              <label>配置置信度</label>
              <CommitNumberControl
                value={confidence}
                min={0}
                max={1}
                step={0.01}
                digits={2}
                onCommit={(value) => updateConfigField("inference", "confidence_threshold", value)}
              />
              <label>配置 NMS</label>
              <CommitNumberControl
                value={nms}
                min={0}
                max={1}
                step={0.01}
                digits={2}
                onCommit={(value) => updateConfigField("inference", "nms_threshold", value)}
              />
              <div className="console-kv compact-kv">
                <span>运行置信度</span><b>{runtimePostprocessAvailable ? formatOptionalNumber(runtimePostprocessConfidence, 2) : NO_SAMPLE}</b>
                <span>运行 NMS</span><b>{runtimePostprocessAvailable ? formatOptionalNumber(runtimePostprocessNms, 2) : NO_SAMPLE}</b>
                <span>应用状态</span><b>{postprocessApplyLabel}</b>
              </div>
              <details className="model-debug-details">
                <summary>当前模型工程详情</summary>
                <div className="model-debug-grid">
                  <span>当前模型</span><b>{activeModelName}</b>
                  <span>运行装载</span><b>{runtimeInference.loaded === true ? "已装载" : "未装载"}</b>
                  <span>登记输入</span><b>{registeredInputShape || "-"}</b>
                  <span>运行输入</span><b>{modelInputWidth > 0 && modelInputHeight > 0 ? `${modelInputWidth}x${modelInputHeight}` : NO_SAMPLE}</b>
                  <span>Artifact</span><b>{artifact ? `${artifact.kind} · ${artifact.status}` : NO_SAMPLE}</b>
                  <span>部署版本</span><b>{version?.version || NO_SAMPLE}</b>
                  <span>运行置信度</span><b>{formatOptionalNumber(runtimePostprocessConfidence, 2)}</b>
                  <span>运行 NMS</span><b>{formatOptionalNumber(runtimePostprocessNms, 2)}</b>
                  <span>运行 ROI</span><b>{runtimeRoiAvailable ? `${runtimeRoiWidth}x${runtimeRoiHeight}` : NO_SAMPLE}</b>
                  <span>后端</span><b>{readString(runtime?.inference?.selected, "auto")}</b>
                  <span>类别数量</span><b>{String(version?.classes.length ?? 0)}</b>
                  <span className="wide">类别名</span><b className="wide">{version?.classes.length ? version.classes.join(", ") : NO_SAMPLE}</b>
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
                previewFps={previewFps}
              />
              <div className="console-kv">
                <span>ROI 输入</span><b>{`${roiInputWidth || "-"}x${roiInputHeight || "-"}`}</b>
                <span>模型输入</span><b>{modelInputWidth && modelInputHeight ? `${modelInputWidth}x${modelInputHeight}` : "-"}</b>
                <span>缩放倍率</span><b>{inputDownscaleFactor === null ? NO_SAMPLE : `${formatNumber(inputDownscaleFactor, 2)}x`}</b>
                <span>输入面积比例</span><b>{inputAreaRatio === null ? NO_SAMPLE : formatPercent(inputAreaRatio)}</b>
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
                <span>推理状态</span><b>{inferenceStatusText}</b>
                <span>推理原因</span><b>{inferenceReason || NO_SAMPLE}</b>
                <span>nvinfer 输入帧</span><b>{formatOptionalInteger(deepstreamInputFrames)}</b>
                <span>nvinfer 输出 Buffer</span><b>{formatOptionalInteger(deepstreamOutputBuffers)}</b>
                <span>元数据提取成功</span><b>{formatOptionalInteger(deepstreamMetadataExtractions)}</b>
                <span>DetectionBatch 已发布</span><b>{formatOptionalInteger(deepstreamPublishedBatches)}</b>
                <span>DetectionBatch 已消费</span><b>{formatOptionalInteger(runtimeMainlineStatus.consumedBatches)}</b>
                <span>目标选择输入</span><b>{formatOptionalInteger(runtimeMainlineStatus.targetingBatches)}</b>
                <span>Buffer PTS 匹配</span><b>{formatOptionalInteger(runtimeInference.timestamp_buffer_pts_matches)}</b>
                <span>FrameMeta PTS 匹配</span><b>{formatOptionalInteger(runtimeInference.timestamp_frame_meta_pts_matches)}</b>
                <span>PTS 关联失败</span><b>{formatOptionalInteger(runtimeInference.timestamp_correlation_misses)}</b>
                <span>采样 Detection generation</span><b>{formatOptionalInteger(sampledDetectionGeneration)}</b>
                <span>统计窗口</span><b>{formatOptionalNumber(telemetryWindowMs, 0, "ms")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="模型输入" />
              <p className="console-section-note">这里只展示当前 Rust 运行态能够证明的尺寸；Tensor 类型、精度和布局未进入运行状态契约时不作推断。</p>
              <div className="console-kv">
                <span>模型名称</span><b>{activeModelName || NO_SAMPLE}</b>
                <span>推理后端</span><b>{selectedRuntimeBackend || NO_SAMPLE}</b>
                <span>ROI 输入尺寸</span><b>{roiInputWidth > 0 && roiInputHeight > 0 ? `${roiInputWidth}x${roiInputHeight}` : NO_SAMPLE}</b>
                <span>模型输入尺寸</span><b>{modelInputWidth > 0 && modelInputHeight > 0 ? `${modelInputWidth}x${modelInputHeight}` : displayedInputShape || NO_SAMPLE}</b>
                <span>运行 ROI</span><b>{runtimeRoiAvailable ? `${runtimeRoiWidth}x${runtimeRoiHeight}` : NO_SAMPLE}</b>
                <span>运行 generation</span><b>{formatOptionalInteger(sampledDetectionGeneration)}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="推理引擎阶段" />
              <p className="console-section-note">从数据进入 nvinfer 到输出离开：包含 DeepStream 输入预处理、TensorRT 执行和自定义 parser 解析；不包含目标跟踪与鼠标控制。</p>
              <div className="console-kv">
                <span>sink → src 总耗时</span><b>{formatOptionalNumber(inferenceTotalMs, 2, "ms")}</b>
                <span>有效计时样本</span><b>{formatOptionalInteger(statistics?.inference_latency_samples)}</b>
                <span>计时范围</span><b>预处理 + TensorRT + parser</b>
                <span>说明</span><b>不包含目标选择、跟踪与控制计算</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="输出与后处理" />
              <div className="console-kv">
                <span>置信度阈值</span><b>{formatOptionalNumber(runtimePostprocessConfidence, 2)}</b>
                <span>NMS IoU 阈值</span><b>{formatOptionalNumber(runtimePostprocessNms, 2)}</b>
                <span>最近检测数</span><b>{formatOptionalInteger(detectionCount)}</b>
                <span>遥测列表截断</span><b>{formatOptionalInteger(vision.detection_items_truncated)}</b>
                <span>最高检测置信度</span><b>{formatOptionalNumber(inferenceHighestConfidence, 3)}</b>
                <span>DetectionBatch 状态</span><b>{inferenceBatchState}</b>
                <span>DetectionBatch published</span><b>{!inferenceRan ? NO_SAMPLE : inferenceBatchPublished ? "是" : "否"}</b>
                <span>DetectionBatch age</span><b>{formatOptionalNumber(detectionDataAgeMs, 2, "ms")}</b>
                <span>控制新鲜度阈值</span><b>{formatOptionalNumber(detectionFreshnessThresholdMs, 2, "ms")}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "control" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="控制状态" value={controlHasSample ? readString(control.global_state, "已计算") : "未执行"} small={controlNoSendReason || NO_SAMPLE} />
            <Metric title="目标选择输入率" value={formatOptionalNumber(targetingBatchFps)} small="DetectionBatch/s" />
            <Metric title="控制观测帧龄" value={formatOptionalNumber(controlFrameAgeMs)} small="当前控制样本 · ms" />
            <Metric title="最近 Track" value={formatOptionalInteger(controlTrackId)} small={`${activeRuntimeClassLabel || "target"} · 最多 5Hz 遥测`} />
            <Metric title="控制误差" value={formatOptionalNumber(predictedErrorDistancePx)} small="px" />
            <Metric title="最近设备接受" value={hasAcceptedCommand ? lastAcceptedCommand : NO_SAMPLE} small="与当前样本独立" />
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
                <span>检测数量</span><b>{formatOptionalInteger(detectionCount)}</b>
                <span>原始 / 合格 / 已选择</span><b>{`${formatOptionalInteger(rawCandidateCount)} / ${formatOptionalInteger(eligibleCandidateCount)} / ${formatOptionalInteger(selectedTargetCount)}`}</b>
                <span>过滤原因</span><b>{targetPipelineRejections || NO_SAMPLE}</b>
                <span>生效类别过滤</span><b>{effectiveClassFilter === "all" ? "全部类别" : effectiveClassFilter === "none" ? "未选择任何类别" : `cls ${effectiveClassFilter}`}</b>
                <span>被类别过滤的 cls</span><b>{rejectedBasicClassIds.length > 0 ? rejectedBasicClassIds.join(", ") : NO_SAMPLE}</b>
                <span>基础过滤前 / 后</span><b>{`${formatOptionalInteger(basicCandidateFilter.raw_candidates)} / ${formatOptionalInteger(basicCandidateFilter.filtered_candidates)}`}</b>
                <span>基础过滤拒绝</span><b>{formatOptionalInteger(basicCandidateFilter.rejected_candidates)}</b>
                <span>候选目标数量</span><b>{formatOptionalInteger(controlCandidateCount)}</b>
                <span>最终选择数量</span><b>{controlHasTarget ? "1" : controlHasSample ? "0" : NO_SAMPLE}</b>
                <span>当前 track_id</span><b>{formatOptionalInteger(controlTrackId)}</b>
                <span>目标类别</span><b>{activeRuntimeClassLabel || NO_SAMPLE}</b>
                <span>目标置信度</span><b>{formatOptionalNumber(target.score, 3)}</b>
                <span>Track 身份置信度</span><b>{formatOptionalNumber(target.identity_confidence, 3)}</b>
                <span>目标选择状态</span><b>{readString(control.selector_state, "") || NO_SAMPLE}</b>
                <span>目标选择原因</span><b>{readString(control.selection_reason, "") || NO_SAMPLE}</b>
                <span>目标框坐标</span><b>{controlHasTarget ? `${formatPoint(target.x1, target.y1)} -> ${formatPoint(target.x2, target.y2)}` : NO_SAMPLE}</b>
                <span>目标框中心</span><b>{formatPoint(target.box_cx ?? target.cx, target.box_cy ?? target.cy, STANDARD_DECIMAL_DIGITS, "px")}</b>
                <span>遥测说明</span><b>目标与框为最近采样快照，最多约 5Hz</b>
              </div>
              {targetPipelineCode === "BASIC_CANDIDATE_REJECTED" && targetPipelineRejections.includes("class_filter") && detectedClassFilterValue ? (
                <div className="control-filter-recovery">
                  <button
                    className="console-button primary"
                    disabled={busy !== null}
                    onClick={() => void updateDetectionClassFilter(detectedClassFilterValue)}
                    type="button"
                  >
                    允许当前检测类别
                  </button>
                  <small>
                    {rustControlPlane
                      ? "保留已有选择，并加入本帧检测到的 cls；保存后重启主链生效。"
                      : "保留已有选择，并加入本帧检测到的 cls；保存后立即热更新。"}
                  </small>
                </div>
              ) : null}
            </div>
            <div className="console-card">
              <SectionTitle title="瞄准点" />
              <div className="console-kv">
                <span>原始瞄准点</span><b>{formatPoint(observedAimX, observedAimY, STANDARD_DECIMAL_DIGITS, "px")}</b>
                <span>目标类别</span><b>{activeRuntimeClassLabel || NO_SAMPLE}</b>
                <span>控制瞄准点</span><b>{formatPoint(predictedAimX, predictedAimY, STANDARD_DECIMAL_DIGITS, "px")}</b>
                <span>位置预测</span><b>{dualPhasePredictionEnabled ? "X / Y 已启用" : "已关闭"}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="观测与控制误差" />
              <div className="console-kv">
                <span>屏幕中心</span><b>{formatPoint(controlCenterX, controlCenterY, STANDARD_DECIMAL_DIGITS, "px")}</b>
                <span>observed error px</span><b>{formatPoint(controlPipeline.observed_error_x_px, controlPipeline.observed_error_y_px, 2, "px")}</b>
                <span>control error px</span><b>{formatPoint(predictedErrorXPx, predictedErrorYPx, 2, "px")}</b>
                <span>误差距离</span><b>{formatOptionalNumber(predictedErrorDistancePx, 2, "px")}</b>
                <span>控制 dt</span><b>{controlMeasurementDtS === null ? NO_SAMPLE : `${(controlMeasurementDtS * 1000).toFixed(3)} ms`}</b>
                <span>控制观测帧龄</span><b>{formatOptionalNumber(controlFrameAgeMs, 2, "ms")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="控制器输出" />
              <div className="console-kv">
                <span>控制模式</span><b>{controlModeLabel}</b>
                <span>移动策略</span><b>{readString(controlPipeline.movement_strategy, "") || NO_SAMPLE}</b>
                <span>FAR / NEAR</span><b>{readString(controlPipeline.mode, "") || NO_SAMPLE}</b>
                <span>完整修正 counts</span><b>{formatPoint(controlPipeline.full_error_counts_x, controlPipeline.full_error_counts_y, 2)}</b>
                <span>Atan 浮点需求</span><b>{formatPoint(controlPipeline.float_demand_x, controlPipeline.float_demand_y, 2)}</b>
                <span>整数输出</span><b>{formatPoint(controlPipeline.integer_command_x, controlPipeline.integer_command_y, 0, "counts")}</b>
                <span>量化余量</span><b>{formatPoint(controlPipeline.quantizer_residual_x, controlPipeline.quantizer_residual_y, 3, "counts")}</b>
                <span>到位区（进入 / 退出）</span><b>{formatPoint(controlPipeline.arrival_enter_counts, controlPipeline.arrival_exit_counts, 2, "counts")}</b>
                <span>每轴到位</span><b>{formatAxisSettlement(controlPipeline.arrival_settled_x, controlPipeline.arrival_settled_y)}</b>
                <span>视觉反馈门控</span><b>{formatFeedbackGate(controlPipeline.actuation_pending_x, controlPipeline.actuation_pending_y)}</b>
                <span>独立压枪状态</span><b>{recoilEnabled ? formatRecoilState(controlPipeline.recoil_state, controlPipeline.recoil_block_reason) : "关闭"}</b>
                <span>基础 / 追加速率</span><b>{`${formatOptionalNumber(controlPipeline.recoil_base_rate_counts_s, 0)} / ${formatOptionalNumber(controlPipeline.recoil_fast_add_rate_counts_s, 0)} counts/s`}</b>
                <span>门控 / 最终速率</span><b>{`${formatOptionalNumber(controlPipeline.recoil_position_gate, 2)} / ${formatOptionalNumber(controlPipeline.recoil_final_rate_counts_s, 0)} `}counts/s</b>
                <span>请求 / 实际输出</span><b>{`${formatOptionalNumber(controlPipeline.recoil_requested_counts_y, 2)} / ${formatOptionalNumber(controlPipeline.recoil_emitted_counts_y, 2)} counts`}</b>
                <span>误差归一化</span><b>{formatOptionalNumber(controlPipeline.recoil_error_y_norm, 3)}</b>
                <span>控制预算</span><b>{formatPoint(control.dx, control.dy, 0, "counts")}</b>
              </div>
            </div>
            <div className="console-card">
              <SectionTitle title="Latest Replace 与设备发送" />
              <p className="console-section-note">控制样本与设备回执分别展示；最近回执不冒充为当前观测的同步发送结果。</p>
              <div className="console-kv">
                <span>触发状态</span><b>{control.trigger_active === true ? "按下" : control.trigger_active === false ? "未按下" : NO_SAMPLE}</b>
                <span>是否允许发包</span><b>{control.will_emit === true ? "是" : control.will_emit === false ? "否" : NO_SAMPLE}</b>
                <span>运行输出门</span><b>{control.output_enabled === true ? "已打开" : control.output_enabled === false ? "已关闭" : NO_SAMPLE}</b>
                <span>不发包原因</span><b>{controlNoSendReason || NO_SAMPLE}</b>
                <span>本轮控制意图</span><b>{formatPoint(control.dx, control.dy, 0, "counts")}</b>
                <span>发送语义</span><b>仅保留最新观测</b>
                <span>设备接受累计</span><b>{formatOptionalInteger(acceptedCommandCount)}</b>
                <span>最近设备已接受</span><b>{hasAcceptedCommand ? lastAcceptedCommand : NO_SAMPLE}</b>
                <span>主链设备通道</span><b>{kmnetRuntimeConnectionLabel}</b>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "params" || activePage === "control-test" ? "console-page active" : "console-page"}>
          {activePage === "params" ? (
          <>
            <div className="console-metrics params-summary-metrics">
              <Metric title="控制模式" value={controlModeLabel} small="单选策略" />
              <Metric title="触发方式" value={triggerModeLabel(triggerMode)} small="trigger" />
              <Metric title="类型瞄点 Y" value={`${Math.round(aimRoleRatios.head * 100)} / ${Math.round(aimRoleRatios.body * 100)} / ${Math.round(aimRoleRatios.other * 100)}`} small="头部 / 身体 / 其他 %" />
              <Metric title="位置预测" value={dualPhasePredictionEnabled ? "X / Y 已启用" : "已关闭"} small={dualPhasePredictionEnabled ? "单目标真实帧间速度" : "当前观测 Atan 反馈"} />
              <Metric title="偏移输出配置" value={outputEnabled ? "允许" : "暂停"} small={outputEnabled ? "实际发送状态见控制页" : "算法仍继续计算"} />
              <Metric title="发送方式" value="最新覆盖" small={`${schedulerIntervalMs.toFixed(STANDARD_DECIMAL_DIGITS)} ms 单槽`} />
            </div>
            <div className={outputEnabled ? "console-card control-output-gate-card enabled" : "console-card control-output-gate-card paused"}>
              <div className="control-output-gate-identity">
                <span className="control-output-gate-icon" aria-hidden="true">
                  <NovaIcon name={outputEnabled ? "device-send" : "pause-output"} size={21} strokeWidth={1.8} />
                </span>
                <div>
                  <span className="class-config-eyebrow">GLOBAL OUTPUT GATE</span>
                  <h3>{outputEnabled ? "允许发送偏移控制量" : "偏移输出已暂停"}</h3>
                  <p>只控制最终鼠标位移是否交付；不会断开 KMNet，也不会停止采集、推理、目标选择和控制量计算。</p>
                </div>
              </div>
              <ModuleSwitch
                label="发送偏移控制量"
                detail={outputEnabled
                  ? kmnetRestartRequired
                    ? "kmNet 新配置等待 novasightd 重启；需要立即停发时请断开 kmNet"
                    : "关闭后立即清空待发送旧命令"
                  : kmnetRestartRequired
                    ? "重启 novasightd 装载 kmNet 新配置后才能打开输出"
                  : !kmnetAutoConnect
                    ? "请先保存 kmNet 配置，并按提示重启 novasightd"
                    : !kmnetExecutorAvailable
                      ? "dry-run 不允许物理输出；请以生产模式启动 novasightd"
                      : !kmnetRuntimeConnected
                        ? "请先连接 kmNet，再打开偏移输出"
                        : "开启后只发送新的实时观测"}
                disabled={busy !== null || kmnetRestartRequired || (!outputEnabled && (!kmnetAutoConnect || !kmnetExecutorAvailable || !kmnetRuntimeConnected))}
                enabled={outputEnabled}
                optimistic={false}
                onToggle={requestOutputGateChange}
              />
            </div>
            <div className="console-card motion-control-mode-card static">
              <div className="motion-control-mode-copy">
                <span className="class-config-eyebrow">生产算法隔离</span>
                <h3>双阶段 Atan 基础反馈</h3>
                <p>{dualPhasePredictionEnabled ? "单目标 X / Y 预测先修正当前位置，随后进入投影、Atan、限幅和量化。" : "设备输出只由当前测量误差、投影、Atan、限幅和量化产生。"}</p>
              </div>
              <ModuleSwitch
                label="启用 X / Y 目标预测"
                detail="只预测 Tracker 已选中的唯一目标；切换目标、时间戳异常或历史不足时自动归零。保存后重启主链生效。"
                enabled={dualPhasePredictionEnabled}
                onToggle={(enabled) => updateDualPhaseField("prediction_enabled", enabled)}
              />
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
                disabled={busy !== null}
                onClick={() => openConfigDialog("class-config")}
                type="button"
              >
                <NovaIcon name="settings" size={16} />
                管理类别配置
              </button>
            </div>
            <div className="console-grid2 params-control-grid compact-content-grid" data-algorithm-page={DEFAULT_CONTROL_ALGORITHM}>
              <div className="console-card">
                <SectionTitle title="控制模式" />
                <div className="console-kv compact-kv" aria-label="控制模式">
                  <span>生产控制器</span><b>{CONTROL_ALGORITHM_LABEL}</b>
                </div>
                <p className="console-section-note">{CONTROL_ALGORITHM_DESCRIPTION}</p>
                <label>触发方式</label>
                <select value={triggerMode} onChange={(event) => void updateConfigField("control", "trigger_mode", event.target.value)}>
                  <option value="hardware">kmNet 硬件按键触发</option>
                  <option value="always">检测到目标后自动控制</option>
                </select>
                <p className="console-section-note">硬件触发直接使用 daemon 缓存的 kmNet 按键状态；自动控制只要求存在合格目标。</p>
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
                <div className="console-kv compact-kv">
                  <span>输出交付</span><b>Latest Replace</b>
                  <span>每个推理结果</span><b>覆盖尚未发送的旧命令</b>
                  <span>待发送容量</span><b>1 条完整命令</b>
                  <span>设备发送校验</span><b>Rust DeviceLane</b>
                  <span>到位判断</span><b>角度投影后的 counts 迟滞区</b>
                </div>
              </div>

              <div className="console-card">
                <SectionTitle title={`控制算法 · ${controlModeLabel}`} />
                <p className="console-section-note">当前生产主链仅使用投影、增益、Atan 响应曲线和单次限幅。</p>
                <div className="advanced-settings-summary">
                  <div><span>FOVX</span><b>{dualPhaseFovX.toFixed(STANDARD_DECIMAL_DIGITS)}°</b></div>
                  <div><span>FAR / NEAR Kp</span><b>{dualPhaseFarKp.toFixed(3)} / {dualPhaseNearKp.toFixed(3)}</b></div>
                  <div><span>位置预测</span><b>{dualPhasePredictionEnabled ? "X / Y 已启用" : "已关闭"}</b></div>
                </div>
                <button className="console-button console-full-button" disabled={busy !== null} onClick={() => openConfigDialog("algorithm")} type="button">
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
                    <span>中心偏移</span><b>{crosshairReferenceReady ? `${crosshairOffsetX.toFixed(STANDARD_DECIMAL_DIGITS)}, ${crosshairOffsetY.toFixed(STANDARD_DECIMAL_DIGITS)} px` : "—"}</b>
                    <span>匹配置信度</span><b>{crosshairConfidence > 0 ? `${(crosshairConfidence * 100).toFixed(STANDARD_DECIMAL_DIGITS)}%` : "—"}</b>
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
                    onClick={requestClearCrosshair}
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
                  <SectionTitle title="Y 轴压枪" />
                  <ModuleSwitch label="启用压枪" detail="仅在总输出已开启、设备可用且检测到真实左键按下时输出。" enabled={recoilEnabled} onToggle={(enabled) => updateControlGroupField("recoil", "enabled", enabled)} />
                  <ModuleSwitch label="只在检测到目标时压枪" detail="开启后没有有效目标就停止压枪；关闭后即使没有目标，真实开火期间也会按补偿速率输出。" enabled={recoilRequireTarget} onToggle={(enabled) => updateControlGroupField("recoil", "require_target", enabled)} />
                  {recoilEnabled ? (
                    <details className="crosshair-advanced-settings">
                      <summary>当前速率补偿参数</summary>
                      <p className="console-section-note">这些参数只控制现有时间速率补偿。枪械轨迹采集会作为独立校准流程生成前馈曲线，不会复用目标误差追加强度。</p>
                      <div className="advanced-settings-grid">
                        <NumberControl label="压枪启动斜坡 ms" detail="从真实左键按下开始，压枪速率逐步进入基础速率。" value={recoilStartupRampMs} min={0} max={1000} step={1} onCommit={(value) => updateControlGroupField("recoil", "startup_ms", value)} />
                        <NumberControl label="基础压枪速率 counts/s" detail="与观测 FPS 无关的时间速率。" value={recoilBaseRate} min={0} max={20000} step={1} onCommit={(value) => updateControlGroupField("recoil", "base_rate_counts_s", value)} />
                        <NumberControl label="最大压枪速率 counts/s" value={readNumber(recoilConfig.max_rate_counts_s, 0)} min={0} max={20000} step={1} onCommit={(value) => updateControlGroupField("recoil", "max_rate_counts_s", value)} />
                        <NumberControl label="目标误差追加强度 counts/s" value={readNumber(recoilConfig.fast_add_gain_counts_s, 0)} min={0} max={20000} step={1} onCommit={(value) => updateControlGroupField("recoil", "fast_add_gain_counts_s", value)} />
                      </div>
                    </details>
                  ) : null}
                </div>

              <div className="console-card">
                <SectionTitle title="目标选择与切换 · 通用参数" />
                <NumberControl label="目标选择半径（640 基准 px）" detail="以 640×640 ROI 为基准；运行时按当前 ROI 尺寸同比缩放，保证 320～640 ROI 使用一致的相对选择范围。" value={targetFovRadiusPx} min={1} max={640} step={1} onCommit={(value) => updatePipelineField("target_fov_radius_px", value)} />
                <div className="target-weight-summary">
                  <div>
                    <span>当前综合分权重</span>
                    <strong>
                      类别 {(normalizedSelectionClassWeight * 100).toFixed(0)}%
                      <i>·</i>
                      距离 {(normalizedSelectionDistanceWeight * 100).toFixed(0)}%
                    </strong>
                    <small>类别偏好与准星距离由此处调整；候选可靠性由系统内部自动处理。</small>
                  </div>
                  <button className="console-button" disabled={busy !== null} onClick={() => openConfigDialog("target-weights")} type="button">
                    <NovaIcon name="settings" size={15} />
                    调整权重
                  </button>
                </div>
                <div className="advanced-settings-summary compact">
                  <div><span>异常框宽高比</span><b>≤ {candidateRatioMaxAspect.toFixed(STANDARD_DECIMAL_DIGITS)}</b></div>
                  <div><span>切换门槛</span><b>{targetSwitchPreferenceAdvantage.toFixed(2)}</b></div>
                  <div><span>确认延迟</span><b>{targetSwitchDelayMs.toFixed(0)} ms</b></div>
                </div>
                <button className="console-button console-full-button" disabled={busy !== null} onClick={() => openConfigDialog("target-advanced")} type="button">
                  <NovaIcon name="settings" size={15} />
                  目标切换高级设置
                </button>
              </div>

              <div className="console-card">
                <SectionTitle title="Tracker · 身份关联" />
                <div className="console-kv compact-kv"><span>关联算法</span><b>Hungarian</b><span>输出状态</span><b>仅 ACTIVE</b></div>
                <div className="advanced-settings-summary compact">
                  <div><span>匹配距离</span><b>{trackerMaxMatchDistance.toFixed(2)}</b></div>
                  <div><span>位置 / IoU / 尺度</span><b>{trackerPositionCostWeight.toFixed(2)} / {trackerIouCostWeight.toFixed(2)} / {trackerScaleCostWeight.toFixed(2)}</b></div>
                  <div><span>丢失保持</span><b>{targetLostGraceMs.toFixed(0)} ms</b></div>
                </div>
                <button className="console-button console-full-button" disabled={busy !== null} onClick={() => openConfigDialog("tracker")} type="button">
                  <NovaIcon name="settings" size={15} />
                  管理 Tracker
                </button>
              </div>
            </div>
          </>
          ) : (
          <>
          <div className="console-metrics">
            <Metric title="设备连接" value={kmnetConnectionLabel} small={kmnetRestartRequired ? `运行 ${effectiveConfigRevision} · 已保存 ${desiredConfigRevision}` : kmnetConnected ? "ready" : kmnetConnecting ? "connecting" : kmnetRetryable ? "retry available" : "unavailable"} />
            <Metric title="执行器可用" value={kmnetRestartRequired ? "等待装载" : kmnetExecutorAvailable ? "可用" : "不可用"} small="kmNet" />
            <Metric title="按键数据" value={kmnetStatus.buttons_available === true ? "可用" : "不可用"} small="最近轮询" />
            <Metric title="自动连接" value={kmnetAutoConnect ? kmnetRestartRequired ? "重启后启用" : "已启用" : "已关闭"} small="startup" />
            <Metric title="主链接受命令" value={formatOptionalInteger(kmnetStatus.accepted_command_count)} small="device receipts" />
            <Metric title="最近接受位移" value={formatPoint(kmnetStatus.last_accepted_dx, kmnetStatus.last_accepted_dy, 0)} small="dx / dy" />
          </div>
          <div className="console-grid2 control-test-grid">
            <div className="console-card">
              <SectionTitle title="kmNet 控制面板" />
              <div className="kmnet-status-grid">
                <div className={kmnetRestartRequired ? "kmnet-status-tile idle" : kmnetConnected ? "kmnet-status-tile good" : kmnetConnectionFailed ? "kmnet-status-tile bad" : "kmnet-status-tile idle"}>
                  <span>连接</span>
                  <b>{kmnetConnectionLabel}</b>
                </div>
                <div className={kmnetStatus.buttons_available === true ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>按键</span>
                  <b>{kmnetStatus.buttons_available === true ? `${kmnetButtonLeft ? "左键" : "-"} / ${kmnetButtonRight ? "右键" : "-"}` : "数据不可用"}</b>
                </div>
                <div className={kmnetExecutorAvailable ? "kmnet-status-tile good" : "kmnet-status-tile bad"}>
                  <span>执行器</span>
                  <b>{kmnetExecutorAvailable ? "可用" : "不可用"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>执行器</span>
                  <b>{readString(executorStatus.selected, NO_SAMPLE)}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>主链接受命令</span>
                  <b>{formatOptionalInteger(kmnetStatus.accepted_command_count)}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>最近接受位移</span>
                  <b>{formatPoint(kmnetStatus.last_accepted_dx, kmnetStatus.last_accepted_dy, 0)}</b>
                </div>
                <div className={control.will_emit === true ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>当前命令门控</span>
                  <b>{control.will_emit === true ? "允许" : control.will_emit === false ? "阻止" : NO_SAMPLE}</b>
                </div>
                <div className={kmnetRuntimeConnected ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>主链设备通道</span>
                  <b>{kmnetRuntimeConnectionLabel}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>设备接受状态</span>
                  <b>{acceptedCommandCount === null ? NO_SAMPLE : hasAcceptedCommand ? "已有协议回执" : "尚无回执"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>连接阶段</span>
                  <b>{readString(kmnetStatus.connection_state, NO_SAMPLE)}</b>
                </div>
                <div className={kmnetConfigurationReady ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>配置装载</span>
                  <b>{kmnetRestartRequired ? `${effectiveConfigRevision} → ${desiredConfigRevision}` : kmnetConfigurationReady ? "已生效" : "未委任"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>设备恢复次数</span>
                  <b>{formatOptionalInteger(kmnetStatus.device_recovery_count)}</b>
                </div>
                <div className={readNumber(kmnetStatus.device_error_count, 0) > 0 ? "kmnet-status-tile bad" : "kmnet-status-tile idle"}>
                  <span>设备错误次数</span>
                  <b>{formatOptionalInteger(kmnetStatus.device_error_count)}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>最近设备错误</span>
                  <b>{kmnetLastDeviceError || kmnetLastError || "无"}</b>
                </div>
              </div>
              {kmnetRestartRequired || kmnetLastError || kmnetConnectionFailed || kmnetConnectionDegraded ? (
                <div className={!kmnetRestartRequired && kmnetConnectionFailed ? "kmnet-connection-notice failed" : "kmnet-connection-notice warn"} role="status">
                  <div>
                    <strong>{kmnetRestartRequired ? "kmNet 配置已保存，等待后端装载" : kmnetConnectionFailed ? "输出设备未连接" : kmnetConnectionDegraded ? "设备连接异常，正在自动恢复" : "最近一次输出失败"}</strong>
                    <span>{kmnetRestartRequired
                      ? `novasightd 当前使用 revision ${effectiveConfigRevision}，已保存 revision ${desiredConfigRevision}。重启进程后才会使用新的地址和 UUID。`
                      : kmnetLastError || (kmnetConnectionFailed ? "请检查地址、端口、UUID 和网络连通性。" : "视觉主链继续运行，物理偏移输出保持关闭。")}</span>
                  </div>
                  {kmnetRestartRequired ? (
                    <small>这是配置生效等待，不是 kmNet 网络连接失败；重启前不会尝试用旧设备对象连接新配置。</small>
                  ) : kmnetRetryable ? (
                    <small>
                      {rustControlPlane
                        ? "低频设备线程会自动重试；也可以点击“立即重试连接”。"
                        : "修改配置后点击“重新连接”，无需重启主链。"}
                    </small>
                  ) : null}
                </div>
              ) : null}
              <div className="console-action-row kmnet-connection-actions">
                <button
                  className="console-button"
                  aria-pressed={kmnetConnected}
                  disabled={
                    busy !== null ||
                    kmnetRestartRequired ||
                    !kmnetAutoConnect ||
                    !kmnetCanConnect
                  }
                  onClick={() => void setKmNetConnection(true)}
                  type="button"
                >
                  {kmnetRestartRequired
                    ? "等待 novasightd 重启"
                    : kmnetRuntimeConnected
                      ? "kmNet 已连接"
                      : kmnetConnecting
                        ? "连接中"
                        : kmnetConnectionDegraded
                          ? "立即重试连接"
                          : kmnetAutoConnect
                            ? "连接 kmNet"
                            : "请先启用自动连接"}
                </button>
                <button
                  className="console-button danger"
                  disabled={busy !== null || !kmnetCanDisconnect}
                  onClick={() => void setKmNetConnection(false)}
                  type="button"
                >
                  断开 kmNet
                </button>
                <span>
                  {kmnetRestartRequired
                    ? "新配置尚未进入当前进程；请停止并重新运行 novasightd。"
                    : !kmnetExecutorAvailable
                    ? "当前是 dry-run 或硬件输出不可用；物理发送必须使用不带 --dry-run 的生产模式启动。"
                    : runtime?.running
                      ? kmnetBlockedReason === "already_connected"
                        ? "设备已连接；断开只停止物理输出，采集、推理和目标计算保持运行。"
                        : "连接操作只影响 kmNet 会话，采集、推理和目标计算保持运行。"
                      : "请先启动主链；Runtime Epoch 建立后才能控制 kmNet 会话。"}
                </span>
              </div>
              <TextControl label="kmnetip" value={kmnetHost} onCommit={(value) => updateConfigField("hardware", "host", value)} />
              <NumberControl label="kmnetport" value={kmnetPort} min={rustControlPlane ? 1 : 0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "port", Math.round(value))} />
              <TextControl label="kmnetuuid" value={kmnetUuid} onCommit={(value) => updateConfigField("hardware", "uuid", value)} />
              <NumberControl label="monitor_port" value={kmnetMonitorPort} min={rustControlPlane ? 1024 : 0} max={rustControlPlane ? 49151 : 65535} step={1} onCommit={(value) => updateConfigField("hardware", "monitor_port", Math.round(value))} />
              <ModuleSwitch
                label={rustControlPlane ? "主链启动时连接设备" : "后端服务启动时自动连接"}
                detail={rustControlPlane
                  ? "修改后需要重启 novasightd；连接失败时视觉主链继续运行，并由低频设备线程自动重连"
                  : "独立于主链启动；连接失败不会阻止采集、推理和鼠标算法运行"}
                enabled={kmnetAutoConnect}
                onToggle={(enabled) => updateConfigField("hardware", "auto_connect", enabled)}
              />
              <label>命令调度</label>
              <div className="console-kv compact-kv">
                <span>执行层</span><b>Latest Replace Scheduler</b>
                <span>行为</span><b>新观测覆盖未发送的旧命令</b>
                <span>待发送容量</span><b>1 条完整命令</b>
                <span>最终校验</span><b>Rust DeviceLane</b>
              </div>
              <NumberControl label="空闲 / 压枪调度间隔 ms" detail="目标跟踪命令不等待此间隔；该值只控制空闲设备轮询与独立压枪节拍。" value={schedulerIntervalMs} min={1} max={10} step={1} onCommit={(value) => updateDualPhaseField("output_interval_ms", Math.round(value))} />
              <div className="console-action-row">
                <button
                  className="console-button"
                  disabled={busy === "kmnet.defaults"}
                  onClick={requestApplyKmNetRecommended}
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
                </div>
                <div className="kmnet-pad">
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(0, -10)}>↑</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(-10, 0)}>←</button>
                  <button type="button" onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy)} disabled={kmnetDiagnosticDisabled}>发送</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(10, 0)}>→</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => void diagnosticMoveHardware(0, 10)}>↓</button>
                </div>
                <div className="kmnet-test-section">
                  <h3>Supervisor 单步诊断</h3>
                  <p>仅在主链停止时，由 Rust supervisor 串行调用 daemon-owned PointerDevice；不会与实时 DeviceLane 竞争。</p>
                  <div className="console-action-row">
                    <button
                      className="console-button"
                      disabled={kmnetDiagnosticDisabled}
                      onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy)}
                      type="button"
                    >
                      发送单步 raw
                    </button>
                  </div>
                </div>
                <div className="kmnet-test-result">
                  <span>最近诊断移动</span>
                  <b>{`${formatPoint(kmnetStatus.last_diagnostic_dx, kmnetStatus.last_diagnostic_dy, 0)} · ${formatOptionalInteger(kmnetStatus.diagnostic_move_count)} 次`}</b>
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
            <Metric title="nvinfer sink → src" value={formatOptionalNumber(inferenceTotalMs)} small="预处理 + TensorRT + parser · ms" />
            <Metric title="结果帧龄" value={formatOptionalNumber(detectionDataAgeMs)} small={`${detectionFreshness} · ms`} />
            <Metric title="结果 FPS" value={formatOptionalNumber(detectionBatchFps)} small="DetectionBatch/s" />
            <Metric title="统计窗口" value={formatOptionalNumber(telemetryWindowMs, 0)} small="ms" />
          </div>
          <div className="console-grid2 latency-analysis-grid">
            <KvCard
              title="当前真实测量"
              rows={[
                ["统计状态", runtimeMetricsStatus],
                ["nvinfer 输入 FPS", formatOptionalNumber(nvinferInputFps, 2, "FPS")],
                ["nvinfer 输出 FPS", formatOptionalNumber(nvinferOutputFps, 2, "FPS")],
                ["DetectionBatch FPS", formatOptionalNumber(detectionBatchFps, 2, "FPS")],
                ["目标选择输入 FPS", formatOptionalNumber(targetingBatchFps, 2, "FPS")],
                ["sink → src 样本", formatOptionalInteger(statistics?.inference_latency_samples)],
                ["控制新鲜度阈值", formatOptionalNumber(detectionFreshnessThresholdMs, 2, "ms")]
              ]}
              notice={<p className="latency-boundary-note">这些值来自 supervisor 低频统计窗口，不在采集推理热路径中为 UI 增加工作。</p>}
            />
            <KvCard
              title="测量边界"
              rows={[
                ["nvinfer sink → src", "包含 DeepStream 预处理、TensorRT 与 parser"],
                ["结果帧龄", "当前时刻 − 最新已发布 DetectionBatch 的采集时间"],
                ["结果 FPS", "统计窗口内已发布 DetectionBatch 增量 / 实际秒数"],
                ["未提供", "采集、解码、ROI、排队、控制计算的独立阶段耗时"]
              ]}
              notice={<p className="latency-boundary-note">未打点的阶段不展示 0、不估算，也不拼成所谓完整链路。</p>}
            />
          </div>
        </section>
      </main>

      {wideThemeGallery ? <ThemeGallery /> : null}

      <AdvancedSettingsDialog
        description="这里只展示 Rust 主链真实读取的投影、预测、Atan、量化与时效安全参数。"
        dirty={configDialogDirty}
        eyebrow="参数设置 / 控制算法"
        footerNote={`当前算法：${controlModeLabel}`}
        onClose={() => void requestDismissConfigDialog("algorithm")}
        onSave={() => void saveConfigDialog("algorithm")}
        open={algorithmSettingsDialogOpen}
        saveError={dialogSaveError}
        saving={dialogSaving}
        title={`${controlModeLabel} · 高级参数`}
      >
        <div className="advanced-settings-grid">
          <NumberControl label="观测新鲜度上限 ms" detail="超过该帧龄的视觉观测不会进入控制器。" value={freshnessThresholdMs} min={1} max={1000} step={0.1} onCommit={(value) => updateDualPhaseField("freshness_threshold_ms", value)} />
          <NumberControl label="水平 FOVX" value={dualPhaseFovX} min={30} max={179} step={0.1} onCommit={(value) => updateDualPhaseField("projection_fov_x_deg", value)} />
          <NumberControl label="每圈 counts" value={dualPhaseCountsPer360} min={1} max={100000} step={1} onCommit={(value) => updateDualPhaseField("projection_counts_per_360", value)} />
          <NumberControl label="近远过渡中心 px" detail="以该误差距离为中心，在前后 25% 区间内平滑融合 NEAR 与 FAR Atan 响应，避免阈值附近突然换挡。" value={dualPhaseNearThreshold} min={0} max={1000} step={0.1} onCommit={(value) => updateDualPhaseField("near_threshold_px", value)} />
          <NumberControl label="FAR Kp" detail="远距离闭环增益；不是 KMNet 设备能力上限。" value={dualPhaseFarKp} min={0.001} max={0.999} step={0.001} onCommit={(value) => updateDualPhaseField("far_kp", value)} />
          <NumberControl label="NEAR Kp" detail="接近准星后的闭环增益，过高会导致左右往返修正。" value={dualPhaseNearKp} min={0.001} max={0.999} step={0.001} onCommit={(value) => updateDualPhaseField("near_kp", value)} />
          <NumberControl label="共享 Atan 尺度 counts" detail="控制响应曲线尺度，不代表协议可发送的最大 counts。" value={dualPhaseAtanScale} min={0.1} max={10000} step={0.1} onCommit={(value) => updateDualPhaseField("atan_scale_counts", value)} />
          <NumberControl label="FAR 单次上限 counts" detail="稳定性保护上限；KMNet 的 signed-16 能力独立校验。" value={dualPhaseFarMaxCounts} min={1} max={2000} step={1} onCommit={(value) => updateDualPhaseField("far_max_counts_per_update", value)} />
          <NumberControl label="NEAR 单次上限 counts" detail="近目标单次修正上限，默认低于 FAR 以抑制过冲。" value={dualPhaseNearMaxCounts} min={1} max={2000} step={1} onCommit={(value) => updateDualPhaseField("near_max_counts_per_update", value)} />
          <NumberControl label="到位半径 counts" detail="每轴进入该范围后清空残差并停止；退出范围自动扩大 1.5 倍形成迟滞。" value={dualPhaseArrivalRadiusCounts} min={0.5} max={100} step={0.5} onCommit={(value) => updateDualPhaseField("arrival_radius_counts", value)} />
          <NumberControl label="量化残差上限 counts" detail="限制不足一个设备计数的累计余量，范围为 0～1。" value={residualCap} min={0} max={1} step={0.01} onCommit={(value) => updateDualPhaseField("residual_cap", value)} />
          <ModuleSwitch label="启用 X / Y 目标预测" detail="使用唯一锁定目标的真实帧间速度；不会预测多个候选目标。" enabled={dualPhasePredictionEnabled} onToggle={(enabled) => updateDualPhaseField("prediction_enabled", enabled)} />
          {dualPhasePredictionEnabled ? (
            <>
              <NumberControl label="预测速度平滑帧数" value={dualPhasePredictionSmoothingFrames} min={0.1} max={120} step={0.1} onCommit={(value) => updateDualPhaseField("velocity_smoothing_frames", value)} />
              <NumberControl label="预测历史重置间隔 ms" detail="相邻有效画面超过该时间后丢弃旧速度，避免断流后沿旧方向预测。" value={dualPhasePredictionHistoryResetGapMs} min={0.1} max={10000} step={0.1} onCommit={(value) => updateDualPhaseField("velocity_history_reset_gap_ms", value)} />
              <NumberControl label="速度离散基础容差 px/ms" detail="三段速度样本的离散程度超过基础值加相对值后，会降低预测可信度。" value={velocitySpreadBasePxMs} min={0.01} max={100} step={0.01} onCommit={(value) => updateDualPhaseField("velocity_spread_base_px_ms", value)} />
              <NumberControl label="速度离散相对容差" value={velocitySpreadRelative} min={0} max={100} step={0.01} onCommit={(value) => updateDualPhaseField("velocity_spread_relative", value)} />
              <NumberControl label="速度变化基础容差 px/ms" detail="限制相邻平滑速度的突变；超过阈值时降低预测可信度。" value={velocityChangeBasePxMs} min={0.01} max={100} step={0.01} onCommit={(value) => updateDualPhaseField("velocity_change_base_px_ms", value)} />
              <NumberControl label="速度变化相对容差" value={velocityChangeRelative} min={0} max={100} step={0.01} onCommit={(value) => updateDualPhaseField("velocity_change_relative", value)} />
              <NumberControl label="预测提前帧数" value={dualPhasePredictionLeadFrames} min={0} max={10} step={0.1} onCommit={(value) => updateDualPhaseField("prediction_lead_frames", value)} />
              <NumberControl label="FAR 预测绝对上限 px" value={dualPhasePredictionFarCapPx} min={0} max={100000} step={0.1} onCommit={(value) => updateDualPhaseField("prediction_far_absolute_cap_px", value)} />
              <NumberControl label="FAR 预测基础上限 px" detail="实际上限取绝对上限与“基础上限 + 相对上限 × 当前误差”中的较小值。" value={dualPhasePredictionFarBaseCapPx} min={0} max={100000} step={0.1} onCommit={(value) => updateDualPhaseField("prediction_far_base_cap_px", value)} />
              <NumberControl label="FAR 预测相对上限" value={dualPhasePredictionFarRelativeCap} min={0} max={100} step={0.01} onCommit={(value) => updateDualPhaseField("prediction_far_relative_cap", value)} />
              <NumberControl label="NEAR 预测绝对上限 px" value={dualPhasePredictionNearCapPx} min={0} max={100000} step={0.1} onCommit={(value) => updateDualPhaseField("prediction_near_absolute_cap_px", value)} />
              <NumberControl label="NEAR 预测基础上限 px" detail="接近目标时的预测基线，避免小误差被过量提前。" value={dualPhasePredictionNearBaseCapPx} min={0} max={100000} step={0.1} onCommit={(value) => updateDualPhaseField("prediction_near_base_cap_px", value)} />
              <NumberControl label="NEAR 预测相对上限" value={dualPhasePredictionNearRelativeCap} min={0} max={100} step={0.01} onCommit={(value) => updateDualPhaseField("prediction_near_relative_cap", value)} />
            </>
          ) : null}
          <NumberControl label="设备反馈等待 ms" detail="设备成功移动后额外等待的最小视觉反馈时间；运行时还会自动加一帧实测采集周期，避免同一旧画面重复驱动。" value={actuationFeedbackDelayMs} min={0} max={100} step={0.5} onCommit={(value) => updateDualPhaseField("actuation_feedback_delay_ms", value)} />
        </div>
      </AdvancedSettingsDialog>

      <AdvancedSettingsDialog
        description="控制异常框过滤、候选切换门槛和防抖确认。设置过严会阻止切换，过松会造成目标跳变。"
        dirty={configDialogDirty}
        eyebrow="参数设置 / 目标选择"
        footerNote="这些设置不会改变框内 aim Y，只影响选择与切换。"
        onClose={() => void requestDismissConfigDialog("target-advanced")}
        onSave={() => void saveConfigDialog("target-advanced")}
        open={targetAdvancedDialogOpen}
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="目标切换高级设置"
      >
        <div className="advanced-settings-grid two-column">
          <NumberControl label="控制目标最低置信度" detail="推理结果通过模型阈值后，还必须达到该值才允许进入目标选择。" value={targetMinConfidence} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("target_min_confidence", value)} />
          <NumberControl label="候选框最大宽高比" detail="拒绝宽高比或高宽比超过此值的异常细长框。值越大越宽松。" value={candidateRatioMaxAspect} min={1} max={20} step={0.1} onCommit={(value) => updatePipelineField("candidate_max_aspect_ratio", value)} />
          <NumberControl label="切换最小优势" detail="新候选综合分减去当前锁定目标综合分，至少达到此值才允许切换。" value={targetSwitchPreferenceAdvantage} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("target_switch_min_preference_advantage", value)} />
          <NumberControl label="切换最小连续性" detail="新候选 Track 的身份连续性至少达到此值，才允许进入切换确认。" value={targetSwitchContinuityScore} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("target_switch_min_continuity_score", value)} />
          <NumberControl label="目标切换确认延迟 ms" detail="新候选持续满足优势和连续性阈值达到此时间后，才正式替换当前目标。" value={targetSwitchDelayMs} min={0} max={500} step={1} onCommit={(value) => updatePipelineField("target_switch_delay_ms", value)} />
        </div>
      </AdvancedSettingsDialog>

      <AdvancedSettingsDialog
        description="这些参数直接进入 Rust Tracker 的跨帧身份关联；设置过松会误关联，过严会频繁断轨。"
        dirty={configDialogDirty}
        eyebrow="参数设置 / Tracker"
        footerNote="关联算法固定为 Hungarian；仅输出 ACTIVE Track。"
        onClose={() => void requestDismissConfigDialog("tracker")}
        onSave={() => void saveConfigDialog("tracker")}
        open={trackerSettingsDialogOpen}
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="Tracker 高级设置"
      >
        <div className="advanced-settings-grid two-column">
          <NumberControl label="归一化匹配距离" value={trackerMaxMatchDistance} min={0.1} max={5} step={0.05} onCommit={(value) => updatePipelineField("tracker_max_match_distance", value)} />
          <NumberControl label="位置代价权重" value={trackerPositionCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("tracker_position_cost_weight", value)} />
          <NumberControl label="IoU 代价权重" value={trackerIouCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("tracker_iou_cost_weight", value)} />
          <NumberControl label="尺度代价权重" detail="候选框尺寸变化参与身份匹配的权重。" value={trackerScaleCostWeight} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("tracker_scale_cost_weight", value)} />
          <NumberControl label="最大尺寸变化倍数" detail="宽或高相对上一帧变化超过该倍数时，不允许关联为同一目标。" value={trackerMaxSizeRatio} min={1} max={100} step={0.1} onCommit={(value) => updatePipelineField("tracker_max_size_ratio", value)} />
          <NumberControl label="最大关联时间间隔 ms" detail="两次观测间隔超过该值时，不使用旧轨迹继续关联。" value={trackerMaxAssociationDtMs} min={1} max={10000} step={1} onCommit={(value) => updatePipelineField("tracker_max_association_dt_ms", value)} />
          <NumberControl label="无时间戳漏检上限（帧）" detail="只用于没有有效捕获时间戳的回放或降级输入；Jetson 正常主链优先使用毫秒保持时间。" value={targetTrackMaxAge} min={1} max={120} step={1} onCommit={(value) => updatePipelineField("target_track_max_age", Math.round(value))} />
          <NumberControl label="目标丢失保持 ms" detail="锁定目标短暂漏检时暂停输出并保留原身份；超过该时间后才允许其他目标接管。" value={targetLostGraceMs} min={1} max={10000} step={1} onCommit={(value) => updatePipelineField("target_track_max_lost_age_ms", value)} />
        </div>
        <div className="advanced-settings-divider">
          <span>Rust Tracker</span>
          <small>{dualPhasePredictionEnabled ? "主链只预测唯一锁定目标的 X / Y 位置；Tracker 切换身份会立即重置预测历史。" : "主链使用有界关联保持目标身份；当前位置直接进入 Atan。"}</small>
        </div>
      </AdvancedSettingsDialog>

      {targetWeightsDialogOpen ? (
        <div
          className="target-weight-dialog-layer"
          onClick={(event) => {
            if (event.target === event.currentTarget && !dialogSaving) {
              void requestDismissConfigDialog("target-weights");
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
                <p>比例决定多个候选同时出现时，类别顺序与准星距离各自占多大影响；置信度只负责候选准入，不参与排序。</p>
              </div>
              <button
                aria-label="关闭权重调整"
                className="launch-dialog-close"
                disabled={dialogSaving}
                onClick={() => void requestDismissConfigDialog("target-weights")}
                title={configDialogDirty ? "关闭；未保存修改会先请求确认" : "关闭"}
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
                    <small>类别比例由用户设置，距离自动使用剩余比例，两项始终合计 100%。</small>
                  </div>
                  <b>类别优先</b>
                </div>
                <div className="target-weight-composition" aria-label="综合目标分数权重占比">
                  <i className="class" style={{ flexGrow: normalizedSelectionClassWeight }} />
                  <i className="distance" style={{ flexGrow: normalizedSelectionDistanceWeight }} />
                </div>
                <div className="target-weight-legend">
                  <span><i className="class" />类别 <b>{(normalizedSelectionClassWeight * 100).toFixed(0)}%</b></span>
                  <span><i className="distance" />距离 <b>{(normalizedSelectionDistanceWeight * 100).toFixed(0)}%</b></span>
                </div>
                <div className="target-weight-controls">
                  <NumberControl label="类别偏好比例" detail="类别顺序按 1、0.5、0.25…递减；距离自动使用剩余比例。提高后更倾向高优先类别。" value={normalizedSelectionClassWeight} min={0} max={1} step={0.01} onCommit={(value) => updatePipelineField("target_selection_class_ratio", value)} />
                </div>
              </section>

            </div>

            <footer className="target-weight-dialog-footer">
              <span className={dialogSaveError ? "dialog-save-status error" : configDialogDirty ? "dialog-save-status dirty" : "dialog-save-status"} role="status" aria-live="polite">
                {dialogSaving
                  ? "正在保存本次修改…"
                  : dialogSaveError
                    ? `保存失败 · ${dialogSaveError}`
                    : configDialogDirty
                      ? "有未保存修改 · 保存后才会同步到运行配置。"
                      : "未修改 · 关闭不会请求后端。"}
              </span>
              <button
                className={`console-button ${configDialogDirty ? "primary dialog-save-button" : "dialog-close-button"}`}
                disabled={dialogSaving}
                onClick={() => void saveConfigDialog("target-weights")}
                type="button"
              >
                {configDialogDirty ? "保存并关闭" : "关闭"}
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {classConfigDialogOpen ? (
        <div
          className="class-config-dialog-layer"
          onClick={(event) => {
            if (event.target === event.currentTarget && !dialogSaving) {
              void requestDismissConfigDialog("class-config");
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
                onClick={() => void requestDismissConfigDialog("class-config")}
                title={configDialogDirty ? "关闭；未保存修改会先请求确认" : "关闭"}
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
                        onClick={() => void updateDetectionClassFilter("all")}
                        type="button"
                      >
                        全部选择
                      </button>
                      <button
                        className="console-button"
                        disabled={busy !== null || selectedDetectionClassIds.size === 0}
                        onClick={() => void updateDetectionClassFilter("none")}
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
              <span className={dialogSaveError ? "dialog-save-status error" : configDialogDirty ? "dialog-save-status dirty" : "dialog-save-status"} role="status" aria-live="polite">
                {dialogSaving
                  ? "正在保存类别配置…"
                  : dialogSaveError
                    ? `保存失败 · ${dialogSaveError}`
                    : configDialogDirty
                      ? "有未保存修改 · 保存后才会同步整份类别配置。"
                      : `未修改 · 当前配置：${activeDetectionProfile}`}
              </span>
              <button
                className={`console-button ${configDialogDirty ? "primary dialog-save-button" : "dialog-close-button"}`}
                disabled={dialogSaving}
                onClick={() => void saveConfigDialog("class-config")}
                type="button"
              >
                {configDialogDirty ? "保存并关闭" : "关闭"}
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

      {modelManagerDialogOpen ? (
        <Suspense fallback={<ModelManagerLoadingDialog onClose={closeModelManager} />}>
          <ModelManagerDialog
            activeModelName={activeModelName}
            onClose={closeModelManager}
            open
            panelProps={{
          root: modelCatalog,
          loading: modelCatalogLoading,
          directoryCount: modelCatalogDirectoryCount,
          modelCount: modelCatalogModelCount,
          selectedPath: selectedModelCatalogPath,
          selectedModel: selectedCatalogModel,
          selectedArtifact: selectedPreviewArtifact,
          selectedVersion: selectedPreviewVersion,
          activeArtifactId: artifact?.id ?? null,
          activeArtifactPath: artifact?.path ?? "",
          runtimeBackend: readString(runtime?.inference?.selected, ""),
          runtimeInputShape: displayedInputShape,
          catalogMessage: modelCatalogMessage,
          switchMessage: modelSwitchMessage,
          busy,
          canSwitch: selectedCatalogModel?.kind === "engine",
          parserPreset,
          onParserPresetChange: setParserPreset,
          onRefresh: () => void refreshModelCatalog(),
          onSelectModel: selectModelFromCatalog,
          onSaveMetadata: (recommendation, tags) => void saveModelMetadata(recommendation, tags),
          onSwitch: () => void switchModel()
            }}
          />
        </Suspense>
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

      <ActionConfirmationDialog
        busy={confirmationBusy}
        error={confirmationError}
        onCancel={() => {
          if (!confirmationBusy) setConfirmationRequest(null);
        }}
        onConfirm={() => void confirmPendingAction()}
        request={confirmationRequest}
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
  optimistic = true,
  onToggle
}: {
  label: string;
  detail: string;
  enabled: boolean;
  disabled?: boolean;
  optimistic?: boolean;
  onToggle: (enabled: boolean) => Promise<void | boolean> | void | boolean;
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
    if (optimistic) {
      setVisualEnabled(next);
    }
    setPending(true);
    try {
      const applied = await onToggle(next);
      if (!optimistic && applied !== false) {
        setVisualEnabled(next);
      }
    } catch {
      setVisualEnabled(enabled);
    } finally {
      setPending(false);
    }
  }, [disabled, enabled, onToggle, optimistic, pending, visualEnabled]);

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
      <i>{pending ? "处理中" : visualEnabled ? "开" : "关"}</i>
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
  roiSize,
  previewFps
}: {
  supported: boolean;
  active: boolean;
  togglePending: boolean;
  onToggle: (enabled: boolean) => void;
  imageAvailable: boolean;
  unavailableReason: string;
  runtime: RuntimeState | null;
  roiSize: number;
  previewFps: number;
}) {
  const previewRef = useRef<HTMLDivElement | null>(null);
  const [streamFailure, setStreamFailure] = useState(false);
  const [streamRetryAttempt, setStreamRetryAttempt] = useState(0);
  const [streamRetryKey, setStreamRetryKey] = useState(0);
  const [transportFps, setTransportFps] = useState(previewFps);
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
    ?? readNullableNumber(mouseObservation.predicted_x_px)
    ?? readNullableNumber(rawAim.aim_roi_x_px)
    ?? targetCx;
  const targetAimY = readNullableNumber(mouseObservation.predicted_aim_y_roi_px)
    ?? readNullableNumber(mouseObservation.predicted_y_px)
    ?? readNullableNumber(rawAim.aim_roi_y_px)
    ?? targetCy;
  const showImage = supported && active && imageAvailable && !streamFailure;
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

  useEffect(() => {
    setStreamFailure(false);
    setStreamRetryAttempt(0);
    setStreamRetryKey(0);
    setTransportFps(previewFps);
  }, [active, configVersion, previewFps]);

  useEffect(() => {
    if (!streamFailure || !supported || !active) {
      return undefined;
    }
    const delayMs = Math.min(1000 * 2 ** Math.max(0, streamRetryAttempt - 1), 8000);
    const timer = window.setTimeout(() => {
      setStreamFailure(false);
      setStreamRetryKey((current) => current + 1);
    }, delayMs);
    return () => window.clearTimeout(timer);
  }, [active, streamFailure, streamRetryAttempt, supported]);

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
          <span>实时预览</span>
          <div className="console-preview-rate-control" role="group" aria-label="远程预览省流档位">
            {[5, 10, 15, 30].filter((fps) => fps <= previewFps).map((fps) => (
              <button
                aria-pressed={transportFps === fps}
                className={transportFps === fps ? "active" : ""}
                key={fps}
                onClick={() => setTransportFps(fps)}
                type="button"
              >
                {fps}
              </button>
            ))}
          </div>
          <button disabled={togglePending} onClick={() => onToggle(false)} type="button">
            {togglePending ? "正在关闭…" : "关闭预览"}
          </button>
        </div>
      ) : null}
      <div className="console-preview-frame">
        {showImage ? (
          <img
            alt="实时画面 / ROI"
            decoding="async"
            height={previewHeight}
            onError={() => {
              setStreamFailure(true);
              setStreamRetryAttempt((current) => current + 1);
            }}
            onLoad={() => {
              setStreamFailure(false);
              setStreamRetryAttempt(0);
            }}
            src={streamUrl(configVersion + streamRetryKey, configVersion, transportFps)}
            width={previewWidth}
          />
        ) : null}
        {supported && active && !showImage ? (
          <div className="console-preview-unavailable" role="status">
            {streamFailure
              ? `预览连接中断，正在自动重试（第 ${streamRetryAttempt} 次）`
              : unavailableReason}
          </div>
        ) : null}
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
