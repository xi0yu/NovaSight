import {
  ChangeEvent,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent
} from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureSelectPayload,
  getApiErrorCode,
  connectKmNet,
  diagnosticMoveKmNet,
  disconnectKmNet,
  clearCrosshairTemplate,
  crosshairTemplatePreviewUrl,
  getConfigSchema,
  getRuntimeConfig,
  learnCrosshair,
  HealthResponse,
  LicenseStatus,
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelCatalogResponse,
  ModelProject,
  ModelVersion,
  ParserPresetId,
  RuntimeConfig,
  ConfigSchemaResponse,
  RuntimeConfigValue,
  RuntimeState,
  RuntimeStatusTopic,
  type RuntimeVisionDetectionState,
  type RuntimeVisionTargetState,
  getCaptureCapabilities,
  getModelArtifacts,
  getModelCatalog,
  getModelVersions,
  selectCaptureProfile,
  setRuntimeOutputGate,
  setCapturePreviewEnabled,
  streamUrl,
  updateRuntimeConfig,
} from "../../api";
import { reportError, reportInfo, reportSuccess, useClearErrorNotices, useErrorNotices } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";
import { useStableSemanticValue } from "../shared/useStableSemanticValue";
import { NovaIcon, ThemeToggle } from "../../components/visual";
import { CurrentModelSummary } from "../models/CurrentModelSummary";
import { ModelSwitchDialog } from "../models/ModelSwitchDialog";
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
import { ControlTracePanel } from "./ControlTracePanel";
import { LaunchReadinessPanel } from "./LaunchReadinessPanel";
import { ProductConfigProfilePanel } from "./ProductConfigProfilePanel";
import {
  InlineNumberControl,
  InlineTextControl,
  ParameterNumberControl,
  ParameterPresetControl,
  SelectControl,
  TextControl
} from "./StudioControls";
import {
  buildAlgorithmParameterGroups,
  buildConfigFieldIndex,
  buildTargetingParameterGroups,
  validateStudioConfigSchema,
  type AlgorithmSettingsSection,
  type AlgorithmNumberParameter,
  type ControlPipelineField,
  type TargetingNumberParameter,
  type TargetingPipelineField
} from "./algorithmParameterModel";
import { CONSOLE_PAGES, DEFAULT_CONSOLE_PAGE, StudioNavigation, type ConsolePage } from "./StudioNavigation";
import { StudioPageHeader } from "./StudioPageHeader";
import { KvCard, Metric, SectionTitle, WorkspaceNotice } from "./StudioPresentation";
import { StudioRuntimeBar } from "./StudioRuntimeBar";
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "./dialogFocus";
import {
  buildLaunchReadiness,
  type LaunchReadinessAction
} from "./launchReadiness";
import { useMainlineLaunch } from "./useMainlineLaunch";
import { useConfigApplyPresentation } from "./useConfigApplyPresentation";
import { useModelSwitchWorkflow } from "./useModelSwitchWorkflow";
import { RuntimeOverviewView } from "./RuntimeOverviewView";
import {
  projectRuntimeState,
  type RuntimeRecoveryAction,
  type RuntimeTransportConfidence,
} from "../runtime/runtimeProjection";
import { buildControlTrace } from "./controlTrace";
import {
  buildProductConfigProfile,
  type ProductConfigAction
} from "./productConfigProfile";
import { persistRuntimeConfigField } from "./runtimeConfigPersistence";
import "./studio-settings.css";

const DEFAULT_CONTROL_ALGORITHM = "continuous_atan_medoid_v2";
const CONTROL_ALGORITHM_LABEL = "连续 Atan 控制";
const CONFIG_SCHEMA_CONTRACT_ERROR_PREFIX = "配置 schema 与 Studio 参数不一致";
type KmnetTestMessageTone = "success" | "warning";
type ParameterPageFieldChange = {
  section: "control" | "pipeline";
  key: string;
  value: RuntimeConfigValue;
};
const loadModelManagerDialog = () => import("../models/ModelManagerDialog");

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [stableValue, setStableValue] = useState(value);

  useEffect(() => {
    if (Object.is(stableValue, value)) {
      return;
    }

    if (delayMs <= 0) {
      setStableValue(value);
      return;
    }

    const timeout = window.setTimeout(() => {
      setStableValue(value);
    }, delayMs);

    return () => window.clearTimeout(timeout);
  }, [value, stableValue, delayMs]);

  return stableValue;
}

const ModelManagerDialog = lazy(() =>
  loadModelManagerDialog().then((module) => ({
    default: module.ModelManagerDialog
  }))
);
const ModelWorkspace = lazy(() =>
  import("../models/ModelWorkspace").then((module) => ({ default: module.ModelWorkspace }))
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
              <span className="class-config-eyebrow">模型管理</span>
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
  license: LicenseStatus | null;
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
  onRuntimeStateChange: (runtime: RuntimeState) => boolean;
  onStatusTopicChange: (topic: RuntimeStatusTopic) => void;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type ConfigDialogId = "class-config" | "target-weights" | "algorithm" | "target-advanced" | "tracker";
const ALGORITHM_SETTINGS_SECTIONS: Array<{
  id: AlgorithmSettingsSection;
  label: string;
  panelId: string;
}> = [
  {
    id: "response",
    label: "控制响应",
    panelId: "algorithm-settings-response"
  },
  {
    id: "prediction",
    label: "目标速度预测",
    panelId: "algorithm-settings-prediction"
  },
  {
    id: "calibration",
    label: "控制标定",
    panelId: "algorithm-settings-calibration"
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

const ARTIFACT_KIND_RANK: Record<string, number> = {
  engine: 0,
  onnx: 1
};
const EMPTY_RECORD: Record<string, unknown> = Object.freeze({});
const EMPTY_AIM_ROLES: Record<string, AimRole> = Object.freeze({});

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : EMPTY_RECORD;
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

function readNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readNullableNumber(value: unknown): number | null {
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
    TARGET_REQUIRED: "等待有效目标",
    INTERVAL_PENDING: "等待压枪间隔",
    OUTPUT_SATURATED: "设备范围已饱和"
  };
  return labels[reason] ?? (reason || "等待真实左键或有效目标");
}

function formatRecoilState(stateValue: unknown, reasonValue: unknown): string {
  const state = readString(stateValue);
  const completedLabels: Record<string, string> = {
    READY: "本轮压枪已就绪",
    APPLIED: "本轮压枪已叠加"
  };
  if (completedLabels[state]) {
    return completedLabels[state];
  }
  const reason = readString(reasonValue);
  if (reason) {
    return formatRecoilBlockReason(reason);
  }
  return state === "IDLE" ? "待机" : state || NO_SAMPLE;
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
    return "直接触发";
  }
  return "按键触发";
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

function formatResponseStage(value: unknown): string {
  switch (readString(value, "")) {
    case "CONTINUOUS":
      return "连续";
    default:
      return NO_SAMPLE;
  }
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

function parameterPageFieldChanges(
  baseline: RuntimeConfig,
  draft: RuntimeConfig
): ParameterPageFieldChange[] {
  const allowedSections = new Set(["control", "pipeline"]);
  const ignoredSections = new Set(["revision", "version", "roi_size"]);
  const unsupportedSections = Array.from(new Set([...Object.keys(baseline), ...Object.keys(draft)]))
    .filter((section) => !ignoredSections.has(section))
    .filter((section) => !allowedSections.has(section))
    .filter((section) => !runtimeConfigValuesEqual(baseline[section], draft[section]));
  if (unsupportedSections.length > 0) {
    throw new Error(`参数页包含不支持热更新的配置区：${unsupportedSections.join("、")}`);
  }

  const baselineControl = asRecord(baseline.control);
  const draftControl = asRecord(draft.control);
  const supportedControlKeys = new Set(["trigger_mode", "recoil"]);
  const unsupportedControlKeys = Array.from(new Set([
    ...Object.keys(baselineControl),
    ...Object.keys(draftControl)
  ]))
    .filter((key) => !supportedControlKeys.has(key))
    .filter((key) => !runtimeConfigValuesEqual(baselineControl[key], draftControl[key]));
  if (unsupportedControlKeys.length > 0) {
    throw new Error(`参数页包含不支持热更新的控制字段：${unsupportedControlKeys.join("、")}`);
  }

  const triggerChange = runtimeConfigValuesEqual(
    baselineControl.trigger_mode,
    draftControl.trigger_mode
  )
    ? null
    : {
        section: "control" as const,
        key: "trigger_mode",
        value: draftControl.trigger_mode as RuntimeConfigValue
      };
  const recoilChange = runtimeConfigValuesEqual(baselineControl.recoil, draftControl.recoil)
    ? null
    : {
        section: "control" as const,
        key: "recoil",
        value: draftControl.recoil as RuntimeConfigValue
      };

  const baselinePipeline = asRecord(baseline.pipeline);
  const draftPipeline = asRecord(draft.pipeline);
  const pipelineChanges = Array.from(new Set([
    ...Object.keys(baselinePipeline),
    ...Object.keys(draftPipeline)
  ]))
    .sort()
    .filter((key) => !runtimeConfigValuesEqual(baselinePipeline[key], draftPipeline[key]))
    .map((key) => ({
      section: "pipeline" as const,
      key,
      value: draftPipeline[key] as RuntimeConfigValue
    }));

  const changes: ParameterPageFieldChange[] = [];
  // Entering hardware-gated mode closes the live path before any less
  // restrictive algorithm edits are installed. Returning to direct trigger
  // happens last, after the rest of the saved parameter set is effective.
  if (triggerChange?.value === "hardware") {
    changes.push(triggerChange);
  }
  changes.push(...pipelineChanges);
  if (recoilChange) {
    changes.push(recoilChange);
  }
  if (triggerChange?.value !== "hardware" && triggerChange) {
    changes.push(triggerChange);
  }
  return changes;
}

function applyParameterPageFieldChanges(
  config: RuntimeConfig,
  changes: ParameterPageFieldChange[]
): RuntimeConfig {
  const next = normalizeRuntimeConfig(config);
  for (const change of changes) {
    next[change.section] = {
      ...asRecord(next[change.section]),
      [change.key]: structuredClone(change.value)
    } as RuntimeConfig[string];
  }
  return next;
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
  server: "NovaSight 服务"
};

function changedRuntimeConfigSections(current: RuntimeConfig, candidate: RuntimeConfig): string[] {
  const ignored = new Set(["revision", "version", "roi_size"]);
  return Array.from(new Set([...Object.keys(current), ...Object.keys(candidate)]))
    .filter((key) => !ignored.has(key) && !runtimeConfigValuesEqual(current[key], candidate[key]))
    .map((key) => CONFIG_SECTION_LABELS[key] ?? key);
}

export function StudioConsoleView({
  license,
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
      if (activePage === "infer") void loadModelManagerDialog();
      return;
    }
    if (activePage === "models") {
      void import("../models/ModelWorkspace");
      void onEnsureProjects(false).catch(() => undefined);
    }
    onStatusTopicChange("summary");
  }, [activePage, onEnsureProjects, onStatusTopicChange]);
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
  const [selectedModelCatalogPath, setSelectedModelCatalogPath] = useState<string>();
  const [kmnetTestDx, setKmnetTestDx] = useState(10);
  const [kmnetTestDy, setKmnetTestDy] = useState(0);
  const [kmnetTestMessage, setKmnetTestMessage] = useState("");
  const [kmnetTestMessageTone, setKmnetTestMessageTone] = useState<KmnetTestMessageTone>("warning");
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [errorCenterOpen, setErrorCenterOpen] = useState(false);
  const errorNotices = useErrorNotices();
  const clearErrorNotices = useClearErrorNotices();
  const [modelManagerDialogOpen, setModelManagerDialogOpen] = useState(false);
  const [modelCatalogMessage, setModelCatalogMessage] = useState("");
  const [classConfigDialogOpen, setClassConfigDialogOpen] = useState(false);
  const [targetWeightsDialogOpen, setTargetWeightsDialogOpen] = useState(false);
  const [algorithmSettingsDialogOpen, setAlgorithmSettingsDialogOpen] = useState(false);
  const [algorithmSettingsSection, setAlgorithmSettingsSection] = useState<AlgorithmSettingsSection>("response");
  const [targetAdvancedDialogOpen, setTargetAdvancedDialogOpen] = useState(false);
  const [trackerSettingsDialogOpen, setTrackerSettingsDialogOpen] = useState(false);
  const [confirmationRequest, setConfirmationRequest] = useState<ActionConfirmationRequest | null>(null);
  const [confirmationBusy, setConfirmationBusy] = useState(false);
  const [confirmationError, setConfirmationError] = useState<string | null>(null);
  const [newClassProfileName, setNewClassProfileName] = useState("");
  const [renamedClassProfileName, setRenamedClassProfileName] = useState("");
  const [classProfileDeleteArmed, setClassProfileDeleteArmed] = useState(false);
  const [crosshairMessage, setCrosshairMessage] = useState("");
  const [crosshairPreviewKey, setCrosshairPreviewKey] = useState(0);
  const [previewActiveOverride, setPreviewActiveOverride] = useState<boolean | null>(null);
  const [previewTogglePending, setPreviewTogglePending] = useState(false);
  const [configDraft, setConfigDraft] = useState<RuntimeConfig | null>(() => cloneRuntimeConfig(runtimeConfig));
  // Tracks the id of a control currently in an interactive edit (slider drag,
  // stepper spin, text input). When non-null, the runtimeConfig -> configDraft
  // sync effect short-circuits to avoid stealing the user's draft value during
  // high-frequency WebSocket partial frames.
  const [draggingControlId, setDraggingControlId] = useState<string | null>(null);
  const handleParameterEditingChange = useCallback((editing: boolean) => {
    setDraggingControlId(editing ? "__editing__" : null);
  }, []);
  const [configSchema, setConfigSchema] = useState<ConfigSchemaResponse | null>(null);
  const [configDialogDirty, setConfigDialogDirty] = useState(false);
  const configDialogSaving = false;
  const dialogSaving = false;
  const [parameterPageDirty, setParameterPageDirty] = useState(false);
  const [parameterPageSaving, setParameterPageSaving] = useState(false);
  const [dialogSaveError, setDialogSaveError] = useState<string | null>(null);
  const configFieldIndex = useMemo(() => buildConfigFieldIndex(configSchema), [configSchema]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const configDraftRef = useRef<RuntimeConfig | null>(cloneRuntimeConfig(runtimeConfig));
  const runtimeConfigLatestRef = useRef<RuntimeConfig | null>(runtimeConfig);
  const activeConfigDialogRef = useRef<ConfigDialogId | null>(null);
  const configDialogBaselineRef = useRef<RuntimeConfig | null>(null);
  const parameterPageBaselineRef = useRef<RuntimeConfig | null>(null);
  const scrollStudioToTop = useCallback(() => {
    window.requestAnimationFrame(() => {
      mainRef.current?.scrollTo({ top: 0, left: 0 });
      window.scrollTo({ top: 0, left: 0 });
    });
  }, []);
  const parameterPageDirtyRef = useRef(false);
  const confirmationBusyRef = useRef(false);
  // Per-call AbortController for kmNet connect/disconnect. If the user
  // rapidly toggles "连接" / "断开", the previous in-flight call is
  // aborted so the backend doesn't apply a stale request.
  const kmnetAbortControllerRef = useRef<AbortController | null>(null);
  const loadedModelProjectIdRef = useRef<number | "">("");
  const loadedModelVersionIdRef = useRef<number | "">("");
  const pendingConfigWritesRef = useRef(0);
  const [pendingConfigWriteCount, setPendingConfigWriteCount] = useState(0);
  const beginPendingConfigWrite = useCallback(() => {
    pendingConfigWritesRef.current += 1;
    setPendingConfigWriteCount(pendingConfigWritesRef.current);
  }, []);
  const finishPendingConfigWrite = useCallback(() => {
    pendingConfigWritesRef.current = Math.max(0, pendingConfigWritesRef.current - 1);
    setPendingConfigWriteCount(pendingConfigWritesRef.current);
  }, []);
  const configWriteSeqRef = useRef(0);
  const configWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const finalizeRuntimeConfigWrite = useCallback(
    (config: RuntimeConfig) => {
      const incomingRevision = readNumber(config.revision, 0);
      const currentRevision = readNumber(runtimeConfigLatestRef.current?.revision, 0);
      if (incomingRevision < currentRevision) {
        // Runtime config writes are serialized. Revision order, not request
        // start order, determines whether a response is stale.
        return;
      }
      runtimeConfigLatestRef.current = config;
      configDraftRef.current = config;
      setConfigDraft(config);
      onRuntimeConfigChange(config);
    },
    [onRuntimeConfigChange]
  );
  const classConfigDialogRef = useRef<HTMLElement | null>(null);
  const targetWeightsDialogRef = useRef<HTMLElement | null>(null);
  const errorCenterDialogRef = useRef<HTMLElement | null>(null);
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

  const setParameterPageDirtyState = useCallback((dirty: boolean) => {
    parameterPageDirtyRef.current = dirty;
    setParameterPageDirty(dirty);
  }, []);

  const stageParameterPageDraft = useCallback((next: RuntimeConfig) => {
    if (!parameterPageBaselineRef.current) {
      const baseline = cloneRuntimeConfig(runtimeConfigLatestRef.current);
      parameterPageBaselineRef.current = baseline;
    }
    configDraftRef.current = next;
    setConfigDraft(next);
    setParameterPageDirtyState(
      !runtimeConfigsEqual(parameterPageBaselineRef.current, next)
    );
    setDialogSaveError(null);
  }, [setParameterPageDirtyState]);

  const discardParameterPageDraft = useCallback(() => {
    const latest = cloneRuntimeConfig(runtimeConfigLatestRef.current);
    configDraftRef.current = latest;
    setConfigDraft(latest);
    parameterPageBaselineRef.current = null;
    setParameterPageDirtyState(false);
    setDialogSaveError(null);
  }, [setParameterPageDirtyState]);

  const focusAlgorithmSettingsSection = useCallback((section: AlgorithmSettingsSection) => {
    setAlgorithmSettingsSection(section);
    window.requestAnimationFrame(() => {
      document.getElementById(`algorithm-settings-${section}-tab`)?.focus();
    });
  }, []);

  const handleAlgorithmSettingsTabKeyDown = useCallback((event: ReactKeyboardEvent<HTMLButtonElement>) => {
    const currentIndex = ALGORITHM_SETTINGS_SECTIONS.findIndex((section) => section.id === algorithmSettingsSection);
    if (currentIndex < 0) {
      return;
    }
    const lastIndex = ALGORITHM_SETTINGS_SECTIONS.length - 1;
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = currentIndex === lastIndex ? 0 : currentIndex + 1;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = currentIndex === 0 ? lastIndex : currentIndex - 1;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = lastIndex;
    } else {
      return;
    }
    event.preventDefault();
    focusAlgorithmSettingsSection(ALGORITHM_SETTINGS_SECTIONS[nextIndex].id);
  }, [algorithmSettingsSection, focusAlgorithmSettingsSection]);

  const applyConfigSchema = useCallback((schema: ConfigSchemaResponse) => {
    const issues = validateStudioConfigSchema(schema);
    setConfigSchema(schema);
    if (issues.length > 0) {
      const summary = issues
        .slice(0, 3)
        .map((issue) => `${issue.path}: ${issue.reason}`)
        .join("；");
      setLocalError(`${CONFIG_SCHEMA_CONTRACT_ERROR_PREFIX}：${summary}`);
      return;
    }
    setLocalError((current) =>
      current?.startsWith(CONFIG_SCHEMA_CONTRACT_ERROR_PREFIX) ? null : current
    );
  }, []);

  useEffect(() => {
    let cancelled = false;
    void getConfigSchema()
      .then((schema) => {
        if (cancelled) {
          return;
        }
        applyConfigSchema(schema);
      })
      .catch((error) => {
        if (!cancelled) {
          reportError(error, { source: "config-schema", title: "配置 schema 读取失败" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [applyConfigSchema]);

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
    if (dialog === "algorithm") {
      setAlgorithmSettingsSection("response");
    }
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
      configDraftRef.current = baseline ?? draft;
      setConfigDraft(baseline ?? draft);
      finishConfigDialog(dialog);
      return;
    }

    stageParameterPageDraft(draft);
    finishConfigDialog(dialog);
    reportInfo(
      "修改已加入参数草稿",
      "这些参数尚未写入配置；请在参数设置页面点击“保存修改”。",
      "config-dialog"
    );
  }, [finishConfigDialog, stageParameterPageDraft]);

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
      configDraftRef.current = baseline ?? draft;
      setConfigDraft(baseline ?? draft);
      finishConfigDialog(dialog);
      return;
    }

    // The explicit "加入草稿并关闭" button keeps the dialog edits; closing via Esc/backdrop
    // is the explicit "放弃" path. Surface a non-blocking info toast so the
    // user can immediately reopen if they change their mind.
    configDraftRef.current = baseline;
    setConfigDraft(baseline);
    finishConfigDialog(dialog);
    reportInfo("已放弃修改", "本弹窗内的草稿已丢弃，可重新打开继续编辑。", "config-dialog");
    return;
  }, [finishConfigDialog]);

  const confirmPendingAction = useCallback(async () => {
    const request = confirmationRequest;
    // Synchronous mutex: state-based `confirmationBusy` only updates on the
    // next React commit, so a double-tap inside the same event loop tick can
    // see `false` both times. The ref gives us a true pre-commit guard.
    if (!request || confirmationBusyRef.current) return;
    confirmationBusyRef.current = true;
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
      confirmationBusyRef.current = false;
      setConfirmationBusy(false);
    }
  }, [confirmationRequest]);

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
    const onPopState = () => {
      const nextPage = pageFromUrl();
      if (
        parameterPageDirtyRef.current
        && nextPage !== "params"
        && !window.confirm("参数设置中还有未保存修改。离开页面将放弃这些修改，确定继续吗？")
      ) {
        writePageToUrl("params", "replace");
        return;
      }
      if (parameterPageDirtyRef.current && nextPage !== "params") {
        discardParameterPageDraft();
      }
      setActivePage(nextPage);
      scrollStudioToTop();
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [discardParameterPageDraft, scrollStudioToTop]);

  const navigatePage = useCallback((page: ConsolePage) => {
    if (
      activePage === "params"
      && page !== "params"
      && parameterPageDirtyRef.current
      && !window.confirm("参数设置中还有未保存修改。离开页面将放弃这些修改，确定继续吗？")
    ) {
      return;
    }
    if (activePage === "params" && page !== "params" && parameterPageDirtyRef.current) {
      discardParameterPageDraft();
    }
    setActivePage(page);
    writePageToUrl(page);
    scrollStudioToTop();
  }, [activePage, discardParameterPageDraft, scrollStudioToTop]);

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!parameterPageDirtyRef.current) {
        return;
      }
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, []);

  useEffect(() => {
    if (!errorCenterOpen) {
      return undefined;
    }
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    // Save the frame id and cancel on cleanup so a fast close (Escape or
    // backdrop click immediately after open) doesn't yank focus back into the
    // just-closed dialog on the next frame.
    const focusFrame = window.requestAnimationFrame(() => errorCenterDialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setErrorCenterOpen(false);
      } else {
        trapDialogTabKey(event, errorCenterDialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [errorCenterOpen]);

  useEffect(() => {
    if (!classConfigDialogOpen) {
      return undefined;
    }
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusFrame = window.requestAnimationFrame(() => classConfigDialogRef.current?.focus());
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
      window.cancelAnimationFrame(focusFrame);
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [classConfigDialogOpen, requestDismissConfigDialog]);

  useEffect(() => {
    return () => {
      // Abort any in-flight kmNet connect/disconnect on unmount so a
      // navigated-away component doesn't keep hitting the backend.
      kmnetAbortControllerRef.current?.abort();
      kmnetAbortControllerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!targetWeightsDialogOpen) {
      return undefined;
    }
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusFrame = window.requestAnimationFrame(() => targetWeightsDialogRef.current?.focus());
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
      window.cancelAnimationFrame(focusFrame);
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [requestDismissConfigDialog, targetWeightsDialogOpen]);

  const capture = runtime?.capture;
  const statistics = runtime?.statistics;
  const config = configDraft ?? runtimeConfig;
  const {
    captureConfig,
    roiConfig,
    crosshairConfig,
    limitsConfig,
    inferenceConfig,
    controlConfig,
    rustPipelineConfig,
    hardwareConfig,
    consumersConfig
  } = useMemo(() => ({
    captureConfig: nestedRecord(config, "capture"),
    roiConfig: nestedRecord(config, "roi"),
    crosshairConfig: nestedRecord(config, "crosshair"),
    limitsConfig: nestedRecord(config, "limits"),
    inferenceConfig: nestedRecord(config, "inference"),
    controlConfig: nestedRecord(config, "control"),
    rustPipelineConfig: nestedRecord(config, "pipeline"),
    hardwareConfig: nestedRecord(config, "hardware"),
    consumersConfig: nestedRecord(config, "consumers")
  }), [config]);
  const configuredCaptureDevice = readString(captureConfig.device, "");
  const configuredCapturePixelFormat = readString(captureConfig.pixel_format, "");
  const configuredCaptureWidth = readNumber(captureConfig.width, 0);
  const configuredCaptureHeight = readNumber(captureConfig.height, 0);
  const configuredCaptureFps = readNumber(captureConfig.fps, 0);
  const configuredRoiLeft = readNumber(captureConfig.roi_left, 0);
  const configuredRoiTop = readNumber(captureConfig.roi_top, 0);
  const configuredRoiWidth = readNumber(captureConfig.roi_width, 0);
  const configuredRoiHeight = readNumber(captureConfig.roi_height, 0);
  const rustControlPlane = Object.prototype.hasOwnProperty.call(
    rustPipelineConfig,
    "projection_fov_x_deg"
  );
  const vision = runtime?.vision;
  const crosshairStatus = vision?.crosshair;
  const crosshairObservation = crosshairStatus?.observation;
  const crosshairTemplate = crosshairStatus?.template;
  const inferenceTrace = vision?.inference;
  const runtimeInference = runtime?.inference;
  const deepstreamStatus = runtime?.pipeline.deepstream;
  const configuredInferenceBackend = readString(inferenceConfig.backend, "deepstream_nvinfer");
  const selectedRuntimeBackend = runtimeInference?.selected ?? configuredInferenceBackend;
  const deepstreamNvinferSelected = selectedRuntimeBackend === "deepstream_nvinfer";
  const runtimeMainlineStatus = getRuntimeMainlineStatus(runtime);
  const runtimeMainlinePresentation = useStableSemanticValue(
    runtimeMainlineStatus,
    `${runtime?.semantic?.phase ?? runtimeMainlineStatus.readinessCode}:${runtime?.semantic?.perception_phase ?? "unknown"}:${runtime?.semantic?.epoch ?? "none"}`,
    280
  );
  const runtimeOutputTrace = runtimeMainlinePresentation.outputTrace;
  const stableRuntimeOutputTrace = useStableSemanticValue(
    runtimeOutputTrace,
    runtimeOutputTrace?.code ?? "unavailable",
    220
  );
  const runtimeInferenceConfigured = runtimeInference?.configured === true;
  const runtimeMainlineRunning = runtimeMainlineStatus.running;
  const runtimePostprocess = runtimeInference?.postprocess;
  const kmnetStatus = runtime?.executor.executors.kmnet;
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
  const crosshairState = crosshairStatus?.state ?? "idle";
  const stableCrosshairState = useStableSemanticValue(crosshairState, crosshairState, 280);
  const crosshairTemplateId = crosshairTemplate?.id ?? "";
  const crosshairRecentSamples = crosshairStatus?.recent_samples ?? 0;
  const crosshairRequiredSamples = crosshairStatus?.required_samples ?? crosshairSampleFrames;
  const crosshairConfidence = crosshairObservation?.confidence ?? 0;
  const crosshairOffsetX = crosshairObservation?.offset_x ?? 0;
  const crosshairOffsetY = crosshairObservation?.offset_y ?? 0;
  const crosshairReferenceReady = crosshairStatus?.control_reference_ready === true;
  const crosshairBranchActive = deepstreamStatus?.crosshair_active === true;
  const crosshairBranchReason = deepstreamStatus?.crosshair_reason ?? "";
  const crosshairProcessingError = crosshairStatus?.last_error ?? "";
  const previewFps = readNumber(limitsConfig.stream_fps, 30);
  const sourceWidth = selectedProfile?.width ?? inferenceTrace?.source_width ?? 0;
  const sourceHeight = selectedProfile?.height ?? inferenceTrace?.source_height ?? 0;
  const roiX = rustControlPlane
    ? configuredRoiLeft
    : sourceWidth > 0 ? Math.max(0, Math.floor((sourceWidth - roiSize) / 2)) : 0;
  const roiY = rustControlPlane
    ? configuredRoiTop
    : sourceHeight > 0 ? Math.max(0, Math.floor((sourceHeight - roiSize) / 2)) : 0;
  const confidence = readNumber(inferenceConfig.confidence_threshold, 0.25);
  const nms = readNumber(inferenceConfig.nms_threshold, 0.45);
  const detectionProfiles = useMemo(
    () => recordList(inferenceConfig.detection_class_profiles),
    [inferenceConfig.detection_class_profiles]
  );
  const activeDetectionProfile = readString(inferenceConfig.detection_class_profile, "default");
  const activeDetectionClass = readString(
    rustControlPlane
      ? rustPipelineConfig.target_class_filter
      : inferenceConfig.detection_class_filter,
    "all"
  );
  const detectionProfileNames = useMemo(() => Object.keys(detectionProfiles), [detectionProfiles]);
  const detectionClasses = detectionProfiles[activeDetectionProfile] ?? detectionProfiles.default ?? [];
  const detectionClassPriority = readString(
    rustControlPlane
      ? rustPipelineConfig.target_class_priority
      : inferenceConfig.detection_class_priority,
    "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  );
  const classPriorityIds = useMemo(
    () => parseClassPriority(detectionClassPriority),
    [detectionClassPriority]
  );
  useEffect(() => {
    setRenamedClassProfileName(activeDetectionProfile);
    setClassProfileDeleteArmed(false);
  }, [activeDetectionProfile]);
  const { aimConfig, rawAimRoleRatios, recoilConfig } = useMemo(() => {
    const aim = nestedRecord(controlConfig, "aim");
    return {
      aimConfig: aim,
      rawAimRoleRatios: nestedRecord(aim, "role_y_ratios"),
      recoilConfig: nestedRecord(controlConfig, "recoil")
    };
  }, [controlConfig]);
  const rustOtherAimRatio = readNumber(
    rustControlPlane ? rustPipelineConfig.target_aim_y_ratio : undefined,
    0.22
  );
  // Memoize so downstream `===` checks (e.g. onEditingChange guards, the
  // P0-A short-circuit, the Slider draftValue external-change detection) stay
  // stable across partial WebSocket frames that don't actually change aim.
  const aimRoleRatios: AimRoleRatios = useMemo(
    () => ({
      head: clampNumber(readNumber(rawAimRoleRatios.head, 0.22), 0, 1),
      body: clampNumber(readNumber(rawAimRoleRatios.body, 0.22), 0, 1),
      other: clampNumber(
        rustControlPlane
          ? rustOtherAimRatio
          : readNumber(rawAimRoleRatios.other, 0.22),
        0,
        1
      )
    }),
    [rawAimRoleRatios, rustControlPlane, rustOtherAimRatio]
  );
  const classRoleProfiles = useMemo(
    () => profileRoleRecords(aimConfig.class_roles),
    [aimConfig.class_roles]
  );
  const activeClassRoles = classRoleProfiles[activeDetectionProfile] ?? EMPTY_AIM_ROLES;
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
  const targetLostGraceMs = readNumber(rustPipelineConfig.target_track_max_lost_age_ms, 120);
  const targetSwitchPreferenceAdvantage = readNumber(rustPipelineConfig.target_switch_min_preference_advantage, 0.08);
  const targetSwitchContinuityScore = readNumber(rustPipelineConfig.target_switch_min_continuity_score, 0.7);
  const targetSwitchDelayMs = readNumber(rustPipelineConfig.target_switch_delay_ms, 50);
  const freshnessThresholdMs = readNumber(rustPipelineConfig.freshness_threshold_ms, 55);
  const controlFovX = readNumber(rustPipelineConfig.projection_fov_x_deg, 105);
  const controlCountsPer360 = readNumber(rustPipelineConfig.projection_counts_per_360, 9980);
  const pResponseScale = readNumber(rustPipelineConfig.p_response_scale, 0.20);
  const pResponseBoost = readNumber(rustPipelineConfig.p_response_boost, 0.50);
  const pResponseCurveShape = readNumber(rustPipelineConfig.p_response_curve_shape, 1);
  const maxOutputXCounts = readNumber(rustPipelineConfig.max_output_x_counts, 127);
  const maxOutputYCounts = readNumber(rustPipelineConfig.max_output_y_counts, 127);
  const controlPredictionEnabled = readBoolean(rustPipelineConfig.prediction_enabled, true);
  const controlPredictionHistoryResetGapMs = readNumber(rustPipelineConfig.velocity_history_reset_gap_ms, 80);
  const controlPredictionLeadMs = readNumber(rustPipelineConfig.prediction_lead_ms, 16);
  const controlPredictionCapPx = readNumber(rustPipelineConfig.prediction_cap_px, 10);
  const predictionActuationDelayMs = readNumber(rustPipelineConfig.prediction_actuation_delay_ms, 4);
  const targetMinConfidence = readNumber(rustPipelineConfig.target_min_confidence, 0.5);
  const trackerScaleCostWeight = readNumber(rustPipelineConfig.tracker_scale_cost_weight, 0.15);
  const trackerMaxSizeRatio = readNumber(rustPipelineConfig.tracker_max_size_ratio, 2.5);
  const trackerMaxAssociationDtMs = readNumber(rustPipelineConfig.tracker_max_association_dt_ms, 150);
  const trackerKalmanAccelerationNoise = readNumber(rustPipelineConfig.tracker_kalman_acceleration_noise, 1200);
  const trackerKalmanMeasurementNoiseX = readNumber(rustPipelineConfig.tracker_kalman_measurement_noise_x, 16);
  const trackerKalmanMeasurementNoiseY = readNumber(rustPipelineConfig.tracker_kalman_measurement_noise_y, 16);
  const trackerKalmanMaxPredictDtMs = readNumber(rustPipelineConfig.tracker_kalman_max_predict_dt_ms, 35);
  const trackerKalmanMaxPredictMissingMs = readNumber(rustPipelineConfig.tracker_kalman_max_predict_missing_ms, 80);
  const trackerKalmanMaxPredictSteps = readNumber(rustPipelineConfig.tracker_kalman_max_predict_steps, 5);
  const trackerKalmanNisThreshold = readNumber(rustPipelineConfig.tracker_kalman_nis_threshold, 9.21);
  const trackerKalmanNisHardReject = readNumber(rustPipelineConfig.tracker_kalman_nis_hard_reject, 16);
  const recoilEnabled = readBoolean(recoilConfig.enabled, false);
  const recoilRequireTarget = readBoolean(recoilConfig.require_target, true);
  const recoilIntervalMs = readNumber(recoilConfig.interval_ms, 16);
  const fireDelayEnabled = readBoolean(rustPipelineConfig.fire_delay_enabled, false);
  const fireDelayMs = readNumber(rustPipelineConfig.fire_delay_ms, 0);
  const recoilYCounts = readNumber(recoilConfig.y_counts, 1);
  const triggerMode = readString(controlConfig.trigger_mode, "always");
  const controlAlgorithmId = readString(configSchema?.algorithm?.id, DEFAULT_CONTROL_ALGORITHM);
  const controlAlgorithmLabel = readString(configSchema?.algorithm?.label, CONTROL_ALGORITHM_LABEL);
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);
  const kmnetAutoConnect = readBoolean(hardwareConfig.auto_connect, false);
  const outputEnabled = readBoolean(controlConfig.output_enabled, false);
  const controlModeLabel = controlAlgorithmLabel;

  useEffect(() => {
    if (
      !runtimeConfig
      || pendingConfigWritesRef.current > 0
      || activeConfigDialogRef.current !== null
      || parameterPageDirtyRef.current
    ) {
      return;
    }
    // Skip syncing while the user is actively editing a parameter; the
    // per-frame partial WebSocket frames would otherwise overwrite the draft.
    if (draggingControlId !== null) {
      return;
    }
    const next = normalizeRuntimeConfig(runtimeConfig);
    const incomingRevision = readNumber(next.revision, 0);
    const currentRevision = readNumber(runtimeConfigLatestRef.current?.revision, 0);
    if (incomingRevision < currentRevision) {
      return;
    }
    runtimeConfigLatestRef.current = next;
    configDraftRef.current = next;
    // Same-value short-circuit: when the partial frame carries no real
    // configuration change, keep the existing reference so downstream
    // controlled components and memos stay stable.
    setConfigDraft((prev) => (runtimeConfigValuesEqual(prev, next) ? prev : next));
  }, [draggingControlId, pendingConfigWriteCount, runtimeConfig]);
  const kmnetConnectedRaw = kmnetStatus?.connected === true;
  const kmnetConnectingRaw = kmnetStatus?.connecting === true;
  const kmnetExecutorAvailable = kmnetStatus?.available === true;
  const kmnetConnectionStateRaw = kmnetStatus?.connection_state
    ?? (kmnetConnectedRaw ? "connected" : kmnetConnectingRaw ? "connecting" : "disconnected");
  const kmnetConnectionState = useDebouncedValue(kmnetConnectionStateRaw, 280);
  const kmnetConnected = kmnetConnectionState === "connected";
  const kmnetRuntimeConnected = kmnetConnected;
  const kmnetConnecting = kmnetConnectionState === "connecting";
  const kmnetConnectionFailed = kmnetConnectionState === "failed";
  const kmnetConnectionDegraded = kmnetConnectionState === "degraded";
  const kmnetLastError = kmnetStatus?.last_error ?? "";
  const kmnetLastDeviceError = kmnetStatus?.last_device_error ?? "";
  const reportedDesiredConfigRevision = runtime?.config.version ?? 0;
  const reportedEffectiveConfigRevision = runtime?.config.effective_version
    ?? reportedDesiredConfigRevision;
  const reportedConfigRestartRequired = runtime?.config?.restart_required === true
    || reportedDesiredConfigRevision !== reportedEffectiveConfigRevision;
  const configApplyPresentation = useConfigApplyPresentation({
    pendingWriteCount: pendingConfigWriteCount,
    restartRequired: reportedConfigRestartRequired,
    desiredRevision: reportedDesiredConfigRevision,
    effectiveRevision: reportedEffectiveConfigRevision
  });
  const configApplyPending = configApplyPresentation.state === "applying";
  const configRestartRequired = configApplyPresentation.state === "pending_process";
  const desiredConfigRevision = configApplyPresentation.desiredRevision;
  const effectiveConfigRevision = configApplyPresentation.effectiveRevision;
  const kmnetRestartRequired = kmnetStatus?.restart_required === true;
  const kmnetConfigurationState = kmnetStatus?.configuration_state
    ?? (kmnetRestartRequired ? "restart_required" : kmnetAutoConnect ? "ready" : "uncommissioned");
  const kmnetConfigurationReady = kmnetStatus?.configuration_ready === true
    || (kmnetConfigurationState === "ready" && !kmnetRestartRequired);
  const kmnetCanConnect = kmnetStatus?.can_connect === true
    || (kmnetConfigurationReady
      && runtime?.running === true
      && !kmnetRuntimeConnected
      && !kmnetConnecting);
  const kmnetCanDisconnect = kmnetStatus?.can_disconnect === true
    || (runtime?.running === true && kmnetRuntimeConnected);
  const kmnetRuntimeConnectionLabel = runtime?.running === true
    ? kmnetRestartRequired
      ? kmnetRuntimeConnected ? "旧会话仍连接" : "等待重载"
      : kmnetRuntimeConnected ? "已连接" : "未连接"
    : "主链未运行";
  const kmnetNoticeCode = kmnetRestartRequired
    ? "restart_required"
    : kmnetConnectionFailed
      ? "failed"
      : kmnetConnectionDegraded
        ? "degraded"
        : kmnetLastDeviceError || kmnetLastError
          ? "recent_error"
          : "none";
  const kmnetNotice = useStableSemanticValue(
    {
      code: kmnetNoticeCode,
      lastError: kmnetLastError,
      lastDeviceError: kmnetLastDeviceError,
      desiredRevision: desiredConfigRevision,
      effectiveRevision: effectiveConfigRevision
    },
    kmnetNoticeCode,
    280
  );
  const kmnetDiagnosticDisabled = kmnetRestartRequired
    || !kmnetExecutorAvailable
    || runtime?.semantic.phase !== "stopped"
    || busy === "kmnet.diagnostic";
  const previewEnabled = consumersConfig.preview !== false;
  const activeModelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const activeModelPublished = runtime?.active_model !== null && runtime?.active_model !== undefined;
  const artifact = runtime?.active_model?.artifact;
  const version = runtime?.active_model?.version;
  const registeredInputShape = version?.input_shape === "engine-probe-required"
    ? "等待 TensorRT engine 探测"
    : version?.input_shape ?? "";
  const displayedInputShape = registeredInputShape;
  const runtimePostprocessConfidence = runtimePostprocess?.confidence_threshold ?? Number.NaN;
  const runtimePostprocessNms = runtimePostprocess?.nms_threshold ?? Number.NaN;
  const runtimeRoiLeft = inferenceTrace?.roi_offset_x ?? Number.NaN;
  const runtimeRoiTop = inferenceTrace?.roi_offset_y ?? Number.NaN;
  const runtimeRoiWidth = inferenceTrace?.roi_width ?? Number.NaN;
  const runtimeRoiHeight = inferenceTrace?.roi_height ?? Number.NaN;
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
      : configApplyPending
        ? "正在应用…"
        : "已保存 · 等待运行态确认";
  const runtimePostprocessAvailable = runtimeMainlineRunning
    && runtimeInference?.loaded === true
    && Number.isFinite(runtimePostprocessConfidence)
    && Number.isFinite(runtimePostprocessNms);
  const postprocessSettingsApplied = runtimePostprocessAvailable
    && Math.abs(runtimePostprocessConfidence - confidence) < 0.0001
    && Math.abs(runtimePostprocessNms - nms) < 0.0001;
  const postprocessApplyLabel = !runtimePostprocessAvailable
    ? "主链未装载"
    : postprocessSettingsApplied
      ? "已生效"
      : configApplyPending
        ? "正在应用…"
        : "已保存 · 等待运行态确认";
  const sortedSwitchableArtifacts = useMemo(() => modelArtifacts
    .filter(
      (item) =>
        item.kind === "engine" &&
        (item.status === "ready" || item.status === "pending" || item.status === "failed") &&
        (selectedModelVersionId === "" || item.version_id === selectedModelVersionId)
    )
    .sort((left, right) => {
      const leftRank = ARTIFACT_KIND_RANK[left.kind] ?? 99;
      const rightRank = ARTIFACT_KIND_RANK[right.kind] ?? 99;
      return leftRank - rightRank || left.path.localeCompare(right.path);
    }), [modelArtifacts, selectedModelVersionId]);
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
  const detectionCount = vision?.detections ?? null;
  const target = vision?.target;
  const activeRuntimeClassId = target?.cls ?? target?.class_id ?? null;
  const activeRuntimeClassLabel = activeRuntimeClassId !== null
    ? `cls ${Math.round(activeRuntimeClassId)}`
    : "";
  const runtimeDetectionItems = vision?.detection_items ?? [];
  const runtimeDetectionClassIds = useMemo(
    () => runtimeDetectionItems.map((item) => item.cls),
    [runtimeDetectionItems]
  );
  const classEditorIds = useMemo(() => Array.from(new Set([
    ...detectionClasses.map((_, classId) => classId),
    ...classPriorityIds,
    ...runtimeDetectionClassIds,
    ...(activeRuntimeClassId !== null ? [activeRuntimeClassId] : [])
  ])).filter((classId) => classId >= 0 && classId <= 255).sort((left, right) => left - right), [
    activeRuntimeClassId,
    classPriorityIds,
    detectionClasses,
    runtimeDetectionClassIds
  ]);
  const orderedClassEditorIds = useMemo(() => [
    ...classPriorityIds.filter((classId) => classEditorIds.includes(classId)),
    ...classEditorIds.filter((classId) => !classPriorityIds.includes(classId))
  ], [classEditorIds, classPriorityIds]);
  const configuredDetectionClassIds = useMemo(
    () => parseDetectionClassFilter(activeDetectionClass),
    [activeDetectionClass]
  );
  const selectedDetectionClassIds = useMemo(
    () => configuredDetectionClassIds ?? new Set(classEditorIds),
    [classEditorIds, configuredDetectionClassIds]
  );
  const control = vision?.control;
  const targetPipeline = vision?.target_pipeline;
  const targetPipelineCode = targetPipeline?.code ?? "";
  const targetPipelineStage = targetPipeline?.stage ?? "";
  const targetPipelineMessage = targetPipeline?.message ?? "";
  const targetPipelineRejections = targetPipeline?.rejection_reasons.join(", ") ?? "";
  const rawCandidateCount = targetPipeline?.counts.raw_candidates ?? null;
  const eligibleCandidateCount = targetPipeline?.counts.eligible_candidates ?? null;
  const selectedTargetCount = targetPipeline?.counts.selected_targets ?? null;
  const controlCandidateFilter = control?.candidate_filter;
  const basicCandidateFilter = controlCandidateFilter?.basic;
  const effectiveClassFilter = controlCandidateFilter?.effective_class_filter ?? activeDetectionClass;
  const rejectedBasicClassIds = basicCandidateFilter?.rejected_class_ids ?? [];
  const recoveryClassIdSet = useMemo(() => new Set([
    ...(configuredDetectionClassIds ?? []),
    ...runtimeDetectionClassIds
  ]), [configuredDetectionClassIds, runtimeDetectionClassIds]);
  const detectedClassFilterValue = useMemo(() => orderedClassEditorIds
    .filter((classId) => recoveryClassIdSet.has(classId))
    .join(","), [orderedClassEditorIds, recoveryClassIdSet]);
  const controlPipeline = control?.pipeline;
  const fireDelayPending = controlPipeline?.fire_delay_pending === true;
  const runtimeFireDelayMs = controlPipeline?.fire_delay_configured_ms ?? fireDelayMs;
  const fireDelayElapsedMs = controlPipeline?.fire_delay_elapsed_ms ?? null;
  const fireDelayRemainingMs = controlPipeline?.fire_delay_remaining_ms ?? null;
  const runtimeRecoilEnabled = controlPipeline?.recoil_enabled ?? null;
  const effectiveRecoilEnabled = runtimeRecoilEnabled ?? recoilEnabled;
  const mouseObservation = control?.mouse_observation;
  const controlHasSample = control?.global_state === "CALCULATED";
  const controlHasTarget = target !== null && target !== undefined;
  const controlCandidateCount = control?.candidates ?? null;
  const controlTrackId = target?.track_id ?? null;
  const controlWidthPx = mouseObservation?.control_width_px ?? null;
  const controlHeightPx = mouseObservation?.control_height_px ?? null;
  const controlCenterX = controlWidthPx === null ? null : controlWidthPx * 0.5;
  const controlCenterY = controlHeightPx === null ? null : controlHeightPx * 0.5;
  const predictedAimX = mouseObservation?.predicted_x_px ?? control?.aim_x ?? null;
  const predictedAimY = mouseObservation?.predicted_y_px ?? control?.aim_y ?? null;
  const observedAimX = mouseObservation?.observed_x_px ?? null;
  const observedAimY = mouseObservation?.observed_y_px ?? null;
  const predictedErrorXPx =
    predictedAimX !== null && controlCenterX !== null ? predictedAimX - controlCenterX : null;
  const predictedErrorYPx =
    predictedAimY !== null && controlCenterY !== null ? predictedAimY - controlCenterY : null;
  const predictedErrorDistancePx =
    predictedErrorXPx !== null && predictedErrorYPx !== null
      ? Math.hypot(predictedErrorXPx, predictedErrorYPx)
      : null;
  const controlMeasurementDtS = controlPipeline?.measurement_dt_s ?? mouseObservation?.measurement_dt_s ?? null;
  const controlMeasurementDtMs = controlMeasurementDtS === null ? null : controlMeasurementDtS * 1000;
  const controlFrameAgeMs = controlPipeline?.frame_age_ms ?? null;
  const hasAcceptedCommand = (kmnetStatus?.accepted_command_count ?? 0) > 0;
  const lastAcceptedCommand = formatPoint(
    kmnetStatus?.last_accepted_dx,
    kmnetStatus?.last_accepted_dy,
    0,
    "counts"
  );
  const controlWillEmitRaw = control?.will_emit ?? null;
  const controlWillEmit = controlWillEmitRaw;
  const controlTriggerActiveRaw = control?.trigger_active ?? null;
  const controlTriggerActive = controlTriggerActiveRaw;
  const controlNoSendReason = fireDelayPending
    ? `按键持续时间尚未超过 ${runtimeFireDelayMs.toFixed(0)} ms，控制算法未启动`
    : stableRuntimeOutputTrace?.detail || (
        !controlHasTarget
          ? targetPipelineMessage || control?.selection_reason || "无目标"
          : controlWillEmit !== true
            ? control?.no_send_reason || control?.reason || "控制门控未通过"
            : !kmnetRuntimeConnected
              ? "主链设备通道未连接"
              : "命令已获准进入设备通道"
      );
  const runtimeOutputEnabled = control?.output_enabled ?? null;
  const deepstreamInputFrames = runtimeMainlineStatus.nvinferInputFrames;
  const deepstreamOutputBuffers = runtimeInference?.output_buffers ?? null;
  const deepstreamMetadataExtractions = runtimeMainlineStatus.metadataExtractions;
  const deepstreamPublishedBatches = runtimeMainlineStatus.publishedBatches;
  const deepstreamInferenceCompleted =
    (deepstreamOutputBuffers ?? 0) > 0 ||
    (deepstreamPublishedBatches ?? 0) > 0;
  const inferenceRan = deepstreamNvinferSelected && deepstreamInferenceCompleted;
  const inferenceReason = runtimeInference?.inference_reason ?? runtimeInference?.reason ?? "";
  const deepstreamPreviewStreamReady =
    deepstreamNvinferSelected &&
    previewEnabled &&
    runtimeInference?.preview_enabled === true &&
    runtimeInference.loaded === true &&
    runtimeInference.terminal_error !== true;
  const runtimePreviewActive = runtimeInference?.preview_active === true;
  const previewRuntimePresentation = {
    streamReady: deepstreamPreviewStreamReady,
    active: runtimePreviewActive,
    reason: runtimeInference?.preview_reason ?? "等待 DeepStream 硬件预览帧"
  };
  const previewActive = previewActiveOverride ?? previewRuntimePresentation.active;
  const previewImageAvailable = deepstreamNvinferSelected
    ? previewRuntimePresentation.streamReady && previewActive
    : runtime?.capture?.available === true;
  const previewUnavailableReason = deepstreamNvinferSelected
    ? previewRuntimePresentation.reason
    : "预览帧尚不可用";

  useEffect(() => {
    if (previewActiveOverride === previewRuntimePresentation.active) {
      setPreviewActiveOverride(null);
    }
  }, [previewActiveOverride, previewRuntimePresentation.active]);

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
  const controlTrace = activePage === "control" ? buildControlTrace({
    runtimeRunning: runtimeMainlineRunning,
    detectionBatchFps,
    publishedBatches: runtimeMainlineStatus.publishedBatches,
    consumedBatches: runtimeMainlineStatus.consumedBatches,
    targetingBatches: runtimeMainlineStatus.targetingBatches,
    detectionDataAgeMs,
    freshnessThresholdMs: detectionFreshnessThresholdMs,
    detectionCount,
    rawCandidateCount,
    eligibleCandidateCount,
    selectedTargetCount,
    targetPipelineStage,
    targetPipelineCode,
    targetPipelineMessage,
    targetPipelineRejections,
    hasTarget: controlHasTarget,
    trackId: controlTrackId,
    classLabel: activeRuntimeClassLabel,
    targetScore: target?.score ?? null,
    observedAim: formatPoint(observedAimX, observedAimY, STANDARD_DECIMAL_DIGITS, "px"),
    controlAim: formatPoint(predictedAimX, predictedAimY, STANDARD_DECIMAL_DIGITS, "px"),
    controlCenter: formatPoint(controlCenterX, controlCenterY, STANDARD_DECIMAL_DIGITS, "px"),
    controlError: formatPoint(predictedErrorXPx, predictedErrorYPx, 2, "px"),
    errorDistancePx: predictedErrorDistancePx,
    controlFrameAgeMs,
    measurementDtMs: controlMeasurementDtMs,
    predictionEnabled: controlPredictionEnabled,
    predictionAllowedX: controlPipeline?.prediction_allowed ?? null,
    predictionAllowedY: controlPipeline?.prediction_allowed_y ?? null,
    predictionSafeOffset: formatPoint(
      controlPipeline?.prediction_safe_offset_x,
      controlPipeline?.prediction_safe_offset_y,
      2,
      "px"
    ),
    controllerActive: controlHasSample,
    controllerMode: controlModeLabel,
    movementStrategy: controlPipeline?.movement_strategy ?? "",
    fullError: formatPoint(controlPipeline?.full_error_counts_x, controlPipeline?.full_error_counts_y, 2, "counts"),
    floatDemand: formatPoint(controlPipeline?.float_demand_x, controlPipeline?.float_demand_y, 2, "counts"),
    maxOutputXCounts,
    maxOutputYCounts,
    integerCommand: formatPoint(controlPipeline?.integer_command_x, controlPipeline?.integer_command_y, 0, "counts"),
    recoilEnabled: effectiveRecoilEnabled,
    hardwareTriggerRequired: triggerMode === "hardware",
    fireDelayEnabled,
    fireDelayConfiguredMs: runtimeFireDelayMs,
    fireDelayPending,
    fireDelayElapsedMs,
    fireDelayRemainingMs,
    recoilState: controlPipeline?.recoil_state ?? "",
    recoilStatus: formatRecoilState(controlPipeline?.recoil_state, controlPipeline?.recoil_block_reason),
    recoilRemainingMs: controlPipeline?.recoil_remaining_ms ?? null,
    recoilRequestedY: controlPipeline?.recoil_requested_counts_y ?? null,
    recoilEmittedY: controlPipeline?.recoil_emitted_counts_y ?? null,
    outputEnabled: runtimeOutputEnabled ?? outputEnabled,
    willEmit: controlWillEmit,
    triggerActive: controlTriggerActive,
    noSendReason: controlNoSendReason,
    kmnetRuntimeConnected,
    kmnetConnectionLabel: kmnetRuntimeConnectionLabel,
    deviceLastError: kmnetLastError || kmnetLastDeviceError,
    outputTrace: stableRuntimeOutputTrace
  }) : null;
  const captureReason = capture?.last_error || (
    capture?.running !== true
      ? "采集尚未启动"
      : nvinferInputFps !== null && nvinferInputFps > 0
        ? "推理正在接收有效输入"
        : (statistics?.nvinfer_input_counter ?? 0) > 0
          ? "已收到输入帧，正在建立速率窗口"
          : "采集已启动，等待第一帧"
  );
  const roiInputWidth = inferenceTrace?.input_width ?? 0;
  const roiInputHeight = inferenceTrace?.input_height ?? 0;
  const modelInputWidth = inferenceTrace?.model_input_width ?? 0;
  const modelInputHeight = inferenceTrace?.model_input_height ?? 0;
  const inputDownscaleFactor =
    roiInputWidth > 0 && roiInputHeight > 0 && modelInputWidth > 0 && modelInputHeight > 0
      ? Math.max(roiInputWidth / modelInputWidth, roiInputHeight / modelInputHeight)
      : null;
  const inputAreaRatio =
    roiInputWidth > 0 && roiInputHeight > 0 && modelInputWidth > 0 && modelInputHeight > 0
      ? (modelInputWidth * modelInputHeight) / (roiInputWidth * roiInputHeight)
      : null;
  // Split `artifact` (rebuilt wholesale on every supervisor publish — see
  // crates/novasight-runtime/src/supervisor.rs:2037-2043) into its identity
  // fields so the memo can stay stable across partial frames that don't
  // actually change the active deployment. Authoritative schema:
  // web/src/contracts/model.ts + crates/novasight-store/src/model_catalog.rs.
  const activeArtifact = activeModelPublished ? artifact ?? null : null;
  const activeArtifactId = activeArtifact?.id ?? null;
  const activeArtifactVersionId = activeArtifact?.version_id ?? null;
  const activeArtifactKind = activeArtifact?.kind ?? null;
  const activeArtifactPath = activeArtifact?.path ?? null;
  const activeArtifactStatus = activeArtifact?.status ?? null;
  const productConfigProfile = useMemo(
    () => activePage === "params" ? buildProductConfigProfile({
      runtimeRunning: runtimeMainlineRunning,
      configRestartRequired,
      configApplyPending,
      desiredRevision: desiredConfigRevision,
      effectiveRevision: effectiveConfigRevision,
      captureConfigured: displayCaptureProfile !== null
        && (configuredCaptureDevice.trim().length > 0 || (capture?.device ?? "").trim().length > 0),
      captureDevice: configuredCaptureDevice || capture?.device || NO_SAMPLE,
      captureProfile: displayCaptureProfile
        ? `${displayCaptureProfile.pixel_format.toUpperCase()} ${displayCaptureProfile.width}x${displayCaptureProfile.height}@${displayCaptureProfile.fps}`
        : NO_SAMPLE,
      captureSource: displayCaptureProfileSource,
      roiLabel: roiApplyLabel,
      roiApplied: roiSettingsApplied,
      modelPublished: activeModelPublished,
      modelRuntimeLoaded: runtimeInference?.loaded === true,
      modelName: activeModelName,
      artifactLabel: activeArtifact ? `${activeArtifact.kind} · ${activeArtifact.status}` : "",
      backendLabel: selectedRuntimeBackend === "deepstream_nvinfer" ? "DeepStream 推理" : selectedRuntimeBackend,
      configuredBackendLabel: configuredInferenceBackend === "deepstream_nvinfer" ? "DeepStream 推理" : configuredInferenceBackend,
      modelInputLabel: modelInputWidth > 0 && modelInputHeight > 0
        ? `${modelInputWidth}x${modelInputHeight}`
        : displayedInputShape || NO_SAMPLE,
      postprocessLabel: postprocessApplyLabel,
      postprocessApplied: postprocessSettingsApplied,
      controlModeLabel,
      triggerModeLabel: triggerModeLabel(triggerMode),
      predictionEnabled: controlPredictionEnabled,
      freshnessThresholdLabel: formatOptionalNumber(detectionFreshnessThresholdMs, 2, "ms"),
      outputEnabled,
      outputRuntimeConnected: kmnetRuntimeConnected,
      kmnetAutoConnect,
      kmnetRuntimeConnected,
      kmnetRestartRequired,
      kmnetConnectionLabel: kmnetRuntimeConnectionLabel,
      kmnetHost,
      kmnetPort: String(kmnetPort),
      kmnetUuid
    }) : null,
    [
      activePage,
      activeModelName,
      activeModelPublished,
      activeArtifactId,
      activeArtifactVersionId,
      activeArtifactKind,
      activeArtifactPath,
      activeArtifactStatus,
      capture?.device,
      configApplyPending,
      configRestartRequired,
      configuredCaptureDevice,
      configuredInferenceBackend,
      controlModeLabel,
      desiredConfigRevision,
      detectionFreshnessThresholdMs,
      displayCaptureProfile,
      displayCaptureProfileSource,
      displayedInputShape,
      controlPredictionEnabled,
      effectiveConfigRevision,
      kmnetAutoConnect,
      kmnetHost,
      kmnetPort,
      kmnetRestartRequired,
      kmnetRuntimeConnected,
      kmnetRuntimeConnectionLabel,
      kmnetUuid,
      modelInputHeight,
      modelInputWidth,
      outputEnabled,
      postprocessApplyLabel,
      postprocessSettingsApplied,
      roiApplyLabel,
      roiSettingsApplied,
      runtimeInference?.loaded,
      runtimeMainlineRunning,
      selectedRuntimeBackend,
      triggerMode
    ]
  );
  const inputDensityWarning = inputDownscaleFactor !== null && inputDownscaleFactor > 1;
  const sampledDetectionGeneration = inferenceTrace?.generation
    ?? runtimeInference?.sampled_detection_generation
    ?? null;
  const inferenceTotalMs = readNullableNumber(statistics?.inference_latency_ms);
  const inferenceHighestConfidence = activePage === "infer"
    ? runtimeDetectionItems.reduce<number | null>(
        (highest, item) => {
          return highest === null ? item.score : Math.max(highest, item.score);
        },
        null
      )
    : null;
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
    const items: Array<{ key: string; title: string; detail: string; time?: number; count?: number }> = [];
    for (const notice of errorNotices) {
      items.push({
        key: `notice-${notice.id}`,
        title: notice.title,
        detail: notice.detail || notice.source,
        time: notice.createdAt,
        count: notice.count
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

  const modelSwitch = useModelSwitchWorkflow({
    applyModelCatalogResult,
    onRefresh,
    parserPreset,
    runtimeMainlineRunning,
    selectedCatalogModel,
    setBusy,
    setConfirmationRequest,
    setLocalError,
    setModelCatalogMessage,
    setModelCatalogRefreshKey,
    setModelDetailsRefreshKey,
    setModelManagerDialogOpen,
    setSelectedModelArtifactId,
    setSelectedModelProjectId,
    setSelectedModelVersionId
  });
  const modelManagerActive = activePage === "models"
    || (activePage === "infer" && modelManagerDialogOpen);

  useEffect(() => {
    if (!modelManagerActive) {
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
  }, [applyModelCatalogResult, modelCatalogRefreshKey, modelManagerActive, onRefresh]);

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
    if (!modelManagerActive) {
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
        setSelectedModelVersionId((current) => {
          // Preserve the user's current version if it's still in the
          // refreshed list; otherwise fall back to the active model
          // version (if any) or the first available version.
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
  }, [modelDetailsRefreshKey, modelManagerActive, runtime?.active_model?.version?.id, selectedModelProjectId]);

  useEffect(() => {
    if (!modelManagerActive) {
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
  }, [modelDetailsRefreshKey, modelManagerActive, selectedModelVersionId]);

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
    let operationFailed = false;
    try {
      await selectCaptureProfile(payload);
    } catch (err) {
      operationFailed = true;
      setLocalError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);
      reportError(err, { source: 'studio', title: '操作失败' });
    }
    try {
      await onRefresh();
    } catch (err) {
      if (!operationFailed) {
        setLocalError(`采集配置处理完成，但最新状态刷新失败：${getErrorMessage(err)}`);
      }
      reportError(err, { source: "capture-refresh", title: "采集状态刷新失败" });
    } finally {
      setBusy(null);
    }
  }, [buildCapturePayload, onRefresh]);

  const {
    emergencyStop: emergencyStopMainline,
    emergencyStopping,
    start: startMainlineLaunch,
    stop: stopMainlineLaunch
  } = useMainlineLaunch({
    onRefresh,
    onRuntimeStateChange,
    runtime,
    setBusy,
    setLocalError
  });

  const runtimePhase = runtime?.semantic.phase ?? null;
  const mainlineLaunchPending = runtimePhase === "starting";
  const runtimeStopping = runtimePhase === "stopping";
  const runtimeControlRequested = ["starting", "waiting_model", "running", "standby"].includes(
    runtimePhase ?? ""
  );
  const runtimeLifecycleActive = runtimeControlRequested || runtimeStopping;
  const diagnosticModeReady = runtimePhase === "stopped";
  const runtimeControlUnavailable = runtime === null || health?.ok !== true;
  const runtimeUnavailableLabel = errors.health
    ? "服务不可达"
    : health === null
      ? "连接中"
      : health.ok !== true
      ? "服务不可达"
      : runtime === null
        ? "等待状态"
        : null;
  const captureMainConfigured = capture?.available === true || runtimeInferenceConfigured;
  const captureStatusText = runtimeUnavailableLabel ?? (capture?.state === "failed" || capture?.state === "unavailable"
    ? "采集异常"
    : capture?.running === true
      ? "运行中"
      : mainlineLaunchPending || capture?.state === "starting"
        ? "启动中"
        : captureMainConfigured
          ? "待启动"
          : "未配置");
  const inferenceStatusText = runtimeUnavailableLabel ?? (runtimeMainlineStatus.failed
      ? "管线故障"
      : runtimePhase === "starting"
        ? "启动中"
      : runtimePhase === "waiting_model"
        ? "等待模型"
      : runtimeMainlineRunning
        ? !activeModelPublished
          ? "主链运行 · 等待模型"
          : runtimeMainlineStatus.hasRuntimeConsumption
          ? "控制链路已读取"
          : runtimeMainlineStatus.hasInferenceSignal
            ? "识别结果已产出"
            : "等待识别结果"
      : runtimeInferenceConfigured
        ? "待启动"
        : "未配置");

  const toggleCapture = useCallback(async () => {
    if (runtimeLifecycleActive) {
      await stopMainlineLaunch();
      return;
    }
    await startMainlineLaunch();
  }, [
    runtimeLifecycleActive,
    startMainlineLaunch,
    stopMainlineLaunch
  ]);

  const updateConfigField = useCallback(
    async (
      section: string,
      key: string,
      value: RuntimeConfigValue,
      options?: { immediate?: boolean; optimistic?: boolean; rethrow?: boolean }
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
      if (activePage === "params" && options?.immediate !== true) {
        stageParameterPageDraft(next);
        return;
      }
      const preservedParameterDraft = options?.immediate === true && parameterPageDirtyRef.current
        ? cloneRuntimeConfig(base)
        : null;
      const optimistic = options?.optimistic !== false;
      const writeSeq = ++configWriteSeqRef.current;
      beginPendingConfigWrite();
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
          persistRuntimeConfigField(section, key, value)
        );
        configWriteQueueRef.current = request.then(
          () => undefined,
          () => undefined
        );
        const result = await request;
        const applied = normalizeRuntimeConfig(result.config);
        // Always advance the canonical persisted revision (guarded by the
        // unified runtime write seq). A newer optimistic edit may still own
        // the visible draft, but the next queued transaction must never be
        // built from an older revision.
        finalizeRuntimeConfigWrite(applied);
        if (preservedParameterDraft) {
          const appliedSection = asRecord(applied[section]);
          const preservedSection = {
            ...asRecord(preservedParameterDraft[section]),
            [key]: appliedSection[key]
          };
          preservedParameterDraft.revision = applied.revision;
          preservedParameterDraft[section] = preservedSection as RuntimeConfig[string];
          const baseline = cloneRuntimeConfig(parameterPageBaselineRef.current);
          if (baseline) {
            baseline.revision = applied.revision;
            baseline[section] = {
              ...asRecord(baseline[section]),
              [key]: appliedSection[key]
            } as RuntimeConfig[string];
            parameterPageBaselineRef.current = baseline;
          }
          configDraftRef.current = preservedParameterDraft;
          setConfigDraft(preservedParameterDraft);
          setParameterPageDirtyState(
            !runtimeConfigsEqual(parameterPageBaselineRef.current, preservedParameterDraft)
          );
        } else if (writeSeq === configWriteSeqRef.current) {
          // The optimistic seq still owns the visible draft; mirror the
          // canonical value into the local refs/state without touching the
          // runtime write seq again (the canonical update above is the one
          // that bumps the visible draft for non-optimistic paths).
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
          // Revert the visible draft to the canonical (pre-edit) state so the
          // user doesn't see the whole form blank. If a newer optimistic
          // edit has already taken over, leave its draft alone.
          const revertTo = runtimeConfigLatestRef.current ?? cloneRuntimeConfig(runtimeConfig);
          configDraftRef.current = revertTo;
          setConfigDraft(revertTo);
        }
        if (options?.rethrow) {
          throw err;
        }
      } finally {
        finishPendingConfigWrite();
        if (!keepsEditorInteractive && writeSeq === configWriteSeqRef.current) {
          setBusy(null);
        }
      }
    },
    [activePage, applyConfigSchema, beginPendingConfigWrite, finalizeRuntimeConfigWrite, finishPendingConfigWrite, onRuntimeConfigChange, runtimeConfig, setParameterPageDirtyState, stageConfigDialogDraft, stageParameterPageDraft]
  );

  const requestOutputGateChange = useCallback((enabled: boolean) => {
    if (!enabled) {
      return updateConfigField(
        "control",
        "output_enabled",
        false,
        { immediate: true, optimistic: false, rethrow: true }
      );
    }
    setConfirmationRequest({
      eyebrow: "物理输出",
      title: "允许发送鼠标偏移？",
      description: "开启后，Rust 控制链产生的新鲜控制量可以通过当前 kmNet 会话发送到物理设备。",
      details: [
        `设备：${kmnetHost || "未填写"}:${kmnetPort || "未填写"} · ${kmnetRuntimeConnected ? "当前已连接" : "当前未连接"}`,
        "被取代命令不会补发；暂停输出或断开 kmNet 会立即清空待发送命令。"
      ],
      confirmLabel: "确认开启输出",
      danger: true,
      onConfirm: () => updateConfigField(
        "control",
        "output_enabled",
        true,
        { immediate: true, optimistic: false, rethrow: true }
      )
    });
    return false;
  }, [kmnetHost, kmnetPort, kmnetRuntimeConnected, updateConfigField]);

  const saveParameterPageDraft = useCallback(async () => {
    if (!parameterPageDirtyRef.current || parameterPageSaving) {
      return;
    }
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
      await Promise.resolve();
    }
    const draft = cloneRuntimeConfig(configDraftRef.current);
    const baseline = cloneRuntimeConfig(
      parameterPageBaselineRef.current ?? runtimeConfigLatestRef.current
    );
    if (!draft || !baseline) {
      return;
    }
    beginPendingConfigWrite();
    setParameterPageSaving(true);
    setBusy("parameter-page.save");
    setDialogSaveError(null);
    setLocalError(null);
    let lastApplied: RuntimeConfig | null = null;
    let changes: ParameterPageFieldChange[] = [];
    let appliedChangeCount = 0;
    let remainingChangeCount = 0;
    try {
      changes = parameterPageFieldChanges(baseline, draft);
      remainingChangeCount = changes.length;
      if (changes.length === 0) {
        configDraftRef.current = baseline;
        setConfigDraft(baseline);
        parameterPageBaselineRef.current = null;
        setParameterPageDirtyState(false);
        return;
      }
      let restartRequired = false;
      let expectedRevision = readNumber(baseline.revision, 0);
      const executeSave = async () => {
        for (const change of changes) {
          const result = await persistRuntimeConfigField(
            change.section,
            change.key,
            change.value,
            expectedRevision
          );
          if (result.apply_mode !== "hot_update") {
            throw new Error(
              `${change.section}.${change.key} 未按热更新契约应用，后端返回 ${result.apply_mode}`
            );
          }
          lastApplied = normalizeRuntimeConfig(result.config);
          expectedRevision = readNumber(lastApplied.revision, expectedRevision);
          appliedChangeCount += 1;
          remainingChangeCount = Math.max(0, changes.length - appliedChangeCount);
          restartRequired ||= result.restart_required;
        }
      };
      const request = configWriteQueueRef.current.then(executeSave);
      configWriteQueueRef.current = request.then(
        () => undefined,
        () => undefined
      );
      await request;
      finalizeRuntimeConfigWrite(lastApplied ?? baseline);
      parameterPageBaselineRef.current = null;
      setParameterPageDirtyState(false);
      reportSuccess(
        "参数已保存并生效",
        restartRequired
          ? "本页参数已热更新；其他进程级配置仍等待服务启动接管。"
          : "修改已写入配置文件并同步到当前进程；没有重建运行 epoch。",
        "parameter-page"
      );
    } catch (error) {
      // The write queue mutates `lastApplied` inside an async closure. Keep the
      // recovery value explicitly wide so TypeScript does not freeze the
      // closure-owned assignment at its initial `null` value.
      let canonical: RuntimeConfig | null = lastApplied;
      try {
        canonical = normalizeRuntimeConfig(await getRuntimeConfig());
      } catch {
        // Keep the newest successful response when the recovery read also fails.
      }
      if (canonical && changes.length > 0) {
        finalizeRuntimeConfigWrite(canonical);
        const retryDraft = applyParameterPageFieldChanges(canonical, changes);
        retryDraft.revision = canonical.revision;
        parameterPageBaselineRef.current = canonical;
        configDraftRef.current = retryDraft;
        setConfigDraft(retryDraft);
        remainingChangeCount = parameterPageFieldChanges(canonical, retryDraft).length;
        setParameterPageDirtyState(remainingChangeCount > 0);
      }
      const message = getErrorMessage(error);
      const progress = remainingChangeCount === 0
        ? "后端当前配置已重新同步，没有剩余未保存修改。"
        : appliedChangeCount > 0
          ? `已保存并热更新 ${appliedChangeCount} 项，剩余 ${remainingChangeCount} 项仍保留为草稿。`
          : `本次没有参数被写入，${remainingChangeCount} 项修改仍保留为草稿。`;
      const failureMessage = `${progress} ${message}`;
      setLocalError(`参数保存未全部完成：${failureMessage}`);
      setDialogSaveError(failureMessage);
      reportError(error, { source: "parameter-page", title: "参数保存失败" });
    } finally {
      setParameterPageSaving(false);
      setBusy(null);
      finishPendingConfigWrite();
    }
  }, [applyConfigSchema, beginPendingConfigWrite, finalizeRuntimeConfigWrite, finishPendingConfigWrite, parameterPageSaving, setParameterPageDirtyState]);

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
      beginPendingConfigWrite();
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
          finalizeRuntimeConfigWrite(latest);
          return persistSection(latest);
        }
      });
      configWriteQueueRef.current = request.then(
        () => undefined,
        () => undefined
      );
      try {
        const result = await request;
        if (!result) {
          return null;
        }
        const applied = normalizeRuntimeConfig(result.config);
        finalizeRuntimeConfigWrite(applied);
        return result;
      } finally {
        finishPendingConfigWrite();
      }
    },
    [applyConfigSchema, beginPendingConfigWrite, finalizeRuntimeConfigWrite, finishPendingConfigWrite, onRuntimeConfigChange]
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

  const updateControlPipelineField = useCallback(
    async (key: ControlPipelineField, value: RuntimeConfigValue) => {
      await updateConfigField("pipeline", key, value);
    },
    [updateConfigField]
  );

  const algorithmParameterGroups = algorithmSettingsDialogOpen
    ? buildAlgorithmParameterGroups({
      pResponseScale,
      pResponseBoost,
      pResponseCurveShape,
      predictionActuationDelayMs,
      controlPredictionLeadMs,
      controlPredictionHistoryResetGapMs,
      controlPredictionCapPx,
      controlFovX,
      controlCountsPer360,
      freshnessThresholdMs
    }, configFieldIndex)
    : null;
  const responseParameters = algorithmParameterGroups?.responseParameters ?? [];
  const predictionCoreParameters = algorithmParameterGroups?.predictionCoreParameters ?? [];
  const predictionCapParameters = algorithmParameterGroups?.predictionCapParameters ?? [];
  const calibrationParameters = algorithmParameterGroups?.calibrationParameters ?? [];

  const buildAlgorithmNumberParameterControl = (parameter: AlgorithmNumberParameter, compact = false) => (
    <ParameterNumberControl
      compact={compact}
      key={parameter.key}
      label={parameter.label}
      detail={parameter.detail}
      formula={parameter.formula}
      value={parameter.value}
      min={parameter.min}
      max={parameter.max}
      recommendedMin={parameter.recommendedMin}
      recommendedMax={parameter.recommendedMax}
      step={parameter.step}
      unit={parameter.unit}
      kind={parameter.kind}
      applyMode={parameter.applyMode ?? "live"}
      riskLevel={parameter.riskLevel}
      onCommit={(value) => updateControlPipelineField(parameter.key, parameter.transform ? parameter.transform(value) : value)}
      onEditingChange={handleParameterEditingChange}
    />
  );
  const renderAlgorithmNumberParameter = (parameter: AlgorithmNumberParameter) =>
    buildAlgorithmNumberParameterControl(parameter, true);

  const targetingParameterGroups = targetAdvancedDialogOpen || trackerSettingsDialogOpen
    ? buildTargetingParameterGroups({
      targetMinConfidence,
      candidateRatioMaxAspect,
      targetSwitchPreferenceAdvantage,
      targetSwitchContinuityScore,
      targetSwitchDelayMs,
      trackerMaxMatchDistance,
      trackerPositionCostWeight,
      trackerIouCostWeight,
      trackerScaleCostWeight,
      trackerMaxSizeRatio,
      trackerMaxAssociationDtMs,
      targetLostGraceMs,
      trackerKalmanAccelerationNoise,
      trackerKalmanMeasurementNoiseX,
      trackerKalmanMeasurementNoiseY,
      trackerKalmanMaxPredictDtMs,
      trackerKalmanMaxPredictMissingMs,
      trackerKalmanMaxPredictSteps,
      trackerKalmanNisThreshold,
      trackerKalmanNisHardReject
    }, configFieldIndex)
    : null;
  const targetAdvancedParameters = targetingParameterGroups?.targetAdvancedParameters ?? [];
  const trackerCoreParameters = targetingParameterGroups?.trackerCoreParameters ?? [];
  const trackerKalmanParameters = targetingParameterGroups?.trackerKalmanParameters ?? [];

  const renderTargetingNumberParameter = (parameter: TargetingNumberParameter) => (
    <ParameterNumberControl
      key={parameter.key}
      label={parameter.label}
      detail={parameter.detail}
      value={parameter.value}
      min={parameter.min}
      max={parameter.max}
      recommendedMin={parameter.recommendedMin}
      recommendedMax={parameter.recommendedMax}
      step={parameter.step}
      unit={parameter.unit}
      kind={parameter.kind}
      compact
      applyMode={parameter.applyMode ?? "live"}
      riskLevel={parameter.riskLevel}
      onCommit={(value) => updatePipelineField(parameter.key, parameter.transform ? parameter.transform(value) : value)}
      onEditingChange={handleParameterEditingChange}
    />
  );

  const handleLearnCrosshair = useCallback(async () => {
    setBusy("crosshair.learn");
    setCrosshairMessage("");
    setLocalError(null);
    try {
      const result = await learnCrosshair();
      setCrosshairPreviewKey(Date.now());
      setCrosshairMessage(`准星模板 ${result.template.id} 已生成，等待连续观测确认。`);
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
      reportInfo(
        "准星模板已清除",
        "按 F8 重新采样并学习模板，可恢复固定 HUD 准星控制。",
        "crosshair"
      );
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
    // No confirmation dialog: clearing the template is reversible by pressing
    // F8 to re-learn, which is the same path the user would take after a
    // confirm anyway. The success toast surfaces the F8 hint.
    void performClearCrosshair();
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

  const setKmNetConnection = useCallback(async (connect: boolean) => {
    setBusy(connect ? "kmnet.connect" : "kmnet.disconnect");
    setLocalError(null);
    setKmnetTestMessage("");
    // Cancel any previous kmNet call still in flight so a rapid toggle
    // doesn't leave the backend applying a stale request.
    kmnetAbortControllerRef.current?.abort();
    const controller = new AbortController();
    kmnetAbortControllerRef.current = controller;
    const signal = controller.signal;
    let physicalDisconnectCompleted = false;
    try {
      if (connect && !kmnetAutoConnect) {
        throw new Error("请先检查地址、端口和 UUID，并开启“主链启动时连接设备”；连接操作不会替你覆盖参数。");
      } else if (!connect) {
        // Physical stop comes first. Persisting the fail-closed output gate is
        // still attempted afterwards, but a storage error cannot keep an
        // already requested device session alive.
        await disconnectKmNet(signal);
        physicalDisconnectCompleted = true;
        if (outputEnabled) {
          beginPendingConfigWrite();
          try {
            const gateResult = await setRuntimeOutputGate(false);
            const applied = normalizeRuntimeConfig(gateResult.config);
            finalizeRuntimeConfigWrite(applied);
          } finally {
            finishPendingConfigWrite();
          }
        }
        setKmnetTestMessageTone("success");
        setKmnetTestMessage(
          "kmNet 已断开；采集、推理与目标计算继续运行，物理偏移输出已关闭。"
        );
      } else {
        await connectKmNet(signal);
        setKmnetTestMessageTone("success");
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
  }, [beginPendingConfigWrite, finalizeRuntimeConfigWrite, finishPendingConfigWrite, kmnetAutoConnect, onRefresh, outputEnabled]);

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
      setKmnetTestMessageTone(result.sent ? "success" : "warning");
      setKmnetTestMessage(
        result.sent
          ? `已通过 ${result.metadata.api_name} 发送 dx=${Math.round(dx)} dy=${Math.round(dy)} · ${result.steps_sent}/1 步`
          : `未发送原始命令：${result.message} · ${result.steps_sent}/1 步`
      );
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 单步移动失败：${getErrorMessage(err)}`);

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
        ? "你确认前，当前配置已被其他操作更新。NovaSight 没有覆盖新 revision，下面的差异已按最新配置重新计算。"
        : "导入会在确认时重新读取当前配置版本，再以该事务基线替换整份运行配置。",
      details: [
        `将修改：${changedSections.join("、") || "没有差异"}`,
        "运行参数会立即应用；监听地址或存储根目录等进程级配置会单独提示。"
      ],
      confirmLabel: canonicalChanged ? "按最新配置导入" : "确认导入配置",
      danger: true,
      onConfirm: async () => {
        setBusy("import");
        beginPendingConfigWrite();
        const executeImport = async (): Promise<boolean> => {
          try {
            const canonical = normalizeRuntimeConfig(await getRuntimeConfig());
            if (!runtimeConfigValuesEqual(canonical.revision, baseline.revision)) {
              finalizeRuntimeConfigWrite(canonical);
              if (changedRuntimeConfigSections(canonical, imported).length === 0) {
                reportSuccess("无需导入配置", "当前配置已经与导入文件一致。", "config-import");
                return true;
              }
              requestConfigImportConfirmation(fileName, imported, canonical, true);
              return false;
            }
            const payload = normalizeRuntimeConfig(imported);
            payload.revision = canonical.revision;
            const result = await updateRuntimeConfig(payload);
            const applied = normalizeRuntimeConfig(result.config);
            finalizeRuntimeConfigWrite(applied);
            reportSuccess(
              "配置导入成功",
              result.restart_required
                ? "运行参数已生效；仅进程级基础配置留待下次服务启动接管。"
                : result.apply_mode === "epoch_reload"
                  ? "配置已由当前进程重载，并进入新的运行 epoch。"
                  : "配置已经即时进入当前运行状态。",
              "config-import"
            );
            return true;
          } catch (error) {
            if (getApiErrorCode(error) === "CONFIG_REVISION_CONFLICT") {
              const canonical = normalizeRuntimeConfig(await getRuntimeConfig());
              finalizeRuntimeConfigWrite(canonical);
              if (changedRuntimeConfigSections(canonical, imported).length === 0) {
                reportSuccess("无需导入配置", "当前配置已经与导入文件一致。", "config-import");
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
          finishPendingConfigWrite();
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
        throw new Error("尚未读取当前配置，不能安全导入。");
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

  const refreshModelCatalog = async () => {
    setBusy("model.refresh");
    setLocalError(null);
    setModelCatalogMessage("");
    try {
      const result = await getModelCatalog(true);
      applyModelCatalogResult(result);
      setModelDetailsRefreshKey((current) => current + 1);
      await onRefresh();
      setModelCatalogMessage(
        `刷新完成：发现 ${result.model_count} 个模型文件；仅刷新目录与元数据，未读取 Engine 内容。`
      );
    } catch (err) {
      setLocalError(`模型列表刷新失败：${getErrorMessage(err)}`);

      reportError(err, { source: 'studio', title: '操作失败' });
    } finally {
      setBusy(null);
    }
  };

  const realtimeStatusText = runtimeDeliveryLabel(realtimeStatus);
  const realtimeStatusDescription = runtimeDeliveryDescription(realtimeStatus);
  const realtimeStatusClass = `console-live ${runtimeDeliveryTone(realtimeStatus)}`;
  const telemetryWindowMs = readNullableNumber(statistics?.telemetry_window_ms);
  const latencySampleCount = readNullableNumber(statistics?.inference_latency_samples);
  const latencyHasSamples = statistics?.metrics_available === true && (
    (latencySampleCount ?? 0) > 0
    || inferenceTotalMs !== null
    || detectionDataAgeMs !== null
    || detectionBatchFps !== null
  );
  const runtimeMetricsStatus = statistics?.metrics_available === true
    ? telemetryWindowMs === null
      ? "已有样本 · 正在建立速率窗口"
      : "真实样本可用"
    : "等待首个运行样本";
  const launchReadiness = useMemo(
    () => buildLaunchReadiness({
      license,
      runtime,
      serviceAvailable: errors.health ? false : health?.ok ?? null
    }),
    [errors.health, health?.ok, license, runtime]
  );
  const runtimeTransportConfidence: RuntimeTransportConfidence =
    realtimeStatus === "connected" || realtimeStatus === "fallback"
      ? "current"
      : realtimeStatus === "stale" || realtimeStatus === "paused"
        ? "stale"
        : "unavailable";
  const runtimeProjection = useMemo(
    () => projectRuntimeState(runtime, runtimeTransportConfidence),
    [runtime, runtimeTransportConfidence]
  );
  const handleLaunchReadinessAction = useCallback((action: LaunchReadinessAction) => {
    if (action === "open-model-manager") {
      navigatePage("models");
      return;
    }
    if (action === "open-capture") {
      navigatePage("capture");
      return;
    }
    if (action === "open-control") {
      navigatePage("control");
      return;
    }
    if (action === "open-latency") {
      navigatePage("latency");
      return;
    }
    if (action === "open-kmnet-test") {
      navigatePage("control-test");
      return;
    }
    if (action === "open-params") {
      navigatePage("params");
      return;
    }
    setLocalError("授权状态已在进入 Studio 前校验；如需更换授权，请返回授权入口。");
  }, [navigatePage, setLocalError]);
  const handleRuntimeRecoveryAction = useCallback((action: RuntimeRecoveryAction) => {
    handleLaunchReadinessAction(action);
  }, [handleLaunchReadinessAction]);
  const handleProductConfigAction = useCallback((action: ProductConfigAction) => {
    if (action === "capture") {
      navigatePage("capture");
      return;
    }
    if (action === "models") {
      navigatePage("models");
      return;
    }
    if (action === "control") {
      navigatePage("params");
      return;
    }
    if (action === "kmnet") {
      navigatePage("control-test");
      return;
    }
    if (action === "advanced") {
      openConfigDialog("target-advanced");
      return;
    }
    openConfigDialog("algorithm");
  }, [navigatePage, openConfigDialog]);

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
            <div className="console-toolbar-item" title={projects[0]?.name ?? "默认项目"}>
              <NovaIcon name="models" size={15} />
              <span><small>项目</small>{projects[0]?.name ?? "默认项目"}</span>
            </div>
            <div
              className="console-toolbar-item"
              title={configuredCaptureDevice || capture?.device || "尚未配置采集设备"}
            >
              <NovaIcon name="jetson" size={15} />
              <span><small>采集设备</small>{configuredCaptureDevice || capture?.device || "未配置"}</span>
            </div>
          </div>
          <div className="console-group">
            <button type="button"
              className={currentErrorDetails.length > 0 ? "error-center-trigger has-errors" : "error-center-trigger"}
              onClick={() => setErrorCenterOpen(true)}
            >
              <NovaIcon name={currentErrorDetails.length > 0 ? "triangle-alert" : "shield-check"} size={15} />
              <span>异常信息</span>
              {currentErrorDetails.length > 0 ? <b>{currentErrorDetails.length}</b> : null}
            </button>
            <ThemeToggle />
            {configApplyPending ? (
              <div
                aria-label="正在保存并应用运行配置"
                className="console-toolbar-pending"
                role="status"
                title="至少有一项配置正在保存并等待运行态确认"
              >
                <NovaIcon name="restart" size={13} />
                <span>保存并应用中…</span>
              </div>
            ) : null}
            <div
              aria-label={`${realtimeStatusText}。${realtimeStatusDescription}运行态 ${formatDate(lastUpdated)}`}
              className={realtimeStatusClass}
              role="status"
              title={`${realtimeStatusDescription} 最近更新 ${formatDate(lastUpdated)}`}
            >
              <span>{realtimeStatusText}</span>
              <small>{formatDate(lastUpdated)}</small>
            </div>
          </div>
        </div>
      </header>

      <aside className="console-sidebar">
        <StudioNavigation activePage={activePage} onNavigate={navigatePage} />
      </aside>

      <main className="console-main" data-page={activePage} ref={mainRef}>
        <StudioPageHeader page={activePage} />
        <StudioRuntimeBar
          page={activePage}
          runtimeAvailable={runtime !== null}
          runtimeLifecycleActive={runtimeLifecycleActive}
          runtimeControlRequested={runtimeControlRequested}
          runtimeStopping={runtimeStopping}
          launchPending={mainlineLaunchPending}
          diagnosticModeReady={diagnosticModeReady}
          busy={busy !== null}
          emergencyStopping={emergencyStopping}
          runtimeControlUnavailable={runtimeControlUnavailable}
          captureStatus={captureStatusText}
          inferenceStatus={inferenceStatusText}
          onToggle={() => void toggleCapture()}
          onEmergencyStop={() => void emergencyStopMainline()}
        />

        {activePage === "overview" ? (
          <RuntimeOverviewView
            runtime={runtime}
            projection={runtimeProjection}
            readiness={launchReadiness}
            lastUpdated={lastUpdated}
            onAction={handleRuntimeRecoveryAction}
          />
        ) : null}

        {activePage === "models" ? (
          <Suspense fallback={<div className="console-info" role="status">正在加载模型管理工作区…</div>}>
            <ModelWorkspace
              activeModelName={activeModelName}
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
                switchMessage: modelSwitch.message,
                busy,
                canSwitch: selectedCatalogModel?.kind === "engine",
                parserPreset,
                onParserPresetChange: setParserPreset,
                onRefresh: () => void refreshModelCatalog(),
                onSelectModel: selectModelFromCatalog,
                onSaveMetadata: (recommendation, tags) => void modelSwitch.saveMetadata(recommendation, tags),
                onSwitch: modelSwitch.switchModel,
              }}
            />
          </Suspense>
        ) : null}

        {activePage !== "overview" && activePage !== "models" && activePage !== "params" && activePage !== "control-test" && runtimeLifecycleActive && runtimeMainlinePresentation.readinessCode !== "ready" ? (
          <div className="console-info" role="status">
            {runtimeMainlinePresentation.readinessLabel}：{runtimeMainlinePresentation.readinessDetail}
          </div>
        ) : null}

        {activePage === "capture" ? (
          <LaunchReadinessPanel
            busy={busy !== null || runtimeStopping || runtimeControlUnavailable}
            onAction={handleLaunchReadinessAction}
            readiness={launchReadiness}
          />
        ) : null}

        {activePage === "capture" ? (
          <section className="console-page">
          {runtime !== null ? (
            <div className="console-metrics">
              <Metric title="采集状态" value={captureStatusText} small={capture?.device || configuredCaptureDevice || "等待设备"} />
              <Metric title="推理输入 FPS" value={formatOptionalNumber(nvinferInputFps)} small="有效输入" />
              <Metric title="配置输入 FPS" value={formatOptionalNumber(configuredCaptureFps, 0)} small="配置值 · 非实时测量" />
              <Metric title="ROI 应用" value={roiApplyLabel} small={runtimeRoiAvailable ? `${runtimeRoiWidth}x${runtimeRoiHeight}` : "等待运行 ROI"} />
            </div>
          ) : null}
          <div className="console-grid2 capture-config-grid compact-content-grid">
              <div className="console-card">
                <SectionTitle title="采集设备" />
                <TextControl
                  label="视频设备"
                  detail="更换设备路径后先检测能力，再明确保存采集配置。"
                  value={device}
                  applyMode="launch"
                  onCommit={(value) => {
                    setDevice(value);
                    setCaps(null);
                    setSelectedChoiceId("");
                  }}
                />
                <div className="capture-format-control">
                  <SelectControl
                    label="采集格式"
                    detail={caps ? "从设备实际返回的格式中选择，然后保存为运行配置。" : "未检测前使用已保存格式；更换采集卡后建议重新检测。"}
                    value={selectedChoice ? choiceId(selectedChoice) : ""}
                    applyMode="launch"
                    options={choices.length > 0
                      ? choices.map((choice) => ({
                          value: choiceId(choice),
                          label: choiceLabel(choice)
                        }))
                      : [{ value: "", label: "请先检测设备能力", disabled: true }]}
                    onCommit={setSelectedChoiceId}
                  />
                  <button className="console-button secondary" disabled={busy === "caps"} onClick={refreshCapabilities} type="button">
                    <NovaIcon name="refresh" size={15} />
                    {busy === "caps" ? "检测中..." : "检测设备能力"}
                  </button>
                  <button
                    className="console-button primary"
                    disabled={busy !== null || runtimeLifecycleActive || runtimeControlUnavailable}
                    onClick={() => void applyCapture()}
                    type="button"
                  >
                    保存采集配置
                  </button>
                </div>
                <p className="console-section-note">
                  {caps
                    ? `已读取 ${choices.length} 组设备格式；更换采集卡或设备路径后请重新检测。`
                    : "当前使用已保存的采集格式，不会在打开页面时自动探测设备。"}
                </p>
                <ParameterPresetControl
                  label="推理画面预览"
                  detail="远程预览最高帧率；远程观看时还可以在画面上选择省流档。"
                  options={[15, 30].map((fps) => ({
                    id: `preview-${fps}`,
                    label: `${fps}fps`,
                    detail: fps === 15 ? "省带宽" : "更顺滑",
                    active: previewFps === fps,
                    disabled: busy === "limits.stream_fps",
                    onSelect: () => updateConfigField("limits", "stream_fps", fps)
                  }))}
                />
              </div>

              <div className="console-card">
                <SectionTitle title="ROI 裁剪" />
                <ParameterNumberControl
                  label="ROI 尺寸"
                  detail="主链按 ROI 正中心裁剪后送入推理；尺寸越大覆盖越广，越小目标细节越密。"
                  value={roiSize}
                  min={256}
                  max={640}
                  step={16}
                  unit="px"
                  onCommit={handleCenteredRoiSizeChange}
                  onEditingChange={handleParameterEditingChange}
                />
                <ParameterPresetControl
                  label="ROI 快捷尺寸"
                  detail="只改中心裁剪尺寸；采集源分辨率不变。"
                  options={ROI_SIZE_CHOICES.map((size) => ({
                    id: `roi-${size}`,
                    label: `${size}`,
                    detail: size <= 320 ? "近距细节" : size >= 560 ? "大视野" : "平衡",
                    active: roiSize === size,
                    disabled: busy === "roi.size" || busy === "capture.roi",
                    onSelect: () => handleCenteredRoiSizeChange(size)
                  }))}
                />
                <div className="console-kv compact-kv">
                  <span>源画面</span><b>{sourceWidth > 0 ? `${sourceWidth}x${sourceHeight}` : NO_SAMPLE}</b>
                  <span>配置 ROI</span><b>{sourceWidth > 0 ? `x=${roiX}, y=${roiY}, ${rustControlPlane ? `${configuredRoiWidth}x${configuredRoiHeight}` : `${roiSize}x${roiSize}`}` : NO_SAMPLE}</b>
                </div>
              </div>
          </div>

          <details className="studio-diagnostic-details">
            <summary>
              <span>
                <b>采集排查信息</b>
                <small>采集原因、运行 ROI 与推理输入健康只在排障时查看。</small>
              </span>
              <i>2 组</i>
            </summary>
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
                  <span>运行 ROI</span><b>{runtimeRoiAvailable ? `x=${runtimeRoiLeft}, y=${runtimeRoiTop}, ${runtimeRoiWidth}x${runtimeRoiHeight}` : NO_SAMPLE}</b>
                  <span>ROI 应用状态</span><b>{roiApplyLabel}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="推理输入健康" />
                <div className="console-kv">
                  <span>推理输入 FPS</span><b>{formatOptionalNumber(nvinferInputFps, STANDARD_DECIMAL_DIGITS, "FPS")}</b>
                  <span>统计窗口</span><b>{formatOptionalNumber(telemetryWindowMs, 0, "ms")}</b>
                  <span>统计状态</span><b>{runtimeMetricsStatus}</b>
                  <span>说明</span><b>当前服务未提供采集卡原始 FPS 与设备协商信息</b>
                </div>
              </div>
            </div>
          </details>
          </section>
        ) : null}

        {activePage === "infer" ? (
          <section className="console-page">
          {runtime !== null ? (
            <div className="console-metrics">
              <Metric title="推理 FPS" value={formatOptionalNumber(nvinferOutputFps)} small="模型实际完成" />
              <Metric title="结果 FPS" value={formatOptionalNumber(detectionBatchFps)} small="识别结果有效产出" />
              <Metric title="结果新鲜度" value={formatOptionalNumber(detectionDataAgeMs)} small={`${detectionFreshness} · ms`} />
              <Metric title="最近检测" value={formatOptionalInteger(detectionCount)} small="最近遥测 · 最多 5Hz" />
            </div>
          ) : null}
          <div className="console-card model-selection-card">
            <SectionTitle title="模型设置" />
            <CurrentModelSummary
              artifactKind={artifact?.kind ?? ""}
              deployed={artifact !== null && artifact !== undefined}
              inputShape={displayedInputShape}
              loaded={runtimeInference?.loaded === true}
              modelName={activeModelName}
              onOpenManager={openModelManager}
            />
          </div>
          <div className="console-grid2 inference-config-grid">
            <div className="console-card">
              <SectionTitle title="推理参数" />
              <ParameterNumberControl
                label="配置置信度"
                detail="低于该分数的检测框会被过滤；调低更敏感，调高更干净。"
                value={confidence}
                min={0}
                max={1}
                recommendedMin={0.05}
                recommendedMax={0.9}
                step={0.01}
                onCommit={(value) => updateConfigField("inference", "confidence_threshold", value)}
                onEditingChange={handleParameterEditingChange}
              />
              <ParameterNumberControl
                label="配置 NMS"
                detail="同一目标附近的重叠框会按该阈值合并；过低容易误删，过高容易重复。"
                value={nms}
                min={0}
                max={1}
                recommendedMin={0.1}
                recommendedMax={0.9}
                step={0.01}
                onCommit={(value) => updateConfigField("inference", "nms_threshold", value)}
                onEditingChange={handleParameterEditingChange}
              />
              <div className="console-kv compact-kv">
                <span>运行置信度</span><b>{runtimePostprocessAvailable ? formatOptionalNumber(runtimePostprocessConfidence, 2) : NO_SAMPLE}</b>
                <span>运行 NMS</span><b>{runtimePostprocessAvailable ? formatOptionalNumber(runtimePostprocessNms, 2) : NO_SAMPLE}</b>
                <span>应用状态</span><b>{postprocessApplyLabel}</b>
              </div>
              <details className="model-debug-details">
                <summary>当前模型详情</summary>
                <div className="model-debug-grid">
                  <span>当前模型</span><b>{activeModelName}</b>
                  <span>运行装载</span><b>{runtimeInference?.loaded === true ? "已装载" : "未装载"}</b>
                  <span>登记输入</span><b>{registeredInputShape || "-"}</b>
                  <span>运行输入</span><b>{modelInputWidth > 0 && modelInputHeight > 0 ? `${modelInputWidth}x${modelInputHeight}` : NO_SAMPLE}</b>
                  <span>模型文件</span><b>{artifact ? `${artifact.kind} · ${artifact.status}` : NO_SAMPLE}</b>
                  <span>部署版本</span><b>{version?.version || NO_SAMPLE}</b>
                  <span>运行方式</span><b>{readString(runtime?.inference?.selected, "auto")}</b>
                  <span>类别数量</span><b>{String(version?.classes.length ?? 0)}</b>
                  <span className="wide">类别名</span><b className="wide">{version?.classes.length ? version.classes.join(", ") : NO_SAMPLE}</b>
                </div>
              </details>
            </div>
            <div className="console-card">
              <SectionTitle title="ROI 输入预览" />
              <PreviewFrame
                supported={activePage === "infer" && previewEnabled && previewRuntimePresentation.streamReady}
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
          <details className="studio-diagnostic-details">
            <summary>
              <span>
                <b>推理排查信息</b>
                <small>累计缓冲、时间戳匹配、结果版本和后处理细节只在排查时查看。</small>
              </span>
              <i>4 组</i>
            </summary>
            <div className="console-grid2 diagnostic-grid" data-layer="inference">
              <div className="console-card">
                <SectionTitle title="推理调度" />
                <div className="console-kv">
                  <span>推理状态</span><b>{inferenceStatusText}</b>
                  <span>推理原因</span><b>{inferenceReason || NO_SAMPLE}</b>
                  <span>推理输入帧</span><b>{formatOptionalInteger(deepstreamInputFrames)}</b>
                  <span>推理输出缓冲</span><b>{formatOptionalInteger(deepstreamOutputBuffers)}</b>
                  <span>元数据提取成功</span><b>{formatOptionalInteger(deepstreamMetadataExtractions)}</b>
                  <span>识别结果已发布</span><b>{formatOptionalInteger(deepstreamPublishedBatches)}</b>
                  <span>识别结果已读取</span><b>{formatOptionalInteger(runtimeMainlineStatus.consumedBatches)}</b>
                  <span>目标选择输入</span><b>{formatOptionalInteger(runtimeMainlineStatus.targetingBatches)}</b>
                  <span>缓冲时间戳匹配</span><b>{formatOptionalInteger(runtimeInference?.timestamp_buffer_pts_matches)}</b>
                  <span>帧元数据时间戳匹配</span><b>{formatOptionalInteger(runtimeInference?.timestamp_frame_meta_pts_matches)}</b>
                  <span>时间戳关联失败</span><b>{formatOptionalInteger(runtimeInference?.timestamp_correlation_misses)}</b>
                  <span>采样结果版本</span><b>{formatOptionalInteger(sampledDetectionGeneration)}</b>
                  <span>统计窗口</span><b>{formatOptionalNumber(telemetryWindowMs, 0, "ms")}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="模型输入" />
                <p className="console-section-note">这里只展示当前运行状态能够确认的尺寸；类型、精度和布局未确认时不作推断。</p>
                <div className="console-kv">
                  <span>模型名称</span><b>{activeModelName || NO_SAMPLE}</b>
                  <span>推理方式</span><b>{selectedRuntimeBackend === "deepstream_nvinfer" ? "DeepStream 推理" : selectedRuntimeBackend || NO_SAMPLE}</b>
                  <span>ROI 输入尺寸</span><b>{roiInputWidth > 0 && roiInputHeight > 0 ? `${roiInputWidth}x${roiInputHeight}` : NO_SAMPLE}</b>
                  <span>模型输入尺寸</span><b>{modelInputWidth > 0 && modelInputHeight > 0 ? `${modelInputWidth}x${modelInputHeight}` : displayedInputShape || NO_SAMPLE}</b>
                  <span>运行 ROI</span><b>{runtimeRoiAvailable ? `${runtimeRoiWidth}x${runtimeRoiHeight}` : NO_SAMPLE}</b>
                  <span>运行结果版本</span><b>{formatOptionalInteger(sampledDetectionGeneration)}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="推理引擎阶段" />
                <p className="console-section-note">从数据进入推理链到输出离开：包含输入预处理、模型推理和结果解析；不包含目标跟踪与鼠标控制。</p>
                <div className="console-kv">
                  <span>sink → src 总耗时</span><b>{formatOptionalNumber(inferenceTotalMs, 2, "ms")}</b>
                  <span>有效计时样本</span><b>{formatOptionalInteger(statistics?.inference_latency_samples)}</b>
                  <span>计时范围</span><b>预处理 + 推理 + 解析</b>
                  <span>说明</span><b>不包含目标选择、跟踪与控制计算</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="输出与后处理" />
                <div className="console-kv">
                  <span>置信度阈值</span><b>{formatOptionalNumber(runtimePostprocessConfidence, 2)}</b>
                  <span>NMS IoU 阈值</span><b>{formatOptionalNumber(runtimePostprocessNms, 2)}</b>
                  <span>最近检测数</span><b>{formatOptionalInteger(detectionCount)}</b>
                  <span>遥测列表截断</span><b>{formatOptionalInteger(vision?.detection_items_truncated)}</b>
                  <span>最高检测置信度</span><b>{formatOptionalNumber(inferenceHighestConfidence, 3)}</b>
                  <span>识别结果状态</span><b>{inferenceBatchState}</b>
                  <span>识别结果已发布</span><b>{!inferenceRan ? NO_SAMPLE : inferenceBatchPublished ? "是" : "否"}</b>
                  <span>识别结果帧龄</span><b>{formatOptionalNumber(detectionDataAgeMs, 2, "ms")}</b>
                  <span>控制新鲜度阈值</span><b>{formatOptionalNumber(detectionFreshnessThresholdMs, 2, "ms")}</b>
                </div>
              </div>
            </div>
          </details>
          </section>
        ) : null}

        {activePage === "control" ? (
          <section className="console-page">
          {runtime === null ? (
            <WorkspaceNotice
              icon="signal-lost"
              title="无法读取控制运行状态"
              detail="页面不会用配置值冒充实时控制结果。请恢复 Web/API 与 novasightd 的状态通道后重新读取。"
              action={(
                <button className="console-button" disabled={busy !== null} onClick={() => void onRefresh()} type="button">
                  <NovaIcon name="refresh" size={15} />
                  重新读取
                </button>
              )}
            />
          ) : (
          <>
          <ControlTracePanel trace={controlTrace!} />
          <details className="studio-diagnostic-details">
            <summary>
              <span>
                <b>控制排查信息</b>
                <small>目标筛选、AimPoint、量化输出与设备回执只在排查控制问题时查看。</small>
              </span>
              <i>5 组</i>
            </summary>
            <div className="console-grid2 diagnostic-grid" data-layer="control">
              <div className="console-card">
                <SectionTitle title="选择主要目标" />
                <div className="console-kv">
                  <span>控制状态</span><b>{controlHasSample ? control?.global_state ?? "已计算" : "未执行"}</b>
                  <span>控制原因</span><b>{control?.reason || control?.selection_reason || NO_SAMPLE}</b>
                  <span>阻断阶段</span><b>{targetPipelineStage || NO_SAMPLE}</b>
                  <span>状态代码</span><b>{targetPipelineCode || NO_SAMPLE}</b>
                  <span>状态信息</span><b>{targetPipelineMessage || NO_SAMPLE}</b>
                  <span>检测数量</span><b>{formatOptionalInteger(detectionCount)}</b>
                  <span>原始 / 合格 / 已选择</span><b>{`${formatOptionalInteger(rawCandidateCount)} / ${formatOptionalInteger(eligibleCandidateCount)} / ${formatOptionalInteger(selectedTargetCount)}`}</b>
                  <span>过滤原因</span><b>{targetPipelineRejections || NO_SAMPLE}</b>
                  <span>生效类别过滤</span><b>{effectiveClassFilter === "all" ? "全部类别" : effectiveClassFilter === "none" ? "未选择任何类别" : `cls ${effectiveClassFilter}`}</b>
                  <span>被类别过滤的 cls</span><b>{rejectedBasicClassIds.length > 0 ? rejectedBasicClassIds.join(", ") : NO_SAMPLE}</b>
                  <span>基础过滤前 / 后</span><b>{`${formatOptionalInteger(basicCandidateFilter?.raw_candidates)} / ${formatOptionalInteger(basicCandidateFilter?.filtered_candidates)}`}</b>
                  <span>基础过滤拒绝</span><b>{formatOptionalInteger(basicCandidateFilter?.rejected_candidates)}</b>
                  <span>候选目标数量</span><b>{formatOptionalInteger(controlCandidateCount)}</b>
                  <span>最终选择数量</span><b>{controlHasTarget ? "1" : controlHasSample ? "0" : NO_SAMPLE}</b>
                  <span>当前 track_id</span><b>{formatOptionalInteger(controlTrackId)}</b>
                  <span>目标类别</span><b>{activeRuntimeClassLabel || NO_SAMPLE}</b>
                  <span>目标置信度</span><b>{formatOptionalNumber(target?.score, 3)}</b>
                  <span>Track 身份置信度</span><b>{formatOptionalNumber(target?.identity_confidence, 3)}</b>
                  <span>目标选择状态</span><b>{control?.selector_state || NO_SAMPLE}</b>
                  <span>目标选择原因</span><b>{control?.selection_reason || NO_SAMPLE}</b>
                  <span>目标框坐标</span><b>{controlHasTarget ? `${formatPoint(target?.x1, target?.y1)} -> ${formatPoint(target?.x2, target?.y2)}` : NO_SAMPLE}</b>
                  <span>目标框中心</span><b>{formatPoint(target?.box_cx ?? target?.cx, target?.box_cy ?? target?.cy, STANDARD_DECIMAL_DIGITS, "px")}</b>
                  <span>遥测说明</span><b>目标与框为最近采样快照，最多约 5Hz</b>
                </div>
                {targetPipelineCode === "BASIC_CANDIDATE_REJECTED" && targetPipelineRejections.includes("class_filter") && detectedClassFilterValue ? (
                  <div className="control-filter-recovery">
                    <button type="button"
                      className="console-button primary"
                      disabled={busy !== null}
                      onClick={() => void updateDetectionClassFilter(detectedClassFilterValue)}
                    >
                      允许当前检测类别
                    </button>
                    <small>
                      {rustControlPlane
                        ? "保留已有选择，并加入本帧检测到的 cls；保存后进入当前目标选择配置。"
                        : "保留已有选择，并加入本帧检测到的 cls；保存后立即热更新。"}
                    </small>
                  </div>
                ) : null}
              </div>
              <div className="console-card">
                <SectionTitle title="目标瞄点" />
                <div className="console-kv">
                  <span>原始瞄准点</span><b>{formatPoint(observedAimX, observedAimY, STANDARD_DECIMAL_DIGITS, "px")}</b>
                  <span>目标类别</span><b>{activeRuntimeClassLabel || NO_SAMPLE}</b>
                  <span>控制瞄准点</span><b>{formatPoint(predictedAimX, predictedAimY, STANDARD_DECIMAL_DIGITS, "px")}</b>
                  <span>目标速度预测</span><b>{controlPredictionEnabled ? "4 点二维 aim 预测" : "已关闭"}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="连续非线性控制" />
                <div className="console-kv">
                  <span>屏幕中心</span><b>{formatPoint(controlCenterX, controlCenterY, STANDARD_DECIMAL_DIGITS, "px")}</b>
                  <span>观测误差</span><b>{formatPoint(controlPipeline?.observed_error_x_px, controlPipeline?.observed_error_y_px, 2, "px")}</b>
                  <span>控制误差</span><b>{formatPoint(predictedErrorXPx, predictedErrorYPx, 2, "px")}</b>
                  <span>误差距离</span><b>{formatOptionalNumber(predictedErrorDistancePx, 2, "px")}</b>
                  <span>控制 dt</span><b>{controlMeasurementDtS === null ? NO_SAMPLE : `${(controlMeasurementDtS * 1000).toFixed(3)} ms`}</b>
                  <span>控制观测帧龄</span><b>{formatOptionalNumber(controlFrameAgeMs, 2, "ms")}</b>
                  <span>预测执行延迟</span><b>{controlPredictionEnabled ? formatOptionalNumber(controlPipeline?.prediction_actuation_delay_ms, 2, "ms") : "已关闭"}</b>
                  <span>预测实际时域</span><b>{controlPredictionEnabled ? formatOptionalNumber(controlPipeline?.prediction_horizon_ms, 2, "ms") : "已关闭"}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="目标速度预测" />
                <p className="console-section-note">三段二维速度只取向量 medoid，再按预测时域计算提前瞄点。</p>
                <div className="console-kv">
                  <span>速度段 X</span><b>{`${formatOptionalNumber(controlPipeline?.velocity_1, 3)} / ${formatOptionalNumber(controlPipeline?.velocity_2, 3)} / ${formatOptionalNumber(controlPipeline?.velocity_3, 3)} px/ms`}</b>
                  <span>速度段 Y</span><b>{`${formatOptionalNumber(controlPipeline?.velocity_y_1, 3)} / ${formatOptionalNumber(controlPipeline?.velocity_y_2, 3)} / ${formatOptionalNumber(controlPipeline?.velocity_y_3, 3)} px/ms`}</b>
                  <span>Medoid 速度</span><b>{formatPoint(controlPipeline?.prediction_velocity, controlPipeline?.prediction_velocity_y, 3, "px/ms")}</b>
                  <span>原始预测</span><b>{formatPoint(controlPipeline?.prediction_raw_offset_x, controlPipeline?.prediction_raw_offset_y, 2, "px")}</b>
                  <span>预测位移上限</span><b>{formatOptionalNumber(controlPipeline?.prediction_allowed_cap_x, 2, "px")}</b>
                  <span>截断后预测</span><b>{formatPoint(controlPipeline?.prediction_safe_offset_x, controlPipeline?.prediction_safe_offset_y, 2, "px")}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="输出限幅与命令" />
                <div className="console-kv">
                  <span>控制算法</span><b>{controlModeLabel}</b>
                  <span>移动策略</span><b>{controlPipeline?.movement_strategy || NO_SAMPLE}</b>
                  <span>响应阶段</span><b>{formatResponseStage(controlPipeline?.mode)}</b>
                  <span>完整修正 counts</span><b>{formatPoint(controlPipeline?.full_error_counts_x, controlPipeline?.full_error_counts_y, 2)}</b>
                  <span>Atan 浮点需求</span><b>{formatPoint(controlPipeline?.float_demand_x, controlPipeline?.float_demand_y, 2)}</b>
                  <span>X / Y 输出上限</span><b>{`${formatOptionalNumber(maxOutputXCounts, 0)} / ${formatOptionalNumber(maxOutputYCounts, 0)} counts`}</b>
                  <span>整数输出</span><b>{formatPoint(controlPipeline?.integer_command_x, controlPipeline?.integer_command_y, 0, "counts")}</b>
                  <span>独立压枪状态</span><b>{effectiveRecoilEnabled ? formatRecoilState(controlPipeline?.recoil_state, controlPipeline?.recoil_block_reason) : "关闭"}</b>
                  <span>开火延迟</span><b>{fireDelayEnabled ? `已开启 · ${fireDelayMs.toFixed(0)} ms` : "已关闭"}</b>
                  <span>压枪间隔 / +Y</span><b>{`${formatOptionalNumber(controlPipeline?.recoil_interval_ms, 0)} ms / ${formatOptionalNumber(controlPipeline?.recoil_y_counts, 0)} counts`}</b>
                  <span>已等待 / 剩余</span><b>{`${formatOptionalNumber(controlPipeline?.recoil_elapsed_since_output_ms, 2)} / ${formatOptionalNumber(controlPipeline?.recoil_remaining_ms, 2)} ms`}</b>
                  <span>本轮请求 / 已发送</span><b>{`${formatOptionalNumber(controlPipeline?.recoil_requested_counts_y, 0)} / ${formatOptionalNumber(controlPipeline?.recoil_emitted_counts_y, 0)} counts`}</b>
                  <span>控制预算</span><b>{formatPoint(control?.dx, control?.dy, 0, "counts")}</b>
                </div>
              </div>
              <div className="console-card">
                <SectionTitle title="最新观测与设备发送" />
                <p className="console-section-note">控制样本与设备回执分别展示；最近回执不冒充为当前观测的同步发送结果。</p>
                <div className="console-kv">
                  <span>触发状态</span><b>{controlTriggerActive === true ? "按下" : controlTriggerActive === false ? "未按下" : NO_SAMPLE}</b>
                  <span>是否允许发包</span><b>{controlWillEmit === true ? "是" : controlWillEmit === false ? "否" : NO_SAMPLE}</b>
                  <span>运行输出门</span><b>{control?.output_enabled === true ? "已打开" : control?.output_enabled === false ? "已关闭" : NO_SAMPLE}</b>
                  <span>不发包原因</span><b>{controlNoSendReason || NO_SAMPLE}</b>
                  <span>本轮控制意图</span><b>{formatPoint(control?.dx, control?.dy, 0, "counts")}</b>
                  <span>发送语义</span><b>仅保留最新观测</b>
                  <span>最近设备已接受</span><b>{hasAcceptedCommand ? lastAcceptedCommand : NO_SAMPLE}</b>
                  <span>设备通道</span><b>{kmnetRuntimeConnectionLabel}</b>
                </div>
              </div>
            </div>
          </details>
          </>
          )}
          </section>
        ) : null}

        {activePage === "params" || activePage === "control-test" ? (
          <section className="console-page">
          {activePage === "params" ? (
          <>
            <div className={parameterPageDirty ? "parameter-save-bar dirty" : "parameter-save-bar"}>
              <span className="parameter-save-bar-icon" aria-hidden="true">
                <NovaIcon name={parameterPageDirty ? "save" : "check-circle"} size={18} />
              </span>
              <div aria-live="polite" role="status">
                <b>{parameterPageDirty ? "修改尚未保存" : "参数已同步"}</b>
                {parameterPageDirty ? <small>保存后立即生效</small> : null}
              </div>
              <div className="parameter-save-bar-actions">
                <button className="console-button" onClick={exportConfig} type="button">
                  <NovaIcon name="export" size={15} />
                  导出
                </button>
                <button className="console-button" onClick={() => fileInputRef.current?.click()} type="button">
                  <NovaIcon name="import" size={15} />
                  导入
                </button>
                <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={importConfig} />
                {parameterPageDirty ? (
                  <button
                    className="console-button"
                    disabled={parameterPageSaving || pendingConfigWriteCount > 0}
                    onClick={discardParameterPageDraft}
                    type="button"
                  >
                    放弃修改
                  </button>
                ) : null}
                <button
                  className="console-button primary"
                  disabled={!parameterPageDirty || parameterPageSaving || pendingConfigWriteCount > 0}
                  onClick={() => void saveParameterPageDraft()}
                  type="button"
                >
                  <NovaIcon name="save" size={15} />
                  {parameterPageSaving ? "正在保存…" : "保存修改"}
                </button>
              </div>
            </div>
            <ol className="control-chain-settings" aria-label="鼠标控制参数链">
              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true">01</span>
                <div className="control-chain-setting-title">
                  <b>触发方式</b>
                  <small>决定控制链在什么条件下开始计算与输出。</small>
                </div>
                <div className="trigger-mode-options" role="group" aria-label="触发方式">
                  <button
                    aria-pressed={triggerMode === "hardware"}
                    className={triggerMode === "hardware" ? "active" : ""}
                    disabled={busy !== null}
                    onClick={() => void updateConfigField("control", "trigger_mode", "hardware")}
                    type="button"
                  >
                    按键触发
                  </button>
                  <button
                    aria-pressed={triggerMode === "always"}
                    className={triggerMode === "always" ? "active" : ""}
                    disabled={busy !== null}
                    onClick={() => void updateConfigField("control", "trigger_mode", "always")}
                    type="button"
                  >
                    直接触发
                  </button>
                </div>
              </li>

              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true">02</span>
                <div className="control-chain-setting-title">
                  <b>开火延迟</b>
                  <small>过滤短按，避免刚触发时立即进入持续控制。</small>
                </div>
                <div className="control-chain-setting-controls">
                  <ModuleSwitch
                    compact
                    label="按键持续门槛"
                    enabled={fireDelayEnabled}
                    onToggle={(enabled) => updateControlPipelineField("fire_delay_enabled", enabled)}
                  />
                  {fireDelayEnabled ? (
                    <label className="control-chain-inline-field">
                      <span>延迟</span>
                      <InlineNumberControl
                        ariaLabel="按键持续开火延迟"
                        value={fireDelayMs}
                        onCommit={(value) => updateControlPipelineField("fire_delay_ms", Math.max(0, Math.min(5000, Math.round(value))))}
                      />
                      <i>ms</i>
                    </label>
                  ) : null}
                </div>
              </li>

              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true">03</span>
                <div className="control-chain-setting-title">
                  <b>算法配置</b>
                  <small>设置预测与连续响应，决定偏移如何被换算。</small>
                </div>
                <div className="control-chain-setting-controls split">
                  <ModuleSwitch
                    compact
                    label="目标速度预测"
                    enabled={controlPredictionEnabled}
                    onToggle={(enabled) => updateControlPipelineField("prediction_enabled", enabled)}
                  />
                  <button
                    className="console-button"
                    disabled={configDialogSaving}
                    onClick={() => openConfigDialog("algorithm")}
                    type="button"
                  >
                    <NovaIcon name="settings" size={15} />
                    算法参数
                  </button>
                </div>
              </li>

              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true">04</span>
                <div className="control-chain-setting-title">
                  <b>压枪</b>
                  <small>在主控制量之外独立叠加向下补偿。</small>
                </div>
                <div className="control-chain-setting-controls recoil">
                  <ModuleSwitch compact label="启用压枪" enabled={recoilEnabled} onToggle={(enabled) => updateControlGroupField("recoil", "enabled", enabled)} />
                  <ModuleSwitch compact label="仅有目标时" enabled={recoilRequireTarget} onToggle={(enabled) => updateControlGroupField("recoil", "require_target", enabled)} />
                  {recoilEnabled ? (
                    <>
                      <label className="control-chain-inline-field">
                        <span>间隔</span>
                        <InlineNumberControl
                          ariaLabel="压枪叠加间隔"
                          value={recoilIntervalMs}
                          onCommit={(value) => updateControlGroupField("recoil", "interval_ms", Math.max(1, Math.min(5000, Math.round(value))))}
                        />
                        <i>ms</i>
                      </label>
                      <label className="control-chain-inline-field">
                        <span>每次 +Y</span>
                        <InlineNumberControl
                          ariaLabel="每次压枪叠加 Y"
                          value={recoilYCounts}
                          onCommit={(value) => updateControlGroupField("recoil", "y_counts", Math.max(1, Math.min(32767, Math.round(value))))}
                        />
                        <i>counts</i>
                      </label>
                    </>
                  ) : null}
                </div>
              </li>

              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true">05</span>
                <div className="control-chain-setting-title">
                  <b>限幅</b>
                  <small>约束每次发送的最大 X/Y 输出，防止突变。</small>
                </div>
                <div className="control-chain-setting-controls">
                  <label className="control-chain-inline-field">
                    <span>X 轴上限</span>
                    <InlineNumberControl
                      ariaLabel="X 轴输出上限"
                      value={maxOutputXCounts}
                      onCommit={(value) => updateControlPipelineField("max_output_x_counts", Math.max(1, Math.min(32767, Math.round(value))))}
                    />
                    <i>counts</i>
                  </label>
                  <label className="control-chain-inline-field">
                    <span>Y 轴上限</span>
                    <InlineNumberControl
                      ariaLabel="Y 轴输出上限"
                      value={maxOutputYCounts}
                      onCommit={(value) => updateControlPipelineField("max_output_y_counts", Math.max(1, Math.min(32767, Math.round(value))))}
                    />
                    <i>counts</i>
                  </label>
                </div>
              </li>

              <li className={outputEnabled ? "console-card control-chain-setting output-enabled" : "console-card control-chain-setting output-paused"}>
                <span className="control-chain-step" aria-hidden="true">06</span>
                <div className="control-chain-setting-title">
                  <b>输出</b>
                  <small>物理设备的最终发送门；关闭时仍保留算法计算。</small>
                </div>
                <ModuleSwitch
                  compact
                  label="发送鼠标偏移"
                  detail={!outputEnabled && kmnetRestartRequired
                    ? "kmNet 正在重载"
                    : !outputEnabled && !kmnetAutoConnect
                      ? "请先启用 kmNet"
                      : !outputEnabled && !kmnetRuntimeConnected
                        ? "请先连接 kmNet"
                        : undefined}
                  disabled={busy !== null || kmnetRestartRequired || (!outputEnabled && (!kmnetAutoConnect || !kmnetExecutorAvailable || !kmnetRuntimeConnected))}
                  enabled={outputEnabled}
                  optimistic={false}
                  onToggle={requestOutputGateChange}
                />
              </li>
            </ol>
            <details className="studio-diagnostic-details parameter-support-details">
              <summary>
                <span>
                  <b>目标与识别设置</b>
                </span>
                <i>展开</i>
              </summary>
              <div className="parameter-support-content">
                <div className="console-card class-config-summary-card">
                  <div className="class-config-summary-main">
                    <div className="class-config-summary-icon" aria-hidden="true">
                      <NovaIcon name="target" size={20} strokeWidth={1.8} />
                    </div>
                    <div>
                      <span className="class-config-eyebrow">类别配置</span>
                      <h3>{activeDetectionProfile}</h3>
                    </div>
                  </div>
                  <button type="button"
                    className="console-button primary"
                    disabled={configDialogSaving}
                    onClick={() => openConfigDialog("class-config")}
                  >
                    <NovaIcon name="settings" size={16} />
                    管理类别配置
                  </button>
                </div>
                <div className="console-grid2 params-control-grid compact-content-grid" data-algorithm-page={controlAlgorithmId}>
              <div className="console-card crosshair-reference-card">
                <SectionTitle title="视觉准星基准" />
                <ModuleSwitch
                  label="启用低频准星观测"
                  enabled={crosshairEnabled}
                  onToggle={(enabled) => updateConfigField("crosshair", "enabled", enabled)}
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
                  <div className="crosshair-reference-status">
                    <span>当前状态</span>
                    <b data-state={stableCrosshairState}>{crosshairStateLabel(stableCrosshairState)}</b>
                  </div>
                </div>
                <div className="crosshair-learn-actions">
                  <button type="button"
                    className="console-button primary"
                    disabled={!crosshairEnabled || !runtimeMainlineRunning || crosshairRecentSamples < crosshairRequiredSamples || busy === "crosshair.learn"}
                    onClick={() => void handleLearnCrosshair()}
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
                  <summary>高级控制原点</summary>
                  <ModuleSwitch
                    compact
                    label="使用视觉准星作为控制原点"
                    enabled={crosshairUseForControl}
                    disabled={!crosshairEnabled || !crosshairTemplateId}
                    onToggle={(enabled) => updateConfigField("crosshair", "use_for_control", enabled)}
                  />
                </details>
                <details className="crosshair-advanced-settings">
                  <summary>观测信息</summary>
                  <div className="console-kv compact-kv crosshair-reference-kv">
                    <span>采样支路</span><b>{crosshairBranchActive ? "运行中" : crosshairEnabled ? "不可用" : "关闭"}</b>
                    <span>控制可用</span><b>{crosshairReferenceReady ? "是" : "否，使用几何中心"}</b>
                    <span>模板</span><b>{crosshairTemplateId || "尚未生成"}</b>
                    <span>中心偏移</span><b>{crosshairReferenceReady ? `${crosshairOffsetX.toFixed(STANDARD_DECIMAL_DIGITS)}, ${crosshairOffsetY.toFixed(STANDARD_DECIMAL_DIGITS)} px` : "—"}</b>
                    <span>匹配置信度</span><b>{crosshairConfidence > 0 ? `${(crosshairConfidence * 100).toFixed(STANDARD_DECIMAL_DIGITS)}%` : "—"}</b>
                    <span>学习帧</span><b>{crosshairRecentSamples}/{crosshairRequiredSamples}</b>
                  </div>
                </details>
                <details className="crosshair-advanced-settings">
                  <summary>采样参数</summary>
                  <div className="advanced-settings-grid">
                    <ParameterNumberControl
                      label="中心搜索区"
                      detail="只截取 ROI 正中心的小区域，不扫描整幅画面。"
                      value={crosshairSearchSize}
                      min={32}
                      max={Math.max(32, roiSize)}
                      step={2}
                      unit="px"
                      kind="stepper"
                      riskLevel="advanced"
                      onCommit={(value) => updateConfigField("crosshair", "search_size", Math.round(value / 2) * 2)}
                    />
                    <ParameterNumberControl
                      label="观测频率"
                      detail="已与推理支路隔离；10 Hz 通常足够验证固定 HUD 准星。"
                      value={crosshairSampleHz}
                      min={1}
                      max={30}
                      step={1}
                      unit="Hz"
                      kind="stepper"
                      riskLevel="advanced"
                      onCommit={(value) => updateConfigField("crosshair", "sample_hz", Math.round(value))}
                    />
                    <ParameterNumberControl
                      label="学习采样帧数"
                      detail="使用多帧中位图减少动态背景对模板的污染。"
                      value={crosshairSampleFrames}
                      min={3}
                      max={15}
                      step={1}
                      unit="帧"
                      kind="stepper"
                      riskLevel="advanced"
                      onCommit={(value) => updateConfigField("crosshair", "sample_frames", Math.round(value))}
                    />
                  </div>
                </details>
              </div>

              <div className="console-card">
                <SectionTitle title="目标选择" />
                <ParameterNumberControl
                  compact
                  label="选择半径"
                  value={targetFovRadiusPx}
                  min={0.000001}
                  max={100000}
                  recommendedMin={1}
                  recommendedMax={640}
                  step={1}
                  unit="px"
                  onCommit={(value) => updatePipelineField("target_fov_radius_px", value)}
                  onEditingChange={handleParameterEditingChange}
                />
                <div className="target-weight-summary">
                  <div>
                    <span>类别 / 距离</span>
                    <strong>
                      {(normalizedSelectionClassWeight * 100).toFixed(0)}%
                      <i>·</i>
                      {(normalizedSelectionDistanceWeight * 100).toFixed(0)}%
                    </strong>
                  </div>
                  <button type="button" className="console-button" disabled={configDialogSaving} onClick={() => openConfigDialog("target-weights")}              >
                    <NovaIcon name="settings" size={15} />
                    权重
                  </button>
                </div>
                <button type="button" className="console-button console-full-button" disabled={configDialogSaving} onClick={() => openConfigDialog("target-advanced")}              >
                  <NovaIcon name="settings" size={15} />
                  目标行为
                </button>
              </div>

              <div className="console-card">
                <SectionTitle title="目标跟踪" />
                <button type="button" className="console-button console-full-button" disabled={configDialogSaving} onClick={() => openConfigDialog("tracker")}              >
                  <NovaIcon name="settings" size={15} />
                  专家跟踪
                </button>
              </div>
                </div>
              </div>
            </details>
            <details className="studio-diagnostic-details parameter-support-details">
              <summary>
                <span>
                  <b>配置生效状态</b>
                </span>
                <i>{productConfigProfile!.attentionCount > 0 ? `${productConfigProfile!.attentionCount} 项需处理` : "正常"}</i>
              </summary>
              <div className="parameter-support-content">
                <ProductConfigProfilePanel
                  busy={busy !== null}
                  onAction={handleProductConfigAction}
                  profile={productConfigProfile!}
                />
              </div>
            </details>
          </>
          ) : (
          <>
          <div className="console-grid2 control-test-grid">
            <div className="console-card">
              <SectionTitle title="kmNet 单步测试" />
              <div className="kmnet-status-grid">
                <div className={diagnosticModeReady ? "kmnet-status-tile good" : "kmnet-status-tile bad"}>
                  <span>主链</span>
                  <b>{diagnosticModeReady ? "已停止" : runtime === null ? "状态不可用" : "未完全停止"}</b>
                </div>
                <div className={kmnetExecutorAvailable ? "kmnet-status-tile good" : "kmnet-status-tile bad"}>
                  <span>硬件适配器</span>
                  <b>{kmnetExecutorAvailable ? "可用" : "不可用"}</b>
                </div>
                <div className={outputEnabled ? "kmnet-status-tile good" : "kmnet-status-tile bad"}>
                  <span>物理输出门</span>
                  <b>{outputEnabled ? "已允许" : "已关闭"}</b>
                </div>
                <div className={kmnetConfigurationReady ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>配置装载</span>
                  <b>{kmnetRestartRequired ? `${effectiveConfigRevision} → ${desiredConfigRevision}` : kmnetConfigurationReady ? "已生效" : "未委任"}</b>
                </div>
              </div>
              {kmnetNotice.code !== "none" ? (
                <div className={kmnetNotice.code === "failed" ? "kmnet-connection-notice failed" : "kmnet-connection-notice warn"} role="status">
                  <div>
                    <strong>{kmnetNotice.code === "restart_required" ? "kmNet 配置尚未进入当前进程" : kmnetNotice.code === "failed" ? "硬件适配器不可用" : kmnetNotice.code === "degraded" ? "最近一次设备会话异常" : "最近一次输出失败"}</strong>
                    <span>{kmnetNotice.code === "restart_required"
                      ? `当前进程使用 revision ${kmnetNotice.effectiveRevision}，已保存 revision ${kmnetNotice.desiredRevision}；单步测试保持锁定。`
                      : kmnetNotice.lastError || kmnetNotice.lastDeviceError || "请检查地址、端口、UUID 和局域网连通性。"}</span>
                  </div>
                  <small>单步命令会独立执行“连接 → 发送一次 → 断开”，不会复用主链实时会话。</small>
                </div>
              ) : null}
              <details className="kmnet-configuration-details" open={!kmnetConfigurationReady || undefined}>
                <summary>
                  <span>
                    <b>设备连接配置</b>
                    <small>{kmnetConfigurationReady ? `${kmnetHost}:${kmnetPort}` : "完成地址、端口与 UUID 后才能测试"}</small>
                  </span>
                  <i>{kmnetConfigurationReady ? "已配置" : "需配置"}</i>
                </summary>
                <div className="kmnet-configuration-content">
                  <TextControl
                    label="kmNet 地址"
                    detail="设备控制器的局域网 IP 或主机名；不要使用示例常量代替设备真实地址。"
                    value={kmnetHost}
                    applyMode="live"
                    riskLevel="advanced"
                    onCommit={(value) => updateConfigField("hardware", "host", value)}
                  />
                  <ParameterNumberControl
                    label="kmNet 控制端口"
                    detail="单步测试连接使用的主控制端口。"
                    value={kmnetPort}
                    min={rustControlPlane ? 1 : 0}
                    max={65535}
                    step={1}
                    kind="stepper"
                    applyMode="live"
                    riskLevel="advanced"
                    onCommit={(value) => updateConfigField("hardware", "port", Math.round(value))}
                  />
                  <TextControl
                    label="kmNet UUID"
                    detail="设备授权标识；必须填写实际设备值，不会自动套用通用 UUID。"
                    value={kmnetUuid}
                    applyMode="live"
                    riskLevel="advanced"
                    onCommit={(value) => updateConfigField("hardware", "uuid", value)}
                  />
                  <ParameterNumberControl
                    label="kmNet 监控端口"
                    detail="主链实时会话读取设备状态时使用的监控端口。"
                    value={kmnetMonitorPort}
                    min={rustControlPlane ? 1024 : 0}
                    max={rustControlPlane ? 49151 : 65535}
                    step={1}
                    kind="stepper"
                    applyMode="live"
                    riskLevel="advanced"
                    onCommit={(value) => updateConfigField("hardware", "monitor_port", Math.round(value))}
                  />
                  <ModuleSwitch
                    label={rustControlPlane ? "主链启动时连接设备" : "NovaSight 启动时自动连接"}
                    detail="只影响实时主链会话；单步测试始终独立连接、发送并断开。"
                    enabled={kmnetAutoConnect}
                    onToggle={(enabled) => updateConfigField("hardware", "auto_connect", enabled)}
                  />
                  <div className="kmnet-live-session">
                    <div>
                      <b>实时主链会话</b>
                      <small>仅在主链运行时手动重连或安全断开；单步测试不需要此操作。</small>
                    </div>
                    <div className="console-action-row">
                      <button
                        aria-pressed={kmnetConnected}
                        className="console-button"
                        disabled={busy !== null || kmnetRestartRequired || !kmnetAutoConnect || !kmnetCanConnect}
                        onClick={() => void setKmNetConnection(true)}
                        type="button"
                      >
                        {kmnetConnecting ? "连接中" : kmnetConnectionDegraded ? "重试实时连接" : "连接实时会话"}
                      </button>
                      <button
                        className="console-button danger"
                        disabled={busy !== null || !kmnetCanDisconnect}
                        onClick={() => void setKmNetConnection(false)}
                        type="button"
                      >
                        断开实时会话
                      </button>
                    </div>
                  </div>
                </div>
              </details>
              <details className="kmnet-diagnostic-details">
                <summary>
                  <span>
                    <b>排查信息</b>
                    <small>运行阶段、会话状态和累计计数只在排查失败时查看。</small>
                  </span>
                  <i>9 项</i>
                </summary>
                <div className="console-kv compact-kv">
                  <span>执行器</span><b>{runtime?.executor.selected || NO_SAMPLE}</b>
                  <span>运行阶段</span><b>{runtimePhase || NO_SAMPLE}</b>
                  <span>配置状态</span><b>{kmnetConfigurationState}</b>
                  <span>实时会话</span><b>{kmnetConnectionState}</b>
                  <span>最近设备错误</span><b>{kmnetLastDeviceError || kmnetLastError || "无"}</b>
                  <span>单步发送次数</span><b>{formatOptionalInteger(kmnetStatus?.diagnostic_move_count)}</b>
                  <span>最近单步位移</span><b>{formatPoint(kmnetStatus?.last_diagnostic_dx, kmnetStatus?.last_diagnostic_dy, 0)}</b>
                  <span>设备恢复次数</span><b>{formatOptionalInteger(kmnetStatus?.device_recovery_count)}</b>
                  <span>设备错误次数</span><b>{formatOptionalInteger(kmnetStatus?.device_error_count)}</b>
                </div>
              </details>
              <div className="kmnet-test-panel">
                <div className="kmnet-test-section">
                  <h3>发送一次受控移动</h3>
                  <p>每次点击只发送一条命令；服务端会再次验证主链已停止、输出门已打开，并在完成后断开设备。</p>
                </div>
                <div className="kmnet-test-inputs">
                  <label>
                    <span>dx</span>
                    <InlineNumberControl
                      ariaLabel="kmNet 测试 dx"
                      disabled={kmnetDiagnosticDisabled}
                      value={kmnetTestDx}
                      onCommit={setKmnetTestDx}
                    />
                  </label>
                  <label>
                    <span>dy</span>
                    <InlineNumberControl
                      ariaLabel="kmNet 测试 dy"
                      disabled={kmnetDiagnosticDisabled}
                      value={kmnetTestDy}
                      onCommit={setKmnetTestDy}
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
                <div className="kmnet-test-result">
                  <span>最近测试移动</span>
                  <b>{formatPoint(kmnetStatus?.last_diagnostic_dx, kmnetStatus?.last_diagnostic_dy, 0)}</b>
                </div>
                {kmnetTestMessage ? <div className={`kmnet-test-message ${kmnetTestMessageTone}`}>{kmnetTestMessage}</div> : null}
              </div>
            </div>
          </div>
          </>
          )}
          </section>
        ) : null}

        {activePage === "latency" ? (
          <section className="console-page">
          {!latencyHasSamples ? (
            <WorkspaceNotice
              icon={runtime === null ? "signal-lost" : runtimeLifecycleActive ? "clock" : "latency"}
              title={runtime === null
                ? "无法读取延迟运行状态"
                : runtimeLifecycleActive
                  ? "正在等待第一批真实统计"
                  : "尚未采集延迟样本"}
              detail={runtime === null
                ? "状态通道恢复前不展示占位数字，也不会把配置值当作实时测量。"
                : runtimeLifecycleActive
                  ? "主链已请求运行；统计窗口完成后，这里会显示推理耗时、结果帧龄与实际吞吐。"
                  : "启动主链后才会建立真实统计窗口；未打点的阶段不会估算为 0。"}
              action={runtime === null ? (
                <button className="console-button" disabled={busy !== null} onClick={() => void onRefresh()} type="button">
                  <NovaIcon name="refresh" size={15} />
                  重新读取
                </button>
              ) : !runtimeLifecycleActive ? (
                <button className="console-button primary" disabled={busy !== null || runtimeControlUnavailable} onClick={() => void toggleCapture()} type="button">
                  <NovaIcon name="start" size={15} />
                  启动主链
                </button>
              ) : undefined}
            />
          ) : (
          <>
            <div className="console-metrics">
              <Metric title="推理链耗时" value={formatOptionalNumber(inferenceTotalMs)} small="预处理 + 推理 + 解析 · ms" />
              <Metric title="结果帧龄" value={formatOptionalNumber(detectionDataAgeMs)} small={`${detectionFreshness} · ms`} />
              <Metric title="结果 FPS" value={formatOptionalNumber(detectionBatchFps)} small="识别结果/s" />
              <Metric title="统计窗口" value={formatOptionalNumber(telemetryWindowMs, 0)} small="ms" />
            </div>
            <div className="console-grid2 latency-analysis-grid">
              <KvCard
                title="当前真实测量"
                rows={[
                  ["统计状态", runtimeMetricsStatus],
                  ["推理输入 FPS", formatOptionalNumber(nvinferInputFps, 2, "FPS")],
                  ["推理输出 FPS", formatOptionalNumber(nvinferOutputFps, 2, "FPS")],
                  ["识别结果 FPS", formatOptionalNumber(detectionBatchFps, 2, "FPS")],
                  ["目标选择输入 FPS", formatOptionalNumber(targetingBatchFps, 2, "FPS")],
                  ["sink → src 样本", formatOptionalInteger(statistics?.inference_latency_samples)],
                  ["控制新鲜度阈值", formatOptionalNumber(detectionFreshnessThresholdMs, 2, "ms")]
                ]}
                notice={<p className="latency-boundary-note">这些值来自低频统计窗口，不在采集推理热路径中为 UI 增加工作。</p>}
              />
              <KvCard
                title="测量边界"
                rows={[
                  ["推理链耗时", "包含预处理、模型推理与结果解析"],
                  ["结果帧龄", "当前时刻 − 最新识别结果的采集时间"],
                  ["结果 FPS", "统计窗口内已发布识别结果增量 / 实际秒数"],
                  ["未提供", "采集、解码、ROI、排队、控制计算的独立阶段耗时"]
                ]}
                notice={<p className="latency-boundary-note">未打点的阶段不展示 0、不估算，也不拼成所谓完整链路。</p>}
              />
            </div>
          </>
          )}
          </section>
        ) : null}
      </main>

      {runtimeLifecycleActive ? (
        <button
          aria-label="紧急停止主链"
          className="console-button danger studio-mobile-emergency-stop"
          disabled={emergencyStopping}
          onClick={() => void emergencyStopMainline()}
          type="button"
        >
          <NovaIcon name="emergency-stop" size={17} />
          {emergencyStopping ? "紧急停止确认中" : "紧急停止"}
        </button>
      ) : null}

      {algorithmSettingsDialogOpen ? (
      <AdvancedSettingsDialog
        dirty={configDialogDirty}
        eyebrow="算法配置"
        footerNote={controlModeLabel}
        onClose={() => void requestDismissConfigDialog("algorithm")}
        onSave={() => void saveConfigDialog("algorithm")}
        open
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="控制参数"
      >
        <div className="algorithm-settings-layout">
          <nav aria-label="控制算法调参分类" className="algorithm-settings-nav" role="tablist">
            {ALGORITHM_SETTINGS_SECTIONS.map((section) => {
              const selected = algorithmSettingsSection === section.id;
              return (
                <button type="button"
                  aria-controls={section.panelId}
                  aria-selected={selected}
                  className={selected ? "active" : ""}
                  id={`${section.panelId}-tab`}
                  key={section.id}
                  onClick={() => focusAlgorithmSettingsSection(section.id)}
                  onKeyDown={handleAlgorithmSettingsTabKeyDown}
                  role="tab"
                  tabIndex={selected ? 0 : -1}
                >
                  <b>{section.label}</b>
                </button>
              );
            })}
          </nav>

          {algorithmSettingsSection === "response" ? (
            <section aria-labelledby="algorithm-settings-response-tab" className="algorithm-settings-panel" id="algorithm-settings-response" role="tabpanel" tabIndex={0}>
              <div className="advanced-settings-grid two-column">
                {responseParameters.map(renderAlgorithmNumberParameter)}
              </div>
            </section>
          ) : null}

          {algorithmSettingsSection === "prediction" ? (
            <section aria-labelledby="algorithm-settings-prediction-tab" className="algorithm-settings-panel" id="algorithm-settings-prediction" role="tabpanel" tabIndex={0}>
              {controlPredictionEnabled ? (
                <>
                  <div className="advanced-settings-grid two-column">
                    {predictionCoreParameters
                      .filter((parameter) => parameter.key === "prediction_lead_ms")
                      .map(renderAlgorithmNumberParameter)}
                    {predictionCapParameters.map(renderAlgorithmNumberParameter)}
                  </div>
                  <details className="algorithm-settings-disclosure">
                    <summary><span><b>高级预测参数</b></span><i>{predictionCoreParameters.length - 1} 项</i></summary>
                    <div className="advanced-settings-grid two-column">
                      {predictionCoreParameters
                        .filter((parameter) => parameter.key !== "prediction_lead_ms")
                        .map(renderAlgorithmNumberParameter)}
                    </div>
                  </details>
                </>
              ) : (
                <div className="algorithm-settings-empty"><b>预测已关闭</b></div>
              )}
            </section>
          ) : null}

          {algorithmSettingsSection === "calibration" ? (
            <section aria-labelledby="algorithm-settings-calibration-tab" className="algorithm-settings-panel" id="algorithm-settings-calibration" role="tabpanel" tabIndex={0}>
              <div className="advanced-settings-grid two-column">
                {calibrationParameters.map(renderAlgorithmNumberParameter)}
              </div>
            </section>
          ) : null}
        </div>
      </AdvancedSettingsDialog>
      ) : null}

      {targetAdvancedDialogOpen ? (
      <AdvancedSettingsDialog
        dirty={configDialogDirty}
        eyebrow="目标选择"
        footerNote="目标行为"
        onClose={() => void requestDismissConfigDialog("target-advanced")}
        onSave={() => void saveConfigDialog("target-advanced")}
        open
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="目标行为"
      >
        <div className="advanced-settings-grid two-column">
          {targetAdvancedParameters.map(renderTargetingNumberParameter)}
        </div>
      </AdvancedSettingsDialog>
      ) : null}

      {trackerSettingsDialogOpen ? (
      <AdvancedSettingsDialog
        dirty={configDialogDirty}
        eyebrow="目标跟踪"
        footerNote="专家跟踪参数"
        onClose={() => void requestDismissConfigDialog("tracker")}
        onSave={() => void saveConfigDialog("tracker")}
        open
        saveError={dialogSaveError}
        saving={dialogSaving}
        title="专家跟踪参数"
      >
        <div className="advanced-settings-grid two-column">
          {trackerCoreParameters.map(renderTargetingNumberParameter)}
        </div>
        <details className="algorithm-settings-disclosure">
          <summary><span><b>卡尔曼参数</b></span><i>{trackerKalmanParameters.length} 项</i></summary>
          <div className="advanced-settings-grid two-column">
            {trackerKalmanParameters.map(renderTargetingNumberParameter)}
          </div>
        </details>
      </AdvancedSettingsDialog>
      ) : null}

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
                <span className="class-config-eyebrow">目标选择</span>
                <h2 id="target-weight-dialog-title">选择权重</h2>
              </div>
              <button type="button"
                aria-label="关闭权重调整"
                className="launch-dialog-close"
                disabled={dialogSaving}
                onClick={() => void requestDismissConfigDialog("target-weights")}
                title={configDialogDirty ? "关闭并放弃本弹窗修改" : "关闭"}
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
                  <ParameterNumberControl
                    compact
                    label="类别偏好比例"
                    detail="类别顺序按 1、0.5、0.25…递减；距离自动使用剩余比例。提高后更倾向高优先类别。"
                    value={normalizedSelectionClassWeight}
                    min={0}
                    max={1}
                    step={0.01}
                    onCommit={(value) => updatePipelineField("target_selection_class_ratio", value)}
                    onEditingChange={handleParameterEditingChange}
                  />
                </div>
              </section>

            </div>

            <footer className="target-weight-dialog-footer">
              <span className={dialogSaveError ? "dialog-save-status error" : configDialogDirty ? "dialog-save-status dirty" : "dialog-save-status"} role="status" aria-live="polite">
                {dialogSaving
                  ? "正在处理…"
                  : dialogSaveError
                    ? dialogSaveError
                    : configDialogDirty
                      ? "修改尚未加入页面草稿"
                      : "未修改"}
              </span>
              <button type="button"
                className={`console-button ${configDialogDirty ? "primary dialog-save-button" : "dialog-close-button"}`}
                disabled={dialogSaving}
                onClick={() => void saveConfigDialog("target-weights")}
              >
                {configDialogDirty ? "加入草稿并关闭" : "关闭"}
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
              <button type="button"
                aria-label="关闭类别配置"
                className="launch-dialog-close"
                disabled={dialogSaving}
                onClick={() => void requestDismissConfigDialog("class-config")}
                title={configDialogDirty ? "关闭并放弃本弹窗修改" : "关闭"}
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
                    <button type="button"
                      aria-current={name === activeDetectionProfile ? "page" : undefined}
                      className={name === activeDetectionProfile ? "active" : ""}
                      key={name}
                      onClick={() => void updateConfigField("inference", "detection_class_profile", name)}
                    >
                      <span>{name}</span>
                      <small>{(detectionProfiles[name] ?? []).filter(Boolean).length} 类</small>
                    </button>
                  ))}
                </div>
                <div className="class-profile-create">
                  <TextControl
                    label="新建配置"
                    detail="输入新配置名后创建空白配置，或复制当前配置作为起点。"
                    placeholder="例如 valorant"
                    value={newClassProfileName}
                    disabled={configDialogSaving}
                    onDraftChange={setNewClassProfileName}
                    onCommit={setNewClassProfileName}
                    onEnter={() => createClassProfile(false)}
                  />
                  <div className="class-profile-create-actions">
                    <button type="button"
                      className="console-button primary"
                      disabled={busy !== null || newClassProfileName.trim() === ""}
                      onClick={() => void createClassProfile(false)}
                    >
                      新建空白
                    </button>
                    <button type="button"
                      className="console-button"
                      disabled={busy !== null || newClassProfileName.trim() === ""}
                      onClick={() => void createClassProfile(true)}
                    >
                      <NovaIcon name="copy" size={15} />
                      复制当前
                    </button>
                  </div>
                </div>
              </aside>

              <div className="class-config-workspace">
                <div className="class-config-profile-bar">
                  <TextControl
                    label="当前配置名称"
                    detail="重命名会同步迁移该配置对应的类别瞄点类型映射。"
                    value={renamedClassProfileName}
                    disabled={configDialogSaving}
                    onDraftChange={setRenamedClassProfileName}
                    onCommit={setRenamedClassProfileName}
                    onEnter={renameClassProfile}
                  />
                  <button type="button"
                    className="console-button"
                    disabled={busy !== null || renamedClassProfileName.trim() === "" || renamedClassProfileName.trim() === activeDetectionProfile}
                    onClick={() => void renameClassProfile()}
                  >
                    <NovaIcon name="edit" size={15} />
                    重命名
                  </button>
                  <button type="button"
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
                  >
                    <NovaIcon name="delete" size={15} />
                    {classProfileDeleteArmed ? "确认删除" : "删除"}
                  </button>
                </div>

                <AimTargetRange
                  disabled={configDialogSaving}
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
                      <button type="button"
                        className="console-button secondary"
                        disabled={busy !== null || selectedDetectionClassIds.size === classEditorIds.length}
                        onClick={() => void updateDetectionClassFilter("all")}
                      >
                        全部选择
                      </button>
                      <button type="button"
                        className="console-button"
                        disabled={busy !== null || selectedDetectionClassIds.size === 0}
                        onClick={() => void updateDetectionClassFilter("none")}
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
                        <button type="button"
                          aria-pressed={selected}
                          className={selected ? "selected" : ""}
                          disabled={busy !== null}
                          key={`class-filter-${classId}`}
                          onClick={() => void toggleDetectionClass(classId)}
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
                        <button type="button"
                          aria-label={`${selectedDetectionClassIds.has(classId) ? "取消" : "允许"} cls ${classId} 参与目标选择`}
                          aria-pressed={selectedDetectionClassIds.has(classId)}
                          className={`class-role-select ${selectedDetectionClassIds.has(classId) ? "selected" : ""}`}
                          disabled={busy !== null}
                          onClick={() => void toggleDetectionClass(classId)}
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
                        <div className="class-priority-control" aria-label={`cls ${classId} 目标优先级，第 ${priorityIndex + 1} 位`}>
                          <span className="class-priority-rank">#{priorityIndex + 1}</span>
                          <button type="button"
                            aria-label={`提高 cls ${classId} 目标优先级`}
                            className="class-priority-button up"
                            disabled={busy !== null || priorityIndex <= 0}
                            onClick={() => void setClassPriorityPosition(classId, priorityIndex - 1)}
                            title="上移"
                          >
                            <NovaIcon name="expand" size={14} />
                          </button>
                          <button type="button"
                            aria-label={`降低 cls ${classId} 目标优先级`}
                            className="class-priority-button down"
                            disabled={busy !== null || priorityIndex >= orderedClassEditorIds.length - 1}
                            onClick={() => void setClassPriorityPosition(classId, priorityIndex + 1)}
                            title="下移"
                          >
                            <NovaIcon name="expand" size={14} />
                          </button>
                        </div>
                        <div className="class-role-segmented" role="group" aria-label={`cls ${classId} 瞄点类型`}>
                          {(["head", "body", "other"] as AimRole[]).map((role) => (
                            <button type="button"
                              aria-pressed={(activeClassRoles[String(classId)] ?? "other") === role}
                              className={`${role} ${(activeClassRoles[String(classId)] ?? "other") === role ? "active" : ""}`}
                              disabled={busy !== null}
                              key={role}
                              onClick={() => void updateClassAimRole(classId, role)}
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
                  ? "正在处理类别配置…"
                  : dialogSaveError
                    ? `处理失败 · ${dialogSaveError}`
                    : configDialogDirty
                      ? "有未确认修改 · 加入页面草稿后仍需点击“保存修改”。"
                      : `未修改 · 当前配置：${activeDetectionProfile}`}
              </span>
              <button type="button"
                className={`console-button ${configDialogDirty ? "primary dialog-save-button" : "dialog-close-button"}`}
                disabled={dialogSaving}
                onClick={() => void saveConfigDialog("class-config")}
              >
                {configDialogDirty ? "加入草稿并关闭" : "关闭"}
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
                <span>系统诊断</span>
                <h2 id="error-center-title">异常信息</h2>
                <p>页面保持安静；网络、服务与操作错误统一收拢在这里。</p>
              </div>
              <button type="button" aria-label="关闭异常信息" onClick={() => setErrorCenterOpen(false)}              >
                <NovaIcon name="x-circle" size={18} />
              </button>
            </header>
            <div className="error-center-body">
              {currentErrorDetails.length > 0 ? currentErrorDetails.map((item) => (
                <article className="error-center-item" key={item.key}>
                  <NovaIcon name="triangle-alert" size={17} />
                  <div>
                    <strong>{item.title}</strong>
                    {(item.count ?? 1) > 1 ? <small>本会话重复 {item.count} 次</small> : null}
                    <p>{item.detail}</p>
                    {item.time ? <time>{formatDate(new Date(item.time))}</time> : null}
                  </div>
                </article>
              )) : (
                <div className="error-center-empty">
                  <NovaIcon name="shield-check" size={28} />
                  <strong>当前没有异常</strong>
                  <span>服务连接和最近操作均未报告错误。</span>
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
                确认并清空本会话记录
              </button>
              <button type="button" className="console-button primary" onClick={() => setErrorCenterOpen(false)}              >完成</button>
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
          switchMessage: modelSwitch.message,
          busy,
          canSwitch: selectedCatalogModel?.kind === "engine",
          parserPreset,
          onParserPresetChange: setParserPreset,
          onRefresh: () => void refreshModelCatalog(),
          onSelectModel: selectModelFromCatalog,
          onSaveMetadata: (recommendation, tags) => void modelSwitch.saveMetadata(recommendation, tags),
          onSwitch: modelSwitch.switchModel
            }}
          />
        </Suspense>
      ) : null}

      {modelSwitch.dialogOpen ? (
      <ModelSwitchDialog
        completedStages={modelSwitch.completedStages}
        currentStage={modelSwitch.stageIndex}
        detail={modelSwitch.progressDetail}
        dialogRef={modelSwitch.dialogRef}
        error={modelSwitch.dialogError}
        modelName={selectedCatalogModel?.relative_path ?? "TensorRT Engine"}
        onClose={modelSwitch.closeDialog}
        open
        status={modelSwitch.dialogStatus}
      />
      ) : null}

      {confirmationRequest ? (
      <ActionConfirmationDialog
        busy={confirmationBusy}
        error={confirmationError}
        onCancel={() => {
          if (!confirmationBusy) setConfirmationRequest(null);
        }}
        onConfirm={() => void confirmPendingAction()}
        request={confirmationRequest}
      />
      ) : null}

    </section>
  );
}

function ModuleSwitch({
  label,
  detail,
  compact = false,
  enabled,
  disabled = false,
  optimistic = true,
  onToggle
}: {
  label: string;
  detail?: string;
  compact?: boolean;
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
    <button type="button"
      aria-busy={pending}
      aria-pressed={visualEnabled}
      className={`module-switch${visualEnabled ? " on" : ""}${compact ? " compact" : ""}`}
      onClick={() => void toggle()}
      disabled={pending || disabled}
    >
      <span>
        <b>{label}</b>
        {detail ? <small>{detail}</small> : null}
      </span>
      <i aria-hidden="true">{pending ? "处理中" : visualEnabled ? "开" : "关"}</i>
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
  target: RuntimeVisionTargetState | null;
  updatedAtMs: number;
};

function useStablePreviewOverlay(
  detections: PreviewDetection[],
  target: RuntimeVisionTargetState | null
): { detections: PreviewDetection[]; target: RuntimeVisionTargetState | null } {
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

function readPreviewDetections(value: RuntimeVisionDetectionState[]): PreviewDetection[] {
  return value.flatMap((item, index) => {
    if (item.w <= 0 || item.h <= 0) {
      return [];
    }
    return [{
      x: item.x,
      y: item.y,
      w: item.w,
      h: item.h,
      cx: item.cx,
      cy: item.cy,
      score: item.score,
      className: `类别${item.class_id}`,
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
  const configVersion = runtime?.config.version ?? 0;
  const vision = runtime?.vision;
  const inferenceTrace = vision?.inference;
  const control = vision?.control;
  const previewWidth = inferenceTrace?.input_width ?? roiSize;
  const previewHeight = inferenceTrace?.input_height ?? roiSize;
  const displaySize = Math.max(previewWidth, previewHeight, roiSize);
  const liveDetections = useMemo(
    () => readPreviewDetections(vision?.detection_items ?? []),
    [vision?.detection_items]
  );
  const liveTarget = vision?.target ?? null;
  const overlay = useStablePreviewOverlay(liveDetections, liveTarget);
  const detections = overlay.detections;
  const target = overlay.target;
  const mouseObservation = control?.mouse_observation;
  const targetDetectionIndex = target?.target_detection_index ?? null;
  const targetCx = target?.cx ?? null;
  const targetCy = target?.cy ?? null;
  const targetAimX = mouseObservation?.predicted_x_px
    ?? targetCx;
  const targetAimY = mouseObservation?.predicted_y_px
    ?? targetCy;
  const showImage = supported && active && imageAvailable && !streamFailure;
  const showOverlay = supported && active && runtime?.running === true && detections.length > 0;
  const selectedDetection = detections.find((item) => (
    targetDetectionIndex !== null
      ? item.index === targetDetectionIndex
      : targetCx !== null && targetCy !== null && Math.abs(item.cx - targetCx) <= 2 && Math.abs(item.cy - targetCy) <= 2
  ));
  const selectedIndex = selectedDetection?.index ?? null;
  const sourceWidth = inferenceTrace?.source_width ?? 0;
  const sourceHeight = inferenceTrace?.source_height ?? 0;
  const roiOffsetX = inferenceTrace?.roi_offset_x ?? 0;
  const roiOffsetY = inferenceTrace?.roi_offset_y ?? 0;
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
          <ParameterPresetControl
            density="compact"
            label="实时预览"
            detail={`上限 ${previewFps}fps`}
            options={[5, 10, 15, 30].filter((fps) => fps <= previewFps).map((fps) => ({
              id: `transport-${fps}`,
              label: `${fps}`,
              detail: "fps",
              active: transportFps === fps,
              onSelect: () => setTransportFps(fps)
            }))}
          />
          <button type="button" disabled={togglePending} onClick={() => onToggle(false)}              >
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
            <p>推理、跟踪与控制继续运行。开启画面会占用图像编码与内存带宽。</p>
            <button type="button" disabled={togglePending} onClick={() => onToggle(true)}              >
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
