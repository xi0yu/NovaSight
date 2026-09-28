import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties
} from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureSelectPayload,
  getApiErrorCode,
  connectKmNet,
  diagnosticMoveKmNet,
  getConfigSchema,
  getRuntimeConfig,
  HealthResponse,
  LicenseStatus,
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelCatalogResponse,
  ModelRecommendation,
  ModelProject,
  ModelVersion,
  ParserPresetId,
  RuntimeConfig,
  ConfigSchemaResponse,
  RuntimeConfigValue,
  RuntimeState,
  type RuntimeVisionDetectionState,
  type RuntimeVisionTargetState,
  getCaptureCapabilities,
  getCaptureDevices,
  getModelArtifacts,
  getModelCatalog,
  createCatalogFolder,
  moveCatalogEngine,
  getModelVersions,
  selectCaptureProfile,
  setCapturePreviewEnabled,
  streamUrl,
  updateRuntimeConfig,
} from "../../api";
import { pushToastRaw, reportError, reportSuccess, useActivityNotices, useClearActivityNotices, useClearErrorNotices, useErrorNotices } from "../../lib/toast";
import { formatRuntimeErrorMessage, getErrorMessage } from "../shared/format";
import { LicenseView } from "../license/LicenseView";
import { ActivityView } from "../activity/ActivityView";
import { useActivityStream } from "../activity/useActivityStream";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";
import { useStableSemanticValue } from "../shared/useStableSemanticValue";
import { NovaIcon } from "../../components/visual";
import { moduleOrderForPage, modulesForPage } from "../../contracts/studioLayout";
import { CurrentModelSummary } from "../models/CurrentModelSummary";
import { ModelSwitchDialog } from "../models/ModelSwitchDialog";
import {
  runtimeDeliveryDescription,
  runtimeDeliveryLabel,
  runtimeDeliveryTone,
  type RuntimeDeliveryStatus
} from "../shared/runtimeDelivery";
import {
  ActionConfirmationDialog,
  type ActionConfirmationRequest
} from "./ActionConfirmationDialog";
import { AimTargetRange } from "./AimTargetRange";
import type { AimRole, AimRoleRatios } from "../targeting/types";
import { ControlTracePanel } from "./ControlTracePanel";
import { ModuleSwitch } from "./ControlSwitches";
import {
  InlineNumberControl,
  InlineTextControl,
  ParameterNumberControl,
  ParameterPresetControl,
  TextControl
} from "./StudioControls";
import {
  buildAlgorithmParameterGroups,
  buildConfigFieldIndex,
  buildTargetingParameterGroups,
  validateStudioConfigSchema,
  type AlgorithmNumberParameter,
  type ControlPipelineField,
  type TargetingNumberParameter,
  type TargetingPipelineField
} from "./algorithmParameterModel";
import { CONSOLE_PAGES, DEFAULT_CONSOLE_PAGE, StudioNavigation, type ConsolePage } from "./StudioNavigation";
import { StudioPageHeader } from "./StudioPageHeader";
import { KvCard, Metric, SectionTitle, WorkspaceNotice } from "./StudioPresentation";
import { StudioRuntimeBar } from "./StudioRuntimeBar";
import { SettingsView } from "./SettingsView";
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "./dialogFocus";
import {
  buildLaunchReadiness,
  type LaunchReadinessAction
} from "./launchReadiness";
import { useMainlineLaunch } from "./useMainlineLaunch";
import { useConfigApplyPresentation } from "./useConfigApplyPresentation";
import { useModelSwitchWorkflow } from "./useModelSwitchWorkflow";
import { useStudioLayout } from "./useStudioLayout";
import { RuntimeOverviewView } from "./RuntimeOverviewView";
import {
  AboutView,
  DeviceStatusView,
  HomeSetupPrompt,
  ManagementView,
  OnboardingView,
  type SetupState
} from "./ProductJourneyViews";
import {
  projectRuntimeState,
  type RuntimeRecoveryAction,
  type RuntimeTransportConfidence,
} from "../runtime/runtimeProjection";
import { buildControlTrace } from "./controlTrace";
import { persistRuntimeConfigField } from "./runtimeConfigPersistence";
import "./studio-settings.css";

const CONTROL_ALGORITHM_LABEL = "连续 Atan 控制";
const CONFIG_SCHEMA_CONTRACT_ERROR_PREFIX = "配置 schema 与 Studio 参数不一致";
type KmnetTestMessageTone = "success" | "warning";
type ParameterPageFieldChange = {
  section: "control" | "inference" | "pipeline";
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
  onLicenseChange: (license: LicenseStatus) => void;
  onRefresh: () => Promise<void>;
  onRuntimeConfigChange: (config: RuntimeConfig) => void;
  onRuntimeStateChange: (runtime: RuntimeState) => boolean;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type CapabilityFormatGroup = {
  pixelFormat: string;
  resolutions: Array<{
    width: number;
    height: number;
    choices: CapabilityChoice[];
  }>;
};

type ConfigDialogId = "class-config";

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

function recordStrings(value: unknown): Record<string, string> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).flatMap(([key, item]) =>
      typeof item === "string" ? [[key, item]] : []
    )
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

function serializeClassAimRatios(
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
        right.width * right.height - left.width * left.height ||
        right.width - left.width ||
        right.fps - left.fps
      );
    });
}

function groupCapabilityChoices(choices: CapabilityChoice[]): CapabilityFormatGroup[] {
  const formats = new Map<string, Map<string, CapabilityChoice[]>>();
  for (const choice of choices) {
    const resolutions = formats.get(choice.pixel_format) ?? new Map<string, CapabilityChoice[]>();
    const key = `${choice.width}x${choice.height}`;
    resolutions.set(key, [...(resolutions.get(key) ?? []), choice]);
    formats.set(choice.pixel_format, resolutions);
  }
  return Array.from(formats, ([pixelFormat, resolutions]) => ({
    pixelFormat,
    resolutions: Array.from(resolutions.values(), (items) => ({
      width: items[0].width,
      height: items[0].height,
      choices: items,
    })),
  }));
}

function canonicalCaptureFormat(value: string): string {
  const normalized = value.trim().toUpperCase();
  if (normalized === "MJPEG") return "MJPG";
  if (normalized === "YUY2") return "YUYV";
  return normalized;
}

function choiceLabel(choice: CapabilityChoice): string {
  const format = canonicalCaptureFormat(choice.pixel_format) === "MJPG" ? "MJPEG (MJPG)" : choice.pixel_format;
  return `${format} / ${choice.width}x${choice.height} / ${choice.fps} FPS`;
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

function preserveOutputGate(candidate: RuntimeConfig, current: RuntimeConfig): RuntimeConfig {
  const next = normalizeRuntimeConfig(candidate);
  const control = { ...asRecord(next.control) };
  const outputEnabled = asRecord(current.control).output_enabled;
  if (outputEnabled === undefined) delete control.output_enabled;
  else control.output_enabled = outputEnabled;
  next.control = control as RuntimeConfig[string];
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

function collapseClassProfiles(config: RuntimeConfig): RuntimeConfig {
  const next = structuredClone(config) as RuntimeConfig;
  const inference = asRecord(next.inference);
  const profile = readString(inference.detection_class_profile, "default");
  const profiles = recordList(inference.detection_class_profiles);
  const priorities = recordStrings(inference.detection_class_priorities);
  const filters = recordStrings(inference.detection_class_filters);
  const control = asRecord(next.control);
  const aim = nestedRecord(control, "aim");
  const roles = profileRoleRecords(aim.class_roles);
  const classes = profiles[profile] ?? profiles.default ?? [];
  const priority = priorities[profile]
    ?? readString(inference.detection_class_priority, "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
  const filter = filters[profile] ?? readString(inference.detection_class_filter, "all");

  next.inference = {
    ...inference,
    detection_class_profile: "default",
    detection_class_profiles: { default: classes },
    detection_class_priority: priority,
    detection_class_priorities: { default: priority },
    detection_class_filter: filter,
    detection_class_filters: { default: filter }
  } as RuntimeConfig[string];
  next.control = {
    ...control,
    aim: {
      ...aim,
      class_roles: { default: roles[profile] ?? roles.default ?? {} }
    }
  } as RuntimeConfig[string];
  return next;
}

function parameterPageFieldChanges(
  baseline: RuntimeConfig,
  draft: RuntimeConfig
): ParameterPageFieldChange[] {
  const allowedSections = new Set(["control", "inference", "pipeline"]);
  const ignoredSections = new Set(["revision", "version", "roi_size"]);
  const unsupportedSections = Array.from(new Set([...Object.keys(baseline), ...Object.keys(draft)]))
    .filter((section) => !ignoredSections.has(section))
    .filter((section) => !allowedSections.has(section))
    .filter((section) => !runtimeConfigValuesEqual(baseline[section], draft[section]));
  if (unsupportedSections.length > 0) {
    throw new Error(`参数页包含不支持保存的配置区：${unsupportedSections.join("、")}`);
  }

  const baselineControl = asRecord(baseline.control);
  const draftControl = asRecord(draft.control);
  const supportedControlKeys = new Set(["recoil", "aim"]);
  const unsupportedControlKeys = Array.from(new Set([
    ...Object.keys(baselineControl),
    ...Object.keys(draftControl)
  ]))
    .filter((key) => !supportedControlKeys.has(key))
    .filter((key) => !runtimeConfigValuesEqual(baselineControl[key], draftControl[key]));
  if (unsupportedControlKeys.length > 0) {
    throw new Error(`参数页包含不支持保存的控制字段：${unsupportedControlKeys.join("、")}`);
  }

  const recoilChange = runtimeConfigValuesEqual(baselineControl.recoil, draftControl.recoil)
    ? null
    : {
        section: "control" as const,
        key: "recoil",
        value: draftControl.recoil as RuntimeConfigValue
      };
  const aimChange = runtimeConfigValuesEqual(baselineControl.aim, draftControl.aim)
    ? null
    : {
        section: "control" as const,
        key: "aim",
        value: draftControl.aim as RuntimeConfigValue
      };

  const baselineInference = asRecord(baseline.inference);
  const draftInference = asRecord(draft.inference);
  const inferenceChanges = Array.from(new Set([
    ...Object.keys(baselineInference),
    ...Object.keys(draftInference)
  ]))
    .sort()
    .filter((key) => !runtimeConfigValuesEqual(baselineInference[key], draftInference[key]))
    .map((key) => ({
      section: "inference" as const,
      key,
      value: draftInference[key] as RuntimeConfigValue
    }));

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
  if (aimChange) changes.push(aimChange);
  changes.push(...inferenceChanges);
  changes.push(...pipelineChanges);
  if (recoilChange) {
    changes.push(recoilChange);
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
  onLicenseChange,
  onRefresh,
  onRuntimeConfigChange,
  onRuntimeStateChange
}: StudioConsoleViewProps) {
  const [activePage, setActivePage] = useState<ConsolePage>(() => pageFromUrl());
  const { events: serverActivityEvents, clear: clearServerActivity } = useActivityStream();
  const studioLayout = useStudioLayout();
  const overviewModules = modulesForPage(studioLayout, "overview");
  const captureModules = modulesForPage(studioLayout, "capture");
  const captureModuleOrder = moduleOrderForPage(studioLayout, "capture");
  const parameterModuleOrder = moduleOrderForPage(studioLayout, "params");
  const activityModules = moduleOrderForPage(studioLayout, "activity");
  const overviewModuleOrder = moduleOrderForPage(studioLayout, "overview");
  const settingsModules = moduleOrderForPage(studioLayout, "settings");
  const aboutModules = moduleOrderForPage(studioLayout, "about");
  const activePageRef = useRef(activePage);
  const pageScrollPositionsRef = useRef<Partial<Record<ConsolePage, number>>>({});

  useEffect(() => {
    if (activePage === "infer") {
      void loadModelManagerDialog();
      return;
    }
    if (activePage === "models" || activePage === "management") {
      void import("../models/ModelWorkspace");
      void onEnsureProjects(false).catch(() => undefined);
    }
  }, [activePage, onEnsureProjects]);
  const [device, setDevice] = useState(
    readString(nestedRecord(runtimeConfig, "capture").device)
  );

  const [caps, setCaps] = useState<CaptureCapabilitiesResponse | null>(null);
  const [captureDevices, setCaptureDevices] = useState<string[]>([]);
  const [deviceDiscoveryStatus, setDeviceDiscoveryStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [deviceDiscoveryError, setDeviceDiscoveryError] = useState<string | null>(null);
  const autoQueriedCaptureDeviceRef = useRef("");
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
  const [modelCatalogError, setModelCatalogError] = useState<string | null>(null);
  const [selectedModelCatalogPath, setSelectedModelCatalogPath] = useState<string>();
  const [kmnetTestDx, setKmnetTestDx] = useState(10);
  const [kmnetTestDy, setKmnetTestDy] = useState(0);
  const [kmnetTestMessage, setKmnetTestMessage] = useState("");
  const [kmnetTestMessageTone, setKmnetTestMessageTone] = useState<KmnetTestMessageTone>("warning");
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [captureActionError, setCaptureActionError] = useState<string | null>(null);
  const [errorCenterOpen, setErrorCenterOpen] = useState(false);
  const errorNotices = useErrorNotices();
  const activityNotices = useActivityNotices();
  const clearActivityNotices = useClearActivityNotices();
  const clearErrorNotices = useClearErrorNotices();
  const lastRuntimeFaultRef = useRef("");
  const [modelManagerDialogOpen, setModelManagerDialogOpen] = useState(false);
  const [modelCatalogMessage, setModelCatalogMessage] = useState("");
  const [classConfigDialogOpen, setClassConfigDialogOpen] = useState(false);
  const [confirmationRequest, setConfirmationRequest] = useState<ActionConfirmationRequest | null>(null);
  const [confirmationBusy, setConfirmationBusy] = useState(false);
  const [confirmationError, setConfirmationError] = useState<string | null>(null);
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
  const [configDialogSaving, setConfigDialogSaving] = useState(false);
  const dialogSaving = configDialogSaving;
  const [parameterPageDirty, setParameterPageDirty] = useState(false);
  const [parameterPageSaving, setParameterPageSaving] = useState(false);
  const [dialogSaveError, setDialogSaveError] = useState<string | null>(null);
  const configFieldIndex = useMemo(() => buildConfigFieldIndex(configSchema), [configSchema]);
  const mainRef = useRef<HTMLElement | null>(null);
  const configDraftRef = useRef<RuntimeConfig | null>(cloneRuntimeConfig(runtimeConfig));
  const runtimeConfigLatestRef = useRef<RuntimeConfig | null>(runtimeConfig);
  const activeConfigDialogRef = useRef<ConfigDialogId | null>(null);
  const configDialogBaselineRef = useRef<RuntimeConfig | null>(null);
  const parameterPageBaselineRef = useRef<RuntimeConfig | null>(null);
  const rememberStudioScroll = useCallback((page: ConsolePage) => {
    pageScrollPositionsRef.current[page] = mainRef.current?.scrollTop ?? 0;
  }, []);
  const restoreStudioScroll = useCallback((page: ConsolePage) => {
    window.requestAnimationFrame(() => {
      mainRef.current?.scrollTo({ top: pageScrollPositionsRef.current[page] ?? 0, left: 0 });
      window.scrollTo({ top: 0, left: 0 });
    });
  }, []);
  const parameterPageDirtyRef = useRef(false);
  const confirmationBusyRef = useRef(false);
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
  const errorCenterDialogRef = useRef<HTMLElement | null>(null);
  const dialogSavingRef = useRef(false);

  const setConfigDialogVisibility = useCallback((dialog: ConfigDialogId, open: boolean) => {
    if (dialog === "class-config") setClassConfigDialogOpen(open);
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

  const requestDiscardParameterPageDraft = useCallback(() => {
    setConfirmationRequest({
      eyebrow: "未保存修改",
      title: "放弃未保存的修改？",
      description: "当前页面的未保存参数会被丢弃，恢复为最新已保存配置。此操作不能撤销。",
      confirmLabel: "确认放弃修改",
      danger: true,
      onConfirm: discardParameterPageDraft
    });
  }, [discardParameterPageDraft]);

  const applyConfigSchema = useCallback((schema: ConfigSchemaResponse) => {
    const issues = validateStudioConfigSchema(schema);
    setConfigSchema(schema);
    if (issues.length > 0) {
      const summary = issues
        .slice(0, 3)
        .map((issue) => `${issue.path}: ${issue.reason}`)
        .join("；");
      const message = `${CONFIG_SCHEMA_CONTRACT_ERROR_PREFIX}：${summary}`;
      setLocalError(message);
      reportError(new Error(message), { source: "config-schema", title: "参数契约不一致", popup: false });
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
          reportError(error, { source: "config-schema", title: "配置 schema 读取失败", popup: false });
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
    setConfigDialogVisibility(dialog, true);
  }, [setConfigDialogVisibility]);

  const saveConfigDialog = useCallback(async (
    dialog: ConfigDialogId,
    physicalOutputAcknowledged = false
  ) => {
    if (dialogSavingRef.current || activeConfigDialogRef.current !== dialog) {
      return;
    }

    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
      await Promise.resolve();
    }

    const rawDraft = configDraftRef.current;
    const baseline = configDialogBaselineRef.current;
    if (!rawDraft || !baseline || runtimeConfigsEqual(rawDraft, baseline)) {
      configDraftRef.current = baseline ?? rawDraft;
      setConfigDraft(baseline ?? rawDraft);
      finishConfigDialog(dialog);
      return;
    }
    const draft = dialog === "class-config" ? collapseClassProfiles(rawDraft) : rawDraft;

    const outputIsActive = runtime?.presentation?.lifecycle?.can_stop === true
      && (runtime?.vision?.control?.output_enabled === true
        || readBoolean(asRecord((runtimeConfigLatestRef.current ?? baseline).control).output_enabled, false));
    if (outputIsActive && !physicalOutputAcknowledged) {
      setConfirmationRequest({
        eyebrow: "应用参数",
        title: "物理输出仍开启，确认应用这些设置？",
        description: "保存后会立即更新当前运行参数，设备的控制量可能随之改变。取消后修改会继续保留在当前窗口。",
        details: ["如需先暂停设备，请取消并回到首页关闭运行。"],
        confirmLabel: "确认保存并应用",
        danger: true,
        onConfirm: () => saveConfigDialog(dialog, true)
      });
      return;
    }

    let dialogChanges: ParameterPageFieldChange[];
    let preservedPageChanges: ParameterPageFieldChange[] = [];
    try {
      dialogChanges = parameterPageFieldChanges(baseline, draft);
      const pageBaseline = parameterPageBaselineRef.current;
      if (pageBaseline) {
        const dialogKeys = new Set(dialogChanges.map((change) => `${change.section}.${change.key}`));
        preservedPageChanges = parameterPageFieldChanges(pageBaseline, baseline)
          .filter((change) => !dialogKeys.has(`${change.section}.${change.key}`));
      }
    } catch (error) {
      setDialogSaveError(getErrorMessage(error));
      return;
    }

    dialogSavingRef.current = true;
    setConfigDialogSaving(true);
    setDialogSaveError(null);
    beginPendingConfigWrite();
    try {
      const canonical = cloneRuntimeConfig(runtimeConfigLatestRef.current);
      if (!canonical) {
        throw new Error("尚未读取运行配置。");
      }
      const payload = preserveOutputGate(
        applyParameterPageFieldChanges(canonical, dialogChanges),
        canonical
      );
      payload.revision = canonical.revision;
      const request = configWriteQueueRef.current.then(() =>
        updateRuntimeConfig(payload, physicalOutputAcknowledged)
      );
      configWriteQueueRef.current = request.then(() => undefined, () => undefined);
      const result = await request;
      if (!result.applied) {
        throw new Error(result.message || "后端未确认这些设置已经生效。");
      }
      const applied = normalizeRuntimeConfig(result.config);
      finalizeRuntimeConfigWrite(applied);
      if (preservedPageChanges.length > 0) {
        const remaining = applyParameterPageFieldChanges(applied, preservedPageChanges);
        remaining.revision = applied.revision;
        parameterPageBaselineRef.current = applied;
        configDraftRef.current = remaining;
        setConfigDraft(remaining);
        setParameterPageDirtyState(true);
      } else {
        parameterPageBaselineRef.current = null;
        setParameterPageDirtyState(false);
      }
      finishConfigDialog(dialog);
      reportSuccess(
        "设置已保存并应用",
        result.restart_required
          ? "运行参数已写入；进程级基础配置将在下次启动时接管。"
          : "后端已确认当前运行配置更新完成。",
        "config-dialog"
      );
    } catch (error) {
      const message = getErrorMessage(error);
      setDialogSaveError(message);
      reportError(error, {
        source: "config-dialog",
        title: "设置应用失败",
        publicDetail: message,
        popup: false
      });
    } finally {
      finishPendingConfigWrite();
      dialogSavingRef.current = false;
      setConfigDialogSaving(false);
    }
  }, [
    beginPendingConfigWrite,
    finalizeRuntimeConfigWrite,
    finishConfigDialog,
    finishPendingConfigWrite,
    runtime,
    setParameterPageDirtyState
  ]);

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

    setConfirmationRequest({
      eyebrow: "未保存修改",
      title: "关闭并放弃本次修改？",
      description: "当前窗口内的修改尚未保存。确认后会丢弃这些修改；参数页已有的未保存修改不受影响。",
      confirmLabel: "放弃弹窗修改",
      danger: true,
      onConfirm: () => {
        configDraftRef.current = baseline;
        setConfigDraft(baseline);
        finishConfigDialog(dialog);
      }
    });
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

  const requestClearErrorHistory = useCallback(() => {
    const ids = errorNotices.map((notice) => notice.id);
    if (ids.length === 0) return;
    setConfirmationRequest({
      eyebrow: "异常信息",
      title: "清空历史错误记录？",
      description: `将清除当前看到的 ${ids.length} 条历史记录，清除后无法恢复；仍在发生的运行故障会继续显示。`,
      confirmLabel: "确认清空历史",
      danger: true,
      onConfirm: () => clearErrorNotices(ids)
    });
  }, [clearErrorNotices, errorNotices]);

  const stageConfigDialogDraft = useCallback((next: RuntimeConfig) => {
    configDraftRef.current = next;
    setConfigDraft(next);
    setConfigDialogDirty(!runtimeConfigsEqual(configDialogBaselineRef.current, next));
    setDialogSaveError(null);
  }, []);

  useEffect(() => {
    if (!classConfigDialogOpen) {
      setDialogSaveError(null);
    }
  }, [classConfigDialogOpen]);

  useEffect(() => {
    writePageToUrl(activePage, "replace");
    const onPopState = () => {
      const nextPage = pageFromUrl();
      rememberStudioScroll(activePageRef.current);
      activePageRef.current = nextPage;
      setActivePage(nextPage);
      restoreStudioScroll(nextPage);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [activePage, rememberStudioScroll, restoreStudioScroll]);

  const navigatePage = useCallback((page: ConsolePage) => {
    if (page === activePage) {
      return;
    }
    rememberStudioScroll(activePage);
    activePageRef.current = page;
    setActivePage(page);
    writePageToUrl(page);
    restoreStudioScroll(page);
  }, [activePage, rememberStudioScroll, restoreStudioScroll]);

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

  const capture = runtime?.capture;
  const statistics = runtime?.statistics;
  const config = configDraft ?? runtimeConfig;
  const {
    captureConfig,
    roiConfig,
    limitsConfig,
    inferenceConfig,
    controlConfig,
    rustPipelineConfig,
    hardwareConfig,
    consumersConfig
  } = useMemo(() => ({
    captureConfig: nestedRecord(config, "capture"),
    roiConfig: nestedRecord(config, "roi"),
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
  const captureDeviceChoices = useMemo(() => configuredCaptureDevice && !captureDevices.includes(configuredCaptureDevice)
    ? [configuredCaptureDevice, ...captureDevices]
    : captureDevices, [captureDevices, configuredCaptureDevice]);
  const configuredRoiLeft = readNumber(captureConfig.roi_left, 0);
  const configuredRoiTop = readNumber(captureConfig.roi_top, 0);
  const configuredRoiWidth = readNumber(captureConfig.roi_width, 0);
  const configuredRoiHeight = readNumber(captureConfig.roi_height, 0);
  const rustControlPlane = Object.prototype.hasOwnProperty.call(
    rustPipelineConfig,
    "projection_fov_x_deg"
  );
  const vision = runtime?.vision;
  const inferenceTrace = vision?.inference;
  const runtimeInference = runtime?.inference;
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
  const configuredCaptureProfile =
    configuredCapturePixelFormat && configuredCaptureWidth > 0 && configuredCaptureHeight > 0 && configuredCaptureFps > 0
      ? {
          pixel_format: configuredCapturePixelFormat.toUpperCase(),
          width: configuredCaptureWidth,
          height: configuredCaptureHeight,
          fps: configuredCaptureFps
        }
      : null;
  const captureProfileMatches = capture?.running === true
    && capture.device === configuredCaptureDevice
    && selectedProfile != null
    && choiceMatchesConfig(selectedProfile, captureConfig);
  const captureProfileState = capture?.running !== true
    ? "pending"
    : configuredCaptureProfile == null || selectedProfile == null
      ? "unknown"
      : captureProfileMatches ? "matched" : "mismatch";
  const displayCaptureProfile = configuredCaptureProfile ?? selectedProfile ?? null;
  const displayCaptureProfileSource = configuredCaptureProfile
    ? "已保存配置"
    : selectedProfile?.source === "configured"
      ? "当前生效配置"
      : selectedProfile?.source || NO_SAMPLE;
  const detectedChoices = useMemo(() => groupCapabilities(caps?.capabilities ?? []), [caps]);
  const capabilityGroups = useMemo(() => groupCapabilityChoices(detectedChoices), [detectedChoices]);
  const roiSize = rustControlPlane
    ? configuredRoiWidth > 0 && configuredRoiHeight > 0
      ? Math.min(configuredRoiWidth, configuredRoiHeight)
      : 640
    : readNumber(roiConfig.size, 640);
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
  const detectionPriorityProfiles = useMemo(
    () => recordStrings(inferenceConfig.detection_class_priorities),
    [inferenceConfig.detection_class_priorities]
  );
  const detectionFilterProfiles = useMemo(
    () => recordStrings(inferenceConfig.detection_class_filters),
    [inferenceConfig.detection_class_filters]
  );
  const activeDetectionClass = readString(
    detectionFilterProfiles[activeDetectionProfile]
      ?? (rustControlPlane
        ? rustPipelineConfig.target_class_filter
        : inferenceConfig.detection_class_filter),
    "all"
  );
  const detectionClasses = detectionProfiles[activeDetectionProfile] ?? detectionProfiles.default ?? [];
  const detectionClassPriority = readString(
    detectionPriorityProfiles[activeDetectionProfile]
      ?? (rustControlPlane
        ? rustPipelineConfig.target_class_priority
        : inferenceConfig.detection_class_priority),
    "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  );
  const classPriorityIds = useMemo(
    () => parseClassPriority(detectionClassPriority),
    [detectionClassPriority]
  );
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
  const targetSelectionWeights = [
    { key: "target_selection_distance_weight", label: "距离", className: "distance", value: readNumber(rustPipelineConfig.target_selection_distance_weight, 0.45), detail: "越靠近准星，候选分越高。" },
    { key: "target_selection_class_weight", label: "内部类别", className: "class", value: readNumber(rustPipelineConfig.target_selection_class_weight, 0.20), detail: "使用当前模型配置中的类别优先顺序；raw cls 本身不锁定身份。" },
    { key: "target_selection_confidence_weight", label: "置信度", className: "confidence", value: readNumber(rustPipelineConfig.target_selection_confidence_weight, 0.15), detail: "本帧 YOLO 置信度越高，候选分越高。" },
    { key: "target_selection_size_weight", label: "大小", className: "size", value: readNumber(rustPipelineConfig.target_selection_size_weight, 0.05), detail: "画面中更大的目标获得少量加分。" },
    { key: "target_selection_continuity_weight", label: "连续性", className: "continuity", value: readNumber(rustPipelineConfig.target_selection_continuity_weight, 0.10), detail: "最近稳定关联过的目标获得加分。" },
    { key: "target_selection_motion_weight", label: "运动趋势", className: "motion", value: readNumber(rustPipelineConfig.target_selection_motion_weight, 0.05), detail: "短时间内向准星靠近的目标获得加分。" }
  ] as const;
  const targetSelectionWeightTotal = targetSelectionWeights.reduce(
    (total, item) => total + Math.max(0, item.value),
    0
  );
  const normalizedTargetSelectionWeights = targetSelectionWeights.map((item) => ({
    ...item,
    share: targetSelectionWeightTotal > 0 ? Math.max(0, item.value) / targetSelectionWeightTotal : 0
  }));
  const dominantTargetSelectionWeight = normalizedTargetSelectionWeights.reduce(
    (dominant, item) => item.share > dominant.share ? item : dominant,
    normalizedTargetSelectionWeights[0]
  );
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
  const trackerClassCostWeight = readNumber(rustPipelineConfig.tracker_class_cost_weight, 0.35);
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
  const targetSelectionMotionHorizonMs = readNumber(rustPipelineConfig.target_selection_motion_horizon_ms, 30);
  const recoilEnabled = readBoolean(recoilConfig.enabled, false);
  const recoilRequireTarget = readBoolean(recoilConfig.require_target, true);
  const recoilIntervalMs = readNumber(recoilConfig.interval_ms, 16);
  const fireDelayEnabled = readBoolean(rustPipelineConfig.fire_delay_enabled, false);
  const fireDelayMs = readNumber(rustPipelineConfig.fire_delay_ms, 0);
  const recoilYCounts = readNumber(recoilConfig.y_counts, 1);
  const hardwareTriggerRequired = readString(controlConfig.trigger_mode, "hardware") === "hardware";
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
  const desiredConfigRevision = configApplyPresentation.desiredRevision;
  const effectiveConfigRevision = configApplyPresentation.effectiveRevision;
  const kmnetRestartRequired = kmnetStatus?.restart_required === true;
  const hardwareControlLicensed = license?.valid === true && license.features.includes("hardware_control");
  const kmnetConfigurationState = kmnetStatus?.configuration_state
    ?? (kmnetRestartRequired ? "restart_required" : kmnetAutoConnect ? "ready" : "uncommissioned");
  const kmnetConfigurationReady = kmnetStatus?.configuration_ready === true
    || (kmnetConfigurationState === "ready" && !kmnetRestartRequired);
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
  const selectedPreviewArtifact = selectedCatalogModel?.artifact_id === selectedSwitchArtifact?.id
    ? selectedSwitchArtifact
    : null;
  const selectedPreviewVersion =
    selectedCatalogModel !== null &&
    typeof selectedModelVersionId === "number" &&
    selectedCatalogModel.version_id === selectedModelVersionId
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
    ? `触发延迟 ${runtimeFireDelayMs.toFixed(0)} ms 尚未结束，控制算法未启动`
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
    hardwareTriggerRequired,
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
  const runtimeFaultDetail = runtimeMainlineStatus.failed
    ? runtimeMainlineStatus.failureMessage || "运行链已报告故障，但未提供原因。"
    : "";
  const runtimeFaultEvidence = runtimeFaultDetail ? JSON.stringify({
    daemon_instance_id: runtime?.semantic.daemon_instance_id,
    epoch: runtime?.semantic.epoch,
    fatal_error: runtime?.fatal_error,
    pipeline_error: runtime?.pipeline.last_error,
    inference_error: runtime?.inference.terminal_error
      ? { reason: runtime.inference.reason, detail: runtime.inference.detail } : null,
    deepstream_error: runtime?.pipeline.deepstream.terminal_error
      ? runtime.pipeline.deepstream.last_error : null,
  }, null, 2) : "";
  useEffect(() => {
    // Repeated snapshots are not new failures. Keep the first observation in the
    // existing session history even after recovery; a new epoch is a new event.
    if (lastRuntimeFaultRef.current === runtimeFaultEvidence) return;
    lastRuntimeFaultRef.current = runtimeFaultEvidence;
    if (runtimeFaultDetail) pushToastRaw({
      tone: "error", title: "运行故障", source: "runtime", status: null,
      detail: runtimeFaultDetail, technicalDetail: runtimeFaultEvidence,
    });
  }, [runtimeFaultDetail, runtimeFaultEvidence]);

  const currentErrorDetails = useMemo(() => {
    const items: Array<{ key: string; title: string; detail: string; technicalDetail?: string; requestId?: string | null; time?: number; count?: number }> = [];
    const noticeStates = new Set(errorNotices.map((notice) => `${notice.source}\u0000${notice.detail}`));
    for (const notice of errorNotices) {
      items.push({
        key: `notice-${notice.id}`,
        title: notice.title,
        detail: formatRuntimeErrorMessage(notice.detail || notice.source),
        technicalDetail: `${notice.requestId ? `排查编号: ${notice.requestId}\n` : ""}${notice.technicalDetail || `来源: ${notice.source}${notice.status === null ? "" : `\nHTTP: ${notice.status}`}\n${notice.detail ?? ""}`}`,
        requestId: notice.requestId,
        time: notice.createdAt,
        count: notice.count
      });
    }
    if (runtimeFaultDetail && !errorNotices.some((notice) => notice.technicalDetail === runtimeFaultEvidence)) {
      items.push({ key: "runtime-fault", title: "运行故障", detail: formatRuntimeErrorMessage(runtimeFaultDetail), technicalDetail: runtimeFaultEvidence });
    }
    for (const [source, detail] of Object.entries(errors)) {
      if (detail && !noticeStates.has(`${source}\u0000${detail}`)) {
        items.push({ key: `state-${source}`, title: `${source} 通道异常`, detail: formatRuntimeErrorMessage(detail), technicalDetail: detail });
      }
    }
    const formattedLocalError = localError ? formatRuntimeErrorMessage(localError) : "";
    const localErrorAlreadyCovered = formattedLocalError !== "" && items.some((item) => (
      formattedLocalError === item.detail ||
      formattedLocalError === `${item.title}：${item.detail}` ||
      formattedLocalError === `${item.title}: ${item.detail}`
    ));
    if (localError && !localErrorAlreadyCovered) {
      items.push({ key: "local", title: "当前操作未完成", detail: formattedLocalError, technicalDetail: localError });
    }
    if (capture?.last_error) {
      items.push({ key: "capture", title: "采集链路异常", detail: formatRuntimeErrorMessage(capture.last_error), technicalDetail: capture.last_error });
    }
    return items.filter((item, index, all) => (
      all.findIndex((candidate) => candidate.title === item.title && candidate.detail === item.detail && candidate.technicalDetail === item.technicalDetail) === index
    ));
  }, [capture?.last_error, errorNotices, errors, localError, runtimeFaultDetail, runtimeFaultEvidence]);
  const activityItems = useMemo(() => [
    ...serverActivityEvents.map((event) => ({
      key: `server-${event.daemon_instance_id}-${event.id}`,
      title: event.title,
      detail: formatRuntimeErrorMessage(event.message),
      technicalDetail: event.technical_detail,
      requestId: event.request_id ?? undefined,
      tag: event.tag,
      time: event.occurred_at_ms,
      count: event.count,
      tone: event.level,
    })),
    ...currentErrorDetails.map((item) => ({ ...item, tone: "error" as const })),
    ...activityNotices.map((notice) => ({
      key: `activity-${notice.id}`,
      title: notice.title,
      detail: notice.detail ?? "操作已完成。",
      time: notice.createdAt,
      count: notice.count,
      tone: notice.tone,
    })),
  ].sort((left, right) => (right.time ?? Number.MAX_SAFE_INTEGER) - (left.time ?? Number.MAX_SAFE_INTEGER)), [activityNotices, currentErrorDetails, serverActivityEvents]);

  useEffect(() => {
    if (configuredCaptureDevice) {
      setDevice(configuredCaptureDevice);
    }
  }, [configuredCaptureDevice]);

  const applyModelCatalogResult = useCallback((result: ModelCatalogResponse) => {
    setModelCatalog(result.root);
    setModelCatalogModelCount(result.model_count);
    setModelCatalogDirectoryCount(result.directory_count);
    setModelCatalogError(null);
  }, []);

  const modelSwitch = useModelSwitchWorkflow({
    applyModelCatalogResult,
    onRefresh,
    parserPreset,
    physicalOutputEnabled: runtimeMainlineRunning && (runtimeOutputEnabled === true || outputEnabled),
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
        setModelCatalogError(getErrorMessage(err));
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

  const refreshCaptureDevices = useCallback(async () => {
    setDeviceDiscoveryStatus("loading");
    setDeviceDiscoveryError(null);
    try {
      const result = await getCaptureDevices();
      setCaptureDevices(result.devices);
      setDeviceDiscoveryStatus("ready");
    } catch (error) {
      const message = `设备扫描失败：${getErrorMessage(error)}`;
      setDeviceDiscoveryStatus("error");
      setDeviceDiscoveryError(message);
      reportError(error, { source: "capture-devices", title: "采集设备扫描失败", popup: false });
    }
  }, []);

  useEffect(() => {
    if (activePage === "capture" && deviceDiscoveryStatus === "idle") {
      void refreshCaptureDevices();
    }
  }, [activePage, deviceDiscoveryStatus, refreshCaptureDevices]);

  const loadCaptureCapabilities = useCallback(async (selectedDevice: string) => {
    setBusy("caps");
    setLocalError(null);
    setCaptureActionError(null);
    setCaps(null);
    try {
      const result = await getCaptureCapabilities(selectedDevice);
      setCaps(result);
      if (!result.available) {
        const reason = result.reason || "设备不可用";
        setLocalError(reason);
        reportError(new Error(reason), { source: "capture-caps", title: "采集设备不可用" });
      }
    } catch (err) {
      const message = `设备能力检测失败：${getErrorMessage(err)}`;
      setLocalError(message);
      setCaptureActionError(message);
      reportError(err, { source: "capture-caps", title: "设备能力检测失败", popup: false });
    } finally {
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    if (
      activePage === "capture"
      && deviceDiscoveryStatus === "ready"
      && configuredCaptureDevice
      && captureDevices.includes(configuredCaptureDevice)
      && autoQueriedCaptureDeviceRef.current !== configuredCaptureDevice
    ) {
      autoQueriedCaptureDeviceRef.current = configuredCaptureDevice;
      void loadCaptureCapabilities(configuredCaptureDevice);
    }
  }, [activePage, captureDevices, configuredCaptureDevice, deviceDiscoveryStatus, loadCaptureCapabilities]);

  const refreshCapabilities = useCallback(async () => {
    await loadCaptureCapabilities(device);
  }, [device, loadCaptureCapabilities]);

  const chooseCaptureDevice = useCallback((selectedDevice: string) => {
    if (busy !== null) return;
    if (selectedDevice !== device) setDevice(selectedDevice);
    setCaps(null);
    setCaptureActionError(null);
    void loadCaptureCapabilities(selectedDevice);
  }, [busy, device, loadCaptureCapabilities]);

  const buildCapturePayload = useCallback((choice: CapabilityChoice): CaptureSelectPayload => ({
    device,
    preference: "manual",
    pixel_format: choice.pixel_format,
    width: choice.width,
    height: choice.height,
    fps: choice.fps
  }), [device]);

  const applyCapture = useCallback(async (choice: CapabilityChoice) => {
    if (!device.trim()) {
      setCaptureActionError("请先填写视频设备并明确选择采集格式，再保存配置。");
      return;
    }
    setBusy("capture");
    setLocalError(null);
    setCaptureActionError(null);
    const payload = buildCapturePayload(choice);
    let operationFailed = false;
    try {
      await selectCaptureProfile(payload);
    } catch (err) {
      operationFailed = true;
      const message = `切换失败，已保留上一组可用配置：${getErrorMessage(err)}`;
      setLocalError(message);
      setCaptureActionError(message);
      reportError(err, { source: "capture-select", title: "采集规格保存失败", popup: false });
    }
    try {
      await onRefresh();
    } catch (err) {
      if (!operationFailed) {
        const message = `采集配置处理完成，但最新状态刷新失败：${getErrorMessage(err)}`;
        setLocalError(message);
        setCaptureActionError(message);
      }
      reportError(err, { source: "capture-refresh", title: "采集状态刷新失败", popup: false });
    } finally {
      setBusy(null);
    }
  }, [buildCapturePayload, device, onRefresh]);

  const {
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
  const mainlineLaunchPending = runtimePhase === "starting" || busy === "runtime.start";
  const runtimeStopping = runtimePhase === "stopping" || busy === "runtime.stop";
  const runtimeControlRequested = runtime?.presentation.lifecycle.can_stop === true;
  const runtimeLifecycleActive = runtimeControlRequested || runtimeStopping;
  const diagnosticModeReady = runtime?.presentation.output.daemon_confirmed_safe === true;
  const runtimeControlUnavailable = runtime === null || health?.ok !== true
    || (realtimeStatus !== "connected" && realtimeStatus !== "fallback");
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

  const updateConfigField = useCallback(
    async (
      section: string,
      key: string,
      value: RuntimeConfigValue,
      options?: { immediate?: boolean; optimistic?: boolean; rethrow?: boolean; physicalOutputAcknowledged?: boolean }
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
      if (
        runtimeLifecycleActive
        && (runtimeOutputEnabled === true || outputEnabled)
        && (
          ["replay", "consumers", "limits", "capture", "inference", "hardware", "pipeline"].includes(section)
          || (section === "control" && key === "recoil")
        )
        && options?.physicalOutputAcknowledged !== true
      ) {
        setConfirmationRequest({
          eyebrow: "运行配置",
          title: "物理输出仍开启，确认重载配置？",
          description: "该修改会重载当前主链；重载完成后仍可能向设备发送控制量。取消后配置保持不变。",
          details: [`修改字段：${section}.${key}`, "如不希望设备继续输出，请先回到首页关闭运行。"],
          confirmLabel: "确认重载并保持输出开启",
          danger: true,
          onConfirm: () => updateConfigField(section, key, value, { ...options, physicalOutputAcknowledged: true })
        });
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
          persistRuntimeConfigField(section, key, value, undefined, options?.physicalOutputAcknowledged === true)
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

        reportError(err, { source: "config-write", title: "配置保存失败" });
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
    [activePage, applyConfigSchema, beginPendingConfigWrite, finalizeRuntimeConfigWrite, finishPendingConfigWrite, onRuntimeConfigChange, outputEnabled, runtimeConfig, runtimeLifecycleActive, runtimeOutputEnabled, setConfirmationRequest, setParameterPageDirtyState, stageConfigDialogDraft, stageParameterPageDraft]
  );

  const requestRuntimeChange = useCallback(async (enabled: boolean) => {
    if (!enabled) {
      let gateError: unknown = null;
      if (outputEnabled || runtimeOutputEnabled === true) {
        try {
          await updateConfigField(
            "control",
            "output_enabled",
            false,
            { immediate: true, optimistic: false, rethrow: true }
          );
        } catch (error) {
          gateError = error;
        }
      }
      const stopped = await stopMainlineLaunch();
      if (gateError) throw gateError;
      return stopped;
    }
    if (!hardwareControlLicensed) {
      setLocalError("当前授权没有硬件控制权限。请先完成授权。");
      navigatePage("license");
      return false;
    }
    if (!kmnetHost.trim() || !kmnetUuid.trim()) {
      setLocalError("请先设置控制设备地址和设备编号。");
      navigatePage("control-test");
      return false;
    }
    let outputGateOpened = false;
    try {
      if (!kmnetAutoConnect) {
        await updateConfigField(
          "hardware",
          "auto_connect",
          true,
          { immediate: true, optimistic: false, rethrow: true, physicalOutputAcknowledged: true }
        );
      }
      if (!outputEnabled && runtimeOutputEnabled !== true) {
        await updateConfigField(
          "control",
          "output_enabled",
          true,
          { immediate: true, optimistic: false, rethrow: true, physicalOutputAcknowledged: true }
        );
        outputGateOpened = true;
      }
      if (!runtimeLifecycleActive) {
        const started = await startMainlineLaunch(true);
        if (!started) throw new Error("服务没有确认启动，请查看异常信息后重试。");
      } else if (!kmnetRuntimeConnected) {
        await connectKmNet(undefined, true);
        await onRefresh();
      }
      reportSuccess("NovaSight 已开启", "采集、推理、控制和设备连接由服务统一维护。", "runtime-switch");
      return true;
    } catch (error) {
      if (outputGateOpened) {
        await updateConfigField(
          "control",
          "output_enabled",
          false,
          { immediate: true, optimistic: false, rethrow: false }
        );
      }
      throw error;
    }
  }, [hardwareControlLicensed, kmnetAutoConnect, kmnetHost, kmnetRuntimeConnected, kmnetUuid, navigatePage, onRefresh, outputEnabled, runtimeLifecycleActive, runtimeOutputEnabled, startMainlineLaunch, stopMainlineLaunch, updateConfigField]);

  const saveParameterPageDraft = useCallback(async (physicalOutputAcknowledged = false) => {
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
    let changes: ParameterPageFieldChange[];
    try {
      changes = parameterPageFieldChanges(baseline, draft);
    } catch (error) {
      const message = getErrorMessage(error);
      setLocalError(`参数保存失败：${message}`);
      setDialogSaveError(message);
      reportError(error, { source: "parameter-page", title: "参数保存失败" });
      return;
    }
    const payload = preserveOutputGate(draft, baseline);
    payload.revision = baseline.revision;
    beginPendingConfigWrite();
    setParameterPageSaving(true);
    setBusy("parameter-page.save");
    setDialogSaveError(null);
    setLocalError(null);
    try {
      if (changes.length === 0) {
        configDraftRef.current = baseline;
        setConfigDraft(baseline);
        parameterPageBaselineRef.current = null;
        setParameterPageDirtyState(false);
        return;
      }
      const executeSave = async () => {
        const result = await updateRuntimeConfig(payload, physicalOutputAcknowledged);
        if (result.apply_mode !== "epoch_reload" || !result.applied || result.rolled_back) {
          throw new Error(`整组参数未在当前进程生效，后端返回 ${result.apply_mode}`);
        }
        return result;
      };
      const request = configWriteQueueRef.current.then(executeSave);
      configWriteQueueRef.current = request.then(
        () => undefined,
        () => undefined
      );
      const result = await request;
      finalizeRuntimeConfigWrite(normalizeRuntimeConfig(result.config));
      parameterPageBaselineRef.current = null;
      setParameterPageDirtyState(false);
      reportSuccess(
        result.restart_required ? "参数已应用，其他配置待重启" : "参数已保存并生效",
        result.restart_required
          ? "本页参数已作为一个整体应用；与本页无关的进程级配置仍等待服务重启。"
          : "本页修改已作为一个整体写入并重新加载；首页运行状态未改变。",
        "parameter-page"
      );
    } catch (error) {
      let canonical: RuntimeConfig | null = null;
      try {
        canonical = normalizeRuntimeConfig(await getRuntimeConfig());
      } catch {
        // Preserve the local draft when the authoritative read also fails.
      }
      if (canonical) {
        if (runtimeConfigsEqual(canonical, { ...payload, revision: canonical.revision })) {
          finalizeRuntimeConfigWrite(canonical);
          parameterPageBaselineRef.current = null;
          setParameterPageDirtyState(false);
          reportSuccess("参数已保存并生效", "响应中断，但重新读取后已确认整组参数生效。", "parameter-page");
          return;
        }
        const retryDraft = applyParameterPageFieldChanges(canonical, changes);
        retryDraft.revision = canonical.revision;
        finalizeRuntimeConfigWrite(canonical);
        parameterPageBaselineRef.current = canonical;
        configDraftRef.current = retryDraft;
        setConfigDraft(retryDraft);
        setParameterPageDirtyState(true);
      }
      const message = getErrorMessage(error);
      const failureMessage = `整组参数未生效，${changes.length} 项修改仍保留在页面中。${message}`;
      setLocalError(`参数保存失败：${failureMessage}`);
      setDialogSaveError(failureMessage);
      reportError(error, { source: "parameter-page", title: "参数保存失败" });
    } finally {
      setParameterPageSaving(false);
      setBusy(null);
      finishPendingConfigWrite();
    }
  }, [beginPendingConfigWrite, finalizeRuntimeConfigWrite, finishPendingConfigWrite, parameterPageSaving, setParameterPageDirtyState]);

  const requestSaveParameterPageDraft = useCallback(() => {
    if (!parameterPageDirtyRef.current || parameterPageSaving) return;
    if (!runtimeLifecycleActive || (runtimeOutputEnabled !== true && !outputEnabled)) {
      void saveParameterPageDraft();
      return;
    }
    setConfirmationRequest({
      eyebrow: "运行参数",
      title: "物理输出仍开启，确认保存参数？",
      description: "保存会把修改应用到当前主链；主链运行时，设备的控制量可能立即改变。取消后修改会继续保留在当前页面。",
      details: ["若要先暂停设备，请取消并回到首页关闭运行。"],
      confirmLabel: "确认保存并保持输出开启",
      danger: true,
      onConfirm: () => saveParameterPageDraft(true)
    });
  }, [outputEnabled, parameterPageSaving, runtimeLifecycleActive, runtimeOutputEnabled, saveParameterPageDraft]);

  const updateConfigSection = useCallback(
    async (
      section: string,
      values: Record<string, RuntimeConfigValue>,
      physicalOutputAcknowledged = false
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
        return updateRuntimeConfig(current, physicalOutputAcknowledged);
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
      const applyRoi = async (physicalOutputAcknowledged = false) => {
        setLocalError(null);
        try {
          await updateConfigSection("capture", {
            roi_left: Math.floor((width - size) / 2),
            roi_top: Math.floor((height - size) / 2),
            roi_width: size,
            roi_height: size
          }, physicalOutputAcknowledged);
        } catch (error) {
          const message = getErrorMessage(error);
          setLocalError(`ROI 配置同步失败：${message}`);
          reportError(error, { source: "capture-roi", title: "ROI 配置未保存" });
        }
      };
      if (runtimeLifecycleActive && (runtimeOutputEnabled === true || outputEnabled)) {
        setConfirmationRequest({
          eyebrow: "采集区域",
          title: "物理输出仍开启，确认调整 ROI？",
          description: "调整会重载当前运行主链；重载完成后仍可能向设备发送控制量。取消后 ROI 保持不变。",
          details: [`新 ROI：${size} × ${size}`, "如不希望设备继续输出，请先回到首页关闭运行。"],
          confirmLabel: "确认重载并保持输出开启",
          danger: true,
          onConfirm: () => applyRoi(true)
        });
        return;
      }
      await applyRoi();
    },
    [
      configuredCaptureHeight,
      configuredCaptureWidth,
      outputEnabled,
      runtimeLifecycleActive,
      rustControlPlane,
      runtimeOutputEnabled,
      setConfirmationRequest,
      sourceHeight,
      sourceWidth,
      updateConfigField,
      updateConfigSection
    ]
  );

  const updateDetectionClassFilter = useCallback(
    async (value: string) => {
      await updateConfigField("inference", "detection_class_filters", {
        ...detectionFilterProfiles,
        [activeDetectionProfile]: value
      } as RuntimeConfigValue);
      await updateConfigField("inference", "detection_class_filter", value);
      if (rustControlPlane) {
        await updateConfigField("pipeline", "target_class_filter", value);
      }
    },
    [activeDetectionProfile, detectionFilterProfiles, rustControlPlane, updateConfigField]
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

  const updateTriggerDelay = useCallback((value: number) => {
    const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
    const next = base ? normalizeRuntimeConfig(base) : null;
    if (!next) return;
    const delayMs = Math.max(0, Math.min(5000, Math.round(value)));
    next.pipeline = {
      ...asRecord(next.pipeline),
      fire_delay_enabled: delayMs > 0,
      fire_delay_ms: delayMs
    } as RuntimeConfig[string];
    stageParameterPageDraft(next);
  }, [runtimeConfig, stageParameterPageDraft]);

  const algorithmParameterGroups = activePage === "params"
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
    buildAlgorithmNumberParameterControl(parameter);

  const targetingParameterGroups = activePage === "params"
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
      trackerClassCostWeight,
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
      trackerKalmanNisHardReject,
      targetSelectionMotionHorizonMs
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
      applyMode={parameter.applyMode ?? "live"}
      riskLevel={parameter.riskLevel}
      onCommit={(value) => updatePipelineField(parameter.key, parameter.transform ? parameter.transform(value) : value)}
      onEditingChange={handleParameterEditingChange}
    />
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
          target_class_aim_y_ratios: serializeClassAimRatios(
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
          target_class_aim_y_ratios: serializeClassAimRatios(
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
          detection_class_priority: current.join(","),
          detection_class_priorities: {
            ...detectionPriorityProfiles,
            [activeDetectionProfile]: current.join(",")
          }
        } as RuntimeConfig[string];
        next.pipeline = {
          ...asRecord(next.pipeline),
          target_class_priority: current.join(",")
        } as RuntimeConfig[string];
        stageConfigDialogDraft(next);
        return;
      }
      await updateConfigField("inference", "detection_class_priorities", {
        ...detectionPriorityProfiles,
        [activeDetectionProfile]: current.join(",")
      } as RuntimeConfigValue);
      await updateConfigField("inference", "detection_class_priority", current.join(","));
    },
    [
      orderedClassEditorIds,
      activeDetectionProfile,
      detectionPriorityProfiles,
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
        Math.round(dy),
        true
      );
      setKmnetTestMessageTone(result.sent ? "success" : "warning");
      setKmnetTestMessage(
        result.sent
          ? `已通过 ${result.metadata.api_name} 发送 dx=${Math.round(dx)} dy=${Math.round(dy)} · ${result.steps_sent}/1 步`
          : `未发送原始命令：${result.message} · ${result.steps_sent}/1 步`
      );
      await onRefresh();
    } catch (err) {
      const message = `单步移动未完成：${getErrorMessage(err)}`;
      setLocalError(message);
      setKmnetTestMessageTone("warning");
      setKmnetTestMessage(message);
      reportError(err, { source: "kmnet-diagnostic", title: "kmNet 单步移动失败", popup: false });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetTestDx, kmnetTestDy, onRefresh]);

  const requestDiagnosticMoveHardware = useCallback((dx: number, dy: number) => {
    const roundedDx = Math.round(dx);
    const roundedDy = Math.round(dy);
    setConfirmationRequest({
      eyebrow: "kmNet 单步测试",
      title: "发送一次物理位移？",
      description: `将向 ${kmnetHost || "未填写"}:${kmnetPort || "未填写"} 发送一次 dx=${roundedDx}、dy=${roundedDy} 的位移命令。`,
      details: ["只发送这一条命令；服务端仍会核对主链已停止、输出已允许，并在测试后断开设备。"],
      confirmLabel: "确认发送一次",
      danger: true,
      onConfirm: () => diagnosticMoveHardware(roundedDx, roundedDy)
    });
  }, [diagnosticMoveHardware, kmnetHost, kmnetPort]);

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
    const changedSections = changedRuntimeConfigSections(baseline, preserveOutputGate(imported, baseline));
    setConfirmationRequest({
      eyebrow: canonicalChanged ? "配置已在后台更新" : "导入运行配置",
      title: canonicalChanged ? "请按最新配置重新确认" : `应用 ${fileName}？`,
      description: canonicalChanged
        ? "你确认前，当前配置已被其他操作更新。NovaSight 没有覆盖新 revision，下面的差异已按最新配置重新计算。"
        : "导入会在确认时重新读取当前配置版本，再以该事务基线替换整份运行配置。",
      details: [
        `将修改：${changedSections.join("、") || "没有差异"}`,
        "运行参数会立即应用；监听地址或存储根目录等进程级配置会单独提示。",
        "导入不会改变首页运行总开关；运行中导入仍可能立即改变控制行为。"
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
              if (changedRuntimeConfigSections(canonical, preserveOutputGate(imported, canonical)).length === 0) {
                reportSuccess("无需导入配置", "当前配置已经与导入文件一致。", "config-import");
                return true;
              }
              requestConfigImportConfirmation(fileName, imported, canonical, true);
              return false;
            }
            const payload = preserveOutputGate(imported, canonical);
            payload.revision = canonical.revision;
            const result = await updateRuntimeConfig(payload, true);
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
              if (changedRuntimeConfigSections(canonical, preserveOutputGate(imported, canonical)).length === 0) {
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

  const importConfig = async (file: File) => {
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
      const payload = preserveOutputGate(decoded as RuntimeConfig, current);
      const changedSections = changedRuntimeConfigSections(current, payload);
      if (changedSections.length === 0) {
        setLocalError("导入文件与当前配置一致，没有需要应用的修改。");
        return;
      }
      requestConfigImportConfirmation(file.name, payload, current);
    } catch (err) {
      setLocalError(`导入失败：${getErrorMessage(err)}`);

      reportError(err, { source: "config-import", title: "配置导入失败" });
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
      setModelCatalogError(getErrorMessage(err));
      setLocalError(`模型列表刷新失败：${getErrorMessage(err)}`);

      reportError(err, { source: "model-catalog", title: "模型列表刷新失败" });
    } finally {
      setBusy(null);
    }
  };

  const createModelFolder = async (relativePath: string) => {
    setBusy("model.folder.create");
    setLocalError(null);
    let created = false;
    try {
      await createCatalogFolder(relativePath);
      created = true;
      const result = await getModelCatalog(true);
      applyModelCatalogResult(result);
      setModelCatalogMessage(`文件夹 ${relativePath} 已创建；模型部署未改变。`);
    } catch (error) {
      const message = created
        ? `文件夹已创建，但列表刷新失败：${getErrorMessage(error)}。请点击“刷新模型”查看。`
        : `新建文件夹失败：${getErrorMessage(error)}`;
      setLocalError(message);
      reportError(error, { source: "model-folder", title: created ? "文件夹列表刷新失败" : "新建文件夹失败", publicDetail: getErrorMessage(error), popup: false });
      throw new Error(message);
    } finally {
      setBusy(null);
    }
  };

  const requestMoveModelEngine = (fromPath: string, toPath: string) => {
    setConfirmationRequest({
      eyebrow: "整理模型文件",
      title: "确认移动或改名模型文件？",
      description: "这会改变设备上未登记 Engine 的文件路径；不会加载模型或切换当前部署。",
      details: [`原位置：${fromPath}`, `新位置：${toPath}`],
      confirmLabel: "确认更改文件路径",
      onConfirm: async () => {
        setBusy("model.file.move");
        setLocalError(null);
        let moved = false;
        try {
          await moveCatalogEngine(fromPath, toPath);
          moved = true;
          const result = await getModelCatalog(true);
          applyModelCatalogResult(result);
          setSelectedModelCatalogPath(toPath);
          setModelCatalogMessage(`模型文件已移至 ${toPath}；当前部署未改变。`);
        } catch (error) {
          const message = moved
            ? `文件已移动，但列表刷新失败：${getErrorMessage(error)}。请点击“刷新模型”重新读取。`
            : `模型文件移动失败：${getErrorMessage(error)}`;
          setLocalError(message);
          reportError(error, { source: "model-file-move", title: moved ? "模型目录刷新失败" : "模型文件移动失败", publicDetail: getErrorMessage(error), popup: false });
          if (!moved) throw error;
        } finally {
          setBusy(null);
        }
      }
    });
  };

  const requestSaveModelMetadata = (recommendation: ModelRecommendation, tags: string[]) => {
    if (selectedCatalogModel?.kind === "engine"
      && (typeof selectedCatalogModel.project_id !== "number" || typeof selectedCatalogModel.artifact_id !== "number")) {
      setConfirmationRequest({
        eyebrow: "整理模型",
        title: "登记后保存模型标签？",
        description: "首次保存会登记这个 Engine。登记后，模型库不能直接移动或改名该文件；如需调整路径，请先整理文件。",
        details: [`文件：${selectedCatalogModel.relative_path}`, "只保存推荐状态与标签，不会切换或加载当前模型。"],
        confirmLabel: "登记并保存",
        onConfirm: () => modelSwitch.saveMetadata(recommendation, tags)
      });
      return;
    }
    void modelSwitch.saveMetadata(recommendation, tags);
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
  const setupState: SetupState = {
    captureReady: configuredCaptureProfile !== null && configuredCaptureDevice.length > 0,
    modelReady: artifact !== undefined,
    configReady: runtime !== null
      && runtimeTransportConfidence === "current"
      && runtimeConfig !== null
      && desiredConfigRevision === effectiveConfigRevision
      && runtime?.config.restart_required !== true,
    runtimeReady: runtime?.running === true
  };
  const setupStatusKnown = runtime !== null && runtimeConfig !== null && runtimeTransportConfidence === "current";
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
    navigatePage("license");
  }, [navigatePage]);
  const handleRuntimeRecoveryAction = useCallback((action: RuntimeRecoveryAction) => {
    handleLaunchReadinessAction(action);
  }, [handleLaunchReadinessAction]);
  return (
    <section className="console-app">
      <header className="console-top">
        <div className="console-brand">
          <div className="console-logo">
            <NovaIcon name="prediction-line" size={24} strokeWidth={1.9} />
          </div>
          <span className="console-brand-name">NovaSight<span>Studio</span></span>
        </div>
        <div className="console-toolbar">
          <div className="console-context-deck" aria-label="当前生产上下文">
            <button
              className="console-context-entry"
              onClick={() => navigatePage("models")}
              title={activeModelName}
              type="button"
            >
              <NovaIcon name="models" size={16} />
              <span><small>当前模型</small><b>{activeModelName}</b></span>
            </button>
            <button
              className="console-context-entry"
              onClick={() => navigatePage("capture")}
              title={configuredCaptureDevice || capture?.device || "尚未配置采集设备"}
              type="button"
            >
              <NovaIcon name="capture" size={16} />
              <span><small>信号输入</small><b>{configuredCaptureDevice || capture?.device || "未配置"}</b></span>
            </button>
          </div>
          <div className="console-safety-deck" aria-label="运行与安全控制">
            <div
              aria-label={`${realtimeStatusText}。${realtimeStatusDescription}运行态 ${formatDate(lastUpdated)}`}
              className={realtimeStatusClass}
              role="status"
              title={`${realtimeStatusDescription} 最近更新 ${formatDate(lastUpdated)}`}
            >
              <span>{realtimeStatusText}</span>
              <small>{formatDate(lastUpdated)}</small>
            </div>
            <button type="button"
              className={currentErrorDetails.length > 0 ? "error-center-trigger has-errors" : "error-center-trigger"}
              aria-label={currentErrorDetails.length > 0 ? `查看异常信息，共 ${currentErrorDetails.length} 条` : "查看异常信息"}
              onClick={() => setErrorCenterOpen(true)}
            >
              <NovaIcon name={currentErrorDetails.length > 0 ? "triangle-alert" : "shield-check"} size={15} />
              <span>{currentErrorDetails.length > 0 ? "查看故障" : "异常"}</span>
              {currentErrorDetails.length > 0 ? <b>{currentErrorDetails.length}</b> : null}
            </button>
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
          </div>
        </div>
      </header>

      <div className="console-sidebar">
        <StudioNavigation activePage={activePage} onNavigate={navigatePage} pages={studioLayout.pages} />
      </div>

      <main className="console-main" data-page={activePage} ref={mainRef}>
        <StudioPageHeader page={activePage} />
        <StudioRuntimeBar
          page={activePage}
          runtimeAvailable={runtime !== null}
          runtimeLifecycleActive={runtimeLifecycleActive}
          diagnosticModeReady={diagnosticModeReady}
          captureStatus={captureStatusText}
          inferenceStatus={inferenceStatusText}
        />

        {activePage === "overview" ? (
          <>
            {overviewModules.has("setup") ? <HomeSetupPrompt state={setupState} statusKnown={setupStatusKnown} onNavigate={navigatePage} /> : null}
            <RuntimeOverviewView
              runtime={runtime}
              projection={runtimeProjection}
              readiness={launchReadiness}
              lastUpdated={lastUpdated}
              onAction={handleRuntimeRecoveryAction}
              controlBusy={busy !== null}
              launchPending={mainlineLaunchPending}
              runtimeStopping={runtimeStopping}
              runtimeControlUnavailable={runtimeControlUnavailable}
              onOpenErrors={() => setErrorCenterOpen(true)}
              onToggle={() => requestRuntimeChange(!runtimeLifecycleActive)}
              moduleOrder={overviewModuleOrder}
            />
          </>
        ) : null}

        {activePage === "onboarding" ? (
          <OnboardingView state={setupState} statusKnown={setupStatusKnown} onNavigate={navigatePage} />
        ) : null}

        {activePage === "device" ? (
          <DeviceStatusView
            runtime={runtime}
            projection={runtimeProjection}
            captureProfile={configuredCaptureProfile ? choiceLabel(configuredCaptureProfile) : ""}
            activeModelName={activeModelName}
            modelLoaded={runtimeInference?.loaded === true}
            lastUpdated={lastUpdated}
            desiredRevision={desiredConfigRevision}
            effectiveRevision={effectiveConfigRevision}
            errorCount={currentErrorDetails.length}
            onNavigate={navigatePage}
            onOpenErrors={() => setErrorCenterOpen(true)}
          />
        ) : null}

        {activePage === "activity" ? (
          <ActivityView
            items={activityItems}
            moduleOrder={activityModules}
            onClear={async () => {
              try {
                await clearServerActivity();
                clearActivityNotices();
                clearErrorNotices();
              } catch (error) {
                reportError(error, { source: "activity-clear", title: "日志清理失败" });
                throw error;
              }
            }}
            onOpenDetails={() => setErrorCenterOpen(true)}
          />
        ) : null}

        {activePage === "settings" ? (
          <SettingsView
            modules={settingsModules}
            desiredRevision={desiredConfigRevision}
            effectiveRevision={effectiveConfigRevision}
            restartRequired={reportedConfigRestartRequired}
            configAvailable={runtimeConfig !== null}
            operationPending={busy !== null || pendingConfigWriteCount > 0}
            parameterChangesPending={parameterPageDirty}
            onExport={exportConfig}
            onImport={importConfig}
            onNavigate={navigatePage}
          />
        ) : null}

        {activePage === "about" ? (
          <AboutView
            license={license}
            serviceConnected={health?.ok === true && runtimeTransportConfidence === "current"}
            layoutRevision={studioLayout.revision}
            modules={aboutModules}
            onNavigate={navigatePage}
          />
        ) : null}

        {activePage === "models" ? (
          <Suspense fallback={<div className="console-info" role="status">正在加载模型管理工作区…</div>}>
            <ModelWorkspace
              activeModelName={activeModelName}
              activeArtifactStatus={artifact?.status ?? null}
              onOpenInference={() => navigatePage("infer")}
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
                activeLoaded: runtimeInference?.loaded === true,
                runtimeBackend: readString(runtime?.inference?.selected, ""),
                runtimeInputShape: displayedInputShape,
                catalogMessage: modelCatalogMessage,
                catalogError: modelCatalogError,
                switchMessage: modelSwitch.message,
                busy,
                canSwitch: selectedCatalogModel?.kind === "engine",
                parserPreset,
                onParserPresetChange: setParserPreset,
                onRefresh: () => void refreshModelCatalog(),
                onCreateFolder: createModelFolder,
                onRequestMove: requestMoveModelEngine,
                onSelectModel: selectModelFromCatalog,
                onSaveMetadata: requestSaveModelMetadata,
                onSwitch: modelSwitch.switchModel,
              }}
            />
          </Suspense>
        ) : null}

        {activePage === "management" ? (
          <ManagementView
            license={license}
            deviceLabel={configuredCaptureDevice || runtime?.capture.device || ""}
            runtimeAvailable={runtime !== null && runtimeTransportConfidence === "current"}
            projectCount={projects.length}
            errorCount={currentErrorDetails.length}
            setupState={setupState}
            onNavigate={navigatePage}
          />
        ) : null}

        {activePage === "license" ? (
          <section className="console-page">
            <LicenseView license={license} onLicenseChange={onLicenseChange} />
          </section>
        ) : null}

        {(["capture", "infer", "control", "latency"] as ConsolePage[]).includes(activePage) && runtimeLifecycleActive && runtimeMainlinePresentation.readinessCode !== "ready" ? (
          <div className="console-info console-info-recovery" role="status">
            <span>{runtimeMainlinePresentation.readinessLabel}：{runtimeMainlinePresentation.readinessDetail}</span>
            {runtimeMainlinePresentation.readinessCode === "no_video" && activePage !== "capture" ? (
              <button className="console-button" onClick={() => navigatePage("capture")} type="button">
                前往采集
              </button>
            ) : null}
          </div>
        ) : null}

        {activePage === "capture" ? (
          <section className="console-page capture-workspace">
          {captureModules.has("source") ? (
          <header className="capture-command-header" data-state={captureProfileState} aria-labelledby="capture-profile-check-title">
            <div className="capture-command-copy">
              <span className="class-config-eyebrow">当前画面输入</span>
              <h2 id="capture-profile-check-title">{configuredCaptureDevice || "尚未配置采集设备"}</h2>
              <p>{configuredCaptureProfile ? choiceLabel(configuredCaptureProfile) : "尚未选择画面规格"}</p>
            </div>
            <div className="capture-runtime-proof">
              <div>
                <span>运行中的画面</span>
                <strong>{capture?.running === true && selectedProfile ? choiceLabel(selectedProfile) : capture?.running === true ? "规格未上报" : "主链未运行"}</strong>
                <small>{capture?.running === true ? capture.device || "设备未上报" : "启动后自动核对"}</small>
              </div>
              <span className="capture-profile-check-state" role="status">
                <NovaIcon name={captureProfileState === "matched" ? "check-circle" : captureProfileState === "mismatch" ? "triangle-alert" : "clock"} size={16} />
                {captureProfileState === "matched" ? "运行规格一致" : captureProfileState === "mismatch" ? "保存与运行不一致" : captureProfileState === "unknown" ? "运行规格未确认" : "等待运行验证"}
              </span>
            </div>
          </header>
          ) : null}
          <div className="capture-workbench">
              {captureModuleOrder.map((moduleId) => {
                if (moduleId === "source") return (
              <section className="capture-source-setup" aria-labelledby="capture-source-title" key={moduleId}>
                <header className="capture-workbench-heading">
                  <span aria-hidden="true"><NovaIcon name="capture-card" size={20} /></span>
                  <div><small>画面来源</small><h3 id="capture-source-title">选择设备和画质</h3><p>读取设备真实能力；点击帧率后立即保存，不需要再次应用。</p></div>
                </header>
                <div className="capture-device-picker">
                  <header>
                    <div><b>选择采集设备</b><small>只展示 Jetson 当前发现的 V4L2 设备；选择后自动读取其真实规格。</small></div>
                    <button className="console-button secondary" disabled={deviceDiscoveryStatus === "loading" || busy !== null} onClick={() => void refreshCaptureDevices()} type="button">
                      <NovaIcon name="refresh" size={15} />
                      {deviceDiscoveryStatus === "loading" ? "正在扫描…" : "重新扫描"}
                    </button>
                  </header>
                  <div className="capture-device-options" role="listbox" aria-label="可用采集设备">
                    {captureDeviceChoices.map((captureDevice) => {
                      const saved = captureDevice === configuredCaptureDevice;
                      const discovered = captureDevices.includes(captureDevice);
                      const selected = captureDevice === device;
                      return (
                        <button
                          aria-selected={selected}
                          className={selected ? "selected" : ""}
                          disabled={busy !== null || runtimeLifecycleActive}
                          key={captureDevice}
                          onClick={() => chooseCaptureDevice(captureDevice)}
                          role="option"
                          type="button"
                        >
                          <span aria-hidden="true"><NovaIcon name="capture-card" size={18} /></span>
                          <span><b>{captureDevice}</b><small>{saved ? "已保存" : "可选择"}{!discovered ? " · 当前未发现" : ""}</small></span>
                          <i>{selected && busy === "caps" ? "读取中" : selected ? "已选" : "选择"}</i>
                        </button>
                      );
                    })}
                    {deviceDiscoveryStatus === "ready" && captureDeviceChoices.length === 0 ? (
                      <div className="capture-device-empty"><b>没有发现采集设备</b><small>连接采集卡后重新扫描，或在下方输入其他设备路径。</small></div>
                    ) : null}
                    {deviceDiscoveryStatus === "error" ? (
                      <div className="capture-device-empty error" role="alert"><b>无法扫描设备</b><small>{deviceDiscoveryError}</small></div>
                    ) : null}
                  </div>
                </div>
                <details className="capture-custom-device">
                  <summary>输入其他设备路径</summary>
                  <div className="capture-device-row">
                    <TextControl
                      label="设备路径"
                      detail="仅在设备未出现在扫描结果时使用；读取成功后仍需选择一个帧率才会保存。"
                      value={device}
                      applyMode="launch"
                      onCommit={(value) => {
                        setDevice(value);
                        setCaps(null);
                        setCaptureActionError(null);
                      }}
                    />
                    <button className="console-button secondary" disabled={busy === "caps" || !device.trim()} onClick={refreshCapabilities} type="button">
                      <NovaIcon name="refresh" size={15} />
                      {busy === "caps" ? "正在读取…" : "读取设备规格"}
                    </button>
                  </div>
                </details>
                <div className="capture-quality-browser" aria-label="分辨率与帧率">
                  <header>
                    <div><b>设备支持的画面规格</b><small>{caps ? `按格式分类，共 ${detectedChoices.length} 个组合；分辨率和帧率从高到低排列。` : "先查询设备，结果会按格式、分辨率和帧率分类。"}</small></div>
                    {configuredCaptureProfile ? <span>当前：{choiceLabel(configuredCaptureProfile)}</span> : null}
                  </header>
                  {capabilityGroups.length > 0 ? capabilityGroups.map((group) => (
                    <section className="capture-format-group" key={group.pixelFormat}>
                      <h4>{canonicalCaptureFormat(group.pixelFormat) === "MJPG" ? "MJPEG / MJPG" : group.pixelFormat}</h4>
                      <div>
                        {group.resolutions.map((resolution) => (
                          <div className="capture-resolution-row" key={`${resolution.width}x${resolution.height}`}>
                            <span><b>{resolution.width} × {resolution.height}</b><small>{(resolution.width * resolution.height / 1_000_000).toFixed(1)} MP</small></span>
                            <div aria-label={`${resolution.width} × ${resolution.height} 可选帧率`}>
                              {resolution.choices.map((choice) => {
                                const active = device === configuredCaptureDevice && configuredCaptureProfile != null && choiceMatchesConfig(choice, captureConfig);
                                return (
                                  <button
                                    aria-pressed={active}
                                    className={active ? "active" : ""}
                                    disabled={busy !== null || runtimeLifecycleActive || runtimeControlUnavailable}
                                    key={choice.fps}
                                    onClick={() => void applyCapture(choice)}
                                    type="button"
                                  >
                                    {busy === "capture" ? "保存中" : `${choice.fps} FPS`}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  )) : (
                    <div className="capture-quality-empty">
                      {caps?.available === false ? "当前设备不可用" : "尚未查询设备规格"}
                    </div>
                  )}
                  {runtimeLifecycleActive ? <p className="capture-save-hint">停止首页运行总开关后可更换画面规格。</p> : <p className="capture-save-hint">点击一个 FPS 即自动验证并保存整组设备配置。</p>}
                </div>
                {captureActionError ? <p className="operation-inline-error" role="alert">{captureActionError}</p> : null}
                <details className="capture-optional-settings">
                  <summary>远程预览画质</summary>
                  <ParameterPresetControl
                    label="预览帧率"
                    detail="只影响浏览器预览流量，不改变采集和推理帧率。"
                    options={[15, 30].map((fps) => ({
                      id: `preview-${fps}`,
                      label: `${fps} FPS`,
                      detail: fps === 15 ? "节省带宽" : "画面更顺滑",
                      active: previewFps === fps,
                      disabled: busy === "limits.stream_fps",
                      onSelect: () => updateConfigField("limits", "stream_fps", fps)
                    }))}
                  />
                </details>
              </section>
              );

                if (moduleId === "roi") return (
              <section className="capture-roi-stage" aria-labelledby="capture-roi-title" key={moduleId}>
                <header className="capture-workbench-heading">
                  <span aria-hidden="true"><NovaIcon name="roi" size={20} /></span>
                  <div><small>模型视野</small><h3 id="capture-roi-title">模型看见画面中央</h3><p>只改变送入模型的中心区域，不改变采集卡分辨率。</p></div>
                </header>
                <div className="capture-roi-visual" aria-label={`源画面中间的 ${roiSize} x ${roiSize} 识别区域`}>
                  <span>源画面 {sourceWidth > 0 ? `${sourceWidth} × ${sourceHeight}` : "待读取"}</span>
                  <div><i /><b>{roiSize} × {roiSize}</b><small>模型识别区域</small></div>
                </div>
                <ParameterPresetControl
                  label="识别范围"
                  detail="近距细节更集中，大视野更容易覆盖快速目标。"
                  options={ROI_SIZE_CHOICES.map((size) => ({
                    id: `roi-${size}`,
                    label: `${size}`,
                    detail: size <= 320 ? "近距细节" : size >= 560 ? "大视野" : "平衡",
                    active: roiSize === size,
                    disabled: busy === "roi.size" || busy === "capture.roi",
                    onSelect: () => handleCenteredRoiSizeChange(size)
                  }))}
                />
                <details className="capture-optional-settings">
                  <summary>自定义识别范围</summary>
                  <ParameterNumberControl
                    label="边长"
                    detail="以 16 像素为一步调整中心正方形区域。"
                    value={roiSize}
                    min={256}
                    max={640}
                    step={16}
                    unit="px"
                    onCommit={handleCenteredRoiSizeChange}
                    onEditingChange={handleParameterEditingChange}
                  />
                  <div className="console-kv compact-kv">
                    <span>位置</span><b>{sourceWidth > 0 ? `x=${roiX}, y=${roiY}` : NO_SAMPLE}</b>
                    <span>保存值</span><b>{sourceWidth > 0 ? rustControlPlane ? `${configuredRoiWidth}x${configuredRoiHeight}` : `${roiSize}x${roiSize}` : NO_SAMPLE}</b>
                  </div>
                </details>
              </section>
              );
                return null;
              })}
          </div>

          {captureModules.has("diagnostics") ? (
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
          ) : null}
          </section>
        ) : null}

        {activePage === "infer" ? (
          <section className="console-page">
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
          {runtime !== null ? (
            <div className="console-metrics consumer-signal-metrics">
              <Metric title="推理 FPS" value={formatOptionalNumber(nvinferOutputFps)} small="模型实际完成" />
              <Metric title="结果 FPS" value={formatOptionalNumber(detectionBatchFps)} small="识别结果有效产出" />
              <Metric title="结果新鲜度" value={formatOptionalNumber(detectionDataAgeMs)} small={`${detectionFreshness}，单位 ms`} />
              <Metric title="最近检测" value={formatOptionalInteger(detectionCount)} small="最近一次状态更新" />
            </div>
          ) : null}
          <div className="console-grid2 inference-config-grid">
            <div className="console-card">
              <SectionTitle title="识别灵敏度" />
              <ParameterNumberControl
                label="最低可信度"
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
                label="重复目标合并"
                detail="NMS 使用这个阈值合并同一目标附近的重叠框；过低容易误删，过高容易重复。"
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
              <SectionTitle title="识别画面" />
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
              detail="Web/API 或 novasightd 状态通道不可用，实时控制结果暂时无法显示。"
              action={(
                <div className="console-action-row">
                  <button className="console-button" onClick={() => setErrorCenterOpen(true)} type="button">
                    <NovaIcon name="triangle-alert" size={15} />
                    查看异常
                  </button>
                  <button className="console-button primary" disabled={busy !== null} onClick={() => void onRefresh()} type="button">
                    <NovaIcon name="refresh" size={15} />
                    重试
                  </button>
                </div>
              )}
            />
          ) : (
          <>
          <ControlTracePanel trace={controlTrace!} />
          <div className="control-config-shortcut">
            <span>需要调整目标选择、跟踪或物理输出？</span>
            <button className="console-button" onClick={() => navigatePage("params")} type="button">
              前往控制参数
            </button>
          </div>
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
                  <span>进入跟踪 / 预算舍弃</span><b>{`${formatOptionalInteger(targetPipeline?.counts.admitted_to_tracking)} / ${formatOptionalInteger(targetPipeline?.counts.dropped_by_budget)}`}</b>
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
                  <span>触发延迟</span><b>{fireDelayMs > 0 ? `${fireDelayMs.toFixed(0)} ms` : "立即触发"}</b>
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
            {parameterPageDirty ? <div className="parameter-save-bar dirty">
              <span className="parameter-save-bar-icon" aria-hidden="true">
                <NovaIcon name="save" size={18} />
              </span>
              <div aria-live="polite" role="status">
                <b>有未应用的修改</b>
                <small>保存后整组参数立即生效；首页运行状态保持不变。</small>
              </div>
              <div className="parameter-save-bar-actions">
                <button
                  className="console-button"
                  disabled={parameterPageSaving || pendingConfigWriteCount > 0}
                  onClick={requestDiscardParameterPageDraft}
                  type="button"
                >
                  放弃修改
                </button>
                <button
                  className="console-button primary"
                  disabled={parameterPageSaving || pendingConfigWriteCount > 0}
                  onClick={requestSaveParameterPageDraft}
                  type="button"
                >
                  <NovaIcon name="save" size={15} />
                  {parameterPageSaving ? "正在保存并应用…" : "保存并应用"}
                </button>
              </div>
            </div> : null}
            {dialogSaveError ? <p className="operation-inline-error" role="alert">参数保存失败：{dialogSaveError}</p> : null}
            <div className="parameter-workspace">
            {parameterModuleOrder.map((moduleId) => {
              if (moduleId === "response") return (
            <section className="parameter-group" data-module="response" id="parameter-start-conditions" aria-labelledby="parameter-start-conditions-title" key={moduleId}>
              <header className="parameter-group-heading">
                <span>触发</span>
                <div><h2 id="parameter-start-conditions-title">触发设置</h2><p>只设置触发信号生效前的等待时间；触发按键由设备绑定负责。</p></div>
              </header>
            <ol className="control-chain-settings" aria-label="触发设置">
              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true"><NovaIcon name="clock" size={16} /></span>
                <div className="control-chain-setting-title">
                  <b>触发延迟</b>
                  <small>收到触发信号后等待多久再执行控制；设为 0 ms 时立即执行。</small>
                </div>
                <div className="control-chain-setting-controls">
                  <label className="control-chain-inline-field">
                    <span>delay</span>
                    <InlineNumberControl
                      ariaLabel="触发延迟"
                      value={fireDelayMs}
                      onCommit={updateTriggerDelay}
                    />
                    <i>ms</i>
                  </label>
                </div>
              </li>

            </ol>
            </section>
            );
              if (moduleId === "motion") return (
            <section className="parameter-group" data-module="motion" id="parameter-motion-response" aria-labelledby="parameter-motion-response-title" key={moduleId}>
              <header className="parameter-group-heading">
                <span>移动</span>
                <div><h2 id="parameter-motion-response-title">移动响应</h2><p>调整移动速度、提前量、补偿和单次输出边界。</p></div>
              </header>
            <ol className="control-chain-settings" aria-label="移动响应设置">
              <li className="console-card control-chain-setting parameter-expanded-setting">
                <span className="control-chain-step" aria-hidden="true"><NovaIcon name="prediction-line" size={16} /></span>
                <div className="control-chain-setting-title">
                  <b>移动手感</b>
                  <small>基础响应决定整体速度；远距增强决定偏差较大时的追赶力度。</small>
                </div>
                <div className="parameter-inline-grid">
                  {responseParameters.map(renderAlgorithmNumberParameter)}
                </div>
              </li>

              <li className="console-card control-chain-setting parameter-expanded-setting">
                <span className="control-chain-step" aria-hidden="true"><NovaIcon name="activity-pulse" size={16} /></span>
                <div className="control-chain-setting-title">
                  <b>移动预测</b>
                  <small>根据近期运动趋势提前移动；快速横移跟不上时再逐步增加。</small>
                </div>
                <div className="parameter-expanded-controls">
                  <ModuleSwitch
                    compact
                    label="预测移动目标"
                    enabled={controlPredictionEnabled}
                    onToggle={(enabled) => updateControlPipelineField("prediction_enabled", enabled)}
                  />
                  {controlPredictionEnabled ? (
                    <div className="parameter-inline-grid">
                      {predictionCoreParameters
                        .filter((parameter) => parameter.key === "prediction_lead_ms")
                        .map(renderAlgorithmNumberParameter)}
                      {predictionCapParameters.map(renderAlgorithmNumberParameter)}
                    </div>
                  ) : null}
                </div>
              </li>

              <li className="console-card control-chain-setting">
                <span className="control-chain-step" aria-hidden="true"><NovaIcon name="auto-tune" size={16} /></span>
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
                <span className="control-chain-step" aria-hidden="true"><NovaIcon name="shield-check" size={16} /></span>
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

            </ol>
            </section>
            );
              if (moduleId === "targeting") return (
            <div className="parameter-module-group" data-module="targeting" key={moduleId}>
            <section className="parameter-group" data-module="targeting" id="parameter-target-lock" aria-labelledby="parameter-target-lock-title">
              <header className="parameter-group-heading">
                <span>目标</span>
                <div><h2 id="parameter-target-lock-title">目标锁定</h2><p>决定哪些目标可以被选中，以及短暂遮挡或更优目标出现时如何处理。</p></div>
              </header>
              <div className="parameter-group-body">
                <div className="class-config-summary-card">
                  <div className="class-config-summary-main">
                    <div className="class-config-summary-icon" aria-hidden="true">
                      <NovaIcon name="target" size={20} strokeWidth={1.8} />
                    </div>
                    <div>
                      <span className="class-config-eyebrow">目标类别</span>
                      <h3>当前目标规则</h3>
                      <small>设置可选类别、优先级和各类别瞄准位置。</small>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="console-button"
                    disabled={configDialogSaving}
                    onClick={() => openConfigDialog("class-config")}
                  >
                    <NovaIcon name="settings" size={16} />
                    编辑目标类别
                  </button>
                </div>

                <div className="parameter-inline-grid parameter-target-basics">
                  <ParameterNumberControl
                    label="可选目标范围"
                    detail="只在准星周围这个半径内选择目标；调小更专注，调大覆盖更多候选。"
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
                  {targetAdvancedParameters.map(renderTargetingNumberParameter)}
                </div>

                <details className="parameter-disclosure">
                  <summary>
                    <span><b>目标偏好</b><small>距离、类别、可信度、大小、连续性和运动趋势共同决定优先目标。</small></span>
                    <i>{dominantTargetSelectionWeight.label} {(dominantTargetSelectionWeight.share * 100).toFixed(0)}%</i>
                  </summary>
                  <div className="target-weight-composition" aria-label="综合目标分数权重占比">
                    {normalizedTargetSelectionWeights.map((item) => (
                      <i className={item.className} key={item.key} style={{ flexGrow: item.share }} />
                    ))}
                  </div>
                  <div className="target-weight-legend">
                    {normalizedTargetSelectionWeights.map((item) => (
                      <span key={item.key}><i className={item.className} />{item.label} <b>{(item.share * 100).toFixed(0)}%</b></span>
                    ))}
                  </div>
                  <div className="parameter-inline-grid">
                    {targetSelectionWeights.map((item) => (
                      <ParameterNumberControl
                        detail={item.detail}
                        key={item.key}
                        label={`${item.label}权重`}
                        max={100}
                        min={0}
                        onCommit={(value) => updatePipelineField(item.key, value)}
                        onEditingChange={handleParameterEditingChange}
                        recommendedMax={1}
                        recommendedMin={0}
                        step={0.01}
                        value={item.value}
                      />
                    ))}
                  </div>
                </details>
              </div>
            </section>
            <details className="parameter-professional-settings" id="parameter-professional-settings">
              <summary>
                <span><b>专业参数</b><small>只在出现误跟、断轨或设备标定偏差时调整。</small></span>
                <i>展开</i>
              </summary>
              <div className="parameter-professional-content">
                <section>
                  <header><h3>预测与设备标定</h3><p>修正断流后的预测历史、画面视场与设备真实移动比例。</p></header>
                  <div className="parameter-inline-grid">
                    {predictionCoreParameters
                      .filter((parameter) => parameter.key !== "prediction_lead_ms")
                      .map(renderAlgorithmNumberParameter)}
                    {calibrationParameters.map(renderAlgorithmNumberParameter)}
                  </div>
                </section>
                <section>
                  <header><h3>短时跟踪匹配</h3><p>这些参数只用于判断相邻画面中的检测结果是否属于同一目标，不把类别当成永久身份。</p></header>
                  <div className="parameter-inline-grid">
                    {trackerCoreParameters.map(renderTargetingNumberParameter)}
                  </div>
                </section>
                <details className="parameter-disclosure nested">
                  <summary><span><b>轨迹滤波参数</b><small>处理框抖动、短暂漏检和异常跳变。</small></span><i>{trackerKalmanParameters.length} 项</i></summary>
                  <div className="parameter-inline-grid">
                    {trackerKalmanParameters.map(renderTargetingNumberParameter)}
                  </div>
                </details>
              </div>
            </details>
            </div>
            );
              return null;
            })}
            </div>
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
                      : formatRuntimeErrorMessage(kmnetNotice.lastError || kmnetNotice.lastDeviceError || "请检查地址、端口、UUID 和局域网连通性。")}</span>
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
                  <div className="kmnet-live-session">
                    <div>
                      <b>实时设备会话</b>
                      <small>连接由首页运行总开关统一维护；关闭运行后不会继续计算或发送新的偏移。</small>
                    </div>
                    <span className={kmnetRuntimeConnected ? "ui-badge success" : kmnetConnecting ? "ui-badge info" : "ui-badge neutral"}>
                      {kmnetRuntimeConnectionLabel}
                    </span>
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
                  <p>选择位移并确认后才发送一条命令；服务端会再次验证主链已停止、输出门已打开，并在完成后断开设备。</p>
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
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => requestDiagnosticMoveHardware(0, -10)}>↑</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => requestDiagnosticMoveHardware(-10, 0)}>←</button>
                  <button type="button" onClick={() => requestDiagnosticMoveHardware(kmnetTestDx, kmnetTestDy)} disabled={kmnetDiagnosticDisabled}>发送</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => requestDiagnosticMoveHardware(10, 0)}>→</button>
                  <button type="button" disabled={kmnetDiagnosticDisabled} onClick={() => requestDiagnosticMoveHardware(0, 10)}>↓</button>
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
                ? "状态通道不可用，暂不显示占位或配置值。"
                : runtimeLifecycleActive
                  ? "主链已请求运行；统计窗口完成后，这里会显示推理耗时、结果帧龄与实际吞吐。"
                  : "启动主链后才会建立真实统计窗口；未打点的阶段不会估算为 0。"}
              action={runtime === null ? (
                <div className="console-action-row">
                  <button className="console-button" onClick={() => setErrorCenterOpen(true)} type="button">
                    <NovaIcon name="triangle-alert" size={15} />
                    查看异常
                  </button>
                  <button className="console-button primary" disabled={busy !== null} onClick={() => void onRefresh()} type="button">
                    <NovaIcon name="refresh" size={15} />
                    重试
                  </button>
                </div>
              ) : !runtimeLifecycleActive ? (
                <button className="console-button primary" onClick={() => navigatePage("overview")} type="button">
                  <NovaIcon name="dashboard" size={15} />
                  前往首页开启
                </button>
              ) : undefined}
            />
          ) : (
          <>
            <div className="console-metrics consumer-signal-metrics">
              <Metric title="推理链耗时" value={formatOptionalNumber(inferenceTotalMs)} small="预处理 + 推理 + 解析 · ms" />
              <Metric title="结果帧龄" value={formatOptionalNumber(detectionDataAgeMs)} small={`${detectionFreshness}，单位 ms`} />
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
                <span className="class-config-eyebrow">参数设置 / 目标类别</span>
                <h2 id="class-config-dialog-title">编辑目标类别</h2>
                <p>raw cls 保留为模型本帧事实；这里单独定义内部名称、优先级和瞄点角色，不把类别当成永久身份。</p>
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
              <div className="class-config-workspace">
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
                  ? "正在保存并应用类别配置…"
                  : dialogSaveError
                    ? `处理失败 · ${dialogSaveError}`
                    : configDialogDirty
                      ? "有未保存修改 · 保存后会直接写入设备并应用。"
                      : "未修改 · 使用单一目标配置"}
              </span>
              <button type="button"
                className={`console-button ${configDialogDirty ? "primary dialog-save-button" : "dialog-close-button"}`}
                disabled={dialogSaving}
                onClick={() => void saveConfigDialog("class-config")}
              >
                {configDialogDirty ? "保存并应用" : "关闭"}
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
                <p>运行、网络与操作故障集中展示；原始详情供定位问题。本页会话内保留最近 20 条记录，刷新后清除。</p>
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
                    {item.requestId ? <small className="error-center-request-id">排查编号 {item.requestId}</small> : null}
                    <details>
                      <summary>原始错误与开发者详情</summary>
                      <pre>{item.technicalDetail || item.detail}</pre>
                    </details>
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
              <div className="error-center-clear-action">
                <button
                  aria-describedby={errorNotices.length === 0 && currentErrorDetails.length > 0 ? "error-center-clear-note" : undefined}
                  className="console-button"
                  disabled={errorNotices.length === 0}
                  onClick={requestClearErrorHistory}
                  title={errorNotices.length === 0 && currentErrorDetails.length > 0 ? "持续异常恢复后会自动消失" : undefined}
                  type="button"
                >
                  {errorNotices.length > 0 ? "清空历史记录" : "没有可清空的记录"}
                </button>
                {errorNotices.length === 0 && currentErrorDetails.length > 0 ? (
                  <small id="error-center-clear-note">当前异常仍由运行状态上报，恢复后会自动消失。</small>
                ) : null}
              </div>
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
          activeLoaded: runtimeInference?.loaded === true,
          runtimeBackend: readString(runtime?.inference?.selected, ""),
          runtimeInputShape: displayedInputShape,
          catalogMessage: modelCatalogMessage,
          catalogError: modelCatalogError,
          switchMessage: modelSwitch.message,
          busy,
          canSwitch: selectedCatalogModel?.kind === "engine",
          parserPreset,
          onParserPresetChange: setParserPreset,
          onRefresh: () => void refreshModelCatalog(),
          onCreateFolder: createModelFolder,
          onRequestMove: requestMoveModelEngine,
          onSelectModel: selectModelFromCatalog,
          onSaveMetadata: requestSaveModelMetadata,
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
