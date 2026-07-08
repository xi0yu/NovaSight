import { ChangeEvent, Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import {
  CaptureCapabilitiesResponse,
  CaptureCapability,
  CaptureSelectPayload,
  connectKmNet,
  diagnosticCircleKmNet,
  diagnosticMoveKmNet,
  disconnectKmNet,
  HealthResponse,
  ModelArtifact,
  ModelProject,
  ModelVersion,
  RuntimeConfig,
  RuntimeState,
  getCaptureCapabilities,
  getModelArtifacts,
  getModelVersions,
  publishModel,
  selectCaptureProfile,
  startRuntimePipeline,
  stopCapture,
  streamUrl,
  updateRuntimeConfig,
  updateRuntimeConfigField
} from "../../api";
import { getErrorMessage } from "../shared/format";

type ConsolePage = "capture" | "infer" | "params" | "stats" | "latency";

const DEFAULT_CONSOLE_PAGE: ConsolePage = "capture";
const CONSOLE_PAGES = new Set<ConsolePage>(["capture", "infer", "params", "stats", "latency"]);

type StudioConsoleViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  runtimeConfig: RuntimeConfig | null;
  projects: ModelProject[];
  errors: Partial<Record<string, string>>;
  lastUpdated: Date | null;
  onRefresh: () => Promise<void>;
};

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

const navItems: { id: ConsolePage; index: string; label: string }[] = [
  { id: "capture", index: "01", label: "采集" },
  { id: "infer", index: "02", label: "模型推理" },
  { id: "params", index: "03", label: "参数设置" },
  { id: "stats", index: "04", label: "统计" },
  { id: "latency", index: "05", label: "采集延迟" }
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
const KMNET_RECOMMENDED = {
  host: "192.168.2.188",
  port: 8888,
  uuid: "12345678",
  monitor_port: 5001
};

const EXPERIMENTAL_ANGLE_DEFAULTS = {
  experimental_angle_kp_x: 0.35,
  experimental_angle_kp_y: 0.24,
  experimental_angle_ki: 0,
  experimental_angle_kd: 0,
  experimental_angle_integral_limit: 0,
  experimental_angle_config_level: "basic",
  experimental_angle_speed: 1,
  experimental_angle_smooth_factor: 0,
  experimental_angle_deadzone_px: 0,
  experimental_angle_derivative_filter: 1,
  experimental_angle_max_step_counts: 80,
  experimental_angle_control_hz: 60,
  experimental_angle_kalman_enabled: true,
  experimental_angle_kalman_process_noise: 2,
  experimental_angle_kalman_measurement_noise: 16,
  experimental_angle_hungarian_enabled: true,
  experimental_angle_matching_distance_px: 140,
  experimental_angle_max_extrapolate_frames: 3,
  experimental_angle_target_filter_enabled: true,
  experimental_angle_target_filter_min_score: 0,
  experimental_angle_target_filter_fov_ratio: 1,
  experimental_angle_target_filter_same_class: false,
  experimental_angle_prediction_lead_ms: 0,
  experimental_angle_extrapolate_confidence_decay: 1,
  experimental_angle_magnet_enabled: false,
  experimental_angle_magnet_radius_px: 120,
  experimental_angle_magnet_strength: 0.25,
  experimental_angle_magnet_curve: 1,
  experimental_angle_magnet_deadzone_px: 0,
  experimental_angle_magnet_max_counts: 20
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

function readTraceStages(value: unknown): Record<string, unknown>[] {
  const stages = asRecord(value).stages;
  return Array.isArray(stages) ? stages.map(asRecord) : [];
}

function triggerModeLabel(value: string): string {
  if (value === "always") {
    return "总是启用";
  }
  return "kmNet 硬件触发";
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, value));
}

function formatNumber(value: unknown, digits = 1): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? number.toFixed(digits) : "待机";
}

function formatPercent(value: unknown, digits = 1): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "待机";
}

function formatShape(value: unknown): string {
  return Array.isArray(value) && value.length > 0 ? value.map((item) => String(item)).join("x") : "-";
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
  const [kmnetTestDx, setKmnetTestDx] = useState(10);
  const [kmnetTestDy, setKmnetTestDy] = useState(0);
  const [kmnetTestMs, setKmnetTestMs] = useState(300);
  const [kmnetBezierX1, setKmnetBezierX1] = useState(-50);
  const [kmnetBezierY1, setKmnetBezierY1] = useState(-60);
  const [kmnetBezierX2, setKmnetBezierX2] = useState(70);
  const [kmnetBezierY2, setKmnetBezierY2] = useState(80);
  const [kmnetTestMessage, setKmnetTestMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [modelSwitchMessage, setModelSwitchMessage] = useState("");
  const [configDraft, setConfigDraft] = useState<RuntimeConfig | null>(() => cloneRuntimeConfig(runtimeConfig));
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const configDraftRef = useRef<RuntimeConfig | null>(cloneRuntimeConfig(runtimeConfig));
  const pendingConfigWritesRef = useRef(0);
  const configWriteSeqRef = useRef(0);

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

  const capture = runtime?.capture;
  const statistics = runtime?.statistics ?? capture?.statistics;
  const captureToTensorMetaMs = readNumber(statistics?.stage_capture_to_tensor_meta_ms, Number.NaN);
  const hasCaptureToTensorMetaMs = Number.isFinite(captureToTensorMetaMs);
  const inferenceLatencyDisplay = hasCaptureToTensorMetaMs
    ? captureToTensorMetaMs
    : (statistics?.stage_engine_ms ?? statistics?.inference_latency);
  const config = configDraft ?? runtimeConfig;
  const captureConfig = nestedRecord(config, "capture");
  const configuredCaptureDevice = readString(captureConfig.device, "");
  const configuredCapturePixelFormat = readString(captureConfig.pixel_format, "");
  const configuredCaptureWidth = readNumber(captureConfig.width, 0);
  const configuredCaptureHeight = readNumber(captureConfig.height, 0);
  const configuredCaptureFps = readNumber(captureConfig.fps, 0);
  const roiConfig = nestedRecord(config, "roi");
  const inferenceConfig = nestedRecord(config, "inference");
  const controlConfig = nestedRecord(config, "control");
  const calibrationConfig = nestedRecord(config, "calibration");
  const hardwareConfig = nestedRecord(config, "hardware");
  const consumersConfig = nestedRecord(config, "consumers");
  const vision = asRecord(runtime?.vision);
  const execution = asRecord(vision.execution);
  const executionIntent = asRecord(execution.intent);
  const executionMeta = asRecord(execution.metadata);
  const inferenceTrace = asRecord(vision.inference);
  const runtimeInference = asRecord(runtime?.inference);
  const deepstreamRuntimeSelected = readString(runtimeInference.selected, "") === "deepstream";
  const deepstreamTerminalError = readBoolean(runtimeInference.terminal_error, false);
  const runtimeInferenceConfigured = runtimeInference.configured === true;
  const runtimeInferenceLoaded = runtimeInference.loaded === true;
  const runtimeInferenceReason = readString(runtimeInference.reason, "");
  const runtimeInferenceDetail = readString(runtimeInference.detail, "");
  const captureMainRunning = deepstreamRuntimeSelected ? runtime?.running === true : capture?.available === true;
  const captureMainConfigured = capture?.available === true || runtimeInferenceConfigured;
  const engineStatusLabel = deepstreamTerminalError
    ? "管线故障"
    : runtimeInferenceLoaded
      ? "已加载"
      : runtimeInferenceConfigured
        ? "待启动"
        : "未配置";
  const deepstreamStatusLabel = deepstreamTerminalError
    ? "管线故障"
    : runtimeInferenceLoaded
      ? "运行中"
      : runtimeInferenceConfigured
        ? "待启动"
        : "未配置";
  const runtimeModelOutput = asRecord(runtimeInference.model_output);
  const runtimePostprocess = asRecord(runtimeInference.postprocess);
  const runtimeModelOutputClassNames = stringArray(runtimeModelOutput.class_names);
  const pipeline = asRecord(runtime?.pipeline);
  const executorStatus = asRecord(runtime?.executor);
  const executors = asRecord(executorStatus.executors);
  const kmnetStatus = asRecord(executors.kmnet);
  const selectedProfile = capture?.profile;
  const runningPixelFormat = selectedProfile?.pixel_format ?? "";
  const runningWidth = selectedProfile?.width ?? 0;
  const runningHeight = selectedProfile?.height ?? 0;
  const runningFps = selectedProfile?.fps ?? 0;
  const choices = useMemo(() => groupCapabilities(caps?.capabilities ?? []), [caps]);
  const selectedChoice =
    choices.find((choice) => choiceId(choice) === selectedChoiceId) ?? choices[0];
  const roiSize = readNumber(roiConfig.size, 640);
  const roiOffsetX = readNumber(roiConfig.offset_x, 0);
  const roiOffsetY = readNumber(roiConfig.offset_y, 0);
  const sourceWidth = selectedProfile?.width ?? readNumber(inferenceTrace.source_width, 0);
  const sourceHeight = selectedProfile?.height ?? readNumber(inferenceTrace.source_height, 0);
  const roiBaseX = sourceWidth > 0 ? Math.max(0, Math.floor((sourceWidth - roiSize) / 2)) : 0;
  const roiBaseY = sourceHeight > 0 ? Math.max(0, Math.floor((sourceHeight - roiSize) / 2)) : 0;
  const roiX = sourceWidth > 0 ? clampNumber(roiBaseX + roiOffsetX, 0, Math.max(0, sourceWidth - roiSize)) : roiOffsetX;
  const roiY = sourceHeight > 0 ? clampNumber(roiBaseY + roiOffsetY, 0, Math.max(0, sourceHeight - roiSize)) : roiOffsetY;
  const confidence = readNumber(inferenceConfig.confidence_threshold, 0.25);
  const nms = readNumber(inferenceConfig.nms_threshold, 0.45);
  const detectionProfiles = recordList(inferenceConfig.detection_class_profiles);
  const activeDetectionProfile = readString(inferenceConfig.detection_class_profile, "default");
  const activeDetectionClass = readString(inferenceConfig.detection_class_filter, "all");
  const detectionProfileNames = Object.keys(detectionProfiles);
  const detectionClasses = detectionProfiles[activeDetectionProfile] ?? detectionProfiles.default ?? [];
  const detectionClassPriority = readString(inferenceConfig.detection_class_priority, "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
  const controlStrategy = "experimental_angle_pid";
  const controlMinConfidence = readNumber(controlConfig.min_confidence, 0);
  const selectionFovRatio = readNumber(controlConfig.fov_ratio, 0.28);
  const targetLostGraceFrames = readNumber(controlConfig.target_lost_grace_frames, 5);
  const candidateRatioMaxAspect = readNumber(controlConfig.candidate_ratio_max_aspect, 6);
  const candidateQualityConfidenceWeight = readNumber(controlConfig.candidate_quality_confidence_weight, 0.7);
  const candidateQualityAreaWeight = readNumber(controlConfig.candidate_quality_area_weight, 0.3);
  const classPriorityQualityMargin = readNumber(controlConfig.class_priority_quality_margin, 0.08);
  const trackerConfirmFrames = readNumber(controlConfig.tracker_confirm_frames, 2);
  const trackerMatchingDistancePx = readNumber(controlConfig.tracker_matching_distance_px, 140);
  const trackerAmbiguityMargin = readNumber(controlConfig.tracker_ambiguity_margin, 0.08);
  const trackerMissingTimeoutMs = readNumber(controlConfig.tracker_missing_timeout_ms, 120);
  const trackerDeleteTimeoutMs = readNumber(controlConfig.tracker_delete_timeout_ms, 250);
  const trackerMatchThreshold = readNumber(controlConfig.tracker_match_threshold, 0.65);
  const trackerMahalanobisGate = readNumber(controlConfig.tracker_mahalanobis_gate, 9.21);
  const targetSwitchPreferenceAdvantage = readNumber(controlConfig.target_switch_min_preference_advantage, 0.08);
  const targetSwitchContinuityScore = readNumber(controlConfig.target_switch_min_continuity_score, 0.7);
  const targetSwitchConfirmFrames = readNumber(controlConfig.target_switch_confirm_frames, 3);
  const kalmanEnabled = readBoolean(controlConfig.kalman_enabled, true);
  const kalmanAccelerationNoise = readNumber(controlConfig.kalman_acceleration_noise, 1200);
  const kalmanMeasurementNoiseX = readNumber(controlConfig.kalman_measurement_noise_x, 16);
  const kalmanMeasurementNoiseY = readNumber(controlConfig.kalman_measurement_noise_y, 16);
  const kalmanMaxPredictMissingMs = readNumber(controlConfig.kalman_max_predict_missing_ms, 80);
  const kalmanMaxPredictSteps = readNumber(controlConfig.kalman_max_predict_steps, 5);
  const kalmanMaxPredictDtMs = readNumber(controlConfig.kalman_max_predict_dt_ms, 35);
  const kalmanMaxPositionSigmaPx = readNumber(controlConfig.kalman_max_position_sigma_px, 45);
  const kalmanMaxCovarianceTrace = readNumber(controlConfig.kalman_max_covariance_trace, 5000);
  const kalmanNisThreshold = readNumber(controlConfig.kalman_nis_threshold, 9.21);
  const kalmanNisHardReject = readNumber(controlConfig.kalman_nis_hard_reject, 16);
  const kalmanMinIdentityConfidence = readNumber(controlConfig.kalman_min_identity_confidence, 0.7);
  const kalmanMinPredictionConfidence = readNumber(controlConfig.kalman_min_prediction_confidence, 0.35);
  const kalmanPredictionDecayTauMs = readNumber(controlConfig.kalman_prediction_decay_tau_ms, 45);
  const moveKind = readString(controlConfig.move_kind, "bezier");
  const moveMs = readNumber(controlConfig.move_ms, 12);
  const commandIntervalMs = readNumber(controlConfig.command_interval_ms, 1);
  const schedulerCommandTtlMs = readNumber(controlConfig.scheduler_command_ttl_ms, 35);
  const schedulerPredictedCommandTtlMs = readNumber(controlConfig.scheduler_predicted_command_ttl_ms, 18);
  const schedulerCancelOnNewFrame = readBoolean(controlConfig.scheduler_cancel_on_new_frame, true);
  const schedulerCancelOnDirectionChange = readBoolean(controlConfig.scheduler_cancel_on_direction_change, true);
  const schedulerCancelOnTrackChange = readBoolean(controlConfig.scheduler_cancel_on_track_change, true);
  const schedulerDeviceErrorCooldownMs = readNumber(controlConfig.scheduler_device_error_cooldown_ms, 50);
  const experimentalAngleKpX = readNumber(controlConfig.experimental_angle_kp_x, 0.35);
  const experimentalAngleKpY = readNumber(controlConfig.experimental_angle_kp_y, 0.24);
  const experimentalAngleKi = readNumber(controlConfig.experimental_angle_ki, 0);
  const experimentalAngleKd = readNumber(controlConfig.experimental_angle_kd, 0);
  const experimentalAngleIntegralLimit = readNumber(controlConfig.experimental_angle_integral_limit, 0);
  const experimentalAngleConfigLevel = readString(controlConfig.experimental_angle_config_level, "basic");
  const experimentalAngleSpeed = readNumber(controlConfig.experimental_angle_speed, 1);
  const experimentalAngleSmoothFactor = readNumber(controlConfig.experimental_angle_smooth_factor, 0);
  const experimentalAngleDeadzonePx = readNumber(controlConfig.experimental_angle_deadzone_px, 0);
  const experimentalAngleDerivativeFilter = readNumber(controlConfig.experimental_angle_derivative_filter, 1);
  const experimentalAngleAdvanced = experimentalAngleConfigLevel === "advanced" || experimentalAngleConfigLevel === "developer";
  const experimentalAngleDeveloper = experimentalAngleConfigLevel === "developer";
  const experimentalAngleFovX = readNumber(calibrationConfig.fov_x_deg, 105);
  const experimentalAngleC360X = readNumber(calibrationConfig.counts_per_360_x, 9980);
  const experimentalAngleC360Y = readNumber(calibrationConfig.counts_per_360_y, experimentalAngleC360X);
  const experimentalAngleMaxStep = readNumber(controlConfig.experimental_angle_max_step_counts, 80);
  const experimentalAngleControlHz = readNumber(controlConfig.experimental_angle_control_hz, 60);
  const experimentalAngleSignX = readNumber(calibrationConfig.axis_sign_x, 1);
  const experimentalAngleSignY = readNumber(calibrationConfig.axis_sign_y, 1);
  const experimentalAngleKalmanEnabled = readBoolean(controlConfig.experimental_angle_kalman_enabled, true);
  const experimentalAngleKalmanProcessNoise = readNumber(controlConfig.experimental_angle_kalman_process_noise, 2);
  const experimentalAngleKalmanMeasurementNoise = readNumber(controlConfig.experimental_angle_kalman_measurement_noise, 16);
  const experimentalAngleHungarianEnabled = readBoolean(controlConfig.experimental_angle_hungarian_enabled, true);
  const experimentalAngleMatchingDistance = readNumber(controlConfig.experimental_angle_matching_distance_px, 140);
  const experimentalAngleMaxExtrapolateFrames = readNumber(controlConfig.experimental_angle_max_extrapolate_frames, 3);
  const experimentalAngleTargetFilterEnabled = readBoolean(controlConfig.experimental_angle_target_filter_enabled, true);
  const experimentalAngleTargetFilterMinScore = readNumber(controlConfig.experimental_angle_target_filter_min_score, 0);
  const experimentalAngleTargetFilterFovRatio = readNumber(controlConfig.experimental_angle_target_filter_fov_ratio, 1);
  const experimentalAngleTargetFilterSameClass = readBoolean(controlConfig.experimental_angle_target_filter_same_class, false);
  const experimentalAnglePredictionLeadMs = readNumber(controlConfig.experimental_angle_prediction_lead_ms, 0);
  const experimentalAngleExtrapolateConfidenceDecay = readNumber(controlConfig.experimental_angle_extrapolate_confidence_decay, 1);
  const experimentalAngleMagnetEnabled = readBoolean(controlConfig.experimental_angle_magnet_enabled, false);
  const experimentalAngleMagnetRadiusPx = readNumber(controlConfig.experimental_angle_magnet_radius_px, 120);
  const experimentalAngleMagnetStrength = readNumber(controlConfig.experimental_angle_magnet_strength, 0.25);
  const experimentalAngleMagnetCurve = readNumber(controlConfig.experimental_angle_magnet_curve, 1);
  const experimentalAngleMagnetDeadzonePx = readNumber(controlConfig.experimental_angle_magnet_deadzone_px, 0);
  const experimentalAngleMagnetMaxCounts = readNumber(controlConfig.experimental_angle_magnet_max_counts, 20);
  const hardwareKind = readString(hardwareConfig.kind, "kmnet") || "kmnet";
  const outputMode = readString(controlConfig.output_mode, "kmnet") || "kmnet";
  const triggerMode = readString(controlConfig.trigger_mode, "hardware");
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);

  useEffect(() => {
    if (!runtimeConfig || pendingConfigWritesRef.current > 0) {
      return;
    }
    const next = normalizeRuntimeConfig(runtimeConfig);
    configDraftRef.current = next;
    setConfigDraft(next);
  }, [runtimeConfig]);
  const kmnetConnected = kmnetStatus.connected === true;
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
  const registeredInputShape = version?.input_shape ?? "";
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
  const switchableArtifacts = modelArtifacts.filter(
    (item) =>
      item.status === "ready" &&
      (item.kind === "onnx" || item.kind === "engine") &&
      (selectedModelVersionId === "" || item.version_id === selectedModelVersionId)
  );
  const sortedSwitchableArtifacts = [...switchableArtifacts].sort((left, right) => {
    const leftRank = ARTIFACT_KIND_RANK[left.kind] ?? 99;
    const rightRank = ARTIFACT_KIND_RANK[right.kind] ?? 99;
    return leftRank - rightRank || left.path.localeCompare(right.path);
  });
  const selectedSwitchArtifact =
    sortedSwitchableArtifacts.find((item) => item.id === selectedModelArtifactId) ??
    sortedSwitchableArtifacts[0] ??
    null;
  const preferredSwitchArtifact = sortedSwitchableArtifacts[0] ?? null;
  const detections = readNumber(vision.detections, 0);
  const target = asRecord(vision.target);
  const control = asRecord(vision.control);
  const selectorDebug = asRecord(control.selector_debug);
  const trackDiagnostics = asRecord(control.track_diagnostics ?? target.track_diagnostics);
  const diagnosticTracks = recordArray(trackDiagnostics.tracks);
  const selectedTrackId = finiteNumber(trackDiagnostics.selected_track_id);
  const selectedTrackDebug =
    selectedTrackId === null
      ? {}
      : diagnosticTracks.find((item) => readNumber(item.track_id, Number.NaN) === selectedTrackId) ?? {};
  const selectedTrackEstimate = asRecord(selectedTrackDebug.estimate);
  const triggerRaw = asRecord(control.trigger_raw);
  const triggerLeft = triggerRaw.left === true;
  const triggerRight = triggerRaw.right === true;
  const triggerAlwaysActive = triggerRaw.source === "always" && triggerRaw.active === true;
  const businessTrace = asRecord(vision.trace);
  const businessTraceStages = readTraceStages(vision.trace);
  const controlPipeline = asRecord(asRecord(vision.control).pipeline);
  const rawDetections = readNumber(inferenceTrace.raw_detections, 0);
  const mappedDetections = readNumber(inferenceTrace.mapped_detections, 0);
  const inferenceRan = inferenceTrace.ran === true;
  const inferenceAvailable = inferenceTrace.available === true;
  const inferenceReason = readString(inferenceTrace.reason, readString(vision.inference_reason, "-"));
  const inferenceDebug = asRecord(inferenceTrace.debug);
  const decodeDebug = asRecord(inferenceDebug.decode);
  const inferenceTimings = asRecord(inferenceDebug.timings);
  const trtTimings = asRecord(decodeDebug.timings);
  const preprocessDebug = asRecord(inferenceDebug.preprocess);
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
  const lastError = localError ?? Object.values(errors)[0] ?? capture?.last_error;

  useEffect(() => {
    if (configuredCaptureDevice) {
      setDevice(configuredCaptureDevice);
    } else if (runtime?.capture?.device) {
      setDevice(runtime.capture.device);
    }
  }, [configuredCaptureDevice, runtime?.capture?.device]);

  useEffect(() => {
    if (!selectedChoiceId && selectedProfile) {
      setSelectedChoiceId(
        `${selectedProfile.pixel_format.toUpperCase()}:${selectedProfile.width}x${selectedProfile.height}@${selectedProfile.fps}`
      );
    }
  }, [selectedChoiceId, selectedProfile]);

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
      setModelVersions([]);
      setSelectedModelVersionId("");
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
      return;
    }
    let cancelled = false;
    setModelVersions([]);
    setSelectedModelVersionId("");
    setModelArtifacts([]);
    setSelectedModelArtifactId("");
    getModelVersions(selectedModelProjectId)
      .then((items) => {
        if (cancelled) {
          return;
        }
        setModelVersions(items);
        setSelectedModelVersionId((current) => {
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
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runtime?.active_model?.version?.id, selectedModelProjectId]);

  useEffect(() => {
    if (selectedModelVersionId === "") {
      setModelArtifacts([]);
      setSelectedModelArtifactId("");
      return;
    }
    let cancelled = false;
    setModelArtifacts([]);
    setSelectedModelArtifactId("");
    getModelArtifacts(selectedModelVersionId)
      .then((items) => {
        if (cancelled) {
          return;
        }
        setModelArtifacts(items);
        const runnable = items
          .filter(
            (item) => item.status === "ready" && (item.kind === "onnx" || item.kind === "engine")
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
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedModelVersionId]);

  const refreshCapabilities = useCallback(async () => {
    setBusy("caps");
    setLocalError(null);
    try {
      const result = await getCaptureCapabilities(device);
      setCaps(result);
      if (!result.available) {
        setLocalError(result.reason || "设备不可用");
      }
      const grouped = groupCapabilities(result.capabilities);
      const configured = grouped.find((choice) => choiceMatchesConfig(choice, {
        pixel_format: configuredCapturePixelFormat,
        width: configuredCaptureWidth,
        height: configuredCaptureHeight,
        fps: configuredCaptureFps
      }));
      const running = grouped.find((choice) => runningPixelFormat && (
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

  const applyCapture = useCallback(async () => {
    const choice = selectedChoice;
    setBusy("capture");
    setLocalError(null);
    const payload: CaptureSelectPayload = choice
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
    try {
      await selectCaptureProfile(payload);
      await onRefresh();
    } catch (err) {
      setLocalError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [device, onRefresh, selectedChoice, selectedProfile]);

  const stopCurrentCapture = useCallback(async () => {
    setBusy("stop");
    setLocalError(null);
    try {
      await stopCapture();
      await onRefresh();
    } catch (err) {
      setLocalError(getErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  const startInferenceThread = useCallback(async () => {
    setBusy("runtime.start");
    setLocalError(null);
    try {
      await startRuntimePipeline();
      await onRefresh();
    } catch (err) {
      setLocalError(`启动推理失败：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [onRefresh]);

  const toggleCapture = useCallback(async () => {
    if (captureMainRunning) {
      await stopCurrentCapture();
      return;
    }
    await applyCapture();
    if (deepstreamRuntimeSelected) {
      await startInferenceThread();
    }
  }, [applyCapture, captureMainRunning, deepstreamRuntimeSelected, startInferenceThread, stopCurrentCapture]);

  const updateConfigField = useCallback(
    async (section: string, key: string, value: number | string | boolean | string[]) => {
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

  const resetExperimentalAngleDefaults = useCallback(async () => {
    const base = configDraftRef.current ?? cloneRuntimeConfig(runtimeConfig);
    const next = base ? normalizeRuntimeConfig(base) : null;
    if (!next) {
      return;
    }
    const writeSeq = ++configWriteSeqRef.current;
    pendingConfigWritesRef.current += 1;
    setBusy("control.experimental_angle_reset");
    setLocalError(null);
    const control = {
      ...asRecord(next.control),
      ...EXPERIMENTAL_ANGLE_DEFAULTS
    };
    next.control = control as RuntimeConfig[string];
    configDraftRef.current = next;
    setConfigDraft(next);
    try {
      const result = await updateRuntimeConfig(next);
      if (writeSeq === configWriteSeqRef.current) {
        const applied = normalizeRuntimeConfig(result.config);
        configDraftRef.current = applied;
        setConfigDraft(applied);
      }
    } catch (err) {
      setLocalError(`实验角度 PID 重置失败：${getErrorMessage(err)}`);
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
  }, [onRefresh, runtimeConfig]);

  const updateHardwareKind = useCallback(
    async (_kind: string) => {
      const next = cloneRuntimeConfig(runtimeConfig);
      if (!next) {
        return;
      }
      setBusy("hardware.kind");
      setLocalError(null);
      next.hardware = {
        ...asRecord(next.hardware),
        kind: "kmnet",
        ...KMNET_RECOMMENDED
      } as RuntimeConfig[string];
      const control = {
        ...asRecord(next.control),
        output_mode: "kmnet"
      };
      next.control = control as RuntimeConfig[string];
      try {
        await updateRuntimeConfig(next);
        await onRefresh();
      } catch (err) {
        setLocalError(`配置同步失败：${getErrorMessage(err)}`);
      } finally {
        setBusy(null);
      }
    },
    [onRefresh, runtimeConfig]
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
      kind: "kmnet",
      ...KMNET_RECOMMENDED
    } as RuntimeConfig[string];
    next.control = {
      ...asRecord(next.control),
      output_mode: "kmnet"
    } as RuntimeConfig[string];
    try {
      await updateRuntimeConfig(next);
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 推荐参数应用失败：${getErrorMessage(err)}`);
    } finally {
      setBusy(null);
    }
  }, [onRefresh, runtimeConfig]);

  const toggleHardwareConnection = useCallback(async () => {
    setBusy("kmnet.toggle");
    setLocalError(null);
    try {
      if (kmnetConnected) {
        await disconnectKmNet();
      } else {
        await connectKmNet();
      }
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet ${kmnetConnected ? "断开" : "连接"}失败：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetConnected, onRefresh]);

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
    } finally {
      setBusy(null);
    }
  };

  const switchModel = async () => {
    if (selectedModelProjectId === "" || selectedSwitchArtifact === null) {
      setLocalError("请选择可推理的 ONNX 或 TensorRT engine 产物。");
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
      const response = await publishModel(selectedModelProjectId, selectedSwitchArtifact.id);
      setModelSwitchMessage(response.report?.message ?? "模型已切换，推理运行态已刷新。");
      await onRefresh();
    } catch (err) {
      setLocalError(`模型切换失败，当前运行模型已保留：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="console-app">
      <header className="console-top">
        <div className="console-brand">
          <div className="console-logo" />
          NovaSight Studio
        </div>
        <div className="console-toolbar">
          <div className="console-group">
            <div>项目：{projects[0]?.name ?? "默认项目"}⌄</div>
            <div><span className="console-dot green" />设备：Jetson Orin Nano⌄</div>
          </div>
          <div className="console-group">
            <div><span className="console-dot" />{health?.ok ? "在线" : "离线"}</div>
            <div className="console-pill">退出</div>
            <div className="console-live">实时推送已连接 · {formatDate(lastUpdated)}</div>
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
            <span>{item.index}</span>
            <b>{item.label}</b>
          </button>
        ))}
      </aside>

      <main className="console-main">
        <section className="console-process">
          <div className="console-process-state">
            <span className="console-dot" />
            采集：{captureMainRunning ? "运行中" : captureMainConfigured ? "已配置" : "已停止"} · 推理：{runtime?.running ? "运行中" : "已停止"}
          </div>
          <button
            className={captureMainRunning ? "console-button danger" : "console-button primary"}
            disabled={busy === "capture" || busy === "stop" || busy === "runtime.start"}
            onClick={() => void toggleCapture()}
            type="button"
          >
            {captureMainRunning ? "▪ 停止采集" : deepstreamRuntimeSelected ? "▶ 启动主链" : "▶ 启动采集"}
          </button>
          {!deepstreamRuntimeSelected && !runtime?.running && capture?.available ? (
            <button
              className="console-button"
              disabled={busy === "runtime.start"}
              onClick={() => void startInferenceThread()}
              type="button"
            >
              恢复推理线程
            </button>
          ) : null}
          <button className="console-button" onClick={exportConfig} type="button">导出配置...</button>
          <button className="console-button" onClick={() => fileInputRef.current?.click()} type="button">导入配置...</button>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={importConfig} />
        </section>

        {lastError ? <div className="console-error">{lastError}</div> : null}

        <section className={activePage === "capture" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="采集 FPS" value={formatNumber(statistics?.capture_fps ?? capture?.fps_capture, 1)} small="FPS" />
            <Metric title="分辨率" value={selectedProfile ? `${selectedProfile.width}x${selectedProfile.height}` : "待机"} small="input" />
            <Metric title="像素格式" value={selectedProfile?.pixel_format ?? "待机"} small="format" />
            <Metric title="丢帧" value={String(statistics?.dropped_counter ?? capture?.frames_dropped ?? 0)} small="drop" />
          </div>

          <div className="console-grid1">
            <div>
              <div className="console-card">
                <h2 className="console-title">采集设备</h2>
                <label>视频设备</label>
                <input value={device} onChange={(event) => setDevice(event.target.value)} />
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
                <div className="console-row">
                  <label>目标帧率</label>
                  <input value={selectedChoice?.fps ?? selectedProfile?.fps ?? ""} readOnly />
                </div>
                <button className="console-button primary" disabled={busy === "caps"} onClick={refreshCapabilities} type="button">
                  {busy === "caps" ? "检测中..." : "检测设备能力"}
                </button>
              </div>

              <div className="console-card">
                <h2 className="console-title">ROI 裁剪</h2>
                <label>ROI 模式</label>
                <select value={roiOffsetX === 0 && roiOffsetY === 0 ? "center" : "manual"} disabled>
                  <option value="center">中心正方形</option>
                  <option value="manual">手动偏移</option>
                </select>
                <label>ROI 尺寸</label>
                <div className="console-row">
                  <input
                    type="range"
                    min="256"
                    max="640"
                    step="64"
                    value={roiSize}
                    onChange={(event) => void updateConfigField("roi", "size", nearestRoiSize(Number(event.target.value)))}
                  />
                  <select value={roiSize} onChange={(event) => void updateConfigField("roi", "size", Number(event.target.value))}>
                    {ROI_SIZE_CHOICES.map((size) => <option key={size} value={size}>{size}</option>)}
                  </select>
                </div>
                <NumberControl label="水平偏移" value={roiOffsetX} min={-1280} max={1280} step={16} onCommit={(value) => updateConfigField("roi", "offset_x", Math.round(value))} />
                <NumberControl label="垂直偏移" value={roiOffsetY} min={-720} max={720} step={16} onCommit={(value) => updateConfigField("roi", "offset_y", Math.round(value))} />
                <div className="console-kv compact-kv">
                  <span>源画面</span><b>{sourceWidth > 0 ? `${sourceWidth}x${sourceHeight}` : "等待采集"}</b>
                  <span>ROI 区域</span><b>{sourceWidth > 0 ? `x=${roiX}, y=${roiY}` : "等待采集"}</b>
                  <span>坐标系</span><b>推理 / 预览 / 控制统一 ROI</b>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={activePage === "infer" ? "console-page active" : "console-page"}>
          <div className="console-tabs">
            <div className="console-tab active"><span className="console-dot dark" />配置1</div>
            <div className="console-tab"><span className="console-dot" />配置2</div>
            <div className="console-tab"><span className="console-dot" />配置3</div>
          </div>
          <div className="console-metrics">
            <Metric title="推理 FPS" value={formatNumber(statistics?.inference_fps, 1)} small="FPS" />
            <Metric title={hasCaptureToTensorMetaMs ? "Tensor Meta延迟" : "推理延迟"} value={formatNumber(inferenceLatencyDisplay, 1)} small="ms" />
            <Metric title="目标数量" value={String(detections)} small="objects" />
            <Metric title="引擎状态" value={engineStatusLabel} small={readString(runtime?.inference?.selected, "engine")} />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">模型设置</h2>
              <label>模型文件</label>
              <select
                value={selectedModelProjectId}
                onChange={(event) => {
                  const nextProjectId =
                    event.target.value === "" ? "" : Number(event.target.value);
                  setSelectedModelProjectId(
                    typeof nextProjectId === "number" && Number.isFinite(nextProjectId)
                      ? nextProjectId
                      : ""
                  );
                  setSelectedModelVersionId("");
                  setSelectedModelArtifactId("");
                  setModelVersions([]);
                  setModelArtifacts([]);
                }}
              >
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>{project.name}</option>
                ))}
                {projects.length === 0 ? <option value="">未发现模型</option> : null}
              </select>
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
                disabled={busy === "model.switch" || selectedModelProjectId === "" || selectedSwitchArtifact === null}
                onClick={switchModel}
                type="button"
              >
                {busy === "model.switch" ? "安全切换中..." : "安全切换模型"}
              </button>
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
              <label>检测类别</label>
              <select
                value={activeDetectionProfile}
                onChange={(event) => void updateConfigField("inference", "detection_class_profile", event.target.value)}
              >
                {(detectionProfileNames.length > 0 ? detectionProfileNames : ["default"]).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
              <div className="console-class-list">
                <button
                  className={activeDetectionClass === "all" ? "console-class-row active" : "console-class-row"}
                  onClick={() => void updateConfigField("inference", "detection_class_filter", "all")}
                  type="button"
                >
                  全部类别
                </button>
                {detectionClasses.map((item, index) => (
                  <button
                    className={activeDetectionClass === String(index) ? "console-class-row active" : "console-class-row"}
                    key={`${index}-${item}`}
                    onClick={() => void updateConfigField("inference", "detection_class_filter", String(index))}
                    type="button"
                  >
                    {item}
                  </button>
                ))}
              </div>
              <TextControl
                label="类别优先级"
                value={detectionClassPriority}
                onCommit={(value) => updateConfigField("inference", "detection_class_priority", value)}
              />
              <p className="console-field-hint">按 class id 从高到低填写，例如 1,0 表示先选头部，再选身体；未列出的类别会排在后面。</p>
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
                    <option key={item.id} value={item.id}>{item.kind} · {item.path}</option>
                  ))}
                  {sortedSwitchableArtifacts.length === 0 ? <option value="">暂无 ONNX / engine ready 产物</option> : null}
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
              <h2 className="console-title">推理输出</h2>
              <PreviewFrame
                enabled={activePage === "infer" && previewEnabled}
                imageEnabled={!deepstreamRuntimeSelected}
                runtime={runtime}
                roiSize={roiSize}
              />
              <div className="business-trace">
                <div className="business-trace-head">
                  <span>主营链路诊断</span>
                  <b>{readString(businessTrace.message, "等待运行状态")}</b>
                </div>
                <div className="business-trace-steps">
                  {businessTraceStages.map((stage) => (
                    <div className={`business-trace-step ${readString(stage.status, "blocked")}`} key={readString(stage.id, readString(stage.label, ""))}>
                      <span>{readString(stage.label, "-")}</span>
                      <b>{readString(stage.message, "-")}</b>
                      <small>{readString(stage.detail, "") || readString(stage.status, "-")}</small>
                    </div>
                  ))}
                  {businessTraceStages.length === 0 ? (
                    <div className="business-trace-step blocked">
                      <span>链路</span>
                      <b>等待运行状态</b>
                      <small>暂无 trace</small>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="console-kv">
                <span>推理状态</span><b>{inferenceRan ? (inferenceAvailable ? "已执行" : "执行失败") : "未执行"}</b>
                <span>推理原因</span><b>{inferenceReason || "-"}</b>
                <span>DeepStream 状态</span><b>{deepstreamStatusLabel}</b>
                <span>DeepStream 原因</span><b>{runtimeInferenceReason || "-"}</b>
                <span className="wide">DeepStream 详情</span><b className="wide">{runtimeInferenceDetail || "-"}</b>
                <span>raw 检测</span><b>{String(rawDetections)}</b>
                <span>前端检测</span><b>{String(mappedDetections)}</b>
                <span>ROI 输入</span><b>{`${roiInputWidth || "-"}x${roiInputHeight || "-"}`}</b>
                <span>模型输入</span><b>{modelInputWidth && modelInputHeight ? `${modelInputWidth}x${modelInputHeight}` : "-"}</b>
                <span>压缩倍率</span><b>{inputDownscaleFactor ? `${formatNumber(inputDownscaleFactor, 2)}x` : "-"}</b>
                <span>有效像素</span><b>{inputPixelRatio ? formatPercent(inputPixelRatio, 1) : "-"}</b>
                <span>输出形状</span><b>{formatShape(decodeDebug.output_shape ?? inferenceDebug.output_shape)}</b>
                <span>运行输出</span><b>{deepstreamModelOutputSummary}</b>
                <span>运行类别</span><b>{formatNumber(runtimeModelOutput.class_count, 0)}</b>
                <span>运行解析器</span><b>{runtimePostprocessParser}</b>
                <span>运行置信度</span><b>{formatNumber(runtimePostprocessConfidence, 2)}</b>
                <span>运行 NMS</span><b>{formatNumber(runtimePostprocessNms, 2)}</b>
                <span className="wide">运行类别名</span><b className="wide">{deepstreamClassNamesSummary}</b>
                <span>输入准备</span><b>{formatNumber(inferenceTimings.numpy_tensor_ms, 1)} ms</b>
                <span>推理总耗时</span><b>{formatNumber(inferenceTimings.execute_total_ms, 1)} ms</b>
                <span>GPU 等待</span><b>{formatNumber(trtTimings.stream_sync_ms, 1)} ms</b>
                <span>解码/NMS</span><b>{formatNumber(trtTimings.decode_ms, 1)} ms</b>
                <span>解码布局</span><b>{readString(decodeDebug.selected_layout, "-")}</b>
                <span>最大分数</span><b>{formatNumber(decodeDebug.max_score, 3)}</b>
                <span>阈值前候选</span><b>{String(readNumber(decodeDebug.raw_candidates, 0))}</b>
                <span>过阈值候选</span><b>{String(readNumber(decodeDebug.threshold_candidates, 0))}</b>
                <span>NMS 后候选</span><b>{String(readNumber(decodeDebug.nms_detections, 0))}</b>
                <span>当前类别</span><b>{readString(target.class_name, "-")}</b>
                <span>最高置信度</span><b>{target.score ? Number(target.score).toFixed(2) : "-"}</b>
                <span>目标框中心</span><b>{`${formatNumber(target.box_cx ?? target.cx, 1)}, ${formatNumber(target.box_cy ?? target.cy, 1)}`}</b>
                <span>瞄准点</span><b>{`${formatNumber(target.aim_x, 1)}, ${formatNumber(target.aim_y, 1)}`}</b>
                <span>候选框数量</span><b>{detections}</b>
                <span>选择状态</span><b>{readString(control.selector_state, "-")}</b>
                <span>选择原因</span><b>{readString(control.selection_reason, "-")}</b>
              </div>
              {inputDensityWarning ? (
                <div className="inference-density-warning">
                  ROI 正在被压缩到模型输入，远距离小目标可能丢失。建议让 ROI 尺寸接近模型输入，或切换到更大输入尺寸的模型。
                </div>
              ) : null}
            </div>
          </div>
        </section>

        <section className={activePage === "params" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="主算法" value="实验角度 PID" small="control" />
            <Metric title="触发方式" value={triggerModeLabel(triggerMode)} small="trigger" />
            <Metric title="Kp X" value={experimentalAngleKpX.toFixed(2)} small="axis x" />
            <Metric title="Kp Y" value={experimentalAngleKpY.toFixed(2)} small="axis y" />
            <Metric title="角度/FOV" value={experimentalAngleFovX.toFixed(0)} small="deg" />
            <Metric title="限幅" value={experimentalAngleMaxStep.toFixed(0)} small="counts" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">鼠标移动算法</h2>
              <label>算法模式</label>
              <select
                value={controlStrategy}
                onChange={(event) => void updateConfigField("control", "strategy", event.target.value)}
              >
                <option value="experimental_angle_pid">实验角度 PID</option>
              </select>
              <label>触发方式</label>
              <select
                value={triggerMode}
                onChange={(event) => void updateConfigField("control", "trigger_mode", event.target.value)}
              >
                <option value="hardware">kmNet 硬件按键触发</option>
                <option value="always">总是启用</option>
              </select>
              <p className="console-field-hint">
                当前运行只允许 kmNet。选择“总是启用”只跳过按键门控，不跳过目标、标定、过期、COOLDOWN 和设备错误保护。
              </p>
              <label>目标保持</label>
              <p className="console-field-hint">
                检测短暂丢失时继续沿用最近目标；超过容忍帧数后释放目标，避免误跟踪。
              </p>
              <NumberControl label="丢失容忍帧" value={targetLostGraceFrames} min={0} max={30} step={1} onCommit={(value) => updateConfigField("control", "target_lost_grace_frames", Math.round(value))} />
              <details className="model-debug-details" open={experimentalAngleAdvanced}>
                <summary>生产候选过滤与目标切换</summary>
                <NumberControl label="最低控制置信度" value={controlMinConfidence} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "min_confidence", value)} />
                <NumberControl label="Selection FOV 比例" value={selectionFovRatio} min={0.01} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "fov_ratio", value)} />
                <NumberControl label="候选框最大宽高比" value={candidateRatioMaxAspect} min={1} max={20} step={0.1} onCommit={(value) => updateConfigField("control", "candidate_ratio_max_aspect", value)} />
                <NumberControl label="质量权重：置信度" value={candidateQualityConfidenceWeight} min={0} max={2} step={0.05} onCommit={(value) => updateConfigField("control", "candidate_quality_confidence_weight", value)} />
                <NumberControl label="质量权重：面积" value={candidateQualityAreaWeight} min={0} max={2} step={0.05} onCommit={(value) => updateConfigField("control", "candidate_quality_area_weight", value)} />
                <NumberControl label="类别优先容忍" value={classPriorityQualityMargin} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "class_priority_quality_margin", value)} />
                <NumberControl label="Track 确认帧数" value={trackerConfirmFrames} min={1} max={10} step={1} onCommit={(value) => updateConfigField("control", "tracker_confirm_frames", Math.round(value))} />
                <NumberControl label="Track 匹配距离 px" value={trackerMatchingDistancePx} min={1} max={1000} step={1} onCommit={(value) => updateConfigField("control", "tracker_matching_distance_px", value)} />
                <NumberControl label="身份歧义边界" value={trackerAmbiguityMargin} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "tracker_ambiguity_margin", value)} />
                <NumberControl label="Track 漏检超时 ms" value={trackerMissingTimeoutMs} min={1} max={1000} step={1} onCommit={(value) => updateConfigField("control", "tracker_missing_timeout_ms", value)} />
                <NumberControl label="Track 删除超时 ms" value={trackerDeleteTimeoutMs} min={1} max={2000} step={1} onCommit={(value) => updateConfigField("control", "tracker_delete_timeout_ms", value)} />
                <NumberControl label="匹配代价上限" value={trackerMatchThreshold} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "tracker_match_threshold", value)} />
                <NumberControl label="马氏门控" value={trackerMahalanobisGate} min={0.001} max={100} step={0.1} onCommit={(value) => updateConfigField("control", "tracker_mahalanobis_gate", value)} />
                <NumberControl label="切换优势阈值" value={targetSwitchPreferenceAdvantage} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "target_switch_min_preference_advantage", value)} />
                <NumberControl label="切换连续性阈值" value={targetSwitchContinuityScore} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "target_switch_min_continuity_score", value)} />
                <NumberControl label="切换确认帧数" value={targetSwitchConfirmFrames} min={1} max={10} step={1} onCommit={(value) => updateConfigField("control", "target_switch_confirm_frames", Math.round(value))} />
              </details>
              <details className="model-debug-details" open={experimentalAngleDeveloper}>
                <summary>生产 Kalman 高级参数</summary>
                <ModuleSwitch label="启用 Kalman 估计" detail="用于 Track 预测、漏检续控和身份稳定" enabled={kalmanEnabled} onToggle={(enabled) => updateConfigField("control", "kalman_enabled", enabled)} />
                <NumberControl label="加速度噪声" value={kalmanAccelerationNoise} min={0.001} max={10000} step={10} onCommit={(value) => updateConfigField("control", "kalman_acceleration_noise", value)} />
                <NumberControl label="X 测量噪声" value={kalmanMeasurementNoiseX} min={0.001} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_measurement_noise_x", value)} />
                <NumberControl label="Y 测量噪声" value={kalmanMeasurementNoiseY} min={0.001} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_measurement_noise_y", value)} />
                <NumberControl label="最大漏检预测 ms" value={kalmanMaxPredictMissingMs} min={1} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_max_predict_missing_ms", value)} />
                <NumberControl label="最大预测步数" value={kalmanMaxPredictSteps} min={1} max={30} step={1} onCommit={(value) => updateConfigField("control", "kalman_max_predict_steps", Math.round(value))} />
                <NumberControl label="单步最大 dt ms" value={kalmanMaxPredictDtMs} min={1} max={200} step={1} onCommit={(value) => updateConfigField("control", "kalman_max_predict_dt_ms", value)} />
                <NumberControl label="最大位置 sigma px" value={kalmanMaxPositionSigmaPx} min={1} max={500} step={1} onCommit={(value) => updateConfigField("control", "kalman_max_position_sigma_px", value)} />
                <NumberControl label="最大协方差迹" value={kalmanMaxCovarianceTrace} min={1} max={100000} step={100} onCommit={(value) => updateConfigField("control", "kalman_max_covariance_trace", value)} />
                <NumberControl label="NIS 阈值" value={kalmanNisThreshold} min={0.001} max={100} step={0.1} onCommit={(value) => updateConfigField("control", "kalman_nis_threshold", value)} />
                <NumberControl label="NIS 硬拒收" value={kalmanNisHardReject} min={0.001} max={200} step={0.1} onCommit={(value) => updateConfigField("control", "kalman_nis_hard_reject", value)} />
                <NumberControl label="最低身份可信度" value={kalmanMinIdentityConfidence} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "kalman_min_identity_confidence", value)} />
                <NumberControl label="最低预测可信度" value={kalmanMinPredictionConfidence} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "kalman_min_prediction_confidence", value)} />
                <NumberControl label="预测衰减 tau ms" value={kalmanPredictionDecayTauMs} min={1} max={1000} step={1} onCommit={(value) => updateConfigField("control", "kalman_prediction_decay_tau_ms", value)} />
              </details>

              <>
                  <label>实验角度 PID 参数</label>
                  <p className="console-field-hint">
                    bbox 中心给 ROI 像素误差；整张采集画面计算焦距；PID 控制角度；最后换算 kmNet counts。Y 方向只交给执行层统一翻转。
                  </p>
                  <button
                    className="console-button"
                    disabled={busy === "control.experimental_angle_reset"}
                    onClick={() => void resetExperimentalAngleDefaults()}
                    type="button"
                  >
                    重置实验角度 PID 默认值
                  </button>
                  <label>配置级别</label>
                  <select value={experimentalAngleConfigLevel} onChange={(event) => void updateConfigField("control", "experimental_angle_config_level", event.target.value)}>
                    <option value="basic">Level 1 普通用户</option>
                    <option value="advanced">Level 2 高级用户</option>
                    <option value="developer">Level 3 开发者</option>
                  </select>
                  <label>Level 1 普通参数</label>
                  <NumberControl label="跟枪速度 Speed" value={experimentalAngleSpeed} min={0} max={3} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_speed", value)} />
                  <NumberControl label="平滑程度 Smooth" value={experimentalAngleSmoothFactor} min={0} max={0.95} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_smooth_factor", value)} />
                  <NumberControl label="预测强度 Prediction ms" value={experimentalAnglePredictionLeadMs} min={0} max={120} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_prediction_lead_ms", value)} />
                  <NumberControl label="最大速度 Max Speed" value={experimentalAngleMaxStep} min={1} max={500} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_max_step_counts", value)} />
                  <NumberControl label="死区 Dead Zone" value={experimentalAngleDeadzonePx} min={0} max={100} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_deadzone_px", value)} />
                  {experimentalAngleAdvanced ? (
                    <>
                      <label>Level 2 高级参数</label>
                  <NumberControl label="Kp X" value={experimentalAngleKpX} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_kp_x", value)} />
                  <NumberControl label="Kp Y" value={experimentalAngleKpY} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_kp_y", value)} />
                  <NumberControl label="Ki" value={experimentalAngleKi} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_ki", value)} />
                  <NumberControl label="Kd" value={experimentalAngleKd} min={-1} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_kd", value)} />
                  <NumberControl label="积分限幅 rad·s" value={experimentalAngleIntegralLimit} min={0} max={2} step={0.001} onCommit={(value) => updateConfigField("control", "experimental_angle_integral_limit", value)} />
                      <NumberControl label="D 项滤波" value={experimentalAngleDerivativeFilter} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_derivative_filter", value)} />
                  <NumberControl label="水平 FOVX" value={experimentalAngleFovX} min={1} max={179} step={1} onCommit={(value) => updateConfigField("calibration", "fov_x_deg", value)} />
                  <NumberControl label="X 每圈 counts" value={experimentalAngleC360X} min={1} max={50000} step={10} onCommit={(value) => updateConfigField("calibration", "counts_per_360_x", value)} />
                  <NumberControl label="Y 每圈 counts" value={experimentalAngleC360Y} min={1} max={50000} step={10} onCommit={(value) => updateConfigField("calibration", "counts_per_360_y", value)} />
                  <NumberControl label="控制频率 Hz" value={experimentalAngleControlHz} min={1} max={240} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_control_hz", value)} />
                  <NumberControl label="X 轴方向" value={experimentalAngleSignX < 0 ? -1 : 1} min={-1} max={1} step={2} onCommit={(value) => updateConfigField("calibration", "axis_sign_x", value < 0 ? -1 : 1)} />
                  <NumberControl label="Y 轴方向" value={experimentalAngleSignY < 0 ? -1 : 1} min={-1} max={1} step={2} onCommit={(value) => updateConfigField("calibration", "axis_sign_y", value < 0 ? -1 : 1)} />
                    </>
                  ) : null}
                  {experimentalAngleDeveloper ? (
                    <>
                      <label>Level 3 开发者参数</label>
                      <p className="console-field-hint">开发者层显示完整 runtime 状态：Error(px/rad)、Velocity(px/s)、Prediction(px)、PID(P/I/D)、Output(counts)、Latency、dt 可在运行反馈里查看。</p>
                      <label>目标过滤</label>
                      <ModuleSwitch label="启用目标过滤" detail="过滤进入 Kalman / 匈牙利的候选框" enabled={experimentalAngleTargetFilterEnabled} onToggle={(enabled) => updateConfigField("control", "experimental_angle_target_filter_enabled", enabled)} />
                      <NumberControl label="过滤置信度" value={experimentalAngleTargetFilterMinScore} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_target_filter_min_score", value)} />
                      <NumberControl label="过滤 FOV 比例" value={experimentalAngleTargetFilterFovRatio} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_target_filter_fov_ratio", value)} />
                      <ModuleSwitch label="只保留同类目标" detail="只让当前目标同类别 detection 进入 tracker" enabled={experimentalAngleTargetFilterSameClass} onToggle={(enabled) => updateConfigField("control", "experimental_angle_target_filter_same_class", enabled)} />
                      <label>Kalman 预测 / 匈牙利匹配</label>
                      <ModuleSwitch label="卡尔曼滤波" detail="稳定并预测检测框中心" enabled={experimentalAngleKalmanEnabled} onToggle={(enabled) => updateConfigField("control", "experimental_angle_kalman_enabled", enabled)} />
                      <NumberControl label="过程噪声" value={experimentalAngleKalmanProcessNoise} min={0.001} max={200} step={0.1} onCommit={(value) => updateConfigField("control", "experimental_angle_kalman_process_noise", value)} />
                      <NumberControl label="测量噪声" value={experimentalAngleKalmanMeasurementNoise} min={0.001} max={500} step={0.1} onCommit={(value) => updateConfigField("control", "experimental_angle_kalman_measurement_noise", value)} />
                      <ModuleSwitch label="匈牙利匹配" detail="多目标时保持 track 与 detection 对应关系" enabled={experimentalAngleHungarianEnabled} onToggle={(enabled) => updateConfigField("control", "experimental_angle_hungarian_enabled", enabled)} />
                      <NumberControl label="匹配距离 px" value={experimentalAngleMatchingDistance} min={1} max={1000} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_matching_distance_px", value)} />
                      <NumberControl label="最大外推帧" value={experimentalAngleMaxExtrapolateFrames} min={0} max={10} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_max_extrapolate_frames", Math.round(value))} />
                      <NumberControl label="外推置信衰减" value={experimentalAngleExtrapolateConfidenceDecay} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_extrapolate_confidence_decay", value)} />
                      <label>磁性吸附</label>
                      <ModuleSwitch label="启用磁性吸附" detail="按目标距离补充 counts，方便单独排查磁性手感" enabled={experimentalAngleMagnetEnabled} onToggle={(enabled) => updateConfigField("control", "experimental_angle_magnet_enabled", enabled)} />
                      <NumberControl label="磁性半径 px" value={experimentalAngleMagnetRadiusPx} min={1} max={1000} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_magnet_radius_px", value)} />
                      <NumberControl label="磁性强度" value={experimentalAngleMagnetStrength} min={0} max={3} step={0.01} onCommit={(value) => updateConfigField("control", "experimental_angle_magnet_strength", value)} />
                      <NumberControl label="磁性曲线" value={experimentalAngleMagnetCurve} min={0.1} max={5} step={0.1} onCommit={(value) => updateConfigField("control", "experimental_angle_magnet_curve", value)} />
                      <NumberControl label="磁性死区 px" value={experimentalAngleMagnetDeadzonePx} min={0} max={100} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_magnet_deadzone_px", value)} />
                      <NumberControl label="磁性限幅 counts" value={experimentalAngleMagnetMaxCounts} min={0} max={200} step={1} onCommit={(value) => updateConfigField("control", "experimental_angle_magnet_max_counts", value)} />
                    </>
                  ) : null}
              </>
            </div>
            <div className="console-card">
              <h2 className="console-title">控制量反馈</h2>
              <div className="console-kv control-feedback-kv">
                <span>当前目标</span><b>{readString(target.class_name, "-")}</b>
                <span>算法模式</span><b>实验角度 PID</b>
                <span>目标序号</span><b>{formatNumber(control.target_detection_index ?? target.target_detection_index, 0)}</b>
                <span>选择状态</span><b>{readString(control.selector_state, "-")}</b>
                <span>选择原因</span><b>{readString(control.selection_reason, "-")}</b>
                <span>Track ID</span><b>{formatNumber(trackDiagnostics.selected_track_id, 0)}</b>
                <span>Track 状态</span><b>{readString(trackDiagnostics.selected_track_state, readString(trackDiagnostics.tracker_state, "-"))}</b>
                <span>连续性</span><b>{formatPercent(trackDiagnostics.selected_continuity_score, 1)}</b>
                <span>身份可信度</span><b>{formatPercent(trackDiagnostics.selected_identity_confidence, 1)}</b>
                <span>missing</span><b>{formatNumber(trackDiagnostics.selected_missing_ms, 1)} ms</b>
                <span>NIS/马氏</span><b>{formatNumber(trackDiagnostics.selected_mahalanobis ?? selectedTrackEstimate.nis, 2)}</b>
                <span>sigma</span><b>{formatNumber(selectedTrackEstimate.position_sigma_px, 1)} px</b>
                <span>cov trace</span><b>{formatNumber(selectedTrackEstimate.cov_trace, 1)}</b>
                <span>预测可信度</span><b>{formatPercent(selectedTrackEstimate.prediction_confidence, 1)}</b>
                <span>切换状态</span><b>{readString(trackDiagnostics.switch_state, "-")}</b>
                <span>aim dx</span><b>{formatNumber(control.aim_error_x, 1)}</b>
                <span>aim dy</span><b>{formatNumber(control.aim_error_y, 1)}</b>
                <span>raw dx</span><b>{formatNumber(control.raw_error_x, 1)}</b>
                <span>raw dy</span><b>{formatNumber(control.raw_error_y, 1)}</b>
                <span>aim y ratio</span><b>{formatNumber(control.aim_y_ratio ?? control.aim_ratio, 0)}%</b>
                <span>控制帧龄</span><b>{formatNumber(control.frame_age_ms, 1)} ms</b>
                <span>aim point</span><b>{`${formatNumber(control.aim_x, 1)}, ${formatNumber(control.aim_y, 1)}`}</b>
                <span>预测点</span><b>{`${formatNumber(controlPipeline.predicted_x, 1)}, ${formatNumber(controlPipeline.predicted_y, 1)}`}</b>
                <span>预测提前</span><b>{formatNumber(controlPipeline.prediction_lead_ms, 1)} ms</b>
                <span>预测速度</span><b>{`${formatNumber(controlPipeline.prediction_velocity_x_px_s, 0)} / ${formatNumber(controlPipeline.prediction_velocity_y_px_s, 0)} px/s`}</b>
                <span>预测补偿</span><b>{`${formatNumber(controlPipeline.prediction_lead_x_px, 1)} / ${formatNumber(controlPipeline.prediction_lead_y_px, 1)} px`}</b>
                <span>Y 坐标约定</span><b>{readString(controlPipeline.coordinate_y, "-")}</b>
                <span>FOV / c360</span><b>{`${formatNumber(controlPipeline.fov_deg, 0)} / ${formatNumber(controlPipeline.c360, 0)}`}</b>
                <span>FOV counts X</span><b>{formatNumber(controlPipeline.fov_counts_x, 1)}</b>
                <span>FOV counts Y</span><b>{formatNumber(controlPipeline.fov_counts_y, 1)}</b>
                <span>动作门控</span><b>{`${formatNumber(controlPipeline.motion_coef_x, 2)} / ${formatNumber(controlPipeline.motion_coef_y, 2)}`}</b>
                <span>预测开关</span><b>{controlPipeline.prediction_enabled === false ? "关闭" : "开启"}</b>
                <span>D 开关</span><b>{controlPipeline.derivative_enabled === false ? "关闭" : "开启"}</b>
                <span>PID P</span><b>{`${formatNumber(controlPipeline.p_x, 1)} / ${formatNumber(controlPipeline.p_y, 1)}`}</b>
                <span>PID I</span><b>{`${formatNumber(controlPipeline.i_x, 1)} / ${formatNumber(controlPipeline.i_y, 1)}`}</b>
                <span>PID D</span><b>{`${formatNumber(controlPipeline.d_x, 1)} / ${formatNumber(controlPipeline.d_y, 1)}`}</b>
                <span>D 原始值</span><b>{`${formatNumber(controlPipeline.raw_d_x, 1)} / ${formatNumber(controlPipeline.raw_d_y, 1)}`}</b>
                <span>D 限幅</span><b>{controlPipeline.d_limited_x || controlPipeline.d_limited_y ? "已触发" : "未触发"}</b>
                <span>策略 dx</span><b>{formatNumber(control.dx, 1)}</b>
                <span>策略 dy</span><b>{formatNumber(control.dy, 1)}</b>
                <span>FOV 内候选</span><b>{formatNumber(control.inside_fov, 0)}</b>
                <span>FOV 半径</span><b>{formatNumber(selectorDebug.fov_radius, 1)}</b>
                <span>过滤后候选</span><b>{formatNumber(selectorDebug.filtered_candidates, 0)}</b>
                <span>目标距离</span><b>{formatNumber(control.distance_px, 1)}</b>
                <span>触发方式</span><b>{triggerModeLabel(readString(control.trigger_mode, triggerMode))}</b>
                <span>kmNet 按键</span><b>{`L:${triggerLeft ? "1" : "0"} R:${triggerRight ? "1" : "0"}`}</b>
                <span>输出状态</span><b>{control.will_emit === true ? "允许输出" : "等待触发"}</b>
                <span>触发要求</span><b>{readString(control.trigger_requirement, control.trigger_required === true ? "需要按键触发" : "无需触发")}</b>
                <span>触发信息</span><b>{readString(control.trigger_reason, "-") || "-"}</b>
                <span>always 状态</span><b>{triggerAlwaysActive ? "已启用" : "-"}</b>
                <span>执行器</span><b>{readString(execution.executor_id, readString(executorStatus.selected, "-"))}</b>
                <span>发送结果</span><b>{execution.sent === true ? "已发送" : execution.sent === false ? "未发送" : "-"}</b>
                <span>移动 API</span><b>{readString(execution.move_kind, moveKind)}</b>
                <span>执行阶段</span><b>{readString(executionMeta.stage, "-")}</b>
                <span>Driver API</span><b>{readString(executionMeta.api_name, "-")}</b>
                <span>Driver rc</span><b>{String(executionMeta.driver_rc ?? "-")}</b>
                <span>最终 dx</span><b>{formatNumber(execution.output_dx ?? executionIntent.dx, 1)}</b>
                <span>最终 dy</span><b>{formatNumber(execution.output_dy ?? executionIntent.dy, 1)}</b>
                <span>Driver dx</span><b>{formatNumber(executionMeta.driver_dx, 1)}</b>
                <span>Driver dy</span><b>{formatNumber(executionMeta.driver_dy, 1)}</b>
                <span>kmNet 次数</span><b>{formatNumber(kmnetStatus.move_count, 0)}</b>
                <span>kmNet 最近</span><b>{`${formatNumber(kmnetStatus.last_dx, 0)} / ${formatNumber(kmnetStatus.last_dy, 0)}`}</b>
                <span>限幅</span><b>{(execution.clipped ?? executionIntent.clipped) === true ? "已限幅" : (execution.clipped ?? executionIntent.clipped) === false ? "未限幅" : "-"}</b>
                <span className="wide">执行信息</span><b className="wide">{readString(execution.message, "-")}</b>
              </div>
            </div>
            <div className="console-card">
              <h2 className="console-title">kmNet 控制面板</h2>
              <div className="kmnet-status-grid">
                <div className={kmnetConnected ? "kmnet-status-tile good" : "kmnet-status-tile idle"}>
                  <span>连接</span>
                  <b>{kmnetConnected ? "已连接" : "未连接"}</b>
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
              </div>
              <div className="kmnet-driver-line">
                <span>{compactDriverSource(kmnetStatus.driver_source)}</span>
                <span>{readString(kmnetStatus.driver_python, "-")}</span>
              </div>
              <label>硬件类型</label>
              <select
                value={hardwareKind}
                onChange={(event) => void updateHardwareKind(event.target.value)}
              >
                <option value="kmnet">kmNet</option>
              </select>
              <label>输出执行器</label>
              <select
                value={outputMode}
                onChange={(event) => void updateConfigField("control", "output_mode", event.target.value)}
              >
                <option value="kmnet">kmNet 实发</option>
              </select>
              <TextControl label="kmnetip" value={kmnetHost} onCommit={(value) => updateConfigField("hardware", "host", value)} />
              <NumberControl label="kmnetport" value={kmnetPort} min={0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "port", Math.round(value))} />
              <TextControl label="kmnetuuid" value={kmnetUuid} onCommit={(value) => updateConfigField("hardware", "uuid", value)} />
              <NumberControl label="monitor_port" value={kmnetMonitorPort} min={0} max={65535} step={1} onCommit={(value) => updateConfigField("hardware", "monitor_port", Math.round(value))} />
              <label>移动 API</label>
              <select
                value={moveKind}
                onChange={(event) => void updateConfigField("control", "move_kind", event.target.value)}
              >
                <option value="raw">move：最快直移</option>
                <option value="enc_raw">enc_move：加密直移</option>
                <option value="auto">move_auto：模拟移动</option>
                <option value="enc_auto">enc_move_auto：加密模拟移动</option>
                <option value="bezier">move_beizer：贝塞尔曲线</option>
                <option value="enc_bezier">enc_move_beizer：加密贝塞尔曲线</option>
              </select>
              <NumberControl label="移动耗时 ms" value={moveMs} min={0} max={100} step={1} onCommit={(value) => updateConfigField("control", "move_ms", Math.round(value))} />
              <label>命令调度</label>
              <NumberControl label="步进间隔 ms" value={commandIntervalMs} min={0} max={100} step={1} onCommit={(value) => updateConfigField("control", "command_interval_ms", value)} />
              <NumberControl label="命令 TTL ms" value={schedulerCommandTtlMs} min={1} max={500} step={1} onCommit={(value) => updateConfigField("control", "scheduler_command_ttl_ms", value)} />
              <NumberControl label="预测命令 TTL ms" value={schedulerPredictedCommandTtlMs} min={1} max={500} step={1} onCommit={(value) => updateConfigField("control", "scheduler_predicted_command_ttl_ms", value)} />
              <NumberControl label="设备错误冷却 ms" value={schedulerDeviceErrorCooldownMs} min={0} max={2000} step={10} onCommit={(value) => updateConfigField("control", "scheduler_device_error_cooldown_ms", value)} />
              <ModuleSwitch label="新帧取消旧命令" detail="新检测帧到达时丢弃未发送的旧命令尾部" enabled={schedulerCancelOnNewFrame} onToggle={(enabled) => updateConfigField("control", "scheduler_cancel_on_new_frame", enabled)} />
              <ModuleSwitch label="方向反转取消" detail="方向变化时取消旧方向 pending，避免追旧世界" enabled={schedulerCancelOnDirectionChange} onToggle={(enabled) => updateConfigField("control", "scheduler_cancel_on_direction_change", enabled)} />
              <ModuleSwitch label="目标切换取消" detail="Track 切换时清空旧目标 pending" enabled={schedulerCancelOnTrackChange} onToggle={(enabled) => updateConfigField("control", "scheduler_cancel_on_track_change", enabled)} />
              <div className="console-action-row">
                <button
                  className={kmnetConnected ? "console-button danger" : "console-button primary"}
                  disabled={busy === "kmnet.toggle"}
                  onClick={() => void toggleHardwareConnection()}
                  type="button"
                >
                  {kmnetConnected ? "断开 kmNet" : "连接 kmNet"}
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
                <div className="kmnet-error">{readString(kmnetStatus.last_error, "")}</div>
              ) : null}
            </div>
          </div>
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
            <KvCard title="推理统计" rows={[
              ["完成帧", String(statistics?.inference_counter ?? 0)],
              ["TensorMeta FPS", formatNumber(statistics?.tensor_meta_fps, 1)],
              ["Postprocess FPS", formatNumber(statistics?.postprocess_fps, 1)],
              ["Batch 消费 FPS", formatNumber(statistics?.detection_batch_fps, 1)],
              ["控制观察 FPS", formatNumber(statistics?.control_observation_fps, 1)],
              ["累计批次", formatNumber(statistics?.published_batches, 0)],
              ["窗口批次", formatNumber(statistics?.window_published_batches, 0)],
              ["窗口 TensorMeta", formatNumber(statistics?.window_tensor_meta_frames, 0)],
              ["窗口后处理", formatNumber(statistics?.window_postprocess_frames, 0)],
              ["时间戳", shortTimestampSource(readString(statistics?.timestamp_source, "-"))],
              ["最后帧龄", formatNumber(statistics?.last_frame_age_ms, 1)],
              ["ROI", formatNumber(statistics?.stage_roi_ms, 1)],
              hasCaptureToTensorMetaMs
                ? ["Tensor Meta", formatNumber(captureToTensorMetaMs, 1)]
                : ["推理总耗时", formatNumber(statistics?.stage_engine_ms, 1)],
              hasCaptureToTensorMetaMs
                ? ["延迟来源", "采集到tensor"]
                : ["TRT执行", formatNumber(statistics?.stage_engine_execute_ms, 1)],
              ["解码/NMS", formatNumber(statistics?.stage_decode_ms, 1)],
              hasCaptureToTensorMetaMs
                ? ["交接等待", formatNumber(statistics?.stage_handoff_ms, 1)]
                : ["映射后处理", formatNumber(statistics?.stage_postprocess_ms, 1)],
              ["控制", formatNumber(statistics?.stage_control_ms, 1)]
            ]} />
            <KvCard title="系统状态" rows={[["CPU", "待机"], ["GPU", "待机"], ["温度", "-"]]} />
          </div>
          <div className="console-card">
            <h2 className="console-title">性能占比</h2>
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
              <h2 className="console-title">延迟链路</h2>
              <div className="console-timeline">
                <Event label="Capture" value={formatNumber(capture?.capture_wait_ms, 2)} width={30} />
                <Event label="Queue" value={formatNumber(statistics?.queue_latency, 1)} width={18} />
                <Event label="ROI" value={formatNumber(statistics?.stage_roi_ms, 1)} width={18} />
                {hasCaptureToTensorMetaMs ? (
                  <Event label="Tensor Meta" value={formatNumber(captureToTensorMetaMs, 1)} width={56} />
                ) : (
                  <>
                    <Event label="推理总耗时" value={formatNumber(statistics?.stage_engine_ms, 1)} width={56} />
                    <Event label="TRT执行" value={formatNumber(statistics?.stage_engine_execute_ms, 1)} width={18} />
                  </>
                )}
                <Event label="解码/NMS" value={formatNumber(statistics?.stage_decode_ms, 1)} width={34} />
                {hasCaptureToTensorMetaMs ? (
                  <Event label="交接等待" value={formatNumber(statistics?.stage_handoff_ms, 1)} width={20} />
                ) : (
                  <Event label="映射后处理" value={formatNumber(statistics?.stage_postprocess_ms, 1)} width={20} />
                )}
                <Event label="Control" value={formatNumber(statistics?.stage_control_ms, 1)} width={14} />
              </div>
            </div>
            <KvCard title="采集诊断" rows={[["状态判断", capture?.available ? "采集中" : "等待数据"], ["队列积压", String(readNumber(asRecord(pipeline.queue).size, 0))], ["建议", capture?.available ? "观察队列等待和帧间隔" : "启动后分析"]]} />
          </div>
        </section>
      </main>
    </section>
  );
}

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  onCommit
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  return (
    <>
      <label>{label}</label>
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

function Metric({ title, value, small }: { title: string; value: string; small: string }) {
  return <div className="console-metric">{title}<br />{value}<small>{small}</small></div>;
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

function PreviewFrame({
  enabled,
  imageEnabled = true,
  runtime,
  roiSize
}: {
  enabled: boolean;
  imageEnabled?: boolean;
  runtime: RuntimeState | null;
  roiSize: number;
}) {
  const configVersion = typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const vision = asRecord(runtime?.vision);
  const inferenceTrace = asRecord(vision.inference);
  const previewWidth = readNumber(inferenceTrace.input_width, roiSize);
  const previewHeight = readNumber(inferenceTrace.input_height, roiSize);
  const displaySize = Math.max(previewWidth, previewHeight, roiSize);
  const detections = readPreviewDetections(vision.detection_items);
  const target = asRecord(vision.target);
  const targetDetectionIndex = readNullableNumber(target.target_detection_index);
  const targetCx = readNullableNumber(target.cx);
  const targetCy = readNullableNumber(target.cy);
  const targetAimX = readNullableNumber(target.aim_x) ?? targetCx;
  const targetAimY = readNullableNumber(target.aim_y) ?? targetCy;
  const showImage = enabled && imageEnabled && runtime?.capture?.available;
  const showOverlay = enabled && detections.length > 0;
  const selectedDetection = detections.find((item) => (
    targetDetectionIndex !== null
      ? item.index === targetDetectionIndex
      : targetCx !== null && targetCy !== null && Math.abs(item.cx - targetCx) <= 2 && Math.abs(item.cy - targetCy) <= 2
  ));
  const selectedIndex = selectedDetection?.index ?? null;
  const centerX = previewWidth / 2;
  const centerY = previewHeight / 2;
  return (
    <div className={showOverlay ? "console-preview has-overlay" : "console-preview"} style={{ "--roi-size": `${displaySize}px` } as CSSProperties}>
      <div className="console-preview-frame">
        {showImage ? <img alt="实时画面 / ROI" src={streamUrl(configVersion, configVersion)} /> : null}
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
                    key={`line-${item.index}-${item.x}-${item.y}`}
                    x1={centerX}
                    y1={centerY}
                    x2={clampNumber(endX, 0, previewWidth)}
                    y2={clampNumber(endY, 0, previewHeight)}
                  />
                );
              })}
            </svg>
            <span className="console-roi-center" />
            {detections.map((item) => {
              const selected = item.index === selectedIndex;
              return (
                <span
                  className={selected ? "console-detection-box primary" : "console-detection-box secondary"}
                  key={`box-${item.index}-${item.x}-${item.y}`}
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

function KvCard({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="console-card">
      <h2 className="console-title">{title}</h2>
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
