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
  updateLocalTrigger,
  updateRuntimeConfig
} from "../../api";
import { getErrorMessage } from "../shared/format";

type ConsolePage = "capture" | "infer" | "params" | "stats" | "latency";

type StudioConsoleViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
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

const ROI_SIZE_CHOICES = [256, 320, 480, 640];
const KMNET_RECOMMENDED = {
  host: "192.168.2.188",
  port: 8888,
  uuid: "12345678",
  monitor_port: 5001
};

const TRIGGER_BINDING_OPTIONS = [
  { value: "", label: "未设置" },
  { value: "MouseLeft", label: "鼠标左键" },
  { value: "MouseRight", label: "鼠标右键" },
  { value: "MouseMiddle", label: "鼠标中键" },
  { value: "KeySpace", label: "空格键" },
  { value: "KeyShiftLeft", label: "左 Shift" },
  { value: "KeyControlLeft", label: "左 Ctrl" },
  { value: "KeyAltLeft", label: "左 Alt" }
];

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

function readNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
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

function readStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function readTraceStages(value: unknown): Record<string, unknown>[] {
  const stages = asRecord(value).stages;
  return Array.isArray(stages) ? stages.map(asRecord) : [];
}

function mouseBindingName(button: number): string {
  if (button === 0) {
    return "MouseLeft";
  }
  if (button === 1) {
    return "MouseMiddle";
  }
  if (button === 2) {
    return "MouseRight";
  }
  return `Mouse${button}`;
}

function keyBindingName(code: string): string {
  return code.startsWith("Key") ? code : `Key${code}`;
}

function triggerModeLabel(value: string): string {
  if (value === "always") {
    return "调试直出";
  }
  if (value === "telemetry") {
    return "本地或硬件按键触发";
  }
  return "绑定按键触发";
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

function cloneRuntimeConfig(runtime: RuntimeState | null): RuntimeConfig | null {
  if (!runtime?.config) {
    return null;
  }
  const next = structuredClone(runtime.config) as RuntimeConfig;
  delete next.version;
  return next;
}

export function StudioConsoleView({
  health,
  runtime,
  projects,
  errors,
  lastUpdated,
  onRefresh
}: StudioConsoleViewProps) {
  const [activePage, setActivePage] = useState<ConsolePage>("capture");
  const [device, setDevice] = useState(
    readString(nestedRecord(runtime?.config, "capture").device, runtime?.capture?.device ?? "/dev/video0")
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
  const [captureBindingSlot, setCaptureBindingSlot] = useState<number | null>(null);
  const [bindingDrafts, setBindingDrafts] = useState<string[] | null>(null);
  const [localTriggerActive, setLocalTriggerActive] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [modelSwitchMessage, setModelSwitchMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pressedBindingsRef = useRef<Set<string>>(new Set());
  const localTriggerActiveRef = useRef(false);

  const capture = runtime?.capture;
  const statistics = runtime?.statistics ?? capture?.statistics;
  const config = runtime?.config;
  const captureConfig = nestedRecord(config, "capture");
  const configuredCaptureDevice = readString(captureConfig.device, "");
  const configuredCapturePixelFormat = readString(captureConfig.pixel_format, "");
  const configuredCaptureWidth = readNumber(captureConfig.width, 0);
  const configuredCaptureHeight = readNumber(captureConfig.height, 0);
  const configuredCaptureFps = readNumber(captureConfig.fps, 0);
  const roiConfig = nestedRecord(config, "roi");
  const inferenceConfig = nestedRecord(config, "inference");
  const controlConfig = nestedRecord(config, "control");
  const hardwareConfig = nestedRecord(config, "hardware");
  const consumersConfig = nestedRecord(config, "consumers");
  const vision = asRecord(runtime?.vision);
  const execution = asRecord(vision.execution);
  const executionIntent = asRecord(execution.intent);
  const executionMeta = asRecord(execution.metadata);
  const yRateLimiterMeta = asRecord(executionMeta.y_rate_limiter ?? executionMeta);
  const inferenceTrace = asRecord(vision.inference);
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
  const pidKpX = readNumber(controlConfig.pid_kp_x, 0.35);
  const pidKpY = readNumber(controlConfig.pid_kp_y, 0.24);
  const pidKd = readNumber(controlConfig.pid_kd, 0.1);
  const kpXMoveMax = readNumber(controlConfig.kp_x_move_max, 150);
  const kpYMoveMax = readNumber(controlConfig.kp_y_move_max, 30);
  const predictionFactor = readNumber(controlConfig.prediction_factor, 0.1);
  const yRateWindowMs = readNumber(controlConfig.y_rate_window_ms, 10);
  const yRateMaxCounts = readNumber(controlConfig.y_rate_max_counts, 0);
  const aimYRatio = readNumber(controlConfig.aim_ratio, 40);
  const targetLostGraceFrames = readNumber(controlConfig.target_lost_grace_frames, 5);
  const moveKind = readString(controlConfig.move_kind, "bezier");
  const moveMs = readNumber(controlConfig.move_ms, 12);
  const hardwareKind = readString(hardwareConfig.kind, "none");
  const outputMode = readString(controlConfig.output_mode, "");
  const triggerMode = readString(controlConfig.trigger_mode, "hardware");
  const triggerBindings = readStringArray(controlConfig.trigger_bindings).slice(0, 2);
  const triggerBindingKey = triggerBindings.join("\u0000");
  const activeTriggerBindings = useMemo(
    () => (bindingDrafts ?? triggerBindings).slice(0, 2),
    [bindingDrafts, triggerBindingKey]
  );
  const activeTriggerBindingKey = activeTriggerBindings.join("\u0000");
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);
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
  const triggerRaw = asRecord(control.trigger_raw);
  const triggerHardwareRaw = asRecord(triggerRaw.hardware);
  const triggerLocalRaw = asRecord(triggerRaw.local);
  const triggerLeft = triggerRaw.left === true || triggerHardwareRaw.left === true;
  const triggerRight = triggerRaw.right === true || triggerHardwareRaw.right === true;
  const triggerLocalActive = triggerRaw.active === true || triggerLocalRaw.active === true;
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

  const toggleCapture = useCallback(async () => {
    if (capture?.available) {
      await stopCurrentCapture();
      return;
    }
    await applyCapture();
  }, [applyCapture, capture?.available, stopCurrentCapture]);

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

  const updateConfigField = useCallback(
    async (section: string, key: string, value: number | string | boolean | string[]) => {
      const next = cloneRuntimeConfig(runtime);
      if (!next) {
        return;
      }
      setBusy(`${section}.${key}`);
      setLocalError(null);
      const sectionValue = {
        ...asRecord(next[section])
      };
      sectionValue[key] = value;
      next[section] = sectionValue as RuntimeConfig[string];
      try {
        await updateRuntimeConfig(next);
        await onRefresh();
      } catch (err) {
        setLocalError(`配置同步失败：${getErrorMessage(err)}`);
        await onRefresh();
      } finally {
        setBusy(null);
      }
    },
    [onRefresh, runtime]
  );

  const setTriggerBinding = useCallback(
    async (slot: number, binding: string) => {
      const next = [...activeTriggerBindings];
      next[slot] = binding;
      const unique = next.filter(Boolean).filter((item, index, arr) => arr.indexOf(item) === index).slice(0, 2);
      setBindingDrafts(unique);
      await updateConfigField("control", "trigger_bindings", unique);
    },
    [activeTriggerBindings, updateConfigField]
  );

  useEffect(() => {
    if (captureBindingSlot === null) {
      setBindingDrafts(triggerBindings);
    }
  }, [captureBindingSlot, triggerBindingKey]);

  useEffect(() => {
    const allowed = new Set(activeTriggerBindings.map((item) => item.toLowerCase()));
    const commit = (nextActive: boolean, bindings: string[], force = false) => {
      if (!force && localTriggerActiveRef.current === nextActive) {
        return;
      }
      localTriggerActiveRef.current = nextActive;
      setLocalTriggerActive(nextActive);
      void updateLocalTrigger(nextActive, bindings);
    };
    const activeBindings = () =>
      [...pressedBindingsRef.current].filter((item) => allowed.has(item.toLowerCase()));
    const refresh = () => {
      const active = activeBindings();
      commit(active.length > 0, active);
    };
    const isInteractiveControlTarget = (target: EventTarget | null) => {
      const element = target instanceof HTMLElement ? target : null;
      return !!element?.closest("button, input, textarea, select, option, label, [contenteditable='true'], [role='button']");
    };
    const isKeyboardControlTarget = (target: EventTarget | null) => {
      const element = target instanceof HTMLElement ? target : null;
      return !!element?.closest("button, input, textarea, select, [contenteditable='true']");
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (captureBindingSlot !== null) {
        event.preventDefault();
        void setTriggerBinding(captureBindingSlot, keyBindingName(event.code));
        setCaptureBindingSlot(null);
        return;
      }
      if (isKeyboardControlTarget(event.target)) {
        return;
      }
      const binding = keyBindingName(event.code);
      if (!allowed.has(binding.toLowerCase())) {
        return;
      }
      pressedBindingsRef.current.add(binding);
      refresh();
    };
    const onKeyUp = (event: KeyboardEvent) => {
      pressedBindingsRef.current.delete(keyBindingName(event.code));
      refresh();
    };
    const onMouseDown = (event: MouseEvent) => {
      const binding = mouseBindingName(event.button);
      if (captureBindingSlot !== null) {
        event.preventDefault();
        void setTriggerBinding(captureBindingSlot, binding);
        setCaptureBindingSlot(null);
        return;
      }
      if (isInteractiveControlTarget(event.target)) {
        return;
      }
      if (!allowed.has(binding.toLowerCase())) {
        return;
      }
      if (binding === "MouseRight") {
        event.preventDefault();
      }
      pressedBindingsRef.current.add(binding);
      refresh();
    };
    const onMouseUp = (event: MouseEvent) => {
      pressedBindingsRef.current.delete(mouseBindingName(event.button));
      refresh();
    };
    const clear = () => {
      pressedBindingsRef.current.clear();
      commit(false, []);
    };
    const heartbeat = window.setInterval(() => {
      if (!localTriggerActiveRef.current) {
        return;
      }
      const active = activeBindings();
      if (active.length > 0) {
        commit(true, active, true);
      } else {
        commit(false, [], true);
      }
    }, 150);
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("mouseup", onMouseUp, true);
    window.addEventListener("blur", clear);
    window.addEventListener("contextmenu", onMouseDown, true);
    return () => {
      window.clearInterval(heartbeat);
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("mousedown", onMouseDown, true);
      window.removeEventListener("mouseup", onMouseUp, true);
      window.removeEventListener("blur", clear);
      window.removeEventListener("contextmenu", onMouseDown, true);
      clear();
    };
  }, [activeTriggerBindingKey, activeTriggerBindings, captureBindingSlot, setTriggerBinding]);

  const updateHardwareKind = useCallback(
    async (kind: string) => {
      const next = cloneRuntimeConfig(runtime);
      if (!next) {
        return;
      }
      setBusy("hardware.kind");
      setLocalError(null);
      next.hardware = {
        ...asRecord(next.hardware),
        kind,
        ...(kind === "kmnet" ? KMNET_RECOMMENDED : {})
      } as RuntimeConfig[string];
      const control = {
        ...asRecord(next.control)
      };
      if (kind === "kmnet" && !["kmnet", "console"].includes(readString(control.output_mode, ""))) {
        control.output_mode = "kmnet";
      }
      if (kind === "none" && readString(control.output_mode, "") === "kmnet") {
        control.output_mode = "silent";
      }
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
    [onRefresh, runtime]
  );

  const applyKmNetRecommended = useCallback(async () => {
    const next = cloneRuntimeConfig(runtime);
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
  }, [onRefresh, runtime]);

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
    if (!runtime?.config) {
      return;
    }
    const blob = new Blob([JSON.stringify(runtime.config, null, 2)], {
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
            onClick={() => setActivePage(item.id)}
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
            采集：{capture?.available ? "运行中" : "已停止"} · 推理：{runtime?.running ? "运行中" : "已停止"}
          </div>
          <button
            className={capture?.available ? "console-button danger" : "console-button primary"}
            disabled={busy === "capture" || busy === "stop"}
            onClick={() => void toggleCapture()}
            type="button"
          >
            {capture?.available ? "▪ 停止采集" : "▶ 启动采集"}
          </button>
          {!runtime?.running && capture?.available ? (
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
            <Metric title="推理延迟" value={formatNumber(statistics?.stage_engine_ms ?? statistics?.inference_latency, 1)} small="ms" />
            <Metric title="目标数量" value={String(detections)} small="objects" />
            <Metric title="引擎状态" value={readString(runtime?.inference?.loaded, "") ? "已加载" : runtime?.inference?.loaded === true ? "已加载" : "未加载"} small={readString(runtime?.inference?.selected, "engine")} />
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
                  <span>ROI</span><b>{roiSize} · 自动缩放</b>
                  <span>后端</span><b>{readString(runtime?.inference?.selected, "auto")}</b>
                  <span>类别数量</span><b>{String(version?.classes.length ?? 0)}</b>
                </div>
              </details>
            </div>
            <div className="console-card">
              <h2 className="console-title">推理输出</h2>
              <PreviewFrame enabled={activePage === "infer" && previewEnabled} runtime={runtime} roiSize={roiSize} />
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
                <span>raw 检测</span><b>{String(rawDetections)}</b>
                <span>前端检测</span><b>{String(mappedDetections)}</b>
                <span>ROI 输入</span><b>{`${roiInputWidth || "-"}x${roiInputHeight || "-"}`}</b>
                <span>模型输入</span><b>{modelInputWidth && modelInputHeight ? `${modelInputWidth}x${modelInputHeight}` : "-"}</b>
                <span>压缩倍率</span><b>{inputDownscaleFactor ? `${formatNumber(inputDownscaleFactor, 2)}x` : "-"}</b>
                <span>有效像素</span><b>{inputPixelRatio ? formatPercent(inputPixelRatio, 1) : "-"}</b>
                <span>输出形状</span><b>{formatShape(decodeDebug.output_shape ?? inferenceDebug.output_shape)}</b>
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
            <Metric title="主算法" value="Kp / Kd" small="control" />
            <Metric title="触发方式" value={triggerModeLabel(triggerMode)} small="trigger" />
            <Metric title="Kp X" value={pidKpX.toFixed(2)} small="axis x" />
            <Metric title="Kp Y" value={pidKpY.toFixed(2)} small="axis y" />
            <Metric title="预测" value={predictionFactor.toFixed(2)} small="lead x" />
            <Metric title="瞄准高度" value={`${aimYRatio.toFixed(0)}%`} small="aim y" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">鼠标移动算法</h2>
              <label>触发方式</label>
              <select
                value={triggerMode}
                onChange={(event) => void updateConfigField("control", "trigger_mode", event.target.value)}
              >
                <option value="hardware">绑定按键触发</option>
                <option value="telemetry">本地或硬件按键触发</option>
                <option value="always">调试直出</option>
              </select>
              <label>本地按键绑定</label>
              <div className="trigger-binding-grid">
                {[0, 1].map((slot) => (
                  <div
                    className={captureBindingSlot === slot ? "trigger-binding capture" : "trigger-binding"}
                    key={slot}
                  >
                    <span>{slot === 0 ? "绑定一" : "绑定二"}</span>
                    <select
                      value={activeTriggerBindings[slot] ?? ""}
                      onChange={(event) => void setTriggerBinding(slot, event.target.value)}
                    >
                      {TRIGGER_BINDING_OPTIONS.map((item) => (
                        <option key={item.value || "empty"} value={item.value}>{item.label}</option>
                      ))}
                    </select>
                    <button
                      className="trigger-capture-button"
                      onClick={() => setCaptureBindingSlot(slot)}
                      type="button"
                    >
                      {captureBindingSlot === slot ? "等待输入" : "捕获其他"}
                    </button>
                  </div>
                ))}
              </div>
              <div className={localTriggerActive ? "trigger-state active" : "trigger-state"}>
                {localTriggerActive ? "本地触发已按下 · 正在续期" : "本地触发未按下"}
              </div>
              <label>目标保持</label>
              <p className="console-field-hint">
                检测短暂丢失时继续沿用最近目标；超过容忍帧数后释放目标，避免误跟踪。
              </p>
              <NumberControl label="丢失容忍帧" value={targetLostGraceFrames} min={0} max={30} step={1} onCommit={(value) => updateConfigField("control", "target_lost_grace_frames", Math.round(value))} />
              <label>鼠标移动算法</label>
              <NumberControl label="Y 下压间隔 ms" value={yRateWindowMs} min={0} max={1000} step={1} onCommit={(value) => updateConfigField("control", "y_rate_window_ms", value)} />
              <NumberControl label="Y 每次下压 counts" value={yRateMaxCounts} min={0} max={200} step={1} onCommit={(value) => updateConfigField("control", "y_rate_max_counts", value)} />
              <NumberControl label="瞄准高度 aim_y_ratio" value={aimYRatio} min={0} max={100} step={1} onCommit={(value) => updateConfigField("control", "aim_ratio", Math.round(value))} />
              <NumberControl label="kp_x" value={pidKpX} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kp_x", value)} />
              <NumberControl label="kp_y" value={pidKpY} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kp_y", value)} />
              <NumberControl label="kd" value={pidKd} min={-1} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kd", value)} />
              <NumberControl label="预测" value={predictionFactor} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "prediction_factor", value)} />
              <p className="console-field-hint">
                预测只补 X 轴：当前帧与上一帧目标 X 偏移差值 × 预测系数，再叠加到当前偏移。
              </p>
              <NumberControl label="kp_x_move_max" value={kpXMoveMax} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "kp_x_move_max", value)} />
              <NumberControl label="kp_y_move_max" value={kpYMoveMax} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "kp_y_move_max", value)} />
            </div>
            <div className="console-card">
              <h2 className="console-title">控制量反馈</h2>
              <div className="console-kv control-feedback-kv">
                <span>当前目标</span><b>{readString(target.class_name, "-")}</b>
                <span>目标序号</span><b>{formatNumber(control.target_detection_index ?? target.target_detection_index, 0)}</b>
                <span>选择状态</span><b>{readString(control.selector_state, "-")}</b>
                <span>选择原因</span><b>{readString(control.selection_reason, "-")}</b>
                <span>aim dx</span><b>{formatNumber(control.aim_error_x, 1)}</b>
                <span>aim dy</span><b>{formatNumber(control.aim_error_y, 1)}</b>
                <span>raw dx</span><b>{formatNumber(control.raw_error_x, 1)}</b>
                <span>raw dy</span><b>{formatNumber(control.raw_error_y, 1)}</b>
                <span>aim y ratio</span><b>{formatNumber(control.aim_y_ratio ?? control.aim_ratio, 0)}%</b>
                <span>aim point</span><b>{`${formatNumber(control.aim_x, 1)}, ${formatNumber(control.aim_y, 1)}`}</b>
                <span>预测点</span><b>{`${formatNumber(controlPipeline.predicted_x, 1)}, ${formatNumber(controlPipeline.predicted_y, 1)}`}</b>
                <span>Y 坐标约定</span><b>{readString(controlPipeline.coordinate_y, "-")}</b>
                <span>FOV / c360</span><b>{`${formatNumber(controlPipeline.fov_deg, 0)} / ${formatNumber(controlPipeline.c360, 0)}`}</b>
                <span>FOV counts X</span><b>{formatNumber(controlPipeline.fov_counts_x, 1)}</b>
                <span>FOV counts Y</span><b>{formatNumber(controlPipeline.fov_counts_y, 1)}</b>
                <span>动作门控</span><b>{`${formatNumber(controlPipeline.motion_coef_x, 2)} / ${formatNumber(controlPipeline.motion_coef_y, 2)}`}</b>
                <span>PID P</span><b>{`${formatNumber(controlPipeline.p_x, 1)} / ${formatNumber(controlPipeline.p_y, 1)}`}</b>
                <span>PID D</span><b>{`${formatNumber(controlPipeline.d_x, 1)} / ${formatNumber(controlPipeline.d_y, 1)}`}</b>
                <span>策略 dx</span><b>{formatNumber(control.dx, 1)}</b>
                <span>策略 dy</span><b>{formatNumber(control.dy, 1)}</b>
                <span>FOV 内候选</span><b>{formatNumber(control.inside_fov, 0)}</b>
                <span>FOV 半径</span><b>{formatNumber(selectorDebug.fov_radius, 1)}</b>
                <span>过滤后候选</span><b>{formatNumber(selectorDebug.filtered_candidates, 0)}</b>
                <span>目标距离</span><b>{formatNumber(control.distance_px, 1)}</b>
                <span>触发方式</span><b>{triggerModeLabel(readString(control.trigger_mode, triggerMode))}</b>
                <span>本地绑定</span><b>{activeTriggerBindings.length ? activeTriggerBindings.join(" / ") : "-"}</b>
                <span>输出状态</span><b>{control.will_emit === true ? "允许输出" : "等待触发"}</b>
                <span>触发要求</span><b>{readString(control.trigger_requirement, control.trigger_required === true ? "需要按键触发" : "无需触发")}</b>
                <span>触发信息</span><b>{readString(control.trigger_reason, "-") || "-"}</b>
                <span>触发 raw</span><b>{`L:${triggerLeft ? "1" : "0"} R:${triggerRight ? "1" : "0"} Local:${triggerLocalActive ? "1" : "0"}`}</b>
                <span>执行器</span><b>{readString(execution.executor_id, readString(executorStatus.selected, "-"))}</b>
                <span>发送结果</span><b>{execution.sent === true ? "已发送" : execution.sent === false ? "未发送" : "-"}</b>
                <span>移动 API</span><b>{readString(execution.move_kind, moveKind)}</b>
                <span>执行阶段</span><b>{readString(executionMeta.stage, "-")}</b>
                <span>Driver API</span><b>{readString(executionMeta.api_name, "-")}</b>
                <span>Driver rc</span><b>{String(executionMeta.driver_rc ?? "-")}</b>
                <span>最终 dx</span><b>{formatNumber(execution.output_dx ?? executionIntent.dx, 1)}</b>
                <span>最终 dy</span><b>{formatNumber(execution.output_dy ?? executionIntent.dy, 1)}</b>
                <span>Y 下压</span><b>{readNumber(yRateLimiterMeta.max_counts, 0) > 0 ? `${formatNumber(yRateLimiterMeta.drop_counts, 0)} -> ${formatNumber(yRateLimiterMeta.final_dy, 0)}` : "关闭"}</b>
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
                <option value="none">不连接硬件</option>
                <option value="kmnet">kmNet</option>
                <option value="makcu">MAKCU</option>
              </select>
              <label>输出执行器</label>
              <select
                value={outputMode}
                onChange={(event) => void updateConfigField("control", "output_mode", event.target.value)}
              >
                <option value="">跟随默认执行器</option>
                <option value="silent">静默吞没</option>
                <option value="dry_run">调试记录</option>
                <option value="console">命令行输出</option>
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
              ["ROI", formatNumber(statistics?.stage_roi_ms, 1)],
              ["推理总耗时", formatNumber(statistics?.stage_engine_ms, 1)],
              ["TRT执行", formatNumber(statistics?.stage_engine_execute_ms, 1)],
              ["解码/NMS", formatNumber(statistics?.stage_decode_ms, 1)],
              ["映射后处理", formatNumber(statistics?.stage_postprocess_ms, 1)],
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

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const commit = useCallback(() => {
    const next = clampNumber(Number(draft.toFixed(digits)), min, max);
    if (Math.abs(next - value) >= step / 2) {
      void onCommit(next);
    } else {
      setDraft(value);
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
        onChange={(event) => setDraft(clampNumber(Number(event.target.value), min, max))}
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
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
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

function PreviewFrame({
  enabled,
  runtime,
  roiSize
}: {
  enabled: boolean;
  runtime: RuntimeState | null;
  roiSize: number;
}) {
  const configVersion = typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const vision = asRecord(runtime?.vision);
  const inferenceTrace = asRecord(vision.inference);
  const previewWidth = readNumber(inferenceTrace.input_width, roiSize);
  const previewHeight = readNumber(inferenceTrace.input_height, roiSize);
  const displaySize = Math.max(previewWidth, previewHeight, roiSize);
  return (
    <div className="console-preview" style={{ "--roi-size": `${displaySize}px` } as CSSProperties}>
      {enabled && runtime?.capture?.available ? <img alt="实时画面 / ROI" src={streamUrl(configVersion, configVersion)} /> : null}
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
