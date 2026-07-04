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

type DetectionOverlay = {
  classId: number;
  className: string;
  score: number;
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
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

function readDetectionItems(value: unknown): DetectionOverlay[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    const record = asRecord(item);
    const x = readNullableNumber(record.x);
    const y = readNullableNumber(record.y);
    const w = readNullableNumber(record.w);
    const h = readNullableNumber(record.h);
    const score = readNullableNumber(record.score);
    const cx = readNullableNumber(record.cx);
    const cy = readNullableNumber(record.cy);
    const classId = readNullableNumber(record.class_id);
    if (x === null || y === null || w === null || h === null || score === null || cx === null || cy === null || w <= 0 || h <= 0) {
      return [];
    }
    return [{
      classId: classId === null ? -1 : Math.trunc(classId),
      className: readString(record.class_name, String(classId ?? "")),
      score,
      x,
      y,
      w,
      h,
      cx,
      cy
    }];
  });
}

function detectionClassName(detection: DetectionOverlay): string {
  const classBucket = detection.classId >= 0 ? detection.classId % 8 : 7;
  return `console-detection-box detection-class-${classBucket}`;
}

function triggerModeLabel(value: string): string {
  if (value === "always") {
    return "调试直出";
  }
  if (value === "telemetry") {
    return "持续计算，按键发送";
  }
  return "硬件按键触发";
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, value));
}

function detectionStyle(detection: DetectionOverlay, width: number, height: number): CSSProperties {
  return {
    left: `${clampPercent((detection.x / width) * 100)}%`,
    top: `${clampPercent((detection.y / height) * 100)}%`,
    width: `${clampPercent((detection.w / width) * 100)}%`,
    height: `${clampPercent((detection.h / height) * 100)}%`
  };
}

function formatNumber(value: unknown, digits = 1): string {
  const number = readNumber(value, Number.NaN);
  return Number.isFinite(number) ? number.toFixed(digits) : "待机";
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
  const [kmnetTestMessage, setKmnetTestMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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
  const vision = asRecord(runtime?.vision);
  const execution = asRecord(vision.execution);
  const executionIntent = asRecord(execution.intent);
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
  const pidKi = readNumber(controlConfig.pid_ki, 0.1);
  const pidKd = readNumber(controlConfig.pid_kd, 0.1);
  const kpXMoveMax = readNumber(controlConfig.kp_x_move_max, 150);
  const kpYMoveMax = readNumber(controlConfig.kp_y_move_max, 30);
  const predictionFactor = readNumber(controlConfig.prediction_factor, 0.1);
  const aimRatio = readNumber(controlConfig.aim_ratio, 40);
  const targetLockEnabled = controlConfig.target_lock_enabled !== false;
  const targetStickyBias = readNumber(controlConfig.target_sticky_bias, 0.25);
  const targetLostGraceFrames = readNumber(controlConfig.target_lost_grace_frames, 5);
  const hardwareKind = readString(hardwareConfig.kind, "none");
  const outputMode = readString(controlConfig.output_mode, "");
  const triggerMode = readString(controlConfig.trigger_mode, "hardware");
  const kmnetHost = readString(hardwareConfig.host, "192.168.2.188");
  const kmnetPort = readNumber(hardwareConfig.port, 8888);
  const kmnetUuid = readString(hardwareConfig.uuid, "12345678");
  const kmnetMonitorPort = readNumber(hardwareConfig.monitor_port, 5001);
  const kmnetConnected = kmnetStatus.connected === true;
  const kmnetDriverAvailable = kmnetStatus.available === true;
  const activeModelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const artifact = runtime?.active_model?.artifact;
  const version = runtime?.active_model?.version;
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
  const controlPipeline = asRecord(asRecord(vision.control).pipeline);
  const rawDetections = readNumber(inferenceTrace.raw_detections, 0);
  const mappedDetections = readNumber(inferenceTrace.mapped_detections, 0);
  const inferenceRan = inferenceTrace.ran === true;
  const inferenceAvailable = inferenceTrace.available === true;
  const inferenceReason = readString(inferenceTrace.reason, readString(vision.inference_reason, "-"));
  const inferenceDebug = asRecord(inferenceTrace.debug);
  const decodeDebug = asRecord(inferenceDebug.decode);
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
    async (section: string, key: string, value: number | string | boolean) => {
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
      } finally {
        setBusy(null);
      }
    },
    [onRefresh, runtime]
  );

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

  const diagnosticMoveHardware = useCallback(async (dx = kmnetTestDx, dy = kmnetTestDy) => {
    setBusy("kmnet.diagnostic");
    setLocalError(null);
    setKmnetTestMessage("");
    try {
      const result = await diagnosticMoveKmNet(Math.round(dx), Math.round(dy));
      setKmnetTestMessage(
        result.sent === true
          ? `已发送 dx=${Math.round(dx)} dy=${Math.round(dy)}`
          : readString(result.message, "未发送")
      );
      await onRefresh();
    } catch (err) {
      setLocalError(`kmNet 诊断移动失败：${getErrorMessage(err)}`);
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [kmnetTestDx, kmnetTestDy, onRefresh]);

  const diagnosticCircleHardware = useCallback(async () => {
    setBusy("kmnet.circle");
    setLocalError(null);
    setKmnetTestMessage("");
    try {
      const result = await diagnosticCircleKmNet(8, 32, 8);
      const failed = asRecord(result.failed);
      if (result.sent === true) {
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
    try {
      await publishModel(selectedModelProjectId, selectedSwitchArtifact.id);
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

          <div className="console-grid2">
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
            <PreviewCard runtime={runtime} title="实时画面" roiSize={roiSize} />
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
            <Metric title="推理延迟" value={formatNumber(pipeline.e2e_latency_ms ?? statistics?.e2e_latency, 1)} small="ms" />
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
                <span>{version?.input_shape ? `输入 ${version.input_shape}` : "等待模型输入信息"}</span>
                <span>{selectedSwitchArtifact?.kind ? selectedSwitchArtifact.kind.toUpperCase() : "无可用产物"}</span>
              </div>
              <button
                className="console-button primary console-full-button"
                disabled={busy === "model.switch" || selectedModelProjectId === "" || selectedSwitchArtifact === null}
                onClick={switchModel}
                type="button"
              >
                {busy === "model.switch" ? "安全切换中..." : "安全切换模型"}
              </button>
              <label>置信度阈值</label>
              <div className="console-row">
                <input type="range" min="0" max="1" step=".01" value={confidence} onChange={(event) => void updateConfigField("inference", "confidence_threshold", Number(event.target.value))} />
                <input value={confidence.toFixed(2)} readOnly />
              </div>
              <label>NMS 阈值</label>
              <div className="console-row">
                <input type="range" min="0" max="1" step=".01" value={nms} onChange={(event) => void updateConfigField("inference", "nms_threshold", Number(event.target.value))} />
                <input value={nms.toFixed(2)} readOnly />
              </div>
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
                  <span>输入尺寸</span><b>{version?.input_shape ?? "-"}</b>
                  <span>ROI</span><b>{roiSize} · 自动缩放</b>
                  <span>后端</span><b>{readString(runtime?.inference?.selected, "auto")}</b>
                  <span>类别数量</span><b>{String(version?.classes.length ?? 0)}</b>
                </div>
              </details>
            </div>
            <div className="console-card">
              <h2 className="console-title">推理输出</h2>
              <PreviewFrame runtime={runtime} roiSize={roiSize} />
              <div className="console-kv">
                <span>推理状态</span><b>{inferenceRan ? (inferenceAvailable ? "已执行" : "执行失败") : "未执行"}</b>
                <span>推理原因</span><b>{inferenceReason || "-"}</b>
                <span>raw 检测</span><b>{String(rawDetections)}</b>
                <span>前端检测</span><b>{String(mappedDetections)}</b>
                <span>输出形状</span><b>{formatShape(decodeDebug.output_shape ?? inferenceDebug.output_shape)}</b>
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
            </div>
          </div>
        </section>

        <section className={activePage === "params" ? "console-page active" : "console-page"}>
          <div className="console-metrics">
            <Metric title="控制策略" value={readString(controlConfig.strategy, "pid")} small="strategy" />
            <Metric title="触发方式" value={triggerModeLabel(triggerMode)} small="trigger" />
            <Metric title="Kp X" value={pidKpX.toFixed(2)} small="axis x" />
            <Metric title="Kp Y" value={pidKpY.toFixed(2)} small="axis y" />
            <Metric title="预测" value={predictionFactor.toFixed(2)} small="lead" />
            <Metric title="瞄准高度" value={`${aimRatio.toFixed(0)}%`} small="aim" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">鼠标移动算法</h2>
              <label>算法模式</label>
              <select
                value={readString(controlConfig.strategy, "pid")}
                onChange={(event) => void updateConfigField("control", "strategy", event.target.value)}
              >
                <option value="pid">PID 平滑追踪</option>
                <option value="proportional">比例速度</option>
                <option value="predictive">预测追踪</option>
              </select>
              <label>触发方式</label>
              <select
                value={triggerMode}
                onChange={(event) => void updateConfigField("control", "trigger_mode", event.target.value)}
              >
                <option value="hardware">硬件按键触发</option>
                <option value="telemetry">持续计算，按键发送</option>
                <option value="always">调试直出</option>
              </select>
              <label>目标锁定</label>
              <select
                value={targetLockEnabled ? "true" : "false"}
                onChange={(event) => void updateConfigField("control", "target_lock_enabled", event.target.value === "true")}
              >
                <option value="true">开启：保持当前目标</option>
                <option value="false">关闭：每帧重新选择</option>
              </select>
              <NumberControl label="目标粘性" value={targetStickyBias} min={0} max={0.9} step={0.05} onCommit={(value) => updateConfigField("control", "target_sticky_bias", value)} />
              <NumberControl label="丢失容忍帧" value={targetLostGraceFrames} min={0} max={30} step={1} onCommit={(value) => updateConfigField("control", "target_lost_grace_frames", Math.round(value))} />
              <NumberControl label="瞄准高度 aim_ratio" value={aimRatio} min={0} max={100} step={1} onCommit={(value) => updateConfigField("control", "aim_ratio", Math.round(value))} />
              <NumberControl label="kp_x" value={pidKpX} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kp_x", value)} />
              <NumberControl label="kp_y" value={pidKpY} min={0} max={2} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kp_y", value)} />
              <NumberControl label="ki" value={pidKi} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "pid_ki", value)} />
              <NumberControl label="kd" value={pidKd} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "pid_kd", value)} />
              <NumberControl label="预测" value={predictionFactor} min={0} max={1} step={0.01} onCommit={(value) => updateConfigField("control", "prediction_factor", value)} />
              <NumberControl label="kp_x_move_max" value={kpXMoveMax} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "kp_x_move_max", value)} />
              <NumberControl label="kp_y_move_max" value={kpYMoveMax} min={0} max={500} step={1} onCommit={(value) => updateConfigField("control", "kp_y_move_max", value)} />
            </div>
            <div className="console-card">
              <h2 className="console-title">控制量反馈</h2>
              <div className="console-kv control-feedback-kv">
                <span>当前目标</span><b>{readString(target.class_name, "-")}</b>
                <span>选择状态</span><b>{readString(control.selector_state, "-")}</b>
                <span>选择原因</span><b>{readString(control.selection_reason, "-")}</b>
                <span>aim dx</span><b>{formatNumber(control.aim_error_x, 1)}</b>
                <span>aim dy</span><b>{formatNumber(control.aim_error_y, 1)}</b>
                <span>raw dx</span><b>{formatNumber(control.raw_error_x, 1)}</b>
                <span>raw dy</span><b>{formatNumber(control.raw_error_y, 1)}</b>
                <span>aim ratio</span><b>{formatNumber(control.aim_ratio, 0)}%</b>
                <span>aim point</span><b>{`${formatNumber(control.aim_x, 1)}, ${formatNumber(control.aim_y, 1)}`}</b>
                <span>预测点</span><b>{`${formatNumber(controlPipeline.predicted_x, 1)}, ${formatNumber(controlPipeline.predicted_y, 1)}`}</b>
                <span>Y 坐标约定</span><b>{readString(controlPipeline.coordinate_y, "-")}</b>
                <span>FOV counts X</span><b>{formatNumber(controlPipeline.fov_counts_x, 1)}</b>
                <span>FOV counts Y</span><b>{formatNumber(controlPipeline.fov_counts_y, 1)}</b>
                <span>速度系数</span><b>{formatNumber(controlPipeline.speed, 2)}</b>
                <span>PID P</span><b>{`${formatNumber(controlPipeline.p_x, 1)} / ${formatNumber(controlPipeline.p_y, 1)}`}</b>
                <span>PID I</span><b>{`${formatNumber(controlPipeline.i_x, 1)} / ${formatNumber(controlPipeline.i_y, 1)}`}</b>
                <span>PID D</span><b>{`${formatNumber(controlPipeline.d_x, 1)} / ${formatNumber(controlPipeline.d_y, 1)}`}</b>
                <span>dx</span><b>{formatNumber(control.dx, 1)}</b>
                <span>dy</span><b>{formatNumber(control.dy, 1)}</b>
                <span>FOV 内候选</span><b>{formatNumber(control.inside_fov, 0)}</b>
                <span>目标距离</span><b>{formatNumber(control.distance_px, 1)}</b>
                <span>触发方式</span><b>{triggerModeLabel(readString(control.trigger_mode, triggerMode))}</b>
                <span>输出状态</span><b>{control.will_emit === true ? "允许输出" : "等待触发"}</b>
                <span>触发要求</span><b>{control.trigger_required === true ? "需要硬件按键" : "调试模式直出"}</b>
                <span>触发信息</span><b>{readString(control.trigger_reason, "-") || "-"}</b>
                <span>执行器</span><b>{readString(execution.executor_id, readString(executorStatus.selected, "-"))}</b>
                <span>发送结果</span><b>{execution.sent === true ? "已发送" : execution.sent === false ? "未发送" : "-"}</b>
                <span>限幅</span><b>{executionIntent.clipped === true ? "已限幅" : executionIntent.clipped === false ? "未限幅" : "-"}</b>
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
                  <b>{kmnetStatus.monitoring === true ? "监听中" : "未监听"}</b>
                </div>
                <div className={kmnetDriverAvailable ? "kmnet-status-tile good" : "kmnet-status-tile bad"}>
                  <span>驱动</span>
                  <b>{kmnetDriverAvailable ? "可用" : "不可用"}</b>
                </div>
                <div className="kmnet-status-tile">
                  <span>平台</span>
                  <b>{`${readString(kmnetStatus.driver_platform, "-")}/${readString(kmnetStatus.driver_machine, "-")}`}</b>
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
                </div>
                <div className="kmnet-pad">
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(0, -10)}>↑</button>
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(-10, 0)}>←</button>
                  <button type="button" onClick={() => void diagnosticMoveHardware(kmnetTestDx, kmnetTestDy)} disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}>发送</button>
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(10, 0)}>→</button>
                  <button type="button" disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"} onClick={() => void diagnosticMoveHardware(0, 10)}>↓</button>
                </div>
                <button
                  className="kmnet-circle-button"
                  disabled={busy === "kmnet.diagnostic" || busy === "kmnet.circle"}
                  onClick={() => void diagnosticCircleHardware()}
                  type="button"
                >
                  {busy === "kmnet.circle" ? "画圆中" : "画圆测试"}
                </button>
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
            <KvCard title="推理统计" rows={[["完成帧", String(statistics?.inference_counter ?? 0)], ["平均耗时", formatNumber(statistics?.e2e_latency, 1)], ["最大耗时", "-"]]} />
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
            <Metric title="队列积压" value={String(readNumber(asRecord(pipeline.queue).size, 0))} small="frames" />
          </div>
          <div className="console-grid2">
            <div className="console-card">
              <h2 className="console-title">延迟链路</h2>
              <div className="console-timeline">
                <Event label="Capture" value={formatNumber(capture?.capture_wait_ms, 2)} width={30} />
                <Event label="Decode" value="--" width={22} />
                <Event label="Preprocess" value="--" width={18} />
                <Event label="Inference" value={formatNumber(statistics?.e2e_latency, 1)} width={56} />
                <Event label="Postprocess" value="--" width={20} />
              </div>
            </div>
            <KvCard title="采集诊断" rows={[["状态判断", capture?.available ? "采集中" : "等待数据"], ["建议", capture?.available ? "观察丢帧和帧间隔" : "启动后分析"], ["峰值延迟", "-"]]} />
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
      <div className="console-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => void onCommit(Number(event.target.value))}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={Number.isInteger(value) ? String(value) : value.toFixed(2)}
          onChange={(event) => {
            const next = Number(event.target.value);
            if (Number.isFinite(next)) {
              void onCommit(next);
            }
          }}
        />
      </div>
    </>
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

function PreviewCard({ runtime, title, roiSize }: { runtime: RuntimeState | null; title: string; roiSize: number }) {
  return (
    <div className="console-card">
      <h2 className="console-title">{title}</h2>
      <PreviewFrame runtime={runtime} roiSize={roiSize} />
    </div>
  );
}

function PreviewFrame({ runtime, roiSize }: { runtime: RuntimeState | null; roiSize: number }) {
  const configVersion = typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const vision = asRecord(runtime?.vision);
  const inferenceTrace = asRecord(vision.inference);
  const detections = readDetectionItems(vision.detection_items);
  const previewWidth = readNumber(inferenceTrace.input_width, roiSize);
  const previewHeight = readNumber(inferenceTrace.input_height, roiSize);
  const displaySize = Math.max(previewWidth, previewHeight, roiSize);
  return (
    <div className="console-preview" style={{ "--roi-size": `${displaySize}px` } as CSSProperties}>
      {runtime?.capture?.available ? <img alt="实时画面 / ROI" src={streamUrl(configVersion, configVersion)} /> : null}
      <div className="console-detection-layer" aria-hidden="true">
        {detections.map((detection, index) => {
          return (
            <div
              className={detectionClassName(detection)}
              key={`${detection.className}-${index}-${detection.x}-${detection.y}`}
              style={detectionStyle(detection, previewWidth, previewHeight)}
            >
              <span>{detection.className || "目标"} {detection.score.toFixed(2)}</span>
            </div>
          );
        })}
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
