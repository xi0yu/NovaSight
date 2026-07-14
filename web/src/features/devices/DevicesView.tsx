import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  type CaptureCapabilitiesResponse,
  type CaptureCapability,
  type CaptureSelectPayload,
  type RuntimeConfig,
  type RuntimeState,
  getCaptureCapabilities,
  getRuntimeConfig,
  selectCaptureProfile,
  selectImageSource,
  stopRuntimePipeline,
  updateRuntimeConfig,
  stopCapture
} from "../../api";
import { Badge, EmptyState, InlineError } from "../../components/ui";
import pipelineVisualUrl from "../../assets/novasight-pipeline-visual.png";
import { reportError } from "../../lib/toast";
import { formatProfile, getErrorMessage } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";

type CapabilityChoice = {
  pixel_format: string;
  width: number;
  height: number;
  fps: number;
};

type CapabilityGroup = {
  pixel_format: string;
  choices: CapabilityChoice[];
};

type DevicesViewProps = {
  runtime: RuntimeState | null;
  runtimeConfig?: RuntimeConfig | null;
  error: string | undefined;
  onRuntimeRefresh: () => Promise<void>;
  onOpenModels: () => void;
  initialSection?: SettingsSection;
};

type SettingsSection = "capture" | "inference";
type CaptureInputSource = "capture" | "image";
type ReadinessTone = "ready" | "warn" | "blocked";

const ROI_SIZE_CHOICES = [640, 480, 320, 256];
const RUNTIME_MAINLINE_BACKENDS = new Set(["deepstream_nvinfer"]);

function getNestedRecord(value: unknown, key: string): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  const child = (value as Record<string, unknown>)[key];
  return typeof child === "object" && child !== null && !Array.isArray(child)
    ? (child as Record<string, unknown>)
    : null;
}

function formatBackendLabel(value: string): string {
  if (value === "deepstream_nvinfer") {
    return "DeepStream NVMM + nvinfer";
  }
  return value || "未选择";
}

function readinessClass(tone: ReadinessTone): string {
  return `readiness-item ${tone}`;
}

function groupCapabilities(caps: CaptureCapability[]): CapabilityGroup[] {
  const grouped = new Map<string, CapabilityChoice[]>();
  caps.forEach((cap) => {
    const pixelFormat = cap.pixel_format.toUpperCase();
    const choices = grouped.get(pixelFormat) ?? [];
    cap.fps_list.forEach((fps) => {
      choices.push({
        pixel_format: pixelFormat,
        width: cap.width,
        height: cap.height,
        fps
      });
    });
    grouped.set(pixelFormat, choices);
  });
  const order = ["MJPG", "NV12", "YUYV", "BGR3"];
  return Array.from(grouped.entries())
    .map(([pixel_format, choices]) => ({
      pixel_format,
      choices: choices.sort((left, right) =>
        right.fps - left.fps || right.width * right.height - left.width * left.height
      )
    }))
    .sort((left, right) => {
      const leftRank = order.includes(left.pixel_format) ? order.indexOf(left.pixel_format) : 99;
      const rightRank = order.includes(right.pixel_format) ? order.indexOf(right.pixel_format) : 99;
      return leftRank - rightRank || left.pixel_format.localeCompare(right.pixel_format);
    });
}

function CapabilityCompactList({
  groups,
  applying,
  disabled,
  onApply
}: {
  groups: CapabilityGroup[];
  applying: string | null;
  disabled?: boolean;
  onApply: (row: CapabilityChoice) => void;
}) {
  if (groups.length === 0) {
    return (
      <EmptyState
        title="尚未读取采集能力"
        detail="点击刷新能力，读取 /dev/video0 支持的格式、分辨率和帧率。"
        command="python3 -m novasight doctor camera --device /dev/video0"
      />
    );
  }

  return (
    <div className="capability-groups compact-capability-groups">
      {groups.map((group, index) => (
        <details className="capability-group" key={group.pixel_format} open={index < 2}>
          <summary>
            <span className="mono">{group.pixel_format}</span>
            <span>{group.choices.length} 组配置</span>
            <strong>{getFormatSummary(group)}</strong>
          </summary>
          <div className="compact-profile-list">
            {group.choices.map((row) => {
              const id = `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`;
              return (
                <button
                  className="compact-profile-row"
                  disabled={applying !== null || disabled}
                  key={id}
                  onClick={() => onApply(row)}
                  type="button"
                >
                  <strong>{row.fps}fps</strong>
                  <span>{row.width}x{row.height}</span>
                  <em className={`hint-chip ${getCapabilityTone(row)}`}>
                    {applying === id ? "应用中" : getCapabilityHint(row)}
                  </em>
                </button>
              );
            })}
          </div>
        </details>
      ))}
    </div>
  );
}

function getFormatSummary(group: CapabilityGroup): string {
  const maxFps = Math.max(...group.choices.map((choice) => choice.fps));
  const maxPixels = Math.max(...group.choices.map((choice) => choice.width * choice.height));
  const maxChoice = group.choices.find((choice) => choice.width * choice.height === maxPixels);
  return `${maxChoice?.width ?? 0}x${maxChoice?.height ?? 0} / ${maxFps}fps`;
}

function getCapabilityHint(row: CapabilityChoice): string {
  if (row.pixel_format === "MJPG" && row.fps >= 120) {
    return "高帧率";
  }
  if (row.pixel_format === "NV12" && row.fps >= 60) {
    return "低延迟";
  }
  if (row.width === 1920 && row.height === 1080) {
    return "均衡";
  }
  return "手动";
}

function getCapabilityTone(row: CapabilityChoice): string {
  if (row.pixel_format === "MJPG" && row.fps >= 120) {
    return "prime";
  }
  if (row.pixel_format === "NV12" && row.fps >= 60) {
    return "cool";
  }
  if (row.width === 1920 && row.height === 1080) {
    return "balanced";
  }
  return "plain";
}

function getFormatDescription(pixelFormat: string): string {
  switch (pixelFormat) {
    case "MJPG":
      return "高帧率，适合 1K240 / 2K144";
    case "NV12":
      return "低转换成本，适合稳定低延迟";
    case "YUYV":
      return "兼容性好，高分辨率帧率较低";
    case "BGR3":
      return "原始 BGR，作为特殊兼容路径";
    default:
      return "采集卡返回的扩展格式";
  }
}

function getFormatPill(pixelFormat: string): { label: string; tone: string } {
  switch (pixelFormat) {
    case "MJPG":
      return { label: "推荐", tone: "prime" };
    case "NV12":
      return { label: "稳定", tone: "cool" };
    case "YUYV":
      return { label: "备用", tone: "amber" };
    default:
      return { label: "高级", tone: "plain" };
  }
}

function isMainstreamProfile(choice: CapabilityChoice): boolean {
  const isStandardRate = [60, 120, 144, 240].includes(choice.fps);
  const isCommonSize =
    (choice.width === 1920 && choice.height === 1080) ||
    (choice.width === 2560 && choice.height === 1440) ||
    (choice.width === 3840 && choice.height === 2160);
  return isStandardRate && isCommonSize;
}

function profilePriority(choice: CapabilityChoice): number {
  const sizeRank = choice.width === 1920 && choice.height === 1080 ? 0 : 1;
  return -choice.fps * 100000 + sizeRank * 10000 - choice.width * choice.height;
}

function getVisibleProfiles(groups: CapabilityGroup[], selectedFormat: string): CapabilityChoice[] {
  const group = groups.find((item) => item.pixel_format === selectedFormat);
  const choices = group?.choices ?? [];
  const mainstream = choices.filter(isMainstreamProfile);
  return (mainstream.length > 0 ? mainstream : choices).sort(
    (left, right) => profilePriority(left) - profilePriority(right)
  ).slice(0, 6);
}

function getProfileTitle(choice: CapabilityChoice): string {
  const heightLabel = choice.height >= 2160 ? "4K" : choice.height >= 1440 ? "2K" : "1K";
  return `${heightLabel} · ${choice.fps}fps`;
}

function getProfileTag(choice: CapabilityChoice): string {
  if (choice.pixel_format === "MJPG" && choice.fps >= 180) {
    return "最高帧率";
  }
  if (choice.pixel_format === "MJPG" && choice.height >= 1440) {
    return "高画质";
  }
  if (choice.pixel_format === "NV12" && choice.fps >= 120) {
    return "低延迟";
  }
  if (choice.height >= 2160) {
    return "预览/录制";
  }
  return "均衡";
}

function formatThreshold(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function readConfiguredCapture(
  runtime: RuntimeState | null,
  runtimeConfig?: RuntimeConfig | null
): {
  device: string;
  pixelFormat: string;
  width: number;
  height: number;
  fps: number;
} {
  const captureConfig = getNestedRecord(runtimeConfig ?? runtime?.config, "capture");
  const configuredDevice = captureConfig?.device;
  const configuredPixelFormat = captureConfig?.pixel_format;
  const configuredWidth = captureConfig?.width;
  const configuredHeight = captureConfig?.height;
  const configuredFps = captureConfig?.fps;
  return {
    device:
      typeof configuredDevice === "string" && configuredDevice.trim()
        ? configuredDevice
        : runtime?.capture?.device || "/dev/video0",
    pixelFormat:
      typeof configuredPixelFormat === "string" && configuredPixelFormat.trim()
        ? configuredPixelFormat.toUpperCase()
        : runtime?.capture?.profile?.pixel_format?.toUpperCase() || "MJPG",
    width: typeof configuredWidth === "number" ? configuredWidth : runtime?.capture?.profile?.width ?? 0,
    height: typeof configuredHeight === "number" ? configuredHeight : runtime?.capture?.profile?.height ?? 0,
    fps: typeof configuredFps === "number" ? configuredFps : runtime?.capture?.profile?.fps ?? 0
  };
}

export function DevicesView({
  runtime,
  runtimeConfig,
  error,
  onRuntimeRefresh,
  onOpenModels,
  initialSection
}: DevicesViewProps) {
  const configuredCapture = readConfiguredCapture(runtime, runtimeConfig);
  const [device, setDevice] = useState(configuredCapture.device);
  const [capabilities, setCapabilities] = useState<CaptureCapabilitiesResponse | null>(null);
  const [captureError, setCaptureError] = useState<string | undefined>(error);
  const [loadingCaps, setLoadingCaps] = useState(false);
  const [applying, setApplying] = useState<string | null>(null);
  const [stoppingCapture, setStoppingCapture] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState(configuredCapture.pixelFormat);
  const [activeSection, setActiveSection] = useState<SettingsSection>("capture");
  const [selectedSource, setSelectedSource] = useState<CaptureInputSource>("capture");
  const [imagePath, setImagePath] = useState("");
  const [imageFps, setImageFps] = useState(15);
  const [configBusy, setConfigBusy] = useState<string | null>(null);
  const capabilityRequestId = useRef(0);

  useEffect(() => {
    if (initialSection) {
      setActiveSection(initialSection);
    }
  }, [initialSection]);

  useEffect(() => {
    if (configuredCapture.device) {
      setDevice(configuredCapture.device);
    }
  }, [configuredCapture.device]);

  useEffect(() => {
    setCaptureError(error);
  }, [error]);

  const groups = useMemo(() => groupCapabilities(capabilities?.capabilities ?? []), [capabilities]);
  const visibleProfiles = useMemo(
    () => getVisibleProfiles(groups, selectedFormat),
    [groups, selectedFormat]
  );

  useEffect(() => {
    const activeFormat =
      runtime?.capture?.profile?.pixel_format?.toUpperCase() || configuredCapture.pixelFormat;
    if (activeFormat) {
      setSelectedFormat(activeFormat);
      return;
    }
    if (groups.length > 0 && !groups.some((group) => group.pixel_format === selectedFormat)) {
      setSelectedFormat(groups[0].pixel_format);
    }
  }, [configuredCapture.pixelFormat, groups, runtime?.capture?.profile?.pixel_format, selectedFormat]);

  const refreshCapabilities = useCallback(async () => {
    const requestId = capabilityRequestId.current + 1;
    capabilityRequestId.current = requestId;
    const requestedDevice = device;
    setLoadingCaps(true);
    setCaptureError(undefined);
    try {
      const result = await getCaptureCapabilities(requestedDevice);
      if (capabilityRequestId.current !== requestId) {
        return;
      }
      setCapabilities(result);
      if (!result.available) {
        const reason = result.reason || "设备不可用";
        setCaptureError(reason);
        reportError(new Error(reason), { source: "capture-caps", title: "采集设备不可用" });
      }
    } catch (err) {
      if (capabilityRequestId.current !== requestId) {
        return;
      }
      setCaptureError(getErrorMessage(err));
      reportError(err, { source: "capture-caps", title: "采集能力读取失败" });
    } finally {
      if (capabilityRequestId.current === requestId) {
        setLoadingCaps(false);
      }
    }
  }, [device]);

  useEffect(() => {
    void refreshCapabilities();
  }, [refreshCapabilities]);

  const applySelection = useCallback(
    async (payload: CaptureSelectPayload, label: string) => {
      setApplying(label);
      setCaptureError(undefined);
      try {
        await selectCaptureProfile(payload);
      } catch (err) {
        setCaptureError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);
        reportError(err, { source: "capture-select", title: "采集配置切换失败" });
        await onRuntimeRefresh();
      } finally {
      }
    },
    [onRuntimeRefresh]
  );

  const stopCaptureSession = useCallback(async () => {
    setStoppingCapture(true);
    setCaptureError(undefined);
    try {
      const selected = typeof runtime?.inference?.selected === "string" ? runtime.inference.selected : "";
      if (RUNTIME_MAINLINE_BACKENDS.has(selected)) {
        await stopRuntimePipeline();
      } else {
        await stopCapture();
      }
    } catch (err) {
      setCaptureError(getErrorMessage(err));
      reportError(err, { source: "capture-stop", title: "停止采集失败" });
      await onRuntimeRefresh();
    } finally {
    }
  }, [onRuntimeRefresh, runtime]);

  const applyImageSource = useCallback(async () => {
    if (!imagePath.trim()) {
      return;
    }
    setApplying("image-source");
    setCaptureError(undefined);
    try {
      await selectImageSource(imagePath.trim(), imageFps);
    } catch (err) {
      setCaptureError(`图片输入源切换失败：${getErrorMessage(err)}`);
      reportError(err, { source: "capture-image", title: "图片输入源切换失败" });
      await onRuntimeRefresh();
    } finally {
    }
  }, [imageFps, imagePath, onRuntimeRefresh]);

  const updateRuntimeField = useCallback(
    async (section: string, key: string, value: string | number | boolean) => {
      if (configBusy) {
        return;
      }
      setConfigBusy(`${section}.${key}`);
      try {
        const nextConfig = await getRuntimeConfig();
        const sectionValue =
          typeof nextConfig[section] === "object" && nextConfig[section] !== null
            ? { ...(nextConfig[section] as Record<string, unknown>) }
            : {};
        sectionValue[key] = value;
        nextConfig[section] = sectionValue as RuntimeConfig[string];
        await updateRuntimeConfig(nextConfig);
        await onRuntimeRefresh();
      } catch (err) {
        setCaptureError(`配置同步失败：${getErrorMessage(err)}`);
        reportError(err, { source: "runtime-config", title: "配置同步失败" });
        await onRuntimeRefresh();
      } finally {
      }
    },
    [configBusy, onRuntimeRefresh]
  );

  const applyPreference = (preference: CaptureSelectPayload["preference"], label: string) =>
    applySelection({ device, preference }, label);

  const capture = runtime?.capture;
  const configSource = runtimeConfig ?? runtime?.config;
  const roiConfig = getNestedRecord(configSource, "roi");
  const inferenceConfig = getNestedRecord(configSource, "inference");
  const roiSize = typeof roiConfig?.size === "number" ? roiConfig.size : 640;
  const sourceConfig = getNestedRecord(configSource, "source");
  const activeSource = String(sourceConfig?.default ?? runtime?.source ?? "null");
  const normalizedActiveSource: CaptureInputSource =
    activeSource === "image" || activeSource.startsWith("image:") ? "image" : "capture";
  const readNumber = (section: Record<string, unknown> | null, key: string, fallback: number) => {
    const value = section?.[key];
    return typeof value === "number" ? value : fallback;
  };
  const readString = (section: Record<string, unknown> | null, key: string, fallback: string) => {
    const value = section?.[key];
    return typeof value === "string" ? value : fallback;
  };
  const readBoolean = (section: Record<string, unknown> | null, key: string, fallback: boolean) => {
    const value = section?.[key];
    return typeof value === "boolean" ? value : fallback;
  };
  const inferenceEnabled = readBoolean(inferenceConfig, "enabled", true);
  const confidenceThreshold = readNumber(inferenceConfig, "confidence_threshold", 0.25);
  const nmsThreshold = readNumber(inferenceConfig, "nms_threshold", 0.45);
  const activeModelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const activeVersion = runtime?.active_model?.version ?? null;
  const activeInputShape = activeVersion?.input_shape ?? "";
  const activeClasses = activeVersion?.classes ?? [];
  const activeClassPreview = activeClasses.slice(0, 6).join(" / ");
  const activeArtifact = runtime?.active_model?.artifact?.kind ?? "未绑定";
  const activeArtifactPath = runtime?.active_model?.artifact?.path ?? "未绑定文件";
  const runnableArtifact = activeArtifact === "onnx" || activeArtifact === "engine";
  const inferenceStatus = runtime?.inference ?? {};
  const inferenceLoaded = inferenceStatus.loaded === true;
  const inferenceSupportsExecution = inferenceStatus.supports_execution !== false;
  const inferenceReason = typeof inferenceStatus.reason === "string" ? inferenceStatus.reason : "";
  const inferredBackend = activeArtifact === "engine" ? "deepstream_nvinfer" : "";
  const inferenceSelected =
    typeof inferenceStatus.selected === "string" ? inferenceStatus.selected : inferredBackend;
  const deepstreamNvinferSelected = inferenceSelected === "deepstream_nvinfer";
  const runtimeMainlineSelected = RUNTIME_MAINLINE_BACKENDS.has(inferenceSelected);
  const runtimeMainlineStatus = getRuntimeMainlineStatus(runtime);
  const runtimeMainlineRunning = runtimeMainlineStatus.running;
  const captureMainRunning = runtimeMainlineSelected ? runtimeMainlineRunning : capture?.available === true;
  const captureProfileConfigured = capture?.available === true || Boolean(capture?.profile);
  const mainlineInputReady = runtimeMainlineStatus.hasInferenceSignal;
  const mainlineRuntimeReady = runtimeMainlineStatus.hasRuntimeConsumption;
  const mainlineBackendLabel = runtimeMainlineSelected
    ? `${formatBackendLabel(inferenceSelected)} 主链`
    : "未选择主链";
  const mainlineRunLabel = runtimeMainlineStatus.failed
    ? "主链故障"
    : runtimeMainlineRunning
      ? mainlineRuntimeReady
        ? "主链已消费"
        : mainlineInputReady
          ? "等待 runtime 消费"
          : "等待 DetectionBatch"
      : captureProfileConfigured
        ? "主链待启动"
        : "未启动";
  const mainlineInputLabel = runtimeMainlineStatus.failed
    ? "管线故障"
    : mainlineRuntimeReady
      ? "DetectionBatch 已消费"
      : mainlineInputReady
        ? "DetectionBatch 已产出"
        : runtimeMainlineRunning
          ? "等待 DetectionBatch"
          : "等待主链启动";
  const inferenceStatusInputShape =
    typeof inferenceStatus.input_shape === "string" ? inferenceStatus.input_shape : "";
  const inferenceStatusOutputShape =
    typeof inferenceStatus.output_shape === "string" ? inferenceStatus.output_shape : "";
  const inferenceChecklist: { label: string; detail: string; tone: ReadinessTone }[] = [
    {
      label: "输入帧",
      detail: runtimeMainlineSelected
        ? runtimeMainlineStatus.failed
          ? runtimeMainlineStatus.failureMessage || `${mainlineBackendLabel}故障`
          : captureMainRunning
            ? mainlineInputReady
              ? `${mainlineBackendLabel}已产出 DetectionBatch`
              : `${mainlineBackendLabel}运行中，${runtimeMainlineStatus.progressSummary}`
          : captureProfileConfigured
            ? "采集 Profile 已配置，启动主链后由后端打开"
            : "先选择采集卡 Profile"
        : capture?.available
          ? "RoiFrame 已可用"
          : "先启动采集卡或图片输入",
      tone: runtimeMainlineSelected
        ? runtimeMainlineStatus.failed
          ? "blocked"
          : mainlineInputReady
            ? "ready"
            : captureMainRunning || captureProfileConfigured
              ? "warn"
              : "blocked"
        : capture?.available
          ? "ready"
          : captureProfileConfigured
            ? "warn"
            : "blocked"
    },
    {
      label: "运行模型",
      detail: runnableArtifact ? `${activeArtifactPath} 可推理` : "请选择 ONNX 或 Engine",
      tone: runnableArtifact ? "ready" : "blocked"
    },
    {
      label: "运行时加载",
      detail: inferenceLoaded ? `${formatBackendLabel(inferenceSelected)} 已加载` : inferenceReason || "等待模型加载",
      tone: inferenceLoaded ? "ready" : "warn"
    },
    {
      label: "检测过滤",
      detail: `置信度 ${formatThreshold(confidenceThreshold)} · NMS ${formatThreshold(nmsThreshold)}`,
      tone: inferenceEnabled ? "ready" : "blocked"
    }
  ];
  useEffect(() => {
    setSelectedSource(normalizedActiveSource);
  }, [normalizedActiveSource]);

  return (
    <div className="capture-setup inference-setup">
      <main className="capture-setup-main">
        <div className="capture-setup-topbar">
          <div>
            <h2>基础设置</h2>
            <p>{deepstreamNvinferSelected ? "采集与 nvinfer 在 NVMM 中运行，控制只消费 DetectionBatch。" : "采集决定输入，推理消费 RoiFrame，算法参数决定控制输出。"}</p>
          </div>
          <div className="panel-actions">
            <button
              className="button"
              disabled={runtimeMainlineSelected ? stoppingCapture || !runtimeMainlineRunning : stoppingCapture || !capture?.available}
              onClick={() => void stopCaptureSession()}
              type="button"
            >
              {stoppingCapture ? "停止中" : runtimeMainlineSelected ? "停止主链" : "停止采集"}
            </button>
            <button className="button" type="button" onClick={refreshCapabilities}>
              {loadingCaps ? "读取中" : "刷新采集卡信息"}
            </button>
          </div>
        </div>

        <InlineError message={captureError} />

        <div className="settings-tabs" role="tablist" aria-label="基础设置分类">
          {[
            ["capture", "采集设置", "设备 / 图片输入 / Profile"],
            ["inference", "推理设置", "ROI / 检测阈值 / 后端"],
          ].map(([id, label, desc]) => (
            <button
              className={activeSection === id ? "settings-tab active" : "settings-tab"}
              key={id}
              onClick={() => setActiveSection(id as SettingsSection)}
              type="button"
            >
              <strong>{label}</strong>
              <span>{desc}</span>
            </button>
          ))}
        </div>

        <div className="settings-flow" aria-label="采集推理控制量流程">
          {[
            {
              id: "capture",
              label: "采集输入",
              value: captureProfileConfigured ? formatProfile(capture) : "未启动",
              detail: runtimeMainlineSelected
                ? mainlineRunLabel
                : normalizedActiveSource === "image" ? "图片输入" : device,
              ready: runtimeMainlineSelected
                ? mainlineRuntimeReady
                : Boolean(capture?.available)
            },
            {
              id: "inference",
              label: "推理消费",
              value: inferenceEnabled ? activeModelName : "已关闭",
              detail: runtimeMainlineSelected
                ? mainlineInputLabel
                : `TensorRT · ${formatThreshold(confidenceThreshold)} 置信度`,
              ready: runtimeMainlineSelected
                ? inferenceEnabled && mainlineInputReady && activeModelName !== "未发布模型"
                : inferenceEnabled && activeModelName !== "未发布模型"
            }
          ].map((step, index) => (
            <button
              className={activeSection === step.id ? "flow-step active" : step.ready ? "flow-step ready" : "flow-step"}
              key={step.id}
              onClick={() => setActiveSection(step.id as SettingsSection)}
              type="button"
            >
              <span className="flow-index">{index + 1}</span>
              <span className="flow-copy">
                <strong>{step.label}</strong>
                <em>{step.value}</em>
                <small>{step.detail}</small>
              </span>
            </button>
          ))}
        </div>

        {activeSection === "capture" ? (
          <section className="settings-console">
            <aside className="settings-console-sidebar">
              <div className="settings-side-head">
                <strong>采集输入源</strong>
                <span>选择后再执行应用动作，不做隐式启动。</span>
              </div>
              <div className="source-segmented" role="tablist" aria-label="采集输入源">
                {[
                  [
                    "capture",
                    "采集卡",
                    captureProfileConfigured
                      ? runtimeMainlineSelected
                        ? `${formatProfile(capture)} · 主链启动时打开`
                        : formatProfile(capture)
                      : "等待启动",
                  ],
                  ["image", "图片输入", deepstreamNvinferSelected ? "DeepStream nvinfer 不支持" : readString(sourceConfig, "image_path", "未配置图片")],
                ].map(([id, label, desc]) => (
                  <button
                    className={selectedSource === id ? "source-option active" : "source-option"}
                    disabled={deepstreamNvinferSelected && id === "image"}
                    key={id}
                    type="button"
                    onClick={() => setSelectedSource(id as CaptureInputSource)}
                  >
                    <span>{label}</span>
                    <small>{desc}</small>
                  </button>
                ))}
              </div>

              <dl className="settings-summary-list">
                <div>
                  <dt>当前来源</dt>
                  <dd>{normalizedActiveSource === "image" ? "图片输入" : "采集卡"}</dd>
                </div>
                <div>
                  <dt>运行状态</dt>
                  <dd>{runtimeMainlineSelected ? mainlineRunLabel : captureMainRunning ? "采集中" : captureProfileConfigured ? "已配置" : "未启动"}</dd>
                </div>
                <div>
                  <dt>当前 Profile</dt>
                  <dd>{formatProfile(capture)}</dd>
                </div>
                <div>
                  <dt>后端</dt>
                  <dd>{runtimeMainlineSelected ? inferenceSelected : capture?.backend ?? "未打开"}</dd>
                </div>
                <div>
                  <dt>{deepstreamNvinferSelected ? "ROI 坐标空间" : "RoiFrame"}</dt>
                  <dd>{roiSize}x{roiSize}</dd>
                </div>
              </dl>
            </aside>

            <div className="settings-console-main">
              <div className="settings-status-strip">
                <div>
                  <span>设备</span>
                  <strong>{device}</strong>
                </div>
                <div>
                  <span>格式</span>
                  <strong>{capture?.profile?.pixel_format ?? selectedFormat}</strong>
                </div>
                <div>
                  <span>帧率</span>
                  <strong>{capture?.profile ? `${capture.profile.fps}fps` : "--"}</strong>
                </div>
                <div>
                  <span>最近错误</span>
                  <strong>{capture?.last_error ?? "无"}</strong>
                </div>
              </div>

              {selectedSource === "capture" ? (
                <div className="settings-panel">
                  <div className="settings-panel-head">
                    <div>
                      <h3>采集卡输入</h3>
                      <p>先选设备与策略，再应用 Profile 启动采集。</p>
                    </div>
                    <div className="settings-panel-actions">
                      <button
                        className="button compact-button"
                        disabled={applying !== null}
                        type="button"
                        onClick={() => applyPreference("auto_high_fps", "auto-high-fps")}
                      >
                        自动高帧率
                      </button>
                      <button
                        className="button compact-button"
                        disabled={applying !== null}
                        type="button"
                        onClick={() => applyPreference("auto_low_latency", "auto-low-latency")}
                      >
                        自动低延迟
                      </button>
                    </div>
                  </div>

                  <div className="control-row">
                    <label className="device-input">
                      <span>设备路径</span>
                      <input value={device} onChange={(event) => setDevice(event.target.value)} />
                    </label>
                    <button className="button" type="button" onClick={refreshCapabilities}>
                      {loadingCaps ? "读取中" : "刷新能力"}
                    </button>
                  </div>

                  <div className="control-block">
                    <div className="control-block-head">
                      <strong>{deepstreamNvinferSelected ? "ROI 控制坐标大小" : "RoiFrame 输出大小"}</strong>
                      <span>{deepstreamNvinferSelected ? "定义检测框映射、目标选择和控制坐标空间；nvinfer 输入尺寸由模型决定。" : "影响采集输出、推理输入和浏览器预览裁剪尺寸。"}</span>
                    </div>
                    <div className="home-chips">
                      {ROI_SIZE_CHOICES.map((size) => (
                        <button
                          className={roiSize === size ? "home-chip active" : "home-chip"}
                          disabled={configBusy === "roi.size"}
                          key={size}
                          onClick={() => void updateRuntimeField("roi", "size", size)}
                          type="button"
                        >
                          {size}x{size}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="control-block">
                    <div className="control-block-head">
                      <strong>像素格式</strong>
                      <span>按项目优先级分组，组内 Profile 按 FPS 降序。</span>
                    </div>
                    <div className="format-tabs">
                      {groups.map((group) => {
                        const pill = getFormatPill(group.pixel_format);
                        return (
                          <button
                            className={group.pixel_format === selectedFormat ? "format-tab active" : "format-tab"}
                            key={group.pixel_format}
                            onClick={() => setSelectedFormat(group.pixel_format)}
                            type="button"
                          >
                            <strong>{group.pixel_format}</strong>
                            <span>{getFormatSummary(group)}</span>
                            <em className={`pill ${pill.tone}`}>{pill.label}</em>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div className="control-block">
                    <div className="control-block-head">
                      <strong>Profile</strong>
                      <span>点击某一行会明确应用并启动采集。</span>
                    </div>
                    {visibleProfiles.length > 0 ? (
                      <div className="profile-list">
                        {visibleProfiles.map((row) => {
                          const id = `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`;
                          const active =
                            (capture?.profile
                              ? capture.profile.pixel_format?.toUpperCase() === row.pixel_format &&
                                capture.profile.width === row.width &&
                                capture.profile.height === row.height &&
                                capture.profile.fps === row.fps
                              : configuredCapture.pixelFormat === row.pixel_format &&
                                configuredCapture.width === row.width &&
                                configuredCapture.height === row.height &&
                                configuredCapture.fps === row.fps);
                          return (
                            <button
                              className={active ? "profile-row active" : "profile-row"}
                              disabled={applying !== null}
                              key={id}
                              onClick={() =>
                                applySelection(
                                  {
                                    device,
                                    preference: "manual",
                                    pixel_format: row.pixel_format,
                                    width: row.width,
                                    height: row.height,
                                    fps: row.fps
                                  },
                                  id
                                )
                              }
                              type="button"
                            >
                              <span className="profile-row-main">
                                <strong>{getProfileTitle(row)}</strong>
                                <small>{row.width}x{row.height} · {row.pixel_format}</small>
                              </span>
                              <span className="profile-row-fps">{row.fps} fps</span>
                              <em>{applying === id ? "应用中" : getProfileTag(row)}</em>
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <EmptyState
                        title="尚未读取采集能力"
                        detail="点击刷新能力，读取设备支持的格式、分辨率和帧率。"
                        command="python3 -m novasight doctor camera --device /dev/video0"
                      />
                    )}
                  </div>

                  <details className="advanced-capability-block advanced-compact">
                    <summary>高级能力：完整 v4l2 枚举</summary>
                    <CapabilityCompactList
                      applying={applying}
                      groups={groups}
                      onApply={(row) =>
                        applySelection(
                          {
                            device,
                            preference: "manual",
                            pixel_format: row.pixel_format,
                            width: row.width,
                            height: row.height,
                            fps: row.fps
                          },
                          `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`
                        )
                      }
                    />
                  </details>
                </div>
              ) : (
                <div className="settings-panel">
                  <div className="settings-panel-head">
                    <div>
                      <h3>图片输入</h3>
                      <p>用于模型、预览和配置链路测试，不占用采集卡。</p>
                    </div>
                  </div>

                  <div className="control-block">
                    <div className="control-block-head">
                      <strong>RoiFrame 输出大小</strong>
                      <span>图片输入同样会按这个尺寸裁剪后进入推理。</span>
                    </div>
                    <div className="home-chips">
                      {ROI_SIZE_CHOICES.map((size) => (
                        <button
                          className={roiSize === size ? "home-chip active" : "home-chip"}
                          disabled={configBusy === "roi.size"}
                          key={size}
                          onClick={() => void updateRuntimeField("roi", "size", size)}
                          type="button"
                        >
                          {size}x{size}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="source-action-grid">
                    <label className="device-input">
                      <span>图片路径</span>
                      <input
                        placeholder="/home/nvidia/NovaSight/data/sample.jpg"
                        value={imagePath}
                        onChange={(event) => setImagePath(event.target.value)}
                      />
                    </label>
                    <div className="control-block inline">
                      <div className="control-block-head">
                        <strong>循环帧率</strong>
                        <span>仅用于测试源节奏。</span>
                      </div>
                      <div className="home-chips">
                        {[1, 5, 15, 30, 60].map((fps) => (
                          <button
                            className={imageFps === fps ? "home-chip active" : "home-chip"}
                            key={fps}
                            type="button"
                            onClick={() => setImageFps(fps)}
                          >
                            {fps}fps
                          </button>
                        ))}
                      </div>
                    </div>
                    <button
                      className="button"
                      type="button"
                      disabled={!imagePath.trim() || applying !== null}
                      onClick={() => void applyImageSource()}
                    >
                      {applying === "image-source" ? "切换中" : "切换到图片输入"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </section>
        ) : null}

        {activeSection === "inference" ? (
          <section className="settings-console commercial-console">
            <aside className="settings-console-sidebar visual-sidebar">
              <div className="settings-side-head">
                <strong>推理链路</strong>
                <span>用户只需要确认三件事：有输入、有模型、阈值合适。</span>
              </div>
              <img className="pipeline-visual" src={pipelineVisualUrl} alt="" aria-hidden="true" />
              <dl className="settings-summary-list">
                <div>
                  <dt>输入</dt>
                  <dd>{runtimeMainlineSelected ? mainlineInputLabel : capture?.available ? "RoiFrame 可用" : "等待采集"}</dd>
                </div>
                <div>
                  <dt>模型</dt>
                  <dd>{activeModelName}</dd>
                </div>
                <div>
                  <dt>输入规格</dt>
                  <dd>{activeInputShape || "未标注"}</dd>
                </div>
                <div>
                  <dt>类别</dt>
                  <dd>{activeClasses.length > 0 ? `${activeClasses.length} 类` : "未标注"}</dd>
                </div>
                <div>
                  <dt>阈值</dt>
                  <dd>{formatThreshold(confidenceThreshold)} / NMS {formatThreshold(nmsThreshold)}</dd>
                </div>
              </dl>
            </aside>

            <div className="settings-console-main">
              <div className="settings-status-strip compact-strip">
                <div>
                  <span>{deepstreamNvinferSelected ? "ROI 坐标" : "RoiFrame"}</span>
                  <strong>{roiSize}x{roiSize}</strong>
                </div>
                <div>
                  <span>输入源</span>
                  <strong>{normalizedActiveSource === "image" ? "图片输入" : "采集卡"}</strong>
                </div>
                <div>
                  <span>自动后端</span>
                  <strong>{formatBackendLabel(inferenceSelected)}</strong>
                </div>
                <div>
                  <span>推理状态</span>
                  <strong>
                    {!inferenceSupportsExecution
                      ? "不支持真实执行"
                      : inferenceLoaded
                        ? `已加载 ${inferenceSelected}`
                        : inferenceReason || "未加载"}
                  </strong>
                </div>
              </div>

              <div className="readiness-grid">
                {inferenceChecklist.map((item) => (
                  <div className={readinessClass(item.tone)} key={item.label}>
                    <span>{item.label}</span>
                    <strong>{item.detail}</strong>
                  </div>
                ))}
              </div>

              <div className="settings-panel">
                <div className="settings-panel-head">
                  <div>
                    <h3>推理配置</h3>
                    <p>推理默认跟随采集流启动；当前阈值会同步到已加载模型运行时，用于过滤检测结果。</p>
                  </div>
                  <button
                    className={inferenceEnabled ? "config-toggle on" : "config-toggle"}
                    type="button"
                    disabled={configBusy === "inference.enabled"}
                    onClick={() => void updateRuntimeField("inference", "enabled", !inferenceEnabled)}
                  >
                    {inferenceEnabled ? "推理已启用" : "推理已关闭"}
                  </button>
                </div>

                <div className="commercial-grid">
                  <div className="commercial-field span-2">
                    <span>当前模型</span>
                    <strong>{activeModelName}</strong>
                  </div>
                  <div className="commercial-field">
                    <span>输入尺寸</span>
                    <strong>{activeInputShape || inferenceStatusInputShape || "未标注"}</strong>
                  </div>
                  <div className="commercial-field">
                    <span>输出尺寸</span>
                    <strong>{inferenceStatusOutputShape || "等待推理运行时"}</strong>
                  </div>
                  <div className="commercial-field span-2">
                    <span>类别</span>
                    <strong>{activeClasses.length > 0 ? `${activeClasses.length} 类` : "未标注类别"}</strong>
                    <small>{activeClassPreview || "模型元数据没有类别名，检测结果只能显示类别编号。"}</small>
                  </div>
                  <div className="commercial-field span-2">
                    <span>自动后端</span>
                    <strong>
                      {activeArtifact === "engine"
                        ? "TensorRT · engine"
                        : activeArtifact === "onnx"
                          ? "ONNX Runtime · onnx"
                          : "等待选择 ONNX 或 engine 模型"}
                    </strong>
                  </div>
                  <NumberField
                    label="置信度阈值"
                    value={confidenceThreshold}
                    busy={configBusy === "inference.confidence_threshold"}
                    min={0}
                    max={1}
                    step={0.01}
                    onCommit={(value) => void updateRuntimeField("inference", "confidence_threshold", value)}
                  />
                  <NumberField
                    label="NMS 阈值"
                    value={nmsThreshold}
                    busy={configBusy === "inference.nms_threshold"}
                    min={0}
                    max={1}
                    step={0.01}
                    onCommit={(value) => void updateRuntimeField("inference", "nms_threshold", value)}
                  />
                </div>

                <div className={!inferenceSupportsExecution || inferenceReason ? "model-binding-card warn" : "model-binding-card"}>
                  <div>
                    <span>模型仓库绑定</span>
                    <strong>{activeModelName}</strong>
                    <p>
                      {!runnableArtifact
                        ? "当前没有可直接推理的 ONNX 或 Engine 模型。"
                        : !inferenceSupportsExecution
                        ? `${activeArtifactPath} · TensorRT engine 当前只能加载/预处理，尚未接入真实执行绑定`
                        : inferenceReason
                          ? `${activeArtifactPath} · ${inferenceReason}`
                          : activeArtifactPath}
                    </p>
                  </div>
                  <div className="model-binding-actions">
                    <Badge tone={activeArtifact === "未绑定" ? "idle" : "good"}>
                      {activeArtifact === "engine" ? "TensorRT 引擎" : activeArtifact === "onnx" ? "ONNX 模型" : activeArtifact}
                    </Badge>
                    <button className="button compact-button" type="button" onClick={onOpenModels}>
                      {runnableArtifact ? "更换模型" : "去模型仓库选择"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>
        ) : null}

      </main>

    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  busy,
  onCommit,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  busy?: boolean;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commit = () => {
    const next = Number(draft);
    if (!Number.isFinite(next)) {
      setDraft(String(value));
      return;
    }
    onCommit(next);
  };

  return (
    <label className="settings-number-field">
      <span>{label}</span>
      <input
        disabled={busy}
        max={max}
        min={min}
        step={step}
        type="number"
        value={draft}
        onBlur={commit}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}
