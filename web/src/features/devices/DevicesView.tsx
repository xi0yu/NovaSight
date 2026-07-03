import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  type CaptureCapabilitiesResponse,
  type CaptureCapability,
  type CaptureSelectPayload,
  type RuntimeConfig,
  type RuntimeState,
  getCaptureCapabilities,
  selectCaptureProfile,
  selectImageSource,
  updateRuntimeConfig,
  stopCapture
} from "../../api";
import { Badge, EmptyState, InlineError } from "../../components/ui";
import pipelineVisualUrl from "../../assets/novasight-pipeline-visual.png";
import { formatProfile, getErrorMessage } from "../shared/format";

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
  error: string | undefined;
  onRuntimeRefresh: () => Promise<void>;
  onOpenModels: () => void;
  initialSection?: SettingsSection;
};

type SettingsSection = "capture" | "inference" | "algorithm";
type CaptureInputSource = "capture" | "image";
type ReadinessTone = "ready" | "warn" | "blocked";

const ROI_SIZE_CHOICES = [640, 480, 320, 256];

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
  if (value === "tensorrt") {
    return "TensorRT";
  }
  if (value === "onnxruntime") {
    return "ONNX Runtime";
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

export function DevicesView({
  runtime,
  error,
  onRuntimeRefresh,
  onOpenModels,
  initialSection
}: DevicesViewProps) {
  const [device, setDevice] = useState(runtime?.capture?.device ?? "/dev/video0");
  const [capabilities, setCapabilities] = useState<CaptureCapabilitiesResponse | null>(null);
  const [captureError, setCaptureError] = useState<string | undefined>(error);
  const [loadingCaps, setLoadingCaps] = useState(false);
  const [applying, setApplying] = useState<string | null>(null);
  const [stoppingCapture, setStoppingCapture] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState("MJPG");
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
    if (runtime?.capture?.device) {
      setDevice(runtime.capture.device);
    }
  }, [runtime?.capture?.device]);

  useEffect(() => {
    setCaptureError(error);
  }, [error]);

  const groups = useMemo(() => groupCapabilities(capabilities?.capabilities ?? []), [capabilities]);
  const visibleProfiles = useMemo(
    () => getVisibleProfiles(groups, selectedFormat),
    [groups, selectedFormat]
  );

  useEffect(() => {
    const activeFormat = runtime?.capture?.profile?.pixel_format?.toUpperCase();
    if (activeFormat) {
      setSelectedFormat(activeFormat);
      return;
    }
    if (groups.length > 0 && !groups.some((group) => group.pixel_format === selectedFormat)) {
      setSelectedFormat(groups[0].pixel_format);
    }
  }, [groups, runtime?.capture?.profile?.pixel_format, selectedFormat]);

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
        setCaptureError(result.reason || "设备不可用");
      }
    } catch (err) {
      if (capabilityRequestId.current !== requestId) {
        return;
      }
      setCaptureError(getErrorMessage(err));
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
        await onRuntimeRefresh();
      } catch (err) {
        setCaptureError(`切换失败，已保留上一组可用配置：${getErrorMessage(err)}`);
        await onRuntimeRefresh();
      } finally {
        setApplying(null);
      }
    },
    [onRuntimeRefresh]
  );

  const stopCaptureSession = useCallback(async () => {
    setStoppingCapture(true);
    setCaptureError(undefined);
    try {
      await stopCapture();
      await onRuntimeRefresh();
    } catch (err) {
      setCaptureError(getErrorMessage(err));
      await onRuntimeRefresh();
    } finally {
      setStoppingCapture(false);
    }
  }, [onRuntimeRefresh]);

  const applyImageSource = useCallback(async () => {
    if (!imagePath.trim()) {
      return;
    }
    setApplying("image-source");
    setCaptureError(undefined);
    try {
      await selectImageSource(imagePath.trim(), imageFps);
      await onRuntimeRefresh();
    } catch (err) {
      setCaptureError(`图片输入源切换失败：${getErrorMessage(err)}`);
      await onRuntimeRefresh();
    } finally {
      setApplying(null);
    }
  }, [imageFps, imagePath, onRuntimeRefresh]);

  const updateRuntimeField = useCallback(
    async (section: string, key: string, value: string | number | boolean) => {
      if (!runtime?.config || configBusy) {
        return;
      }
      setConfigBusy(`${section}.${key}`);
      const nextConfig = structuredClone(runtime.config) as RuntimeConfig;
      delete nextConfig.version;
      const sectionValue =
        typeof nextConfig[section] === "object" && nextConfig[section] !== null
          ? { ...(nextConfig[section] as Record<string, unknown>) }
          : {};
      sectionValue[key] = value;
      nextConfig[section] = sectionValue as RuntimeConfig[string];
      try {
        await updateRuntimeConfig(nextConfig);
        await onRuntimeRefresh();
      } catch (err) {
        setCaptureError(`配置同步失败：${getErrorMessage(err)}`);
      } finally {
        setConfigBusy(null);
      }
    },
    [configBusy, onRuntimeRefresh, runtime?.config]
  );

  const applyPreference = (preference: CaptureSelectPayload["preference"], label: string) =>
    applySelection({ device, preference }, label);

  const capture = runtime?.capture;
  const roiConfig = getNestedRecord(runtime?.config, "roi");
  const inferenceConfig = getNestedRecord(runtime?.config, "inference");
  const controlConfig = getNestedRecord(runtime?.config, "control");
  const roiSize = typeof roiConfig?.size === "number" ? roiConfig.size : 640;
  const sourceConfig = getNestedRecord(runtime?.config, "source");
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
  const inferredBackend =
    activeArtifact === "engine" ? "tensorrt" : activeArtifact === "onnx" ? "onnxruntime" : "";
  const inferenceSelected =
    typeof inferenceStatus.selected === "string" ? inferenceStatus.selected : inferredBackend;
  const inferenceStatusInputShape =
    typeof inferenceStatus.input_shape === "string" ? inferenceStatus.input_shape : "";
  const inferenceStatusOutputShape =
    typeof inferenceStatus.output_shape === "string" ? inferenceStatus.output_shape : "";
  const inferenceChecklist: { label: string; detail: string; tone: ReadinessTone }[] = [
    {
      label: "输入帧",
      detail: capture?.available ? "RoiFrame 已可用" : "先启动采集卡或图片输入",
      tone: capture?.available ? "ready" : "blocked"
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
  const controlStrategy = readString(controlConfig, "strategy", "pid");
  const fovRatio = readNumber(controlConfig, "fov_ratio", 0.28);
  const maxAbsDx = readNumber(controlConfig, "max_abs_dx", 120);
  const maxAbsDy = readNumber(controlConfig, "max_abs_dy", 120);
  const minConfidence = readNumber(controlConfig, "min_confidence", 0);
  const pidKpX = readNumber(controlConfig, "pid_kp_x", 0.35);
  const pidKpY = readNumber(controlConfig, "pid_kp_y", 0.35);
  const pidKi = readNumber(controlConfig, "pid_ki", 0.1);
  const pidKd = readNumber(controlConfig, "pid_kd", 0.1);
  const pidIntegralLimit = readNumber(controlConfig, "pid_integral_limit", 250);
  const pidMoveLimit = readNumber(controlConfig, "pid_move_limit", 120);

  useEffect(() => {
    setSelectedSource(normalizedActiveSource);
  }, [normalizedActiveSource]);

  return (
    <div className="capture-setup inference-setup">
      <main className="capture-setup-main">
        <div className="capture-setup-topbar">
          <div>
            <h2>基础设置</h2>
            <p>采集决定输入，推理消费 RoiFrame，算法参数决定控制输出。</p>
          </div>
          <div className="panel-actions">
            <button
              className="button"
              disabled={stoppingCapture || !capture?.available}
              onClick={() => void stopCaptureSession()}
              type="button"
            >
              {stoppingCapture ? "停止中" : "停止采集"}
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
            ["algorithm", "算法设置", "FOV / PID / 输出限幅"],
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
              value: capture?.available ? formatProfile(capture) : "未启动",
              detail: normalizedActiveSource === "image" ? "图片输入" : device,
              ready: Boolean(capture?.available)
            },
            {
              id: "inference",
              label: "推理消费",
              value: inferenceEnabled ? activeModelName : "已关闭",
              detail: `${inferenceSelected === "tensorrt" ? "TensorRT" : "ONNX"} · ${formatThreshold(confidenceThreshold)} 置信度`,
              ready: inferenceEnabled && activeModelName !== "未发布模型"
            },
            {
              id: "algorithm",
              label: "控制输出",
              value: controlStrategy === "pid" ? "PID 平滑追踪" : "预测追踪",
              detail: `X/Y 限幅 ${maxAbsDx}/${maxAbsDy}`,
              ready: maxAbsDx > 0 && maxAbsDy > 0
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
                  ["capture", "采集卡", capture?.available ? formatProfile(capture) : "等待启动"],
                  ["image", "图片输入", readString(sourceConfig, "image_path", "未配置图片")],
                ].map(([id, label, desc]) => (
                  <button
                    className={selectedSource === id ? "source-option active" : "source-option"}
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
                  <dd>{capture?.available ? "采集中" : "未启动"}</dd>
                </div>
                <div>
                  <dt>当前 Profile</dt>
                  <dd>{formatProfile(capture)}</dd>
                </div>
                <div>
                  <dt>后端</dt>
                  <dd>{capture?.backend ?? "未打开"}</dd>
                </div>
                <div>
                  <dt>RoiFrame</dt>
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
                      <strong>RoiFrame 输出大小</strong>
                      <span>影响采集输出、推理输入和浏览器预览裁剪尺寸。</span>
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
                            capture?.profile?.pixel_format?.toUpperCase() === row.pixel_format &&
                            capture.profile.width === row.width &&
                            capture.profile.height === row.height &&
                            capture.profile.fps === row.fps;
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
                  <dd>{capture?.available ? "RoiFrame 可用" : "等待采集"}</dd>
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
                  <span>RoiFrame</span>
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

                <div className="settings-next-row">
                  <span>推理参数确认后，下一步调整 FOV、PID 和输出限幅，避免控制量过冲。</span>
                  <button className="button compact-button" type="button" onClick={() => setActiveSection("algorithm")}>
                    继续调控制量
                  </button>
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

        {activeSection === "algorithm" ? (
          <section className="settings-console commercial-console">
            <aside className="settings-console-sidebar">
              <div className="settings-side-head">
                <strong>控制量调整</strong>
                <span>把检测结果变成可控范围内的输出，核心是 FOV、PID 和限幅。</span>
              </div>
              <dl className="settings-summary-list">
                <div>
                  <dt>策略</dt>
                  <dd>{controlStrategy === "pid" ? "PID 平滑追踪" : "预测追踪"}</dd>
                </div>
                <div>
                  <dt>FOV</dt>
                  <dd>{formatThreshold(fovRatio)}</dd>
                </div>
                <div>
                  <dt>限幅</dt>
                  <dd>X {maxAbsDx} / Y {maxAbsDy}</dd>
                </div>
                <div>
                  <dt>PID</dt>
                  <dd>Kp {pidKpX}/{pidKpY} · Ki {pidKi} · Kd {pidKd} · 积分 {pidIntegralLimit}</dd>
                </div>
              </dl>
              <div className="control-safety-note">
                <strong>安全顺序</strong>
                <span>先缩小输出上限，再提高 Kp；如果抖动明显，优先降低 Kd 或提高置信度门槛。</span>
              </div>
            </aside>

            <div className="settings-console-main">
              <div className="settings-status-strip compact-strip">
                <div>
                  <span>最低置信度</span>
                  <strong>{formatThreshold(minConfidence)}</strong>
                </div>
                <div>
                  <span>Kp X/Y</span>
                  <strong>{pidKpX} / {pidKpY}</strong>
                </div>
                <div>
                  <span>PID 上限</span>
                  <strong>{pidMoveLimit}</strong>
                </div>
                <div>
                  <span>输出上限</span>
                  <strong>{maxAbsDx} / {maxAbsDy}</strong>
                </div>
              </div>

              <div className="settings-panel">
                <div className="settings-panel-head">
                  <div>
                    <h3>算法与输出边界</h3>
                    <p>参数会实时同步到后端运行配置。FOV 决定可追踪区域，限幅决定最终控制量边界。</p>
                  </div>
                </div>

                <div className="algorithm-choice-row">
                  {["pid", "predictive"].map((strategy) => (
                    <button
                      className={controlStrategy === strategy ? "algorithm-choice active" : "algorithm-choice"}
                      key={strategy}
                      type="button"
                      onClick={() => void updateRuntimeField("control", "strategy", strategy)}
                    >
                      <strong>{strategy === "pid" ? "PID 平滑追踪" : "预测追踪"}</strong>
                      <span>{strategy === "pid" ? "Kp X/Y 分轴，Ki/Kd 共用" : "基于目标位移提前量"}</span>
                    </button>
                  ))}
                </div>

                <div className="commercial-grid">
                  <NumberField
                    label="FOV 比例"
                    value={fovRatio}
                    busy={configBusy === "control.fov_ratio"}
                    min={0.01}
                    max={1}
                    step={0.01}
                    onCommit={(value) => void updateRuntimeField("control", "fov_ratio", value)}
                  />
                  <NumberField
                    label="X 限幅"
                    value={maxAbsDx}
                    busy={configBusy === "control.max_abs_dx"}
                    min={0}
                    step={1}
                    onCommit={(value) => void updateRuntimeField("control", "max_abs_dx", value)}
                  />
                  <NumberField
                    label="Y 限幅"
                    value={maxAbsDy}
                    busy={configBusy === "control.max_abs_dy"}
                    min={0}
                    step={1}
                    onCommit={(value) => void updateRuntimeField("control", "max_abs_dy", value)}
                  />
                  <NumberField
                    label="最低置信度"
                    value={minConfidence}
                    busy={configBusy === "control.min_confidence"}
                    min={0}
                    max={1}
                    step={0.01}
                    onCommit={(value) => void updateRuntimeField("control", "min_confidence", value)}
                  />
                </div>

                <div className="control-block">
                  <div className="control-block-head">
                    <strong>PID 参数</strong>
                    <span>Kp 分 X/Y，Ki 和 Kd 共用；控制量上限会截断 PID 计算结果。</span>
                  </div>
                  <div className="commercial-grid dense">
                    {[
                      ["pid_kp_x", "Kp X", 0.35],
                      ["pid_kp_y", "Kp Y", 0.35],
                      ["pid_ki", "Ki", 0.1],
                      ["pid_kd", "Kd", 0.1],
                      ["pid_integral_limit", "积分上限", 250],
                      ["pid_move_limit", "控制量上限", 120],
                    ].map(([key, label, fallback]) => (
                      <NumberField
                        key={key}
                        label={String(label)}
                        value={readNumber(controlConfig, String(key), Number(fallback))}
                        busy={configBusy === `control.${key}`}
                        min={0}
                        step={0.01}
                        onCommit={(value) => void updateRuntimeField("control", String(key), value)}
                      />
                    ))}
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
