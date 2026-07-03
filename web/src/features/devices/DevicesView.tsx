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
import { EmptyState, InlineError } from "../../components/ui";
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
};

type SettingsSection = "capture" | "inference" | "algorithm";

function getNestedRecord(value: unknown, key: string): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  const child = (value as Record<string, unknown>)[key];
  return typeof child === "object" && child !== null && !Array.isArray(child)
    ? (child as Record<string, unknown>)
    : null;
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

export function DevicesView({
  runtime,
  error,
  onRuntimeRefresh
}: DevicesViewProps) {
  const [device, setDevice] = useState(runtime?.capture?.device ?? "/dev/video0");
  const [capabilities, setCapabilities] = useState<CaptureCapabilitiesResponse | null>(null);
  const [captureError, setCaptureError] = useState<string | undefined>(error);
  const [loadingCaps, setLoadingCaps] = useState(false);
  const [applying, setApplying] = useState<string | null>(null);
  const [stoppingCapture, setStoppingCapture] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState("MJPG");
  const [activeSection, setActiveSection] = useState<SettingsSection>("capture");
  const [imagePath, setImagePath] = useState("");
  const [imageFps, setImageFps] = useState(15);
  const [configBusy, setConfigBusy] = useState<string | null>(null);
  const capabilityRequestId = useRef(0);

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
    async (section: string, key: string, value: string | number) => {
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
  const controlConfig = getNestedRecord(runtime?.config, "control");
  const roiSize = typeof roiConfig?.size === "number" ? roiConfig.size : 640;
  const activeSource = String(getNestedRecord(runtime?.config, "source")?.default ?? runtime?.source ?? "null");
  const readNumber = (section: Record<string, unknown> | null, key: string, fallback: number) => {
    const value = section?.[key];
    return typeof value === "number" ? value : fallback;
  };
  const readString = (section: Record<string, unknown> | null, key: string, fallback: string) => {
    const value = section?.[key];
    return typeof value === "string" ? value : fallback;
  };

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

        {activeSection === "capture" ? (
          <>
            <section className="setup-card">
              <div className="section-title">1. 采集输入源</div>
              <div className="hint">采集是唯一的启动入口；采集有输出后，推理默认消费最新 RoiFrame。</div>
              <div className="source-mode-grid">
                <div className={activeSource === "capture" ? "source-mode active" : "source-mode"}>
                  <strong>采集卡</strong>
                  <span>{capture?.available ? formatProfile(capture) : "未启动，选择 Profile 后启动"}</span>
                </div>
                <div className={activeSource === "image" || activeSource.startsWith("image:") ? "source-mode active" : "source-mode"}>
                  <strong>图片输入</strong>
                  <span>用于模型和 UI 链路测试，不占用采集卡。</span>
                </div>
              </div>
            </section>

            <section className="setup-card">
              <div className="section-title">2. 采集卡能力</div>
              <div className="hint">按像素格式分组，组内优先展示最高 FPS；完整能力折叠在 More。</div>
              <div className="format-choice-grid">
                {groups.map((group) => {
                  const pill = getFormatPill(group.pixel_format);
                  return (
                    <button
                      className={
                        group.pixel_format === selectedFormat
                          ? "format-choice active"
                          : "format-choice"
                      }
                      key={group.pixel_format}
                      onClick={() => setSelectedFormat(group.pixel_format)}
                      type="button"
                    >
                      <span className="format-key">{group.pixel_format}</span>
                      <span className="format-detail">{getFormatDescription(group.pixel_format)}</span>
                      <span className={`pill ${pill.tone}`}>{pill.label}</span>
                    </button>
                  );
                })}
              </div>
            </section>

            <section className="setup-card">
              <div className="section-title">3. 分辨率 / 帧率</div>
              <div className="hint">只展示当前格式里最有价值的组合，默认按 FPS 从高到低排序。</div>
              {visibleProfiles.length > 0 ? (
                <div className="profile-choice-grid">
                  {visibleProfiles.map((row) => {
                    const id = `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`;
                    const active =
                      capture?.profile?.pixel_format?.toUpperCase() === row.pixel_format &&
                      capture.profile.width === row.width &&
                      capture.profile.height === row.height &&
                      capture.profile.fps === row.fps;
                    return (
                      <button
                        className={active ? "profile-choice active" : "profile-choice"}
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
                        <span>
                          <strong>{getProfileTitle(row)}</strong>
                          <small>
                            {row.width}x{row.height} {row.pixel_format}
                          </small>
                        </span>
                        <em>{applying === id ? "应用中" : getProfileTag(row)}</em>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <EmptyState
                  title="尚未读取采集能力"
                  detail="点击刷新采集卡信息，读取 /dev/video0 支持的格式、分辨率和帧率。"
                  command="python3 -m novasight doctor camera --device /dev/video0"
                />
              )}
              <details className="advanced-capability-block">
                <summary>More：显示全部格式内排序后的能力组合</summary>
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
            </section>

            <section className="setup-card">
              <div className="section-title">4. 图片输入源</div>
              <div className="hint">用于离线验证模型、预览和配置链路；真机采集调试时可随时切回采集卡。</div>
              <div className="image-source-row">
                <input
                  className="home-inline-input"
                  placeholder="/home/nvidia/NovaSight/data/sample.jpg"
                  value={imagePath}
                  onChange={(event) => setImagePath(event.target.value)}
                />
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
                <button
                  className="button"
                  type="button"
                  disabled={!imagePath.trim() || applying !== null}
                  onClick={() => void applyImageSource()}
                >
                  {applying === "image-source" ? "切换中" : "切换到图片输入"}
                </button>
              </div>
            </section>

            <section className="advanced-summary">
              <label className="device-input">
                <span>设备</span>
                <input value={device} onChange={(event) => setDevice(event.target.value)} />
              </label>
              <button
                className="button"
                disabled={applying !== null}
                onClick={() => applyPreference("auto_high_fps", "高帧率")}
                type="button"
              >
                自动高帧率
              </button>
              <button
                className="button"
                disabled={applying !== null}
                onClick={() => applyPreference("auto_low_latency", "低延迟")}
                type="button"
              >
                自动低延迟
              </button>
              <span>采集参数修改会重新打开采集源；性能指挥台只负责观察结果。</span>
            </section>
          </>
        ) : null}

        {activeSection === "inference" ? (
          <>
        <section className="setup-card">
          <div className="section-title">1. 推理输入</div>
          <div className="hint">采集流存在时，推理默认消费最新 RoiFrame；推理状态由采集帧驱动。</div>
          <div className="inference-setting-grid">
            <div className="pipeline-step">
              <b>RoiFrame</b>
              <span>{roiSize}x{roiSize} · {capture?.profile?.pixel_format ?? "未选择"}</span>
            </div>
            <div className="pipeline-step">
              <b>采集来源</b>
              <span>{formatProfile(capture)}</span>
            </div>
            <div className="pipeline-step">
              <b>推理策略</b>
              <span>最新帧 · 跳过旧帧</span>
            </div>
          </div>
        </section>

        <section className="setup-card">
          <div className="section-title">2. 检测参数</div>
          <div className="hint">这些参数后续会接入运行配置热更新；当前先固定展示目标结构。</div>
          <div className="inference-setting-grid">
            <div className="pipeline-step">
              <b>置信度阈值</b>
              <span>0.25</span>
            </div>
            <div className="pipeline-step">
              <b>NMS 阈值</b>
              <span>0.45</span>
            </div>
            <div className="pipeline-step">
              <b>类别策略</b>
              <span>使用模型类别</span>
            </div>
          </div>
        </section>

        <section className="setup-card">
          <div className="section-title">3. 推理后端</div>
          <div className="hint">TensorRT 闭环接入后，这里会显示 engine、binding、输入尺寸和执行 provider。</div>
          <div className="capture-pipeline">
            <div className="pipeline-step">
              <b>模型</b>
              <span>{runtime?.active_model?.project?.name ?? "未发布模型"}</span>
            </div>
            <div className="pipeline-step">
              <b>产物</b>
              <span>{runtime?.active_model?.artifact?.kind ?? "未绑定"}</span>
            </div>
            <div className="pipeline-step">
              <b>执行</b>
              <span>{capture?.available ? "等待采集帧驱动" : "采集未启动"}</span>
            </div>
            <div className="pipeline-step">
              <b>输出</b>
              <span>{runtime?.executor?.selected ?? "未选择"}</span>
            </div>
          </div>
        </section>
          </>
        ) : null}

        {activeSection === "algorithm" ? (
          <>
            <section className="setup-card">
              <div className="section-title">1. 算法选择</div>
              <div className="hint">控制算法参数实时同步到运行配置，FOV 会影响预览层绘制。</div>
              <div className="algorithm-choice-row">
                {["pid", "predictive"].map((strategy) => (
                  <button
                    className={
                      readString(controlConfig, "strategy", "pid") === strategy
                        ? "algorithm-choice active"
                        : "algorithm-choice"
                    }
                    key={strategy}
                    type="button"
                    onClick={() => void updateRuntimeField("control", "strategy", strategy)}
                  >
                    <strong>{strategy === "pid" ? "PID 平滑追踪" : "预测追踪"}</strong>
                    <span>{strategy === "pid" ? "支持 X/Y 分轴参数" : "基于目标位移提前量"}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="setup-card">
              <div className="section-title">2. FOV 与输出限制</div>
              <div className="settings-form-grid">
                <NumberField
                  label="FOV 比例"
                  value={readNumber(controlConfig, "fov_ratio", 0.28)}
                  busy={configBusy === "control.fov_ratio"}
                  min={0.01}
                  max={1}
                  step={0.01}
                  onCommit={(value) => void updateRuntimeField("control", "fov_ratio", value)}
                />
                <NumberField
                  label="X 限幅"
                  value={readNumber(controlConfig, "max_abs_dx", 120)}
                  busy={configBusy === "control.max_abs_dx"}
                  min={0}
                  step={1}
                  onCommit={(value) => void updateRuntimeField("control", "max_abs_dx", value)}
                />
                <NumberField
                  label="Y 限幅"
                  value={readNumber(controlConfig, "max_abs_dy", 120)}
                  busy={configBusy === "control.max_abs_dy"}
                  min={0}
                  step={1}
                  onCommit={(value) => void updateRuntimeField("control", "max_abs_dy", value)}
                />
                <NumberField
                  label="最低置信度"
                  value={readNumber(controlConfig, "min_confidence", 0)}
                  busy={configBusy === "control.min_confidence"}
                  min={0}
                  max={1}
                  step={0.01}
                  onCommit={(value) => void updateRuntimeField("control", "min_confidence", value)}
                />
              </div>
            </section>

            <section className="setup-card">
              <div className="section-title">3. PID 参数</div>
              <div className="settings-form-grid dense">
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
            </section>
          </>
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
