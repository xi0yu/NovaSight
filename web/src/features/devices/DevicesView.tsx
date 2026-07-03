import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  type CaptureCapabilitiesResponse,
  type CaptureCapability,
  type CaptureSelectPayload,
  type CaptureState,
  type RuntimeState,
  getCaptureCapabilities,
  selectCaptureProfile,
  stopCapture,
  streamUrl
} from "../../api";
import { VideoPanel } from "../../components/studio";
import { EmptyState, InlineError } from "../../components/ui";
import { formatProfile, getErrorMessage } from "../shared/format";
import { InferenceControl } from "../shared/InferenceControl";

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
  canControlRuntime: boolean;
  runtime: RuntimeState | null;
  error: string | undefined;
  onRuntimeRefresh: () => Promise<void>;
  onInferenceControlCommand: (action: "start" | "stop") => void;
  runtimeCommandBusy: boolean;
};

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

function CapabilityTable({
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
    <div className="capability-groups">
      {groups.map((group, index) => (
        <details className="capability-group" key={group.pixel_format} open={index < 2}>
          <summary>
            <span className="mono">{group.pixel_format}</span>
            <span>{group.choices.length} 组配置</span>
            <strong>{getFormatSummary(group)}</strong>
          </summary>
          <div className="table-wrap capability-table">
            <table>
              <thead>
                <tr>
                  <th>分辨率</th>
                  <th>帧率</th>
                  <th>建议</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {group.choices.map((row) => {
                  const id = `${row.pixel_format}-${row.width}-${row.height}-${row.fps}`;
                  return (
                    <tr key={id}>
                      <td className="mono">
                        {row.width}x{row.height}
                      </td>
                      <td className="mono">{row.fps}</td>
                      <td>
                        <span className={`hint-chip ${getCapabilityTone(row)}`}>
                          {getCapabilityHint(row)}
                        </span>
                      </td>
                      <td>
                        <button
                          className="button compact-button"
                          disabled={applying !== null || disabled}
                          onClick={() => onApply(row)}
                          type="button"
                        >
                          {applying === id ? "应用中" : "应用并启动"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
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
  const formatRank =
    choice.pixel_format === "MJPG"
      ? 0
      : choice.pixel_format === "NV12"
        ? 1
        : choice.pixel_format === "YUYV"
          ? 2
          : 3;
  const sizeRank = choice.width === 1920 && choice.height === 1080 ? 0 : 1;
  return formatRank * 100000000 - choice.fps * 100000 + sizeRank * 10000 - choice.width * choice.height;
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

function CaptureFeedback({
  capture,
  runtimeRunning,
  roiSize,
  streamKey,
  configVersion
}: {
  capture: CaptureState | undefined;
  runtimeRunning: boolean;
  roiSize: number;
  streamKey: number;
  configVersion: number;
}) {
  return (
    <aside className="capture-feedback">
      <div>
        <div className="section-title">实时反馈</div>
        <div className="hint">配置不是终点，跑起来后的结果才是主界面重点。</div>
      </div>
      <VideoPanel
        available={Boolean(capture?.available)}
        caption={`ROI ${roiSize}x${roiSize}`}
        className="capture-preview-card"
        profile={formatProfile(capture)}
        running={runtimeRunning}
        src={capture?.available ? streamUrl(streamKey, configVersion) : undefined}
      />
      <div className="small-card">
        <div className="metric-row">
          <span>采集 FPS</span>
          <b>{capture ? capture.fps_capture.toFixed(1) : "--"}</b>
        </div>
        <div className="metric-row">
          <span>预览 FPS</span>
          <b>{capture ? capture.preview_fps.toFixed(1) : "--"}</b>
        </div>
        <div className="metric-row">
          <span>读取等待</span>
          <b>{capture ? `${capture.capture_wait_ms.toFixed(2)} ms` : "--"}</b>
        </div>
        <div className="metric-row">
          <span>丢帧</span>
          <b>{capture?.frames_dropped ?? 0}</b>
        </div>
      </div>
      <div className="small-card capture-advice">
        {capture?.last_error
          ? capture.last_error
          : "当前模式适合验证采集吞吐。若推理或 UI 反馈跟不上，优先切换到 1K120 NV12 或降低 ROI 输入。"}
      </div>
      <div className="footer-note">
        能力列表会缓存到当前页面；刷新按钮用于重新读取采集卡变化。
      </div>
    </aside>
  );
}

export function DevicesView({
  canControlRuntime,
  runtime,
  error,
  onRuntimeRefresh,
  onInferenceControlCommand,
  runtimeCommandBusy
}: DevicesViewProps) {
  const [device, setDevice] = useState(runtime?.capture?.device ?? "/dev/video0");
  const [capabilities, setCapabilities] = useState<CaptureCapabilitiesResponse | null>(null);
  const [captureError, setCaptureError] = useState<string | undefined>(error);
  const [loadingCaps, setLoadingCaps] = useState(false);
  const [applying, setApplying] = useState<string | null>(null);
  const [stoppingCapture, setStoppingCapture] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState("MJPG");
  const [streamKey, setStreamKey] = useState(Date.now());
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
        setStreamKey(Date.now());
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
      setStreamKey(Date.now());
      await onRuntimeRefresh();
    } catch (err) {
      setCaptureError(getErrorMessage(err));
      await onRuntimeRefresh();
    } finally {
      setStoppingCapture(false);
    }
  }, [onRuntimeRefresh]);

  const applyPreference = (preference: CaptureSelectPayload["preference"], label: string) =>
    applySelection({ device, preference }, label);

  const capture = runtime?.capture;
  const runtimeRunning = Boolean(runtime?.running);
  const configVersion =
    typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const roiSize =
    typeof runtime?.config?.roi_size === "number" ? runtime.config.roi_size : 640;

  return (
    <div className="capture-setup">
      <main className="capture-setup-main">
        <div className="capture-setup-topbar">
          <div>
            <h2>采集配置</h2>
            <p>先选择目标采集方式，复杂能力已自动折叠。</p>
          </div>
          <div className="panel-actions">
            {canControlRuntime ? (
              <InferenceControl
                busy={runtimeCommandBusy}
                running={runtimeRunning}
                onCommand={onInferenceControlCommand}
              />
            ) : (
              <span className="panel-action-note">当前授权不包含运行控制能力</span>
            )}
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

        <section className="setup-card">
          <div className="section-title">1. 选择采集格式</div>
          <div className="hint">默认只显示主流格式，BGR3 / 特殊尺寸 / 29.97fps 等放到更多里。</div>
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
          <div className="section-title">2. 选择分辨率 / 帧率</div>
          <div className="hint">这里不展示所有枚举，只展示对项目有意义的主流 Profile。</div>
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
            <summary>More：显示非主流分辨率和完整 v4l2 能力枚举</summary>
            <CapabilityTable
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
          <div className="section-title">3. 当前采集链路</div>
          <div className="hint">用户关注的是采集之后能不能顺畅进入推理和控制反馈。</div>
          <div className="capture-pipeline">
            <div className="pipeline-step">
              <b>采集</b>
              <span>{formatProfile(capture)}</span>
            </div>
            <div className="pipeline-step">
              <b>解码/转换</b>
              <span>{capture?.backend ?? "等待打开"}</span>
            </div>
            <div className="pipeline-step">
              <b>推理输入</b>
              <span>ROI {roiSize}</span>
            </div>
            <div className="pipeline-step">
              <b>UI 反馈</b>
              <span>延迟 / FPS / 丢帧</span>
            </div>
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
          <span>原始能力列表已折叠，最近错误会在右侧实时反馈展示。</span>
        </section>
      </main>

      <CaptureFeedback
        capture={capture}
        configVersion={configVersion}
        roiSize={roiSize}
        runtimeRunning={runtimeRunning}
        streamKey={streamKey}
      />
    </div>
  );
}
