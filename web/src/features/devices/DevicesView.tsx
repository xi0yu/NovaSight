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
import { EmptyState, InlineError, Panel } from "../../components/ui";
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

function CaptureDiagnostics({ capture }: { capture: CaptureState | undefined }) {
  if (!capture) {
    return <EmptyState title="没有采集状态" detail="后端没有返回摄像头诊断信息。" />;
  }

  return (
    <dl className="metric-list">
      <div>
        <dt>设备</dt>
        <dd className="mono">{capture.device}</dd>
      </div>
      <div>
        <dt>后端</dt>
        <dd className="mono">{capture.backend ?? "未选择"}</dd>
      </div>
      <div>
        <dt>配置</dt>
        <dd>{formatProfile(capture)}</dd>
      </div>
      <div>
        <dt>采集帧率</dt>
        <dd className="mono">{capture.fps_capture.toFixed(1)} fps</dd>
      </div>
      <div>
        <dt>预览目标</dt>
        <dd className="mono">{capture.preview_target_fps} fps</dd>
      </div>
      <div>
        <dt>预览输出</dt>
        <dd className="mono">{capture.preview_fps.toFixed(1)} fps</dd>
      </div>
      <div>
        <dt>帧间隔</dt>
        <dd className="mono">{capture.frame_period_ms.toFixed(2)} ms</dd>
      </div>
      <div>
        <dt>读取等待</dt>
        <dd className="mono">{capture.capture_wait_ms.toFixed(2)} ms</dd>
      </div>
      <div>
        <dt>丢帧</dt>
        <dd className="mono">{capture.frames_dropped}</dd>
      </div>
      <div>
        <dt>预览帧</dt>
        <dd className="mono">{capture.preview_output_frames}</dd>
      </div>
      <div>
        <dt>预览丢帧</dt>
        <dd className="mono">{capture.preview_dropped}</dd>
      </div>
      <div>
        <dt>恢复次数</dt>
        <dd className="mono">{capture.recoveries}</dd>
      </div>
      <div>
        <dt>最近错误</dt>
        <dd>{capture.last_error ?? "无"}</dd>
      </div>
    </dl>
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
    <div className="capture-workbench">
      <Panel
        title="采集工作台"
        eyebrow="设备能力与配置切换"
        action={
          <div className="panel-actions">
            {canControlRuntime ? (
              <InferenceControl
                busy={runtimeCommandBusy}
                running={runtimeRunning}
                onCommand={onInferenceControlCommand}
              />
            ) : (
              <span className="panel-action-note">当前授权不包含 runtime 控制能力</span>
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
              {loadingCaps ? "读取中" : "刷新能力"}
            </button>
          </div>
        }
      >
        <InlineError message={captureError} />
        {capture?.available ? (
          <div className="inline-note">
            采集会话已打开。浏览器预览按目标帧率取样，推理控制读取同一个最新帧队列。
          </div>
        ) : (
          <div className="inline-note">
            请选择格式并应用，后端会打开正式采集会话；预览不会单独占用采集卡。
          </div>
        )}
        <div className="capture-toolbar">
          <label className="device-input">
            <span>设备</span>
            <input value={device} onChange={(event) => setDevice(event.target.value)} />
          </label>
          <div className="preference-actions" aria-label="推荐配置">
            <button
              className="button"
              disabled={applying !== null}
              onClick={() => applyPreference("auto_high_fps", "高帧率")}
              type="button"
            >
              高帧率
            </button>
            <button
              className="button"
              disabled={applying !== null}
              onClick={() => applyPreference("auto_low_latency", "低延迟")}
              type="button"
            >
              低延迟
            </button>
            <button
              className="button"
              disabled={applying !== null}
              onClick={() => applyPreference("auto_balanced", "均衡")}
              type="button"
            >
              均衡
            </button>
          </div>
        </div>
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
      </Panel>

      <Panel title="实时预览" eyebrow="MJPEG 采集流">
        <VideoPanel
          available={Boolean(capture?.available)}
          caption={`中心 ROI ${roiSize}x${roiSize}，预览限速输出，采集与推理控制不依赖浏览器帧率`}
          className="live-preview"
          profile={formatProfile(capture)}
          running={runtimeRunning}
          src={capture?.available ? streamUrl(streamKey, configVersion) : undefined}
        />
      </Panel>

      <Panel title="当前配置" eyebrow="采集状态">
        <CaptureDiagnostics capture={capture} />
      </Panel>
    </div>
  );
}
