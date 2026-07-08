import { useEffect, useMemo, useState, type CSSProperties } from "react";

import {
  type CaptureState,
  type HealthResponse,
  type RuntimeConfig,
  type RuntimeState,
  type Statistics,
  getRuntimeConfig,
  streamUrl,
  updateRuntimeConfig
} from "../../api";
import { StatusIndicator } from "../../components/ui";
import { statusTone } from "../shared/format";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";

type DashboardViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  runtimeConfig?: RuntimeConfig | null;
  loading: boolean;
  errors: {
    health?: string;
    runtime?: string;
  };
  onRefresh: () => void | Promise<void>;
};

type Tone = "good" | "info" | "warn" | "bad";
type DetectionOverlay = {
  className: string;
  score: number;
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
};

function formatNumber(value: number | undefined, digits = 1): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function readStatistics(runtime: RuntimeState | null): Statistics {
  const stats = runtime?.statistics ?? runtime?.capture?.statistics;
  return {
    capture_counter: stats?.capture_counter ?? 0,
    inference_counter: stats?.inference_counter ?? 0,
    dropped_counter: stats?.dropped_counter ?? runtime?.capture?.frames_dropped ?? 0,
    skipped_counter: stats?.skipped_counter ?? 0,
    capture_fps: stats?.capture_fps ?? runtime?.capture?.fps_capture ?? 0,
    inference_fps: stats?.inference_fps ?? 0,
    queue_latency: stats?.queue_latency ?? 0,
    inference_latency: stats?.inference_latency ?? 0,
    e2e_latency: stats?.e2e_latency ?? 0
  };
}

function captureMode(capture: CaptureState | undefined): string {
  if (!capture?.profile) {
    return "未打开";
  }
  const { pixel_format, height, fps } = capture.profile;
  const size = height >= 2160 ? "4K" : height >= 1440 ? "2K" : "1K";
  return `${pixel_format} · ${size}${fps} · ROI`;
}

function metricTone(value: number, target: number): Tone {
  if (target <= 0 || value <= 0) return "warn";
  const ratio = value / target;
  if (ratio >= 0.88) return "good";
  if (ratio >= 0.6) return "info";
  return "warn";
}

function MetricTile({
  label,
  value,
  detail,
  tone
}: {
  label: string;
  value: string;
  detail: string;
  tone: Tone;
}) {
  return (
    <div className="home-metric">
      <div className="home-metric-key">{label}</div>
      <div className={`home-metric-value ${tone}`}>{value}</div>
      <div className="home-metric-sub">{detail}</div>
    </div>
  );
}

function ConsumerRow({
  icon,
  title,
  detail,
  enabled,
  busy,
  onToggle
}: {
  icon: string;
  title: string;
  detail: string;
  enabled: boolean;
  busy?: boolean;
  onToggle: (enabled: boolean) => void;
}) {
  return (
    <div className="consumer-row">
      <div className="consumer-icon">{icon}</div>
      <div>
        <h4>{title}</h4>
        <p>{detail}</p>
      </div>
      <button
        className={enabled ? "consumer-switch on" : "consumer-switch"}
        type="button"
        aria-pressed={enabled}
        disabled={busy}
        onClick={() => onToggle(!enabled)}
      >
        <span />
      </button>
    </div>
  );
}

function useDisplayTick(intervalMs = 250): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setTick((value) => value + 1), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return tick;
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function readNestedBoolean(
  config: unknown,
  section: string,
  key: string,
  fallback: boolean
): boolean {
  const value = asRecord(asRecord(config)[section])[key];
  return typeof value === "boolean" ? value : fallback;
}

function readNestedNumber(
  config: unknown,
  section: string,
  key: string,
  fallback: number
): number {
  const value = asRecord(asRecord(config)[section])[key];
  return typeof value === "number" ? value : fallback;
}

function readNumberRecord(value: Record<string, unknown>, key: string): number | null {
  const item = value[key];
  return typeof item === "number" && Number.isFinite(item) ? item : null;
}

function readStringRecord(value: Record<string, unknown>, key: string): string {
  const item = value[key];
  return typeof item === "string" ? item : "";
}

function readDetectionItems(value: unknown): DetectionOverlay[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    const record = asRecord(item);
    const displayBox = asRecord(record.display_box);
    const roiBox = asRecord(record.roi_box);
    const box =
      Object.keys(displayBox).length > 0
        ? displayBox
        : Object.keys(roiBox).length > 0
          ? roiBox
          : record;
    const x = readNumberRecord(box, "x");
    const y = readNumberRecord(box, "y");
    const w = readNumberRecord(box, "w");
    const h = readNumberRecord(box, "h");
    const score = readNumberRecord(record, "score");
    const cx = readNumberRecord(box, "cx");
    const cy = readNumberRecord(box, "cy");
    if (
      x === null ||
      y === null ||
      w === null ||
      h === null ||
      score === null ||
      cx === null ||
      cy === null ||
      w <= 0 ||
      h <= 0
    ) {
      return [];
    }
    const className =
      readStringRecord(record, "class_name") ||
      String(readNumberRecord(record, "class_id") ?? "");
    return [
      {
        className,
        score,
        x,
        y,
        w,
        h,
        cx,
        cy
      }
    ];
  });
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, value));
}

function detectionStyle(
  detection: DetectionOverlay,
  roiOffsetX: number,
  roiOffsetY: number,
  roiWidth: number,
  roiHeight: number,
  coordinateSpace: string
): CSSProperties {
  const x = coordinateSpace === "roi" ? detection.x : detection.x - roiOffsetX;
  const y = coordinateSpace === "roi" ? detection.y : detection.y - roiOffsetY;
  const left = (x / roiWidth) * 100;
  const top = (y / roiHeight) * 100;
  const width = (detection.w / roiWidth) * 100;
  const height = (detection.h / roiHeight) * 100;
  return {
    left: `${clampPercent(left)}%`,
    top: `${clampPercent(top)}%`,
    width: `${clampPercent(width)}%`,
    height: `${clampPercent(height)}%`
  };
}

function pointStyle(x: number | null, y: number | null, width: number, height: number): CSSProperties | null {
  if (x === null || y === null || width <= 0 || height <= 0) {
    return null;
  }
  return {
    left: `${clampPercent((x / width) * 100)}%`,
    top: `${clampPercent((y / height) * 100)}%`
  };
}

function isSelectedDetection(
  detection: DetectionOverlay,
  targetCx: number | null,
  targetCy: number | null
): boolean {
  if (targetCx === null || targetCy === null) {
    return false;
  }
  return Math.abs(detection.cx - targetCx) <= 2 && Math.abs(detection.cy - targetCy) <= 2;
}

export function DashboardView({
  health,
  runtime,
  runtimeConfig,
  loading,
  errors,
  onRefresh
}: DashboardViewProps) {
  const displayTick = useDisplayTick();
  const capture = runtime?.capture;
  const stats = useMemo(() => readStatistics(runtime), [runtime, displayTick]);
  const targetFps = capture?.profile?.fps ?? 120;
  const configSource = runtimeConfig ?? runtime?.config;
  const runtimeInference = asRecord(runtime?.inference);
  const deepstreamRuntimeSelected = readStringRecord(runtimeInference, "selected") === "deepstream";
  const runtimeMainlineStatus = getRuntimeMainlineStatus(runtime);
  const runtimeMainlineRunning = runtimeMainlineStatus.running;
  const captureMainRunning = deepstreamRuntimeSelected ? runtimeMainlineRunning : capture?.available === true;
  const captureProfileConfigured = capture?.available === true || Boolean(capture?.profile);
  const captureStateLabel = deepstreamRuntimeSelected
    ? runtimeMainlineStatus.failed
      ? "主链故障"
      : runtimeMainlineRunning
        ? runtimeMainlineStatus.hasRuntimeConsumption
          ? "主链已消费"
          : runtimeMainlineStatus.hasInferenceSignal
            ? "等待消费"
            : "等待推理输出"
        : captureProfileConfigured
          ? "主链待启动"
          : "未启动"
    : captureMainRunning
      ? "运行中"
      : captureProfileConfigured
        ? "已配置"
        : "未启动";
  const roiSize = readNestedNumber(configSource, "roi", "size", 640);
  const configVersion =
    typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const modelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const e2eText = stats.e2e_latency > 0 ? `${formatNumber(stats.e2e_latency, 1)}ms` : "--";
  const [consumerBusy, setConsumerBusy] = useState<string | null>(null);
  const previewEnabled = readNestedBoolean(configSource, "consumers", "preview", true);
  const inferenceEnabled = readNestedBoolean(configSource, "consumers", "inference", true);
  const recordingEnabled = readNestedBoolean(configSource, "consumers", "recording", false);
  const vision = asRecord(runtime?.vision);
  const inferenceTrace = asRecord(vision.inference);
  const target = asRecord(vision.target);
  const control = asRecord(vision.control);
  const targetName =
    typeof target.class_name === "string" && target.class_name ? target.class_name : "";
  const targetScore = readNumberRecord(target, "score");
  const targetCx = readNumberRecord(target, "cx");
  const targetCy = readNumberRecord(target, "cy");
  const aimX = readNumberRecord(target, "aim_x");
  const aimY = readNumberRecord(target, "aim_y");
  const controlDx = readNumberRecord(control, "dx");
  const controlDy = readNumberRecord(control, "dy");
  const willEmit = control.will_emit === true;
  const inferenceReason =
    typeof vision.inference_reason === "string" ? vision.inference_reason : "";
  const inferenceRan = inferenceTrace.ran === true;
  const inferenceAvailable = inferenceTrace.available === true;
  const inferenceFrameId = readNumberRecord(inferenceTrace, "frame_id");
  const rawDetections = readNumberRecord(inferenceTrace, "raw_detections");
  const mappedDetections = readNumberRecord(inferenceTrace, "mapped_detections");
  const inputWidth = readNumberRecord(inferenceTrace, "input_width");
  const inputHeight = readNumberRecord(inferenceTrace, "input_height");
  const inputImageWidth = readNumberRecord(inferenceTrace, "input_image_width");
  const inputImageHeight = readNumberRecord(inferenceTrace, "input_image_height");
  const inputFormat =
    typeof inferenceTrace.input_pixel_format === "string" ? inferenceTrace.input_pixel_format : "--";
  const inferenceTraceReason =
    typeof inferenceTrace.reason === "string" ? inferenceTrace.reason : inferenceReason;
  const inferenceDebug = asRecord(inferenceTrace.debug);
  const debugEngine = readStringRecord(inferenceDebug, "engine");
  const debugOutputName = readStringRecord(inferenceDebug, "output_name");
  const debugOutputDtype = readStringRecord(inferenceDebug, "output_dtype");
  const debugOutputShape = Array.isArray(inferenceDebug.output_shape)
    ? inferenceDebug.output_shape.join("x")
    : "";
  const detections = readDetectionItems(vision.detection_items);
  const roiOffsetX = readNumberRecord(inferenceTrace, "roi_offset_x") ?? 0;
  const roiOffsetY = readNumberRecord(inferenceTrace, "roi_offset_y") ?? 0;
  const detectionCoordinateSpace = readStringRecord(inferenceTrace, "detection_coordinate_space") || "source";
  const overlayWidth = inputWidth ?? roiSize;
  const overlayHeight = inputHeight ?? roiSize;
  const aimPointStyle = pointStyle(aimX, aimY, overlayWidth, overlayHeight);

  async function updateConsumer(key: "preview" | "inference" | "recording", enabled: boolean) {
    if (consumerBusy) {
      return;
    }
    setConsumerBusy(key);
    try {
      const nextConfig = await getRuntimeConfig();
      (nextConfig as Record<string, unknown>).consumers = {
        ...asRecord(nextConfig.consumers),
        [key]: enabled
      };
      await updateRuntimeConfig(nextConfig);
      await onRefresh();
    } finally {
      setConsumerBusy(null);
    }
  }

  return (
    <div className="home-workspace">
      <aside className="home-side">
        <section className="home-card">
          <div className="home-card-head">
            <div>
              <div className="home-card-title">采集状态</div>
              <div className="home-card-desc">性能页只展示当前链路状态，采集参数进入基础设置调整。</div>
            </div>
          </div>

          <div className="home-section">
            <div className="home-field">
              <div className="home-field-label">
                <span>采集源</span>
                <span>{captureStateLabel}</span>
              </div>
              <div className="home-selectbox">
                <span>{capture?.device ?? "/dev/video0"}</span>
                <span>{deepstreamRuntimeSelected ? "deepstream" : capture?.backend ?? "未打开"}</span>
              </div>
              <div className="home-tiny">
                {deepstreamRuntimeSelected && runtimeMainlineRunning
                  ? runtimeMainlineStatus.progressSummary
                  : "启动、停止、格式和分辨率选择统一在基础设置 / 采集设置。"}
              </div>
            </div>

            <div className="home-field">
              <div className="home-field-label">
                <span>当前 Profile</span>
                <span>{capture?.profile?.selection_reason ?? "等待采集"}</span>
              </div>
              <div className="home-status-line">
                <strong>{capture?.profile?.pixel_format ?? "--"}</strong>
                <span>
                  {capture?.profile
                    ? `${capture.profile.width}x${capture.profile.height} @ ${capture.profile.fps}fps`
                    : "未选择"}
                </span>
              </div>
            </div>

            <div className="home-field">
              <div className="home-field-label">
                <span>RoiFrame 输出</span>
                <span>GPU 路线</span>
              </div>
              <div className="home-chips">
                {[320, 416, 640, 960].map((size) => (
                  <span className={roiSize === size ? "home-chip active" : "home-chip"} key={size}>
                    {size}
                  </span>
                ))}
              </div>
              <div className="home-tiny">
                目标对象不是 CPU Mat，而是可被多路消费的 RoiFrame / GpuFrameView。
              </div>
            </div>
          </div>
        </section>

        <section className="home-card">
          <div className="home-card-head">
            <div>
              <div className="home-card-title">输出对象</div>
              <div className="home-card-desc">当前管线产出 RoiFrame，不直接绑定某个消费端。</div>
            </div>
          </div>
          <div className="home-section">
            <div className="home-field">
              <div className="home-field-label">
                <span>RoiFrame</span>
                <span>#{stats.capture_counter || "--"}</span>
              </div>
              <div className="home-tiny">
                width={roiSize} · height={roiSize}<br />
                capture_ts_ns · frame_id · gpu_ptr · pitch<br />
                可被推理、推流、录制同时消费。
              </div>
            </div>
          </div>
        </section>
      </aside>

      <section className="home-main-view">
        <div className="home-pipeline-strip">
          <div className="home-mini-node">
            <div className="k">采集输入</div>
            <div className="v">{capture?.profile ? `${capture.profile.width}x${capture.profile.height} · ${capture.profile.fps}fps` : "等待采集"}</div>
            <div className="s">{capture?.profile?.pixel_format ?? "未选择"} · {deepstreamRuntimeSelected ? "DeepStream 启动时打开" : capture?.backend ?? "未打开"}</div>
          </div>
          <div className="home-arrow">→</div>
          <div className="home-mini-node">
            <div className="k">GPU 处理</div>
            <div className="v">Decode / Convert / Crop</div>
            <div className="s">目标：避免 CPU Copy</div>
          </div>
          <div className="home-arrow">→</div>
          <div className="home-mini-node">
            <div className="k">统一输出</div>
            <div className="v">RoiFrame {roiSize}x{roiSize}</div>
            <div className="s">供多路消费者使用</div>
          </div>
        </div>

        <section className="home-card home-video-card">
          <div className="home-video-head">
            <div>
              <div className="home-card-title">实时画面</div>
              <div className="home-card-desc">主页主角是画面与 ROI，而不是采集卡能力表。</div>
            </div>
            <div className="home-video-actions">
              <StatusIndicator tone={statusTone(health?.ok)}>
                {loading ? "后端检查中" : health?.ok ? "后端已连接" : "后端离线"}
              </StatusIndicator>
              <button className="button compact-button" type="button" onClick={onRefresh}>
                刷新状态
              </button>
            </div>
          </div>

          <div
            className="home-video"
            style={{ "--roi-display-size": `${roiSize}px` } as CSSProperties}
          >
            {capture?.available && previewEnabled && !deepstreamRuntimeSelected ? (
              <img alt="实时采集画面" src={streamUrl(configVersion, configVersion)} />
            ) : null}
            <div className="home-video-grid" />
            <div className="home-video-scan" />
            <div className="home-detection-layer" aria-hidden="true">
              {detections.map((detection, index) => {
                const selected = isSelectedDetection(detection, targetCx, targetCy);
                return (
                  <div
                    className="home-detection-box"
                    key={`${detection.className}-${index}-${detection.x}-${detection.y}`}
                    style={detectionStyle(
                      detection,
                      roiOffsetX,
                      roiOffsetY,
                      overlayWidth,
                      overlayHeight,
                      detectionCoordinateSpace
                    )}
                  >
                    <span>
                      {selected ? "当前 " : ""}{detection.className || "目标"} {detection.score.toFixed(2)}
                    </span>
                  </div>
                );
              })}
              {aimPointStyle ? <div className="home-aim-point" style={aimPointStyle} /> : null}
            </div>
            <div className="home-hud home-hud-left">
              <span>预览 {deepstreamRuntimeSelected ? "Tensor Overlay" : previewEnabled ? `${capture?.preview_target_fps ?? 30}fps` : "已关闭"}</span>
              <span>{captureMode(capture)}</span>
              <span>ROI {roiSize}</span>
              <span>GPU 路线</span>
            </div>
            <div className="home-hud home-hud-right">
              <span>预览不影响推理链路</span>
            </div>
          </div>
        </section>

        <div className="home-bottom-timeline">
          <div className="home-stage">
            <div className="k">采集等待</div>
            <div className="v">{formatNumber(capture?.capture_wait_ms, 2)}ms</div>
          </div>
          <div className="home-stage">
            <div className="k">帧间隔</div>
            <div className="v">{formatNumber(capture?.frame_period_ms, 2)}ms</div>
          </div>
          <div className="home-stage">
            <div className="k">采集计数</div>
            <div className="v">{stats.capture_counter}</div>
          </div>
          <div className="home-stage">
            <div className="k">推理计数</div>
            <div className="v">{stats.inference_counter}</div>
          </div>
          <div className="home-stage">
            <div className="k">端到端延迟</div>
            <div className="v">{e2eText}</div>
          </div>
        </div>
      </section>

      <aside className="home-metrics">
        <section className="home-card">
          <div className="home-card-head">
            <div>
              <div className="home-card-title">运行反馈</div>
              <div className="home-card-desc">显示每 250ms 刷新，统计最近 1 秒窗口的数据。</div>
            </div>
          </div>
          <div className="home-metric-grid">
            <MetricTile
              label="采集 FPS"
              value={formatNumber(stats.capture_fps)}
              detail={`目标 ${targetFps}`}
              tone={metricTone(stats.capture_fps, targetFps)}
            />
            <MetricTile
              label="推理 FPS"
              value={stats.inference_fps > 0 ? formatNumber(stats.inference_fps) : "--"}
              detail="最新帧模式"
              tone="info"
            />
            <MetricTile
              label="E2E 延迟"
              value={e2eText}
              detail="采集到控制发出"
              tone={stats.e2e_latency > 0 && stats.e2e_latency <= 30 ? "good" : "warn"}
            />
            <MetricTile
              label="丢帧"
              value={String(stats.dropped_counter)}
              detail="最近 1 秒采集丢帧"
              tone={stats.dropped_counter > 0 ? "warn" : "good"}
            />
            <MetricTile
              label="跳帧"
              value={String(stats.skipped_counter)}
              detail="最近 1 秒推理跳过旧帧"
              tone={stats.skipped_counter > 0 ? "warn" : "good"}
            />
            <MetricTile
              label="GPU 内存"
              value="目标"
              detail="NVMM/CUDA 路径"
              tone="good"
            />
          </div>
          <div className="home-notice">
            预览流建议限制为 15-30fps；推理链路继续消费 RoiFrame 或最新帧，不让 UI 预览拖慢核心链路。
          </div>
          <div className={inferenceAvailable ? "home-notice good" : "home-notice"}>
            <strong>推理结果</strong><br />
            状态：{inferenceRan ? (inferenceAvailable ? "已执行" : "执行失败") : "尚未执行"}
            {inferenceFrameId !== null ? ` · 帧 #${inferenceFrameId}` : ""}
            <br />
            输入：{inputWidth !== null && inputHeight !== null ? `${inputWidth}x${inputHeight}` : "--"} · {inputFormat}
            {inputImageWidth !== null && inputImageHeight !== null ? ` · 缓冲区 ${inputImageWidth}x${inputImageHeight}` : ""}
            <br />
            检测：raw {rawDetections ?? 0} · mapped {mappedDetections ?? 0}
            {inferenceTraceReason ? <><br />原因：{inferenceTraceReason}</> : null}
            {debugEngine || debugOutputShape ? (
              <>
                <br />
                调试：{debugEngine || "runtime"}
                {debugOutputName ? ` · ${debugOutputName}` : ""}
                {debugOutputShape ? ` · ${debugOutputShape}` : ""}
                {debugOutputDtype ? ` · ${debugOutputDtype}` : ""}
              </>
            ) : null}
          </div>
          <div className="home-notice">
            <strong>目标与控制量</strong><br />
            {targetName
              ? `目标 ${targetName} · ${(targetScore ?? 0).toFixed(2)} · (${formatNumber(targetCx ?? undefined, 0)}, ${formatNumber(targetCy ?? undefined, 0)})`
              : inferenceReason
                ? `暂无推理目标：${inferenceReason}`
                : "暂无推理目标。"}
            <br />
            {controlDx !== null && controlDy !== null
              ? `控制量 dx=${formatNumber(controlDx, 1)} · dy=${formatNumber(controlDy, 1)} · ${willEmit ? "允许输出" : "等待硬件触发"}`
              : "暂无控制量。"}
          </div>
          {errors.health || errors.runtime ? (
            <div className="home-notice bad">{errors.health ?? errors.runtime}</div>
          ) : null}
        </section>

        <section className="home-card">
          <div className="home-card-head">
            <div>
              <div className="home-card-title">消费者</div>
              <div className="home-card-desc">RoiFrame 可以被多个模块订阅。</div>
            </div>
          </div>
          <div className="consumer-list">
            <ConsumerRow
              icon="TRT"
              title="TensorRT 推理"
              detail={inferenceEnabled ? modelName : "已从运行配置关闭"}
              enabled={inferenceEnabled}
              busy={consumerBusy === "inference"}
              onToggle={(enabled) => void updateConsumer("inference", enabled)}
            />
            <ConsumerRow
              icon="WEB"
              title="浏览器预览"
              detail={`${capture?.preview_target_fps ?? 30}fps 降采样推流`}
              enabled={previewEnabled}
              busy={consumerBusy === "preview"}
              onToggle={(enabled) => void updateConsumer("preview", enabled)}
            />
            <ConsumerRow
              icon="REC"
              title="录制回放"
              detail={recordingEnabled ? "保存帧生命周期与结果" : "当前未写入回放文件"}
              enabled={recordingEnabled}
              busy={consumerBusy === "recording"}
              onToggle={(enabled) => void updateConsumer("recording", enabled)}
            />
          </div>
        </section>
      </aside>
    </div>
  );
}
