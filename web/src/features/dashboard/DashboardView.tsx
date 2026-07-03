import { useEffect, useMemo, useState } from "react";

import {
  type CaptureState,
  type HealthResponse,
  type RuntimeState,
  type Statistics,
  streamUrl
} from "../../api";
import { StatusIndicator } from "../../components/ui";
import { formatProfile, statusTone } from "../shared/format";
import { InferenceControl } from "../shared/InferenceControl";

type DashboardViewProps = {
  canControlRuntime: boolean;
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  loading: boolean;
  errors: {
    health?: string;
    runtime?: string;
  };
  onRefresh: () => void;
  onInferenceControlCommand: (action: "start" | "stop") => void;
  runtimeCommandBusy: boolean;
};

type Tone = "good" | "info" | "warn" | "bad";

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
  enabled
}: {
  icon: string;
  title: string;
  detail: string;
  enabled: boolean;
}) {
  return (
    <div className="consumer-row">
      <div className="consumer-icon">{icon}</div>
      <div>
        <h4>{title}</h4>
        <p>{detail}</p>
      </div>
      <div className={enabled ? "consumer-switch on" : "consumer-switch"} />
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

export function DashboardView({
  canControlRuntime,
  health,
  runtime,
  loading,
  errors,
  onRefresh,
  onInferenceControlCommand,
  runtimeCommandBusy
}: DashboardViewProps) {
  const displayTick = useDisplayTick();
  const capture = runtime?.capture;
  const runtimeRunning = Boolean(runtime?.running);
  const stats = useMemo(() => readStatistics(runtime), [runtime, displayTick]);
  const targetFps = capture?.profile?.fps ?? 120;
  const roiSize =
    typeof runtime?.config?.roi_size === "number" ? runtime.config.roi_size : 640;
  const configVersion =
    typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const modelName = runtime?.active_model?.project?.name ?? "未发布模型";
  const e2eText = stats.e2e_latency > 0 ? `${formatNumber(stats.e2e_latency, 1)}ms` : "--";

  return (
    <div className="home-workspace">
      <aside className="home-side">
        <section className="home-card">
          <div className="home-card-head">
            <div>
              <div className="home-card-title">采集配置</div>
              <div className="home-card-desc">这里只放最常用选择，高级能力在采集页展开。</div>
            </div>
          </div>

          <div className="home-section">
            <div className="home-field">
              <div className="home-field-label">
                <span>采集源</span>
                <span>已缓存</span>
              </div>
              <button className="home-selectbox" type="button" onClick={onRefresh}>
                <span>{capture?.device ?? "/dev/video0"}</span>
                <span>{loading ? "读取中" : "刷新"}</span>
              </button>
              <div className="home-tiny">启动时读取设备能力，手动刷新可重新扫描。</div>
            </div>

            <div className="home-field">
              <div className="home-field-label">
                <span>采集格式</span>
                <span>主流</span>
              </div>
              <div className="home-chips">
                {["MJPG", "NV12", "YUYV", "More"].map((item) => (
                  <span
                    className={capture?.profile?.pixel_format === item ? "home-chip active" : "home-chip"}
                    key={item}
                  >
                    {item}
                  </span>
                ))}
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
            <div className="s">{capture?.profile?.pixel_format ?? "未选择"} · {capture?.backend ?? "未打开"}</div>
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
              {canControlRuntime ? (
                <InferenceControl
                  busy={runtimeCommandBusy}
                  running={runtimeRunning}
                  onCommand={onInferenceControlCommand}
                />
              ) : null}
            </div>
          </div>

          <div className="home-video">
            {capture?.available ? (
              <img alt="实时采集画面" src={streamUrl(configVersion, configVersion)} />
            ) : null}
            <div className="home-video-grid" />
            <div className="home-video-scan" />
            <div className="home-roi" data-label={`ROI ${roiSize}x${roiSize}`} />
            <div className="home-target" />
            <div className="home-hud home-hud-left">
              <span>Preview {capture?.preview_target_fps ?? 30}fps</span>
              <span>{captureMode(capture)}</span>
              <span>ROI {roiSize}</span>
              <span>GPU Path</span>
            </div>
            <div className="home-hud home-hud-right">
              <span>直播感预览</span>
              <span>不影响推理链路</span>
            </div>
          </div>
        </section>

        <div className="home-bottom-timeline">
          <div className="home-stage">
            <div className="k">Capture Wait</div>
            <div className="v">{formatNumber(capture?.capture_wait_ms, 2)}ms</div>
          </div>
          <div className="home-stage">
            <div className="k">Frame Period</div>
            <div className="v">{formatNumber(capture?.frame_period_ms, 2)}ms</div>
          </div>
          <div className="home-stage">
            <div className="k">Capture Counter</div>
            <div className="v">{stats.capture_counter}</div>
          </div>
          <div className="home-stage">
            <div className="k">Inference Counter</div>
            <div className="v">{stats.inference_counter}</div>
          </div>
          <div className="home-stage">
            <div className="k">E2E Latency</div>
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
              label="Capture FPS"
              value={formatNumber(stats.capture_fps)}
              detail={`目标 ${targetFps}`}
              tone={metricTone(stats.capture_fps, targetFps)}
            />
            <MetricTile
              label="Infer FPS"
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
              label="Dropped"
              value={String(stats.dropped_counter)}
              detail="累计采集丢帧"
              tone={stats.dropped_counter > 0 ? "warn" : "good"}
            />
            <MetricTile
              label="Skipped"
              value={String(stats.skipped_counter)}
              detail="推理跳过旧帧"
              tone={stats.skipped_counter > 0 ? "warn" : "good"}
            />
            <MetricTile
              label="GPU Memory"
              value="目标"
              detail="NVMM/CUDA 路径"
              tone="good"
            />
          </div>
          <div className="home-notice">
            预览流建议限制为 15-30fps；推理链路继续消费 RoiFrame 或最新帧，不让 UI 预览拖慢核心链路。
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
            <ConsumerRow icon="TRT" title="TensorRT 推理" detail={modelName} enabled={runtimeRunning} />
            <ConsumerRow icon="WEB" title="浏览器预览" detail={`${capture?.preview_target_fps ?? 30}fps 降采样推流`} enabled={Boolean(capture?.available)} />
            <ConsumerRow icon="REC" title="录制回放" detail="保存帧生命周期与结果" enabled={false} />
          </div>
        </section>
      </aside>
    </div>
  );
}

export { InferenceControl };
