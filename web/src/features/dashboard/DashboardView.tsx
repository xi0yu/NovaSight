import { EmptyState, InlineError, Panel, StatusIndicator } from "../../components/ui";
import { VideoPanel } from "../../components/studio";
import {
  type CaptureState,
  type ExecutorStatus,
  type HealthResponse,
  type RuntimeState,
  streamUrl
} from "../../api";
import { Field } from "../shared/Field";
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

type CommandMetricProps = {
  label: string;
  value: string;
  unit?: string;
  detail: string;
  ratio: number;
  tone: "good" | "warn" | "bad" | "idle";
};

function clampRatio(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function formatNumber(value: number | undefined, digits = 1): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function captureTargetFps(capture: CaptureState | undefined): number {
  return capture?.profile?.fps ?? 120;
}

function metricTone(
  value: number | undefined,
  target: number,
  mode: "higher" | "lower"
): CommandMetricProps["tone"] {
  if (typeof value !== "number" || !Number.isFinite(value) || target <= 0) {
    return "idle";
  }
  const ratio = value / target;
  if (mode === "higher") {
    if (ratio >= 0.88) return "good";
    if (ratio >= 0.65) return "warn";
    return "bad";
  }
  if (ratio <= 0.55) return "good";
  if (ratio <= 1) return "warn";
  return "bad";
}

function CommandMetric({ label, value, unit, detail, ratio, tone }: CommandMetricProps) {
  return (
    <div className={`command-metric tone-${tone}`}>
      <div className="command-metric-head">
        <span>{label}</span>
        <strong>
          {value}
          {unit ? <small>{unit}</small> : null}
        </strong>
      </div>
      <div className="command-meter" aria-hidden="true">
        <span style={{ width: `${clampRatio(ratio) * 100}%` }} />
      </div>
      <p>{detail}</p>
    </div>
  );
}

function PerformanceDeck({ capture }: { capture: CaptureState | undefined }) {
  const targetFps = captureTargetFps(capture);
  const targetPeriod = targetFps > 0 ? 1000 / targetFps : 8.33;
  const previewTarget = capture?.preview_target_fps || 30;
  const dropBudget = Math.max(1, (capture?.preview_output_frames ?? 0) * 0.02);

  return (
    <div className="command-metric-grid">
      <CommandMetric
        label="采集吞吐"
        value={formatNumber(capture?.fps_capture)}
        unit="fps"
        detail={`目标 ${targetFps} fps，来自采集线程。`}
        ratio={(capture?.fps_capture ?? 0) / targetFps}
        tone={metricTone(capture?.fps_capture, targetFps, "higher")}
      />
      <CommandMetric
        label="帧间隔"
        value={formatNumber(capture?.frame_period_ms, 2)}
        unit="ms"
        detail={`目标 ${targetPeriod.toFixed(2)} ms，越低越稳。`}
        ratio={targetPeriod / Math.max(capture?.frame_period_ms ?? targetPeriod * 2, 0.01)}
        tone={metricTone(capture?.frame_period_ms, targetPeriod, "lower")}
      />
      <CommandMetric
        label="读取等待"
        value={formatNumber(capture?.capture_wait_ms, 2)}
        unit="ms"
        detail="采集 read 等待时间，异常升高会直接影响控制延迟。"
        ratio={targetPeriod / Math.max(capture?.capture_wait_ms ?? targetPeriod * 2, 0.01)}
        tone={metricTone(capture?.capture_wait_ms, targetPeriod, "lower")}
      />
      <CommandMetric
        label="预览输出"
        value={formatNumber(capture?.preview_fps)}
        unit="fps"
        detail={`浏览器目标 ${previewTarget} fps，不代表采集主链路。`}
        ratio={(capture?.preview_fps ?? 0) / previewTarget}
        tone={metricTone(capture?.preview_fps, previewTarget, "higher")}
      />
      <CommandMetric
        label="采集丢帧"
        value={String(capture?.frames_dropped ?? 0)}
        detail="Latest-Frame 队列覆盖旧帧，控制链路优先处理最新画面。"
        ratio={1 - (capture?.frames_dropped ?? 0) / 240}
        tone={(capture?.frames_dropped ?? 0) > 30 ? "warn" : "good"}
      />
      <CommandMetric
        label="预览丢帧"
        value={String(capture?.preview_dropped ?? 0)}
        detail="预览限速丢帧是预期行为，不应拖慢采集和推理。"
        ratio={1 - (capture?.preview_dropped ?? 0) / dropBudget}
        tone={(capture?.preview_dropped ?? 0) > dropBudget ? "warn" : "good"}
      />
    </div>
  );
}

function PipelineRail({ runtime }: { runtime: RuntimeState | null }) {
  const capture = runtime?.capture;
  const executor = runtime?.executor;
  const selectedExecutor = executor?.selected;
  const executorAvailable = selectedExecutor
    ? executor?.executors[selectedExecutor]?.available
    : false;
  const modelName = runtime?.active_model?.project?.name ?? "未发布模型";

  const steps = [
    {
      label: "采集",
      value: capture?.available ? formatProfile(capture) : "未打开",
      tone: capture?.available ? "good" : "warn"
    },
    {
      label: "推理",
      value: runtime?.running ? "运行中" : "待机",
      tone: runtime?.running ? "good" : "idle"
    },
    {
      label: "模型",
      value: modelName,
      tone: runtime?.active_model?.artifact ? "good" : "warn"
    },
    {
      label: "输出",
      value: selectedExecutor ?? "未选择",
      tone: executorAvailable ? "good" : "warn"
    }
  ] as const;

  return (
    <div className="pipeline-rail" aria-label="运行链路">
      {steps.map((step) => (
        <div className={`pipeline-node tone-${step.tone}`} key={step.label}>
          <span>{step.label}</span>
          <strong>{step.value}</strong>
        </div>
      ))}
    </div>
  );
}

function CaptureDiagnosticStrip({ capture }: { capture: CaptureState | undefined }) {
  if (!capture?.last_error) {
    return (
      <div className="diagnostic-strip clean">
        <strong>采集诊断</strong>
        <span>当前没有后端错误。若 Jetson 端失败，GStreamer bus 诊断会显示在这里。</span>
      </div>
    );
  }

  return (
    <div className="diagnostic-strip fault">
      <strong>采集诊断</strong>
      <code>{capture.last_error}</code>
    </div>
  );
}

function RuntimeFields({ runtime }: { runtime: RuntimeState | null }) {
  const capture = runtime?.capture;
  const configVersion =
    typeof runtime?.config?.version === "number" ? String(runtime.config.version) : "0";
  return (
    <div className="field-grid command-fields">
      <Field label="设备" value={capture?.device ?? "/dev/video0"} mono />
      <Field label="后端" value={capture?.backend ?? "未打开"} mono />
      <Field label="配置版本" value={configVersion} mono />
      <Field label="模型文件" value={runtime?.active_model?.artifact?.path ?? "未绑定"} mono />
    </div>
  );
}

function ExecutorTable({ executor }: { executor: ExecutorStatus | undefined }) {
  const entries = Object.entries(executor?.executors ?? {});

  if (entries.length === 0) {
    return <EmptyState title="没有执行器状态" detail="后端没有返回控制输出执行器。" />;
  }

  return (
    <div className="table-wrap compact-table">
      <table>
        <thead>
          <tr>
            <th>执行器</th>
            <th>当前选择</th>
            <th>可用</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([id, state]) => (
            <tr key={id}>
              <td className="mono">{id}</td>
              <td>{executor?.selected === id ? "是" : "否"}</td>
              <td>
                <StatusIndicator tone={state.available ? "good" : "bad"}>
                  {state.available ? "可用" : "不可用"}
                </StatusIndicator>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
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
  const capture = runtime?.capture;
  const runtimeRunning = Boolean(runtime?.running);
  const configVersion =
    typeof runtime?.config?.version === "number" ? runtime.config.version : 0;
  const roiSize =
    typeof runtime?.config?.roi_size === "number" ? runtime.config.roi_size : 640;

  return (
    <div className="command-center">
      <Panel
        title="性能态势"
        eyebrow="实时链路"
        action={
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
            <button className="button" type="button" onClick={onRefresh}>
              {loading ? "刷新中" : "刷新"}
            </button>
          </div>
        }
      >
        <InlineError message={errors.health ?? errors.runtime} />
        <div className="command-hero">
          <div className="command-status-stack">
            <StatusIndicator tone={statusTone(health?.ok)}>
              {loading ? "后端检查中" : health?.ok ? "后端已连接" : "后端离线"}
            </StatusIndicator>
            <StatusIndicator tone={runtimeRunning ? "good" : "idle"}>
              {runtimeRunning ? "推理控制运行中" : "推理控制待机"}
            </StatusIndicator>
            <StatusIndicator tone={capture?.available ? "good" : "warn"}>
              {capture?.available ? "采集已打开" : "采集未打开"}
            </StatusIndicator>
          </div>
          <PerformanceDeck capture={capture} />
        </div>
        <PipelineRail runtime={runtime} />
      </Panel>

      <Panel title="现场画面" eyebrow="浏览器预览">
        <VideoPanel
          available={Boolean(capture?.available)}
          caption={`预览限速输出，ROI ${roiSize}x${roiSize}，主采集链路不依赖浏览器帧率`}
          className="dashboard-preview"
          profile={formatProfile(capture)}
          running={runtimeRunning}
          src={capture?.available ? streamUrl(configVersion, configVersion) : undefined}
        />
      </Panel>

      <Panel title="链路诊断" eyebrow="状态与错误">
        <CaptureDiagnosticStrip capture={capture} />
        <RuntimeFields runtime={runtime} />
      </Panel>

      <Panel title="控制输出" eyebrow="执行器">
        <InlineError message={errors.runtime} />
        <ExecutorTable executor={runtime?.executor} />
      </Panel>
    </div>
  );
}

export { InferenceControl };
