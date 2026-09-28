import type { ReactNode } from "react";

import type { RuntimeState } from "../../api";
import { NovaIcon } from "../../components/visual";
import type { RuntimeProjection, RuntimeRecoveryAction } from "../runtime/runtimeProjection";
import { RuntimeSwitch } from "./ControlSwitches";
import type { LaunchReadinessSummary } from "./launchReadiness";

import "./runtime-overview.css";

type RuntimeOverviewViewProps = {
  runtime: RuntimeState | null;
  projection: RuntimeProjection | null;
  readiness: LaunchReadinessSummary;
  lastUpdated: Date | null;
  onAction: (action: RuntimeRecoveryAction) => void;
  controlBusy: boolean;
  launchPending: boolean;
  runtimeStopping: boolean;
  runtimeControlUnavailable: boolean;
  onOpenErrors?: () => void;
  onToggle: () => void | Promise<void | boolean>;
  moduleOrder?: readonly string[];
};

const DEFAULT_MODULES = ["runtime", "metrics", "pipeline", "diagnostics"];

function toneForProjection(projection: RuntimeProjection): string {
  if (projection.transport !== "current" || projection.output.state === "unknown") return "unknown";
  if (projection.lifecycle.state === "faulted" || projection.perception.state === "faulted") return "danger";
  if (projection.daemonConfirmedSafe) return "safe";
  if (projection.output.state === "armed") return "armed";
  return "attention";
}

function formatEvidenceTime(value: Date | null): string {
  return value ? value.toLocaleTimeString("zh-CN", { hour12: false }) : "尚未取得";
}

function formatLiveMetric(available: boolean, value: number | null, digits: number, unit: string): string {
  return available && value !== null ? `${value.toFixed(digits)}${unit}` : "等待样本";
}

export function RuntimeOverviewView({
  runtime,
  projection,
  readiness,
  lastUpdated,
  onAction,
  controlBusy,
  launchPending,
  runtimeStopping,
  runtimeControlUnavailable,
  onOpenErrors,
  onToggle,
  moduleOrder = DEFAULT_MODULES,
}: RuntimeOverviewViewProps) {
  if (!runtime || !projection) {
    return (
      <section className="runtime-overview runtime-overview-offline" role="status" aria-label="当前运行结论">
        <article className="runtime-overview-stage">
          <header className="runtime-overview-stage-header">
            <span><NovaIcon name="capture-card" size={16} />实时画面</span>
          </header>
          <div className="runtime-overview-stage-canvas">
            <div className="runtime-overview-stage-visual" aria-hidden="true"><i /><i /><i /></div>
            <div className="runtime-overview-stage-placeholder">
              <span aria-hidden="true"><NovaIcon name="prediction-line" size={22} strokeWidth={1.6} /></span>
              <strong>等待画面</strong>
              <small>服务连接后自动显示</small>
            </div>
          </div>
        </article>
        <div className="runtime-overview-offline-actions">
          <article>
            <span className="runtime-overview-offline-icon" aria-hidden="true"><NovaIcon name="daemon" size={18} /></span>
            <div>
              <small>当前运行结论</small>
              <strong>{runtimeControlUnavailable ? "无法确认运行状态" : "正在读取运行状态"}</strong>
              <p>{runtimeControlUnavailable ? "检查服务连接或查看异常记录。" : "等待服务返回完整状态。"}</p>
            </div>
            {runtimeControlUnavailable && onOpenErrors ? (
              <button className="console-button" onClick={onOpenErrors} type="button">查看异常信息</button>
            ) : <span className="runtime-overview-waiting">请稍候</span>}
          </article>
          <article>
            <span className="runtime-overview-offline-icon safe" aria-hidden="true"><NovaIcon name="shield-check" size={18} /></span>
            <div>
              <small>硬件安全</small>
              <strong>输出保持锁定</strong>
              <p>收到完整运行状态前，不会推断或发送硬件输出。</p>
            </div>
            <span className="runtime-overview-locked">已锁定</span>
          </article>
        </div>
      </section>
    );
  }

  const tone = toneForProjection(projection);
  const kmnet = runtime.executor.executors.kmnet;
  const phase = runtime.semantic.phase;
  const lifecycleActive = runtime.presentation.lifecycle.can_stop || phase === "stopping";
  const stopping = phase === "stopping" || runtimeStopping;
  const starting = phase === "starting" || launchPending;
  const metricsCurrent = projection.transport === "current";
  const metricsAvailable = metricsCurrent && runtime.statistics.metrics_available === true;
  const missingMetricLabel = metricsCurrent ? "等待样本" : "状态已过期";
  const runtimeControlLabel = stopping ? "正在停止" : starting ? "正在启动" : lifecycleActive ? "运行中" : "已关闭";
  const runtimeControlTone = phase === "faulted"
    ? "danger"
    : starting || stopping || phase === "waiting_model"
      ? "warning"
      : lifecycleActive ? "success" : "idle";
  const chainStale = !metricsCurrent;
  const chain = [
    { label: "采集输入", value: chainStale ? "状态已过期" : runtime.capture.running ? "正在接收" : runtime.capture.available ? "待运行" : "不可用", live: metricsCurrent && runtime.capture.running },
    { label: "感知数据", value: projection.perception.label, live: metricsCurrent && projection.perception.state === "current" },
    { label: "目标选择", value: chainStale ? "状态已过期" : runtime.vision.target ? "已选择目标" : "未选择目标", live: metricsCurrent && runtime.vision.target !== null },
    { label: "硬件输出", value: projection.output.label, live: metricsCurrent && projection.output.state === "armed" },
  ];

  const modules: Record<string, ReactNode> = {
    runtime: (
      <article className={`runtime-overview-conclusion ${tone}`}>
        <div className="runtime-overview-identity">
          <span className="runtime-overview-conclusion-icon" aria-hidden="true">
            <NovaIcon name={tone === "safe" ? "shield-check" : tone === "danger" ? "triangle-alert" : "activity-pulse"} size={24} />
          </span>
          <div><small>{metricsCurrent ? "当前设备" : "最近设备"}</small><strong>{runtime.capture.device || "设备未确认"}</strong></div>
        </div>
        <div className="runtime-overview-conclusion-copy">
          <small>现在</small>
          <h2>{projection.conclusion}</h2>
          <p>{readiness.detail}</p>
        </div>
        <div className="runtime-overview-actions">
          {projection.nextAction ? (
            <button className="console-button" type="button" onClick={() => onAction(projection.nextAction!)}>{projection.nextActionLabel ?? "处理问题"}</button>
          ) : null}
          <RuntimeSwitch
            label="运行总开关"
            enabled={lifecycleActive}
            pending={starting || stopping}
            disabled={controlBusy || runtimeControlUnavailable || !metricsCurrent}
            status={runtimeControlLabel}
            tone={runtimeControlTone}
            onToggle={onToggle}
          />
        </div>
      </article>
    ),
    metrics: (
      <dl className={`runtime-overview-metrics${metricsAvailable ? " is-live" : ""}`} aria-label="核心运行数据">
        <div><dt>推理输入</dt><dd>{metricsAvailable ? formatLiveMetric(true, runtime.statistics.nvinfer_input_fps, 0, " FPS") : missingMetricLabel}</dd></div>
        <div><dt>检测结果</dt><dd>{metricsAvailable ? formatLiveMetric(true, runtime.statistics.detection_batch_fps, 0, " FPS") : missingMetricLabel}</dd></div>
        <div><dt>推理耗时</dt><dd>{metricsAvailable ? formatLiveMetric(true, runtime.statistics.inference_latency_ms, 1, " ms") : missingMetricLabel}</dd></div>
        <div><dt>结果帧龄</dt><dd>{metricsAvailable ? formatLiveMetric(true, runtime.statistics.detection_data_age_ms, 1, " ms") : missingMetricLabel}</dd></div>
      </dl>
    ),
    pipeline: (
      <section className="runtime-overview-pipeline" aria-labelledby="runtime-pipeline-title">
        <header><div><small>实时路径</small><h3 id="runtime-pipeline-title">画面如何变成输出</h3></div><span>{metricsCurrent && runtime.semantic.phase === "running" ? "LIVE" : chainStale ? "状态已过期" : "状态快照"}</span></header>
        <ol className="runtime-pipeline-list">
          {chain.map((item, index) => <li key={item.label}><i>{index + 1}</i><span>{item.label}<strong>{item.value}</strong></span></li>)}
        </ol>
        <div className="runtime-overview-axes">
          <article><span><NovaIcon name="daemon" size={17} />运行</span><div><strong>{projection.lifecycle.label}</strong><p>{projection.lifecycle.detail}</p></div></article>
          <article><span><NovaIcon name="inference" size={17} />识别</span><div><strong>{projection.perception.label}</strong><p>{projection.perception.detail}</p></div></article>
          <article><span><NovaIcon name="device-send" size={17} />输出</span><div><strong>{projection.output.label}</strong><p>{projection.output.detail}</p></div></article>
        </div>
      </section>
    ),
    diagnostics: (
      <details className="runtime-overview-evidence">
        <summary><span><small>诊断</small><strong>运行诊断证据</strong></span><span>快照与输出门控，共 5 项</span></summary>
        <dl>
          <div><dt>Daemon</dt><dd title={runtime.semantic.daemon_instance_id}>{runtime.semantic.daemon_instance_id.slice(0, 12)}</dd></div>
          <div><dt>序列</dt><dd>#{runtime.semantic.snapshot_sequence}</dd></div>
          <div><dt>Studio 收到</dt><dd>{formatEvidenceTime(lastUpdated)}</dd></div>
          <div><dt>kmNet 运行连接</dt><dd>{kmnet?.runtime_connected === true ? "已连接" : kmnet?.runtime_connected === false ? "已断开" : "未知"}</dd></div>
          <div><dt>当前样本可输出</dt><dd>{runtime.vision.control.will_emit === true ? "是" : runtime.vision.control.will_emit === false ? "否" : "无样本"}</dd></div>
        </dl>
      </details>
    ),
  };

  return (
    <section className="runtime-overview" aria-label="运行总览">
      {moduleOrder.map((moduleId) => modules[moduleId] ? <div data-module={moduleId} key={moduleId}>{modules[moduleId]}</div> : null)}
    </section>
  );
}
