import type { RuntimeState } from "../../api";
import { NovaIcon } from "../../components/visual";
import type { RuntimeProjection, RuntimeRecoveryAction } from "../runtime/runtimeProjection";
import type { LaunchReadinessSummary } from "./launchReadiness";

import "./runtime-overview.css";

type RuntimeOverviewViewProps = {
  runtime: RuntimeState | null;
  projection: RuntimeProjection | null;
  readiness: LaunchReadinessSummary;
  lastUpdated: Date | null;
  onAction: (action: RuntimeRecoveryAction) => void;
};

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

export function RuntimeOverviewView({
  runtime,
  projection,
  readiness,
  lastUpdated,
  onAction,
}: RuntimeOverviewViewProps) {
  if (!runtime || !projection) {
    return (
      <section className="runtime-overview-empty" role="status">
        <NovaIcon name="daemon" size={22} />
        <div>
          <strong>正在读取 novasightd 权威状态</strong>
          <p>在完整运行快照到达前，Studio 不会推断感知或硬件输出状态。</p>
        </div>
      </section>
    );
  }

  const tone = toneForProjection(projection);
  const kmnet = runtime.executor.executors.kmnet;
  const chain = [
    { label: "采集输入", value: runtime.capture.running ? "正在接收" : runtime.capture.available ? "待运行" : "不可用" },
    { label: "感知数据", value: projection.perception.label },
    { label: "目标选择", value: runtime.vision.target ? "已选择目标" : "未选择目标" },
    { label: "硬件输出", value: projection.output.label },
  ];

  return (
    <section className="runtime-overview" aria-label="运行总览">
      <article className={`runtime-overview-conclusion ${tone}`}>
        <span className="runtime-overview-conclusion-icon" aria-hidden="true">
          <NovaIcon
            name={tone === "safe" ? "shield-check" : tone === "danger" ? "triangle-alert" : "activity-pulse"}
            size={24}
          />
        </span>
        <div>
          <small>当前结论</small>
          <h2>{projection.conclusion}</h2>
          <p>{readiness.detail}</p>
        </div>
        {projection.nextAction ? (
          <button className="console-button" type="button" onClick={() => onAction(projection.nextAction!)}>
            {projection.nextActionLabel ?? "处理问题"}
          </button>
        ) : null}
      </article>

      <div className="runtime-overview-axes">
        <article data-state={projection.lifecycle.state}>
          <span><NovaIcon name="daemon" size={17} />运行生命周期</span>
          <strong>{projection.lifecycle.label}</strong>
          <p>{projection.lifecycle.detail}</p>
        </article>
        <article data-state={projection.perception.state}>
          <span><NovaIcon name="inference" size={17} />感知数据</span>
          <strong>{projection.perception.label}</strong>
          <p>{projection.perception.detail}</p>
        </article>
        <article data-state={projection.output.state}>
          <span><NovaIcon name="device-send" size={17} />硬件输出</span>
          <strong>{projection.output.label}</strong>
          <p>{projection.output.detail}</p>
        </article>
      </div>

      <div className="runtime-overview-grid">
        <article className="runtime-overview-chain">
          <header>
            <div>
              <small>真实主链</small>
              <h3>采集 → 感知 → 目标 → 输出</h3>
            </div>
            <span>{runtime.semantic.phase === "running" ? "实时" : "状态快照"}</span>
          </header>
          <ol>
            {chain.map((item, index) => (
              <li key={item.label}>
                <i>{index + 1}</i>
                <span>{item.label}<strong>{item.value}</strong></span>
              </li>
            ))}
          </ol>
        </article>

        <details className="runtime-overview-evidence">
          <summary>
            <span>
              <small>诊断</small>
              <strong>运行诊断证据</strong>
            </span>
            <span>快照与输出门控 · 5 项</span>
          </summary>
          <dl>
            <div><dt>Daemon</dt><dd title={runtime.semantic.daemon_instance_id}>{runtime.semantic.daemon_instance_id.slice(0, 12)}</dd></div>
            <div><dt>序列</dt><dd>#{runtime.semantic.snapshot_sequence}</dd></div>
            <div><dt>Studio 收到</dt><dd>{formatEvidenceTime(lastUpdated)}</dd></div>
            <div><dt>kmNet 运行连接</dt><dd>{kmnet?.runtime_connected === true ? "已连接" : kmnet?.runtime_connected === false ? "已断开" : "未知"}</dd></div>
            <div><dt>当前样本可输出</dt><dd>{runtime.vision.control.will_emit === true ? "是" : runtime.vision.control.will_emit === false ? "否" : "无样本"}</dd></div>
          </dl>
        </details>
      </div>
    </section>
  );
}
