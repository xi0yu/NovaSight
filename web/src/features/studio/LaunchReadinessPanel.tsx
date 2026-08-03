import { NovaIcon, type NovaIconName } from "../../components/visual";
import type {
  LaunchReadinessAction,
  LaunchReadinessItem,
  LaunchReadinessState,
  LaunchReadinessSummary
} from "./launchReadiness";

const ITEM_ICONS: Record<LaunchReadinessItem["id"], NovaIconName> = {
  license: "shield-check",
  model: "models",
  capture: "capture",
  deepstream: "tensorrt",
  control: "control",
  kmnet: "hid"
};

const STATE_ICONS: Record<LaunchReadinessState, NovaIconName> = {
  ready: "check-circle",
  action: "clock-alert",
  blocked: "triangle-alert",
  idle: "empty-circle"
};

const STATE_LABELS: Record<LaunchReadinessState, string> = {
  ready: "就绪",
  action: "待处理",
  blocked: "阻断",
  idle: "等待"
};

export function LaunchReadinessPanel({
  readiness,
  busy,
  onAction
}: {
  readiness: LaunchReadinessSummary;
  busy: boolean;
  onAction: (action: LaunchReadinessAction) => void;
}) {
  return (
    <section className={`launch-readiness-panel ${readiness.state}`} aria-labelledby="launch-readiness-title">
      <div className="launch-readiness-main">
        <header className="launch-readiness-header">
          <div>
            <span className="class-config-eyebrow">MAINLINE COMMISSIONING</span>
            <h2 id="launch-readiness-title">{readiness.title}</h2>
            <p>{readiness.detail}</p>
          </div>
          <div className="launch-readiness-score" aria-label={`主链准备度 ${readiness.readyCount}/${readiness.blockingCount}`}>
            <strong>{readiness.score}</strong>
            <span>%</span>
            <small>{readiness.readyCount}/{readiness.blockingCount} 主链项</small>
          </div>
        </header>

        <div className="launch-readiness-actions">
          {readiness.primaryAction ? (
            <button
              className="console-button primary"
              disabled={busy}
              onClick={() => onAction(readiness.primaryAction as LaunchReadinessAction)}
              type="button"
            >
              <NovaIcon name={readiness.primaryAction === "start-mainline" ? "start" : "forward"} size={16} />
              {readiness.primaryActionLabel}
            </button>
          ) : null}
          {readiness.secondaryAction ? (
            <button
              className="console-button"
              disabled={busy}
              onClick={() => onAction(readiness.secondaryAction as LaunchReadinessAction)}
              type="button"
            >
              <NovaIcon name="settings" size={16} />
              {readiness.secondaryActionLabel}
            </button>
          ) : null}
        </div>

        <ol className="launch-readiness-steps" aria-label="主链准备检查项">
          {readiness.items.map((item) => (
            <li className={`launch-readiness-step ${item.state}`} key={item.id}>
              <span className="launch-readiness-step-icon" aria-hidden="true">
                <NovaIcon name={ITEM_ICONS[item.id]} size={18} />
              </span>
              <div className="launch-readiness-step-copy">
                <div>
                  <strong>{item.label}</strong>
                  <span>
                    <NovaIcon name={STATE_ICONS[item.state]} size={13} />
                    {STATE_LABELS[item.state]}
                  </span>
                </div>
                <p>{item.detail}</p>
                <small>{item.evidence}</small>
              </div>
              {item.action ? (
                <button
                  className="launch-readiness-step-action"
                  disabled={busy}
                  onClick={() => onAction(item.action as LaunchReadinessAction)}
                  type="button"
                >
                  {item.actionLabel}
                </button>
              ) : null}
            </li>
          ))}
        </ol>
      </div>

      <aside className="launch-product-config" aria-label="运行配置摘要">
        <div className="launch-product-config-heading">
          <NovaIcon name="settings" size={16} />
          <strong>运行配置摘要</strong>
        </div>
        <dl>
          {readiness.productConfig.map((row) => (
            <div key={row.label}>
              <dt>{row.label}</dt>
              <dd>
                <strong>{row.value}</strong>
                <span>{row.detail}</span>
              </dd>
            </div>
          ))}
        </dl>
      </aside>
    </section>
  );
}
