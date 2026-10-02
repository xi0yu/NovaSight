import { NovaIcon } from "../../components/visual";
import type {
  LaunchReadinessAction,
  LaunchReadinessSummary
} from "./launchReadiness";

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
    <section
      aria-labelledby="launch-readiness-title"
      className={`launch-readiness-panel ${readiness.state}`}
    >
      <div className="launch-readiness-main">
        <header className="launch-readiness-header">
          <div>
            <span className="class-config-eyebrow">运行状态</span>
            <h2 id="launch-readiness-title">{readiness.title}</h2>
            <p>{readiness.detail}</p>
          </div>
          {readiness.primaryAction ? (
            <button
              className="console-button primary"
              disabled={busy}
              onClick={() => onAction(readiness.primaryAction as LaunchReadinessAction)}
              type="button"
            >
              <NovaIcon name="forward" size={16} />
              {readiness.primaryActionLabel}
            </button>
          ) : null}
        </header>
      </div>
    </section>
  );
}
