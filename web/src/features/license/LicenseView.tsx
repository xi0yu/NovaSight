import { type LicenseStatus } from "../../api";
import { NovaIcon, ThemeToggle } from "../../components/visual";
import { Panel } from "../../components/ui";
import { LicensePanel } from "./LicensePanel";
import { type LicenseConnectionIssue } from "./connectionIssue";
import { LICENSE_CACHE_KEY } from "./storage";

type LicenseProps = {
  license: LicenseStatus | null;
  loading?: boolean;
  issue?: LicenseConnectionIssue | null;
  onRefresh?: () => void;
  onTemporaryRecovery?: () => void;
  temporaryRecoveryLoading?: boolean;
  onLicenseChange: (license: LicenseStatus) => void;
};

export function LicenseGate({
  license,
  loading,
  issue,
  onRefresh,
  onTemporaryRecovery,
  temporaryRecoveryLoading,
  onLicenseChange
}: Required<Pick<LicenseProps, "loading" | "onRefresh" | "onTemporaryRecovery" | "temporaryRecoveryLoading">> &
  Pick<LicenseProps, "license" | "issue" | "onLicenseChange">) {
  const serviceUnavailable = !loading && license === null && issue !== null;

  return (
    <main className="app-shell license-shell">
      <section className="license-gate">
        <div className="brand-block">
          <span className="brand-mark">
            <NovaIcon name="prediction-line" size={24} strokeWidth={1.9} />
          </span>
          <div>
            <h1>NovaSight</h1>
            <p>{serviceUnavailable ? "连接本机服务后进入 Jetson 实时视觉工作台" : "申请临时授权后进入 Jetson 实时视觉工作台"}</p>
          </div>
          <ThemeToggle />
        </div>
        {serviceUnavailable ? (
          <section className="service-connection-panel" role="alert" aria-live="polite">
            <span className="service-connection-icon" aria-hidden="true">
              <NovaIcon name="backend-api" size={24} />
            </span>
            <div>
              <h2>{issue?.title ?? "尚未连接到 NovaSight 后端"}</h2>
              <p>{issue?.description}</p>
              <div className="service-connection-guidance">
                <strong>建议操作</strong>
                <span>{issue?.recovery}</span>
              </div>
              <div className="service-connection-actions">
                <button className="button" type="button" onClick={onRefresh}>重新连接</button>
                {issue?.kind === "service-error" ? (
                  <button
                    className="button compact-button"
                    type="button"
                    disabled={temporaryRecoveryLoading}
                    onClick={onTemporaryRecovery}
                  >
                    {temporaryRecoveryLoading ? "正在尝试…" : "尝试 Debug 临时权限"}
                  </button>
                ) : null}
              </div>
            </div>
          </section>
        ) : loading ? (
          <section className="service-connection-panel waiting" role="status" aria-live="polite">
            <span className="service-connection-icon" aria-hidden="true">
              <NovaIcon name="activity-pulse" size={24} />
            </span>
            <div>
              <h2>正在连接 NovaSight 后端</h2>
              <p>正在读取本机服务与授权状态，请稍候。</p>
            </div>
          </section>
        ) : (
          <>
            <LicensePanel license={license} onLicenseChange={onLicenseChange} />
            <button className="button compact-button" type="button" onClick={onRefresh}>
              重新校验
            </button>
          </>
        )}
      </section>
    </main>
  );
}

export function LicenseView({ license, onLicenseChange }: LicenseProps) {
  return (
    <div className="view-grid">
      <Panel title="授权管理" eyebrow="本机授权">
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
      </Panel>
    </div>
  );
}

export { LicensePanel };
export { LICENSE_CACHE_KEY };
