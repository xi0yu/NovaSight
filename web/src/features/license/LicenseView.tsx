import { type LicenseStatus } from "../../api";
import { NovaIcon } from "../../components/visual";
import { Panel } from "../../components/ui";
import { LicenseActivationForm, LicensePanel } from "./LicensePanel";
import { type LicenseConnectionIssue } from "./connectionIssue";

type LicenseProps = {
  license: LicenseStatus | null;
  loading?: boolean;
  issue?: LicenseConnectionIssue | null;
  onRefresh?: () => void;
  onLicenseChange: (license: LicenseStatus) => void;
};

export function LicenseGate({
  license,
  loading,
  issue,
  onRefresh,
  onLicenseChange
}: Required<Pick<LicenseProps, "loading" | "onRefresh">> &
  Pick<LicenseProps, "license" | "issue" | "onLicenseChange">) {
  const serviceUnavailable = !loading && license === null && issue !== null;
  const gateSubtitle = loading
    ? "正在校验本机服务与授权状态"
    : serviceUnavailable
      ? "连接本机服务后进入 Jetson 实时视觉工作台"
      : license?.valid
        ? "授权已激活，进入 Jetson 实时视觉工作台"
        : "输入临时授权码或正式许可证后进入工作台";

  return (
    <main className="app-shell license-shell">
      <section className="license-gate">
        <div className="brand-block">
          <span className="brand-mark">
            <NovaIcon name="prediction-line" size={24} strokeWidth={1.9} />
          </span>
          <div>
            <h1>NovaSight</h1>
            <p>{gateSubtitle}</p>
          </div>
        </div>
        {serviceUnavailable ? (
          <section className="service-connection-panel" role="alert" aria-live="polite">
            <span className="service-connection-icon" aria-hidden="true">
              <NovaIcon name="backend-api" size={24} />
            </span>
            <div>
              <h2>{issue?.title ?? "尚未连接到 NovaSight 服务"}</h2>
              <p>{issue?.description}</p>
              <div className="service-connection-guidance">
                <strong>建议操作</strong>
                <span>{issue?.recovery}</span>
              </div>
              <div className="service-connection-actions">
                <button className="button" type="button" onClick={onRefresh}>重新连接</button>
              </div>
            </div>
          </section>
        ) : loading ? (
          <section className="service-connection-panel waiting" role="status" aria-live="polite">
            <span className="service-connection-icon" aria-hidden="true">
              <NovaIcon name="activity-pulse" size={24} />
            </span>
            <div>
              <h2>正在连接 NovaSight 服务</h2>
              <p>正在读取本机服务与授权状态，请稍候。</p>
            </div>
          </section>
        ) : (
          <LicenseActivationForm
            gate
            temporarySupported={license?.temporary_access_supported === true}
            onLicenseChange={onLicenseChange}
          />
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
