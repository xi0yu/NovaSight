import { type LicenseStatus } from "../../api";
import { NovaIcon, ThemeToggle } from "../../components/visual";
import { InlineError, Panel } from "../../components/ui";
import { LicensePanel } from "./LicensePanel";
import { LICENSE_CACHE_KEY } from "./storage";

type LicenseProps = {
  license: LicenseStatus | null;
  loading?: boolean;
  error?: string;
  onRefresh?: () => void;
  onLicenseChange: (license: LicenseStatus) => void;
};

export function LicenseGate({
  license,
  loading,
  error,
  onRefresh,
  onLicenseChange
}: Required<Pick<LicenseProps, "loading" | "onRefresh">> &
  Pick<LicenseProps, "license" | "error" | "onLicenseChange">) {
  const serviceUnavailable = !loading && license === null && Boolean(error);

  return (
    <main className="app-shell license-shell">
      <section className="license-gate">
        <div className="brand-block">
          <span className="brand-mark">
            <NovaIcon name="prediction-line" size={24} strokeWidth={1.9} />
          </span>
          <div>
            <h1>NovaSight</h1>
            <p>{serviceUnavailable ? "连接本机服务后进入 Jetson 实时视觉工作台" : "完成本机授权后进入 Jetson 实时视觉工作台"}</p>
          </div>
          <ThemeToggle />
        </div>
        {serviceUnavailable ? (
          <section className="service-connection-panel" role="alert" aria-live="polite">
            <span className="service-connection-icon" aria-hidden="true">
              <NovaIcon name="backend-api" size={24} />
            </span>
            <div>
              <h2>无法读取 NovaSight 后端</h2>
              <p>授权状态尚未读取，因此这里不是卡密错误。请确认本机后端服务已经启动并能正常响应。</p>
              <InlineError message={error} />
              <button className="button" type="button" onClick={onRefresh}>重新连接</button>
            </div>
          </section>
        ) : loading ? (
          <section className="service-connection-panel waiting" role="status" aria-live="polite">
            <span className="service-connection-icon" aria-hidden="true">
              <NovaIcon name="activity-pulse" size={24} />
            </span>
            <div>
              <h2>正在连接 NovaSight 后端</h2>
              <p>正在读取本机服务与授权状态，确认结果前不会显示卡密输入。</p>
            </div>
          </section>
        ) : (
          <>
            <InlineError message={error} />
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
      <Panel title="卡密管理" eyebrow="本机授权">
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
      </Panel>
    </div>
  );
}

export { LicensePanel };
export { LICENSE_CACHE_KEY };
