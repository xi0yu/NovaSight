import { type LicenseStatus } from "../../api";
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
  return (
    <main className="app-shell license-shell">
      <section className="license-gate">
        <div className="brand-block">
          <span className="brand-mark"><span>NS</span></span>
          <div>
            <h1>NovaSight</h1>
            <p>请输入卡密后进入 Jetson 实时视觉工作台</p>
          </div>
        </div>
        <InlineError message={error} />
        {loading ? <div className="inline-note">正在校验本机授权状态。</div> : null}
        <LicensePanel license={license} onLicenseChange={onLicenseChange} />
        <button className="button compact-button" type="button" onClick={onRefresh}>
          重新校验
        </button>
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
