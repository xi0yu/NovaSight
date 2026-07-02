import { useCallback, useState } from "react";

import {
  type LicenseStatus,
  TEST_MAX_LICENSE_KEY,
  clearLicenseKey,
  saveLicenseKey
} from "../../api";
import { InlineError, Panel, StatusIndicator } from "../../components/ui";
import { Field } from "../shared/Field";
import { formatEpoch, getErrorMessage } from "../shared/format";

export const LICENSE_CACHE_KEY = "novasight.license.valid";

type LicenseProps = {
  license: LicenseStatus | null;
  loading?: boolean;
  error?: string;
  onRefresh?: () => void;
  onLicenseChange: (license: LicenseStatus) => void;
};

export function LicensePanel({
  license,
  onLicenseChange
}: Pick<LicenseProps, "license" | "onLicenseChange">) {
  const [licenseInput, setLicenseInput] = useState("");
  const [message, setMessage] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();

  const saveLicense = useCallback(async () => {
    setError(undefined);
    setMessage(undefined);
    try {
      const status = await saveLicenseKey(licenseInput);
      localStorage.setItem(LICENSE_CACHE_KEY, status.valid ? "1" : "0");
      onLicenseChange(status);
      setLicenseInput("");
      setMessage("卡密已激活，界面不回显明文。");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [licenseInput, onLicenseChange]);

  const clearLicense = useCallback(async () => {
    setError(undefined);
    setMessage(undefined);
    try {
      const status = await clearLicenseKey();
      localStorage.removeItem(LICENSE_CACHE_KEY);
      onLicenseChange(status);
      setMessage("卡密已清除。");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [onLicenseChange]);

  return (
    <div className="license-panel">
      <InlineError message={error} />
      {message ? <div className="inline-note">{message}</div> : null}
      <StatusIndicator tone={license?.valid ? "good" : "idle"}>
        {license?.valid ? "授权有效" : "未授权"}
      </StatusIndicator>
      <div className="field-grid">
        <Field label="授权等级" value={license?.tier || "无"} mono />
        <Field label="指纹" value={license?.fingerprint || "无"} mono />
        <Field label="创建时间" value={formatEpoch(license?.created_at)} />
        <Field label="激活时间" value={formatEpoch(license?.activated_at)} />
        <Field label="到期时间" value={formatEpoch(license?.expires_at)} />
        <Field
          label="期限"
          value={
            license?.duration_value
              ? `${license.duration_value} ${license.duration_unit}`
              : "无"
          }
        />
      </div>
      <label className="config-field">
        <span>卡密</span>
        <input
          type="password"
          value={licenseInput}
          placeholder="输入授权码，激活后不回显"
          onChange={(event) => setLicenseInput(event.target.value)}
        />
      </label>
      <div className="preference-actions">
        <button className="button" type="button" onClick={saveLicense} disabled={!licenseInput.trim()}>
          激活卡密
        </button>
        <button className="button" type="button" onClick={() => setLicenseInput(TEST_MAX_LICENSE_KEY)}>
          填入测试卡密
        </button>
        <button className="button" type="button" onClick={clearLicense}>
          清除
        </button>
      </div>
      <div className="license-features">
        {(license?.features ?? []).map((feature) => (
          <span key={feature}>{feature}</span>
        ))}
      </div>
    </div>
  );
}

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
