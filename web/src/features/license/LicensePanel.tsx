import { useCallback, useState } from "react";

import {
  type LicenseStatus,
  clearLicenseKey,
  saveLicenseKey
} from "../../api";
import { InlineError, StatusIndicator } from "../../components/ui";
import { Field } from "../shared/Field";
import { formatEpoch, getErrorMessage } from "../shared/format";
import { LICENSE_CACHE_KEY } from "./storage";

export function LicensePanel({
  license,
  onLicenseChange
}: {
  license: LicenseStatus | null;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
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
