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
      <div className={license?.valid ? "license-hero valid" : "license-hero"}>
        <div>
          <span>NovaSight 授权</span>
          <strong>{license?.valid ? formatTier(license.tier) : "等待激活"}</strong>
          <p>{license?.valid ? "已解锁核心能力，当前设备可进入完整工作台。" : "激活卡密后解锁采集、推理、插件和配置能力。"}</p>
        </div>
        <StatusIndicator tone={license?.valid ? "good" : "idle"}>
          {license?.valid ? "尊享权益已启用" : "未授权"}
        </StatusIndicator>
      </div>
      <div className="field-grid">
        <Field label="授权等级" value={license?.tier ? formatTier(license.tier) : "无"} />
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
          <span key={feature}>{formatFeature(feature)}</span>
        ))}
      </div>
    </div>
  );
}

function formatTier(tier: string): string {
  if (!tier) {
    return "无";
  }
  if (tier === "test_max") {
    return "测试全权限";
  }
  if (tier === "pro") {
    return "专业版";
  }
  if (tier === "premium" || tier === "ultimate") {
    return "旗舰版";
  }
  return tier;
}

function formatFeature(feature: string): string {
  const labels: Record<string, string> = {
    capture: "真机采集",
    runtime: "运行控制",
    models: "模型仓库",
    plugins: "算法插件",
    tensorrt: "TensorRT",
    hardware_control: "硬件控制",
    config_read: "读取配置",
    config_write: "写入配置"
  };
  return labels[feature] ?? feature;
}
