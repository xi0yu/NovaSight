import { useCallback, useState } from "react";

import {
  type LicenseStatus,
  clearLicenseKey,
  requestTemporaryLicense,
  saveLicenseKey
} from "../../api";
import { InlineError, StatusIndicator } from "../../components/ui";
import { reportError } from "../../lib/toast";
import { Field } from "../shared/Field";
import { formatEpoch } from "../shared/format";
import {
  describeLicenseActionFailure,
  type LicenseActionFailure
} from "./connectionIssue";
import { LICENSE_CACHE_KEY } from "./storage";

export function LicensePanel({
  license,
  onLicenseChange
}: {
  license: LicenseStatus | null;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
  const [licenseInput, setLicenseInput] = useState("");
  const [requesting, setRequesting] = useState(false);
  const [activating, setActivating] = useState(false);
  const [message, setMessage] = useState<string | undefined>();
  const [failure, setFailure] = useState<LicenseActionFailure | null>(null);
  const currentTime = new Date().toLocaleString("zh-CN", { hour12: false });

  const requestTemporary = useCallback(async () => {
    setFailure(null);
    setMessage(undefined);
    setRequesting(true);
    try {
      const response = await requestTemporaryLicense();
      const status = response.status;
      if (!response.granted || !status.valid) {
        localStorage.setItem(LICENSE_CACHE_KEY, status.valid ? "1" : "0");
        onLicenseChange(status);
        setFailure({
          title: "临时授权未通过",
          message: status.valid
            ? "当前已有有效的正式授权，无需申请临时授权。"
            : "当前后端未开放临时授权，请使用 Debug 测试版本，或改用正式授权凭证。"
        });
        return;
      }
      localStorage.setItem(LICENSE_CACHE_KEY, status.valid ? "1" : "0");
      onLicenseChange(status);
      setMessage("临时授权已通过，可以进入 NovaSight 工作台。");
    } catch (err) {
      const nextFailure = describeLicenseActionFailure(err, "temporary");
      setFailure(nextFailure);
      reportError(err, {
        source: "license-temporary",
        title: nextFailure.title,
        publicDetail: nextFailure.message,
        exposeStatus: false
      });
    } finally {
      setRequesting(false);
    }
  }, [onLicenseChange]);

  const activateSignedLicense = useCallback(async () => {
    setFailure(null);
    setMessage(undefined);
    setActivating(true);
    try {
      const status = await saveLicenseKey(licenseInput);
      localStorage.setItem(LICENSE_CACHE_KEY, status.valid ? "1" : "0");
      onLicenseChange(status);
      if (!status.valid) {
        setFailure({
          title: "正式授权未生效",
          message: "该授权凭证已过期或当前不可用，请更换有效凭证。"
        });
        return;
      }
      setLicenseInput("");
      setMessage("正式授权已激活，授权凭证不会在界面中回显。");
    } catch (err) {
      const nextFailure = describeLicenseActionFailure(err, "activate");
      setFailure(nextFailure);
      reportError(err, {
        source: "license-activate",
        title: nextFailure.title,
        publicDetail: nextFailure.message,
        exposeStatus: false
      });
    } finally {
      setActivating(false);
    }
  }, [licenseInput, onLicenseChange]);

  const clearLicense = useCallback(async () => {
    setFailure(null);
    setMessage(undefined);
    try {
      const status = await clearLicenseKey();
      localStorage.removeItem(LICENSE_CACHE_KEY);
      onLicenseChange(status);
      setMessage("当前授权已退出。");
    } catch (err) {
      const nextFailure = describeLicenseActionFailure(err, "clear");
      setFailure(nextFailure);
      reportError(err, {
        source: "license-clear",
        title: nextFailure.title,
        publicDetail: nextFailure.message,
        exposeStatus: false
      });
    }
  }, [onLicenseChange]);

  return (
    <div className="license-panel">
      <InlineError message={failure?.message} title={failure?.title} />
      {message ? <div className="inline-note">{message}</div> : null}
      <div className={license?.valid ? "license-hero valid" : "license-hero"}>
        <div>
          <span>NovaSight 授权</span>
          <strong>{license?.valid ? formatTier(license.tier) : "等待授权"}</strong>
          <p>{license?.valid ? "授权状态有效，当前设备可以进入完整工作台。" : "测试阶段可直接申请临时授权；正式环境使用签名授权凭证。"}</p>
        </div>
        <StatusIndicator tone={license?.valid ? "good" : "idle"}>
          {license?.valid ? "授权有效" : "尚未授权"}
        </StatusIndicator>
      </div>
      <div className="field-grid">
        <Field label="授权类型" value={license?.tier ? formatTier(license.tier) : "未授权"} />
        <Field label="当前状态" value={license?.valid ? "可用" : "等待申请"} />
        <Field label="当前时间" value={currentTime} />
        <Field label="激活时间" value={formatEpoch(license?.activated_at)} />
        <Field label="到期时间" value={formatEpoch(license?.expires_at)} />
        <Field
          label="期限"
          value={formatDuration(license?.duration_value, license?.duration_unit)}
        />
      </div>
      <div className="temporary-license-action">
        <div>
          <strong>测试临时授权</strong>
          <span>Debug 后端直接批准，有效期 24 小时；不需要共享或固定测试码。</span>
        </div>
        <button
          className="button"
          type="button"
          onClick={requestTemporary}
          disabled={requesting || license?.valid === true}
        >
          {requesting ? "正在申请…" : license?.valid ? "授权已生效" : "申请临时授权"}
        </button>
      </div>
      <div className="signed-license-action">
        <div>
          <strong>正式授权</strong>
          <span>使用经过签名的授权凭证；激活后不会保存或回显凭证明文。</span>
        </div>
        <div className="signed-license-controls">
          <label className="config-field">
            <span>授权凭证</span>
            <input
              type="password"
              value={licenseInput}
              placeholder="输入 NS1 签名授权凭证"
              autoComplete="off"
              onChange={(event) => setLicenseInput(event.target.value)}
            />
          </label>
          <button
            className="button"
            type="button"
            onClick={activateSignedLicense}
            disabled={activating || !licenseInput.trim()}
          >
            {activating ? "正在验证…" : "激活正式授权"}
          </button>
        </div>
      </div>
      {license?.configured ? (
        <button className="button compact-button" type="button" onClick={clearLicense}>
          退出当前授权
        </button>
      ) : null}
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
  if (tier === "temporary") {
    return "临时测试授权";
  }
  if (tier === "pro") {
    return "专业版";
  }
  if (tier === "premium" || tier === "ultimate") {
    return "旗舰版";
  }
  return tier;
}

function formatDuration(value: number | null | undefined, unit: string | undefined): string {
  if (!value || !unit) {
    return "无";
  }
  const labels: Record<string, string> = {
    day: "天",
    days: "天",
    month: "个月",
    months: "个月",
    year: "年",
    years: "年"
  };
  return `${value} ${labels[unit] ?? unit}`;
}

function formatFeature(feature: string): string {
  const labels: Record<string, string> = {
    capture: "真机采集",
    runtime: "运行控制",
    models: "模型仓库",
    tensorrt: "TensorRT",
    hardware_control: "硬件控制",
    config_read: "读取配置",
    config_write: "写入配置"
  };
  return labels[feature] ?? feature;
}
