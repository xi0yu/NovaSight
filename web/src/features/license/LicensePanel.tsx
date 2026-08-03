import { useCallback, useEffect, useState } from "react";

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

const LICENSE_CLEAR_CONFIRMATION_MESSAGE = "再次点击将在当前设备退出授权；5 秒后自动取消确认。";

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
  const [clearing, setClearing] = useState(false);
  const [clearArmed, setClearArmed] = useState(false);
  const [message, setMessage] = useState<string | undefined>();
  const [failure, setFailure] = useState<LicenseActionFailure | null>(null);
  const currentTime = new Date().toLocaleString("zh-CN", { hour12: false });
  const temporarySupported = license?.temporary_access_supported === true;
  const traceSource = license?.token_id || license?.license_id || license?.fingerprint || "";
  const statusTraceCode = formatTraceFragment(traceSource);
  const fingerprintTraceCode = formatTraceFragment(license?.fingerprint);
  const licenseTraceCode = formatTraceFragment(license?.license_id);
  const tokenTraceCode = formatTraceFragment(license?.token_id);
  const keyTraceCode = formatTraceFragment(license?.key_id);
  const validFrom = license?.valid
    ? formatEpoch(license.not_before ?? license.created_at)
    : "等待签发";
  const activatedAt = license?.valid
    ? formatEpoch(license.activated_at ?? license.created_at)
    : "等待激活";
  const licenseStatusTitle = license?.valid
    ? "授权已激活"
    : "等待授权激活";
  const backendMessage = license?.message?.trim();
  const features = license?.features ?? [];

  const requestTemporary = useCallback(async () => {
    setFailure(null);
    setMessage(undefined);
    setRequesting(true);
    try {
      const response = await requestTemporaryLicense();
      const status = response.status;
      if (!response.supported) {
        onLicenseChange(status);
        setFailure({
          title: "当前构建不支持临时权限",
          message: "Release 后端只接受正式签名授权；请启动 Debug 构建的 novasightd 进行开发验证。"
        });
        return;
      }
      if (!response.granted || !status.valid) {
        onLicenseChange(status);
        setFailure({
          title: "临时授权未通过",
          message: status.valid
            ? "当前已有有效的正式授权，无需申请临时授权。"
            : "当前后端未开放临时授权，请使用 Debug 测试版本，或改用正式授权凭证。"
        });
        return;
      }
      onLicenseChange(status);
      setMessage("当前 Debug 后端进程已开放开发权限，可以进入 NovaSight 工作台。");
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

  useEffect(() => {
    if (!clearArmed) return undefined;
    const timeout = window.setTimeout(() => {
      setClearArmed(false);
      setMessage((current) => current === LICENSE_CLEAR_CONFIRMATION_MESSAGE ? undefined : current);
    }, 5000);
    return () => window.clearTimeout(timeout);
  }, [clearArmed]);

  useEffect(() => {
    setClearArmed(false);
  }, [license?.token_id, license?.valid]);

  const clearLicense = useCallback(async () => {
    if (!clearArmed) {
      setClearArmed(true);
      setMessage(LICENSE_CLEAR_CONFIRMATION_MESSAGE);
      return;
    }
    setFailure(null);
    setMessage(undefined);
    setClearing(true);
    try {
      const status = await clearLicenseKey();
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
    } finally {
      setClearing(false);
      setClearArmed(false);
    }
  }, [clearArmed, onLicenseChange]);

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
      <div className={license?.valid ? "license-status-card valid" : "license-status-card"}>
        <div className="license-status-copy">
          <span>LICENSE STATUS</span>
          <strong>{licenseStatusTitle}</strong>
          <p>
            授权凭证由后端保存和校验；界面只展示追踪片段，不回显完整凭证。
          </p>
        </div>
        <dl className="license-status-trace">
          <div>
            <dt>状态追踪码</dt>
            <dd>{statusTraceCode}</dd>
          </div>
          <div>
            <dt>设备指纹</dt>
            <dd>{fingerprintTraceCode}</dd>
          </div>
          <div>
            <dt>授权编号</dt>
            <dd>{licenseTraceCode}</dd>
          </div>
          <div>
            <dt>签发密钥</dt>
            <dd>{keyTraceCode}</dd>
          </div>
          <div>
            <dt>凭证片段</dt>
            <dd>{tokenTraceCode}</dd>
          </div>
          <div>
            <dt>生效时间</dt>
            <dd>{validFrom}</dd>
          </div>
          <div>
            <dt>激活时间</dt>
            <dd>{activatedAt}</dd>
          </div>
        </dl>
      </div>
      {backendMessage ? (
        <div className="license-provenance-note">
          <strong>授权说明</strong>
          <span>{backendMessage}</span>
        </div>
      ) : null}
      <div className="field-grid">
        <Field label="授权类型" value={license?.tier ? formatTier(license.tier) : "未授权"} />
        <Field label="凭证格式" value={formatCredentialFormat(license?.credential_format)} />
        <Field label="授权编号" value={license?.license_id || "无"} />
        <Field label="当前状态" value={license?.valid ? "可用" : "等待申请"} />
        <Field label="当前时间" value={currentTime} />
        <Field label="激活时间" value={formatEpoch(license?.activated_at)} />
        <Field
          label="到期时间"
          value={license?.tier === "temporary" ? "进程退出即失效" : formatEpoch(license?.expires_at)}
        />
        <Field
          label="期限"
          value={license?.tier === "temporary"
            ? "当前后端进程"
            : formatDuration(license?.duration_value, license?.duration_unit)}
        />
      </div>
      <div className="temporary-license-action">
        <div>
          <strong>测试临时授权</strong>
          <span>{temporarySupported
            ? "由本机 Debug 构建直接批准，只在当前后端进程内有效；不访问外部授权服务，也不写入授权文件。"
            : "当前是 Release 构建，不开放临时权限；正式使用需要签名授权凭证。"}</span>
        </div>
        <button
          className="button"
          type="button"
          onClick={requestTemporary}
          disabled={requesting || license?.valid === true || !temporarySupported}
        >
          {requesting
            ? "正在申请…"
            : license?.valid
              ? "授权已生效"
              : temporarySupported
                ? "申请临时权限"
                : "当前构建不支持"}
        </button>
      </div>
      <div className="signed-license-action">
        <div>
          <strong>正式授权</strong>
          <span>优先使用 RS256 JWT，兼容旧版 NS1。界面不保存或回显凭证，后端会在受限权限文件中保留签名片段用于重启复核。</span>
        </div>
        <div className="signed-license-controls">
          <label className="config-field">
            <span>授权凭证</span>
            <input
              type="password"
              value={licenseInput}
              placeholder="粘贴 RS256 JWT 授权凭证"
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
        <button className="button compact-button" disabled={clearing} type="button" onClick={clearLicense}>
          {clearing ? "正在退出…" : clearArmed ? "确认退出授权" : "退出当前授权"}
        </button>
      ) : null}
      <div className={features.length > 0 ? "license-features" : "license-features empty"}>
        {features.length > 0
          ? features.map((feature) => (
              <span key={feature}>{formatFeature(feature)}</span>
            ))
          : <span>等待授权功能范围</span>}
      </div>
    </div>
  );
}

function formatTraceFragment(value: string | null | undefined): string {
  const normalized = value?.trim();
  if (!normalized) {
    return "未生成";
  }
  if (normalized.length <= 16) {
    return normalized.toUpperCase();
  }
  return `${normalized.slice(0, 8).toUpperCase()}-${normalized.slice(-6).toUpperCase()}`;
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
    second: "秒",
    seconds: "秒",
    day: "天",
    days: "天",
    month: "个月",
    months: "个月",
    year: "年",
    years: "年"
  };
  return `${value} ${labels[unit] ?? unit}`;
}

function formatCredentialFormat(format: string | null | undefined): string {
  const labels: Record<string, string> = {
    jwt_rs256: "JWT · RS256",
    legacy_ns1: "旧版 NS1",
    debug_session: "Debug 进程授权",
    legacy_document: "旧版授权文件"
  };
  return format ? labels[format] ?? format : "未授权";
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
