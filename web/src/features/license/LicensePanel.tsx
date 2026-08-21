import { type FormEvent, useCallback, useEffect, useState } from "react";

import {
  type LicenseStatus,
  clearLicenseKey,
  saveLicenseKey
} from "../../api";
import { Button, InlineError, StatusIndicator } from "../../components/ui";
import { reportError } from "../../lib/toast";
import { Field } from "../shared/Field";
import { formatEpoch } from "../shared/format";
import {
  describeLicenseActionFailure,
  type LicenseActionFailure
} from "./connectionIssue";

const LICENSE_CLEAR_CONFIRMATION_MESSAGE = "再次点击将先紧急停止当前设备并验证输出安全，再退出授权；5 秒后自动取消确认。";

export function LicensePanel({
  license,
  onLicenseChange
}: {
  license: LicenseStatus | null;
  onLicenseChange: (license: LicenseStatus) => void;
}) {
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
          <p>{license?.valid ? "授权状态有效，当前设备可以进入完整工作台。" : "临时授权码与正式许可证使用同一验证入口。"}</p>
        </div>
        <StatusIndicator tone={license?.valid ? "good" : "idle"}>
          {license?.valid ? "授权有效" : "尚未授权"}
        </StatusIndicator>
      </div>
      <div className={license?.valid ? "license-status-card valid" : "license-status-card"}>
        <div className="license-status-copy">
          <span>授权状态</span>
          <strong>{licenseStatusTitle}</strong>
          <p>
            授权凭证由 NovaSight 服务保存和校验；界面只展示追踪片段，不回显完整凭证。
          </p>
        </div>
        <details className="compact-settings-details">
          <summary>授权诊断信息 · 签发与凭证追踪 · 7 项</summary>
          <dl className="license-status-trace">
            <div>
              <dt>状态追踪码</dt>
              <dd>{statusTraceCode}</dd>
            </div>
            <div>
              <dt>授权凭据指纹</dt>
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
        </details>
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
          value={license?.credential_format === "ephemeral_code" ? "进程退出即失效" : formatEpoch(license?.expires_at)}
        />
        <Field
          label="期限"
          value={license?.credential_format === "ephemeral_code"
            ? "当前服务进程"
            : formatDuration(license?.duration_value, license?.duration_unit)}
        />
      </div>
      <LicenseActivationForm
        temporarySupported={temporarySupported}
        onLicenseChange={onLicenseChange}
      />
      {license?.configured ? (
        <button className="button compact-button" disabled={clearing} type="button" onClick={clearLicense}>
          {clearing ? "正在停止设备并退出…" : clearArmed ? "确认：先紧急停止设备，再退出授权" : "退出当前授权"}
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

export function LicenseActivationForm({
  temporarySupported,
  onLicenseChange,
  gate = false
}: {
  temporarySupported: boolean;
  onLicenseChange: (license: LicenseStatus) => void;
  gate?: boolean;
}) {
  const [credential, setCredential] = useState("");
  const [activating, setActivating] = useState(false);
  const [message, setMessage] = useState<string | undefined>();
  const [failure, setFailure] = useState<LicenseActionFailure | null>(null);

  const activate = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalized = credential.trim();
    if (!normalized) {
      setFailure({ title: "请输入授权码", message: "授权码不能为空。" });
      return;
    }
    setFailure(null);
    setMessage(undefined);
    setActivating(true);
    try {
      const status = await saveLicenseKey(normalized);
      onLicenseChange(status);
      if (!status.valid) {
        setFailure({
          title: "授权未生效",
          message: "该授权码已过期或当前不可用，请更换有效授权码。"
        });
        return;
      }
      setCredential("");
      setMessage(status.tier === "temporary"
        ? "本次进程临时授权已生效，重启后需要使用新临时码。"
        : "正式许可证已激活，完整凭证不会在界面中回显。");
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
  }, [credential, onLicenseChange]);

  return (
    <div className={gate ? "license-activation-form is-gate" : "license-activation-form"}>
      <div className="license-activation-copy">
        <strong>{gate ? "输入授权码" : "激活授权"}</strong>
        <span>{temporarySupported
          ? "可使用本次 Debug 启动生成的临时授权码，或正式签名许可证。两者经过同一服务端验证流程。"
          : "请输入正式签名许可证。授权码只发送给本机服务验证，不会在页面中保存或回显。"}</span>
      </div>
      <form className="license-activation-controls" onSubmit={activate}>
        <label className="config-field">
          <span>授权码</span>
          <input
            type="password"
            value={credential}
            placeholder={temporarySupported ? "临时授权码或正式许可证" : "正式许可证"}
            autoComplete="off"
            autoFocus={gate}
            spellCheck={false}
            disabled={activating}
            onChange={(event) => setCredential(event.target.value)}
          />
        </label>
        <Button variant="primary" type="submit" loading={activating} leadingIcon="shield-check">
          验证并继续
        </Button>
      </form>
      {failure ? <InlineError message={failure.message} title={failure.title} /> : null}
      {message ? <div className="inline-note">{message}</div> : null}
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
    ephemeral_code: "本次启动临时码",
    legacy_document: "旧版授权文件"
  };
  return format ? labels[format] ?? format : "未授权";
}

function formatFeature(feature: string): string {
  const labels: Record<string, string> = {
    capture: "真机采集",
    runtime: "运行控制",
    models: "模型仓库",
    plugins: "旧版插件能力",
    tensorrt: "TensorRT",
    hardware_control: "硬件控制",
    config_read: "读取配置",
    config_write: "写入配置"
  };
  return labels[feature] ?? feature;
}
