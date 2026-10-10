import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  type LicenseStatus,
  clearLicenseKey,
  getLicenseStatus,
  saveLicenseKey
} from "../../api";
import { Button, InlineError, StatusIndicator } from "../../components/ui";
import { reportError } from "../../lib/toast";
import { NovaIcon, type NovaIconName } from "../../components/visual";
import type { LicenseFeature } from "../../contracts/license";
import { formatEpoch } from "../shared/format";
import { formatLicenseCategory } from "./format";
import {
  describeLicenseActionFailure,
  type LicenseActionFailure
} from "./connectionIssue";

const FEATURE_GROUPS: { title: string; icon: NovaIconName; features: LicenseFeature[] }[] = [
  { title: "采集与运行", icon: "capture", features: ["capture", "runtime"] },
  { title: "模型与推理", icon: "ai-model", features: ["models", "tensorrt", "plugins"] },
  { title: "控制与配置", icon: "device-send", features: ["hardware_control", "config_read", "config_write"] },
];

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
  const [now, setNow] = useState(Date.now);
  const expiresAt = license?.expires_at ? license.expires_at * 1000 : null;
  const remaining = expiresAt === null ? null : Math.max(0, Math.ceil((expiresAt - now) / 1000));
  const expiredNotified = useRef<number | null>(null);
  useEffect(() => {
    if (expiresAt === null) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [expiresAt]);
  useEffect(() => {
    if (license?.valid && expiresAt !== null && now >= expiresAt && expiredNotified.current !== expiresAt) {
      expiredNotified.current = expiresAt;
      void getLicenseStatus().then(onLicenseChange).catch(() => { expiredNotified.current = null; });
    }
  }, [now, expiresAt, license, onLicenseChange]);
  const licenseValid = license?.valid === true;
  const temporarySupported = license?.temporary_access_supported === true;
  const features = licenseValid ? license.features : [];

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
  }, [license?.token_id, licenseValid]);

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
        exposeStatus: false,
        popup: false,
      });
    } finally {
      setClearing(false);
      setClearArmed(false);
    }
  }, [clearArmed, onLicenseChange]);

  return (
    <div className="license-panel">
      <InlineError message={failure?.message} title={failure?.title} />
      {message ? <div className="inline-note" role="status">{message}</div> : null}
      <section className="license-summary" aria-label="设备授权摘要">
        <header>
          <div className="license-summary-title"><NovaIcon name="shield-check" size={24} /><h3>设备授权</h3></div>
          <StatusIndicator tone={licenseValid ? "good" : "idle"}>
            {licenseValid ? "授权有效" : license ? "尚未授权" : "授权待核实"}
          </StatusIndicator>
        </header>
        <dl className="license-device-expiry" aria-label="设备授权状态">
          <div className="license-deadline"><dt>到期时间</dt><dd>{license?.expires_at ? formatEpoch(license.expires_at) : "尚未确认"}<small className="license-expiry-countdown">{remaining === null ? "到期时间待核实" : remaining === 0 ? (licenseValid ? "正在核验到期状态" : "授权不可用") : `倒计时 ${String(Math.floor(remaining / 3600)).padStart(2, "0")}:${String(Math.floor(remaining / 60) % 60).padStart(2, "0")}:${String(remaining % 60).padStart(2, "0")}`}</small></dd></div>
          <div><dt>设备模型 ID</dt><dd>{license?.device_model || "设备型号待核实"}</dd></div>
          <div><dt>授权类别</dt><dd><span className="license-category">{formatLicenseCategory(license)}</span></dd></div>
        </dl>
      </section>
      <section className="license-scope" aria-labelledby="license-scope-title">
        <h3 id="license-scope-title">授权范围</h3>
        {features.length > 0 ? <div className="license-scope-grid">
          {FEATURE_GROUPS.map((group) => {
            const enabled = group.features.filter((feature) => features.includes(feature));
            return enabled.length > 0 ? <article key={group.title}>
              <h4><NovaIcon name={group.icon} size={20} />{group.title}</h4>
              <ul>{enabled.map((feature) => <li key={feature}><NovaIcon name="check-circle" size={16} />{formatFeature(feature)}</li>)}</ul>
            </article> : null;
          })}
        </div> : <p className="license-scope-empty">{licenseValid ? "此授权未包含可用功能。" : license ? "授权生效后，可在这里查看可用功能。" : "正在核实授权范围。"}</p>}
      </section>
      <section className="license-management" aria-labelledby="license-management-title">
        <h3 id="license-management-title">授权管理</h3>
        <details className="license-change-disclosure" open={!licenseValid || undefined}>
          <summary>{licenseValid ? "更换授权码" : "输入授权码"}<NovaIcon name="expand" size={18} /></summary>
          <LicenseActivationForm
            active={licenseValid}
            temporarySupported={temporarySupported}
            onLicenseChange={onLicenseChange}
          />
        </details>
        {license?.configured ? <div className="license-exit">
          <p>退出授权将先停止设备输出。</p>
          <Button loading={clearing} onClick={clearLicense}>
            {clearing ? "正在停止设备并退出…" : clearArmed ? "确认：先紧急停止设备，再退出授权" : "退出当前授权"}
          </Button>
        </div> : null}
      </section>
    </div>
  );
}

export function LicenseActivationForm({
  active = false,
  temporarySupported,
  onLicenseChange,
  gate = false
}: {
  active?: boolean;
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
    if (!normalized && !temporarySupported) {
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
        ? "内测授权已生效，有效期内无需重复授权，原期限不会重置。"
        : "正式许可证已激活，完整凭证不会在界面中回显。");
    } catch (err) {
      const nextFailure = describeLicenseActionFailure(err, "activate");
      setFailure(nextFailure);
      reportError(err, {
        source: "license-activate",
        title: nextFailure.title,
        publicDetail: nextFailure.message,
        exposeStatus: false,
        popup: false,
      });
    } finally {
      setActivating(false);
    }
  }, [credential, onLicenseChange, temporarySupported]);

  return (
    <div className={gate ? "license-activation-form is-gate" : "license-activation-form"}>
      <div className="license-activation-copy">
        <strong>{gate ? "输入授权码" : active ? "更换授权" : "激活授权"}</strong>
        <span>{temporarySupported
          ? "留空即可申请 1 天内测授权；已有有效授权沿用原到期时间。"
          : "请输入正式签名许可证。授权码只发送给本机服务验证，不会在页面中保存或回显。"}</span>
      </div>
      <form className="license-activation-controls" onSubmit={activate}>
        <label className="config-field">
          <span>授权码</span>
          <input
            type="password"
            value={credential}
            placeholder={temporarySupported ? "留空申请内测授权，或填写正式许可证" : "正式许可证"}
            autoComplete="off"
            autoFocus={gate}
            spellCheck={false}
            disabled={activating}
            onChange={(event) => {
              setCredential(event.target.value);
              setFailure(null);
              setMessage(undefined);
            }}
          />
        </label>
        <Button variant="primary" type="submit" loading={activating} leadingIcon="shield-check">
          {active ? "验证并更换" : "验证并继续"}
        </Button>
      </form>
      {failure ? <InlineError message={failure.message} title={failure.title} /> : null}
      {message ? <div className="inline-note">{message}</div> : null}
    </div>
  );
}

function formatFeature(feature: string): string {
  const labels: Record<string, string> = {
    capture: "画面采集",
    runtime: "运行控制",
    models: "模型仓库",
    plugins: "插件扩展",
    tensorrt: "TensorRT 加速",
    hardware_control: "硬件控制",
    config_read: "读取配置",
    config_write: "写入配置"
  };
  return labels[feature] ?? feature;
}
