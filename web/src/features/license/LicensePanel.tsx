import { type FormEvent, useCallback, useEffect, useState } from "react";

import {
  type LicenseStatus,
  clearLicenseKey,
  saveLicenseKey
} from "../../api";
import { Button, InlineError } from "../../components/ui";
import { reportError, reportSuccess } from "../../lib/toast";
import { NovaIcon } from "../../components/visual";
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
  const licenseValid = license?.valid === true;
  const temporarySupported = license?.temporary_access_supported === true;

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
      reportSuccess("设备授权已退出", "设备输出已先停止。", "license-clear", "feedback");
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
      reportSuccess("设备授权已更新", "有效期以服务端确认的到期时间为准。", "license-activate", "feedback");
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
