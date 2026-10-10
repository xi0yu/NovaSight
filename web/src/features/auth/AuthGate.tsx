import {
  type FormEvent,
  type ReactNode,
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
  useRef
} from "react";

import {
  ApiError,
  AuthSession,
  establishAuthSession,
  getApiErrorCode,
  recoverAuthSession,
  setBusinessConfirmed,
  getHealth,
  logoutAuthSession
} from "../../api";
import { Button, LoadedImage } from "../../components/ui";
import { StartupScreen } from "../../components/StartupScreen";
import { NovaIcon } from "../../components/visual";
import { reportError, reportInfo } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import { transitionAuth, type AuthAuthority } from "./authStateMachine";
import type { AuthState } from "../../contracts/auth";
import daylightArt from "../../assets/themes/nova-daylight.webp";

type AuthGateProps = {
  children: ReactNode;
};

type GatewayState = "checking" | "reachable" | "unreachable";
type DaemonState = "checking" | "reachable" | "unreachable";

const WorkspaceRecovery = createContext({ revision: 0, confirm: (_ready: boolean, _revision: number) => {} });
export const useWorkspaceRecovery = () => useContext(WorkspaceRecovery);
const WorkspaceAuthorization = createContext(true);
export const useWorkspaceAuthorization = () => useContext(WorkspaceAuthorization);

export function AuthGate({ children }: AuthGateProps) {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [businessReady, setBusinessReady] = useState(false);
  const [revision, setRevision] = useState(0);
  const revisionRef = useRef(0);
  const stateRef = useRef<AuthState>("checking");
  const sessionRef = useRef<AuthSession | null>(null);
  const requestRef = useRef(0);
  const beginBusinessRecovery = useCallback(() => {
    revisionRef.current += 1;
    setRevision(revisionRef.current);
    setBusinessReady(false);
    setBusinessConfirmed(false);
  }, []);
  const transition = useCallback((next: AuthState, authority: AuthAuthority = "client") => {
    const allowed = transitionAuth(stateRef.current, next, authority);
    stateRef.current = allowed;
    setAuthState(allowed);
  }, []);
  const confirm = useCallback((ready: boolean, evidenceRevision: number) => {
    if (evidenceRevision !== revisionRef.current || !sessionRef.current?.authenticated || stateRef.current === "reconnecting") return;
    setBusinessReady(ready);
    setBusinessConfirmed(ready);
  }, []);
  const [workspaceOpened, setWorkspaceOpened] = useState(false);
  const [gatewayState, setGatewayState] = useState<GatewayState>("checking");
  const [daemonState, setDaemonState] = useState<DaemonState>("checking");
  const [accessCode, setAccessCode] = useState("");
  const [showAccessCode, setShowAccessCode] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [issue, setIssue] = useState("");

  const verifyDaemon = useCallback(async () => {
    try {
      const health = await getHealth();
      setDaemonState(health.ok ? "reachable" : "unreachable");
      if (!health.ok) throw new Error("服务不可达");
    } catch {
      setDaemonState("unreachable");
      if (stateRef.current !== "reconnecting") beginBusinessRecovery();
      transition("reconnecting");
    }
  }, [beginBusinessRecovery, transition]);

  const applySession = useCallback((next: AuthSession, userConfirmed = false) => {
    const nextState = next.auth_state ?? (next.authenticated ? "authorized" : "required");
    if (nextState === "checking") { transition("checking"); return; }
    if (next.authenticated) {
      if (transitionAuth(stateRef.current, "authorized", userConfirmed ? "user-confirmed" : "server") !== "authorized") return;
      if (!sessionRef.current?.authenticated || stateRef.current === "reconnecting") beginBusinessRecovery();
      sessionRef.current = next;
      setWorkspaceOpened(true);
      setSession(next);
      transition("authorized", userConfirmed ? "user-confirmed" : "server");
      setIssue("");
      void verifyDaemon();
      return;
    }
    requestRef.current += 1;
    sessionRef.current = null;
    setSession(null);
    setBusinessReady(false);
    setBusinessConfirmed(false);
    transition(nextState, nextState === "signed_out" ? "user" : "server");
    setIssue(nextState === "expired" ? "授权已到期，请提供有效授权码。" : nextState === "revoked" ? "授权已撤销，请提供有效授权码。" : "");
  }, [beginBusinessRecovery, transition, verifyDaemon]);

  const refreshSession = useCallback(async (force = false) => {
    if (["required", "expired", "revoked", "signed_out"].includes(stateRef.current)) return;
    const request = ++requestRef.current;
    try {
      const next = await recoverAuthSession(force);
      if (request !== requestRef.current) return;
      setGatewayState("reachable");
      applySession(next);
    } catch (error) {
      if (request !== requestRef.current || (error instanceof DOMException && error.name === "AbortError")) return;
      setGatewayState("unreachable");
      if (stateRef.current !== "reconnecting") beginBusinessRecovery();
      transition("reconnecting");
    }
  }, [applySession, beginBusinessRecovery, transition]);

  useEffect(() => {
    if (new URLSearchParams(window.location.hash.slice(1)).has("access")) {
      window.history.replaceState(window.history.state, document.title, `${window.location.pathname}${window.location.search}`);
    }
    void refreshSession();
    return () => { requestRef.current += 1; };
  }, [refreshSession]);

  useEffect(() => {
    const received = (event: Event) => {
      const next = (event as CustomEvent<AuthSession & { user_initiated?: boolean }>).detail;
      applySession(next, next.user_initiated === true);
    };
    const checking = () => {
      if (!["required", "expired", "revoked", "signed_out"].includes(stateRef.current)) transition("checking");
    };
    const refreshing = () => transition("refreshing");
    const userAuthorization = () => transition("checking", "user");
    const reconnecting = () => {
      if (["required", "expired", "revoked", "signed_out"].includes(stateRef.current)) return;
      if (stateRef.current !== "reconnecting") beginBusinessRecovery();
      transition("reconnecting");
    };
    const recheck = () => { void refreshSession(true); };
    const wake = () => {
      if (document.visibilityState === "hidden" || ["required", "expired", "revoked", "signed_out"].includes(stateRef.current)) return;
      beginBusinessRecovery();
      transition("checking");
      void refreshSession();
    };
    window.addEventListener("novasight:auth-refreshing", refreshing);
    window.addEventListener("novasight:auth-user-authorization", userAuthorization);
    window.addEventListener("novasight:auth-session", received);
    window.addEventListener("novasight:auth-checking", checking);
    window.addEventListener("novasight:auth-reconnecting", reconnecting);
    window.addEventListener("novasight:auth-required", recheck);
    window.addEventListener("online", wake);
    window.addEventListener("focus", wake);
    window.addEventListener("offline", reconnecting);
    document.addEventListener("visibilitychange", wake);
    return () => {
      window.removeEventListener("novasight:auth-refreshing", refreshing);
      window.removeEventListener("novasight:auth-user-authorization", userAuthorization);
      window.removeEventListener("novasight:auth-session", received);
      window.removeEventListener("novasight:auth-checking", checking);
      window.removeEventListener("novasight:auth-reconnecting", reconnecting);
      window.removeEventListener("novasight:auth-required", recheck);
      window.removeEventListener("online", wake);
      window.removeEventListener("focus", wake);
      window.removeEventListener("offline", reconnecting);
      document.removeEventListener("visibilitychange", wake);
    };
  }, [applySession, beginBusinessRecovery, refreshSession, transition]);

  useEffect(() => {
    if (!session?.expires_at) return;
    // Server-relative time is a wake-up hint, never terminal authorization evidence.
    const remaining = session.expires_at - (session.server_time ?? Math.floor(Date.now() / 1000));
    const timer = window.setTimeout(() => void refreshSession(true), Math.max(1000, Math.min(2147483647, (remaining - 60) * 1000)));
    return () => window.clearTimeout(timer);
  }, [session, refreshSession]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (stateRef.current === "reconnecting" || stateRef.current === "checking") void refreshSession();
      else if (stateRef.current === "authorized") void verifyDaemon();
    }, 5000);
    return () => window.clearInterval(interval);
  }, [refreshSession, verifyDaemon]);

  const remindedDeadlineRef = useRef<number | null>(null);
  useEffect(() => {
    const deadline = session?.license_expires_at;
    if (!session?.authenticated || !deadline || session.server_time === undefined || document.visibilityState === "hidden") return;
    const remaining = deadline - session.server_time;
    const key = "novasight.license-expiry-notice";
    let reminded = remindedDeadlineRef.current === deadline;
    try { reminded ||= window.sessionStorage.getItem(key) === String(deadline); } catch { /* In-memory deduplication remains available. */ }
    if (reminded || remaining <= 0) return;
    if (remaining <= 3600) {
      remindedDeadlineRef.current = deadline;
      try { window.sessionStorage.setItem(key, String(deadline)); } catch { /* Storage may be disabled. */ }
      reportInfo("授权将在 1 小时内到期", "到期后采集与输出会停止，请提前保存配置。", "license-expiry", "feedback");
      return;
    }
    // A local timer asks the server; it never proves license expiry or eligibility for a warning.
    const timer = window.setTimeout(() => { void recoverAuthSession().catch(() => undefined); }, Math.min(2147483647, (remaining - 3600) * 1000));
    return () => window.clearTimeout(timer);
  }, [session]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const code = accessCode.trim();
    requestRef.current += 1;
    setSubmitting(true);
    setIssue("");
    void establishAuthSession(code)
      .then((next) => {
        setGatewayState("reachable");
        setAccessCode("");
        applySession(next, true);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setGatewayState(error instanceof ApiError ? "reachable" : "unreachable");
        const code = getApiErrorCode(error);
        setIssue(
          code === "LICENSE_ACTIVATION_RATE_LIMITED"
            ? "验证次数过多，请等待 60 秒后重试。"
            : `验证失败：${getErrorMessage(error)}`
        );
      })
      .finally(() => setSubmitting(false));
  };

  const logout = () => {
    requestRef.current += 1;
    transition("signed_out", "user");
    setSession(null);
    sessionRef.current = null;
    setBusinessConfirmed(false);
    setBusinessReady(false);
    setSubmitting(true);
    void logoutAuthSession().catch((error: unknown) => {
      setIssue("已退出此页面，服务端撤销尚未确认；连接恢复后可重试退出。");
      reportError(error, { source: "auth-logout", title: "服务端退出尚未确认", popup: false });
    }).finally(() => setSubmitting(false));
  };

  const authenticated = session?.authenticated === true && !["required", "expired", "revoked", "signed_out"].includes(authState);
  const needsAuthorization = ["required", "expired", "revoked", "signed_out"].includes(authState);
  const login = !needsAuthorization ? null : (
    <main className="auth-shell welcome-shell">
      <section className="welcome-story" aria-labelledby="welcome-title">
        <LoadedImage className="welcome-character" src={daylightArt} width="1536" height="1024" alt="" aria-hidden="true" />
        <header className="welcome-brand">
          <span className="auth-brand-mark" aria-hidden="true">
            <NovaIcon name="prediction-line" size={22} strokeWidth={1.8} />
          </span>
          <strong>NovaSight</strong>
          <span>Edge AI Vision</span>
        </header>
        <div className="welcome-copy">
          <span className="auth-eyebrow">你的视觉工作台</span>
          <h1 id="welcome-title">你好，<br />欢迎来到 NovaSight。</h1>
          <p>连接画面，调整到你的节奏。<br />从这里，开始新的体验。</p>
        </div>
        <ol className="welcome-sequence" aria-label="开始使用的三个步骤">
          <li><span>01</span><NovaIcon name="capture" size={18} /><strong>连接画面</strong></li>
          <li><span>02</span><NovaIcon name="models" size={18} /><strong>选择模型</strong></li>
          <li><span>03</span><NovaIcon name="play-circle" size={18} /><strong>安全运行</strong></li>
        </ol>
        <div className="welcome-live-note">
          <span><i aria-hidden="true" />实时状态</span>
          <p>授权后读取这台设备的真实运行状态。</p>
        </div>
      </section>
      <section className="auth-card" aria-labelledby="auth-title">
        <header className="auth-card-heading">
          <div>
            <span className="auth-eyebrow">开始使用</span>
            <h2 id="auth-title">设备授权</h2>
          </div>
        </header>
        <p className="auth-intro">只需一次授权。进入后会先带你核对画面、模型与运行状态。</p>
        <form onSubmit={submit}>
          <label htmlFor="web-access-code">授权码</label>
          <div className="auth-input-wrap">
            <input
              id="web-access-code"
              type={showAccessCode ? "text" : "password"}
              autoComplete="one-time-code"
              spellCheck={false}
              value={accessCode}
              onChange={(event) => {
                setAccessCode(event.target.value);
                setIssue("");
              }}
              disabled={submitting}
              aria-describedby="auth-code-help"
            />
            <button
              type="button"
              aria-label={showAccessCode ? "隐藏授权码" : "显示授权码"}
              onClick={() => setShowAccessCode((current) => !current)}
            >
              <NovaIcon name={showAccessCode ? "hide" : "show"} size={17} />
            </button>
          </div>
          <small id="auth-code-help">内测版留空即可申请 1 天授权；正式版须填写授权码。</small>
          <Button variant="primary" type="submit" loading={submitting} leadingIcon="shield-check">
            进入控制台
          </Button>
        </form>
        {issue ? <div className="auth-issue" role="alert"><NovaIcon name="triangle-alert" size={16} /><span>{issue}</span></div> : null}
        {gatewayState === "unreachable" ? (
          <Button variant="secondary" onClick={() => { transition("checking"); void refreshSession(); }}>
            重新检查连接
          </Button>
        ) : null}
        <footer className="auth-context">
          <span className="auth-gateway-state" data-state={gatewayState}>
            <i aria-hidden="true" />
            {gatewayState === "reachable" ? "服务已连接" : gatewayState === "unreachable" ? "服务不可达" : "正在检查连接"}
          </span>
          <span>授权不会自动启动主链或开启物理输出</span>
        </footer>
      </section>
    </main>
  );
  return <WorkspaceAuthorization.Provider value={authenticated && authState !== "reconnecting"}>
    <WorkspaceRecovery.Provider value={{ revision, confirm }}>
      {workspaceOpened ? <div className="authenticated-workspace" hidden={needsAuthorization} data-auth-state={authState}>
        {session ? <SessionBar daemonState={daemonState} issue={issue} loggingOut={submitting} onLogout={logout} /> : null}
        {!businessReady ? <aside className="auth-recovery-notice" role="status">
          <NovaIcon name="activity-pulse" size={18} />
          <span>{authState === "reconnecting" ? "正在恢复连接，页面内容已保留" : "正在核对设备状态"}</span>
          <Button variant="ghost" size="compact" onClick={() => { beginBusinessRecovery(); void refreshSession(true); }}>重新连接</Button>
        </aside> : null}
        {children}
      </div> : null}
      {!workspaceOpened && !needsAuthorization ? <StartupScreen message={authState === "reconnecting" ? "正在恢复连接，你的授权与页面会保留" : undefined}>
        {authState === "reconnecting" ? <Button variant="secondary" onClick={() => void refreshSession()}>重新连接</Button> : null}
      </StartupScreen> : null}
      {login}
    </WorkspaceRecovery.Provider>
  </WorkspaceAuthorization.Provider>;
}

function SessionBar({
  daemonState,
  issue,
  loggingOut,
  onLogout
}: {
  daemonState: DaemonState;
  issue: string;
  loggingOut: boolean;
  onLogout: () => void;
}) {
  return (
    <div className="auth-session-bar" role="status">
      <div className="auth-session-identity" title="设备授权详情见关于">
        <NovaIcon name="shield-check" size={15} />
        <strong>operator</strong>
        <span>已授权</span>
      </div>
      {daemonState === "reachable" ? null : (
        <div className="auth-session-route" aria-label="服务连接状态">
          <span className={daemonState === "unreachable" ? "is-error" : "is-checking"}>
            {daemonState === "unreachable" ? "服务不可达" : "正在检查服务"}
          </span>
        </div>
      )}
      <Button variant="ghost" size="compact" onClick={onLogout} loading={loggingOut} leadingIcon="lock">
        退出登录
      </Button>
      {issue ? <span className="auth-session-issue" role="alert">{issue}</span> : null}
    </div>
  );
}
