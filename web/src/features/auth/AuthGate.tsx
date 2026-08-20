import {
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";

import {
  ApiError,
  AuthSession,
  establishAuthSession,
  getApiErrorCode,
  getAuthSession,
  getHealth,
  logoutAuthSession
} from "../../api";
import { Button } from "../../components/ui";
import { NovaIcon } from "../../components/visual";
import { getErrorMessage } from "../shared/format";

type AuthGateProps = {
  children: ReactNode;
};

type GatewayState = "checking" | "reachable" | "unreachable";
type DaemonState = "checking" | "reachable" | "unreachable";

function accessCodeFromFragment(): string {
  const fragment = window.location.hash.startsWith("#")
    ? window.location.hash.slice(1)
    : window.location.hash;
  return new URLSearchParams(fragment).get("access")?.trim() ?? "";
}

function clearAccessFragment(): void {
  if (!window.location.hash) return;
  window.history.replaceState(
    window.history.state,
    document.title,
    `${window.location.pathname}${window.location.search}`
  );
}

function sessionExpired(session: AuthSession): boolean {
  return session.expires_at !== null && session.expires_at * 1000 <= Date.now();
}

export function AuthGate({ children }: AuthGateProps) {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [gatewayState, setGatewayState] = useState<GatewayState>("checking");
  const [daemonState, setDaemonState] = useState<DaemonState>("checking");
  const [accessCode, setAccessCode] = useState("");
  const [showAccessCode, setShowAccessCode] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [issue, setIssue] = useState("");
  const startupAccessCodeRef = useRef(accessCodeFromFragment());

  const verifyDaemon = useCallback(async () => {
    setDaemonState("checking");
    try {
      const health = await getHealth();
      setDaemonState(health.ok ? "reachable" : "unreachable");
    } catch {
      setDaemonState("unreachable");
    }
  }, []);

  const applySession = useCallback((next: AuthSession, unauthenticatedIssue = "") => {
    if (next.authenticated && !sessionExpired(next)) {
      setSession(next);
      setIssue("");
      void verifyDaemon();
      return;
    }
    setSession(null);
    setDaemonState("checking");
    setIssue(unauthenticatedIssue);
  }, [verifyDaemon]);

  const refreshSession = useCallback(async (failureMessage: string) => {
    try {
      const next = await getAuthSession();
      setGatewayState("reachable");
      applySession(next, failureMessage);
    } catch (error) {
      setGatewayState("unreachable");
      setSession(null);
      setIssue(`Web/API 安全层不可用：${getErrorMessage(error)}`);
    }
  }, [applySession]);

  useEffect(() => {
    const controller = new AbortController();
    const fragmentCode = startupAccessCodeRef.current;
    clearAccessFragment();
    const request = fragmentCode
      ? establishAuthSession(fragmentCode, controller.signal)
      : getAuthSession(controller.signal);
    void request
      .then((next) => {
        startupAccessCodeRef.current = "";
        setGatewayState("reachable");
        applySession(
          next,
          fragmentCode
            ? "自动验证未建立会话，请检查本次启动的接入码。"
            : ""
        );
      })
      .catch((error: unknown) => {
        if (typeof error === "object" && error !== null && "name" in error && error.name === "AbortError") {
          return;
        }
        startupAccessCodeRef.current = "";
        setGatewayState(error instanceof ApiError ? "reachable" : "unreachable");
        setIssue(
          fragmentCode
            ? `自动验证失败：${getErrorMessage(error)}`
            : `无法检查 Web 会话：${getErrorMessage(error)}`
        );
      });
    return () => controller.abort();
  }, [applySession]);

  useEffect(() => {
    const requireAuthentication = () => {
      setSession(null);
      setDaemonState("checking");
      setIssue("Web 会话已失效。运行态没有被修改，请重新验证当前浏览器。");
    };
    const refreshCsrf = () => {
      void refreshSession("会话安全令牌已轮换，请重试刚才的操作。");
    };
    window.addEventListener("novasight:auth-required", requireAuthentication);
    window.addEventListener("novasight:csrf-rejected", refreshCsrf);
    return () => {
      window.removeEventListener("novasight:auth-required", requireAuthentication);
      window.removeEventListener("novasight:csrf-rejected", refreshCsrf);
    };
  }, [refreshSession]);

  useEffect(() => {
    if (!session?.expires_at) return;
    const remaining = Math.max(0, session.expires_at * 1000 - Date.now());
    const timer = window.setTimeout(() => {
      setSession(null);
      setIssue("operator 会话已到期。运行态保持不变，请重新验证后继续操作。");
    }, remaining);
    return () => window.clearTimeout(timer);
  }, [session]);

  useEffect(() => {
    if (!session?.authenticated) return;
    const interval = window.setInterval(() => void verifyDaemon(), 30_000);
    return () => window.clearInterval(interval);
  }, [session?.authenticated, verifyDaemon]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const code = accessCode.trim();
    if (!code) {
      setIssue("请输入启动器本次生成的 Web 接入码。");
      return;
    }
    setSubmitting(true);
    setIssue("");
    void establishAuthSession(code)
      .then((next) => {
        setGatewayState("reachable");
        setAccessCode("");
        applySession(next, "Web/API 未建立 operator 会话。");
      })
      .catch((error: unknown) => {
        setGatewayState(error instanceof ApiError ? "reachable" : "unreachable");
        const code = getApiErrorCode(error);
        setIssue(
          code === "AUTH_RATE_LIMITED"
            ? "验证失败次数过多。请等待 60 秒后使用本次启动的新接入码重试。"
            : `验证失败：${getErrorMessage(error)}`
        );
      })
      .finally(() => setSubmitting(false));
  };

  const logout = () => {
    setSubmitting(true);
    void logoutAuthSession()
      .catch((error: unknown) => {
        setIssue(`退出会话失败：${getErrorMessage(error)}`);
      })
      .finally(() => {
        setSession(null);
        setDaemonState("checking");
        setSubmitting(false);
      });
  };

  if (session?.authenticated) {
    return (
      <div className="authenticated-workspace">
        <SessionBar session={session} daemonState={daemonState} loggingOut={submitting} onLogout={logout} />
        {children}
      </div>
    );
  }

  return (
    <main className="auth-shell">
      <section className="auth-card" aria-labelledby="auth-title">
        <header className="auth-card-heading">
          <span className="auth-brand-mark" aria-hidden="true">
            <NovaIcon name="prediction-line" size={22} strokeWidth={1.8} />
          </span>
          <div>
            <span className="auth-eyebrow">NovaSight Studio</span>
            <h1 id="auth-title">验证后继续</h1>
          </div>
        </header>
        <p className="auth-intro">输入启动器显示的本次接入码。</p>
        <form onSubmit={submit}>
          <label htmlFor="web-access-code">接入码</label>
          <div className="auth-input-wrap">
            <input
              id="web-access-code"
              type={showAccessCode ? "text" : "password"}
              autoComplete="one-time-code"
              autoFocus
              spellCheck={false}
              value={accessCode}
              onChange={(event) => setAccessCode(event.target.value)}
              disabled={submitting}
              aria-describedby="auth-code-help"
            />
            <button
              type="button"
              aria-label={showAccessCode ? "隐藏接入码" : "显示接入码"}
              onClick={() => setShowAccessCode((current) => !current)}
            >
              <NovaIcon name={showAccessCode ? "hide" : "show"} size={17} />
            </button>
          </div>
          <small id="auth-code-help">使用启动器提供的链接时会自动验证。</small>
          <Button variant="primary" type="submit" loading={submitting} leadingIcon="shield-check">
            进入控制台
          </Button>
        </form>
        {issue ? <div className="auth-issue" role="alert"><NovaIcon name="triangle-alert" size={16} /><span>{issue}</span></div> : null}
        {gatewayState === "unreachable" ? (
          <Button variant="secondary" onClick={() => void refreshSession("")}>
            重新检查连接
          </Button>
        ) : null}
        <footer className="auth-context">
          <span className="auth-gateway-state" data-state={gatewayState}>
            <i aria-hidden="true" />
            {gatewayState === "reachable" ? "服务已连接" : gatewayState === "unreachable" ? "服务不可达" : "正在检查连接"}
          </span>
          <span>验证不会改变设备运行状态</span>
        </footer>
      </section>
    </main>
  );
}

function SessionBar({
  session,
  daemonState,
  loggingOut,
  onLogout
}: {
  session: AuthSession;
  daemonState: DaemonState;
  loggingOut: boolean;
  onLogout: () => void;
}) {
  const expiresAt = session.expires_at
    ? new Date(session.expires_at * 1000).toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit" })
    : "—";
  return (
    <div className="auth-session-bar" role="status">
      <div className="auth-session-identity">
        <NovaIcon name="shield-check" size={15} />
        <strong>operator</strong>
        <span>受保护会话至 {expiresAt}</span>
      </div>
      <div className="auth-session-route" aria-label="认证链路状态">
        <span className="is-ready">浏览器</span><i />
        <span className="is-ready">Web/API</span><i />
        <span className={daemonState === "reachable" ? "is-ready" : daemonState === "unreachable" ? "is-error" : "is-checking"}>
          novasightd {daemonState === "reachable" ? "已连接" : daemonState === "unreachable" ? "不可达" : "检查中"}
        </span>
      </div>
      <Button variant="ghost" size="compact" onClick={onLogout} loading={loggingOut} leadingIcon="lock">
        退出会话
      </Button>
    </div>
  );
}
