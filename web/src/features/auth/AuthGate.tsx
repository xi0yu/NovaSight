import {
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
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
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";

type AuthGateProps = {
  children: ReactNode;
};

type GatewayState = "checking" | "reachable" | "unreachable";
type DaemonState = "checking" | "reachable" | "unreachable";

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
    // Drop obsolete launcher credentials from old bookmarks, without using or storing them.
    if (new URLSearchParams(window.location.hash.slice(1)).has("access")) {
      window.history.replaceState(window.history.state, document.title,
        `${window.location.pathname}${window.location.search}`);
    }
    void getAuthSession(controller.signal)
      .then((next) => {
        setGatewayState("reachable");
        applySession(next);
      })
      .catch((error: unknown) => {
        if (typeof error === "object" && error !== null && "name" in error && error.name === "AbortError") {
          return;
        }
        setGatewayState(error instanceof ApiError ? "reachable" : "unreachable");
        setIssue(`无法连接授权服务：${getErrorMessage(error)}`);
      });
    return () => controller.abort();
  }, [applySession]);

  useEffect(() => {
    const requireAuthentication = () => {
      setSession(null);
      setDaemonState("checking");
      setIssue("会话已失效，请重新输入授权码。运行状态未改变。");
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
      setIssue("会话已到期，请重新输入授权码。运行状态未改变。");
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
      setIssue("请输入授权码。");
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
          code === "LICENSE_ACTIVATION_RATE_LIMITED"
            ? "验证次数过多，请等待 60 秒后重试。"
            : `验证失败：${getErrorMessage(error)}`
        );
      })
      .finally(() => setSubmitting(false));
  };

  const logout = () => {
    setSubmitting(true);
    void logoutAuthSession()
      .then(() => {
        setSession(null);
        setDaemonState("checking");
        setIssue("");
      })
      .catch((error: unknown) => {
        setIssue(`退出会话失败：${getErrorMessage(error)}`);
        reportError(error, { source: "auth-logout", title: "退出会话失败" });
      })
      .finally(() => setSubmitting(false));
  };

  if (session?.authenticated) {
    return (
      <div className="authenticated-workspace">
        <SessionBar session={session} daemonState={daemonState} issue={issue} loggingOut={submitting} onLogout={logout} />
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
            <h1 id="auth-title">授权后进入</h1>
          </div>
        </header>
        <p className="auth-intro">只需输入一次授权码，即可进入工作台。</p>
        <form onSubmit={submit}>
          <label htmlFor="web-access-code">授权码</label>
          <div className="auth-input-wrap">
            <input
              id="web-access-code"
              type={showAccessCode ? "text" : "password"}
              autoComplete="one-time-code"
              autoFocus
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
          <small id="auth-code-help">支持正式授权码或本次启动的临时授权码；不另设接入码。</small>
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
          <span>授权不会自动启动主链或开启物理输出</span>
        </footer>
      </section>
    </main>
  );
}

function SessionBar({
  session,
  daemonState,
  issue,
  loggingOut,
  onLogout
}: {
  session: AuthSession;
  daemonState: DaemonState;
  issue: string;
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
      {issue ? <span className="auth-session-issue" role="alert">{issue}</span> : null}
    </div>
  );
}
