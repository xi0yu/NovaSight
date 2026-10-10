import { useEffect, useState } from "react";
import { TOAST_DURATION_MS, useToasts, useDismissToast } from "../lib/toast";
import { NovaIcon } from "./visual";

const TONE_TITLE: Record<"info" | "warn" | "error" | "success", string> = {
  info: "提示",
  warn: "请注意",
  error: "操作失败",
  success: "已完成"
};

function ToastTimer({ createdAt }: { createdAt: number }) {
  const [mountedAt] = useState(Date.now);
  return <span className="toast-card-timer" aria-hidden="true"><span style={{ animationDuration: `${TOAST_DURATION_MS}ms`, animationDelay: `${Math.min(0, createdAt - mountedAt)}ms` }} /></span>;
}

export function ToastHost() {
  const toasts = useToasts();
  const dismiss = useDismissToast();
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (toasts.length === 0) return;
    setNow(Date.now());
    const interval = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(interval);
  }, [toasts.length]);

  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="toast-host" role="region" aria-label="系统提示">
      <output className="toast-host-list" aria-live="polite" aria-relevant="additions text">
        {toasts.map((toast) => (
          <article
            key={toast.id}
            className={`toast-card tone-${toast.tone}`}
            data-source={toast.source}
            data-status={toast.status ?? ""}
          >
            <div className="toast-card-head">
              <NovaIcon name={toast.tone === "success" ? "check-circle" : toast.tone === "error" ? "triangle-alert" : "notification"} size={20} aria-hidden="true" />
              <span className="toast-card-tag" aria-hidden="true">
                {TONE_TITLE[toast.tone]}
              </span>
              <span className="toast-card-countdown" aria-hidden="true">{Math.max(0, Math.min(TOAST_DURATION_MS / 1000, Math.ceil((toast.createdAt + TOAST_DURATION_MS - now) / 1000)))} 秒后关闭</span>
              <button
                type="button"
                className="toast-card-close"
                onClick={() => dismiss(toast.id)}
                aria-label="关闭提示"
              >
                ×
              </button>
            </div>
            <p className="toast-card-title">{toast.title}{toast.count > 1 ? <small className="toast-card-repeat"> · {toast.count} 次</small> : null}</p>
            {toast.detail ? <p className="toast-card-detail">{toast.detail}</p> : null}
            <ToastTimer key={toast.createdAt} createdAt={toast.createdAt} />
            {toast.requestId ? <details className="toast-card-request-id"><summary>排查信息</summary><span>来源：{toast.source}<br />编号：{toast.requestId}</span></details> : null}
          </article>
        ))}
      </output>
    </div>
  );
}
