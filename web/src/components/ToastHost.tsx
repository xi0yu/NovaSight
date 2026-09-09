import { useToasts, useDismissToast } from "../lib/toast";

const TONE_TITLE: Record<"info" | "warn" | "error" | "success", string> = {
  info: "提示",
  warn: "请注意",
  error: "操作失败",
  success: "已完成"
};

export function ToastHost() {
  const toasts = useToasts();
  const dismiss = useDismissToast();

  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="toast-host" role="region" aria-label="系统提示">
      <output className="toast-host-list" aria-live="polite" aria-relevant="additions">
        {toasts.map((toast) => (
          <article
            key={toast.id}
            className={`toast-card tone-${toast.tone}`}
            data-source={toast.source}
            data-status={toast.status ?? ""}
          >
            <div className="toast-card-head">
              <span className="toast-card-tag" aria-hidden="true">
                {TONE_TITLE[toast.tone]}
              </span>
              <span className="toast-card-source">{toast.source}</span>
              <button
                type="button"
                className="toast-card-close"
                onClick={() => dismiss(toast.id)}
                aria-label="关闭提示"
              >
                ×
              </button>
            </div>
            <p className="toast-card-title">{toast.title}</p>
            {toast.detail ? <p className="toast-card-detail">{toast.detail}</p> : null}
          </article>
        ))}
      </output>
    </div>
  );
}
