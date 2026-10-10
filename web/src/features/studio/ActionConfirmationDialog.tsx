import { useId, useRef, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";
import { useDialogInteraction } from "./dialogFocus";

export type ActionConfirmationRequest = {
  eyebrow: string;
  title: string;
  description: string;
  details?: string[];
  content?: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  handoffOnConfirm?: boolean;
  onConfirm: () => void | boolean | Promise<void | boolean>;
  onCancel?: () => void;
};

export function ActionConfirmationDialog({
  request,
  busy,
  error,
  onCancel,
  onConfirm
}: {
  request: ActionConfirmationRequest;
  busy: boolean;
  error?: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const titleId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  useDialogInteraction(dialogRef, true, onCancel, { blocked: busy });

  return (
    <div
      className="action-confirmation-layer"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onCancel();
      }}
    >
      <section
        aria-busy={busy}
        aria-labelledby={titleId}
        aria-describedby={`${titleId}-description`}
        aria-modal="true"
        className={request.content ? "action-confirmation-dialog has-content" : "action-confirmation-dialog"}
        ref={dialogRef}
        role="alertdialog"
        tabIndex={-1}
      >
        <header>
          <span className={request.danger ? "action-confirmation-icon danger" : "action-confirmation-icon"} aria-hidden="true">
            <NovaIcon name={request.danger ? "triangle-alert" : "help"} size={20} />
          </span>
          <div>
            <span className="class-config-eyebrow">{request.eyebrow}</span>
            <h2 id={titleId}>{request.title}</h2>
            <p id={`${titleId}-description`}>{request.description}</p>
          </div>
        </header>
        {request.details?.length ? (
          <ul>
            {request.details.map((detail) => <li key={detail}>{detail}</li>)}
          </ul>
        ) : null}
        {request.content}
        {error ? <p className="action-confirmation-error" role="alert">{error}</p> : null}
        <footer>
          <button className="console-button" disabled={busy} onClick={onCancel} type="button">
            取消
          </button>
          <button type="button"
            className={`console-button ${request.danger ? "danger" : "primary"}`}
            disabled={busy}
            onClick={onConfirm}
          >
            {busy ? "正在处理…" : request.confirmLabel}
          </button>
        </footer>
      </section>
    </div>
  );
}
