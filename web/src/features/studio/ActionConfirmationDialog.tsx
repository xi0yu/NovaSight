import { useEffect, useId, useRef } from "react";

import { NovaIcon } from "../../components/visual";
import { trapDialogTabKey } from "./dialogFocus";

export type ActionConfirmationRequest = {
  eyebrow: string;
  title: string;
  description: string;
  details?: string[];
  confirmLabel: string;
  cancelLabel?: string;
  danger?: boolean;
  handoffOnConfirm?: boolean;
  onConfirm: () => void | boolean | Promise<void | boolean>;
};

export function ActionConfirmationDialog({
  request,
  busy,
  error,
  onCancel,
  onConfirm
}: {
  request: ActionConfirmationRequest | null;
  busy: boolean;
  error?: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const titleId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  const busyRef = useRef(busy);
  const onCancelRef = useRef(onCancel);
  busyRef.current = busy;
  onCancelRef.current = onCancel;

  useEffect(() => {
    if (!request) return undefined;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const frame = window.requestAnimationFrame(() => dialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busyRef.current) {
        event.preventDefault();
        event.stopImmediatePropagation();
        onCancelRef.current();
      } else if (event.key === "Tab") {
        event.stopImmediatePropagation();
        trapDialogTabKey(event, dialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown, { capture: true });
    return () => {
      window.cancelAnimationFrame(frame);
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown, { capture: true });
      previousFocus?.focus();
    };
  }, [request]);

  if (!request) return null;

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
        aria-modal="true"
        className="action-confirmation-dialog"
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
            <p>{request.description}</p>
          </div>
        </header>
        {request.details?.length ? (
          <ul>
            {request.details.map((detail) => <li key={detail}>{detail}</li>)}
          </ul>
        ) : null}
        {error ? <p className="action-confirmation-error" role="alert">{error}</p> : null}
        <footer>
          <button className="console-button" disabled={busy} onClick={onCancel} type="button">
            {request.cancelLabel ?? "取消"}
          </button>
          <button
            className={`console-button ${request.danger ? "danger" : "primary"}`}
            disabled={busy}
            onClick={onConfirm}
            type="button"
          >
            {busy ? "正在处理…" : request.confirmLabel}
          </button>
        </footer>
      </section>
    </div>
  );
}
