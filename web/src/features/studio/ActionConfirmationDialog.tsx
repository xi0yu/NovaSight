import { useEffect, useId, useRef, useState } from "react";

import { NovaIcon } from "../../components/visual";
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "./dialogFocus";

export type ActionConfirmationField = {
  key: string;
  label: string;
  current: string;
  recommended: string;
  selected: boolean;
};

export type ActionConfirmationRequest = {
  eyebrow: string;
  title: string;
  description: string;
  details?: string[];
  // Optional field-level diff. When present the dialog renders one row per
  // field with a checkbox; the user picks which to apply. Used for bulk
  // recommendations (e.g. kmNet defaults) where a blanket apply forces the
  // user to accept all-or-nothing. `onConfirm` receives the current set of
  // selected field keys (empty array if none are ticked).
  fields?: ActionConfirmationField[];
  confirmLabel: string;
  cancelLabel?: string;
  danger?: boolean;
  handoffOnConfirm?: boolean;
  onConfirm: (selectedKeys?: string[]) => void | boolean | Promise<void | boolean>;
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
  onConfirm: (selectedKeys?: string[]) => void;
}) {
  const titleId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  const busyRef = useRef(busy);
  const onCancelRef = useRef(onCancel);
  busyRef.current = busy;
  onCancelRef.current = onCancel;

  useEffect(() => {
    if (!request) return undefined;
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
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
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown, { capture: true });
      previousFocus?.focus();
    };
  }, [request]);

  // Local toggle state for the optional field picker. The parent provides
  // the initial `selected` value; the dialog owns the live state so toggles
  // don't round-trip through React state and re-create the request.
  // `fields` is hoisted so the early-return `if (!request) return null`
  // doesn't lose the reference; we re-read `request?.fields` when needed.
  const [fieldSelection, setFieldSelection] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    for (const field of request?.fields ?? []) {
      initial[field.key] = field.selected;
    }
    return initial;
  });
  useEffect(() => {
    // Resync when a new request replaces the previous one (e.g. user opens
    // the dialog again with different defaults).
    const next: Record<string, boolean> = {};
    for (const field of request?.fields ?? []) {
      next[field.key] = field.selected;
    }
    setFieldSelection(next);
  }, [request]);

  // When the request has no `fields`, the selectedKeys argument is irrelevant.
  // When `fields` is present, pass the current selection to the parent so the
  // parent can decide which subset of the bulk apply to execute.
  const getSelectedKeys = (): string[] => {
    const currentFields = request?.fields;
    if (!currentFields?.length) return [];
    return currentFields
      .map((item) => item.key)
      .filter((key) => fieldSelection[key] ?? request?.fields?.find((f) => f.key === key)?.selected ?? false);
  };

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
        {request.fields?.length ? (
          <ul className="action-confirmation-fields" role="group" aria-label="可选应用字段">
            {request.fields.map((field) => {
              const checked = fieldSelection[field.key] ?? field.selected;
              return (
                <li key={field.key}>
                  <label className={checked ? "selected" : ""}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={busy}
                      onChange={() => {
                        setFieldSelection((prev) => ({
                          ...prev,
                          [field.key]: !checked
                        }));
                      }}
                    />
                    <span className="action-confirmation-field-label">{field.label}</span>
                    <span className="action-confirmation-field-diff">
                      <s>{field.current}</s>
                      <span aria-hidden="true">→</span>
                      <b>{field.recommended}</b>
                    </span>
                  </label>
                </li>
              );
            })}
          </ul>
        ) : null}
        {error ? <p className="action-confirmation-error" role="alert">{error}</p> : null}
        <footer>
          <button className="console-button" disabled={busy} onClick={onCancel} type="button">
            {request.cancelLabel ?? "取消"}
          </button>
          <button type="button"
            className={`console-button ${request.danger ? "danger" : "primary"}`}
            disabled={busy}
            onClick={() => onConfirm(getSelectedKeys())}
          >
            {busy ? "正在处理…" : request.confirmLabel}
          </button>
        </footer>
      </section>
    </div>
  );
}
