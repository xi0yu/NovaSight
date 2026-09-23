import { useEffect, useId, useRef, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "./dialogFocus";

export function AdvancedSettingsDialog({
  open,
  eyebrow,
  title,
  description,
  footerNote,
  dirty = false,
  saving = false,
  saveError = null,
  onClose,
  onSave,
  children
}: {
  open: boolean;
  eyebrow: string;
  title: string;
  description?: string;
  footerNote: string;
  dirty?: boolean;
  saving?: boolean;
  saveError?: string | null;
  onClose: () => void;
  onSave: () => void;
  children: ReactNode;
}) {
  const titleId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  const savingRef = useRef(saving);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    savingRef.current = saving;
  }, [saving]);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.requestAnimationFrame(() => dialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!savingRef.current) {
          onCloseRef.current();
        }
      } else {
        trapDialogTabKey(event, dialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [open]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="advanced-settings-dialog-layer"
      onClick={(event) => {
        if (event.target === event.currentTarget && !saving) {
          onClose();
        }
      }}
    >
      <section
        aria-labelledby={titleId}
        aria-modal="true"
        aria-busy={saving}
        className="advanced-settings-dialog"
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <header className="advanced-settings-dialog-header">
          <div>
            <span className="class-config-eyebrow">{eyebrow}</span>
            <h2 id={titleId}>{title}</h2>
            {description ? <p>{description}</p> : null}
          </div>
          <button
            aria-label={`关闭${title}`}
            className="launch-dialog-close"
            disabled={saving}
            onClick={onClose}
            title={dirty ? "关闭并放弃本弹窗修改" : "关闭"}
            type="button"
          >
            <NovaIcon name="x-circle" size={18} />
          </button>
        </header>
        <div
          className="advanced-settings-dialog-body"
          {...({ inert: saving ? "" : undefined } as { inert?: string })}
        >
          {children}
        </div>
        <footer className="advanced-settings-dialog-footer">
          <span className={saveError ? "dialog-save-status error" : dirty ? "dialog-save-status dirty" : "dialog-save-status"} role="status" aria-live="polite">
            {saving
              ? "正在处理…"
              : saveError
                ? saveError
                : dirty
                  ? "修改尚未加入页面草稿"
                  : footerNote}
          </span>
          <button
            className={`console-button ${dirty ? "primary dialog-save-button" : "dialog-close-button"}`}
            disabled={saving}
            onClick={dirty ? onSave : onClose}
            type="button"
          >
            {dirty ? "加入草稿并关闭" : "关闭"}
          </button>
        </footer>
      </section>
    </div>
  );
}
