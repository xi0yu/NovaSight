import { useEffect, useId, useRef, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";
import { trapDialogTabKey } from "./dialogFocus";

export function AdvancedSettingsDialog({
  open,
  eyebrow,
  title,
  description,
  footerNote,
  saving = false,
  saveError = null,
  onClose,
  children
}: {
  open: boolean;
  eyebrow: string;
  title: string;
  description: string;
  footerNote: string;
  saving?: boolean;
  saveError?: string | null;
  onClose: () => void;
  children: ReactNode;
}) {
  const titleId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  const savingRef = useRef(saving);

  useEffect(() => {
    savingRef.current = saving;
  }, [saving]);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => dialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!savingRef.current) {
          onClose();
        }
      } else {
        trapDialogTabKey(event, dialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [onClose, open]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="advanced-settings-dialog-layer"
      onMouseDown={(event) => {
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
            <p>{description}</p>
          </div>
          <button aria-label={`关闭${title}`} className="launch-dialog-close" disabled={saving} onClick={onClose} type="button">
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
          <span className={saveError ? "dialog-save-status error" : "dialog-save-status"} role="status" aria-live="polite">
            {saving
              ? "正在自动保存并同步运行配置…"
              : saveError
                ? `保存失败 · ${saveError}`
                : `已自动保存 · ${footerNote}`}
          </span>
          <button className="console-button primary" disabled={saving} onClick={onClose} type="button">关闭</button>
        </footer>
      </section>
    </div>
  );
}
