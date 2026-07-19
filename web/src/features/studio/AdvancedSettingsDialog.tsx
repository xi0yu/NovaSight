import { useEffect, useId, useRef, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";
import { trapDialogTabKey } from "./dialogFocus";

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
  children
}: {
  open: boolean;
  eyebrow: string;
  title: string;
  description: string;
  footerNote: string;
  dirty?: boolean;
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
            <p>{description}</p>
          </div>
          <button
            aria-label={dirty ? `保存并关闭${title}` : `关闭${title}`}
            className="launch-dialog-close"
            disabled={saving}
            onClick={onClose}
            title={dirty ? "关闭并保存本次修改" : "关闭"}
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
              ? "正在保存本次修改…"
              : saveError
                ? `保存失败 · ${saveError}`
                : dirty
                  ? "有未保存修改 · 关闭时将一次同步到运行配置。"
                  : `未修改 · ${footerNote}`}
          </span>
          <button
            className={`console-button ${dirty ? "primary dialog-save-button" : "dialog-close-button"}`}
            disabled={saving}
            onClick={onClose}
            type="button"
          >
            {dirty ? "关闭并保存" : "关闭"}
          </button>
        </footer>
      </section>
    </div>
  );
}
