import { useEffect, useRef } from "react";

import { NovaIcon } from "../../components/visual";
import { trapDialogTabKey } from "../studio/dialogFocus";
import { ModelSelectionPanel, type ModelSelectionPanelProps } from "./ModelSelectionPanel";

export function ModelManagerDialog({
  activeModelName,
  onClose,
  open,
  panelProps
}: {
  activeModelName: string;
  onClose: () => void;
  open: boolean;
  panelProps: ModelSelectionPanelProps;
}) {
  const dialogRef = useRef<HTMLElement | null>(null);
  const busyRef = useRef(panelProps.busy);
  busyRef.current = panelProps.busy;

  useEffect(() => {
    if (!open) return undefined;
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    const frame = window.requestAnimationFrame(() => {
      dialogRef.current?.focus();
      dialogRef.current
        ?.querySelector<HTMLElement>('.model-catalog-row[data-active="true"]')
        ?.scrollIntoView({ block: "center" });
    });
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && busyRef.current === null) {
        onClose();
      } else {
        trapDialogTabKey(event, dialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [onClose, open, panelProps.activeArtifactId, panelProps.root]);

  if (!open) return null;

  return (
    <div
      className="model-manager-dialog-layer"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && busyRef.current === null) onClose();
      }}
    >
      <section
        aria-labelledby="model-manager-dialog-title"
        aria-modal="true"
        className="model-manager-dialog"
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <header className="model-manager-dialog-header">
          <div className="model-manager-dialog-title">
            <span className="model-manager-dialog-icon" aria-hidden="true">
              <NovaIcon name="models" size={22} strokeWidth={1.7} />
            </span>
            <div>
              <span className="class-config-eyebrow">MODEL VAULT</span>
              <h2 id="model-manager-dialog-title">模型管理与切换</h2>
              <p>按推荐状态与标签整理本机 Engine；确认切换后才读取 TensorRT 契约。</p>
            </div>
          </div>
          <div className="model-manager-active-pill">
            <i aria-hidden="true" />
            <span>当前</span>
            <b>{activeModelName}</b>
          </div>
          <button aria-label="关闭模型管理" className="launch-dialog-close" disabled={panelProps.busy !== null} onClick={onClose} type="button">
            <NovaIcon name="x-circle" size={18} />
          </button>
        </header>
        <div className="model-manager-dialog-body">
          <ModelSelectionPanel {...panelProps} />
        </div>
      </section>
    </div>
  );
}
