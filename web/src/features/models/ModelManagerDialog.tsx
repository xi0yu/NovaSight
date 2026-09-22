import { useEffect, useRef } from "react";

import { NovaIcon } from "../../components/visual";
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "../studio/dialogFocus";
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
  const scrolledToActiveRef = useRef(false);
  busyRef.current = panelProps.busy;
  const activeFile = panelProps.activeArtifactPath.split(/[\\/]/).pop();

  useEffect(() => {
    if (!open) return undefined;
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    scrolledToActiveRef.current = false;
    const frame = window.requestAnimationFrame(() => dialogRef.current?.focus());
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
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [onClose, open]);

  useEffect(() => {
    if (!open || !panelProps.root || scrolledToActiveRef.current) return;
    const activeRow = dialogRef.current?.querySelector<HTMLElement>('.model-catalog-row[data-active="true"]');
    if (!activeRow) return;
    activeRow.scrollIntoView({ block: "center" });
    scrolledToActiveRef.current = true;
  }, [open, panelProps.activeArtifactId, panelProps.root]);

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
              <span className="class-config-eyebrow">模型管理</span>
              <h2 id="model-manager-dialog-title">模型库</h2>
              <p>查找并整理设备上的模型文件；只有确认切换才会影响推理。</p>
            </div>
          </div>
          <div className={`model-manager-active-pill ${activeFile ? "" : "inactive"}`}>
            <i aria-hidden="true" />
            <span>{activeFile ? panelProps.activeLoaded ? "运行已装载" : "已部署 · 未装载" : "未部署"}</span>
            <b title={panelProps.activeArtifactPath}>{activeFile || "尚无模型"}</b>
            {activeFile ? <small title={activeModelName}>项目：{activeModelName}</small> : null}
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
