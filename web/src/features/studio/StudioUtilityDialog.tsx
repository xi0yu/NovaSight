import { useRef, type ReactNode } from "react";
import { NovaIcon } from "../../components/visual";
import { useDialogInteraction } from "./dialogFocus";
import type { ConsoleUtility } from "./StudioNavigation";
import "./studio-utility-dialog.css";

export function StudioUtilityDialog({ kind, children, onClose }: {
  kind: ConsoleUtility; children: ReactNode; onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  const title = kind === "settings" ? "设置" : "关于 NovaSight";
  useDialogInteraction(dialogRef, true, onClose);
  return <div className="studio-utility-dialog-layer" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="studio-utility-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-utility-title" ref={dialogRef} tabIndex={-1}>
      <header><div><NovaIcon name={kind === "settings" ? "settings" : "help"} size={22} /><h2 id="studio-utility-title">{title}</h2></div>
        <button className="console-button" type="button" aria-label={`关闭${kind === "settings" ? "设置" : "关于"}`} onClick={onClose}><NovaIcon name="x-circle" size={18} /></button>
      </header>
      <div className="studio-utility-dialog-content">{children}</div>
    </section>
  </div>;
}
