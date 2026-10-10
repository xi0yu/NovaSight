import { useEffect, useRef, type ReactNode } from "react";
import { NovaIcon } from "../../components/visual";
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "./dialogFocus";
import type { ConsoleUtility } from "./StudioNavigation";
import "./studio-utility-dialog.css";

export function StudioUtilityDialog({ kind, children, onClose }: {
  kind: ConsoleUtility; children: ReactNode; onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef(onClose); closeRef.current = onClose;
  const title = kind === "settings" ? "设置" : "关于 NovaSight";
  useEffect(() => {
    acquireBodyScrollLock();
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const background = Array.from(dialogRef.current?.closest(".console-app")?.querySelectorAll<HTMLElement>(":scope > .console-main, :scope > .console-sidebar") ?? []);
    const inertStates = background.map((element) => element.inert);
    background.forEach((element) => { element.inert = true; });
    dialogRef.current?.focus();
    const keydown = (event: KeyboardEvent) => {
      const focusedDialog = document.activeElement?.closest('[role="dialog"], [role="alertdialog"]');
      if (focusedDialog && focusedDialog !== dialogRef.current) return;
      if (event.key === "Escape") { event.preventDefault(); event.stopImmediatePropagation(); closeRef.current(); }
      else if (event.key === "Tab") { event.stopImmediatePropagation(); trapDialogTabKey(event, dialogRef.current); }
    };
    document.addEventListener("keydown", keydown, { capture: true });
    return () => {
      releaseBodyScrollLock();
      document.removeEventListener("keydown", keydown, { capture: true });
      background.forEach((element, index) => { element.inert = inertStates[index]; });
      previous?.focus();
    };
  }, []);
  return <div className="studio-utility-dialog-layer" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="studio-utility-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-utility-title" ref={dialogRef} tabIndex={-1}>
      <header><div><NovaIcon name={kind === "settings" ? "settings" : "help"} size={22} /><h2 id="studio-utility-title">{title}</h2></div>
        <button className="console-button" type="button" aria-label={`关闭${kind === "settings" ? "设置" : "关于"}`} onClick={onClose}><NovaIcon name="x-circle" size={18} /></button>
      </header>
      <div className="studio-utility-dialog-content">{children}</div>
    </section>
  </div>;
}
