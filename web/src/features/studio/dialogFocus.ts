import { useEffect, useRef, type RefObject } from "react";

const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "summary",
  '[tabindex]:not([tabindex="-1"])'
].join(",");

export function trapDialogTabKey(event: KeyboardEvent, dialog: HTMLElement | null): void {
  if (event.key !== "Tab" || !dialog) {
    return;
  }
  const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter((element) => {
      if (element.tabIndex < 0 || element.closest('[hidden], [inert], [aria-hidden="true"]')) return false;
      const closedDetails = element.closest("details:not([open])");
      if (closedDetails && element !== closedDetails.querySelector("summary")) return false;
      if (typeof element.checkVisibility === "function" && !element.checkVisibility({ checkVisibilityCSS: true })) return false;
      const style = window.getComputedStyle(element);
      return style.display !== "none" && style.visibility !== "hidden";
    });
  if (focusable.length === 0) {
    event.preventDefault();
    dialog.focus();
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (document.activeElement === dialog || !dialog.contains(document.activeElement)) {
    event.preventDefault();
    (event.shiftKey ? last : first).focus();
  } else if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

// Module-level stack: each dialog that wants to lock body scroll calls
// `acquireBodyScrollLock()` in its open path and `releaseBodyScrollLock()`
// in cleanup. The first caller captures the original `overflow` value;
// subsequent ones just increment the counter. When the counter drops back
// to 0 the original value is restored. This avoids the save/restore race
// when 2+ dialogs are open at the same time (e.g. error center over a
// config dialog) and the cleanup order would otherwise restore a
// "hidden" value that another dialog is still relying on.
let lockCount = 0;
let savedOverflow = "";

export function acquireBodyScrollLock(): void {
  if (lockCount === 0) {
    savedOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  lockCount += 1;
}

export function releaseBodyScrollLock(): void {
  if (lockCount === 0) {
    return;
  }
  lockCount -= 1;
  if (lockCount === 0) {
    document.body.style.overflow = savedOverflow;
  }
}
const dialogStack: HTMLElement[] = [];
const inertLocks = new Map<HTMLElement, { count: number; previous: boolean }>();

export function useDialogInteraction(
  dialogRef: RefObject<HTMLElement>,
  open: boolean,
  onDismiss: () => void,
  options: { blocked?: boolean; initialFocus?: string } = {},
): void {
  const latest = useRef({ onDismiss, ...options });
  latest.current = { onDismiss, ...options };
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!open || !dialog) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const workspace = dialog.closest(".console-app");
    // Keep the emergency-stop toolbar reachable while ordinary workspace controls are inert.
    const background: HTMLElement[] = [];
    if (workspace) {
      for (let layer: HTMLElement = dialog; layer !== workspace && layer.parentElement; layer = layer.parentElement) {
        for (const sibling of Array.from(layer.parentElement.children)) {
          if (sibling instanceof HTMLElement && sibling !== layer && !sibling.querySelector(".console-emergency-stop")) background.push(sibling);
        }
      }
    }
    for (const element of background) {
      const lock = inertLocks.get(element) ?? { count: 0, previous: element.inert };
      lock.count += 1;
      inertLocks.set(element, lock);
      element.inert = true;
    }
    acquireBodyScrollLock();
    dialogStack.push(dialog);
    const target = latest.current.initialFocus ? dialog.querySelector<HTMLElement>(latest.current.initialFocus) : null;
    (target ?? dialog).focus();
    const keydown = (event: KeyboardEvent) => {
      if (dialogStack[dialogStack.length - 1] !== dialog) return;
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (!latest.current.blocked) latest.current.onDismiss();
      } else if (event.key === "Tab") {
        event.stopImmediatePropagation();
        trapDialogTabKey(event, dialog);
      }
    };
    document.addEventListener("keydown", keydown, { capture: true });
    return () => {
      document.removeEventListener("keydown", keydown, { capture: true });
      const wasTop = dialogStack[dialogStack.length - 1] === dialog;
      dialogStack.splice(dialogStack.indexOf(dialog), 1);
      releaseBodyScrollLock();
      for (const element of background) {
        const lock = inertLocks.get(element)!;
        lock.count -= 1;
        if (lock.count === 0) { element.inert = lock.previous; inertLocks.delete(element); }
      }
      if (wasTop) {
        if (previousFocus?.isConnected && !previousFocus.closest("[inert]")) previousFocus.focus();
        else dialogStack[dialogStack.length - 1]?.focus();
      }
    };
  }, [dialogRef, open]);
}
