const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])'
].join(",");

export function trapDialogTabKey(event: KeyboardEvent, dialog: HTMLElement | null): void {
  if (event.key !== "Tab" || !dialog) {
    return;
  }
  const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter((element) => element.getAttribute("aria-hidden") !== "true");
  if (focusable.length === 0) {
    event.preventDefault();
    dialog.focus();
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (document.activeElement === dialog) {
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
