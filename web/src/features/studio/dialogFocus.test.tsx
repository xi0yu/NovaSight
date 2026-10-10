import { act, fireEvent, render, screen } from "@testing-library/react";
import { useRef, useState } from "react";
import { expect, it, vi } from "vitest";
import { ActionConfirmationDialog, type ActionConfirmationRequest } from "./ActionConfirmationDialog";
import { StudioUtilityDialog } from "./StudioUtilityDialog";
import { useDialogInteraction } from "./dialogFocus";

const request: ActionConfirmationRequest = { eyebrow: "设置", title: "确认操作", description: "核对后继续", confirmLabel: "继续", onConfirm: () => {}, content: <input aria-label="编辑内容" defaultValue="draft" /> };

it("blocks background controls for a dialog inside the page while retaining emergency stop", () => {
  function PageDialog() {
    const ref = useRef<HTMLElement>(null);
    useDialogInteraction(ref, true, () => {});
    return <section ref={ref} role="dialog" tabIndex={-1}><button>弹窗操作</button></section>;
  }
  const view = render(<div className="console-app">
    <header><button className="console-emergency-stop">紧急停止</button></header>
    <aside><button>导航</button></aside>
    <main><button>页面操作</button><div><PageDialog /></div></main>
  </div>);
  expect(screen.getByRole("button", { name: "页面操作" }).inert).toBe(true);
  expect(screen.getByRole("button", { name: "导航" }).closest("aside")?.inert).toBe(true);
  expect(screen.getByRole("button", { name: "弹窗操作" }).closest("section")?.inert).not.toBe(true);
  expect(screen.getByRole("button", { name: "紧急停止" }).closest("header")?.inert).not.toBe(true);
  view.unmount();
});

it("keeps the edited input focused across updates and blocks dismissal while processing", () => {
  vi.useFakeTimers();
  const close = vi.fn();
  const props = { error: null, onConfirm: vi.fn(), onCancel: close };
  const view = render(<ActionConfirmationDialog {...props} busy request={request} />);
  act(() => vi.advanceTimersByTime(20));
  const input = screen.getByRole("textbox", { name: "编辑内容" });
  act(() => input.focus());
  view.rerender(<ActionConfirmationDialog {...props} busy request={{ ...request }} />);
  act(() => vi.advanceTimersByTime(20));
  expect(input).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(close).not.toHaveBeenCalled();
  view.rerender(<ActionConfirmationDialog {...props} busy={false} request={{ ...request }} />);
  fireEvent.keyDown(document, { key: "Escape" });
  expect(close).toHaveBeenCalledOnce();
  view.unmount();
  vi.useRealTimers();
});

it("closes only the top dialog and returns focus to the underlying operation", () => {
  vi.useFakeTimers();
  function Workspace() {
    const [outer, setOuter] = useState(true);
    const [inner, setInner] = useState(false);
    return <div className="console-app">
      <main className="console-main"><button>工作区操作</button></main>
      {outer ? <StudioUtilityDialog kind="settings" onClose={() => setOuter(false)}>
        <button onClick={() => setInner(true)}>打开确认</button>
      </StudioUtilityDialog> : null}
      {inner ? <ActionConfirmationDialog request={request} busy={false} onCancel={() => setInner(false)} onConfirm={() => {}} /> : null}
    </div>;
  }
  const view = render(<Workspace />);
  act(() => vi.advanceTimersByTime(20));
  const opener = screen.getByRole("button", { name: "打开确认" });
  act(() => opener.focus());
  fireEvent.click(opener);
  act(() => vi.advanceTimersByTime(20));
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  expect(screen.getByRole("dialog", { name: "设置" })).toBeVisible();
  expect(opener).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  view.unmount();
  vi.useRealTimers();
});
