import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { InlineNumberControl, ParameterNumberControl } from "./StudioControls";
import { AlgorithmTabs } from "./AlgorithmTabs";
import { trapDialogTabKey } from "./dialogFocus";

it("does not turn an empty diagnostic number into a physical zero command", async () => {
  const onCommit = vi.fn();
  render(<InlineNumberControl ariaLabel="kmNet 测试 dx" value={10} onCommit={onCommit} />);
  const input = screen.getByRole("textbox", { name: "kmNet 测试 dx" });
  await userEvent.clear(input);
  await userEvent.tab();
  expect(onCommit).not.toHaveBeenCalled();
  expect(input).toHaveValue("10");
});

it("names each stepper action after its own parameter", () => {
  render(<ParameterNumberControl label="kmNet 控制端口" kind="stepper" value={8888} min={1} max={65535} step={1} onCommit={vi.fn()} />);
  expect(screen.getByRole("button", { name: "减少kmNet 控制端口" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "增加kmNet 控制端口" })).toBeInTheDocument();
});

it("traps focus without entering closed details or hidden controls", () => {
  const { container } = render(<section tabIndex={-1}><button>第一项</button><details><summary>技术详情</summary><button>隐藏操作</button></details><button hidden>隐藏按钮</button></section>);
  const dialog = container.querySelector("section")!;
  dialog.focus();
  trapDialogTabKey(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, cancelable: true }), dialog);
  expect(screen.getByText("技术详情")).toHaveFocus();
});

it("moves tab selection and keyboard focus together, including wrap and end keys", async () => {
  function Workspace() {
    const [section, setSection] = useState<"response" | "targeting" | "motion" | "advanced">("response");
    return <><AlgorithmTabs value={section} onChange={setSection} /><div role="tabpanel" id="algorithm-parameter-panel" aria-labelledby={`algorithm-tab-${section}`}>{section}</div></>;
  }
  render(<Workspace />);
  const user = userEvent.setup();
  await user.tab();
  expect(screen.getByRole("tab", { name: "范围与触发" })).toHaveFocus();
  await user.keyboard("{ArrowRight}");
  expect(screen.getByRole("tab", { name: "目标与瞄点" })).toHaveFocus();
  expect(screen.getByRole("tabpanel", { name: "目标与瞄点" })).toHaveTextContent("targeting");
  await user.keyboard("{End}{ArrowRight}");
  expect(screen.getByRole("tab", { name: "范围与触发" })).toHaveAttribute("aria-selected", "true");
  await user.keyboard("{ArrowLeft}");
  expect(screen.getByRole("tab", { name: "进阶调校" })).toHaveFocus();
  await user.keyboard("{Home}");
  expect(screen.getByRole("tab", { name: "范围与触发" })).toHaveFocus();
});
