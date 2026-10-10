import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { InlineNumberControl, ParameterNumberControl } from "./StudioControls";
import { AlgorithmTabs } from "./AlgorithmTabs";

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

it("keeps invalid numeric drafts local and clamps valid edits to the accepted output range", async () => {
  const saved: number[] = [];
  function Workspace() {
    const [value, setValue] = useState(127);
    return <ParameterNumberControl label="输出上限" value={value} min={0} max={1000} step={1}
      onCommit={(next) => { saved.push(next); setValue(next); }} />;
  }
  render(<Workspace />);
  const input = screen.getByRole("textbox", { name: "输出上限 数值" });
  for (const invalid of ["", " ", "abc", "NaN", "Infinity"]) {
    fireEvent.change(input, { target: { value: invalid } });
    fireEvent.blur(input);
    expect(input).toHaveValue("127");
    expect(saved).toEqual([]);
  }
  for (const [typed, expected] of [["-1", 0], ["1001", 1000], ["0", 0], ["1000", 1000]] as const) {
    fireEvent.change(input, { target: { value: typed } });
    fireEvent.blur(input);
    await waitFor(() => expect(input).toHaveValue(String(expected)));
    expect(saved[saved.length - 1]).toBe(expected);
  }
});
