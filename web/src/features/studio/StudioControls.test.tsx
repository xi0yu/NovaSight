import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { InlineNumberControl, ParameterNumberControl } from "./StudioControls";

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
