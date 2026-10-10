import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { TargetRangeControls } from "./TargetRangeControls";

it("edits the full search-radius range and quick presets without the removed explanations", async () => {
  const saved = vi.fn();
  function Workspace() {
    const [radius, setRadius] = useState(180);
    return <TargetRangeControls radius={radius} onRadiusCommit={(value) => { saved(value); setRadius(value); }} />;
  }
  render(<Workspace />);
  expect(screen.queryByText(/自动保存并应用|了解范围判定|预览搜索范围|范围只负责筛选|力度独立调整/)).not.toBeInTheDocument();
  const slider = screen.getByRole("slider", { name: "搜索半径 滑块" });
  expect(slider).toHaveAttribute("min", "0");
  expect(slider).toHaveAttribute("max", "1000");
  for (const radius of [0, 1000]) {
    fireEvent.change(slider, { target: { value: String(radius) } });
    fireEvent.blur(slider);
    await waitFor(() => expect(screen.getByRole("textbox", { name: "搜索半径 数值" })).toHaveValue(String(radius)));
  }
  await userEvent.click(screen.getByRole("button", { name: "大范围，320 px" }));
  expect(screen.getByRole("textbox", { name: "搜索半径 数值" })).toHaveValue("320");
  expect(saved).toHaveBeenLastCalledWith(320);
});
