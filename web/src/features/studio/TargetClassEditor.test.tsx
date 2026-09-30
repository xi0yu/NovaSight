import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { TargetClassEditor, parseClassValues, setClassValue } from "./TargetClassEditor";

describe("class point workbench", () => {
  it("edits eight classes side by side without dropping legacy configuration", () => {
    const onWeight = vi.fn();
    const onToggle = vi.fn();
    const onName = vi.fn();
    function Harness() {
      const [xs, setXs] = useState<Record<string, number>>({ 15: 0.7 });
      const [ys, setYs] = useState<Record<string, number>>({});
      return <TargetClassEditor ids={Array.from({ length: 16 }, (_, id) => id)} names={[]} selected={new Set([0, 1, 15])} weights={{ 0: 0.8, 1: 0.4, 15: 0.7 }} xs={xs} ys={ys} defaultY={0.22}
        saved={{ names: [], selected: new Set([0, 1, 15]), weights: { 0: 0.8, 1: 0.4, 15: 0.7 }, xs: { 15: 0.7 }, ys: {}, defaultY: 0.22 }}
        onReset={(id) => { setXs((old) => { const next = { ...old }; delete next[id]; return next; }); setYs((old) => { const next = { ...old }; delete next[id]; return next; }); }}
        onPoint={(id, x, y) => { setXs((old) => ({ ...old, [id]: x })); setYs((old) => ({ ...old, [id]: y })); }} onWeight={onWeight} onToggle={onToggle} onName={onName} />;
    }
    render(<Harness />);
    expect(screen.getAllByRole("checkbox")).toHaveLength(8);
    expect(screen.getAllByRole("button", { name: /瞄点平面/ })).toHaveLength(8);
    expect(screen.getByRole("textbox", { name: "cls 7 名称" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "cls 8 名称" })).toBeNull();
    expect(screen.getByRole("note")).toHaveTextContent("cls15");
    expect(screen.getByRole("checkbox", { name: "cls 0 参与目标选择" })).toBeChecked();
    expect(screen.getByRole("button", { name: "cls 0 上部" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.keyDown(screen.getByRole("button", { name: /cls 0 瞄点平面/ }), { key: "ArrowRight" });
    expect(screen.getByRole("textbox", { name: "cls 0 水平位置 数值" })).toHaveValue("51");
    expect(screen.getByRole("status", { name: "cls 0 已修改" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "cls 0 上部" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("textbox", { name: "cls 1 水平位置 数值" })).toHaveValue("50");
    fireEvent.click(screen.getByRole("button", { name: "cls 1 中心" }));
    expect(screen.getByRole("textbox", { name: "cls 1 垂直位置 数值" })).toHaveValue("50");
    expect(screen.getByRole("button", { name: "cls 1 中心" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "撤销 cls 0 修改" }));
    expect(screen.queryByRole("status", { name: "cls 0 已修改" })).toBeNull();
    expect(screen.getByRole("textbox", { name: "cls 1 垂直位置 数值" })).toHaveValue("50");
    const weight = screen.getByRole("slider", { name: "cls 1 类别偏好 滑块" });
    fireEvent.change(weight, { target: { value: "0.8" } }); fireEvent.blur(weight);
    expect(onWeight).toHaveBeenCalledWith(1, 0.8);
    expect(onToggle).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("checkbox", { name: "cls 7 参与目标选择" }));
    expect(onToggle).toHaveBeenCalledWith(7);
    const name = screen.getByRole("textbox", { name: "cls 1 名称" });
    fireEvent.change(name, { target: { value: "身体" } }); fireEvent.blur(name);
    expect(onName).toHaveBeenCalledWith(1, "身体");
    expect(parseClassValues(setClassValue("0:0.8,1:0.4,15:0.7", 1, 0.7))).toEqual({ 0: 0.8, 1: 0.7, 15: 0.7 });
  });
});
