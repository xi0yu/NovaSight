import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { TargetClassEditor, parseClassValues, setClassValue } from "./TargetClassEditor";

describe("class point workbench", () => {
  it("edits class points independently and keeps selection separate from weight", () => {
    const onWeight = vi.fn();
    const onToggle = vi.fn();
    function Harness() {
      const [xs, setXs] = useState<Record<string, number>>({});
      const [ys, setYs] = useState<Record<string, number>>({});
      return <TargetClassEditor ids={Array.from({ length: 16 }, (_, id) => id)} names={[]} selected={new Set([0, 1])} weights={{ 0: 0.8, 1: 0.4 }} xs={xs} ys={ys} defaultY={0.22}
        onPoint={(id, x, y) => { setXs((old) => ({ ...old, [id]: x })); setYs((old) => ({ ...old, [id]: y })); }} onWeight={onWeight} onToggle={onToggle} onName={() => {}} />;
    }
    render(<Harness />);
    expect(screen.queryByRole("img")).toBeNull();
    fireEvent.keyDown(screen.getByRole("button", { name: /cls 0 瞄点平面/ }), { key: "ArrowRight" });
    expect(screen.getByRole("textbox", { name: "cls 0 水平位置 数值" })).toHaveValue("51");
    fireEvent.click(screen.getByRole("button", { name: "cls 1" }));
    expect(screen.getByRole("textbox", { name: "cls 1 水平位置 数值" })).toHaveValue("50");
    fireEvent.click(screen.getByRole("button", { name: "中心" }));
    expect(screen.getByRole("textbox", { name: "cls 1 垂直位置 数值" })).toHaveValue("50");
    const weight = screen.getByRole("slider", { name: "cls 1 优先权重 滑块" });
    fireEvent.change(weight, { target: { value: "0.8" } }); fireEvent.blur(weight);
    expect(onWeight).toHaveBeenCalledWith(1, 0.8);
    expect(onToggle).not.toHaveBeenCalled();
    expect(screen.getAllByRole("button", { name: /^cls \d+$/ })).toHaveLength(16);
    fireEvent.click(screen.getByRole("button", { name: "cls 15" }));
    fireEvent.keyDown(screen.getByRole("button", { name: /cls 15 瞄点平面/ }), { key: "Home" });
    expect(screen.getByRole("textbox", { name: "cls 15 垂直位置 数值" })).toHaveValue("50");
    fireEvent.click(screen.getByRole("checkbox", { name: "cls 15 参与目标选择" }));
    expect(onToggle).toHaveBeenCalledWith(15);
    expect(screen.queryByRole("slider", { name: "cls 1 优先权重 滑块" })).toBeNull();
    expect(parseClassValues(setClassValue("0:0.8,1:0.4", 1, 0.7))).toEqual({ 0: 0.8, 1: 0.7 });
  });
});
