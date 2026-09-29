import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TargetRangeControls } from "./TargetRangeControls";

describe("capsule range tuning", () => {
  it("always uses a capsule and previews a draft without sending a control command", async () => {
    const onScaleCommit = vi.fn();
    render(<TargetRangeControls scale={1} onScaleCommit={onScaleCommit} />);
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.getByRole("status").textContent).toContain("范围外");
    const slider = screen.getByRole("slider", { name: "范围比例 滑块" });
    expect(slider.getAttribute("max")).toBe("500");
    expect(slider.getAttribute("step")).toBe("10");
    fireEvent.change(slider, { target: { value: "200" } });
    expect(screen.getByRole("status").textContent).toContain("范围条件满足");
    expect(onScaleCommit).not.toHaveBeenCalled();
    fireEvent.blur(slider);
    await waitFor(() => expect(onScaleCommit).toHaveBeenCalledWith(2));
    fireEvent.click(screen.getByRole("button", { name: "贴近目标" }));
    expect(screen.getByRole("status").textContent).toContain("范围条件满足");
  });
  it("offers the maximum 500% preset", () => {
    const onScaleCommit = vi.fn();
    render(<TargetRangeControls scale={1} onScaleCommit={onScaleCommit} />);
    fireEvent.click(screen.getByRole("button", { name: "500%" }));
    expect(onScaleCommit).toHaveBeenCalledWith(5);
  });
});
