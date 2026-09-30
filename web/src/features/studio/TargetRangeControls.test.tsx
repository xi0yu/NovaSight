import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TargetRangeControls } from "./TargetRangeControls";

describe("search circle tuning", () => {
  it("previews radius and boundary changes locally, then saves only the radius", async () => {
    const onRadiusCommit = vi.fn();
    const { rerender } = render(<TargetRangeControls radius={180} onRadiusCommit={onRadiusCommit} />);
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.getByRole("status").textContent).toContain("范围外");
    const slider = screen.getByRole("slider", { name: "搜索半径 滑块" });
    expect(slider.getAttribute("max")).toBe("500");
    expect(slider.getAttribute("step")).toBe("10");
    fireEvent.change(slider, { target: { value: "220" } });
    expect(screen.getByRole("status").textContent).toContain("范围条件满足");
    expect(onRadiusCommit).not.toHaveBeenCalled();
    fireEvent.blur(slider);
    await waitFor(() => expect(onRadiusCommit).toHaveBeenCalledWith(220));
    rerender(<TargetRangeControls radius={220} onRadiusCommit={onRadiusCommit} />);
    fireEvent.click(screen.getByRole("button", { name: "远离准星" }));
    expect(screen.getByRole("status").textContent).toContain("范围外");
    fireEvent.click(screen.getByRole("button", { name: "靠近准星" }));
    expect(screen.getByRole("status").textContent).toContain("范围条件满足");
    expect(onRadiusCommit).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "320 px" }));
    expect(onRadiusCommit).toHaveBeenLastCalledWith(320);
  });
});
