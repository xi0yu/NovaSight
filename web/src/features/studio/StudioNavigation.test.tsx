import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CONSOLE_PAGES, StudioNavigation } from "./StudioNavigation";

describe("StudioNavigation", () => {
  it("shows six task-first entries without a hidden second navigation level", async () => {
    const onNavigate = vi.fn();
    render(<StudioNavigation activePage="capture" onNavigate={onNavigate} />);

    expect(screen.getAllByRole("button")).toHaveLength(6);
    expect(screen.getByRole("button", { name: "设备管理" })).toHaveAttribute("aria-current", "page");
    expect(screen.queryByRole("region", { name: "当前设备的深入功能" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "算法参数" }));
    expect(onNavigate).toHaveBeenCalledWith("params");
  });

  it("keeps deep model routes valid while grouping them under device management", () => {
    render(<StudioNavigation activePage="models" onNavigate={vi.fn()} />);

    expect(CONSOLE_PAGES.has("models")).toBe(true);
    expect(screen.getByRole("button", { name: "设备管理" })).toHaveAttribute("aria-current", "page");
  });

  it("uses the verified operations layout instead of hard-coded nav labels", () => {
    render(
      <StudioNavigation
        activePage="overview"
        onNavigate={vi.fn()}
        pages={[
          { id: "overview", label: "运行首页", modules: [] },
          { id: "activity", label: "链路记录", modules: [] },
        ]}
      />
    );

    expect(screen.getByRole("button", { name: "运行首页" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "链路记录" })).toBeInTheDocument();
  });
});
