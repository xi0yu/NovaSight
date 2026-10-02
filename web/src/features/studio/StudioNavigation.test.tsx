import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CONSOLE_PAGES, StudioNavigation } from "./StudioNavigation";

describe("StudioNavigation", () => {
  it("keeps deep model routes valid while grouping them under device management", () => {
    render(<StudioNavigation activePage="models" onNavigate={vi.fn()} />);

    expect(CONSOLE_PAGES.has("models")).toBe(true);
    expect(screen.getByRole("button", { name: "设备管理" })).toHaveAttribute("aria-current", "page");
  });

  it("maps configured labels to page IDs when navigating", async () => {
    const onNavigate = vi.fn();
    render(
      <StudioNavigation
        activePage="overview"
        onNavigate={onNavigate}
        pages={[
          { id: "overview", label: "运行首页", modules: [] },
          { id: "activity", label: "链路记录", modules: [] },
        ]}
      />
    );

    expect(screen.getByRole("button", { name: "运行首页" })).toHaveAttribute("aria-current", "page");
    await userEvent.click(screen.getByRole("button", { name: "链路记录" }));
    expect(onNavigate).toHaveBeenCalledWith("activity");
  });
});
