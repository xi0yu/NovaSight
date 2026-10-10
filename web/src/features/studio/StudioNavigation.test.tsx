import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DEFAULT_STUDIO_LAYOUT } from "../../contracts/studioLayout";
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

it("keeps output last and separates application dialogs from workspace pages even with an old layout", async () => {
  const onNavigate = vi.fn();
  const legacyPages = [...DEFAULT_STUDIO_LAYOUT.pages].sort((a, b) => a.id === "control-test" ? -1 : b.id === "control-test" ? 1 : 0);
  render(<StudioNavigation activePage="params" activeUtility="settings" onNavigate={onNavigate} pages={legacyPages} />);
  const primary = document.querySelector<HTMLElement>(".console-primary-navigation")!;
  expect(within(primary).getAllByRole("button").map((button) => button.dataset.page)).toEqual(["overview", "capture", "params", "fire", "activity", "control-test"]);
  const settings = screen.getByRole("button", { name: "设置" });
  expect(settings).toHaveAttribute("aria-haspopup", "dialog");
  expect(settings).toHaveAttribute("aria-expanded", "true");
  expect(settings).not.toHaveAttribute("aria-current");
  expect(screen.getByRole("button", { name: "算法参数" })).toHaveAttribute("aria-current", "page");
  await userEvent.click(settings);
  expect(onNavigate).toHaveBeenCalledWith("settings");
});

it("keeps advanced control tuning under algorithm parameters", () => {
  render(<StudioNavigation activePage="control" onNavigate={vi.fn()} />);
  expect(screen.getByRole("button", { name: "算法参数" })).toHaveAttribute("aria-current", "page");
  expect(screen.getByRole("button", { name: "设备管理" })).not.toHaveAttribute("aria-current");
});
