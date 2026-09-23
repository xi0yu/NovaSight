import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CONSOLE_PAGES, StudioNavigation } from "./StudioNavigation";

describe("StudioNavigation", () => {
  it("exposes models as an object-first workspace", async () => {
    const onNavigate = vi.fn();
    render(<StudioNavigation activePage="capture" onNavigate={onNavigate} />);

    expect(CONSOLE_PAGES.has("models")).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "模型" }));
    expect(onNavigate).toHaveBeenCalledWith("models");
  });

  it("lets an authenticated operator open license management from Studio", async () => {
    const onNavigate = vi.fn();
    render(<StudioNavigation activePage="capture" onNavigate={onNavigate} />);

    await userEvent.click(screen.getByRole("button", { name: "授权" }));
    expect(onNavigate).toHaveBeenCalledWith("license");
  });

  it("keeps professional device tools one level deeper", () => {
    render(<StudioNavigation activePage="capture" onNavigate={vi.fn()} />);

    expect(screen.getByRole("button", { name: "设备" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("region", { name: "当前设备的深入功能" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "参数" })).toBeInTheDocument();
  });
});
