import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CONSOLE_PAGES, StudioNavigation } from "./StudioNavigation";

describe("StudioNavigation", () => {
  it("exposes model management as a deep-link workspace", async () => {
    const onNavigate = vi.fn();
    render(<StudioNavigation activePage="capture" onNavigate={onNavigate} />);

    expect(CONSOLE_PAGES.has("models")).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "模型管理" }));
    expect(onNavigate).toHaveBeenCalledWith("models");
  });

  it("lets an authenticated operator open license management from Studio", async () => {
    const onNavigate = vi.fn();
    render(<StudioNavigation activePage="capture" onNavigate={onNavigate} />);

    await userEvent.click(screen.getByRole("button", { name: "授权管理" }));
    expect(onNavigate).toHaveBeenCalledWith("license");
  });

  it("keeps navigation groups out of the page heading hierarchy", () => {
    render(<StudioNavigation activePage="capture" onNavigate={vi.fn()} />);

    expect(screen.queryByRole("heading", { name: "运行工作台" })).not.toBeInTheDocument();
    expect(screen.getByText("运行工作台")).toBeInTheDocument();
  });
});
