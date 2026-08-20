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
});
