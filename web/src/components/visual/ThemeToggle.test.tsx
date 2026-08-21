import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { ThemeToggle } from "./ThemeToggle";
import { THEME_STORAGE_KEY } from "./themeRegistry";

describe("ThemeToggle", () => {
  beforeEach(() => {
    window.localStorage.removeItem(THEME_STORAGE_KEY);
  });

  it("closes the open theme menu with Escape and restores toggle focus", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);
    const toggle = screen.getByLabelText(/当前主题/);

    await user.click(toggle);
    expect(document.querySelector("details.theme-picker")).toHaveAttribute("open");

    await user.keyboard("{Escape}");
    expect(document.querySelector("details.theme-picker")).not.toHaveAttribute("open");
    expect(toggle).toHaveFocus();
  });
});
