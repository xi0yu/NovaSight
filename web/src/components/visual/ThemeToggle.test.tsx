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
    const toggle = screen.getByRole("button", { name: /主题设置/ });

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    await user.keyboard("{Escape}");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveFocus();
  });

  it("applies a visibly selected theme and persists it", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);
    const toggle = screen.getByRole("button", { name: /主题设置/ });

    await user.click(toggle);
    await user.click(screen.getByRole("button", { name: /黑灰红/ }));

    expect(document.documentElement).toHaveAttribute("data-theme", "graphite-red");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("graphite-red");
    expect(toggle).toHaveTextContent("黑灰红");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });
});
