import { describe, expect, it } from "vitest";

import { DEFAULT_THEME, isThemeMode, resolveStoredTheme } from "./themeRegistry";

describe("theme registry", () => {
  it("uses one product appearance instead of exposing a theme picker", () => {
    expect(DEFAULT_THEME).toBe("arena-signal");
    expect(isThemeMode("arena-signal")).toBe(true);
  });

  it("migrates every old local preference to the product theme", () => {
    expect(isThemeMode("studio")).toBe(false);
    expect(resolveStoredTheme("light")).toBe(DEFAULT_THEME);
    expect(resolveStoredTheme("graphite-red")).toBe(DEFAULT_THEME);
    expect(resolveStoredTheme("unknown")).toBe(DEFAULT_THEME);
  });
});
