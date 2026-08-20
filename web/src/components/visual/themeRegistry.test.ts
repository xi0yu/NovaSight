import { describe, expect, it } from "vitest";

import { DEFAULT_THEME, THEME_OPTIONS, isThemeMode, resolveStoredTheme } from "./themeRegistry";

describe("theme registry", () => {
  it("includes the approved pink-white and black-gray-red palettes", () => {
    expect(THEME_OPTIONS.map((option) => option.id)).toEqual(expect.arrayContaining([
      "rose-white",
      "graphite-red",
    ]));
  });

  it("accepts both new themes from local storage", () => {
    expect(isThemeMode("rose-white")).toBe(true);
    expect(isThemeMode("graphite-red")).toBe(true);
    expect(resolveStoredTheme("rose-white")).toBe("rose-white");
    expect(resolveStoredTheme("graphite-red")).toBe("graphite-red");
  });

  it("keeps legacy light and dark preferences compatible", () => {
    expect(resolveStoredTheme("light")).toBe("elysia");
    expect(resolveStoredTheme("dark")).toBe("tayama");
    expect(resolveStoredTheme("unknown")).toBe(DEFAULT_THEME);
  });
});
