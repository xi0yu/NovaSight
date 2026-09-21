import { describe, expect, it } from "vitest";

import { DEFAULT_THEME, THEME_OPTIONS, isThemeMode, resolveStoredTheme } from "./themeRegistry";

describe("theme registry", () => {
  it("offers exactly three supported appearances", () => {
    expect(THEME_OPTIONS.map((option) => option.id)).toEqual(["studio", "rose-white", "graphite-red"]);
  });

  it("accepts product themes from local storage", () => {
    expect(isThemeMode("studio")).toBe(true);
    expect(resolveStoredTheme("studio")).toBe("studio");
    expect(isThemeMode("rose-white")).toBe(true);
    expect(isThemeMode("graphite-red")).toBe(true);
    expect(isThemeMode("frontier-industrial")).toBe(false);
    expect(resolveStoredTheme("rose-white")).toBe("rose-white");
    expect(resolveStoredTheme("graphite-red")).toBe("graphite-red");
    expect(resolveStoredTheme("frontier-industrial")).toBe("graphite-red");
  });

  it("keeps legacy light and dark preferences compatible", () => {
    expect(DEFAULT_THEME).toBe("studio");
    expect(resolveStoredTheme("light")).toBe("rose-white");
    expect(resolveStoredTheme("dark")).toBe("graphite-red");
    expect(resolveStoredTheme("momo")).toBe("rose-white");
    expect(resolveStoredTheme("tayama")).toBe("graphite-red");
    expect(resolveStoredTheme("unknown")).toBe(DEFAULT_THEME);
  });
});
