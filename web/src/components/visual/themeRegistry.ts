export type ThemeMode = "arena-signal";

export const THEME_STORAGE_KEY = "novasight.theme";
export const DEFAULT_THEME: ThemeMode = "arena-signal";

export function isThemeMode(value: string | null): value is ThemeMode {
  return value === DEFAULT_THEME;
}

export function resolveStoredTheme(_value: string | null): ThemeMode {
  return DEFAULT_THEME;
}
