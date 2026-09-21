export type ThemeMode = "studio" | "rose-white" | "graphite-red";

export type ThemeOption = {
  id: ThemeMode;
  label: string;
  character: string;
  palette: string;
};

export const THEME_STORAGE_KEY = "novasight.theme";
export const DEFAULT_THEME: ThemeMode = "studio";

export const THEME_OPTIONS: ThemeOption[] = [
  { id: "studio", label: "专业工作台", character: "NovaSight · 标准界面", palette: "冰川灰 · 深海蓝 · 信号青" },
  { id: "rose-white", label: "粉白清昼", character: "Product · 明亮界面", palette: "柔粉 · 雾白 · 石墨" },
  { id: "graphite-red", label: "黑灰红", character: "NovaSight · 深色界面", palette: "曜黑 · 石墨灰 · 安全红" }
];

export function isThemeMode(value: string | null): value is ThemeMode {
  return THEME_OPTIONS.some((option) => option.id === value);
}

export function resolveStoredTheme(value: string | null): ThemeMode {
  if (isThemeMode(value)) {
    return value;
  }
  if (value === "dark" || value === "tayama" || value === "frontier-industrial") {
    return "graphite-red";
  }
  if (value === "light" || ["momo", "elysia", "rem", "lusha"].includes(value ?? "")) {
    return "rose-white";
  }
  return DEFAULT_THEME;
}
