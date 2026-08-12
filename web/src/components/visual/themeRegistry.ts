export type ThemeMode = "momo" | "elysia" | "rem" | "lusha" | "tayama";

export type ThemeOption = {
  id: ThemeMode;
  label: string;
  character: string;
  palette: string;
};

export const THEME_STORAGE_KEY = "novasight.theme";
export const DEFAULT_THEME: ThemeMode = "momo";

export const THEME_OPTIONS: ThemeOption[] = [
  { id: "momo", label: "桃粉工作台", character: "Momo · 成年向导", palette: "桃粉 · 莓红 · 奶白" },
  { id: "elysia", label: "樱晶庭院", character: "Sakura · 视觉档案", palette: "樱花白 · 粉金" },
  { id: "rem", label: "苍雪校准", character: "Azure · 稳定档案", palette: "冰蓝 · 瓷白" },
  { id: "lusha", label: "鎏金王庭", character: "Gilded · 典藏档案", palette: "暖褐 · 象牙白 · 金" },
  { id: "tayama", label: "绯夜模式", character: "Crimson · 夜间档案", palette: "墨黑 · 深红" }
];

export function isThemeMode(value: string | null): value is ThemeMode {
  return THEME_OPTIONS.some((option) => option.id === value);
}

export function resolveStoredTheme(value: string | null): ThemeMode {
  if (isThemeMode(value)) {
    return value;
  }
  if (value === "dark") {
    return "tayama";
  }
  if (value === "light") {
    return "elysia";
  }
  return DEFAULT_THEME;
}
