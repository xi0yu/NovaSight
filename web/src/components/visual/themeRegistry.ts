export type ThemeMode = "momo" | "elysia" | "rem" | "lusha" | "tayama";

export type ThemeOption = {
  id: ThemeMode;
  label: string;
  character: string;
  palette: string;
};

export type ThemeStory = {
  id: ThemeMode;
  index: string;
  work: string;
  character: string;
  title: string;
  line: string;
  tags: readonly string[];
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

const THEME_STORY_DETAILS: Record<ThemeMode, Omit<ThemeStory, "id" | "index">> = {
  momo: {
    work: "NOVA ROSE",
    character: "Momo",
    title: "桃粉工作台",
    line: "个人授权以清晰状态卡呈现；每一步启动都有下一步。",
    tags: ["成年向导", "授权状态", "粉色工作台"]
  },
  elysia: {
    work: "NOVA SAKURA",
    character: "Sakura",
    title: "樱晶庭院",
    line: "把视觉链路收进温柔的粉白层次，保留工作台的清晰边界。",
    tags: ["樱色档案", "水晶花庭", "柔和校准"]
  },
  rem: {
    work: "NOVA AZURE",
    character: "Azure",
    title: "苍雪校准",
    line: "蓝色只服务稳定和状态，不把画面变成另一个系统。",
    tags: ["低温校准", "蓝白档案", "稳定优先"]
  },
  lusha: {
    work: "NOVA GILDED",
    character: "Gilded",
    title: "鎏金王庭",
    line: "金色只出现在收藏和档案层，运行控件保持克制。",
    tags: ["典藏档案", "暖金边界", "典藏感"]
  },
  tayama: {
    work: "NOVA CRIMSON",
    character: "Crimson",
    title: "绯夜模式",
    line: "夜间主题降低亮面干扰，把红色留给目标和危险动作。",
    tags: ["夜间工作台", "深红信号", "成熟漫画"]
  }
};

export const THEME_STORIES: ThemeStory[] = THEME_OPTIONS.map((option, index) => ({
  id: option.id,
  index: String(index + 1).padStart(2, "0"),
  ...THEME_STORY_DETAILS[option.id]
}));

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
