import { useEffect, useRef, useState } from "react";

import { NovaIcon } from "./NovaIcon";

type ThemeMode = "momo" | "elysia" | "rem" | "lusha" | "tayama";

type ThemeOption = {
  id: ThemeMode;
  label: string;
  character: string;
  palette: string;
};

const THEME_STORAGE_KEY = "novasight.theme";
const DEFAULT_THEME: ThemeMode = "momo";
const THEME_OPTIONS: ThemeOption[] = [
  { id: "momo", label: "桃粉礼物", character: "Momo · 成年向导", palette: "礼物粉 · 莓红 · 奶白" },
  { id: "elysia", label: "樱晶庭院", character: "Sakura · 视觉档案", palette: "樱花白 · 粉金" },
  { id: "rem", label: "苍雪校准", character: "Azure · 稳定档案", palette: "冰蓝 · 瓷白" },
  { id: "lusha", label: "鎏金王庭", character: "Gilded · 典藏档案", palette: "暖褐 · 象牙白 · 金" },
  { id: "tayama", label: "绯夜模式", character: "Crimson · 夜间档案", palette: "墨黑 · 深红" }
];

function isThemeMode(value: string | null): value is ThemeMode {
  return THEME_OPTIONS.some((option) => option.id === value);
}

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return DEFAULT_THEME;
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (isThemeMode(storedTheme)) {
    return storedTheme;
  }

  // Preserve the intent of the retired two-theme system.
  if (storedTheme === "dark") {
    return "tayama";
  }
  if (storedTheme === "light") {
    return "elysia";
  }
  return DEFAULT_THEME;
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [pickerOpen, setPickerOpen] = useState(false);
  const pickerRef = useRef<HTMLDetailsElement>(null);
  const activeTheme = THEME_OPTIONS.find((option) => option.id === theme) ?? THEME_OPTIONS[0];

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  function selectTheme(nextTheme: ThemeMode) {
    setTheme(nextTheme);
    if (pickerRef.current) {
      pickerRef.current.open = false;
    }
    setPickerOpen(false);
  }

  return (
    <details
      className="theme-picker"
      onToggle={(event) => setPickerOpen(event.currentTarget.open)}
      ref={pickerRef}
    >
      <summary
        aria-label={`当前主题：${activeTheme.label}，角色：${activeTheme.character}`}
        className="theme-toggle"
        title="选择网站主题"
      >
        <span className={`theme-swatch theme-swatch-${activeTheme.id}`} aria-hidden="true" />
        <span className="theme-toggle-copy">
          <strong>{activeTheme.label}</strong>
          <small>{activeTheme.character}</small>
        </span>
        <NovaIcon className="theme-toggle-chevron" name="expand" size={14} />
      </summary>
      {pickerOpen ? <div className="theme-menu" aria-label="网站主题">
        <div className="theme-menu-heading">
          <span>THEME ARCHIVE</span>
          <strong>选择视觉主题</strong>
        </div>
        {THEME_OPTIONS.map((option) => {
          const selected = option.id === theme;
          return (
            <button
              aria-pressed={selected}
              className="theme-option"
              key={option.id}
              onClick={() => selectTheme(option.id)}
              type="button"
            >
              <span className={`theme-option-art theme-option-art-${option.id}`} aria-hidden="true" />
              <span className="theme-option-copy">
                <strong>{option.label}</strong>
                <small>{option.character} · {option.palette}</small>
              </span>
              {selected ? <NovaIcon name="check-circle" size={17} /> : null}
            </button>
          );
        })}
      </div> : null}
    </details>
  );
}
