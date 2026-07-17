import { useEffect, useRef, useState } from "react";

import { NovaIcon } from "./NovaIcon";

type ThemeMode = "elysia" | "rem" | "lusha" | "tayama";

type ThemeOption = {
  id: ThemeMode;
  label: string;
  character: string;
  palette: string;
};

const THEME_STORAGE_KEY = "novasight.theme";
const DEFAULT_THEME: ThemeMode = "elysia";
const THEME_OPTIONS: ThemeOption[] = [
  { id: "elysia", label: "樱晶花园", character: "爱莉希雅", palette: "樱花白 · 粉金" },
  { id: "rem", label: "苍雪女仆", character: "雷姆", palette: "冰蓝 · 瓷白" },
  { id: "lusha", label: "鎏金王庭", character: "露莎公主", palette: "暖褐 · 象牙白 · 金" },
  { id: "tayama", label: "绯夜烟巷", character: "田山小姐", palette: "墨黑 · 深红" }
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
  return storedTheme === "dark" ? "tayama" : DEFAULT_THEME;
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
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
  }

  return (
    <details className="theme-picker" ref={pickerRef}>
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
      <div className="theme-menu" aria-label="网站主题">
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
              <span className={`theme-swatch theme-swatch-${option.id}`} aria-hidden="true" />
              <span className="theme-option-copy">
                <strong>{option.label}</strong>
                <small>{option.character} · {option.palette}</small>
              </span>
              {selected ? <NovaIcon name="check-circle" size={17} /> : null}
            </button>
          );
        })}
      </div>
    </details>
  );
}
