import { useEffect, useRef, useState } from "react";

import { NovaIcon } from "./NovaIcon";
import { DEFAULT_THEME, THEME_OPTIONS, THEME_STORAGE_KEY, resolveStoredTheme, type ThemeMode } from "./themeRegistry";

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return DEFAULT_THEME;
  }

  return resolveStoredTheme(window.localStorage.getItem(THEME_STORAGE_KEY));
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
            <button type="button"
              aria-pressed={selected}
              className="theme-option"
              key={option.id}
              onClick={() => selectTheme(option.id)}
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
