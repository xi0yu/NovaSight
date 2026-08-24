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
  const pickerRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const activeTheme = THEME_OPTIONS.find((option) => option.id === theme) ?? THEME_OPTIONS[0];

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (!pickerOpen) return;

    function closePickerFromOutside(event: PointerEvent) {
      if (!(event.target instanceof Node) || pickerRef.current?.contains(event.target)) return;
      setPickerOpen(false);
    }

    document.addEventListener("pointerdown", closePickerFromOutside);
    return () => document.removeEventListener("pointerdown", closePickerFromOutside);
  }, [pickerOpen]);

  function selectTheme(nextTheme: ThemeMode) {
    setTheme(nextTheme);
    setPickerOpen(false);
    toggleRef.current?.focus();
  }

  function closePickerFromKeyboard() {
    setPickerOpen(false);
    toggleRef.current?.focus();
  }

  return (
    <div
      className={`theme-picker${pickerOpen ? " is-open" : ""}`}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !pickerOpen) return;
        event.preventDefault();
        closePickerFromKeyboard();
      }}
      ref={pickerRef}
    >
      <button
        aria-controls="novasight-theme-menu"
        aria-expanded={pickerOpen}
        aria-label={`主题设置，当前：${activeTheme.label}`}
        className="theme-toggle"
        onClick={() => setPickerOpen((open) => !open)}
        ref={toggleRef}
        title="选择网站主题"
        type="button"
      >
        <span className={`theme-swatch theme-swatch-${activeTheme.id}`} aria-hidden="true" />
        <span className="theme-toggle-copy">
          <strong>主题</strong>
          <small>{activeTheme.label}</small>
        </span>
        <NovaIcon className="theme-toggle-chevron" name="expand" size={14} />
      </button>
      {pickerOpen ? <div className="theme-menu" id="novasight-theme-menu" role="group" aria-label="网站主题">
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
              {selected ? <span className="theme-option-state"><NovaIcon name="check-circle" size={15} />当前</span> : null}
            </button>
          );
        })}
      </div> : null}
    </div>
  );
}
