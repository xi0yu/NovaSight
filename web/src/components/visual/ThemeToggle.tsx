import { useEffect, useState } from "react";

import { NovaIcon } from "./NovaIcon";

type ThemeMode = "light" | "dark";

const THEME_STORAGE_KEY = "novasight.theme";

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "light";
  }
  return window.localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const nextTheme = theme === "dark" ? "light" : "dark";
  const currentLabel = theme === "dark" ? "绯夜红黑" : "樱花白";
  const nextLabel = nextTheme === "dark" ? "绯夜红黑" : "樱花白";

  return (
    <button
      aria-label={`当前为${currentLabel}主题，切换到${nextLabel}主题`}
      className="theme-toggle"
      onClick={() => setTheme(nextTheme)}
      title={`切换到${nextLabel}`}
      type="button"
    >
      <NovaIcon name={theme === "dark" ? "show" : "hide"} size={15} />
      <span>{currentLabel}</span>
    </button>
  );
}
