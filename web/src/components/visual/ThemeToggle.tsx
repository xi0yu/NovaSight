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

  return (
    <button
      aria-label={theme === "dark" ? "切换到浅色模式" : "切换到深色模式"}
      className="theme-toggle"
      onClick={() => setTheme(nextTheme)}
      title={theme === "dark" ? "浅色模式" : "深色模式"}
      type="button"
    >
      <NovaIcon name={theme === "dark" ? "show" : "hide"} size={15} />
      <span>{theme === "dark" ? "Dark" : "Light"}</span>
    </button>
  );
}
