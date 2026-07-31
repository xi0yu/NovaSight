import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { resolveStoredTheme, THEME_STORAGE_KEY } from "./components/visual/themeRegistry";
import { installGlobalErrorGuards } from "./lib/errorGuards";
import "./design/tokens.css";
import "./styles.css";

const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
document.documentElement.dataset.theme = resolveStoredTheme(storedTheme);

installGlobalErrorGuards();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
