import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { resolveStoredTheme, THEME_STORAGE_KEY } from "./components/visual/themeRegistry";
import { installGlobalErrorGuards } from "./lib/errorGuards";
import "./design/tokens.css";
import "./styles.css";

installGlobalErrorGuards();

try {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  document.documentElement.dataset.theme = resolveStoredTheme(storedTheme);
} catch {
  document.documentElement.dataset.theme = resolveStoredTheme(null);
}

const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <AppErrorBoundary>
        <App />
      </AppErrorBoundary>
    </React.StrictMode>
  );
} else {
  const message = document.createElement("p");
  message.setAttribute("role", "alert");
  message.textContent = "NovaSight 启动失败：页面缺少 #root 挂载节点。";
  document.body.replaceChildren(message);
}
