import React from "react";
import ReactDOM from "react-dom/client";
import "./design/tokens.css";
import "./styles.css";

import App from "./App";
import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { installGlobalErrorGuards } from "./lib/errorGuards";

installGlobalErrorGuards();

document.documentElement.dataset.theme = "arena-signal";

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
