import "./previewNetwork";
import React from "react";
import ReactDOM from "react-dom/client";
import "./design/tokens.css";
import "./styles.css";

import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { PreviewStates } from "./features/studio/PreviewStates";
import { installGlobalErrorGuards } from "./lib/errorGuards";

import "./design/studio-theme.css";

installGlobalErrorGuards();
document.documentElement.dataset.theme = "nova-daylight";

const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <AppErrorBoundary><PreviewStates /></AppErrorBoundary>
    </React.StrictMode>
  );
}
