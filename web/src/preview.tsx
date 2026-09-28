import React from "react";
import ReactDOM from "react-dom/client";

import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { FrontendPreview } from "./features/studio/FrontendPreview";
import { installGlobalErrorGuards } from "./lib/errorGuards";
import "./design/tokens.css";
import "./styles.css";

installGlobalErrorGuards();
document.documentElement.dataset.theme = "arena-signal";

const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <AppErrorBoundary><FrontendPreview /></AppErrorBoundary>
    </React.StrictMode>
  );
}
