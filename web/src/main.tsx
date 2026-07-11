import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { installGlobalErrorGuards } from "./lib/errorGuards";
import "./design/tokens.css";
import "./styles.css";

installGlobalErrorGuards();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
