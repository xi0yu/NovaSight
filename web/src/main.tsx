import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { installGlobalErrorGuards } from "./lib/errorGuards";
import "./design/tokens.css";
import "./styles.css";

const storedTheme = window.localStorage.getItem("novasight.theme");
document.documentElement.dataset.theme = ["elysia", "rem", "lusha", "tayama"].includes(storedTheme ?? "")
  ? storedTheme ?? "elysia"
  : storedTheme === "dark" ? "tayama" : "elysia";

installGlobalErrorGuards();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
