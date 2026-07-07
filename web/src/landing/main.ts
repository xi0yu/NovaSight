import "./landing.css";
import { mountLanding } from "./landing";

// 入口根：复用 Vite 注入的 #landing-root
const root = document.getElementById("landing-root");
if (!root) {
  throw new Error("landing-root not found");
}
mountLanding(root);
