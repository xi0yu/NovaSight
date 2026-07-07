import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5174",
      "/healthz": "http://127.0.0.1:5174",
      "/ws": {
        target: "ws://127.0.0.1:5174",
        ws: true,
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        // 控制台 React 工程 (不改动, 维持原行为)
        main: resolve(__dirname, "index.html"),
        // 独立营销入口页 (新增, 访问 /landing/)
        landing: resolve(__dirname, "landing.html"),
      },
    },
  },
});
