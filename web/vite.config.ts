import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5174",
      "/healthz": "http://127.0.0.1:5174"
    }
  }
});
