import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 7351,
    strictPort: true,
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
    outDir: "../out/web",
    emptyOutDir: true,
  },
});
