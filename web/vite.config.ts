import { defineConfig } from "vite";
import endpointContract from "../deploy/studio-endpoints.json";

const studioEndpoint = endpointContract.studio;
const rustDevEndpoint = endpointContract.frontend_development_api;
const RUST_DEV_ORIGIN = `http://${rustDevEndpoint.host}:${rustDevEndpoint.port}`;

export default defineConfig({
  server: {
    host: studioEndpoint.host,
    port: studioEndpoint.port,
    strictPort: true,
    // novasight-web validates the browser Origin against this public Host.
    proxy: {
      "/api": {
        target: RUST_DEV_ORIGIN,
        changeOrigin: false,
      },
      "/healthz": {
        target: RUST_DEV_ORIGIN,
        changeOrigin: false,
      },
      "/ws": {
        target: RUST_DEV_ORIGIN,
        changeOrigin: false,
        ws: true,
      },
    },
  },
  build: {
    outDir: "../out/web",
    emptyOutDir: true,
  },
});
