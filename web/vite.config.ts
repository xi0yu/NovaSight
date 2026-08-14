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
    proxy: {
      "/api": RUST_DEV_ORIGIN,
      "/healthz": RUST_DEV_ORIGIN,
      "/ws": {
        target: RUST_DEV_ORIGIN,
        ws: true,
      },
    },
  },
  build: {
    outDir: "../out/web",
    emptyOutDir: true,
  },
});
