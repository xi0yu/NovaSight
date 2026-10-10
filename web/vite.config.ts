import { defineConfig } from "vite";
import { randomUUID } from "node:crypto";
import { cpSync, mkdirSync, readdirSync, renameSync, rmSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
const buildVersion = randomUUID();
const publishedRoot = fileURLToPath(new URL("../out/web", import.meta.url));
const stagedRoot = fileURLToPath(new URL(`../out/web-build-${buildVersion}`, import.meta.url));
import endpointContract from "../deploy/studio-endpoints.json";

const studioEndpoint = endpointContract.studio;
const rustDevEndpoint = endpointContract.frontend_development_api;
const RUST_DEV_ORIGIN = `http://${rustDevEndpoint.host}:${rustDevEndpoint.port}`;

export default defineConfig({
  define: { __NOVASIGHT_BUILD_VERSION__: JSON.stringify(buildVersion) },
  plugins: [{ name: "novasight-version", apply: "build", generateBundle() {
    this.emitFile({ type: "asset", fileName: "version.json", source: JSON.stringify({ version: buildVersion }) });
  }, closeBundle() {
    mkdirSync(publishedRoot, { recursive: true });
    for (const entry of readdirSync(stagedRoot)) {
      if (entry === "index.html" || entry === "version.json") continue;
      cpSync(join(stagedRoot, entry), join(publishedRoot, entry), { recursive: true, force: false });
    }
    // Publish the entry only after every hashed resource exists; publish the version hint last.
    for (const entry of ["index.html", "version.json"]) {
      cpSync(join(stagedRoot, entry), join(publishedRoot, `.${entry}.${buildVersion}`));
      renameSync(join(publishedRoot, `.${entry}.${buildVersion}`), join(publishedRoot, entry));
    }
    rmSync(stagedRoot, { recursive: true });
  } }],
  server: {
    host: "127.0.0.1",
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
    outDir: stagedRoot,
    emptyOutDir: true,
  },
});
