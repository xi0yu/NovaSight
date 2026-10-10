import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e", testMatch: "auth-session-contract.spec.ts", workers: 2,
  outputDir: "../out/playwright-auth-static", reporter: "list",
  use: { ...devices["Desktop Chrome"], baseURL: "http://127.0.0.1:7352", trace: "retain-on-failure" },
  webServer: {
    command: "node scripts/serve-auth-static.mjs",
    url: "http://127.0.0.1:7352", reuseExistingServer: false, timeout: 30_000,
  },
});
