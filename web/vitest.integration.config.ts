import { defineConfig } from "vitest/config";
import unitConfig from "./vitest.config";

export default defineConfig({
  ...unitConfig,
  test: {
    ...unitConfig.test,
    include: ["src/**/*.integration.test.{ts,tsx,mjs}"],
    exclude: [],
  },
});
