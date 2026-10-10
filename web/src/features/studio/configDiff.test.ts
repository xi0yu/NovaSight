import { expect, it } from "vitest";
import { configDifferences } from "./configDiff";

it("shows exact changes, process scope and redacted nested credentials", () => {
  const changes = configDifferences(
    { revision: 1, pipeline: { p_response_scale: 0.5 }, server: { port: 7351 }, hardware: { mouse_protocols: [], password: "old-secret" } },
    { revision: 2, pipeline: { p_response_scale: 0.8 }, server: { port: 7352 }, hardware: { mouse_protocols: [{ id: "one", token: "new-secret" }], password: "new-secret" } },
  );
  expect(changes.find((item) => item.path === "pipeline.p_response_scale")).toMatchObject({ before: "0.5", after: "0.8", restart: false });
  expect(changes.find((item) => item.path === "server.port")?.restart).toBe(true);
  expect(JSON.stringify(changes)).not.toMatch(/old-secret|new-secret/);
  expect(changes.some((item) => item.path === "revision")).toBe(false);
  expect(configDifferences({ revision: 1, pipeline: { enabled: true } }, { revision: 2, pipeline: { enabled: true } })).toEqual([]);
});
