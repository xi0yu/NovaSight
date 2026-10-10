import { expect, it, vi } from "vitest";
import { publishModel, rollbackModel } from "../api";


it("requests physical-output acknowledgement only after operator consent", async () => {
  const headers: Array<string | null> = [];
  vi.stubGlobal("fetch", async (_url: RequestInfo | URL, init?: RequestInit) => {
    headers.push(new Headers(init?.headers).get("x-novasight-physical-output-ack"));
    return new Response("blocked", { status: 428 });
  });
  await expect(publishModel(1, 2)).rejects.toThrow();
  await expect(publishModel(1, 2, "auto", true)).rejects.toThrow();
  await expect(rollbackModel(1)).rejects.toThrow();
  await expect(rollbackModel(1, true)).rejects.toThrow();
  expect(headers).toEqual([null, "confirmed", null, "confirmed"]);
});
