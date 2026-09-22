import { afterEach, describe, expect, it, vi } from "vitest";

import { publishModel, rollbackModel, type ModelPublishResponse } from "../../api";
import { isActiveModelNoOp } from "./useModelSwitchWorkflow";

function response(overrides: Partial<ModelPublishResponse["report"]> = {}): ModelPublishResponse {
  return {
    deployment: { artifact_id: 42 },
    report: {
      applied: false,
      rolled_back: false,
      artifact_id: 42,
      ...overrides,
    },
  } as unknown as ModelPublishResponse;
}

describe("model switch no-op", () => {
  it("accepts selecting the already active artifact as a no-op success", () => {
    expect(isActiveModelNoOp(response(), 42)).toBe(true);
  });

  it("does not hide rollback or mismatched-artifact failures", () => {
    expect(isActiveModelNoOp(response({ rolled_back: true }), 42)).toBe(false);
    expect(isActiveModelNoOp(response({ artifact_id: 41 }), 42)).toBe(false);
    expect(isActiveModelNoOp(response(), 41)).toBe(false);
  });
});

afterEach(() => vi.unstubAllGlobals());

it("sends physical-output acknowledgement only when explicitly requested", async () => {
  const fetchMock = vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) => new Response("blocked", { status: 428 }));
  vi.stubGlobal("fetch", fetchMock);
  await expect(publishModel(1, 2)).rejects.toThrow();
  await expect(publishModel(1, 2, "auto", true)).rejects.toThrow();
  await expect(rollbackModel(1)).rejects.toThrow();
  await expect(rollbackModel(1, true)).rejects.toThrow();
  expect(fetchMock.mock.calls.map(([, init]) => new Headers(init?.headers).get("x-novasight-physical-output-ack")))
    .toEqual([null, "confirmed", null, "confirmed"]);
});
