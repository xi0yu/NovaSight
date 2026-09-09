import { describe, expect, it } from "vitest";

import type { ModelPublishResponse } from "../../api";
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
