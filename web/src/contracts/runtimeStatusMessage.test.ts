import { describe, expect, it } from "vitest";

import { RuntimeContractError, decodeRuntimeStatusMessage } from "./runtime";

describe("runtime status message identity", () => {
  it("decodes a heartbeat with a non-empty daemon identity", () => {
    expect(decodeRuntimeStatusMessage({
      kind: "runtime_heartbeat",
      daemon_instance_id: "daemon-a",
      snapshot_sequence: 7,
    })).toEqual({
      kind: "runtime_heartbeat",
      daemon_instance_id: "daemon-a",
      snapshot_sequence: 7,
    });
  });

  it("fails closed when a heartbeat omits or blanks daemon identity", () => {
    expect(() => decodeRuntimeStatusMessage({
      kind: "runtime_heartbeat",
      snapshot_sequence: 7,
    })).toThrow(RuntimeContractError);
    expect(() => decodeRuntimeStatusMessage({
      kind: "runtime_heartbeat",
      daemon_instance_id: "   ",
      snapshot_sequence: 7,
    })).toThrow(/non-empty string/);
  });
});
