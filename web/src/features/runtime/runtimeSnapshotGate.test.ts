import { describe, expect, it } from "vitest";

import {
  EMPTY_RUNTIME_SNAPSHOT_CURSOR,
  gateRuntimeSnapshot,
  type RuntimeSnapshotCursor,
} from "./runtimeSnapshotGate";

const established: RuntimeSnapshotCursor = {
  daemonInstanceId: "daemon-a",
  observedSequence: 12,
  appliedSequence: 10,
};

describe("runtime snapshot gate", () => {
  it("requires a full snapshot to establish or replace daemon identity", () => {
    expect(gateRuntimeSnapshot(EMPTY_RUNTIME_SNAPSHOT_CURSOR, {
      daemonInstanceId: "daemon-a",
      snapshotSequence: 1,
      kind: "heartbeat",
      full: false,
    }).reason).toBe("bootstrap_snapshot_required");

    expect(gateRuntimeSnapshot(established, {
      daemonInstanceId: "daemon-b",
      snapshotSequence: 1,
      kind: "snapshot",
      full: false,
    }).reason).toBe("daemon_reset_snapshot_required");

    const reset = gateRuntimeSnapshot(established, {
      daemonInstanceId: "daemon-b",
      snapshotSequence: 1,
      kind: "snapshot",
      full: true,
    });
    expect(reset.accept).toBe(true);
    expect(reset.cursor).toEqual({
      daemonInstanceId: "daemon-b",
      observedSequence: 1,
      appliedSequence: 1,
    });
  });

  it("lets a heartbeat fence older snapshots without repainting", () => {
    const heartbeat = gateRuntimeSnapshot(established, {
      daemonInstanceId: "daemon-a",
      snapshotSequence: 14,
      kind: "heartbeat",
      full: false,
    });
    expect(heartbeat.accept).toBe(false);
    expect(heartbeat.cursor.observedSequence).toBe(14);
    expect(heartbeat.cursor.appliedSequence).toBe(10);

    expect(gateRuntimeSnapshot(heartbeat.cursor, {
      daemonInstanceId: "daemon-a",
      snapshotSequence: 13,
      kind: "snapshot",
      full: false,
    }).reason).toBe("sequence_regressed");
  });

  it("accepts the matching snapshot at the observed heartbeat sequence", () => {
    const result = gateRuntimeSnapshot(established, {
      daemonInstanceId: "daemon-a",
      snapshotSequence: 12,
      kind: "snapshot",
      full: false,
    });
    expect(result.accept).toBe(true);
    expect(result.cursor.appliedSequence).toBe(12);
  });

  it("accepts an equal-sequence topic-complete replacement", () => {
    const result = gateRuntimeSnapshot({ ...established, observedSequence: 10 }, {
      daemonInstanceId: "daemon-a",
      snapshotSequence: 10,
      kind: "snapshot",
      full: false,
    });
    expect(result.accept).toBe(true);
    expect(result.cursor.appliedSequence).toBe(10);
  });

  it("drops callbacks from an obsolete browser request generation", () => {
    const result = gateRuntimeSnapshot(established, {
      daemonInstanceId: "daemon-a",
      snapshotSequence: 13,
      kind: "snapshot",
      full: true,
      requestGeneration: 4,
      currentRequestGeneration: 5,
    });
    expect(result.accept).toBe(false);
    expect(result.reason).toBe("stale_request");
  });
});
