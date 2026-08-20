export type RuntimeSnapshotCursor = {
  daemonInstanceId: string | null;
  observedSequence: number;
  appliedSequence: number;
};

export type RuntimeSnapshotEnvelope = {
  daemonInstanceId: string;
  snapshotSequence: number;
  kind: "heartbeat" | "snapshot";
  full: boolean;
  requestGeneration?: number;
  currentRequestGeneration?: number;
};

export type RuntimeSnapshotGateResult = {
  accept: boolean;
  cursor: RuntimeSnapshotCursor;
  reason:
    | "accepted"
    | "heartbeat_observed"
    | "stale_request"
    | "bootstrap_snapshot_required"
    | "daemon_reset_snapshot_required"
    | "sequence_regressed"
    | "already_applied";
};

export const EMPTY_RUNTIME_SNAPSHOT_CURSOR: RuntimeSnapshotCursor = {
  daemonInstanceId: null,
  observedSequence: -1,
  appliedSequence: -1,
};

export function gateRuntimeSnapshot(
  current: RuntimeSnapshotCursor,
  envelope: RuntimeSnapshotEnvelope,
): RuntimeSnapshotGateResult {
  if (
    envelope.requestGeneration !== undefined
    && envelope.currentRequestGeneration !== undefined
    && envelope.requestGeneration !== envelope.currentRequestGeneration
  ) {
    return { accept: false, cursor: current, reason: "stale_request" };
  }

  if (current.daemonInstanceId === null) {
    if (envelope.kind !== "snapshot" || !envelope.full) {
      return { accept: false, cursor: current, reason: "bootstrap_snapshot_required" };
    }
    return {
      accept: true,
      cursor: {
        daemonInstanceId: envelope.daemonInstanceId,
        observedSequence: envelope.snapshotSequence,
        appliedSequence: envelope.snapshotSequence,
      },
      reason: "accepted",
    };
  }

  if (current.daemonInstanceId !== envelope.daemonInstanceId) {
    if (envelope.kind !== "snapshot" || !envelope.full) {
      return { accept: false, cursor: current, reason: "daemon_reset_snapshot_required" };
    }
    return {
      accept: true,
      cursor: {
        daemonInstanceId: envelope.daemonInstanceId,
        observedSequence: envelope.snapshotSequence,
        appliedSequence: envelope.snapshotSequence,
      },
      reason: "accepted",
    };
  }

  if (envelope.snapshotSequence < current.observedSequence) {
    return { accept: false, cursor: current, reason: "sequence_regressed" };
  }

  if (envelope.kind === "heartbeat") {
    return {
      accept: false,
      cursor: {
        ...current,
        observedSequence: Math.max(current.observedSequence, envelope.snapshotSequence),
      },
      reason: "heartbeat_observed",
    };
  }

  if (envelope.snapshotSequence < current.appliedSequence) {
    return {
      accept: false,
      cursor: {
        ...current,
        observedSequence: Math.max(current.observedSequence, envelope.snapshotSequence),
      },
      reason: "already_applied",
    };
  }

  return {
    accept: true,
    cursor: {
      daemonInstanceId: envelope.daemonInstanceId,
      observedSequence: Math.max(current.observedSequence, envelope.snapshotSequence),
      appliedSequence: envelope.snapshotSequence,
    },
    reason: "accepted",
  };
}
