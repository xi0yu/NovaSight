import { useStableSemanticValue } from "../shared/useStableSemanticValue";

export type ConfigApplyPresentationState = "applying" | "live" | "pending_process";

export interface ConfigApplyPresentation {
  state: ConfigApplyPresentationState;
  desiredRevision: number;
  effectiveRevision: number;
}

export function useConfigApplyPresentation({
  pendingWriteCount,
  restartRequired,
  desiredRevision,
  effectiveRevision,
  settleMs = 280
}: {
  pendingWriteCount: number;
  restartRequired: boolean;
  desiredRevision: number;
  effectiveRevision: number;
  settleMs?: number;
}): ConfigApplyPresentation {
  const reportedState: ConfigApplyPresentationState = pendingWriteCount > 0
    ? "applying"
    : restartRequired
      ? "pending_process"
      : "live";
  const reportedPresentation = {
    state: reportedState,
    desiredRevision,
    effectiveRevision
  };
  const stablePresentation = useStableSemanticValue(
    reportedPresentation,
    reportedState,
    reportedState === "applying" ? 0 : settleMs
  );

  // A write is user-visible immediately. Its exit waits for one stable
  // runtime status window, preventing a late pre-write WebSocket frame from
  // flashing "pending process" before the effective revision catches up.
  return reportedState === "applying" ? reportedPresentation : stablePresentation;
}
