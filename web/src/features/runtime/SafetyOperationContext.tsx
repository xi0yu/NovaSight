import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

import type { RuntimeState } from "../../api";
import { NovaIcon } from "../../components/visual";
import { isDaemonConfirmedSafe } from "./runtimeProjection";

import "./safety-operation.css";

type SafetyOperationEvidence = {
  phase: "idle" | "requesting" | "confirmed" | "unconfirmed";
  requestedDaemonId: string | null;
  requestedSequence: number | null;
  confirmedSequence: number | null;
  causalityBlocked: boolean;
  message: string;
};

type SafetyOperationContextValue = {
  beginEmergencyStop: (runtime: RuntimeState | null) => void;
  reconcileEmergencyStop: (runtime: RuntimeState) => "confirmed" | "safe_causality_unknown" | "not_safe" | "idle";
  markEmergencyStopCausalityUnknown: (message: string) => void;
  markEmergencyStopUnconfirmed: (message: string) => void;
};

const INITIAL_EVIDENCE: SafetyOperationEvidence = {
  phase: "idle",
  requestedDaemonId: null,
  requestedSequence: null,
  confirmedSequence: null,
  causalityBlocked: false,
  message: "",
};

const SafetyOperationContext = createContext<SafetyOperationContextValue | null>(null);

export function SafetyOperationProvider({ children }: { children: ReactNode }) {
  const [evidence, setEvidence] = useState<SafetyOperationEvidence>(INITIAL_EVIDENCE);
  const evidenceRef = useRef<SafetyOperationEvidence>(INITIAL_EVIDENCE);
  const updateEvidence = useCallback((next: SafetyOperationEvidence) => {
    evidenceRef.current = next;
    setEvidence(next);
  }, []);
  const beginEmergencyStop = useCallback((runtime: RuntimeState | null) => {
    updateEvidence({
      phase: "requesting",
      requestedDaemonId: runtime?.semantic.daemon_instance_id ?? null,
      requestedSequence: runtime?.semantic.snapshot_sequence ?? null,
      confirmedSequence: null,
      causalityBlocked: false,
      message: "正在请求 novasightd 立即关闭输出，并等待最终安全快照。",
    });
  }, [updateEvidence]);
  const reconcileEmergencyStop = useCallback((runtime: RuntimeState) => {
    const current = evidenceRef.current;
    if (current.phase === "idle" || current.phase === "confirmed") {
      return "idle";
    }
    if (!isDaemonConfirmedSafe(runtime)) {
      return "not_safe";
    }
    const sameDaemon = current.requestedDaemonId !== null
      && current.requestedDaemonId === runtime.semantic.daemon_instance_id;
    const newerSnapshot = current.requestedSequence !== null
      && runtime.semantic.snapshot_sequence > current.requestedSequence;
    if (sameDaemon && newerSnapshot && !current.causalityBlocked) {
      updateEvidence({
        ...current,
        phase: "confirmed",
        confirmedSequence: runtime.semantic.snapshot_sequence,
        message: "同一 novasightd 已用更新快照确认主链停止、kmNet 断开且不存在可发送输出。",
      });
      return "confirmed";
    }
    updateEvidence({
      ...current,
      phase: "unconfirmed",
      causalityBlocked: true,
      confirmedSequence: runtime.semantic.snapshot_sequence,
      message: "当前 novasightd 已确认安全，但 daemon 身份或快照序列无法证明这是本次紧急停止的操作回执。",
    });
    return "safe_causality_unknown";
  }, [updateEvidence]);
  const markEmergencyStopCausalityUnknown = useCallback((message: string) => {
    updateEvidence({ ...evidenceRef.current, phase: "unconfirmed", causalityBlocked: true, message });
  }, [updateEvidence]);
  const markEmergencyStopUnconfirmed = useCallback((message: string) => {
    updateEvidence({ ...evidenceRef.current, phase: "unconfirmed", message });
  }, [updateEvidence]);
  const value = useMemo(() => ({
    beginEmergencyStop,
    reconcileEmergencyStop,
    markEmergencyStopCausalityUnknown,
    markEmergencyStopUnconfirmed,
  }), [beginEmergencyStop, markEmergencyStopCausalityUnknown, reconcileEmergencyStop, markEmergencyStopUnconfirmed]);

  return (
    <SafetyOperationContext.Provider value={value}>
      {children}
      {evidence.phase !== "idle" ? (
        <aside className={`safety-operation-receipt ${evidence.phase}`} role="status" aria-live="polite">
          <NovaIcon name={evidence.phase === "confirmed" ? "shield-check" : "emergency-stop"} size={18} />
          <div>
            <strong>
              {evidence.phase === "requesting"
                ? "紧急停止确认中"
                : evidence.phase === "confirmed"
                  ? "紧急停止已确认"
                  : "紧急停止尚未确认"}
            </strong>
            <span>{evidence.message}</span>
            {evidence.requestedDaemonId ? (
              <small>
                {evidence.requestedDaemonId.slice(0, 10)} · 请求 #{evidence.requestedSequence ?? "—"}
                {evidence.confirmedSequence === null ? "" : ` · 确认 #${evidence.confirmedSequence}`}
              </small>
            ) : null}
          </div>
          <button type="button" onClick={() => updateEvidence(INITIAL_EVIDENCE)} aria-label="关闭紧急停止凭证">×</button>
        </aside>
      ) : null}
    </SafetyOperationContext.Provider>
  );
}

export function useSafetyOperation(): SafetyOperationContextValue {
  const value = useContext(SafetyOperationContext);
  if (!value) throw new Error("useSafetyOperation must be used inside SafetyOperationProvider");
  return value;
}
