import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  emergencyStopRuntimePipeline,
  getRuntimeState,
  selectCaptureProfile,
  startRuntimePipeline,
  stopRuntimePipeline,
  type CaptureSelectPayload,
  type CaptureState,
  type RuntimeState
} from "../../api";
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import {
  getRuntimeMainlineStatus,
  type RuntimeMainlineStatus
} from "../shared/runtimeStatus";
import { acquireBodyScrollLock, releaseBodyScrollLock } from "./dialogFocus";

export type MainlineLaunchStatus = "idle" | "running" | "success" | "failed" | "cancelled";
export type MainlineLaunchStepState = "pending" | "running" | "success" | "failed";

export type MainlineLaunchStage = {
  title: string;
  caption: string;
};

type UseMainlineLaunchInput = {
  activeModelPublished: boolean;
  buildCapturePayload: () => CaptureSelectPayload;
  onRefresh: () => Promise<void>;
  onRuntimeStateChange: (runtime: RuntimeState) => void;
  setBusy: (busy: string | null) => void;
  setLocalError: (message: string | null) => void;
  runtimeFatalError: boolean;
  runtimeInferenceDetail: string;
  runtimeInferenceReason: string;
  runtimeMainlineRunning: boolean;
  runtimeMainlineSelected: boolean;
  runtimeMainlineStatus: RuntimeMainlineStatus;
};

const LAUNCH_STATUS_REQUEST_TIMEOUT_MS = 15000;

// Output delivery is configured independently. Mainline launch proves the
// runtime owner is active; capture/inference evidence is required only when
// an active model exists.
export const MAINLINE_LAUNCH_STAGES: MainlineLaunchStage[] = [
  {
    title: "检查模型状态",
    caption: "没有已发布 Engine 时主链仍会启动，并保持等待模型。"
  },
  {
    title: "检查运行环境",
    caption: "确认 Studio 已连接到 Jetson 运行服务。"
  },
  {
    title: "应用采集配置",
    caption: "按当前设备、格式、分辨率与帧率选择采集配置。"
  },
  {
    title: "启动视觉控制",
    caption: "请求 NovaSight 启动采集、画面识别与控制功能。"
  },
  {
    title: "激活鼠标算法",
    caption: "有模型时确认算法读取识别结果；无模型时确认主链已进入等待状态。"
  }
];

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function readBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

export function resolveMainlineLaunchStepState(
  index: number,
  status: MainlineLaunchStatus,
  activeIndex: number,
  completedStages: number
): MainlineLaunchStepState {
  if (status === "success" || index < completedStages) {
    return "success";
  }
  if (status === "failed" && index === activeIndex) {
    return "failed";
  }
  if (status === "running" && index === activeIndex) {
    return "running";
  }
  return "pending";
}

function captureLaunchFailureMessage(state: CaptureState): string {
  return (
    readString(state.last_error, "") ||
    "采集配置未进入可用状态。"
  );
}

function assertCaptureLaunchState(state: CaptureState) {
  if (state.available !== true) {
    throw new Error(`采集阶段失败：${captureLaunchFailureMessage(state)}`);
  }
}

// Structured error envelope: { stage, action, detail, hint? }. All launch
// failure messages should flow through this so users see (a) which stage
// failed, (b) what the user/system was trying to do, and (c) what to try
// next. `detail` carries the underlying error message; the helper formats
// the final string used by both `setLocalError` and the error reporter.
export type LaunchErrorEnvelope = {
  stage: string;
  action: string;
  detail: string;
  hint?: string;
};

function formatLaunchError(envelope: LaunchErrorEnvelope): string {
  const head = `${envelope.stage}失败：${envelope.action}`;
  if (envelope.hint) {
    return `${head} · ${envelope.detail} · 建议：${envelope.hint}`;
  }
  return `${head} · ${envelope.detail}`;
}

export function useMainlineLaunch({
  activeModelPublished,
  buildCapturePayload,
  onRefresh,
  onRuntimeStateChange,
  runtimeFatalError,
  runtimeInferenceDetail,
  runtimeInferenceReason,
  runtimeMainlineRunning,
  runtimeMainlineSelected,
  runtimeMainlineStatus,
  setBusy,
  setLocalError
}: UseMainlineLaunchInput) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [status, setStatus] = useState<MainlineLaunchStatus>("idle");
  const [stageIndex, setStageIndex] = useState(0);
  const [completedStages, setCompletedStages] = useState(0);
  const [error, setError] = useState("");
  const [progressDetail, setProgressDetail] = useState("");
  const [toastVisible, setToastVisible] = useState(false);
  const [accepted, setAccepted] = useState(false);
  const [message, setMessage] = useState("");
  const cancelledRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const timerResolveRef = useRef<(() => void) | null>(null);
  const toastTimerRef = useRef<number | null>(null);
  // AbortController for the in-flight launch polling loop. cancel() and the
  // hook cleanup abort this so we don't keep hitting getRuntimeState for up
  // to LAUNCH_STATUS_REQUEST_TIMEOUT_MS (15s) after the user gives up.
  const launchAbortControllerRef = useRef<AbortController | null>(null);
  // Timer for auto-closing the dialog after a successful launch so the user
  // doesn't have to click the X. Cleared on cancel and on unmount.
  const successAutoCloseTimerRef = useRef<number | null>(null);
  // True while a cancel is in flight (between the user clicking the cancel
  // button and the emergency-stop roundtrip finishing). Blocks a re-launch
  // during this window so the new start's startRuntimePipeline doesn't race
  // with the old cancel's emergency stop.
  const cancelInFlightRef = useRef(false);
  // Mirror of the ref so the launch dialog's "重新启动" button can show a
  // disabled state while a previous cancel is still round-tripping. The
  // ref is the source of truth; the state is just for rendering.
  const [cancelInFlight, setCancelInFlight] = useState(false);
  useEffect(() => {
    if (cancelInFlightRef.current && !cancelInFlight) {
      setCancelInFlight(true);
    } else if (!cancelInFlightRef.current && cancelInFlight) {
      setCancelInFlight(false);
    }
  });

  const clearAccepted = useCallback(() => {
    setAccepted(false);
    setMessage("");
  }, []);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (timerResolveRef.current) {
      timerResolveRef.current();
      timerResolveRef.current = null;
    }
  }, []);

  const waitForLaunchFeedback = useCallback((ms: number) => new Promise<void>((resolve) => {
    clearTimer();
    timerResolveRef.current = resolve;
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      timerResolveRef.current = null;
      resolve();
    }, ms);
  }), [clearTimer]);

  const resetDialog = useCallback(() => {
    clearTimer();
    if (successAutoCloseTimerRef.current !== null) {
      window.clearTimeout(successAutoCloseTimerRef.current);
      successAutoCloseTimerRef.current = null;
    }
    cancelledRef.current = false;
    setStatus("idle");
    setStageIndex(0);
    setCompletedStages(0);
    setError("");
    setProgressDetail("");
  }, [clearTimer]);

  const openDialog = useCallback(() => {
    resetDialog();
    setDialogOpen(true);
  }, [resetDialog]);

  const closeDialog = useCallback(() => {
    if (status === "running") {
      return;
    }
    if (successAutoCloseTimerRef.current !== null) {
      window.clearTimeout(successAutoCloseTimerRef.current);
      successAutoCloseTimerRef.current = null;
    }
    setDialogOpen(false);
  }, [status]);

  const showToast = useCallback(() => {
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    setToastVisible(true);
    toastTimerRef.current = window.setTimeout(() => {
      toastTimerRef.current = null;
      setToastVisible(false);
    }, 2600);
  }, []);

  useEffect(() => {
    if (!dialogOpen) {
      return undefined;
    }
    acquireBodyScrollLock();
    return () => {
      releaseBodyScrollLock();
    };
  }, [dialogOpen]);

  useEffect(() => {
    if (!dialogOpen || status === "running") {
      return undefined;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setDialogOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [dialogOpen, status]);

  useEffect(() => () => {
    clearTimer();
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
      toastTimerRef.current = null;
    }
    if (successAutoCloseTimerRef.current !== null) {
      window.clearTimeout(successAutoCloseTimerRef.current);
      successAutoCloseTimerRef.current = null;
    }
    // Abort any in-flight launch polling on unmount so a navigated-away
    // component doesn't keep hitting the backend for up to 15s.
    launchAbortControllerRef.current?.abort();
    launchAbortControllerRef.current = null;
  }, [clearTimer]);

  useEffect(() => {
    if (!runtimeMainlineSelected) {
      clearAccepted();
      return;
    }
    // Only clear `accepted` when the runtime is healthy. If the runtime
    // reports `failed` or a fatal error after a successful stage 3, keep
    // `accepted` so the post-start error branch below can surface the
    // failure to the user.
    if (runtimeMainlineRunning && !runtimeMainlineStatus.failed && !runtimeFatalError) {
      clearAccepted();
      return;
    }
    if (
      accepted &&
      (runtimeMainlineStatus.failed || runtimeFatalError)
    ) {
      clearAccepted();
      setLocalError(`主链启动未确认：${runtimeMainlineStatus.readinessDetail || runtimeInferenceDetail || runtimeInferenceReason || "NovaSight 服务未进入运行状态。"}`);
    }
  }, [
    accepted,
    clearAccepted,
    runtimeFatalError,
    runtimeInferenceDetail,
    runtimeInferenceReason,
    runtimeMainlineRunning,
    runtimeMainlineSelected,
    runtimeMainlineStatus.failed,
    runtimeMainlineStatus.readinessDetail,
    setLocalError
  ]);

  const startInferenceThread = useCallback(async () => {
    setBusy("runtime.start");
    setLocalError(null);
    try {
      const runtimeStart = asRecord(await startRuntimePipeline());
      if (readBoolean(runtimeStart.failed) || !readBoolean(runtimeStart.running)) {
        throw new Error(readString(runtimeStart.last_error, "NovaSight 服务未确认推理管线运行。"));
      }
      await onRefresh();
    } catch (err) {
      setLocalError(formatLaunchError({
        stage: "启动推理",
        action: "请求 nova 控制链进入运行态",
        detail: getErrorMessage(err),
        hint: "检查采集设备与 kmNet 通路，再重试启动。"
      }));
      reportError(err, { source: "studio", title: "启动推理失败" });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [onRefresh, setBusy, setLocalError]);

  const waitForRuntimeEvidence = useCallback(async (
    stageTitle: string,
    hasEvidence: (state: RuntimeState) => boolean,
    missingMessage: string,
    timeoutMs = 6000,
    intervalMs = 600,
    signal?: AbortSignal
  ) => {
    const deadline = Date.now() + timeoutMs;
    let lastSummary = "";
    while (Date.now() <= deadline) {
      if (cancelledRef.current) {
        throw new Error("launch cancelled");
      }
      if (signal?.aborted) {
        throw new Error("launch cancelled");
      }
      const state = await getRuntimeState(signal, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
      const runtimeStatus = getRuntimeMainlineStatus(state);
      lastSummary = runtimeStatus.progressSummary;
      setProgressDetail(
        runtimeStatus.outputTrace?.detail ||
          (runtimeStatus.progressSummary ? `当前计数：${runtimeStatus.progressSummary}` : "等待主链启动状态。")
      );
      if (runtimeStatus.failed) {
        throw new Error(`${stageTitle}失败：${runtimeStatus.outputTrace?.detail || runtimeStatus.readinessDetail || missingMessage}`);
      }
      if (!runtimeStatus.running) {
        await waitForLaunchFeedback(intervalMs);
        continue;
      }
      if (hasEvidence(state)) {
        setProgressDetail(
          runtimeStatus.outputTrace?.detail ||
            (runtimeStatus.progressSummary ? `已收到启动状态：${runtimeStatus.progressSummary}` : "已收到启动状态。")
        );
        return state;
      }
      await waitForLaunchFeedback(intervalMs);
    }
    throw new Error(`${stageTitle}失败：${missingMessage}${lastSummary ? ` 当前计数：${lastSummary}` : ""}`);
  }, [waitForLaunchFeedback]);

  const start = useCallback(async () => {
    if (status === "running") {
      return;
    }
    if (cancelInFlightRef.current) {
      // A previous cancel is still round-tripping; don't start a new launch
      // that would race with the in-flight emergency stop.
      return;
    }
    cancelledRef.current = false;
    // Cancel any pending success-auto-close from a previous launch so the
    // timer doesn't fire and close the dialog mid-run.
    if (successAutoCloseTimerRef.current !== null) {
      window.clearTimeout(successAutoCloseTimerRef.current);
      successAutoCloseTimerRef.current = null;
    }
    // Fresh controller for this launch attempt. Any prior controller is
    // aborted first so we don't leak a stale one if start() runs twice.
    launchAbortControllerRef.current?.abort();
    const controller = new AbortController();
    launchAbortControllerRef.current = controller;
    const signal = controller.signal;
    setBusy("runtime.start");
    setLocalError(null);
    setStatus("running");
    setError("");
    setMessage("");
    setProgressDetail("正在提交启动请求，等待启动阶段反馈。");
    setStageIndex(0);
    setCompletedStages(0);
    const ensureNotCancelled = () => {
      if (cancelledRef.current) {
        throw new Error("launch cancelled");
      }
    };
    const runStage = async (index: number, action?: () => Promise<void>) => {
      ensureNotCancelled();
      setStageIndex(index);
      if (action) {
        await action();
      }
      ensureNotCancelled();
      setCompletedStages(index + 1);
    };
    const stageLabel = (index: number): string =>
      MAINLINE_LAUNCH_STAGES[index]?.title ?? `阶段 ${index + 1}`;

    try {
      await runStage(0, async () => {
        setProgressDetail(activeModelPublished
          ? `已确认活动模型：${stageLabel(0)}`
          : "当前没有活动模型；主链将先启动并等待模型发布。"
        );
      });
      await runStage(1, async () => {
        setProgressDetail(`正在采集运行态：${stageLabel(1)}`);
        if (signal.aborted) {
          throw new Error("launch cancelled");
        }
        const state = await getRuntimeState(signal, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
        const runtimeStatus = getRuntimeMainlineStatus(state);
        if (runtimeStatus.failed) {
          throw new Error(runtimeStatus.readinessDetail);
        }
      });
      await runStage(2, async () => {
        setProgressDetail(`正在选择采集规格：${stageLabel(2)}`);
        const captureState = await selectCaptureProfile(buildCapturePayload(), signal);
        assertCaptureLaunchState(captureState);
      });
      await runStage(3, async () => {
        setProgressDetail(`正在启动主链：${stageLabel(3)}`);
        const runtimeStart = asRecord(await startRuntimePipeline(signal));
        const accepted = readBoolean(runtimeStart.running, true);
        if (!accepted) {
          const reason = readString(runtimeStart.last_error, "NovaSight 服务未确认主链运行。");
          throw new Error(reason);
        }
        setAccepted(true);
        setMessage("NovaSight 已确认主链运行，正在确认识别结果已进入控制链路。");
      });
      await runStage(4, async () => {
        if (!activeModelPublished) {
          const state = await waitForRuntimeEvidence(
            "进入模型等待状态",
            (state) => getRuntimeMainlineStatus(state).running,
            "控制主链尚未进入运行态。",
            6000,
            600,
            signal
          );
          const runtimeStatus = getRuntimeMainlineStatus(state);
          setProgressDetail(`主链已运行，等待模型发布。${runtimeStatus.progressSummary ? ` ${runtimeStatus.progressSummary}` : ""}`);
          return;
        }
        const state = await waitForRuntimeEvidence(
          "激活鼠标算法",
          (state) => getRuntimeMainlineStatus(state).hasRuntimeConsumption,
          "控制链路尚未读取识别结果，目标选择、目标速度预测与连续非线性控制没有输入。",
          6000,
          600,
          signal
        );
        const runtimeStatus = getRuntimeMainlineStatus(state);
        setProgressDetail(`${runtimeStatus.readinessLabel}：${runtimeStatus.readinessDetail}`);
      });
      setStatus("success");
      setCompletedStages(MAINLINE_LAUNCH_STAGES.length);
      setProgressDetail((detail) => detail || "主链启动完成，运行状态已确认。");
      await onRefresh();
      showToast();
      // Auto-close the dialog after the toast finishes so the user doesn't
      // have to dismiss it manually. Cancelled / failed flows keep manual
      // close so the user can read the message.
      if (successAutoCloseTimerRef.current !== null) {
        window.clearTimeout(successAutoCloseTimerRef.current);
      }
      successAutoCloseTimerRef.current = window.setTimeout(() => {
        successAutoCloseTimerRef.current = null;
        closeDialog();
      }, 1500);
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      // If the polling was aborted (cancel or unmount), the request fails
      // with an AbortError. Don't surface that as a launch failure — cancel
      // already updated status to "cancelled" in its own handler.
      if (launchAbortControllerRef.current?.signal.aborted || cancelledRef.current) {
        return;
      }
      if (cancelledRef.current) {
        setStatus("cancelled");
        setError("");
        clearAccepted();
        setProgressDetail("启动已取消，已停止继续等待启动阶段反馈。");
        return;
      }
      setStatus("failed");
      setError(errorMessage);
      clearAccepted();
      setLocalError(formatLaunchError({
        stage: "启动主链",
        action: "等待推理或控制链进入运行态超时",
        detail: errorMessage,
        hint: "查看错误中心的后端上下文，或刷新运行态后重试。"
      }));
      reportError(err, { source: "mainline-launch", title: "启动主链失败" });
      try {
        const stoppedState = await stopRuntimePipeline();
        onRuntimeStateChange(stoppedState);
      } catch {
        // Keep the original launch error visible; refresh below exposes stop failures if backend reports them.
      }
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [
    buildCapturePayload,
    clearAccepted,
    activeModelPublished,
    onRefresh,
    onRuntimeStateChange,
    setBusy,
    setLocalError,
    showToast,
    status,
    waitForRuntimeEvidence
  ]);

  const cancel = useCallback(async () => {
    if (status !== "running") {
      closeDialog();
      return;
    }
    cancelledRef.current = true;
    cancelInFlightRef.current = true;
    // Abort the in-flight launch polling immediately so we don't keep
    // hitting getRuntimeState for the remainder of the 15s timeout window.
    launchAbortControllerRef.current?.abort();
    launchAbortControllerRef.current = null;
    clearTimer();
    setStatus("cancelled");
    setError("");
    setProgressDetail("正在请求立即停止输出并中止启动，不再等待普通生命周期锁。");
    clearAccepted();
    setBusy(null);
    // The post-cancel confirmation fetch uses its own controller so the
    // launch abort above doesn't tear it down — we still want to confirm
    // that the backend accepted the emergency stop.
    const cancelController = new AbortController();
    try {
      await emergencyStopRuntimePipeline(cancelController.signal);
      const stoppedState = await getRuntimeState(
        cancelController.signal,
        LAUNCH_STATUS_REQUEST_TIMEOUT_MS
      );
      onRuntimeStateChange(stoppedState);
      setProgressDetail("NovaSight 已确认紧急停止；待发送输出已失效，启动流程已取消。");
      await onRefresh();
    } catch (err) {
      // AbortError after a follow-up cancel is expected; don't surface it.
      if (cancelController.signal.aborted) {
        return;
      }
      setProgressDetail("紧急停止未完成；后端可能仍在运行，请进入错误中心确认状态。");
      setLocalError(formatLaunchError({
        stage: "取消启动",
        action: "调用紧急停止以终止运行态",
        detail: getErrorMessage(err),
        hint: "手动进入错误中心查看后端是否已停止；必要时重新启动 nova 控制链。"
      }));
      reportError(err, { source: "mainline-cancel", title: "取消启动失败" });
    } finally {
      cancelInFlightRef.current = false;
    }
  }, [
    clearAccepted,
    clearTimer,
    closeDialog,
    onRefresh,
    onRuntimeStateChange,
    setBusy,
    setLocalError,
    status
  ]);

  const progress = useMemo(
    () => status === "success"
      ? 100
      : Math.round((completedStages / MAINLINE_LAUNCH_STAGES.length) * 100),
    [completedStages, status]
  );

  const summary = useMemo(
    () => status === "success"
      ? "主链启动完成"
      : status === "failed"
        ? "启动在当前步骤中断"
        : status === "cancelled"
          ? "启动流程已取消"
          : status === "running"
            ? `正在执行第 ${Math.min(stageIndex + 1, MAINLINE_LAUNCH_STAGES.length)} 项`
            : "等待用户确认启动",
    [stageIndex, status]
  );

  return {
    accepted,
    cancel,
    cancelInFlight,
    clearAccepted,
    closeDialog,
    completedStages,
    dialogOpen,
    error,
    message,
    openDialog,
    progress,
    progressDetail,
    stageIndex,
    stages: MAINLINE_LAUNCH_STAGES,
    start,
    startInferenceThread,
    status,
    summary,
    toastVisible
  };
}
