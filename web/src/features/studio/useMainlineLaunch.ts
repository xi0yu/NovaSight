import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  emergencyStopRuntimePipeline,
  getApiErrorCode,
  getModelCatalog,
  getRuntimeState,
  selectCaptureProfile,
  startRuntimePipeline,
  stopRuntimePipeline,
  type CaptureSelectPayload,
  type CaptureState,
  type ModelCatalogDirectory,
  type ModelCatalogModel,
  type ModelCatalogResponse,
  type ModelProject,
  type RuntimeState
} from "../../api";
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import {
  getRuntimeMainlineStatus,
  type RuntimeMainlineStatus
} from "../shared/runtimeStatus";

export type MainlineLaunchStatus = "idle" | "running" | "success" | "failed" | "cancelled";
export type MainlineLaunchStepState = "pending" | "running" | "success" | "failed";

export type MainlineLaunchStage = {
  title: string;
  caption: string;
};

type UseMainlineLaunchInput = {
  activeModelPublished: boolean;
  buildCapturePayload: () => CaptureSelectPayload;
  navigateToInference: () => void;
  onEnsureProjects: (force?: boolean) => Promise<ModelProject[]>;
  onRefresh: () => Promise<void>;
  onRuntimeStateChange: (runtime: RuntimeState) => void;
  applyModelCatalogResult: (result: ModelCatalogResponse) => void;
  setBusy: (busy: string | null) => void;
  setLocalError: (message: string | null) => void;
  setModelCatalogLoading: (loading: boolean) => void;
  setModelCatalogMessage: (message: string) => void;
  setModelManagerDialogOpen: (open: boolean) => void;
  setSelectedModelCatalogPath: (path: string) => void;
  runtimeFatalError: boolean;
  runtimeInferenceDetail: string;
  runtimeInferenceReason: string;
  runtimeMainlineRunning: boolean;
  runtimeMainlineSelected: boolean;
  runtimeMainlineStatus: RuntimeMainlineStatus;
};

const LAUNCH_STATUS_REQUEST_TIMEOUT_MS = 15000;
const MODEL_GUIDANCE_OPENED_ERROR = "MODEL_GUIDANCE_OPENED";
const MODEL_UNAVAILABLE_ERROR_CODE = "model_unavailable";

// Output delivery is configured and connected independently. Mainline launch
// only proves capture, inference and mouse-algorithm consumption are ready.
export const MAINLINE_LAUNCH_STAGES: MainlineLaunchStage[] = [
  {
    title: "检查模型配置",
    caption: "没有已发布 Engine 时转入模型管理，配置完成后再继续启动。"
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
    caption: "确认目标选择、目标速度预测与连续非线性控制已开始读取识别结果；输出设备不影响本步骤。"
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

function modelLaunchCandidateRank(model: ModelCatalogModel): number {
  const recommendationRank = model.recommendation === "recommended"
    ? 0
    : model.recommendation === "unrated"
      ? 10
      : 20;
  const status = model.artifact_status ?? model.scan_status;
  const statusRank = status === "ready"
    ? 0
    : status === "pending" || status === "need_confirm"
      ? 1
      : status === "failed"
        ? 4
        : 8;
  return recommendationRank + statusRank;
}

function flattenLaunchCatalogModels(root: ModelCatalogDirectory | null): ModelCatalogModel[] {
  if (!root) {
    return [];
  }
  return root.children.flatMap((node) =>
    node.type === "model" ? [node] : flattenLaunchCatalogModels(node)
  );
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

export function useMainlineLaunch({
  activeModelPublished,
  applyModelCatalogResult,
  buildCapturePayload,
  navigateToInference,
  onEnsureProjects,
  onRefresh,
  onRuntimeStateChange,
  runtimeFatalError,
  runtimeInferenceDetail,
  runtimeInferenceReason,
  runtimeMainlineRunning,
  runtimeMainlineSelected,
  runtimeMainlineStatus,
  setBusy,
  setLocalError,
  setModelCatalogLoading,
  setModelCatalogMessage,
  setModelManagerDialogOpen,
  setSelectedModelCatalogPath
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
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
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
  }, [clearTimer]);

  useEffect(() => {
    if (!runtimeMainlineSelected) {
      clearAccepted();
      return;
    }
    if (runtimeMainlineRunning) {
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

  const ensurePublishedModelBeforeMainline = useCallback(async (options?: { force?: boolean }) => {
    if (activeModelPublished && options?.force !== true) {
      return true;
    }
    const openingMessage = options?.force === true
      ? "NovaSight 服务报告当前模型不可用，已转入模型管理。"
      : "启动前没有已发布 TensorRT Engine，已转入模型管理。";
    setLocalError(openingMessage);
    setModelCatalogMessage(openingMessage);
    setProgressDetail("未找到当前模型，正在读取模型目录并打开模型管理。");
    navigateToInference();
    setDialogOpen(false);
    setModelManagerDialogOpen(true);
    setModelCatalogLoading(true);
    try {
      await onEnsureProjects(false).catch(() => undefined);
      const result = await getModelCatalog(true);
      applyModelCatalogResult(result);
      const engineCandidates = flattenLaunchCatalogModels(result.root)
        .filter((model) => model.kind === "engine")
        .sort((left, right) =>
          modelLaunchCandidateRank(left) - modelLaunchCandidateRank(right) ||
          left.relative_path.localeCompare(right.relative_path)
        );
      const selectedCandidate = engineCandidates[0];
      if (!selectedCandidate) {
        const noModelMessage = "启动前没有找到 .engine 模型。请把 TensorRT engine 放入 models 目录，刷新模型后再切换。";
        setLocalError(noModelMessage);
        setModelCatalogMessage(noModelMessage);
        return false;
      }
      setSelectedModelCatalogPath(selectedCandidate.relative_path);
      const selectedMessage = `已预选 ${selectedCandidate.relative_path}。请点击“验证并切换到所选模型”，完成后再次启动主链。`;
      setLocalError(selectedMessage);
      setModelCatalogMessage(selectedMessage);
      if (result.updated_files > 0) {
        await onRefresh();
      }
    } catch (err) {
      const failureMessage = `模型管理打开失败：${getErrorMessage(err)}`;
      setLocalError(failureMessage);
      setModelCatalogMessage(failureMessage);
      reportError(err, { source: "mainline-model-guide", title: "模型管理打开失败" });
    } finally {
      setModelCatalogLoading(false);
    }
    return false;
  }, [
    activeModelPublished,
    applyModelCatalogResult,
    navigateToInference,
    onEnsureProjects,
    onRefresh,
    setLocalError,
    setModelCatalogLoading,
    setModelCatalogMessage,
    setModelManagerDialogOpen,
    setSelectedModelCatalogPath
  ]);

  const startInferenceThread = useCallback(async () => {
    if (!(await ensurePublishedModelBeforeMainline())) {
      return;
    }
    setBusy("runtime.start");
    setLocalError(null);
    try {
      const runtimeStart = asRecord(await startRuntimePipeline());
      if (readBoolean(runtimeStart.failed) || !readBoolean(runtimeStart.running)) {
        throw new Error(readString(runtimeStart.last_error, "NovaSight 服务未确认推理管线运行。"));
      }
      await onRefresh();
    } catch (err) {
      if (getApiErrorCode(err) === MODEL_UNAVAILABLE_ERROR_CODE) {
        await ensurePublishedModelBeforeMainline({ force: true });
        return;
      }
      setLocalError(`启动推理失败：${getErrorMessage(err)}`);
      reportError(err, { source: "studio", title: "操作失败" });
      await onRefresh();
    } finally {
      setBusy(null);
    }
  }, [ensurePublishedModelBeforeMainline, onRefresh, setBusy, setLocalError]);

  const waitForRuntimeEvidence = useCallback(async (
    stageTitle: string,
    hasEvidence: (state: RuntimeState) => boolean,
    missingMessage: string,
    timeoutMs = 6000,
    intervalMs = 600
  ) => {
    const deadline = Date.now() + timeoutMs;
    let lastSummary = "";
    while (Date.now() <= deadline) {
      if (cancelledRef.current) {
        throw new Error("launch cancelled");
      }
      const state = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
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
    cancelledRef.current = false;
    setBusy("runtime.start");
    setLocalError(null);
    setStatus("running");
    setError("");
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

    try {
      await runStage(0, async () => {
        const modelReady = await ensurePublishedModelBeforeMainline();
        if (!modelReady) {
          throw new Error(MODEL_GUIDANCE_OPENED_ERROR);
        }
      });
      await runStage(1, async () => {
        const state = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
        const runtimeStatus = getRuntimeMainlineStatus(state);
        if (runtimeStatus.failed) {
          throw new Error(runtimeStatus.readinessDetail);
        }
      });
      await runStage(2, async () => {
        const captureState = await selectCaptureProfile(buildCapturePayload());
        assertCaptureLaunchState(captureState);
      });
      await runStage(3, async () => {
        const runtimeStart = asRecord(await startRuntimePipeline());
        const accepted = readBoolean(runtimeStart.running, true);
        if (!accepted) {
          const reason = readString(runtimeStart.last_error, "NovaSight 服务未确认主链运行。");
          throw new Error(reason);
        }
        setAccepted(true);
        setMessage("NovaSight 已确认主链运行，正在确认识别结果已进入控制链路。");
      });
      await runStage(4, async () => {
        const state = await waitForRuntimeEvidence(
          "激活鼠标算法",
          (state) => getRuntimeMainlineStatus(state).hasRuntimeConsumption,
          "控制链路尚未读取识别结果，目标选择、目标速度预测与连续非线性控制没有输入。"
        );
        const runtimeStatus = getRuntimeMainlineStatus(state);
        setProgressDetail(`${runtimeStatus.readinessLabel}：${runtimeStatus.readinessDetail}`);
      });
      setStatus("success");
      setCompletedStages(MAINLINE_LAUNCH_STAGES.length);
      setProgressDetail((detail) => detail || "主链启动完成，运行状态已确认。");
      await onRefresh();
      showToast();
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      if (errorMessage === MODEL_GUIDANCE_OPENED_ERROR) {
        setStatus("idle");
        setError("");
        setProgressDetail("已转入模型管理，请完成模型切换后重新启动。");
        clearAccepted();
        return;
      }
      if (getApiErrorCode(err) === MODEL_UNAVAILABLE_ERROR_CODE) {
        await ensurePublishedModelBeforeMainline({ force: true });
        setStatus("idle");
        setError("");
        setProgressDetail("已转入模型管理，请完成模型切换后重新启动。");
        clearAccepted();
        return;
      }
      if (cancelledRef.current) {
        setStatus("cancelled");
        setError("");
        setProgressDetail("启动已取消，已停止继续等待启动阶段反馈。");
        return;
      }
      setStatus("failed");
      setError(errorMessage);
      clearAccepted();
      setLocalError(`启动主链失败：${errorMessage}`);
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
    ensurePublishedModelBeforeMainline,
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
    clearTimer();
    setStatus("cancelled");
    setError("");
    setProgressDetail("正在请求立即停止输出并中止启动，不再等待普通生命周期锁。");
    clearAccepted();
    setBusy(null);
    try {
      await emergencyStopRuntimePipeline();
      const stoppedState = await getRuntimeState(undefined, LAUNCH_STATUS_REQUEST_TIMEOUT_MS);
      onRuntimeStateChange(stoppedState);
      setProgressDetail("NovaSight 已确认紧急停止；旧输出已失效，启动流程已取消。");
      await onRefresh();
    } catch (err) {
      setLocalError(`取消启动失败：${getErrorMessage(err)}`);
      reportError(err, { source: "mainline-cancel", title: "取消启动失败" });
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
