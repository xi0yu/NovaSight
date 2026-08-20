import { useCallback, useEffect, useRef, useState } from "react";

import {
  emergencyStopRuntimePipeline,
  getRuntimeState,
  startRuntimePipeline,
  stopRuntimePipeline,
  type RuntimeState
} from "../../api";
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import { isDaemonConfirmedSafe } from "../runtime/runtimeProjection";
import { useSafetyOperation } from "../runtime/SafetyOperationContext";

type UseMainlineLaunchInput = {
  onRefresh: () => Promise<void>;
  onRuntimeStateChange: (runtime: RuntimeState) => boolean;
  runtime: RuntimeState | null;
  setBusy: (busy: string | null) => void;
  setLocalError: (message: string | null) => void;
};

const START_RECONCILE_TIMEOUT_MS = 5000;

function runtimeAcceptedStart(runtime: RuntimeState): boolean {
  return ["starting", "waiting_model", "running", "standby"].includes(
    runtime.semantic.phase
  );
}

function runtimeAcceptedEmergencyStop(runtime: RuntimeState): boolean {
  return isDaemonConfirmedSafe(runtime);
}

function stopError(detail: string): string {
  return `停止主链未确认：${detail} · 建议：查看异常信息中的后端原因并确认当前运行状态。`;
}

function launchError(detail: string): string {
  return `启动主链失败：请求 NovaSight 使用已保存配置进入运行态 · ${detail} · 建议：查看异常信息中的后端原因后重试。`;
}

export function useMainlineLaunch({
  onRefresh,
  onRuntimeStateChange,
  runtime,
  setBusy,
  setLocalError
}: UseMainlineLaunchInput) {
  const mountedRef = useRef(true);
  const {
    beginEmergencyStop,
    markEmergencyStopCausalityUnknown,
    reconcileEmergencyStop,
    markEmergencyStopUnconfirmed,
  } = useSafetyOperation();
  const requestControllerRef = useRef<AbortController | null>(null);
  const emergencyInFlightRef = useRef(false);
  const [emergencyStopping, setEmergencyStopping] = useState(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (emergencyInFlightRef.current) {
        markEmergencyStopCausalityUnknown("页面会话已结束，紧急停止请求因果未知。请重新认证后核对当前运行状态；系统不会自动重发请求。");
      }
      requestControllerRef.current?.abort();
      requestControllerRef.current = null;
    };
  }, [markEmergencyStopCausalityUnknown]);

  const start = useCallback(async () => {
    if (requestControllerRef.current !== null) {
      return;
    }
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setBusy("runtime.start");
    setLocalError(null);

    try {
      let requestError: unknown = null;
      try {
        await startRuntimePipeline(controller.signal);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        requestError = error;
      }

      if (requestError === null) {
        try {
          await onRefresh();
        } catch (error) {
          if (!mountedRef.current) {
            return;
          }
          const detail = getErrorMessage(error);
          setLocalError(`运行请求已完成，但最新运行状态刷新失败：${detail}`);
          reportError(error, { source: "mainline-refresh", title: "运行状态刷新失败" });
        }
        return;
      }

      let acceptedByRuntime = false;
      try {
        const runtime = await getRuntimeState(controller.signal, START_RECONCILE_TIMEOUT_MS);
        if (!mountedRef.current) {
          return;
        }
        onRuntimeStateChange(runtime);
        acceptedByRuntime = runtimeAcceptedStart(runtime);
      } catch (error) {
        if (!mountedRef.current) {
          return;
        }
        reportError(error, { source: "mainline-reconcile", title: "启动状态确认失败" });
      }

      if (!acceptedByRuntime) {
        const detail = getErrorMessage(requestError);
        setLocalError(launchError(detail));
        reportError(requestError, { source: "mainline-launch", title: "启动主链失败" });
      }

      try {
        await onRefresh();
      } catch (error) {
        if (mountedRef.current) {
          reportError(error, { source: "mainline-refresh", title: "运行状态刷新失败" });
        }
      }
    } finally {
      const ownsRequest = requestControllerRef.current === controller;
      if (ownsRequest) {
        requestControllerRef.current = null;
      }
      if (mountedRef.current && ownsRequest) {
        setBusy(null);
      }
    }
  }, [onRefresh, onRuntimeStateChange, setBusy, setLocalError]);

  const stop = useCallback(async () => {
    if (requestControllerRef.current !== null) {
      return;
    }
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setBusy("runtime.stop");
    setLocalError(null);

    try {
      let requestError: unknown = null;
      try {
        const runtime = await stopRuntimePipeline(controller.signal);
        if (!mountedRef.current) {
          return;
        }
        onRuntimeStateChange(runtime);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        requestError = error;
      }

      let acceptedByRuntime = false;
      try {
        const runtime = await getRuntimeState(controller.signal, START_RECONCILE_TIMEOUT_MS);
        if (!mountedRef.current) {
          return;
        }
        const acceptedByGate = onRuntimeStateChange(runtime);
        acceptedByRuntime = acceptedByGate && runtimeAcceptedEmergencyStop(runtime);
      } catch (error) {
        if (!mountedRef.current) {
          return;
        }
        reportError(error, {
          source: "mainline-stop-reconcile",
          title: "停止状态确认失败"
        });
      }
      if (!acceptedByRuntime) {
        const detail = requestError === null
          ? "后端未返回严格的停止、kmNet 断开、输出阻止状态"
          : getErrorMessage(requestError);
        setLocalError(stopError(detail));
        if (requestError !== null) {
          reportError(requestError, { source: "mainline-stop", title: "停止主链失败" });
        }
      }

      try {
        await onRefresh();
      } catch (error) {
        if (mountedRef.current) {
          reportError(error, { source: "mainline-stop-refresh", title: "停止状态刷新失败" });
        }
      }
    } finally {
      const ownsRequest = requestControllerRef.current === controller;
      if (ownsRequest) {
        requestControllerRef.current = null;
      }
      if (mountedRef.current && ownsRequest) {
        setBusy(null);
      }
    }
  }, [onRefresh, onRuntimeStateChange, setBusy, setLocalError]);

  const emergencyStop = useCallback(async () => {
    if (emergencyInFlightRef.current) {
      return;
    }
    emergencyInFlightRef.current = true;
    const interruptedLifecycleRequest = requestControllerRef.current !== null;
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setEmergencyStopping(true);
    setLocalError(null);
    beginEmergencyStop(runtime);

    try {
      let requestError: unknown = null;
      try {
        await emergencyStopRuntimePipeline(controller.signal);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        requestError = error;
      }

      let acceptedByRuntime = false;
      let currentSafeButCausalityUnknown = false;
      try {
        const runtime = await getRuntimeState(controller.signal, START_RECONCILE_TIMEOUT_MS);
        if (!mountedRef.current) {
          return;
        }
        const acceptedByGate = onRuntimeStateChange(runtime);
        if (acceptedByGate && runtimeAcceptedEmergencyStop(runtime)) {
          if (requestError === null) {
            acceptedByRuntime = reconcileEmergencyStop(runtime) === "confirmed";
          } else {
            currentSafeButCausalityUnknown = true;
            markEmergencyStopCausalityUnknown("当前 novasightd 已确认安全，但紧急停止请求返回错误，无法证明该安全快照是本次操作的回执。");
          }
        }
      } catch (error) {
        if (!mountedRef.current) {
          return;
        }
        reportError(error, { source: "emergency-stop-reconcile", title: "紧急停止状态确认失败" });
      }
      if (!acceptedByRuntime) {
        const detail = requestError === null
          ? "后端未返回严格的停止、kmNet 断开、输出阻止状态"
          : getErrorMessage(requestError);
        setLocalError(currentSafeButCausalityUnknown
          ? `当前输出已确认安全，但紧急停止请求因果未确认：${detail}`
          : `紧急停止未确认：${detail} · 请查看异常信息并确认物理输出状态。`);
        if (!currentSafeButCausalityUnknown) {
          markEmergencyStopUnconfirmed(`无法取得严格安全快照：${detail}`);
        }
        if (requestError !== null) {
          reportError(requestError, { source: "emergency-stop", title: "紧急停止失败" });
        }
      }

      try {
        await onRefresh();
      } catch (error) {
        if (mountedRef.current) {
          reportError(error, { source: "emergency-stop-refresh", title: "紧急停止状态刷新失败" });
        }
      }
    } finally {
      const ownsRequest = requestControllerRef.current === controller;
      if (ownsRequest) {
        requestControllerRef.current = null;
      }
      emergencyInFlightRef.current = false;
      if (mountedRef.current) {
        setEmergencyStopping(false);
        if (interruptedLifecycleRequest) {
          setBusy(null);
        }
      }
    }
  }, [beginEmergencyStop, markEmergencyStopCausalityUnknown, markEmergencyStopUnconfirmed, onRefresh, onRuntimeStateChange, reconcileEmergencyStop, runtime, setBusy, setLocalError]);

  return { emergencyStop, emergencyStopping, start, stop };
}
