import { useCallback, useEffect, useRef, useState } from "react";

import { StatusIndicator } from "./components/ui";
import {
  HealthResponse,
  LicenseStatus,
  ModelProject,
  RuntimeConfig,
  RuntimeState,
  getHealth,
  getLicenseStatus,
  getModelProjects,
  getRuntimeConfig,
  getRuntimeState,
  statusWebSocketUrl,
} from "./api";
import { ToastHost } from "./components/ToastHost";
import { reportError, reportInfo, reportSuccess } from "./lib/toast";
import {
  clearWebSocketFailure,
  reportWebSocketFailure,
  reportWebSocketRecovered
} from "./lib/errorGuards";
import { LicenseGate } from "./features/license/LicenseView";
import { LICENSE_CACHE_KEY } from "./features/license/storage";
import { StudioConsoleView } from "./features/studio/StudioConsoleView";
import { VisualSystemView } from "./features/visual-system/VisualSystemView";
import { formatTime, getErrorMessage } from "./features/shared/format";
import {
  runtimeDeliveryLabel,
  runtimeDeliveryTone,
  type RuntimeDeliveryStatus
} from "./features/shared/runtimeDelivery";

type ErrorKey = "health" | "runtime" | "config" | "projects" | "capture";
type LoadState = {
  loading: boolean;
  errors: Partial<Record<ErrorKey, string>>;
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  config: RuntimeConfig | null;
  projects: ModelProject[];
  lastUpdated: Date | null;
};

const initialState: LoadState = {
  loading: true,
  errors: {},
  health: null,
  runtime: null,
  config: null,
  projects: [],
  lastUpdated: null
};

const RUNTIME_FALLBACK_INTERVAL_MS = 3000;
const HEALTH_FALLBACK_INTERVAL_MS = 30000;
const STATUS_STALE_AFTER_MS = 3000;
const STATUS_FIRST_MESSAGE_TIMEOUT_MS = 8000;
const STATUS_RECONNECT_BASE_MS = 1000;
const STATUS_RECONNECT_MAX_MS = 15000;

function withoutError(
  errors: LoadState["errors"],
  key: ErrorKey
): LoadState["errors"] {
  const nextErrors = { ...errors };
  delete nextErrors[key];
  return nextErrors;
}

function isAbortError(error: unknown): boolean {
  return typeof error === "object" && error !== null && "name" in error && error.name === "AbortError";
}

const visualSystemMode = new URLSearchParams(window.location.search).get("visual-system") === "1";

export default function App() {
  return visualSystemMode ? <VisualSystemView /> : <StudioApp />;
}

function StudioApp() {
  const [state, setState] = useState<LoadState>(initialState);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [realtimeStatus, setRealtimeStatus] = useState<RuntimeDeliveryStatus>("disconnected");
  const [pageVisible, setPageVisible] = useState(() => document.visibilityState !== "hidden");
  const [networkOnline, setNetworkOnline] = useState(() => navigator.onLine);
  const loadRequestSeqRef = useRef(0);
  const licenseRequestSeqRef = useRef(0);
  const runtimeRevisionRef = useRef(0);
  const initialLoadStartedRef = useRef(false);
  const backgroundLoadInFlightRef = useRef(false);
  const healthLoadInFlightRef = useRef(false);
  const foregroundLoadInFlightRef = useRef(false);
  const runtimeFallbackAbortRef = useRef<AbortController | null>(null);
  const healthRefreshAbortRef = useRef<AbortController | null>(null);
  const [licenseLoading, setLicenseLoading] = useState(true);
  const [licenseError, setLicenseError] = useState<string | undefined>();

  const applyRuntimeState = useCallback((runtime: RuntimeState) => {
    const receivedAt = Date.now();
    runtimeRevisionRef.current += 1;
    setState((current) => ({
      ...current,
      errors: withoutError(current.errors, "runtime"),
      runtime,
      lastUpdated: new Date(receivedAt)
    }));
  }, []);

  const applyRuntimeConfig = useCallback((config: RuntimeConfig) => {
    setState((current) => ({
      ...current,
      errors: withoutError(current.errors, "config"),
      config
    }));
  }, []);

  const loadLicense = useCallback(async () => {
    const requestSeq = licenseRequestSeqRef.current + 1;
    licenseRequestSeqRef.current = requestSeq;
    setLicenseLoading(true);
    setLicenseError(undefined);
    try {
      const cached = localStorage.getItem(LICENSE_CACHE_KEY) === "1";
      const status = await getLicenseStatus();
      if (requestSeq !== licenseRequestSeqRef.current) {
        return status;
      }
      setLicense(status);
      if (status.valid) {
        localStorage.setItem(LICENSE_CACHE_KEY, "1");
        if (cached) {
          reportInfo("授权已从本地缓存命中", "本次刷新沿用上一次的授权结论。", "license");
        }
      } else {
        localStorage.removeItem(LICENSE_CACHE_KEY);
      }
      return status;
    } catch (err) {
      if (requestSeq !== licenseRequestSeqRef.current) {
        return null;
      }
      setLicense(null);
      setLicenseError(getErrorMessage(err));
      localStorage.removeItem(LICENSE_CACHE_KEY);
      reportError(err, {
        source: "license",
        title: "授权校验失败",
        fallback: "无法连接 NovaSight 后端"
      });
      return null;
    } finally {
      if (requestSeq === licenseRequestSeqRef.current) {
        setLicenseLoading(false);
      }
    }
  }, []);

  const handleLicenseChange = useCallback((status: LicenseStatus) => {
    setLicense(status);
    if (status.valid) {
      localStorage.setItem(LICENSE_CACHE_KEY, "1");
    } else {
      localStorage.removeItem(LICENSE_CACHE_KEY);
      loadRequestSeqRef.current += 1;
      setState(initialState);
      setRealtimeStatus("disconnected");
      runtimeRevisionRef.current = 0;
      initialLoadStartedRef.current = false;
      backgroundLoadInFlightRef.current = false;
      healthLoadInFlightRef.current = false;
      foregroundLoadInFlightRef.current = false;
      runtimeFallbackAbortRef.current?.abort();
      healthRefreshAbortRef.current?.abort();
      runtimeFallbackAbortRef.current = null;
      healthRefreshAbortRef.current = null;
      clearWebSocketFailure("status");
    }
  }, []);

  const load = useCallback(async () => {
    const requestSeq = loadRequestSeqRef.current + 1;
    const runtimeRevisionAtRequest = runtimeRevisionRef.current;
    loadRequestSeqRef.current = requestSeq;
    foregroundLoadInFlightRef.current = true;
    setState((current) => ({ ...current, loading: true, errors: {} }));
    try {
      const [health, runtime, config, projects] = await Promise.allSettled([
        getHealth(),
        getRuntimeState(),
        getRuntimeConfig(),
        getModelProjects()
      ]);
      const runtimeSuperseded = runtimeRevisionRef.current !== runtimeRevisionAtRequest;
      const sourceMap: Array<[PromiseSettledResult<unknown>, string]> = [
        [health, "health"],
        [runtime, "runtime"],
        [config, "config"],
        [projects, "projects"]
      ];
      const seenReason = new Set<unknown>();
      for (const [result, source] of sourceMap) {
        if (source === "runtime" && runtimeSuperseded) {
          continue;
        }
        if (result.status === "rejected" && !seenReason.has(result.reason)) {
          seenReason.add(result.reason);
          reportError(result.reason, {
            source,
            title: source === "runtime" ? "运行态失败" : "请求失败"
          });
        }
      }
      setState((current) => {
        if (requestSeq !== loadRequestSeqRef.current) {
          return current;
        }
        const errors: LoadState["errors"] = {};
        if (health.status === "rejected") {
          errors.health = getErrorMessage(health.reason);
        }
        if (runtime.status === "rejected" && !runtimeSuperseded) {
          errors.runtime = getErrorMessage(runtime.reason);
        }
        if (config.status === "rejected") {
          errors.config = getErrorMessage(config.reason);
        }
        if (projects.status === "rejected") {
          errors.projects = getErrorMessage(projects.reason);
        }
        const shouldApplyRuntime =
          runtime.status === "fulfilled" && runtimeRevisionRef.current === runtimeRevisionAtRequest;
        if (shouldApplyRuntime) {
          runtimeRevisionRef.current += 1;
        }
        return {
          loading: false,
          errors,
          health: health.status === "fulfilled" ? health.value : current.health,
          runtime: shouldApplyRuntime ? runtime.value : current.runtime,
          config: config.status === "fulfilled" ? config.value : current.config,
          projects: projects.status === "fulfilled" ? projects.value : current.projects,
          lastUpdated: shouldApplyRuntime ? new Date() : current.lastUpdated
        };
      });
    } finally {
      foregroundLoadInFlightRef.current = false;
    }
  }, []);

  const refreshRuntime = useCallback(async () => {
    if (backgroundLoadInFlightRef.current || foregroundLoadInFlightRef.current) {
      return;
    }
    backgroundLoadInFlightRef.current = true;
    const runtimeRevisionAtRequest = runtimeRevisionRef.current;
    const abortController = new AbortController();
    runtimeFallbackAbortRef.current = abortController;
    try {
      const runtime = await getRuntimeState(abortController.signal);
      if (runtimeRevisionRef.current !== runtimeRevisionAtRequest) {
        return;
      }
      applyRuntimeState(runtime);
      setRealtimeStatus((current) => current === "connected" ? current : "fallback");
    } catch (err) {
      if (isAbortError(err)) {
        return;
      }
      setState((current) => ({
        ...current,
        errors: { ...current.errors, runtime: getErrorMessage(err) }
      }));
      setRealtimeStatus((current) => current === "connected" ? current : "disconnected");
    } finally {
      if (runtimeFallbackAbortRef.current === abortController) {
        runtimeFallbackAbortRef.current = null;
      }
      backgroundLoadInFlightRef.current = false;
    }
  }, [applyRuntimeState]);

  const refreshHealth = useCallback(async () => {
    if (healthLoadInFlightRef.current) {
      return;
    }
    healthLoadInFlightRef.current = true;
    const abortController = new AbortController();
    healthRefreshAbortRef.current = abortController;
    try {
      const health = await getHealth(abortController.signal);
      setState((current) => ({
        ...current,
        errors: withoutError(current.errors, "health"),
        health
      }));
    } catch (err) {
      if (isAbortError(err)) {
        return;
      }
      setState((current) => ({
        ...current,
        errors: { ...current.errors, health: getErrorMessage(err) },
        health: null
      }));
    } finally {
      if (healthRefreshAbortRef.current === abortController) {
        healthRefreshAbortRef.current = null;
      }
      healthLoadInFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    void loadLicense();
  }, [loadLicense]);

  useEffect(() => {
    if (license?.valid && networkOnline && pageVisible && !initialLoadStartedRef.current) {
      initialLoadStartedRef.current = true;
      void load();
    }
  }, [license?.valid, load, networkOnline, pageVisible]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      const visible = document.visibilityState !== "hidden";
      setPageVisible(visible);
      if (!visible) {
        runtimeFallbackAbortRef.current?.abort();
        healthRefreshAbortRef.current?.abort();
      }
      if (visible && license?.valid && initialLoadStartedRef.current) {
        void refreshRuntime();
        void refreshHealth();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, [license?.valid, refreshHealth, refreshRuntime]);

  useEffect(() => {
    const handleOnline = () => {
      setNetworkOnline(true);
      if (pageVisible && license?.valid && initialLoadStartedRef.current) {
        void refreshRuntime();
        void refreshHealth();
      }
    };
    const handleOffline = () => {
      setNetworkOnline(false);
      runtimeFallbackAbortRef.current?.abort();
      healthRefreshAbortRef.current?.abort();
      setState((current) => ({
        ...current,
        errors: { ...current.errors, health: "浏览器网络已断开" },
        health: null
      }));
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [license?.valid, pageVisible, refreshHealth, refreshRuntime]);

  useEffect(() => {
    if (!license?.valid || !pageVisible) {
      setRealtimeStatus("paused");
      clearWebSocketFailure("status");
      return undefined;
    }
    if (!networkOnline) {
      setRealtimeStatus("offline");
      return undefined;
    }

    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let staleTimer: number | null = null;
    let reconnectAttempt = 0;
    let latestMessageAt = 0;
    let connectionStartedAt = 0;

    const scheduleReconnect = () => {
      if (!active || reconnectTimer !== null) {
        return;
      }
      const delay = Math.min(
        STATUS_RECONNECT_BASE_MS * 2 ** reconnectAttempt,
        STATUS_RECONNECT_MAX_MS
      );
      reconnectAttempt += 1;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, delay);
    };

    const connect = () => {
      if (!active) {
        return;
      }
      setRealtimeStatus((current) => current === "fallback" ? current : "connecting");
      let failureReported = false;
      let closedForStaleData = false;
      latestMessageAt = 0;
      connectionStartedAt = Date.now();
      if (staleTimer !== null) {
        window.clearInterval(staleTimer);
      }
      const nextSocket = new WebSocket(statusWebSocketUrl());
      socket = nextSocket;
      nextSocket.onerror = (event) => {
        if (!active) {
          return;
        }
        setRealtimeStatus((current) => current === "fallback" ? current : "disconnected");
        failureReported = true;
        if (reportWebSocketFailure(event, "status")) {
          void refreshHealth();
        }
      };
      nextSocket.onclose = (event) => {
        if (!active || socket !== nextSocket) {
          return;
        }
        socket = null;
        setRealtimeStatus((current) =>
          current === "fallback" ? current : closedForStaleData ? "stale" : "disconnected"
        );
        if (!closedForStaleData && !failureReported && event.code !== 1000 && event.code !== 1001) {
          if (reportWebSocketFailure(event.reason || `code=${event.code}`, "status")) {
            void refreshHealth();
          }
        }
        scheduleReconnect();
      };
      nextSocket.onmessage = (event) => {
        if (!active || socket !== nextSocket) {
          return;
        }
        try {
          const runtime = JSON.parse(String(event.data)) as RuntimeState;
          const receivedAt = Date.now();
          latestMessageAt = receivedAt;
          reconnectAttempt = 0;
          applyRuntimeState(runtime);
          setRealtimeStatus("connected");
          reportWebSocketRecovered("status");
        } catch {
          // Ignore malformed status frames; REST refresh still provides recovery.
        }
      };

      staleTimer = window.setInterval(() => {
        if (socket !== nextSocket) {
          return;
        }
        const now = Date.now();
        if (latestMessageAt > 0 && now - latestMessageAt > STATUS_STALE_AFTER_MS) {
          closedForStaleData = true;
          setRealtimeStatus("stale");
          nextSocket.close(4000, "status stream stale");
          return;
        }
        if (latestMessageAt === 0 && now - connectionStartedAt > STATUS_FIRST_MESSAGE_TIMEOUT_MS) {
          failureReported = true;
          setRealtimeStatus((current) => current === "fallback" ? current : "disconnected");
          if (reportWebSocketFailure("first status frame timed out", "status")) {
            void refreshHealth();
          }
          nextSocket.close();
        }
      }, 1000);
    };

    connect();
    return () => {
      active = false;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (staleTimer !== null) {
        window.clearInterval(staleTimer);
      }
      socket?.close(1000, "client suspended");
    };
  }, [applyRuntimeState, license?.valid, networkOnline, pageVisible, refreshHealth]);

  const realtimeConnected = realtimeStatus === "connected";

  useEffect(() => {
    if (!license?.valid || !networkOnline || !pageVisible || realtimeConnected) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      void refreshRuntime();
    }, RUNTIME_FALLBACK_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [license?.valid, networkOnline, pageVisible, realtimeConnected, refreshRuntime]);

  useEffect(() => {
    if (!license?.valid || !networkOnline || !pageVisible || realtimeConnected) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      void refreshHealth();
    }, HEALTH_FALLBACK_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [license?.valid, networkOnline, pageVisible, realtimeConnected, refreshHealth]);

  if (!license?.valid) {
    return (
      <LicenseGate
        license={license}
        loading={licenseLoading}
        error={licenseError}
        onRefresh={() => void loadLicense()}
        onLicenseChange={handleLicenseChange}
      />
    );
  }

  const hasErrors = Object.keys(state.errors).length > 0;
  const consoleStatus = (
    <>
      <StatusIndicator tone={hasErrors ? "bad" : state.health?.ok ? "good" : "warn"}>
        {hasErrors ? "后端 部分异常" : state.health?.ok ? "后端 已连接" : "后端 检查中"}
      </StatusIndicator>
      <StatusIndicator tone={runtimeDeliveryTone(realtimeStatus)}>
        {runtimeDeliveryLabel(realtimeStatus)}
      </StatusIndicator>
      <span className="last-updated">更新于 {formatTime(state.lastUpdated)}</span>
    </>
  );

  return (
    <>
      <StudioConsoleView
        health={state.health}
        runtime={state.runtime}
        runtimeConfig={state.config}
        projects={state.projects}
        errors={state.errors}
        lastUpdated={state.lastUpdated}
        realtimeStatus={realtimeStatus}
        onRefresh={load}
        onRuntimeConfigChange={applyRuntimeConfig}
        onRuntimeStateChange={applyRuntimeState}
      />
      <ToastHost />
      <div className="visually-hidden">{consoleStatus}</div>
    </>
  );
}
