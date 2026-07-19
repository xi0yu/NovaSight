import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";

import { StatusIndicator } from "./components/ui";
import {
  HealthResponse,
  LicenseStatus,
  ModelProject,
  RuntimeConfig,
  RuntimeState,
  RuntimeStatusFrame,
  RuntimeStatusTopic,
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

function statusTopicFromPage(): RuntimeStatusTopic {
  const page = new URLSearchParams(window.location.search).get("page");
  if (page === "infer" || page === "control" || page === "latency" || page === "capture") {
    return page;
  }
  return "summary";
}

function isRuntimeStatusFrame(value: unknown): value is RuntimeStatusFrame {
  if (typeof value !== "object" || value === null) return false;
  const frame = value as Partial<RuntimeStatusFrame>;
  return frame.kind === "runtime_snapshot" && typeof frame.full === "boolean" &&
    typeof frame.state === "object" && frame.state !== null;
}

function mergeRuntimePatch(current: RuntimeState, patch: Partial<RuntimeState>): RuntimeState {
  return {
    ...current,
    ...patch,
    executor: patch.executor ?? current.executor,
    capture: patch.capture ? { ...current.capture, ...patch.capture } : current.capture,
    statistics: patch.statistics
      ? { ...(current.statistics ?? {}), ...patch.statistics }
      : current.statistics,
    inference: patch.inference
      ? { ...current.inference, ...patch.inference }
      : current.inference,
    config: patch.config ? { ...current.config, ...patch.config } : current.config,
    pipeline: patch.pipeline
      ? {
          ...current.pipeline,
          ...patch.pipeline,
          ...(typeof patch.pipeline.deepstream === "object" && patch.pipeline.deepstream !== null
            ? {
                deepstream: {
                  ...((typeof current.pipeline.deepstream === "object" && current.pipeline.deepstream !== null)
                    ? current.pipeline.deepstream as Record<string, unknown>
                    : {}),
                  ...patch.pipeline.deepstream as Record<string, unknown>
                }
              }
            : {})
        }
      : current.pipeline,
    power_saving: patch.power_saving
      ? { ...(current.power_saving ?? {}), ...patch.power_saving }
      : current.power_saving,
    vision: patch.vision ?? current.vision
  };
}

const visualSystemMode = new URLSearchParams(window.location.search).get("visual-system") === "1";

const StudioConsoleView = lazy(() =>
  import("./features/studio/StudioConsoleView").then((module) => ({
    default: module.StudioConsoleView
  }))
);
const VisualSystemView = lazy(() =>
  import("./features/visual-system/VisualSystemView").then((module) => ({
    default: module.VisualSystemView
  }))
);
const MotionProfileStudio = lazy(() =>
  import("./features/motion/MotionProfileStudio").then((module) => ({
    default: module.MotionProfileStudio
  }))
);

function RouteLoadingShell() {
  return (
    <main className="route-loading-shell" role="status">
      <span className="route-loading-mark" aria-hidden="true" />
      <strong>NovaSight Studio</strong>
      <small>正在载入控制台…</small>
    </main>
  );
}

export default function App() {
  const page = new URLSearchParams(window.location.search).get("page");
  if (page === "motion-profile") {
    return <Suspense fallback={<RouteLoadingShell />}><MotionProfileStudio /></Suspense>;
  }
  if (visualSystemMode) {
    return <Suspense fallback={<RouteLoadingShell />}><VisualSystemView /></Suspense>;
  }
  return <StudioApp />;
}

function StudioApp() {
  const [state, setState] = useState<LoadState>(initialState);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [realtimeStatus, setRealtimeStatus] = useState<RuntimeDeliveryStatus>("disconnected");
  const [statusTopic, setStatusTopic] = useState<RuntimeStatusTopic>(() => statusTopicFromPage());
  const [pageVisible, setPageVisible] = useState(() => document.visibilityState !== "hidden");
  const [networkOnline, setNetworkOnline] = useState(() => navigator.onLine);
  const loadRequestSeqRef = useRef(0);
  const licenseRequestSeqRef = useRef(0);
  const runtimeRevisionRef = useRef(0);
  const initialLoadStartedRef = useRef(false);
  const backgroundLoadInFlightRef = useRef(false);
  const healthLoadInFlightRef = useRef(false);
  const foregroundLoadInFlightRef = useRef(false);
  const projectsLoadedRef = useRef(false);
  const projectsRef = useRef<ModelProject[]>([]);
  const projectsLoadInFlightRef = useRef<Promise<ModelProject[]> | null>(null);
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

  const applyRuntimeFrame = useCallback((frame: RuntimeStatusFrame) => {
    if (frame.full) {
      applyRuntimeState(frame.state as RuntimeState);
      return;
    }
    const receivedAt = Date.now();
    setState((current) => {
      if (current.runtime === null) {
        // The initial full REST request owns construction of RuntimeState.
        // A partial frame may arrive first on a fast local WebSocket.
        return current;
      }
      return {
        ...current,
        errors: withoutError(current.errors, "runtime"),
        runtime: mergeRuntimePatch(current.runtime, frame.state),
        lastUpdated: new Date(receivedAt)
      };
    });
  }, [applyRuntimeState]);

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
      projectsLoadedRef.current = false;
      projectsRef.current = [];
      projectsLoadInFlightRef.current = null;
      runtimeFallbackAbortRef.current?.abort();
      healthRefreshAbortRef.current?.abort();
      runtimeFallbackAbortRef.current = null;
      healthRefreshAbortRef.current = null;
      clearWebSocketFailure("status");
    }
  }, []);

  const loadProjects = useCallback(async (force = false): Promise<ModelProject[]> => {
    if (!force && projectsLoadedRef.current) {
      return projectsRef.current;
    }
    if (projectsLoadInFlightRef.current) {
      return projectsLoadInFlightRef.current;
    }
    const request = getModelProjects()
      .then((projects) => {
        projectsLoadedRef.current = true;
        projectsRef.current = projects;
        setState((current) => ({
          ...current,
          errors: withoutError(current.errors, "projects"),
          projects
        }));
        return projects;
      })
      .catch((error) => {
        setState((current) => ({
          ...current,
          errors: { ...current.errors, projects: getErrorMessage(error) }
        }));
        reportError(error, { source: "projects", title: "模型项目读取失败" });
        throw error;
      })
      .finally(() => {
        projectsLoadInFlightRef.current = null;
      });
    projectsLoadInFlightRef.current = request;
    return request;
  }, []);

  const load = useCallback(async () => {
    const requestSeq = loadRequestSeqRef.current + 1;
    const runtimeRevisionAtRequest = runtimeRevisionRef.current;
    loadRequestSeqRef.current = requestSeq;
    foregroundLoadInFlightRef.current = true;
    setState((current) => ({ ...current, loading: true, errors: {} }));
    try {
      const healthRequest = getHealth()
        .then((health) => {
          if (requestSeq !== loadRequestSeqRef.current) return;
          setState((current) => ({
            ...current,
            errors: withoutError(current.errors, "health"),
            health
          }));
        })
        .catch((error) => {
          if (requestSeq !== loadRequestSeqRef.current) return;
          setState((current) => ({
            ...current,
            errors: { ...current.errors, health: getErrorMessage(error) },
            health: null
          }));
          reportError(error, { source: "health", title: "后端健康检查失败" });
        });
      const runtimeRequest = getRuntimeState()
        .then((runtime) => {
          if (
            requestSeq !== loadRequestSeqRef.current ||
            runtimeRevisionRef.current !== runtimeRevisionAtRequest
          ) return;
          applyRuntimeState(runtime);
        })
        .catch((error) => {
          if (
            requestSeq !== loadRequestSeqRef.current ||
            runtimeRevisionRef.current !== runtimeRevisionAtRequest
          ) return;
          setState((current) => ({
            ...current,
            errors: { ...current.errors, runtime: getErrorMessage(error) }
          }));
          reportError(error, { source: "runtime", title: "运行态失败" });
        });
      const configRequest = getRuntimeConfig()
        .then((config) => {
          if (requestSeq !== loadRequestSeqRef.current) return;
          applyRuntimeConfig(config);
        })
        .catch((error) => {
          if (requestSeq !== loadRequestSeqRef.current) return;
          setState((current) => ({
            ...current,
            errors: { ...current.errors, config: getErrorMessage(error) }
          }));
          reportError(error, { source: "config", title: "配置读取失败" });
        });
      const projectsRequest = projectsLoadedRef.current
        ? loadProjects(true).catch(() => undefined)
        : Promise.resolve();
      await Promise.allSettled([
        healthRequest,
        runtimeRequest,
        configRequest,
        projectsRequest
      ]);
    } finally {
      if (requestSeq === loadRequestSeqRef.current) {
        setState((current) => ({ ...current, loading: false }));
      }
      foregroundLoadInFlightRef.current = false;
    }
  }, [applyRuntimeConfig, applyRuntimeState, loadProjects]);

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
      const nextSocket = new WebSocket(statusWebSocketUrl(statusTopic));
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
          const payload = JSON.parse(String(event.data)) as RuntimeState | RuntimeStatusFrame;
          const receivedAt = Date.now();
          latestMessageAt = receivedAt;
          reconnectAttempt = 0;
          if (isRuntimeStatusFrame(payload)) {
            applyRuntimeFrame(payload);
          } else {
            applyRuntimeState(payload);
          }
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
  }, [applyRuntimeFrame, applyRuntimeState, license?.valid, networkOnline, pageVisible, refreshHealth, statusTopic]);

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
      <Suspense fallback={<RouteLoadingShell />}>
        <StudioConsoleView
          health={state.health}
          runtime={state.runtime}
          runtimeConfig={state.config}
          projects={state.projects}
          errors={state.errors}
          lastUpdated={state.lastUpdated}
          realtimeStatus={realtimeStatus}
          onEnsureProjects={loadProjects}
          onRefresh={load}
          onRuntimeConfigChange={applyRuntimeConfig}
          onRuntimeStateChange={applyRuntimeState}
          onStatusTopicChange={setStatusTopic}
        />
      </Suspense>
      <ToastHost />
      <div className="visually-hidden">{consoleStatus}</div>
    </>
  );
}
