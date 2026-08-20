import {
  lazy,
  memo,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";

import { StatusIndicator } from "./components/ui";
import {
  ApiError,
  HealthResponse,
  LicenseStatus,
  ModelProject,
  RuntimeConfig,
  RuntimeState,
  RuntimeStatusFrame,
  RuntimeStatusTopic,
  decodeRuntimeStatusMessage,
  getHealth,
  getApiErrorCode,
  getLicenseStatus,
  getModelProjects,
  getRuntimeConfig,
  getRuntimeState,
  statusWebSocketUrl,
} from "./api";
import { ToastHost } from "./components/ToastHost";
import { AuthGate } from "./features/auth/AuthGate";
import { reportError } from "./lib/toast";
import {
  clearWebSocketFailure,
  reportWebSocketFailure,
  reportWebSocketRecovered
} from "./lib/errorGuards";
import { LicenseGate } from "./features/license/LicenseView";
import {
  describeLicenseConnectionIssue,
  type LicenseConnectionIssue
} from "./features/license/connectionIssue";
import { formatTime, getErrorMessage } from "./features/shared/format";
import {
  runtimeDeliveryLabel,
  runtimeDeliveryTone,
  type RuntimeDeliveryStatus
} from "./features/shared/runtimeDelivery";
import { useStableSemanticValue } from "./features/shared/useStableSemanticValue";
import {
  EMPTY_RUNTIME_SNAPSHOT_CURSOR,
  gateRuntimeSnapshot,
  type RuntimeSnapshotCursor,
} from "./features/runtime/runtimeSnapshotGate";
import { SafetyOperationProvider, useSafetyOperation } from "./features/runtime/SafetyOperationContext";

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
const LICENSE_STARTUP_RETRY_DELAYS_MS = [250, 750] as const;

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

function isTransientLicenseTransportError(error: unknown): boolean {
  if (error instanceof ApiError) {
    return getApiErrorCode(error) === "" && (error.status === 408 || error.status >= 500);
  }
  return error instanceof TypeError;
}

async function getLicenseStatusWithStartupRetry(): Promise<LicenseStatus> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await getLicenseStatus();
    } catch (error) {
      const delay = LICENSE_STARTUP_RETRY_DELAYS_MS[attempt];
      if (delay === undefined || !isTransientLicenseTransportError(error)) {
        throw error;
      }
      await new Promise<void>((resolve) => window.setTimeout(resolve, delay));
    }
  }
}

function statusTopicFromPage(): RuntimeStatusTopic {
  const page = new URLSearchParams(window.location.search).get("page");
  if (page === "infer" || page === "control" || page === "latency" || page === "capture") {
    return page;
  }
  return "summary";
}

function isJsonRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function jsonValuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => jsonValuesEqual(value, right[index]));
  }
  if (!isJsonRecord(left) || !isJsonRecord(right)) {
    return false;
  }
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  for (const key of leftKeys) {
    if (!Object.prototype.hasOwnProperty.call(right, key)) {
      return false;
    }
    if (!jsonValuesEqual(left[key], right[key])) {
      return false;
    }
  }
  return true;
}

// Heartbeat sequence/timestamp changes are transport metadata, not UI state.
// Compare every other field so idle status frames do not repaint the Studio.
function runtimePayloadEqual(left: RuntimeState, right: RuntimeState): boolean {
  return left.semantic.daemon_instance_id === right.semantic.daemon_instance_id
    && left.semantic.phase === right.semantic.phase
    && left.semantic.perception_phase === right.semantic.perception_phase
    && left.semantic.epoch === right.semantic.epoch
    && left.running === right.running
    && left.source === right.source
    && left.model_catalog_error === right.model_catalog_error
    && jsonValuesEqual(left.active_model, right.active_model)
    && jsonValuesEqual(left.executor, right.executor)
    && jsonValuesEqual(left.capture, right.capture)
    && jsonValuesEqual(left.statistics, right.statistics)
    && jsonValuesEqual(left.inference, right.inference)
    && jsonValuesEqual(left.config, right.config)
    && jsonValuesEqual(left.pipeline, right.pipeline)
    && jsonValuesEqual(left.vision, right.vision)
    && jsonValuesEqual(left.fatal_error, right.fatal_error);
}

const visualSystemMode = new URLSearchParams(window.location.search).get("visual-system") === "1";

// Throttles the "更新于 HH:MM:SS" text node to ≤1Hz. The parent re-renders
// on observable WebSocket snapshot changes; without throttling the timestamp paints
// N times per second even when the formatted string is unchanged. The
// component only commits a new `displayed` value once per 1000ms.
const LastUpdatedText = memo(function LastUpdatedText({ date }: { date: Date | null }) {
  const [displayed, setDisplayed] = useState<Date | null>(date);
  const lastShownRef = useRef<number>(date?.getTime() ?? 0);
  useEffect(() => {
    if (!date) {
      if (displayed !== null) {
        setDisplayed(null);
      }
      return;
    }
    const now = date.getTime();
    const elapsed = now - lastShownRef.current;
    if (elapsed >= 1000) {
      lastShownRef.current = now;
      setDisplayed(date);
      return;
    }
    const remaining = 1000 - elapsed;
    const timer = window.setTimeout(() => {
      lastShownRef.current = date.getTime();
      setDisplayed(date);
    }, remaining);
    return () => window.clearTimeout(timer);
  }, [date, displayed]);
  return <span className="last-updated">更新于 {formatTime(displayed)}</span>;
});

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
  return (
    <SafetyOperationProvider>
      <AuthGate>
        {visualSystemMode
          ? <Suspense fallback={<RouteLoadingShell />}><VisualSystemView /></Suspense>
          : <StudioApp />}
      </AuthGate>
    </SafetyOperationProvider>
  );
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
  const runtimeRequestGenerationRef = useRef(0);
  const runtimeSnapshotCursorRef = useRef<RuntimeSnapshotCursor>(EMPTY_RUNTIME_SNAPSHOT_CURSOR);
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
  const [licenseIssue, setLicenseIssue] = useState<LicenseConnectionIssue | null>(null);
  const { reconcileEmergencyStop } = useSafetyOperation();

  const applyRuntimeHeartbeat = useCallback((daemonInstanceId: string, snapshotSequence: number) => {
    const result = gateRuntimeSnapshot(runtimeSnapshotCursorRef.current, {
      daemonInstanceId,
      snapshotSequence,
      kind: "heartbeat",
      full: false,
    });
    runtimeSnapshotCursorRef.current = result.cursor;
    return result.reason === "heartbeat_observed";
  }, []);

  const applyRuntimeState = useCallback((
    runtime: RuntimeState,
    options: { full?: boolean; requestGeneration?: number; source?: "stream" } = {},
  ) => {
    const currentDaemonId = runtimeSnapshotCursorRef.current.daemonInstanceId;
    const fullSnapshot = options.full ?? true;
    const switchingDaemon = currentDaemonId !== null
      && currentDaemonId !== runtime.semantic.daemon_instance_id;
    const mayResetDaemonIdentity = options.source === "stream"
      || options.requestGeneration !== undefined;
    if (
      options.source === "stream"
      && fullSnapshot
      && switchingDaemon
    ) {
      // A full stream snapshot is the first authoritative evidence of a daemon
      // restart. Invalidate every older REST callback before switching identity.
      runtimeRequestGenerationRef.current += 1;
    }
    const result = gateRuntimeSnapshot(runtimeSnapshotCursorRef.current, {
      daemonInstanceId: runtime.semantic.daemon_instance_id,
      snapshotSequence: runtime.semantic.snapshot_sequence,
      kind: "snapshot",
      // Unversioned action callbacks may update the current daemon, but they
      // cannot resurrect an older daemon after a stream/REST identity switch.
      full: switchingDaemon && !mayResetDaemonIdentity ? false : fullSnapshot,
      requestGeneration: options.requestGeneration,
      currentRequestGeneration: runtimeRequestGenerationRef.current,
    });
    runtimeSnapshotCursorRef.current = result.cursor;
    if (!result.accept) {
      return false;
    }
    const receivedAt = Date.now();
    setState((current) => {
      if (current.runtime !== null && runtimePayloadEqual(current.runtime, runtime)) {
        return current;
      }
      return {
        ...current,
        errors: withoutError(current.errors, "runtime"),
        runtime,
        lastUpdated: new Date(receivedAt)
      };
    });
    return true;
  }, []);

  const applyRuntimeFrame = useCallback((frame: RuntimeStatusFrame) => {
    return applyRuntimeState(frame.state, { full: frame.full, source: "stream" });
  }, [applyRuntimeState]);

  useEffect(() => {
    if (state.runtime) {
      reconcileEmergencyStop(state.runtime);
    }
  }, [reconcileEmergencyStop, state.runtime]);

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
    setLicenseIssue(null);
    try {
      const status = await getLicenseStatusWithStartupRetry();
      if (requestSeq !== licenseRequestSeqRef.current) {
        return status;
      }
      setLicense(status);
      return status;
    } catch (err) {
      if (requestSeq !== licenseRequestSeqRef.current) {
        return null;
      }
      setLicense(null);
      const issue = describeLicenseConnectionIssue(err);
      setLicenseIssue(issue);
      reportError(err, {
        source: "license",
        title: issue.title,
        publicDetail: `${issue.description} ${issue.recovery}`,
        exposeStatus: false
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
    if (!status.valid) {
      loadRequestSeqRef.current += 1;
      setState(initialState);
      setRealtimeStatus("disconnected");
      runtimeRequestGenerationRef.current += 1;
      runtimeSnapshotCursorRef.current = EMPTY_RUNTIME_SNAPSHOT_CURSOR;
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
    const runtimeRequestGeneration = runtimeRequestGenerationRef.current + 1;
    runtimeRequestGenerationRef.current = runtimeRequestGeneration;
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
          if (requestSeq !== loadRequestSeqRef.current) return;
          applyRuntimeState(runtime, { requestGeneration: runtimeRequestGeneration });
        })
        .catch((error) => {
          if (requestSeq !== loadRequestSeqRef.current) return;
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
    const runtimeRequestGeneration = runtimeRequestGenerationRef.current + 1;
    runtimeRequestGenerationRef.current = runtimeRequestGeneration;
    const abortController = new AbortController();
    runtimeFallbackAbortRef.current = abortController;
    try {
      const runtime = await getRuntimeState(abortController.signal);
      applyRuntimeState(runtime, { requestGeneration: runtimeRequestGeneration });
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
      let nextSocket: WebSocket;
      try {
        nextSocket = new WebSocket(statusWebSocketUrl(statusTopic));
      } catch (error) {
        failureReported = true;
        setRealtimeStatus((current) => current === "fallback" ? current : "disconnected");
        if (reportWebSocketFailure(error, "status")) {
          void refreshHealth();
        }
        scheduleReconnect();
        return;
      }
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
        if (event.code === 4401) {
          failureReported = true;
          void loadLicense().finally(scheduleReconnect);
          return;
        }
        if (event.code === 4403) {
          failureReported = true;
          window.dispatchEvent(new CustomEvent("novasight:auth-required"));
          return;
        }
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
          if (typeof event.data !== "string") {
            throw new Error("状态流必须发送 UTF-8 JSON 文本帧");
          }
          const payload = decodeRuntimeStatusMessage(JSON.parse(event.data));
          let trustedMessage = false;
          if ("kind" in payload && payload.kind === "runtime_heartbeat") {
            trustedMessage = applyRuntimeHeartbeat(payload.daemon_instance_id, payload.snapshot_sequence);
          } else if ("kind" in payload) {
            trustedMessage = applyRuntimeFrame(payload);
          } else {
            trustedMessage = applyRuntimeState(payload, { full: true, source: "stream" });
          }
          if (trustedMessage) {
            latestMessageAt = Date.now();
            reconnectAttempt = 0;
            setRealtimeStatus("connected");
            reportWebSocketRecovered("status");
          }
        } catch (error) {
          failureReported = true;
          setRealtimeStatus((current) => current === "fallback" ? current : "disconnected");
          if (reportWebSocketFailure(error, "status")) {
            void refreshHealth();
          }
          void refreshRuntime();
          nextSocket.close(4002, "invalid status contract");
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
  }, [applyRuntimeFrame, applyRuntimeHeartbeat, applyRuntimeState, license?.valid, loadLicense, networkOnline, pageVisible, refreshHealth, refreshRuntime, statusTopic]);

  const realtimeConnected = realtimeStatus === "connected";
  const stableRealtimeStatus = useStableSemanticValue(realtimeStatus, realtimeStatus, 280);
  const displayedRealtimeStatus = realtimeStatus === "offline" || realtimeStatus === "paused"
    ? realtimeStatus
    : stableRealtimeStatus;

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
        issue={licenseIssue}
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
      <StatusIndicator tone={runtimeDeliveryTone(displayedRealtimeStatus)}>
        {runtimeDeliveryLabel(displayedRealtimeStatus)}
      </StatusIndicator>
      <LastUpdatedText date={state.lastUpdated} />
    </>
  );

  return (
    <>
      <Suspense fallback={<RouteLoadingShell />}>
        <StudioConsoleView
          license={license}
          health={state.health}
          runtime={state.runtime}
          runtimeConfig={state.config}
          projects={state.projects}
          errors={state.errors}
          lastUpdated={state.lastUpdated}
          realtimeStatus={displayedRealtimeStatus}
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
