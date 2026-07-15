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
import { reportWebSocketFailure } from "./lib/errorGuards";
import { LicenseGate } from "./features/license/LicenseView";
import { LICENSE_CACHE_KEY } from "./features/license/storage";
import { StudioConsoleView } from "./features/studio/StudioConsoleView";
import { VisualSystemView } from "./features/visual-system/VisualSystemView";
import { formatTime, getErrorMessage } from "./features/shared/format";

type ErrorKey = "health" | "runtime" | "config" | "projects" | "capture";
type RealtimeStatus = "connecting" | "connected" | "stale" | "disconnected";
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

function realtimeTone(status: RealtimeStatus): "good" | "warn" | "bad" | "idle" {
  switch (status) {
    case "connected":
      return "good";
    case "stale":
    case "connecting":
      return "warn";
    case "disconnected":
      return "bad";
    default:
      return "idle";
  }
}

function realtimeLabel(status: RealtimeStatus): string {
  switch (status) {
    case "connected":
      return "实时 已连接";
    case "stale":
      return "实时 数据陈旧";
    case "connecting":
      return "实时 连接中";
    case "disconnected":
    default:
      return "实时 已断开";
  }
}

const visualSystemMode = new URLSearchParams(window.location.search).get("visual-system") === "1";

export default function App() {
  return visualSystemMode ? <VisualSystemView /> : <StudioApp />;
}

function StudioApp() {
  const [state, setState] = useState<LoadState>(initialState);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("disconnected");
  const [lastWsMessageAt, setLastWsMessageAt] = useState<number | null>(null);
  const loadRequestSeqRef = useRef(0);
  const licenseRequestSeqRef = useRef(0);
  const runtimeStateReceivedAtRef = useRef(0);
  const backgroundLoadInFlightRef = useRef(false);
  const foregroundLoadInFlightRef = useRef(false);
  const [licenseLoading, setLicenseLoading] = useState(
    localStorage.getItem(LICENSE_CACHE_KEY) === "1"
  );
  const [licenseError, setLicenseError] = useState<string | undefined>();

  const applyRuntimeState = useCallback((runtime: RuntimeState) => {
    const receivedAt = Date.now();
    runtimeStateReceivedAtRef.current = receivedAt;
    setState((current) => ({
      ...current,
      runtime,
      lastUpdated: new Date(receivedAt)
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
      setLastWsMessageAt(null);
      runtimeStateReceivedAtRef.current = 0;
      backgroundLoadInFlightRef.current = false;
      foregroundLoadInFlightRef.current = false;
    }
  }, []);

  const load = useCallback(async (options?: { background?: boolean }) => {
    const background = options?.background === true;
    if (background && foregroundLoadInFlightRef.current) {
      return;
    }
    const requestSeq = loadRequestSeqRef.current + 1;
    loadRequestSeqRef.current = requestSeq;
    if (!background) {
      foregroundLoadInFlightRef.current = true;
      setState((current) => ({ ...current, loading: true, errors: {} }));
    }
    try {
      const [health, runtime, config, projects] = await Promise.allSettled([
        getHealth(),
        getRuntimeState(),
        getRuntimeConfig(),
        getModelProjects()
      ]);
      const responseReceivedAt = Date.now();
      const sourceMap: Array<[PromiseSettledResult<unknown>, string]> = [
        [health, "health"],
        [runtime, "runtime"],
        [config, "config"],
        [projects, "projects"]
      ];
      const seenReason = new Set<unknown>();
      for (const [result, source] of sourceMap) {
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
        if (runtime.status === "rejected") {
          errors.runtime = getErrorMessage(runtime.reason);
        }
        if (config.status === "rejected") {
          errors.config = getErrorMessage(config.reason);
        }
        if (projects.status === "rejected") {
          errors.projects = getErrorMessage(projects.reason);
        }
        const shouldApplyRuntime =
          runtime.status === "fulfilled" && runtimeStateReceivedAtRef.current <= responseReceivedAt;
        if (shouldApplyRuntime) {
          runtimeStateReceivedAtRef.current = responseReceivedAt;
        }
        return {
          loading: false,
          errors,
          health: health.status === "fulfilled" ? health.value : current.health,
          runtime: shouldApplyRuntime ? runtime.value : current.runtime,
          config: config.status === "fulfilled" ? config.value : current.config,
          projects: projects.status === "fulfilled" ? projects.value : current.projects,
          lastUpdated: new Date()
        };
      });
    } finally {
      if (!background) {
        foregroundLoadInFlightRef.current = false;
      }
    }
  }, []);

  useEffect(() => {
    void loadLicense();
  }, [loadLicense]);

  useEffect(() => {
    if (license?.valid) {
      void load();
    }
  }, [license?.valid, load]);

  useEffect(() => {
    if (!license?.valid) {
      setRealtimeStatus("disconnected");
      setLastWsMessageAt(null);
      return undefined;
    }

    setRealtimeStatus("connecting");

    let active = true;
    const socket = new WebSocket(statusWebSocketUrl());
    socket.onerror = (event) => {
      if (active) {
        setRealtimeStatus("disconnected");
        reportWebSocketFailure(event, "status");
      }
    };
    socket.onclose = (event) => {
      if (active) {
        setRealtimeStatus("disconnected");
        if (event.code !== 1000 && event.code !== 1001) {
          reportWebSocketFailure(event.reason || `code=${event.code}`, "status");
        }
      }
    };
    socket.onmessage = (event) => {
      if (!active) {
        return;
      }
      try {
        const runtime = JSON.parse(String(event.data)) as RuntimeState;
        const receivedAt = Date.now();
        applyRuntimeState(runtime);
        setLastWsMessageAt(receivedAt);
        setRealtimeStatus("connected");
      } catch {
        // Ignore malformed status frames; REST refresh still provides recovery.
      }
    };
    return () => {
      active = false;
      socket.close();
    };
  }, [applyRuntimeState, license?.valid]);

  useEffect(() => {
    if (realtimeStatus !== "connected" || lastWsMessageAt === null) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      if (Date.now() - lastWsMessageAt > 2000) {
        setRealtimeStatus("stale");
      }
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, [lastWsMessageAt, realtimeStatus]);

  useEffect(() => {
    if (!license?.valid || realtimeStatus === "connected") {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      if (backgroundLoadInFlightRef.current || foregroundLoadInFlightRef.current) {
        return;
      }
      backgroundLoadInFlightRef.current = true;
      void load({ background: true }).finally(() => {
        backgroundLoadInFlightRef.current = false;
      });
    }, 3000);

    return () => window.clearInterval(intervalId);
  }, [license?.valid, load, realtimeStatus]);

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
      <StatusIndicator tone={realtimeTone(realtimeStatus)}>
        {realtimeLabel(realtimeStatus)}
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
        onRuntimeStateChange={applyRuntimeState}
      />
      <ToastHost />
      <div className="visually-hidden">{consoleStatus}</div>
    </>
  );
}
