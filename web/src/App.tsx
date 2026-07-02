import { useCallback, useEffect, useState } from "react";

import { StudioShell } from "./app/StudioShell";
import { type StudioViewId } from "./app/navigation";
import { PermissionGuard } from "./components/studio";
import { LoadingSkeleton, StatusIndicator } from "./components/ui";
import {
  HealthResponse,
  LicenseStatus,
  ModelProject,
  PluginInfo,
  RuntimeState,
  getHealth,
  getLicenseStatus,
  getModelProjects,
  getPlugins,
  getRuntimeState,
  startInferenceControl,
  statusWebSocketUrl,
  stopInferenceControl,
} from "./api";
import { ConfigView } from "./features/config/ConfigView";
import { DashboardView } from "./features/dashboard/DashboardView";
import { DevicesView } from "./features/devices/DevicesView";
import { LicenseGate, LicenseView } from "./features/license/LicenseView";
import { LICENSE_CACHE_KEY } from "./features/license/storage";
import { ModelsView } from "./features/models/ModelsView";
import { PluginsView } from "./features/plugins/PluginsView";
import { formatTime, getErrorMessage } from "./features/shared/format";

type ErrorKey = "health" | "runtime" | "plugins" | "projects" | "capture";
type RealtimeStatus = "connecting" | "connected" | "stale" | "disconnected";
type GuardedViewId = Exclude<StudioViewId, "license">;

type LoadState = {
  loading: boolean;
  errors: Partial<Record<ErrorKey, string>>;
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  plugins: PluginInfo[];
  projects: ModelProject[];
  lastUpdated: Date | null;
};

const initialState: LoadState = {
  loading: true,
  errors: {},
  health: null,
  runtime: null,
  plugins: [],
  projects: [],
  lastUpdated: null
};

const viewFeatureMap: Partial<Record<GuardedViewId, LicenseStatus["features"][number]>> = {
  dashboard: "runtime",
  devices: "capture",
  models: "models",
  config: "config_read",
  plugins: "plugins"
};

function withoutError(
  errors: LoadState["errors"],
  key: ErrorKey
): LoadState["errors"] {
  const next = { ...errors };
  delete next[key];
  return next;
}

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

function canAccessView(view: StudioViewId, license: LicenseStatus | null): boolean {
  if (view === "license") {
    return true;
  }

  const requiredFeature = viewFeatureMap[view];
  if (!requiredFeature) {
    return true;
  }

  return license?.features.includes(requiredFeature) ?? false;
}

function getFallbackView(license: LicenseStatus | null): StudioViewId {
  const fallbackOrder: StudioViewId[] = [
    "dashboard",
    "devices",
    "models",
    "config",
    "plugins",
    "license"
  ];

  return fallbackOrder.find((view) => canAccessView(view, license)) ?? "license";
}

export default function App() {
  const [activeView, setActiveView] = useState<StudioViewId>("license");
  const [state, setState] = useState<LoadState>(initialState);
  const [runtimeCommandBusy, setRuntimeCommandBusy] = useState(false);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("disconnected");
  const [lastWsMessageAt, setLastWsMessageAt] = useState<number | null>(null);
  const [licenseLoading, setLicenseLoading] = useState(
    localStorage.getItem(LICENSE_CACHE_KEY) === "1"
  );
  const [licenseError, setLicenseError] = useState<string | undefined>();

  const loadLicense = useCallback(async () => {
    setLicenseLoading(true);
    setLicenseError(undefined);
    try {
      const status = await getLicenseStatus();
      setLicense(status);
      if (status.valid) {
        localStorage.setItem(LICENSE_CACHE_KEY, "1");
      } else {
        localStorage.removeItem(LICENSE_CACHE_KEY);
      }
      return status;
    } catch (err) {
      setLicenseError(getErrorMessage(err));
      localStorage.removeItem(LICENSE_CACHE_KEY);
      return null;
    } finally {
      setLicenseLoading(false);
    }
  }, []);

  const handleLicenseChange = useCallback((status: LicenseStatus) => {
    setLicense(status);
    if (status.valid) {
      localStorage.setItem(LICENSE_CACHE_KEY, "1");
    } else {
      localStorage.removeItem(LICENSE_CACHE_KEY);
      setState(initialState);
      setRealtimeStatus("disconnected");
      setLastWsMessageAt(null);
    }
  }, []);

  const load = useCallback(async () => {
    setState((current) => ({ ...current, loading: true, errors: {} }));
    const [health, runtime, plugins, projects] = await Promise.allSettled([
      getHealth(),
      getRuntimeState(),
      getPlugins(),
      getModelProjects()
    ]);
    setState((current) => {
      const errors: LoadState["errors"] = {};
      if (health.status === "rejected") {
        errors.health = getErrorMessage(health.reason);
      }
      if (runtime.status === "rejected") {
        errors.runtime = getErrorMessage(runtime.reason);
      }
      if (plugins.status === "rejected") {
        errors.plugins = getErrorMessage(plugins.reason);
      }
      if (projects.status === "rejected") {
        errors.projects = getErrorMessage(projects.reason);
      }
      return {
        loading: false,
        errors,
        health: health.status === "fulfilled" ? health.value : current.health,
        runtime: runtime.status === "fulfilled" ? runtime.value : current.runtime,
        plugins: plugins.status === "fulfilled" ? plugins.value : current.plugins,
        projects: projects.status === "fulfilled" ? projects.value : current.projects,
        lastUpdated: new Date()
      };
    });
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
      return;
    }

    if (canAccessView(activeView, license)) {
      return;
    }

    setActiveView(getFallbackView(license));
  }, [activeView, license]);

  useEffect(() => {
    if (!license?.valid) {
      setRealtimeStatus("disconnected");
      setLastWsMessageAt(null);
      return undefined;
    }

    setRealtimeStatus("connecting");

    let active = true;
    const socket = new WebSocket(statusWebSocketUrl());
    socket.onerror = () => {
      if (active) {
        setRealtimeStatus("disconnected");
      }
    };
    socket.onclose = () => {
      if (active) {
        setRealtimeStatus("disconnected");
      }
    };
    socket.onmessage = (event) => {
      try {
        const runtime = JSON.parse(String(event.data)) as RuntimeState;
        const receivedAt = Date.now();
        setState((current) => ({
          ...current,
          runtime,
          lastUpdated: new Date()
        }));
        if (active) {
          setLastWsMessageAt(receivedAt);
          setRealtimeStatus("connected");
        }
      } catch {
        // Ignore malformed status frames; REST refresh still provides recovery.
      }
    };
    return () => {
      active = false;
      socket.close();
    };
  }, [license?.valid]);

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

  const handleInferenceControlCommand = useCallback(
    async (action: "start" | "stop") => {
      setRuntimeCommandBusy(true);
      setState((current) => ({
        ...current,
        errors: withoutError(current.errors, "runtime")
      }));
      try {
        if (action === "start") {
          await startInferenceControl();
        } else {
          await stopInferenceControl();
        }
        await load();
      } catch (err) {
        setState((current) => ({
          ...current,
          errors: { ...current.errors, runtime: getErrorMessage(err) }
        }));
      } finally {
        setRuntimeCommandBusy(false);
      }
    },
    [load]
  );

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

  const activeModel = state.runtime?.active_model ?? null;
  const hasErrors = Object.keys(state.errors).length > 0;
  const shellCopy: Record<StudioViewId, { title: string; subtitle: string }> = {
    dashboard: {
      title: "Dashboard",
      subtitle: "Track runtime health, pipeline readiness, and the current control path."
    },
    devices: {
      title: "Devices",
      subtitle: "Configure capture profiles, inspect camera capability sets, and monitor preview."
    },
    models: {
      title: "Models",
      subtitle: "Review active deployments and the registered project inventory."
    },
    config: {
      title: "Config",
      subtitle: "Adjust runtime configuration and compare the live state with saved settings."
    },
    plugins: {
      title: "Plugins",
      subtitle: "Inspect loaded vision and control modules in the production chain."
    },
    license: {
      title: "License",
      subtitle: "Manage the local activation key and verify feature entitlement status."
    }
  };
  const shellStatus = (
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
    <StudioShell
      activeView={activeView}
      title={shellCopy[activeView].title}
      subtitle={shellCopy[activeView].subtitle}
      status={shellStatus}
      onNavigate={setActiveView}
    >
      {hasErrors ? (
        <div className="alert" role="alert">
          <strong>后端请求异常</strong>
          <span>{Object.values(state.errors).join(" / ")}</span>
          <button className="button compact-button" type="button" onClick={load}>
            重试
          </button>
        </div>
      ) : null}

      {state.loading && !state.runtime ? (
        <LoadingSkeleton />
      ) : null}

      <div className="view-stack">
        {activeView === "dashboard" ? (
          <PermissionGuard feature="runtime" license={license}>
            <DashboardView
              health={state.health}
              loading={state.loading}
              runtime={state.runtime}
              errors={state.errors}
              onRefresh={load}
              onInferenceControlCommand={(action) => void handleInferenceControlCommand(action)}
              runtimeCommandBusy={runtimeCommandBusy}
            />
          </PermissionGuard>
        ) : null}
        {activeView === "devices" ? (
          <PermissionGuard feature="capture" license={license}>
            <DevicesView
              runtime={state.runtime}
              error={state.errors.capture}
              onRuntimeRefresh={load}
              onInferenceControlCommand={(action) => void handleInferenceControlCommand(action)}
              runtimeCommandBusy={runtimeCommandBusy}
            />
          </PermissionGuard>
        ) : null}
        {activeView === "models" ? (
          <PermissionGuard feature="models" license={license}>
            <ModelsView
              projects={state.projects}
              activeModel={activeModel}
              error={state.errors.projects}
            />
          </PermissionGuard>
        ) : null}
        {activeView === "plugins" ? (
          <PermissionGuard feature="plugins" license={license}>
            <PluginsView plugins={state.plugins} error={state.errors.plugins} />
          </PermissionGuard>
        ) : null}
        {activeView === "config" ? (
          <PermissionGuard feature="config_read" license={license}>
            <ConfigView
              runtime={state.runtime}
              license={license}
              onRuntimeRefresh={load}
              onLicenseChange={handleLicenseChange}
            />
          </PermissionGuard>
        ) : null}
        {activeView === "license" ? (
          <LicenseView
            license={license}
            loading={licenseLoading}
            error={licenseError}
            onRefresh={() => void loadLicense()}
            onLicenseChange={handleLicenseChange}
          />
        ) : null}
      </div>
    </StudioShell>
  );
}
