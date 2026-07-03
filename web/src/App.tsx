import { useCallback, useEffect, useState } from "react";

import { StudioShell } from "./app/StudioShell";
import { type StudioViewId } from "./app/navigation";
import { PermissionGuard } from "./components/studio";
import { LoadingSkeleton, StatusIndicator } from "./components/ui";
import {
  HealthResponse,
  LicenseStatus,
  ModelProject,
  RuntimeState,
  getHealth,
  getLicenseStatus,
  getModelProjects,
  getRuntimeState,
  statusWebSocketUrl,
} from "./api";
import { ConfigView } from "./features/config/ConfigView";
import { DashboardView } from "./features/dashboard/DashboardView";
import { DevicesView } from "./features/devices/DevicesView";
import { LicenseGate, LicenseView } from "./features/license/LicenseView";
import { LICENSE_CACHE_KEY } from "./features/license/storage";
import { ModelsView } from "./features/models/ModelsView";
import { formatTime, getErrorMessage } from "./features/shared/format";

type ErrorKey = "health" | "runtime" | "projects" | "capture";
type RealtimeStatus = "connecting" | "connected" | "stale" | "disconnected";
type GuardedViewId = Exclude<StudioViewId, "license">;
type DeviceSettingsSection = "capture" | "inference" | "algorithm";

type LoadState = {
  loading: boolean;
  errors: Partial<Record<ErrorKey, string>>;
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  projects: ModelProject[];
  lastUpdated: Date | null;
};

const initialState: LoadState = {
  loading: true,
  errors: {},
  health: null,
  runtime: null,
  projects: [],
  lastUpdated: null
};

const viewFeatureMap: Partial<Record<GuardedViewId, LicenseStatus["features"][number]>> = {
  devices: "capture",
  models: "models",
  config: "config_read"
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
    "license"
  ];

  return fallbackOrder.find((view) => canAccessView(view, license)) ?? "license";
}

export default function App() {
  const [activeView, setActiveView] = useState<StudioViewId>("dashboard");
  const [state, setState] = useState<LoadState>(initialState);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("disconnected");
  const [lastWsMessageAt, setLastWsMessageAt] = useState<number | null>(null);
  const [deviceSettingsSection, setDeviceSettingsSection] = useState<DeviceSettingsSection>("capture");
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
    const [health, runtime, projects] = await Promise.allSettled([
      getHealth(),
      getRuntimeState(),
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
      if (projects.status === "rejected") {
        errors.projects = getErrorMessage(projects.reason);
      }
      return {
        loading: false,
        errors,
        health: health.status === "fulfilled" ? health.value : current.health,
        runtime: runtime.status === "fulfilled" ? runtime.value : current.runtime,
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
      title: "性能指挥台",
      subtitle: "采集吞吐、链路延迟、推理状态和控制路径。"
    },
    devices: {
      title: "基础设置",
      subtitle: "采集设置、推理设置和算法参数统一管理。"
    },
    models: {
      title: "模型仓库",
      subtitle: "面向使用场景选择模型，而不是管理开发产物。"
    },
    config: {
      title: "运行配置",
      subtitle: "运行参数、保存差异和硬件配置。"
    },
    license: {
      title: "授权许可",
      subtitle: "本机授权、功能权益和到期状态。"
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
          <DashboardView
            health={state.health}
            loading={state.loading}
            runtime={state.runtime}
            errors={state.errors}
            onRefresh={load}
          />
        ) : null}
        {activeView === "devices" ? (
          <PermissionGuard feature="capture" license={license}>
            <DevicesView
              runtime={state.runtime}
              error={state.errors.capture}
              onRuntimeRefresh={load}
              onOpenModels={() => setActiveView("models")}
              initialSection={deviceSettingsSection}
            />
          </PermissionGuard>
        ) : null}
        {activeView === "models" ? (
          <PermissionGuard feature="models" license={license}>
            <ModelsView
              projects={state.projects}
              activeModel={activeModel}
              runtimeInference={state.runtime?.inference}
              error={state.errors.projects}
              onRuntimeRefresh={load}
              onOpenInference={() => {
                setDeviceSettingsSection("inference");
                setActiveView("devices");
              }}
            />
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
