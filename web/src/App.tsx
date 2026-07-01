import { useCallback, useEffect, useMemo, useState } from "react";

import {
  ApiError,
  ActiveModel,
  ExecutorStatus,
  HealthResponse,
  ModelProject,
  PluginInfo,
  RuntimeState,
  getHealth,
  getModelProjects,
  getPlugins,
  getRuntimeState
} from "./api";

type TabId = "dashboard" | "live" | "models" | "plugins" | "settings";

type LoadState = {
  loading: boolean;
  error: string | null;
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  plugins: PluginInfo[];
  projects: ModelProject[];
  lastUpdated: Date | null;
};

const tabs: Array<{ id: TabId; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "live", label: "Live View" },
  { id: "models", label: "Models" },
  { id: "plugins", label: "Plugins" },
  { id: "settings", label: "Settings" }
];

const initialState: LoadState = {
  loading: true,
  error: null,
  health: null,
  runtime: null,
  plugins: [],
  projects: [],
  lastUpdated: null
};

function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unable to reach NovaSight backend";
}

function formatTime(date: Date | null): string {
  if (!date) {
    return "Never";
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

function statusTone(value: boolean | undefined): "good" | "warn" | "bad" {
  if (value === true) {
    return "good";
  }
  if (value === false) {
    return "bad";
  }
  return "warn";
}

function StatusPill({
  tone,
  children
}: {
  tone: "good" | "warn" | "bad" | "idle";
  children: React.ReactNode;
}) {
  return <span className={`status-pill ${tone}`}>{children}</span>;
}

function Field({
  label,
  value,
  mono = false
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <span className={mono ? "field-value mono" : "field-value"}>{value}</span>
    </div>
  );
}

function Panel({
  title,
  eyebrow,
  children,
  action
}: {
  title: string;
  eyebrow?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
          <h2>{title}</h2>
        </div>
        {action ? <div className="panel-action">{action}</div> : null}
      </div>
      {children}
    </section>
  );
}

function EmptyState({
  title,
  detail,
  command
}: {
  title: string;
  detail: string;
  command?: string;
}) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
      {command ? <code>{command}</code> : null}
    </div>
  );
}

function DashboardView({
  health,
  runtime,
  loading,
  onRefresh
}: {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  const executor = runtime?.executor;
  const selectedExecutor = executor?.selected;
  const selectedAvailability = selectedExecutor
    ? executor?.executors[selectedExecutor]?.available
    : undefined;
  const activeModel = runtime?.active_model;

  return (
    <div className="view-grid dashboard-grid">
      <Panel
        title="Runtime"
        eyebrow="Core state"
        action={
          <button className="button" type="button" onClick={onRefresh}>
            Refresh
          </button>
        }
      >
        <div className="metric-grid">
          <div className="metric">
            <span>Backend</span>
            <StatusPill tone={statusTone(health?.ok)}>
              {loading ? "Checking" : health?.ok ? "Healthy" : "Offline"}
            </StatusPill>
          </div>
          <div className="metric">
            <span>Runtime</span>
            <StatusPill tone={runtime?.running ? "good" : "idle"}>
              {runtime?.running ? "Running" : "Standby"}
            </StatusPill>
          </div>
          <div className="metric">
            <span>Executor</span>
            <strong>{selectedExecutor ?? "Unresolved"}</strong>
          </div>
          <div className="metric">
            <span>Availability</span>
            <StatusPill tone={statusTone(selectedAvailability)}>
              {selectedAvailability === undefined
                ? "Unknown"
                : selectedAvailability
                  ? "Available"
                  : "Unavailable"}
            </StatusPill>
          </div>
        </div>
        <div className="field-grid">
          <Field label="Source" value={runtime?.source ?? "No source configured"} mono />
          <Field
            label="Active model"
            value={activeModel?.project?.name ?? "No published model"}
          />
          <Field
            label="Artifact"
            value={activeModel?.artifact ? activeModel.artifact.path : "No artifact bound"}
            mono
          />
        </div>
      </Panel>

      <Panel title="Executor Matrix" eyebrow="Selection and readiness">
        <ExecutorTable executor={executor} />
      </Panel>
    </div>
  );
}

function ExecutorTable({ executor }: { executor: ExecutorStatus | undefined }) {
  const entries = Object.entries(executor?.executors ?? {});

  if (entries.length === 0) {
    return (
      <EmptyState
        title="No executors reported"
        detail="The backend did not return executor status."
      />
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Executor</th>
            <th>Selected</th>
            <th>Available</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([id, state]) => (
            <tr key={id}>
              <td className="mono">{id}</td>
              <td>{executor?.selected === id ? "Yes" : "No"}</td>
              <td>
                <StatusPill tone={state.available ? "good" : "bad"}>
                  {state.available ? "Available" : "Unavailable"}
                </StatusPill>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LiveView({ runtime }: { runtime: RuntimeState | null }) {
  return (
    <div className="view-grid live-grid">
      <Panel title="Frame Monitor" eyebrow="Source feed">
        <div className="video-shell">
          <div className="reticle" />
          <div className="video-status">
            <StatusPill tone={runtime?.running ? "good" : "idle"}>
              {runtime?.running ? "Streaming" : "Standby"}
            </StatusPill>
            <span className="mono">{runtime?.source ?? "source:none"}</span>
          </div>
          <div className="scan-lines" />
        </div>
      </Panel>
      <Panel title="Frame Telemetry" eyebrow="Current sample">
        <div className="field-grid compact">
          <Field label="Input" value={runtime?.source ?? "No source"} mono />
          <Field label="Frame rate" value="0 fps" />
          <Field label="Inference latency" value="No sample" />
          <Field label="Control intents" value="0 pending" />
          <Field label="Capture state" value={runtime?.running ? "Armed" : "Paused"} />
        </div>
      </Panel>
    </div>
  );
}

function ModelsView({
  projects,
  activeModel
}: {
  projects: ModelProject[];
  activeModel: ActiveModel | null;
}) {
  return (
    <div className="view-grid models-grid">
      <Panel title="Active Model" eyebrow="Runtime deployment">
        {activeModel ? (
          <div className="field-grid">
            <Field label="Project" value={activeModel.project?.name ?? "Unknown project"} />
            <Field label="Deployment ID" value={activeModel.deployment.id} mono />
            <Field
              label="Artifact"
              value={activeModel.artifact?.path ?? "Artifact missing"}
              mono
            />
            <Field label="Artifact status" value={activeModel.artifact?.status ?? "Unknown"} />
          </div>
        ) : (
          <EmptyState
            title="No active model"
            detail="Publish a ready artifact from the registry before arming runtime inference."
            command="POST /api/models/projects/{project_id}/publish"
          />
        )}
      </Panel>
      <Panel title="Projects" eyebrow="Registry">
        {projects.length > 0 ? (
          <div className="project-list">
            {projects.map((project) => (
              <article className="project-row" key={project.id}>
                <div>
                  <strong>{project.name}</strong>
                  <span>{project.description || "No description"}</span>
                </div>
                <code>#{project.id}</code>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="Registry is empty"
            detail="Create a model project, add a version, convert an artifact, then publish it."
            command="POST /api/models/projects"
          />
        )}
      </Panel>
    </div>
  );
}

function PluginsView({ plugins }: { plugins: PluginInfo[] }) {
  const groups = useMemo(
    () => ({
      vision: plugins.filter((plugin) => plugin.kind === "vision"),
      control: plugins.filter((plugin) => plugin.kind === "control"),
      other: plugins.filter((plugin) => plugin.kind !== "vision" && plugin.kind !== "control")
    }),
    [plugins]
  );

  if (plugins.length === 0) {
    return (
      <Panel title="Plugins" eyebrow="Runtime chain">
        <EmptyState
          title="No plugins loaded"
          detail="The runtime returned an empty plugin chain."
        />
      </Panel>
    );
  }

  return (
    <div className="view-grid plugins-grid">
      <PluginGroup title="Vision" plugins={groups.vision} />
      <PluginGroup title="Control" plugins={groups.control} />
      {groups.other.length > 0 ? <PluginGroup title="Other" plugins={groups.other} /> : null}
    </div>
  );
}

function PluginGroup({ title, plugins }: { title: string; plugins: PluginInfo[] }) {
  return (
    <Panel title={`${title} Plugins`} eyebrow="Loaded modules">
      {plugins.length > 0 ? (
        <div className="plugin-list">
          {plugins.map((plugin) => (
            <article className="plugin-row" key={plugin.plugin_id}>
              <div>
                <strong className="mono">{plugin.plugin_id}</strong>
                <span>{plugin.kind}</span>
              </div>
              <StatusPill tone={plugin.enabled ? "good" : "idle"}>
                {plugin.enabled ? "Enabled" : "Disabled"}
              </StatusPill>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState title={`No ${title.toLowerCase()} plugins`} detail="No modules in this lane." />
      )}
    </Panel>
  );
}

function SettingsView({ runtime }: { runtime: RuntimeState | null }) {
  return (
    <div className="view-grid settings-grid">
      <Panel title="Runtime Configuration" eyebrow="Loaded config">
        <div className="field-grid">
          <Field label="Source default" value={runtime?.source ?? "Not loaded"} mono />
          <Field label="Executor default" value={runtime?.executor.selected ?? "Not loaded"} mono />
          <Field label="Model binding" value={runtime?.active_model ? "Published" : "Unset"} />
        </div>
      </Panel>
      <Panel title="Operations" eyebrow="Local console">
        <div className="settings-list">
          <label>
            <span>Capture guard</span>
            <select defaultValue="manual">
              <option value="manual">Manual arm</option>
              <option value="boot">Arm on boot</option>
            </select>
          </label>
          <label>
            <span>Telemetry interval</span>
            <select defaultValue="1s">
              <option value="500ms">500 ms</option>
              <option value="1s">1 s</option>
              <option value="5s">5 s</option>
            </select>
          </label>
          <label className="check-row">
            <input type="checkbox" defaultChecked />
            <span>Show unavailable executors</span>
          </label>
        </div>
      </Panel>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [state, setState] = useState<LoadState>(initialState);

  const load = useCallback(async () => {
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const [health, runtime, plugins, projects] = await Promise.all([
        getHealth(),
        getRuntimeState(),
        getPlugins(),
        getModelProjects()
      ]);
      setState({
        loading: false,
        error: null,
        health,
        runtime,
        plugins,
        projects,
        lastUpdated: new Date()
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        loading: false,
        error: getErrorMessage(error),
        lastUpdated: new Date()
      }));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const activeModel = state.runtime?.active_model ?? null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">NS</span>
          <div>
            <h1>NovaSight</h1>
            <p>Jetson realtime vision console</p>
          </div>
        </div>
        <div className="topbar-status">
          <StatusPill tone={state.error ? "bad" : state.health?.ok ? "good" : "warn"}>
            {state.error ? "API fault" : state.health?.ok ? "Connected" : "Connecting"}
          </StatusPill>
          <span className="last-updated">Updated {formatTime(state.lastUpdated)}</span>
        </div>
      </header>

      <nav className="tabs" aria-label="Console views">
        {tabs.map((tab) => (
          <button
            aria-current={activeTab === tab.id ? "page" : undefined}
            className={activeTab === tab.id ? "tab active" : "tab"}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {state.error ? (
        <div className="alert" role="alert">
          <strong>Backend request failed</strong>
          <span>{state.error}</span>
          <button className="button compact-button" type="button" onClick={load}>
            Retry
          </button>
        </div>
      ) : null}

      {state.loading && !state.runtime ? (
        <div className="loading-grid" aria-label="Loading console data">
          <div />
          <div />
          <div />
        </div>
      ) : null}

      <div className="view-stack">
        {activeTab === "dashboard" ? (
          <DashboardView
            health={state.health}
            loading={state.loading}
            runtime={state.runtime}
            onRefresh={load}
          />
        ) : null}
        {activeTab === "live" ? <LiveView runtime={state.runtime} /> : null}
        {activeTab === "models" ? (
          <ModelsView projects={state.projects} activeModel={activeModel} />
        ) : null}
        {activeTab === "plugins" ? <PluginsView plugins={state.plugins} /> : null}
        {activeTab === "settings" ? <SettingsView runtime={state.runtime} /> : null}
      </div>
    </main>
  );
}
