# Studio Production UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing NovaSight React console into a production-grade Studio workbench without adding unsupported Admin features.

**Architecture:** Keep backend APIs unchanged. First introduce design tokens, reusable UI primitives, and a Studio shell around the existing behavior; then move the current monolithic views into feature modules; finally improve realtime freshness, stream error states, and model registry coverage against existing endpoints.

**Tech Stack:** React 18, TypeScript, Vite, CSS variables, existing fetch/WebSocket API layer. No new package dependencies in the first pass.

---

## File Structure

- Create `web/src/design/tokens.css`: dark-first and light-mode semantic tokens.
- Create `web/src/app/navigation.ts`: Studio view metadata and reserved Admin namespace comments.
- Create `web/src/app/StudioShell.tsx`: sidebar, top bar, page header, mobile/bottom navigation.
- Create `web/src/components/ui/*.tsx`: reusable Button, Panel, StatusIndicator, EmptyState, InlineError, LoadingSkeleton, Badge, ConfirmDialog primitives.
- Create `web/src/components/studio/*.tsx`: MetricCard, VideoPanel, PermissionGuard, ConfigForm helpers.
- Create `web/src/features/license/LicenseView.tsx`: extracted license gate and license panel.
- Create `web/src/features/dashboard/DashboardView.tsx`: runtime summary and dashboard panels.
- Create `web/src/features/devices/DevicesView.tsx`: capture workbench and diagnostics.
- Create `web/src/features/models/ModelsView.tsx`: projects, versions, artifacts, jobs, publish/rollback UI backed by existing model endpoints.
- Create `web/src/features/config/ConfigView.tsx`: schema-driven config editor with dirty state and save warnings.
- Create `web/src/features/plugins/PluginsView.tsx`: plugin and executor status.
- Modify `web/src/api.ts`: add missing model registry endpoint helpers and export license feature type.
- Modify `web/src/App.tsx`: reduce to app state orchestration and view composition.
- Modify `web/src/styles.css`: replace current page styling with tokenized shell/component styles.

## Task 1: Token Foundation And Shell Skeleton

**Files:**
- Create: `web/src/design/tokens.css`
- Create: `web/src/app/navigation.ts`
- Create: `web/src/app/StudioShell.tsx`
- Modify: `web/src/main.tsx`
- Modify: `web/src/styles.css`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Add design tokens**

Create `web/src/design/tokens.css`:

```css
:root {
  color-scheme: dark;
  --bg-app: #080b0f;
  --bg-subtle: #0c1117;
  --surface-1: #111820;
  --surface-2: #151e28;
  --surface-3: #1b2633;
  --border-subtle: rgba(148, 163, 184, 0.16);
  --border-strong: rgba(148, 163, 184, 0.28);
  --text-primary: #eef4f8;
  --text-secondary: #9aa8b5;
  --text-muted: #667584;
  --text-disabled: #46525f;
  --primary: #33c7d6;
  --primary-strong: #17a8b8;
  --success: #3ddc97;
  --warning: #f6c350;
  --danger: #ff6b66;
  --info: #67a8ff;
  --radius-control: 6px;
  --radius-panel: 8px;
  --radius-dialog: 10px;
  --font-ui: "Aptos", "Segoe UI", "PingFang SC", "Microsoft YaHei UI", system-ui, -apple-system, sans-serif;
  --font-mono: "SFMono-Regular", "Cascadia Mono", Consolas, monospace;
}

[data-theme="light"] {
  color-scheme: light;
  --bg-app: #f5f7f9;
  --bg-subtle: #edf1f5;
  --surface-1: #ffffff;
  --surface-2: #f8fafc;
  --surface-3: #eef3f7;
  --border-subtle: rgba(15, 23, 42, 0.12);
  --border-strong: rgba(15, 23, 42, 0.22);
  --text-primary: #111827;
  --text-secondary: #526170;
  --text-muted: #738190;
  --text-disabled: #a3adb8;
  --primary: #087f8c;
  --primary-strong: #066671;
}
```

- [ ] **Step 2: Import tokens**

Modify `web/src/main.tsx` so token CSS loads before app CSS:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./design/tokens.css";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 3: Add navigation metadata**

Create `web/src/app/navigation.ts`:

```ts
export type StudioViewId = "dashboard" | "devices" | "models" | "config" | "plugins" | "license";

export type StudioNavItem = {
  id: StudioViewId;
  label: string;
  hint: string;
  feature?: string;
};

export const studioNavItems: StudioNavItem[] = [
  { id: "dashboard", label: "Dashboard", hint: "Runtime" },
  { id: "devices", label: "Devices", hint: "Capture", feature: "capture" },
  { id: "models", label: "Models", hint: "Registry", feature: "models" },
  { id: "config", label: "Config", hint: "Runtime", feature: "config_read" },
  { id: "plugins", label: "Plugins", hint: "Pipeline", feature: "plugins" },
  { id: "license", label: "License", hint: "Access" }
];
```

- [ ] **Step 4: Add shell component**

Create `web/src/app/StudioShell.tsx` with:

```tsx
import { StudioNavItem, StudioViewId, studioNavItems } from "./navigation";

type StudioShellProps = {
  activeView: StudioViewId;
  title: string;
  subtitle: string;
  status: React.ReactNode;
  children: React.ReactNode;
  onNavigate: (view: StudioViewId) => void;
};

export function StudioShell({
  activeView,
  title,
  subtitle,
  status,
  children,
  onNavigate
}: StudioShellProps) {
  return (
    <main className="studio-shell">
      <aside className="studio-sidebar" aria-label="NovaSight Studio navigation">
        <div className="studio-brand">
          <span className="brand-mark"><span>NS</span></span>
          <div>
            <strong>NovaSight</strong>
            <span>Studio</span>
          </div>
        </div>
        <nav className="studio-nav">
          {studioNavItems.map((item: StudioNavItem) => (
            <button
              aria-current={activeView === item.id ? "page" : undefined}
              className={activeView === item.id ? "nav-item active" : "nav-item"}
              key={item.id}
              onClick={() => onNavigate(item.id)}
              type="button"
            >
              <span>{item.label}</span>
              <small>{item.hint}</small>
            </button>
          ))}
        </nav>
        <div className="admin-reserved">Admin module reserved for future SaaS APIs.</div>
      </aside>
      <section className="studio-main">
        <header className="studio-topbar">
          <div>
            <p className="breadcrumb">NovaSight Platform / Studio</p>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div className="topbar-status">{status}</div>
        </header>
        <div className="studio-content">{children}</div>
      </section>
      <nav className="mobile-nav" aria-label="Mobile Studio navigation">
        {studioNavItems.slice(0, 5).map((item) => (
          <button
            aria-current={activeView === item.id ? "page" : undefined}
            className={activeView === item.id ? "mobile-nav-item active" : "mobile-nav-item"}
            key={item.id}
            onClick={() => onNavigate(item.id)}
            type="button"
          >
            {item.label}
          </button>
        ))}
      </nav>
    </main>
  );
}
```

- [ ] **Step 5: Wire shell around existing views**

Modify `web/src/App.tsx` only enough to map old tabs to new view IDs:

```ts
type TabId = "dashboard" | "devices" | "models" | "plugins" | "config" | "license";
```

Replace tab labels with `StudioShell`. Keep the existing view components and data loading unchanged.

- [ ] **Step 6: Verify shell build**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0 and Vite build output.

- [ ] **Step 7: Commit**

```bash
git add web/src/design/tokens.css web/src/app/navigation.ts web/src/app/StudioShell.tsx web/src/main.tsx web/src/styles.css web/src/App.tsx
git commit -m "Build Studio shell and token foundation"
```

## Task 2: Shared UI Primitives

**Files:**
- Create: `web/src/components/ui/Button.tsx`
- Create: `web/src/components/ui/Panel.tsx`
- Create: `web/src/components/ui/StatusIndicator.tsx`
- Create: `web/src/components/ui/EmptyState.tsx`
- Create: `web/src/components/ui/InlineError.tsx`
- Create: `web/src/components/ui/LoadingSkeleton.tsx`
- Create: `web/src/components/ui/Badge.tsx`
- Create: `web/src/components/ui/index.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [ ] **Step 1: Create Button primitive**

Create `web/src/components/ui/Button.tsx`:

```tsx
import { ButtonHTMLAttributes } from "react";

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: "default" | "compact";
  loading?: boolean;
};

export function Button({
  className = "",
  disabled,
  children,
  variant = "secondary",
  size = "default",
  loading = false,
  ...props
}: ButtonProps) {
  return (
    <button
      className={`ui-button ${variant} ${size} ${className}`.trim()}
      disabled={disabled || loading}
      type="button"
      {...props}
    >
      {loading ? "处理中" : children}
    </button>
  );
}
```

- [ ] **Step 2: Create Panel primitive**

Create `web/src/components/ui/Panel.tsx`:

```tsx
export function Panel({
  title,
  eyebrow,
  action,
  children
}: {
  title: string;
  eyebrow?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="ui-panel">
      <div className="ui-panel-header">
        <div>
          {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
          <h2>{title}</h2>
        </div>
        {action ? <div className="ui-panel-action">{action}</div> : null}
      </div>
      <div className="ui-panel-body">{children}</div>
    </section>
  );
}
```

- [ ] **Step 3: Create state primitives**

Create `StatusIndicator`, `EmptyState`, `InlineError`, `LoadingSkeleton`, and `Badge` with the same public behavior currently embedded in `App.tsx`. Use class names `status-pill`, `empty-state`, `inline-error`, `loading-grid`, and `badge` to minimize CSS churn.

- [ ] **Step 4: Export primitives**

Create `web/src/components/ui/index.ts`:

```ts
export { Badge } from "./Badge";
export { Button } from "./Button";
export { EmptyState } from "./EmptyState";
export { InlineError } from "./InlineError";
export { LoadingSkeleton } from "./LoadingSkeleton";
export { Panel } from "./Panel";
export { StatusIndicator } from "./StatusIndicator";
```

- [ ] **Step 5: Replace local primitives in App**

Modify `web/src/App.tsx` to import primitives from `./components/ui` and delete the duplicate local `Panel`, `EmptyState`, `InlineError`, and `StatusPill` implementations. Keep `Field` local until feature extraction.

- [ ] **Step 6: Verify primitives**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/ui web/src/App.tsx web/src/styles.css
git commit -m "Extract Studio UI primitives"
```

## Task 3: Feature Extraction Without Behavior Changes

**Files:**
- Create: `web/src/features/license/LicenseView.tsx`
- Create: `web/src/features/dashboard/DashboardView.tsx`
- Create: `web/src/features/devices/DevicesView.tsx`
- Create: `web/src/features/models/ModelsView.tsx`
- Create: `web/src/features/plugins/PluginsView.tsx`
- Create: `web/src/features/config/ConfigView.tsx`
- Create: `web/src/features/shared/format.ts`
- Create: `web/src/features/shared/Field.tsx`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Move formatting helpers**

Create `web/src/features/shared/format.ts` and move these helpers from `App.tsx`:

```ts
export function formatTime(date: Date | null): string { /* existing implementation */ }
export function formatEpoch(seconds: number | null | undefined): string { /* existing implementation */ }
export function statusTone(value: boolean | undefined): "good" | "warn" | "bad" { /* existing implementation */ }
```

- [ ] **Step 2: Move Field component**

Create `web/src/features/shared/Field.tsx` using the current `Field` implementation.

- [ ] **Step 3: Extract LicenseView**

Move `LicensePanel` and `LicenseGate` into `web/src/features/license/LicenseView.tsx`. Keep props identical:

```ts
type LicenseViewProps = {
  license: LicenseStatus | null;
  loading: boolean;
  error: string | undefined;
  onRefresh: () => void;
  onLicenseChange: (license: LicenseStatus) => void;
};
```

- [ ] **Step 4: Extract DashboardView**

Move `OverviewView`, `ExecutorTable`, and `InferenceControl` into `web/src/features/dashboard/DashboardView.tsx`. Rename exported component to `DashboardView`.

- [ ] **Step 5: Extract DevicesView**

Move `CaptureWorkbench`, capability helpers, `CapabilityTable`, and `CaptureDiagnostics` into `web/src/features/devices/DevicesView.tsx`.

- [ ] **Step 6: Extract ModelsView, PluginsView, ConfigView**

Move existing same-name view logic into their feature files without adding new API behavior yet.

- [ ] **Step 7: Reduce App**

Modify `web/src/App.tsx` so it owns only:

- active view state
- license state
- aggregate load state
- WebSocket subscription
- runtime start/stop command handler
- composition of feature views

- [ ] **Step 8: Verify extraction**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0.

- [ ] **Step 9: Commit**

```bash
git add web/src/App.tsx web/src/features
git commit -m "Split Studio views into feature modules"
```

## Task 4: Permission And Freshness States

**Files:**
- Create: `web/src/components/studio/PermissionGuard.tsx`
- Create: `web/src/components/studio/MetricCard.tsx`
- Create: `web/src/components/studio/VideoPanel.tsx`
- Create: `web/src/components/studio/index.ts`
- Modify: `web/src/api.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/features/dashboard/DashboardView.tsx`
- Modify: `web/src/features/devices/DevicesView.tsx`

- [ ] **Step 1: Add feature type**

Modify `web/src/api.ts`:

```ts
export type LicenseFeature =
  | "capture"
  | "runtime"
  | "models"
  | "plugins"
  | "tensorrt"
  | "hardware_control"
  | "config_read"
  | "config_write";
```

- [ ] **Step 2: Add PermissionGuard**

Create `web/src/components/studio/PermissionGuard.tsx`:

```tsx
import { LicenseFeature, LicenseStatus } from "../../api";
import { EmptyState } from "../ui";

export function PermissionGuard({
  license,
  feature,
  children
}: {
  license: LicenseStatus | null;
  feature: LicenseFeature;
  children: React.ReactNode;
}) {
  if (!license?.features.includes(feature)) {
    return (
      <EmptyState
        title="当前授权不包含此能力"
        detail={`需要 ${feature} feature 才能使用该模块。`}
      />
    );
  }
  return <>{children}</>;
}
```

- [ ] **Step 3: Track WebSocket freshness**

Modify `App.tsx` to add:

```ts
type RealtimeStatus = "connecting" | "connected" | "stale" | "disconnected";
```

Set status to `connected` when a frame arrives. Use a 2 second interval to mark `stale` when `Date.now() - lastWsMessageAt > 2000`. Set `disconnected` in `onclose` and `onerror`.

- [ ] **Step 4: Add VideoPanel**

Create `web/src/components/studio/VideoPanel.tsx` that wraps the MJPEG `img`, tracks `onLoad` and `onError`, and shows overlay text for `offline`, `loading`, or `error`.

- [ ] **Step 5: Use permission and freshness states**

Wrap Devices with `capture`, Models with `models`, Plugins with `plugins`, Config with `config_read`. Display realtime status in `StudioShell` top bar.

- [ ] **Step 6: Verify**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add web/src
git commit -m "Add Studio permission and realtime freshness states"
```

## Task 5: Model Registry UI Expansion

**Files:**
- Modify: `web/src/api.ts`
- Modify: `web/src/features/models/ModelsView.tsx`

- [ ] **Step 1: Add missing model types and helpers**

Extend `web/src/api.ts` with types for model versions, conversion jobs, publish and rollback. Add helpers:

```ts
export function getModelVersions(projectId: number): Promise<ModelVersion[]>;
export function getModelArtifacts(versionId: number): Promise<ModelArtifact[]>;
export function getConversionJobs(versionId?: number): Promise<ConversionJob[]>;
export function publishModel(projectId: number, artifactId: number): Promise<Deployment>;
export function rollbackModel(projectId: number): Promise<Deployment>;
```

- [ ] **Step 2: Add selected project/version state**

Modify `ModelsView` to select the first project by default, load versions for the selected project, select the first version, then load artifacts and jobs.

- [ ] **Step 3: Add publish confirmation**

Before `publishModel`, show `window.confirm` with project, artifact path, artifact kind, and artifact status. If kind is not `engine`, include a warning in the message.

- [ ] **Step 4: Add rollback confirmation**

Before `rollbackModel`, show `window.confirm` with selected project name and id.

- [ ] **Step 5: Preserve empty/error states**

If projects, versions, artifacts, or jobs are empty, render `EmptyState` inside the relevant panel. If a request fails, show `InlineError` in that panel and keep the last selected parent visible.

- [ ] **Step 6: Verify**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add web/src/api.ts web/src/features/models/ModelsView.tsx
git commit -m "Expand Studio model registry UI"
```

## Task 6: Config Diff And Save Safety

**Files:**
- Modify: `web/src/features/config/ConfigView.tsx`

- [ ] **Step 1: Add dirty detection**

Store `initialConfig` when schema loads. Compare `JSON.stringify(initialConfig)` and `JSON.stringify(config)` to determine dirty state.

- [ ] **Step 2: Add field-level dirty marks**

For each schema field, compare `getConfigValue(initialConfig, field.path)` and `getConfigValue(config, field.path)`. Render a `Badge` with `Changed` when different.

- [ ] **Step 3: Add save confirmation for restart or dangerous fields**

Before saving, calculate changed paths. If any changed field has `restart_required` or is in the dangerous list from the spec, show a `window.confirm` message listing paths. Cancel save if rejected.

- [ ] **Step 4: Update initialConfig after save**

When save succeeds, set both `config` and `initialConfig` to the returned config.

- [ ] **Step 5: Verify**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add web/src/features/config/ConfigView.tsx
git commit -m "Add Studio config diff and save safety"
```

## Task 7: Visual QA And Final Verification

**Files:**
- Modify only files needed for defects found during QA.

- [ ] **Step 1: Run full frontend checks**

Run: `pnpm --dir web typecheck`

Expected: exit 0.

Run: `pnpm --dir web build`

Expected: exit 0.

- [ ] **Step 2: Start backend**

Run: `python3 -m novasight --host 127.0.0.1 --port 5174`

Expected: backend starts and serves `/healthz`.

- [ ] **Step 3: Start frontend**

Run: `pnpm --dir web dev`

Expected: Vite serves `http://127.0.0.1:5173`.

- [ ] **Step 4: Browser QA**

Verify these states manually or with browser automation:

- License gate renders and can fill test key.
- Studio shell renders after valid license.
- Dashboard does not show fake telemetry.
- Devices page can refresh capabilities or show a clear device error.
- Models page handles empty registry.
- Config page loads schema and marks changed fields.
- Plugins page handles empty or loaded plugin groups.
- Mobile width does not overlap text or controls.

- [ ] **Step 5: Commit QA fixes**

```bash
git add web/src
git commit -m "Polish Studio production UI"
```

## Self-Review

Spec coverage:

- Studio shell, dark/light tokens, reusable components, feature extraction, permission states, realtime freshness, models expansion, and config safety are covered.
- Admin, recordings, logs, users, orders, and batch licenses are explicitly excluded.
- GPU/CPU/temperature/memory telemetry remains unavailable unless backend support is later added.

Placeholder scan:

- The plan has no `TBD` or `TODO` items.
- Steps that involve implementation name concrete files, commands, and expected verification.

Type consistency:

- `StudioViewId`, `LicenseFeature`, and component names are defined before use.
- Existing API names remain compatible with `web/src/api.ts`.
