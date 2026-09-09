# NovaSight Studio Production UI Design

## Decision

NovaSight Platform will split product surfaces into two layers:

- NovaSight Studio: the current Jetson realtime vision workbench. This is the first implementation target.
- NovaSight Admin: the future SaaS administration surface. It remains a routed, documented extension point until backend APIs exist.

This phase must not create fake Admin, user, order, revenue, or batch license features in the frontend. The production work is limited to the capabilities already represented by the codebase: license activation, runtime state/control, capture capabilities and MJPEG preview, model registry, plugins, executors, and runtime configuration.

## Existing Product Surface

Current frontend:

- `web/src/App.tsx` owns all UI state and view rendering.
- `web/src/api.ts` centralizes API paths and fetch helpers.
- Views are tab-based: `overview`, `capture`, `models`, `plugins`, `settings`.
- The UI gates access through local license activation.

Current backend capabilities:

- License: `GET /api/license`, `POST /api/license/activate`, `PUT /api/license`, `DELETE /api/license`.
- Runtime: `GET /api/runtime/state`, `POST /api/runtime/start`, `POST /api/runtime/stop`, `ws://.../ws/status`.
- Config: `GET /api/config`, `GET /api/config/schema`, `PUT /api/config`.
- Capture: `GET /api/capture/capabilities`, `GET /api/capture/state`, `POST /api/capture/select`, `POST /api/capture/stop`, `GET /api/capture/stream.mjpg`.
- Models: projects, versions, artifacts, conversion jobs, publish, rollback.
- Plugins and executors: read-only list/status endpoints.

## Scope

### In Scope

- Replace the tab-like console with a Studio application shell.
- Add a design-token layer for dark-first and light mode.
- Split the current monolithic UI into reusable Studio components.
- Preserve current API behavior while improving UI states.
- Make the current views feel like a production engineering tool:
  - Dashboard
  - Devices / Capture
  - Models
  - Config
  - Plugins / Executors
  - License
- Add explicit empty, loading, error, no-license, disconnected, stale, and permission-state patterns.
- Use license features as the first permission primitive.

### Out of Scope

- SaaS Admin implementation.
- User account login, registration, password reset, and RBAC.
- Orders, revenue, audit-log management, batch license generation, or card-key inventory.
- Real recordings UI unless a recordings API is added.
- Real logs UI unless a logs query or streaming API is added.
- Fake telemetry for GPU, CPU, temperature, or memory. These may appear only as unavailable/extension states until an API exists.

## Recommended Architecture

```text
web/src
├── app
│   ├── App.tsx
│   ├── StudioShell.tsx
│   └── navigation.ts
├── api
│   ├── client.ts
│   ├── capture.ts
│   ├── license.ts
│   ├── models.ts
│   └── runtime.ts
├── components
│   ├── ui
│   └── studio
├── features
│   ├── dashboard
│   ├── devices
│   ├── models
│   ├── config
│   ├── plugins
│   └── license
├── design
│   └── tokens.css
└── main.tsx
```

This structure is a target. Implementation should be staged so that small, verified changes preserve the current working console.

## Navigation Design

Studio routes or route-like view IDs:

```text
/studio/dashboard
/studio/devices
/studio/models
/studio/config
/studio/plugins
/studio/license
```

Admin route namespace is reserved:

```text
/admin/*
```

The product switcher can show Admin as disabled or hidden until the backend exists. It must not route users to fake pages.

Studio shell:

- Left sidebar: primary modules.
- Top bar: current device, runtime status, WebSocket freshness, current model, license tier, theme toggle.
- Content header: page title, breadcrumb, primary page action.
- Main content: dense panels and tables with engineering-focused state displays.

Narrow screens:

- Sidebar collapses to icon rail at tablet width.
- Mobile uses a compact top bar and bottom navigation for Dashboard, Devices, Models, Config, License.
- Complex tables become stacked rows or horizontally scrollable only where unavoidable.

## Design System

Use CSS variables with semantic names. Dark mode is default.

```css
:root {
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
  --primary: #33c7d6;
  --success: #3ddc97;
  --warning: #f6c350;
  --danger: #ff6b66;
  --info: #67a8ff;
}
```

Light mode should be implemented through `[data-theme="light"]`, not a separate stylesheet.

Typography:

- UI: system stack, with Chinese support.
- Metrics and logs: `SFMono-Regular`, `Cascadia Mono`, `Consolas`, monospace.
- Use tabular numerals for metrics.

Spacing:

- 4px base grid.
- Panels use 16px or 20px padding.
- Dashboard panel gaps use 12px or 16px.
- Dense tables use 36px rows; default tables use 44px rows.

Shape:

- Buttons, inputs, selects: 6px radius.
- Panels and cards: 8px radius.
- Dialogs and drawers: 10px radius.
- Avoid nested cards. Use section bands, dividers, or compact panels instead.

Motion:

- Keep transitions short: 120-180ms.
- Metric refresh should not cause layout shift.
- Video disconnect overlays fade in while preserving the frame area.

## Core Components

Required reusable components:

- `Button`
- `IconButton`
- `Input`
- `Select`
- `Switch`
- `Slider`
- `Card` / `Panel`
- `MetricCard`
- `DataTable`
- `StatusIndicator`
- `Badge`
- `EmptyState`
- `InlineError`
- `LoadingSkeleton`
- `ConfirmDialog`
- `PermissionGuard`
- `VideoPanel`
- `ConfigForm`

`PermissionGuard` initially checks license features:

```ts
type Feature =
  | "capture"
  | "runtime"
  | "models"
  | "plugins"
  | "tensorrt"
  | "hardware_control"
  | "config_read"
  | "config_write";
```

Future RBAC should be added behind the same component and helper API.

## Page Design

### License

Purpose: activate or clear the local license and explain access state.

States:

- Loading: checking `/api/license`.
- Empty: no configured license.
- Valid: show tier, fingerprint, created, activated, expires, features.
- Invalid/expired: show message and activation form.
- Failure: show backend error without losing typed input.

The license key is password-style input and must never be echoed after activation.

### Dashboard

Purpose: summarize runtime health and the active vision pipeline.

Panels:

- Runtime status: backend health, running/idle, fatal error.
- Capture metrics: capture FPS, preview FPS, frame period, wait time, drops, recoveries.
- Video preview: MJPEG stream with stale/offline overlay.
- Active model: project, artifact, status, path.
- Executor status: selected executor and availability.
- Recent alerts: derived from fatal error, capture last error, disconnected WebSocket, license expiry.

GPU, CPU, temperature, and memory are shown only as unavailable placeholders until telemetry exists.

### Devices

Purpose: operate capture device and hardware/executor status.

Panels:

- Device selector.
- Capability groups by pixel format.
- Recommended profile actions: high FPS, low latency, balanced.
- Current capture diagnostics.
- Hardware box config summary from config schema.
- Executor status table.

Actions:

- Refresh capabilities.
- Apply and start capture.
- Stop capture.
- Restart inference/control only if runtime feature exists.

### Models

Purpose: manage the model registry that already exists in the backend.

Panels:

- Active model.
- Projects list.
- Versions for selected project.
- Artifacts for selected version.
- Conversion jobs.
- Publish / rollback actions.

Rules:

- Publishing must require confirmation.
- Non-engine artifacts must warn that inference may be disabled.
- Missing artifact path/status must be visible.

### Config

Purpose: edit runtime configuration through `/api/config/schema`.

Behavior:

- Render fields from schema.
- Show dirty fields.
- Show a save diff before applying.
- Mark `restart_required` fields.
- If runtime is running and restart is required, show a restart-required warning.
- Reject invalid min/max values before sending where schema provides limits.

Dangerous fields:

- `capture.device`
- `capture.pixel_format`
- `capture.width`
- `capture.height`
- `capture.fps`
- `hardware.kind`
- `hardware.host`
- `hardware.port`
- `hardware.serial_port`

Changing dangerous fields should require confirmation when runtime or capture is active.

### Plugins / Executors

Purpose: show loaded algorithm modules and control output availability.

Current backend is read-only, so controls must not imply enable/disable unless an API is added.

Panels:

- Vision plugins.
- Control plugins.
- Other plugins.
- Executor status.

Empty states must explain that the backend returned no modules, not that the feature is broken.

## Realtime Data Flow

Initial load uses REST:

```text
license -> health/runtime/plugins/projects
```

Runtime updates use WebSocket:

```text
/ws/status -> runtime state -> dashboard/devices/config shell status
```

UI must track freshness:

- `connected`: recent WebSocket frame received.
- `stale`: no frame for more than 2 seconds.
- `disconnected`: socket closed or failed.

The UI should not re-render all panels at 10Hz. High-frequency runtime payloads should be stored centrally and rendered at a lower cadence where charts or metric cards need it.

## Error Handling

Common patterns:

- Backend offline: full-width alert with retry.
- Unauthorized/license required: route to License view.
- Capture unavailable: keep previous healthy state visible and show the error.
- Stream unavailable: video panel overlay, not broken image chrome.
- Config rejected: field-level or form-level error from API detail.
- WebSocket failure: stale badge and REST refresh button.

## Implementation Plan Boundary

The first implementation plan should be small enough to verify:

1. Add token/theme foundation and Studio shell.
2. Move existing App views into feature components without behavior changes.
3. Add shared UI primitives used by the moved views.
4. Improve state handling for WebSocket freshness and video stream errors.
5. Expand Models UI only against existing model registry endpoints.

Admin, recordings, and logs must wait for backend API design.

## Verification

Each implementation step must run:

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

When backend behavior is touched later, also run:

```bash
pytest -q
```

For frontend layout changes, use browser QA before declaring the UI ready.

## Open Follow-Up API Needs

These are not blockers for Studio productionization:

- Host telemetry: GPU, CPU, memory, temperature.
- Logs query or streaming API.
- Recordings API.
- User/RBAC API.
- SaaS Admin APIs for users, roles, orders, audit logs, and batch licenses.
