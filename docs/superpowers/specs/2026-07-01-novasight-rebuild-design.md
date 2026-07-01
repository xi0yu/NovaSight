# NovaSight Rebuild Design

Date: 2026-07-01

## Goal

Rebuild NovaSight in the current project directory from the old
`~/Workspace/Projects/jetson_cam_demo/untitled` project, keeping the useful
real-time vision ideas while removing historical weight in model management,
configuration management, algorithm management, and project structure.

The first version is a Jetson-first modular monolith with a lightweight Web UI.
It should run as one backend process plus one browser console, but the codebase
must keep runtime, model repository, plugins, executors, API, and UI concerns
separate.

## Scope

In scope:

- Jetson-first runtime pipeline.
- Camera/image/video input.
- Preprocessing, inference, tracking, and state publication.
- Model repository with upload/registration, TensorRT conversion jobs, artifacts,
  publishing, and rollback.
- Visual analysis plugins.
- Control plugins that output structured intents.
- Pluggable executors with `dry_run` as the default and kmNet/HID as an optional
  Jetson executor.
- FastAPI HTTP/WebSocket API.
- Lightweight React/Vite management UI.

Out of scope for the first version:

- Training, labeling, dataset management, or evaluation leaderboards.
- Cloud model market, multi-device sync, or remote fleet management.
- Full migration of old control algorithms as defaults.
- Marketing site, large dashboard product shell, or unrelated landing pages.

## Architecture

NovaSight will be a modular monolith:

```text
novasight/
  runtime/
  model_registry/
  plugins/
  executors/
  config/
  api/
  web/
```

The runtime pipeline only consumes released model artifacts and enabled plugin
configuration. Model upload, conversion, validation, and release are background
management operations and must not block or mutate an active runtime loop
directly.

The Web UI is an operating console. It observes state and calls API operations;
it never owns business truth and never writes YAML or SQLite directly.

## Runtime Boundary

`runtime/` owns the real-time path:

- Source acquisition.
- Frame metadata.
- Preprocessing.
- Inference through the currently deployed artifact.
- Tracking.
- Plugin context creation.
- Result publication to API/WebSocket streams.

Runtime must avoid knowing how models are uploaded or converted. Its dependency
is a deployment resolver that returns a verified runtime artifact and its class
definition.

## Model Repository

The model repository treats model assets as first-class records, not config
fields.

Core records:

- `ModelProject`: logical model family, such as `person-detector`.
- `ModelVersion`: a concrete version with source files, class definitions,
  input shape policy, task type, and target backend.
- `ModelArtifact`: runtime output such as `.onnx`, `.engine`, checksum, and
  conversion log reference.
- `ConversionJob`: asynchronous job that turns `.pt` or `.onnx` into a TensorRT
  `.engine`, including parameters, status, timestamps, and failure logs.
- `Deployment`: the active model version/artifact used by runtime, with rollback
  history.

Storage:

- Files live under `data/models/<project>/<version>/`.
- SQLite lives at `data/novasight.db`.
- Conversion logs live beside the model version they belong to.

First-version capabilities:

- Upload or register `.pt` and `.onnx`.
- Enter or import class definitions.
- Select fixed or dynamic input shape policy.
- Create TensorRT conversion jobs on Jetson.
- View conversion status and logs.
- Publish a successful artifact.
- Roll back to an earlier deployed artifact.

## Plugin System

Plugins are split into visual analysis plugins and control plugins.

Visual analysis plugins receive frame metadata, detections, tracks, model class
information, and their own config. They return structured analysis results.
Examples include target filtering, region statistics, track summaries, and
alerts.

Control plugins receive the same perception context plus their own state and
configuration. They return a `ControlIntent` and do not call hardware APIs.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlIntent:
    dx: float
    dy: float
    action: str | None
    confidence: float
    reason: str
    plugin_id: str
```

Plugin declarations include:

- `plugin_id`.
- `kind`: `vision` or `control`.
- Configuration schema.
- Input requirements.
- `process(context) -> PluginResult | ControlIntent`.

Old algorithms such as `v3`, `straight`, `chris`, and `sunset` are not copied
as default first-version code. They can be migrated later only by adapting to
the new plugin protocol: no global config reads, no direct HID import, and no
hardware sends from plugin code.

## Executor Boundary

Executors translate `ControlIntent` into external action.

First-version executors:

- `dry_run`: default executor. Records and streams intents but performs no
  external action.
- `kmnet`: optional Jetson executor. Wraps kmNet/HID behavior behind a narrow
  adapter.

This keeps control algorithms testable and makes hardware behavior explicit,
observable, and disableable.

## Configuration

Configuration is split into three ownership layers:

- Runtime YAML: `config/novasight.yaml` for port, log level, default source,
  frame limits, and default executor mode.
- Model repository SQLite: model projects, versions, artifacts, conversion jobs,
  deployments.
- Plugin SQLite state: installed plugins, enabled state, plugin config, and
  runtime stats.

YAML must not become the truth source for models or plugin internals.

## API

The API layer is the only mutation boundary for UI and tools.

Required API groups:

- Health and runtime state.
- Source selection.
- Model projects, versions, artifacts, conversion jobs, publish, and rollback.
- Plugin listing, configuration, enable/disable.
- Executor listing and selection.
- WebSocket streams for frames, runtime state, model job updates, plugin output,
  and control intents.

Operations that affect runtime state, such as publishing a model or switching an
executor, must be validated and reversible where practical.

## Web UI

The first UI has five pages:

- Dashboard: source, FPS, inference time, active model version, plugin status,
  executor mode.
- Live View: frame stream, boxes, ROI/filter overlays, plugin outputs.
- Models: upload/register models, start TensorRT conversion, inspect logs,
  publish, and roll back versions.
- Plugins: enable/disable visual and control plugins, edit plugin config.
- Settings: camera source, runtime parameters, executor selection. `dry_run` is
  the default; `kmnet` is opt-in.

The UI should be operational and dense, not a marketing site. It should use a
clear console layout and avoid old project baggage such as landing pages or
large decorative product shells.

## Migration Strategy

Use the old project as a reference, not as a source tree to copy.

Keep ideas:

- ROI/canvas coordinate discipline.
- Strict separation of runtime artifacts from user-facing selection.
- Runtime observability for FPS, inference time, and backend status.
- Tests around config patching, geometry, model selection, plugin isolation, and
  control intent behavior.

Avoid copying:

- Build/cache output directories.
- Large old Web shell.
- Legacy AI-agent docs that conflict with the new project.
- Direct hardware calls from algorithm code.
- Algorithm-specific fields in global config.
- Old compatibility layers for retired YAML shapes unless explicitly needed.

## Testing Strategy

Initial test coverage should focus on boundaries:

- Model repository state transitions: upload/register, conversion job creation,
  artifact success/failure, publish, rollback.
- Runtime deployment resolver behavior.
- Plugin contract validation and isolation.
- Control plugin to executor flow with `dry_run`.
- YAML config load/save without model/plugin truth leakage.
- API tests for model publish, plugin enable, and executor selection.

Hardware-specific kmNet behavior should be behind integration tests or explicit
Jetson-only checks. Unit tests should run on Mac without hardware.

## Open Implementation Decisions

- Exact Python package name: likely `novasight`.
- Exact frontend styling system: decide during implementation after scaffolding
  the Web UI.
- Whether TensorRT conversion is invoked by `trtexec`, Python TensorRT APIs, or
  both. First implementation should use the simplest Jetson-reliable path and
  record the command and logs.
- Whether SQLite access uses the standard library directly or a tiny repository
  wrapper. Prefer a small wrapper before adding a heavy ORM.
