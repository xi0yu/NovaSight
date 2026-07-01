# NovaSight Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable NovaSight vertical slice: modular Python backend, SQLite model registry, plugin/executor contracts, FastAPI API, and a lightweight React management console.

**Architecture:** Implement a Jetson-first modular monolith with strict internal boundaries. The runtime reads released model artifacts and enabled plugin/executor state; model conversion and publishing happen through repository/API operations and do not mutate runtime internals directly.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, PyYAML, SQLite standard library, pytest, httpx, React, Vite, TypeScript.

---

## File Structure

Create these top-level files:

- `pyproject.toml`: Python packaging, dependencies, pytest, ruff.
- `.gitignore`: ignore local data, virtualenvs, caches, model artifacts, web build output.
- `README.md`: short local run instructions.
- `config/novasight.example.yaml`: runtime-only config example.

Create these Python package files:

- `novasight/__init__.py`: package version.
- `novasight/__main__.py`: `python -m novasight` entry.
- `novasight/main.py`: CLI parsing and Uvicorn startup.
- `novasight/config/__init__.py`: config exports.
- `novasight/config/runtime.py`: runtime YAML dataclasses and load/save.
- `novasight/model_registry/__init__.py`: registry exports.
- `novasight/model_registry/schema.py`: dataclasses and enum literals for registry records.
- `novasight/model_registry/store.py`: SQLite schema and repository operations.
- `novasight/plugins/__init__.py`: plugin exports.
- `novasight/plugins/contracts.py`: plugin context/result/intent dataclasses and protocols.
- `novasight/plugins/runtime.py`: plugin registry and execution coordinator.
- `novasight/plugins/builtin.py`: first built-in plugins.
- `novasight/executors/__init__.py`: executor exports.
- `novasight/executors/contracts.py`: executor protocol and execution result.
- `novasight/executors/dry_run.py`: default executor.
- `novasight/executors/kmnet.py`: optional kmNet adapter with graceful unavailable state.
- `novasight/executors/runtime.py`: executor registry and selection.
- `novasight/runtime/__init__.py`: runtime exports.
- `novasight/runtime/state.py`: runtime status dataclasses.
- `novasight/runtime/service.py`: initial runtime service that resolves deployment and runs plugin/executor flow.
- `novasight/api/__init__.py`: API exports.
- `novasight/api/app.py`: FastAPI app factory.
- `novasight/api/routes_health.py`: health/runtime endpoints.
- `novasight/api/routes_models.py`: model registry endpoints.
- `novasight/api/routes_plugins.py`: plugin endpoints.
- `novasight/api/routes_executors.py`: executor endpoints.

Create these tests:

- `tests/test_config_runtime.py`
- `tests/test_model_registry.py`
- `tests/test_plugin_executor_flow.py`
- `tests/test_api.py`

Create these web files:

- `web/package.json`
- `web/index.html`
- `web/tsconfig.json`
- `web/vite.config.ts`
- `web/src/main.tsx`
- `web/src/App.tsx`
- `web/src/api.ts`
- `web/src/styles.css`

---

### Task 1: Project Scaffold and Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `novasight/__init__.py`
- Create: `novasight/__main__.py`
- Create: `novasight/main.py`
- Create: package `__init__.py` files under `config`, `model_registry`, `plugins`, `executors`, `runtime`, `api`
- Test: command-line import checks

- [ ] **Step 1: Create Python project metadata**

Write `pyproject.toml`:

```toml
[project]
name = "novasight"
version = "0.1.0"
description = "Jetson-first realtime vision console with model registry and plugin runtime"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pyyaml>=6.0",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
    "httpx>=0.27",
    "pytest>=8.0",
    "ruff>=0.6",
]

[project.scripts]
novasight = "novasight.main:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["novasight*"]
exclude = ["tests*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

- [ ] **Step 2: Create ignore rules**

Write `.gitignore`:

```gitignore
.DS_Store
.idea/
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
data/
config/novasight.yaml
web/node_modules/
web/dist/
web/.vite/
```

- [ ] **Step 3: Create package directories and package markers**

Create:

```text
novasight/__init__.py
novasight/__main__.py
novasight/main.py
novasight/config/__init__.py
novasight/model_registry/__init__.py
novasight/plugins/__init__.py
novasight/executors/__init__.py
novasight/runtime/__init__.py
novasight/api/__init__.py
```

Write `novasight/__init__.py`:

```python
"""NovaSight realtime vision console."""

__version__ = "0.1.0"
```

Write `novasight/__main__.py`:

```python
from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Write temporary `novasight/main.py`:

```python
from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("novasight")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5174)
    parser.add_argument("--config", default="config/novasight.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"NovaSight scaffold ready on {args.host}:{args.port}")
    return 0
```

- [ ] **Step 4: Create README**

Write `README.md`:

```markdown
# NovaSight

NovaSight is a Jetson-first realtime vision console rebuilt from the old jetcam
prototype with cleaner model, configuration, plugin, executor, API, and UI
boundaries.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m novasight
pytest -q
```

The default runtime config path is `config/novasight.yaml`. Keep model assets
and SQLite state under `data/`, which is ignored by git.
```

- [ ] **Step 5: Verify scaffold**

Run:

```bash
python -m novasight
```

Expected:

```text
NovaSight scaffold ready on 0.0.0.0:5174
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md novasight
git commit -m "chore: scaffold novasight package"
```

---

### Task 2: Runtime YAML Configuration

**Files:**
- Create: `config/novasight.example.yaml`
- Create: `novasight/config/runtime.py`
- Modify: `novasight/config/__init__.py`
- Test: `tests/test_config_runtime.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config_runtime.py`:

```python
from pathlib import Path

from novasight.config import RuntimeConfig, load_runtime_config, save_runtime_config


def test_runtime_config_defaults_do_not_contain_model_truth() -> None:
    cfg = RuntimeConfig()

    assert cfg.web.port == 5174
    assert cfg.executor.default == "dry_run"
    assert not hasattr(cfg, "model_path")
    assert not hasattr(cfg, "plugin_settings")


def test_runtime_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    cfg = RuntimeConfig()
    cfg.web.port = 6000
    cfg.source.default = "image:/tmp/frame.jpg"
    cfg.executor.default = "dry_run"

    save_runtime_config(cfg, path)
    loaded = load_runtime_config(path)

    assert loaded.web.port == 6000
    assert loaded.source.default == "image:/tmp/frame.jpg"
    assert loaded.executor.default == "dry_run"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_config_runtime.py -q
```

Expected: fail with `ModuleNotFoundError` or missing `RuntimeConfig`.

- [ ] **Step 3: Implement runtime config**

Write `novasight/config/runtime.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 5174


@dataclass
class SourceConfig:
    default: str = "null"
    target_fps: int = 60


@dataclass
class RuntimeLimitsConfig:
    max_frame_queue: int = 2
    stream_fps: int = 30


@dataclass
class ExecutorConfig:
    default: str = "dry_run"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class RuntimeConfig:
    web: WebConfig = field(default_factory=WebConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    limits: RuntimeLimitsConfig = field(default_factory=RuntimeLimitsConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


T = TypeVar("T")


def _build_dataclass(cls: type[T], raw: dict[str, Any]) -> T:
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        current = getattr(cls(), item.name) if callable(cls) else None
        value = raw[item.name]
        if is_dataclass(current) and isinstance(value, dict):
            values[item.name] = _build_dataclass(type(current), value)
        else:
            values[item.name] = value
    return cls(**values)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return RuntimeConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {cfg_path}")
    return _build_dataclass(RuntimeConfig, raw)


def save_runtime_config(cfg: RuntimeConfig, path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
```

Write `novasight/config/__init__.py`:

```python
from .runtime import RuntimeConfig, load_runtime_config, save_runtime_config

__all__ = ["RuntimeConfig", "load_runtime_config", "save_runtime_config"]
```

Create `config/novasight.example.yaml`:

```yaml
web:
  host: 0.0.0.0
  port: 5174
source:
  default: null
  target_fps: 60
limits:
  max_frame_queue: 2
  stream_fps: 30
executor:
  default: dry_run
logging:
  level: INFO
```

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest tests/test_config_runtime.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add config/novasight.example.yaml novasight/config tests/test_config_runtime.py
git commit -m "feat: add runtime yaml config"
```

---

### Task 3: SQLite Model Registry

**Files:**
- Create: `novasight/model_registry/schema.py`
- Create: `novasight/model_registry/store.py`
- Modify: `novasight/model_registry/__init__.py`
- Test: `tests/test_model_registry.py`

- [ ] **Step 1: Write failing model registry tests**

Create `tests/test_model_registry.py`:

```python
from pathlib import Path

from novasight.model_registry import ModelRegistry


def test_model_registry_publish_and_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "novasight.db"
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=db_path, data_dir=data_dir)

    project = registry.create_project("person-detector", "Person detector")
    v1 = registry.create_version(
        project_id=project.id,
        version="v1",
        source_kind="onnx",
        source_path="/tmp/person-v1.onnx",
        classes=["person"],
        input_shape="1x3x640x640",
    )
    artifact1 = registry.create_artifact(
        version_id=v1.id,
        kind="onnx",
        path=str(data_dir / "person-detector" / "v1" / "model.onnx"),
        checksum="sha256:v1",
        status="ready",
    )
    registry.publish(project.id, artifact1.id)

    active = registry.get_deployment(project.id)
    assert active is not None
    assert active.artifact_id == artifact1.id

    v2 = registry.create_version(
        project_id=project.id,
        version="v2",
        source_kind="onnx",
        source_path="/tmp/person-v2.onnx",
        classes=["person", "head"],
        input_shape="1x3x960x960",
    )
    artifact2 = registry.create_artifact(
        version_id=v2.id,
        kind="engine",
        path=str(data_dir / "person-detector" / "v2" / "model.engine"),
        checksum="sha256:v2",
        status="ready",
    )
    registry.publish(project.id, artifact2.id)
    registry.rollback(project.id)

    rolled_back = registry.get_deployment(project.id)
    assert rolled_back is not None
    assert rolled_back.artifact_id == artifact1.id


def test_conversion_job_tracks_status_and_log(tmp_path: Path) -> None:
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=tmp_path / "models")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project_id=project.id,
        version="v1",
        source_kind="pt",
        source_path="/tmp/demo.pt",
        classes=["target"],
        input_shape="dynamic",
    )

    job = registry.create_conversion_job(
        version_id=version.id,
        target_kind="engine",
        command=["trtexec", "--onnx=demo.onnx", "--saveEngine=demo.engine"],
    )
    registry.finish_conversion_job(job.id, status="failed", log="TensorRT parse error")

    loaded = registry.get_conversion_job(job.id)
    assert loaded is not None
    assert loaded.status == "failed"
    assert "TensorRT parse error" in loaded.log
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_model_registry.py -q
```

Expected: fail because `ModelRegistry` is not implemented.

- [ ] **Step 3: Implement registry records**

Write `novasight/model_registry/schema.py` with dataclasses for:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProject:
    id: int
    name: str
    description: str


@dataclass(frozen=True)
class ModelVersion:
    id: int
    project_id: int
    version: str
    source_kind: str
    source_path: str
    classes: list[str]
    input_shape: str


@dataclass(frozen=True)
class ModelArtifact:
    id: int
    version_id: int
    kind: str
    path: str
    checksum: str
    status: str


@dataclass(frozen=True)
class ConversionJob:
    id: int
    version_id: int
    target_kind: str
    command: list[str]
    status: str
    log: str


@dataclass(frozen=True)
class Deployment:
    id: int
    project_id: int
    artifact_id: int
    previous_artifact_id: int | None
```

- [ ] **Step 4: Implement SQLite repository**

Write `novasight/model_registry/store.py` with a `ModelRegistry` class that:

- Opens SQLite connections with `row_factory = sqlite3.Row`.
- Creates tables in `__init__`.
- Stores `classes` and `command` as JSON strings.
- Creates version asset directories under `data_dir/<project-name>/<version>/`.
- Rejects `publish()` for artifacts whose status is not `ready`.
- Implements `rollback()` by swapping `artifact_id` with `previous_artifact_id` when present.

The exact public methods required by tests:

```python
create_project(name: str, description: str) -> ModelProject
create_version(project_id: int, version: str, source_kind: str, source_path: str, classes: list[str], input_shape: str) -> ModelVersion
create_artifact(version_id: int, kind: str, path: str, checksum: str, status: str) -> ModelArtifact
create_conversion_job(version_id: int, target_kind: str, command: list[str]) -> ConversionJob
finish_conversion_job(job_id: int, status: str, log: str) -> None
get_conversion_job(job_id: int) -> ConversionJob | None
publish(project_id: int, artifact_id: int) -> Deployment
rollback(project_id: int) -> Deployment
get_deployment(project_id: int) -> Deployment | None
list_projects() -> list[ModelProject]
list_versions(project_id: int) -> list[ModelVersion]
list_artifacts(version_id: int) -> list[ModelArtifact]
```

- [ ] **Step 5: Export registry API**

Write `novasight/model_registry/__init__.py`:

```python
from .schema import ConversionJob, Deployment, ModelArtifact, ModelProject, ModelVersion
from .store import ModelRegistry

__all__ = [
    "ConversionJob",
    "Deployment",
    "ModelArtifact",
    "ModelProject",
    "ModelRegistry",
    "ModelVersion",
]
```

- [ ] **Step 6: Run registry tests**

Run:

```bash
pytest tests/test_model_registry.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add novasight/model_registry tests/test_model_registry.py
git commit -m "feat: add sqlite model registry"
```

---

### Task 4: Plugin Contracts and Dry-Run Executor Flow

**Files:**
- Create: `novasight/plugins/contracts.py`
- Create: `novasight/plugins/runtime.py`
- Create: `novasight/plugins/builtin.py`
- Modify: `novasight/plugins/__init__.py`
- Create: `novasight/executors/contracts.py`
- Create: `novasight/executors/dry_run.py`
- Create: `novasight/executors/kmnet.py`
- Create: `novasight/executors/runtime.py`
- Modify: `novasight/executors/__init__.py`
- Test: `tests/test_plugin_executor_flow.py`

- [ ] **Step 1: Write failing plugin/executor tests**

Create `tests/test_plugin_executor_flow.py`:

```python
from novasight.executors import ExecutorRegistry
from novasight.plugins import Detection, FrameContext, PluginRuntime, Track


def test_builtin_control_plugin_outputs_intent_and_dry_run_records_it() -> None:
    plugins = PluginRuntime.with_builtin_plugins()
    executors = ExecutorRegistry.with_builtin_executors(default="dry_run")
    context = FrameContext(
        frame_id=42,
        width=1280,
        height=720,
        detections=[Detection(cls=0, score=0.91, x=600, y=300, w=80, h=120)],
        tracks=[Track(track_id=7, cls=0, score=0.91, x=600, y=300, w=80, h=120)],
        classes=["target"],
    )

    results = plugins.process(context)
    intents = [item for item in results.control_intents if item.plugin_id == "control.center_target"]
    assert len(intents) == 1
    assert intents[0].dx < 0
    assert intents[0].dy > 0

    execution = executors.execute(intents[0])
    assert execution.executor_id == "dry_run"
    assert execution.sent is False
    assert execution.intent == intents[0]


def test_kmnet_executor_is_unavailable_without_driver() -> None:
    executors = ExecutorRegistry.with_builtin_executors(default="kmnet")

    status = executors.status()

    assert status["selected"] == "kmnet"
    assert status["executors"]["kmnet"]["available"] is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_plugin_executor_flow.py -q
```

Expected: fail because plugin and executor modules are missing.

- [ ] **Step 3: Implement plugin contracts**

Write `novasight/plugins/contracts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    cls: int
    score: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class Track:
    track_id: int
    cls: int
    score: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginResult:
    plugin_id: str
    kind: str
    payload: dict


@dataclass(frozen=True)
class ControlIntent:
    dx: float
    dy: float
    action: str | None
    confidence: float
    reason: str
    plugin_id: str


@dataclass(frozen=True)
class PluginBatchResult:
    plugin_results: list[PluginResult]
    control_intents: list[ControlIntent]


class VisionPlugin(Protocol):
    plugin_id: str
    kind: str

    def process(self, context: FrameContext) -> PluginResult: ...


class ControlPlugin(Protocol):
    plugin_id: str
    kind: str

    def process(self, context: FrameContext) -> ControlIntent | None: ...
```

- [ ] **Step 4: Implement built-in plugins and runtime**

Write `novasight/plugins/builtin.py` with:

- `TrackStatsPlugin`: returns count of detections and tracks.
- `CenterTargetControlPlugin`: selects the highest score track when present, otherwise the highest score detection. It returns `dx = target.cx - frame_width / 2` and `dy = frame_height / 2 - target.cy`.

Write `novasight/plugins/runtime.py`:

- Holds lists of enabled vision and control plugins.
- `with_builtin_plugins()` enables both built-ins.
- `process(context)` returns `PluginBatchResult`.
- Drops `None` control intents.

Write `novasight/plugins/__init__.py` exporting:

```python
from .contracts import ControlIntent, Detection, FrameContext, PluginBatchResult, PluginResult, Track
from .runtime import PluginRuntime

__all__ = [
    "ControlIntent",
    "Detection",
    "FrameContext",
    "PluginBatchResult",
    "PluginResult",
    "PluginRuntime",
    "Track",
]
```

- [ ] **Step 5: Implement executors**

Write `novasight/executors/contracts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from novasight.plugins import ControlIntent


@dataclass(frozen=True)
class ExecutionResult:
    executor_id: str
    sent: bool
    intent: ControlIntent
    message: str


class Executor(Protocol):
    executor_id: str

    def available(self) -> bool: ...

    def execute(self, intent: ControlIntent) -> ExecutionResult: ...
```

Write `novasight/executors/dry_run.py`:

```python
from __future__ import annotations

from novasight.plugins import ControlIntent

from .contracts import ExecutionResult


class DryRunExecutor:
    executor_id = "dry_run"

    def __init__(self) -> None:
        self.history: list[ControlIntent] = []

    def available(self) -> bool:
        return True

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        self.history.append(intent)
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=False,
            intent=intent,
            message="recorded intent without external action",
        )
```

Write `novasight/executors/kmnet.py`:

```python
from __future__ import annotations

from novasight.plugins import ControlIntent

from .contracts import ExecutionResult


class KmNetExecutor:
    executor_id = "kmnet"

    def __init__(self) -> None:
        try:
            import kmNet  # type: ignore
        except Exception:
            self._driver = None
        else:
            self._driver = kmNet

    def available(self) -> bool:
        return self._driver is not None

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        if self._driver is None:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=intent,
                message="kmNet driver is unavailable",
            )
        self._driver.move(int(round(intent.dx)), int(round(-intent.dy)))
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=True,
            intent=intent,
            message="sent kmNet move",
        )
```

Write `novasight/executors/runtime.py`:

- Registers `DryRunExecutor` and `KmNetExecutor`.
- Selects the requested default if present.
- `execute(intent)` delegates to selected executor.
- `status()` returns selected executor and availability map.

Write `novasight/executors/__init__.py` exporting `ExecutionResult` and `ExecutorRegistry`.

- [ ] **Step 6: Run plugin/executor tests**

Run:

```bash
pytest tests/test_plugin_executor_flow.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add novasight/plugins novasight/executors tests/test_plugin_executor_flow.py
git commit -m "feat: add plugin and executor runtime"
```

---

### Task 5: Runtime Service and FastAPI Backend

**Files:**
- Create: `novasight/runtime/state.py`
- Create: `novasight/runtime/service.py`
- Modify: `novasight/runtime/__init__.py`
- Create: `novasight/api/app.py`
- Create: `novasight/api/routes_health.py`
- Create: `novasight/api/routes_models.py`
- Create: `novasight/api/routes_plugins.py`
- Create: `novasight/api/routes_executors.py`
- Modify: `novasight/api/__init__.py`
- Modify: `novasight/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_api.py`:

```python
from pathlib import Path

from fastapi.testclient import TestClient

from novasight.api import create_app


def test_health_and_runtime_state(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    state = client.get("/api/runtime/state")
    assert state.status_code == 200
    body = state.json()
    assert body["executor"]["selected"] == "dry_run"
    assert body["active_model"] is None


def test_model_project_version_artifact_publish_flow(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)

    project = client.post(
        "/api/models/projects",
        json={"name": "demo", "description": "Demo model"},
    ).json()
    version = client.post(
        f"/api/models/projects/{project['id']}/versions",
        json={
            "version": "v1",
            "source_kind": "onnx",
            "source_path": "/tmp/demo.onnx",
            "classes": ["target"],
            "input_shape": "1x3x640x640",
        },
    ).json()
    artifact = client.post(
        f"/api/models/versions/{version['id']}/artifacts",
        json={
            "kind": "onnx",
            "path": "/tmp/demo.onnx",
            "checksum": "sha256:demo",
            "status": "ready",
        },
    ).json()

    published = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": artifact["id"]},
    )

    assert published.status_code == 200
    assert published.json()["artifact_id"] == artifact["id"]


def test_plugin_and_executor_endpoints(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)

    plugins = client.get("/api/plugins").json()
    assert "control.center_target" in {item["plugin_id"] for item in plugins}

    executors = client.get("/api/executors").json()
    assert executors["selected"] == "dry_run"
    assert executors["executors"]["dry_run"]["available"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_api.py -q
```

Expected: fail because API app is missing.

- [ ] **Step 3: Implement runtime service**

Write `novasight/runtime/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeState:
    running: bool
    source: str
    active_model: dict | None
    executor: dict
```

Write `novasight/runtime/service.py`:

- Accepts `RuntimeConfig`, `ModelRegistry`, `PluginRuntime`, `ExecutorRegistry`.
- `state()` returns `RuntimeState`.
- `active_model` is `None` until a deployment exists.
- No camera or inference loop is required in Phase 1; this is the management slice.

Write `novasight/runtime/__init__.py` exporting `RuntimeService` and `RuntimeState`.

- [ ] **Step 4: Implement FastAPI app factory**

Write `novasight/api/app.py`:

- `create_app(data_dir: Path | str = "data", config_path: Path | str = "config/novasight.yaml") -> FastAPI`.
- Creates `RuntimeConfig`, `ModelRegistry`, `PluginRuntime`, `ExecutorRegistry`, `RuntimeService`.
- Stores them on `app.state`.
- Includes routers from health, models, plugins, executors.

Write route modules:

- `routes_health.py`: `GET /healthz`, `GET /api/runtime/state`.
- `routes_models.py`: project/version/artifact/job/publish/rollback endpoints used by tests.
- `routes_plugins.py`: `GET /api/plugins`.
- `routes_executors.py`: `GET /api/executors`.

Use plain dict responses from dataclasses with `dataclasses.asdict()`.

Write `novasight/api/__init__.py`:

```python
from .app import create_app

__all__ = ["create_app"]
```

- [ ] **Step 5: Wire CLI to Uvicorn**

Replace `novasight/main.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_runtime_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("novasight")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default="config/novasight.yaml")
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_runtime_config(args.config)
    host = args.host or cfg.web.host
    port = args.port or cfg.web.port
    app = create_app(data_dir=Path(args.data_dir), config_path=Path(args.config))
    uvicorn.run(app, host=host, port=port)
    return 0
```

- [ ] **Step 6: Run API tests**

Run:

```bash
pytest tests/test_api.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Run backend smoke test**

Run:

```bash
python -m novasight --host 127.0.0.1 --port 5174
```

Expected: Uvicorn starts and logs that it is running on `http://127.0.0.1:5174`.
Stop it with `Ctrl-C`.

- [ ] **Step 8: Commit**

```bash
git add novasight/runtime novasight/api novasight/main.py tests/test_api.py
git commit -m "feat: expose management api"
```

---

### Task 6: Lightweight Web Console

**Files:**
- Create: `web/package.json`
- Create: `web/index.html`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/api.ts`
- Create: `web/src/styles.css`

- [ ] **Step 1: Create Vite package**

Write `web/package.json`:

```json
{
  "name": "novasight-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 5173",
    "build": "tsc && vite build",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^5.4.0",
    "typescript": "^5.5.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {}
}
```

- [ ] **Step 2: Create Vite/TypeScript config**

Write `web/index.html`, `web/tsconfig.json`, and `web/vite.config.ts`.

`vite.config.ts` must proxy `/api` and `/healthz` to `http://127.0.0.1:5174`.

- [ ] **Step 3: Implement API client**

Write `web/src/api.ts`:

```ts
export type RuntimeState = {
  running: boolean;
  source: string;
  active_model: null | Record<string, unknown>;
  executor: {
    selected: string;
    executors: Record<string, { available: boolean }>;
  };
};

export type PluginInfo = {
  plugin_id: string;
  kind: string;
  enabled: boolean;
};

export async function getRuntimeState(): Promise<RuntimeState> {
  const res = await fetch("/api/runtime/state");
  if (!res.ok) throw new Error(`runtime state failed: ${res.status}`);
  return res.json();
}

export async function getPlugins(): Promise<PluginInfo[]> {
  const res = await fetch("/api/plugins");
  if (!res.ok) throw new Error(`plugins failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 4: Implement console UI**

Write `web/src/App.tsx` with:

- Top navigation tabs: Dashboard, Live View, Models, Plugins, Settings.
- Dashboard cards for source, executor, active model, and executor availability.
- Plugins page showing built-in plugins from `/api/plugins`.
- Models, Live View, and Settings pages as operational empty states that name the intended workflow without marketing copy.
- No landing page and no nested card-heavy marketing layout.

Write `web/src/styles.css` with a restrained console design:

- Neutral background.
- Compact tabs.
- Dense status panels.
- Cards radius at or below `8px`.
- No gradient orb or decorative hero.

- [ ] **Step 5: Typecheck/build**

Run:

```bash
pnpm --dir web install
pnpm --dir web typecheck
pnpm --dir web build
```

Expected: typecheck and build succeed.

If network is unavailable and dependencies cannot be installed, request escalation for `pnpm --dir web install`.

- [ ] **Step 6: Commit**

```bash
git add web
git commit -m "feat: add lightweight web console"
```

---

### Task 7: Phase 1 Integration Verification

**Files:**
- Modify: `README.md`
- Optional modify: `.gitignore` if generated files reveal missing ignore rules

- [ ] **Step 1: Run Python test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run web checks**

Run:

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

Expected: both commands pass.

- [ ] **Step 3: Run backend and manually smoke API**

Start backend:

```bash
python -m novasight --host 127.0.0.1 --port 5174
```

In a separate shell:

```bash
curl -s http://127.0.0.1:5174/healthz
curl -s http://127.0.0.1:5174/api/runtime/state
curl -s http://127.0.0.1:5174/api/plugins
```

Expected:

- `/healthz` returns `{"ok": true}`.
- Runtime state reports `selected` executor as `dry_run`.
- Plugins include `vision.track_stats` and `control.center_target`.

- [ ] **Step 4: Update README with actual commands**

Update `README.md` with:

```markdown
## Run Backend

```bash
python -m novasight --host 127.0.0.1 --port 5174
```

## Run Web Console

```bash
pnpm --dir web install
pnpm --dir web dev
```

Open http://127.0.0.1:5173 during development. Vite proxies API calls to the
backend on port 5174.
```

- [ ] **Step 5: Commit verification/docs updates**

```bash
git add README.md .gitignore
git commit -m "docs: add phase 1 run instructions"
```

---

## Phase 1 Done Criteria

Phase 1 is complete when:

- `python -m novasight --host 127.0.0.1 --port 5174` starts the backend.
- `pytest -q` passes.
- `pnpm --dir web typecheck` passes.
- `pnpm --dir web build` passes.
- The Web UI can display runtime state and plugin state from the backend.
- Model registry can create projects, versions, artifacts, conversion jobs, publish a ready artifact, and roll back.
- Control plugins produce `ControlIntent` values and the selected executor handles them through `dry_run` by default.
