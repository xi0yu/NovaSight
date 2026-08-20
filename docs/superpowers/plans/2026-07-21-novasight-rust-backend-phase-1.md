# NovaSight Rust Backend Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the root `rust/` workspace with a production-shaped `relink_server` binary and a Mac-runnable replay slice that preserves the current health/runtime HTTP and status WebSocket contracts without taking ownership away from the Python production entry.

**Architecture:** `novasight-core` owns typed runtime state and the replay pipeline; `novasight-api` projects immutable snapshots through Axum; `novasight-store` loads deployment configuration; `novasight-platform-jetson` provides platform clocks now and vendor adapters later; `rust/bins/relink_server` is the only Rust composition root. Phase 1 is replay/DryRun only and cannot send kmNet commands.

**Tech Stack:** Rust 2024, Tokio 1, Axum 0.8, Serde/serde_yaml, thiserror 2, UUID 1, tower-http 0.6, tracing 0.1, clap 4, async-trait 0.1.

## Global Constraints

- Preserve all Python/C++ source, tests, tools, YAML, SQLite, model and License files.
- Add all production code to the existing root `rust/` workspace; do not expand `base/relink_server`.
- `novasight-core` must not depend on Axum, SQLite, GStreamer, CUDA, TensorRT or kmNet.
- `relink_server` remains replay/DryRun until later Jetson gates; it must never send a live device command in Phase 1.
- Keep current paths `/healthz`, `/api/runtime/state`, `/api/runtime/start`, `/api/runtime/stop` and `/ws/status`.
- Keep current `RuntimeState` required fields: `running`, `source`, `active_model`, `executor`, `capture`, `inference`, `config`, `pipeline`, `fatal_error`.
- Use `CLOCK_MONOTONIC` semantics for runtime timing; wall clock is not used for frame age or expiry.
- All channels are bounded; snapshots use Tokio `watch` capacity-one semantics.
- Do not add placeholders, fake success paths or hardware no-ops. The Phase 1 source and device are explicit replay and recording adapters.

---

## Target File Map

```text
rust/
├── Cargo.toml                                      # workspace members and shared dependency versions
├── config/novasightd.example.yaml                 # deploy-time server/replay/path configuration
├── bins/relink_server/
│   ├── Cargo.toml                                  # binary dependencies
│   └── src/
│       ├── main.rs                                 # CLI and process exit
│       ├── bootstrap.rs                            # composition root
│       └── shutdown.rs                             # Ctrl-C/SIGTERM coordination
└── crates/
    ├── novasight-core/
    │   ├── Cargo.toml
    │   ├── src/
    │   │   ├── lib.rs                              # public module surface
    │   │   ├── error.rs                            # typed core errors
    │   │   ├── ports.rs                            # Clock/PerceptionSource/PointerDevice seams
    │   │   ├── perception/{mod.rs,types.rs,replay.rs}
    │   │   ├── targeting/mod.rs                    # deterministic nearest-center replay targeting
    │   │   ├── control/mod.rs                      # explicit proportional replay algorithm
    │   │   ├── output/mod.rs                       # typed command and recording receipt
    │   │   ├── telemetry/mod.rs                    # immutable OperationalSnapshot
    │   │   └── runtime/{mod.rs,state.rs,manager.rs,session.rs}
    │   └── tests/{replay_slice.rs,runtime_lifecycle.rs}
    ├── novasight-api/
    │   ├── Cargo.toml
    │   ├── src/
    │   │   ├── lib.rs
    │   │   ├── app.rs                              # Router assembly
    │   │   ├── state.rs                            # ApiState handles only
    │   │   ├── error.rs                            # AppError to HTTP mapping
    │   │   ├── dto/{mod.rs,runtime.rs}
    │   │   ├── routes/{mod.rs,health.rs,runtime.rs}
    │   │   └── websocket/{mod.rs,status.rs}
    │   └── tests/{runtime_contract.rs,status_websocket.rs}
    ├── novasight-store/
    │   ├── Cargo.toml
    │   ├── src/lib.rs
    │   ├── src/config/{mod.rs,model.rs,repository.rs}
    │   └── tests/config_repository.rs
    └── novasight-platform-jetson/
        ├── Cargo.toml
        ├── src/lib.rs
        ├── src/clock.rs
        └── tests/clock.rs
```

---

### Task 1: Lock Workspace Boundaries and Shared Dependencies

**Files:**
- Modify: `rust/Cargo.toml`
- Create: `rust/crates/novasight-core/Cargo.toml`
- Create: `rust/crates/novasight-core/src/lib.rs`
- Create: `rust/crates/novasight-api/Cargo.toml`
- Create: `rust/crates/novasight-api/src/lib.rs`
- Create: `rust/crates/novasight-store/Cargo.toml`
- Create: `rust/crates/novasight-store/src/lib.rs`
- Create: `rust/crates/novasight-platform-jetson/Cargo.toml`
- Create: `rust/crates/novasight-platform-jetson/src/lib.rs`

**Interfaces:**
- Produces workspace package names `novasight-core`, `novasight-api`, `novasight-store`, and `novasight-platform-jetson`.
- Existing package `novasight-deepstream-bridge` remains unchanged and in the workspace.

- [ ] **Step 1: Replace the workspace manifest with explicit members and shared dependencies**

```toml
[workspace]
members = [
  "crates/novasight-api",
  "crates/novasight-core",
  "crates/novasight-deepstream-bridge",
  "crates/novasight-platform-jetson",
  "crates/novasight-store",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Proprietary"

[workspace.dependencies]
async-trait = "0.1"
axum = { version = "0.8", features = ["ws"] }
clap = { version = "4", features = ["derive"] }
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
thiserror = "2"
tokio = { version = "1", features = ["macros", "net", "rt-multi-thread", "signal", "sync", "time"] }
tower = { version = "0.5", features = ["util"] }
tower-http = { version = "0.6", features = ["cors", "trace"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
uuid = { version = "1", features = ["serde", "v4"] }
```

- [ ] **Step 2: Create the four library package manifests with path-only internal dependencies**

`novasight-core` depends only on `async-trait`, `serde`, `thiserror`, `tokio`, and `uuid`. `novasight-api` depends on Axum/Tower plus `novasight-core`. `novasight-store` depends on Serde/YAML plus `novasight-core`. `novasight-platform-jetson` depends on `novasight-core`.

- [ ] **Step 3: Create minimal library roots**

```rust
// crates/novasight-core/src/lib.rs
#![forbid(unsafe_code)]

// crates/novasight-api/src/lib.rs
#![forbid(unsafe_code)]

// crates/novasight-store/src/lib.rs
#![forbid(unsafe_code)]

// crates/novasight-platform-jetson/src/lib.rs
#![deny(unsafe_op_in_unsafe_fn)]
```

- [ ] **Step 4: Run workspace metadata and check**

Run: `cargo metadata --no-deps --format-version 1` from `rust/`.  
Expected: the existing bridge and four new library packages are listed exactly once.

Run: `cargo check --workspace`.  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/crates/novasight-{core,api,store,platform-jetson}
git commit -m "build(rust): establish NovaSight backend workspace"
```

### Task 2: Implement Typed Replay Domain Slice

**Files:**
- Create: `rust/crates/novasight-core/src/error.rs`
- Create: `rust/crates/novasight-core/src/ports.rs`
- Create: `rust/crates/novasight-core/src/perception/mod.rs`
- Create: `rust/crates/novasight-core/src/perception/types.rs`
- Create: `rust/crates/novasight-core/src/perception/replay.rs`
- Create: `rust/crates/novasight-core/src/targeting/mod.rs`
- Create: `rust/crates/novasight-core/src/control/mod.rs`
- Create: `rust/crates/novasight-core/src/output/mod.rs`
- Create: `rust/crates/novasight-core/tests/replay_slice.rs`
- Modify: `rust/crates/novasight-core/src/lib.rs`

**Interfaces:**
- Produces `FrameStamp`, `Detection`, `DetectionBatch`, `SelectedTarget`, `ControlDecision`, `DeviceCommand`, `DeviceReceipt`.
- Produces seams `Clock`, `PerceptionSource`, and `PointerDevice`.
- Produces `ReplayPerceptionSource`, `NearestCenterTargeting`, `ProportionalReplayControl`, and `RecordingPointerDevice`.

- [ ] **Step 1: Write a failing end-to-end replay test**

```rust
#[tokio::test]
async fn replay_batch_becomes_recorded_device_receipt() {
    let batch = DetectionBatch::fixture(
        FrameStamp::new(RuntimeEpoch(1), 7, 1_000_000_000),
        640,
        640,
        vec![Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9).unwrap()],
    ).unwrap();
    let target = NearestCenterTargeting::default().select(&batch).expect("target");
    let decision = ProportionalReplayControl::new(1.0)
        .decide(&batch, &target, 1_010_000_000)
        .expect("decision");
    let device = RecordingPointerDevice::default();
    let receipt = device.send(decision.into_command()).expect("receipt");

    assert_eq!(receipt.epoch, RuntimeEpoch(1));
    assert_eq!(receipt.generation, 7);
    assert_eq!(device.receipts().len(), 1);
}
```

- [ ] **Step 2: Run the test and verify missing symbols**

Run: `cargo test -p novasight-core --test replay_slice`.  
Expected: FAIL because domain modules do not exist.

- [ ] **Step 3: Implement bounded, validated domain types**

Use newtypes for `RuntimeEpoch`, `Generation`, and `MonotonicNanos`. `Detection::new` rejects non-finite values, non-positive dimensions, and confidence outside `0.0..=1.0`. `DetectionBatch::new` rejects zero geometry, more than 256 detections, and mismatched coordinate-space geometry.

```rust
pub const MAX_DETECTIONS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct RuntimeEpoch(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Generation(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct MonotonicNanos(pub u64);
```

- [ ] **Step 4: Implement real transition adapters**

`ReplayPerceptionSource` owns a finite `VecDeque<DetectionBatch>` and returns batches in order. `RecordingPointerDevice` records every attempted command and returns a receipt. Neither is a placeholder: both are deterministic adapters used by replay acceptance tests and later regression fixtures.

```rust
#[async_trait]
pub trait PerceptionSource: Send {
    async fn next_batch(&mut self) -> Result<Option<DetectionBatch>, AppError>;
}

pub trait PointerDevice: Send + Sync {
    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError>;
}
```

- [ ] **Step 5: Implement deterministic replay targeting and control**

`NearestCenterTargeting` chooses the detection whose center has the smallest squared distance to the declared batch center, breaking ties by object ID. `ProportionalReplayControl` is explicitly named and scoped to replay; it computes integer counts from target center delta and configured gain. It is not used as the final production control algorithm.

- [ ] **Step 6: Run core tests**

Run: `cargo test -p novasight-core --test replay_slice`.  
Expected: PASS with one test.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/novasight-core
git commit -m "feat(core): add typed replay perception-to-device slice"
```

### Task 3: Add RuntimeManager, RuntimeSession, and Immutable Snapshots

**Files:**
- Create: `rust/crates/novasight-core/src/telemetry/mod.rs`
- Create: `rust/crates/novasight-core/src/runtime/mod.rs`
- Create: `rust/crates/novasight-core/src/runtime/state.rs`
- Create: `rust/crates/novasight-core/src/runtime/manager.rs`
- Create: `rust/crates/novasight-core/src/runtime/session.rs`
- Create: `rust/crates/novasight-core/tests/runtime_lifecycle.rs`
- Modify: `rust/crates/novasight-core/src/lib.rs`

**Interfaces:**
- Produces `RuntimeHandle::start`, `RuntimeHandle::stop`, `RuntimeHandle::snapshot`, and `RuntimeHandle::subscribe`.
- Produces `OperationalSnapshot`, `RuntimePhase`, `RunIntent`, `RuntimeCommandReceipt`, and `RuntimeManager::spawn`.
- Consumes the replay source/control/device interfaces from Task 2.

- [ ] **Step 1: Write lifecycle tests for idempotency and epoch isolation**

```rust
#[tokio::test]
async fn repeated_start_keeps_one_session_and_stop_blocks_old_receipts() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture());
    let first = runtime.start().await.unwrap();
    let repeated = runtime.start().await.unwrap();
    assert_eq!(first.epoch, repeated.epoch);
    assert_eq!(runtime.snapshot().phase, RuntimePhase::Running);

    runtime.stop().await.unwrap();
    let stopped = runtime.snapshot();
    assert_eq!(stopped.phase, RuntimePhase::Stopped);
    assert!(!stopped.running);
}

#[tokio::test]
async fn restart_allocates_new_epoch() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture());
    let first = runtime.start().await.unwrap();
    runtime.stop().await.unwrap();
    let second = runtime.start().await.unwrap();
    assert!(second.epoch > first.epoch);
}
```

- [ ] **Step 2: Verify tests fail**

Run: `cargo test -p novasight-core --test runtime_lifecycle`.  
Expected: FAIL because runtime modules do not exist.

- [ ] **Step 3: Implement state and snapshot types**

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePhase { Stopped, Starting, Running, Standby, Stopping, Faulted }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunIntent { Stopped, Running }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationalSnapshot {
    pub phase: RuntimePhase,
    pub run_intent: RunIntent,
    pub epoch: Option<RuntimeEpoch>,
    pub running: bool,
    pub source: String,
    pub last_generation: Option<Generation>,
    pub processed_batches: u64,
    pub device_receipts: u64,
    pub fatal_error: Option<ErrorSnapshot>,
}
```

- [ ] **Step 4: Implement a bounded command actor and capacity-one snapshot watch**

`RuntimeManager::spawn` creates `mpsc::channel(16)` for commands and `watch::channel(Arc<OperationalSnapshot>)` for snapshots. Start is idempotent while Running. Stop closes the Session, waits for its task, clears the latest command, publishes Stopped, and only then acknowledges.

```rust
impl RuntimeHandle {
    pub async fn start(&self) -> Result<RuntimeCommandReceipt, AppError>;
    pub async fn stop(&self) -> Result<RuntimeCommandReceipt, AppError>;
    pub fn snapshot(&self) -> Arc<OperationalSnapshot>;
    pub fn subscribe(&self) -> watch::Receiver<Arc<OperationalSnapshot>>;
}
```

- [ ] **Step 5: Implement RuntimeSession replay ownership**

The session owns one source and one device lease. It stops through a cancellation watch, processes each batch in order, rejects wrong epochs, runs targeting/control, records the receipt, and publishes counters. No task outlives `RuntimeSession::stop`.

- [ ] **Step 6: Run lifecycle and full core tests**

Run: `cargo test -p novasight-core`.  
Expected: PASS; no test sleeps on wall time longer than 100ms.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/novasight-core
git commit -m "feat(core): add runtime manager epochs and snapshots"
```

### Task 4: Externalize Server, Replay, and Path Configuration

**Files:**
- Create: `rust/crates/novasight-store/src/config/mod.rs`
- Create: `rust/crates/novasight-store/src/config/model.rs`
- Create: `rust/crates/novasight-store/src/config/repository.rs`
- Create: `rust/crates/novasight-store/tests/config_repository.rs`
- Modify: `rust/crates/novasight-store/src/lib.rs`
- Create: `rust/config/novasightd.example.yaml`

**Interfaces:**
- Produces `AppConfig`, `ServerConfig`, `ReplayConfig`, `PathConfig`, and `YamlConfigRepository::load`.
- Preserves unconsumed legacy sections in `legacy: BTreeMap<String, serde_yaml::Value>` for later typed migrations; it does not discard them.

- [ ] **Step 1: Write configuration compatibility tests**

```rust
#[test]
fn loads_current_project_yaml_without_dropping_legacy_sections() {
    let project_config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../config/novasight.yaml");
    let config = YamlConfigRepository::load(project_config).unwrap();
    assert_eq!(config.server.port, 5174);
    assert!(config.legacy.contains_key("capture"));
    assert!(config.legacy.contains_key("inference"));
}

#[test]
fn missing_file_is_an_error_not_silent_defaults() {
    let error = YamlConfigRepository::load("fixtures/missing.yaml").unwrap_err();
    assert_eq!(error.code(), "CONFIG_NOT_FOUND");
}
```

- [ ] **Step 2: Verify tests fail**

Run: `cargo test -p novasight-store --test config_repository`.  
Expected: FAIL because config types do not exist.

- [ ] **Step 3: Implement typed deploy configuration with legacy preservation**

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub replay: ReplayConfig,
    #[serde(default)]
    pub paths: PathConfig,
    #[serde(flatten)]
    pub legacy: BTreeMap<String, serde_yaml::Value>,
}
```

Defaults preserve current deployment values: host `0.0.0.0`, port `5174`, data directory `data`, model directory `data/models`, database `data/novasight.db`, license `data/license.json`. The repository only applies defaults after a file is successfully opened and parsed.

- [ ] **Step 4: Add a complete example config**

```yaml
schema_version: 1
revision: 0
server:
  host: 0.0.0.0
  port: 5174
replay:
  enabled: true
  frame_interval_ms: 16
  output_gate_open: false
paths:
  data_dir: data
  model_dir: data/models
  database: data/novasight.db
  license: data/license.json
  python_executable: python3
```

- [ ] **Step 5: Run store tests**

Run: `cargo test -p novasight-store`.  
Expected: PASS and current `config/novasight.yaml` loads.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/novasight-store rust/config/novasightd.example.yaml
git commit -m "feat(store): load external Rust backend configuration"
```

### Task 5: Provide the Monotonic Platform Clock

**Files:**
- Create: `rust/crates/novasight-platform-jetson/src/clock.rs`
- Create: `rust/crates/novasight-platform-jetson/tests/clock.rs`
- Modify: `rust/crates/novasight-platform-jetson/src/lib.rs`

**Interfaces:**
- Implements `novasight_core::ports::Clock` as `SystemMonotonicClock`.
- Does not expose wall-clock values.

- [ ] **Step 1: Write a monotonicity test**

```rust
#[test]
fn system_clock_never_moves_backwards_in_sample_window() {
    let clock = SystemMonotonicClock::default();
    let first = clock.now();
    let second = clock.now();
    assert!(second >= first);
}
```

- [ ] **Step 2: Implement using a process-local Instant origin**

```rust
#[derive(Debug)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl Clock for SystemMonotonicClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
}
```

- [ ] **Step 3: Run the test and commit**

Run: `cargo test -p novasight-platform-jetson`.  
Expected: PASS.

```bash
git add rust/crates/novasight-platform-jetson
git commit -m "feat(platform): provide monotonic runtime clock"
```

### Task 6: Implement Compatible Axum Runtime HTTP Routes

**Files:**
- Create: `rust/crates/novasight-api/src/state.rs`
- Create: `rust/crates/novasight-api/src/error.rs`
- Create: `rust/crates/novasight-api/src/dto/mod.rs`
- Create: `rust/crates/novasight-api/src/dto/runtime.rs`
- Create: `rust/crates/novasight-api/src/routes/mod.rs`
- Create: `rust/crates/novasight-api/src/routes/health.rs`
- Create: `rust/crates/novasight-api/src/routes/runtime.rs`
- Create: `rust/crates/novasight-api/src/app.rs`
- Create: `rust/crates/novasight-api/tests/runtime_contract.rs`
- Modify: `rust/crates/novasight-api/src/lib.rs`

**Interfaces:**
- Produces `build_router(ApiState) -> Router`.
- Consumes only `RuntimeHandle` and immutable snapshots.
- Preserves the required fields consumed by `web/src/api.ts`.

- [ ] **Step 1: Write failing route contract tests**

```rust
#[tokio::test]
async fn health_matches_existing_contract() {
    let response = app().oneshot(Request::get("/healthz").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json(response).await, json!({"ok": true}));
}

#[tokio::test]
async fn runtime_state_contains_all_required_frontend_fields() {
    let body = get_json(app(), "/api/runtime/state").await;
    for key in ["running", "source", "active_model", "executor", "capture", "inference", "config", "pipeline", "fatal_error"] {
        assert!(body.get(key).is_some(), "missing {key}");
    }
}
```

- [ ] **Step 2: Verify tests fail**

Run: `cargo test -p novasight-api --test runtime_contract`.  
Expected: FAIL because `build_router` does not exist.

- [ ] **Step 3: Implement compatibility DTOs without arbitrary core JSON**

`RuntimeStateResponse` is an API-only projection. Nested Phase 1 sections are typed structs serialized as objects; unsupported online hardware is reported explicitly as unavailable/replay, never as ready.

```rust
#[derive(Serialize)]
pub struct RuntimeStateResponse {
    pub running: bool,
    pub source: String,
    pub active_model: Option<serde_json::Value>,
    pub executor: ExecutorStatusResponse,
    pub capture: CaptureStateResponse,
    pub statistics: StatisticsResponse,
    pub inference: InferenceStateResponse,
    pub config: RuntimeConfigSummaryResponse,
    pub pipeline: PipelineStateResponse,
    pub power_saving: RuntimePowerSavingResponse,
    pub vision: VisionStateResponse,
    pub fatal_error: Option<ErrorResponse>,
}
```

- [ ] **Step 4: Implement handlers as command/query adapters**

- `GET /healthz` returns `{"ok": true}`.
- `GET /api/runtime/state` projects `RuntimeHandle::snapshot()`.
- `POST /api/runtime/start` accepts no request body, awaits `RuntimeHandle::start()`, and returns `204 No Content`; `GET /api/runtime/state` and `/ws/status` remain the only runtime-state authorities.
- `POST /api/runtime/stop` awaits complete stop then returns the full RuntimeState projection.

- [ ] **Step 5: Run route contracts**

Run: `cargo test -p novasight-api --test runtime_contract`.  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/novasight-api
git commit -m "feat(api): expose compatible runtime HTTP contracts"
```

### Task 7: Implement Capacity-One Status WebSocket

**Files:**
- Create: `rust/crates/novasight-api/src/websocket/mod.rs`
- Create: `rust/crates/novasight-api/src/websocket/status.rs`
- Create: `rust/crates/novasight-api/tests/status_websocket.rs`
- Modify: `rust/crates/novasight-api/src/app.rs`

**Interfaces:**
- Adds `GET /ws/status` WebSocket upgrade.
- Emits `RuntimeStatusFrame { kind: "runtime_snapshot", topic, full, state }`.
- Supports `full`, `summary`, `capture`, `infer`, `control`, and `latency`; unknown topics normalize to `full` for current compatibility.

- [ ] **Step 1: Write a serialization and slow-consumer contract test**

```rust
#[test]
fn status_frame_matches_frontend_envelope() {
    let frame = RuntimeStatusFrame::full(RuntimeStateResponse::fixture_stopped());
    let value = serde_json::to_value(frame).unwrap();
    assert_eq!(value["kind"], "runtime_snapshot");
    assert_eq!(value["topic"], "full");
    assert_eq!(value["full"], true);
    assert!(value["state"].is_object());
}
```

Also test a `watch` receiver that skips intermediate snapshots and observes the newest epoch/generation.

- [ ] **Step 2: Implement WebSocket projection from `RuntimeHandle::subscribe`**

On connect, send the current snapshot immediately. Then await `receiver.changed()` and serialize only the newest borrowed snapshot. Disconnect ends only that client task and never mutates Runtime.

- [ ] **Step 3: Run API tests and commit**

Run: `cargo test -p novasight-api`.  
Expected: PASS.

```bash
git add rust/crates/novasight-api
git commit -m "feat(api): stream latest runtime snapshots over WebSocket"
```

### Task 8: Compose and Smoke-Test the relink_server Binary

**Files:**
- Modify: `rust/Cargo.toml`
- Create: `rust/bins/relink_server/Cargo.toml`
- Create: `rust/bins/relink_server/src/main.rs`
- Create: `rust/bins/relink_server/src/bootstrap.rs`
- Create: `rust/bins/relink_server/src/shutdown.rs`
- Create: `rust/bins/relink_server/tests/cli.rs`

**Interfaces:**
- Produces CLI `relink_server --config <path>`.
- Loads configuration before binding.
- Creates `SystemMonotonicClock`, replay dependencies, `RuntimeManager`, `ApiState`, and Axum Router.
- Handles Ctrl-C and SIGTERM through graceful Axum shutdown and `RuntimeHandle::stop`.

- [ ] **Step 1: Write CLI failure and help tests**

```rust
#[test]
fn missing_config_exits_nonzero_with_stable_reason() {
    let output = Command::new(env!("CARGO_BIN_EXE_relink_server"))
        .args(["--config", "fixtures/missing.yaml"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("CONFIG_NOT_FOUND"));
}
```

- [ ] **Step 2: Implement typed CLI and bootstrap**

```rust
#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "config/novasight.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> ExitCode {
    match bootstrap::run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(code = error.code(), error = %error, "backend stopped");
            ExitCode::FAILURE
        }
    }
}
```

`bootstrap::run` validates `replay.enabled == true` and `output_gate_open == false` in Phase 1. Any attempt to open the output gate is a configuration error, not an ignored value.

- [ ] **Step 3: Implement graceful shutdown**

Wait for Ctrl-C on all platforms and SIGTERM on Unix. Trigger Axum graceful shutdown, call `RuntimeHandle::stop`, await acknowledgement, then return. Bind failures surface a stable `SERVER_BIND_FAILED` error.

- [ ] **Step 4: Run focused and workspace verification**

Run: `cargo test --workspace`.  
Expected: all bridge, core, store, platform, API, and binary tests PASS.

Run: `cargo clippy --workspace --all-targets -- -D warnings`.  
Expected: PASS with zero warnings.

Run: `cargo fmt --all -- --check`.  
Expected: PASS.

- [ ] **Step 5: Smoke-test the actual binary**

Run the binary with `rust/config/novasightd.example.yaml`, request `/healthz`, start runtime, read state, stop runtime, and connect once to `/ws/status`. Expected observations:

- health body exactly `{"ok":true}`;
- first start accepted and returns epoch `1`;
- repeated start returns epoch `1` and no second Session;
- state reports source `replay`, running `true`, and at least one processed batch;
- WebSocket envelope has `kind=runtime_snapshot`;
- stop returns running `false`;
- process exits cleanly after SIGTERM.

- [ ] **Step 6: Verify the existing frontend contract remains buildable**

Run: `pnpm --dir web build`.  
Expected: PASS; no changes to `web/src/api.ts` are required.

- [ ] **Step 7: Commit**

```bash
git add rust/bins/relink_server rust/Cargo.lock
git commit -m "feat(rust): ship replay-capable relink_server entry"
```

---

## Phase 1 Completion Gate

Phase 1 is complete only when:

- root `rust/` contains one workspace and one lockfile;
- `base/relink_server` and all Python/C++ files remain unchanged;
- `cargo test --workspace`, Clippy and rustfmt pass;
- the real binary completes the start/replay/snapshot/stop smoke path;
- `/healthz`, runtime state/start/stop and `/ws/status` satisfy current frontend contracts;
- duplicate start creates no second Session;
- stop acknowledges only after the Session is gone;
- output gate cannot be opened and no kmNet adapter is linked;
- current `config/novasight.yaml` loads without dropping untyped legacy sections;
- current frontend build passes.

After this gate, write a separate Phase 2 plan for geometry, freshness, TargetingCore and production control parity using captured Python replay fixtures. Do not begin DeepStream or kmNet integration inside Phase 1.
