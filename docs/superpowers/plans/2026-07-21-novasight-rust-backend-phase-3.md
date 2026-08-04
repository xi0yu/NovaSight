# NovaSight Rust Backend Phase 3 Implementation Plan

> **Execution:** Each commit keeps the workspace buildable and tests green. No commit deletes a live caller before its replacement is in place.

**Goal:** Reshape the existing Phase 1 + Phase 2 Rust workspace into the long-term shape that matches the Jetson daemon target: one `novasightd` process owning an `Application` that hosts a `RuntimeSupervisor`, a `PipelineRuntime` running on dedicated `std::thread` workers, and an HTTP + WebSocket + Unix Domain Socket transport. Phase 2 algorithm contracts, fixtures, and 139 tests are preserved untouched.

**Scope boundary:** Python code, frontend, native C++ code, and the DeepStream bridge prototype stay out of scope. This plan is pure Rust reshuffling plus the empty skeleton of the long-term architecture.

**Compatibility rule:** Every commit must satisfy `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo fmt --all -- --check`. The `relink_server` binary and `novasight-store` crate remain in the workspace as legacy until the very last step.

---

## Architecture summary

```text
Jetson (single machine)
├── /run/novasight/novasightd.sock   (Unix Domain Socket for CLI)
└── novasightd
    └── HTTP 0.0.0.0:9070 + Web /api/v1/events
```

```text
novasightctl
└── Unix Domain Socket (no HTTP fallback in v1)
```

```text
novasightd internal:
Tokio runtime        ── Application (config + supervisor + api)
                         │
                         ├─ RuntimeSupervisor  (mpsc<RuntimeCommand> + oneshot reply)
                         │     └─ PipelineRuntime (4 workers on std::thread)
                         │           ├─ CaptureWorker
                         │           ├─ InferenceWorker
                         │           ├─ ControlWorker
                         │           └─ DeviceWorker
                         │
                         └─ ApiServer (TCP + Unix, same Router, same DTO)
```

---

## Final crate layout

```text
rust/
├── Cargo.toml                     # workspace members
├── rust-toolchain.toml
│
├── apps/
│   ├── novasightd/                # new: long-running daemon
│   └── novasightctl/              # new: CLI client
│
├── crates/
│   ├── novasight-core/            # keep + extend
│   ├── novasight-runtime/          # new
│   ├── novasight-api/              # keep
│   ├── novasight-client/           # new
│   ├── novasight-pipeline/         # new
│   ├── novasight-platform-jetson/  # keep
│   ├── novasight-store/            # legacy in this phase; renamed next phase
│   └── novasight-deepstream-bridge/# remove
│
├── bins/
│   └── relink_server/             # legacy in this phase; remove at end
│
├── fixtures/                      # keep (Phase 2 fixtures live here)
│
└── ...
```

Final dependency graph:

```text
apps/novasightd
    ├── novasight-runtime
    ├── novasight-api
    └── novasight-platform-jetson

apps/novasightctl
    └── novasight-client

novasight-api
    ├── novasight-core
    └── novasight-runtime

novasight-runtime
    ├── novasight-core
    ├── novasight-pipeline
    └── novasight-store   (legacy during phase; novasight-config in next phase)

novasight-pipeline
    └── novasight-core

novasight-client
    └── novasight-core
```

---

## Commit plan

### Commit 1 — Empty skeleton, no behavior change

**Add files:**
- `rust/apps/novasightd/Cargo.toml`
- `rust/apps/novasightd/src/main.rs` (empty fn main returning Ok(()))
- `rust/apps/novasightctl/Cargo.toml`
- `rust/apps/novasightctl/src/main.rs` (empty fn main returning Ok(()))
- `rust/crates/novasight-runtime/Cargo.toml` (empty lib, depends on core only)
- `rust/crates/novasight-runtime/src/lib.rs`
- `rust/crates/novasight-client/Cargo.toml` (empty lib, depends on core only)
- `rust/crates/novasight-client/src/lib.rs`
- `rust/crates/novasight-pipeline/Cargo.toml` (empty lib, depends on core only)
- `rust/crates/novasight-pipeline/src/lib.rs`

**Modify:**
- `rust/Cargo.toml` — add the 5 new members to `members` while keeping all existing members

**Verification:**
- `cargo build --workspace` succeeds
- `cargo test --workspace` passes (no test count change)
- `cargo clippy --workspace --all-targets -- -D warnings` clean
- Old `relink_server` binary still runs
- Old `novasight-store` tests still pass

**Commit message:**
`chore(workspace): add empty apps/novasightd, apps/novasightctl, runtime, client, pipeline skeletons`

### Commit 2 — RuntimeSupervisor + RuntimeHandle

**Add files:**
- `rust/crates/novasight-runtime/src/lib.rs` — `pub mod config; pub mod snapshot; pub mod command; pub mod supervisor;`
- `rust/crates/novasight-runtime/src/snapshot.rs` — `RuntimeSnapshot`, `DaemonSnapshot`, `PipelineSnapshot`, `SubsystemSnapshots`, `SubsystemSnapshot`, all enums
- `rust/crates/novasight-runtime/src/command.rs` — `RuntimeCommand` enum with `oneshot::Sender<Reply>` per variant
- `rust/crates/novasight-runtime/src/supervisor.rs` — `RuntimeSupervisor`, `RuntimeHandle` (Clone), supervisor loop
- `rust/crates/novasight-runtime/src/config.rs` — `AppConfig` (placeholder, no behavior yet)

**Tests:** add to `rust/crates/novasight-runtime/tests/`:
- `runtime_command_round_trip.rs` — Start/Stop/Restart/EmergencyStop each reply with snapshot
- `supervisor_shutdowns_on_daemon_command.rs` — ShutdownDaemon drops the loop
- `runtime_handle_is_clone_send_sync.rs` — handle can be cloned across tasks

**Verification:**
- Old test count unchanged (runtime crate has only new tests, no test removed)
- Old `relink_server` and `novasight-store` still untouched

**Commit message:**
`feat(runtime): add RuntimeSupervisor and RuntimeHandle with mpsc command/oneshot reply`

### Commit 3 — Fake Pipeline + LatestSlot + std::thread workers

**Add files:**
- `rust/crates/novasight-pipeline/src/lib.rs` — `pub mod worker; pub mod slot; pub mod pipeline;`
- `rust/crates/novasight-pipeline/src/slot.rs` — `LatestSlot<T>` per the proposal (RwLock + AtomicU64 version)
- `rust/crates/novasight-pipeline/src/worker.rs` — `Worker` trait
- `rust/crates/novasight-pipeline/src/pipeline.rs` — `PipelineRuntime`, `PipelineState`
- `rust/crates/novasight-pipeline/src/fake/` (mod) — `FakeCapture`, `FakeInference`, `FakeControl`, `FakeDevice`, each implementing `Worker` and producing/consuming typed slot values

**Modify:**
- `rust/crates/novasight-runtime/src/supervisor.rs` — supervisor loop now owns a `PipelineRuntime` and starts/stops it in response to `Start`/`Stop`/`Restart` commands

**Tests:** add to `rust/crates/novasight-pipeline/tests/`:
- `latest_slot_replaces_and_clamps_version.rs`
- `fake_pipeline_start_stop_round_trip.rs`
- `worker_stop_drains_its_thread.rs`
- `pipeline_state_machine_transitions.rs`

**Verification:**
- Old test count unchanged
- `cargo test -p novasight-pipeline` passes with new fake pipeline tests
- Old `relink_server` and `novasight-store` still untouched

**Commit message:**
`feat(pipeline): add LatestSlot, Worker trait, and Fake pipeline on std::thread`

### Commit 4 — `novasightd` Application

**Add files:**
- `rust/crates/novasight-runtime/src/application.rs` — `Application` struct, `bootstrap()`, `run_until_shutdown()`, `run_preflight_check()`

**Modify:**
- `rust/apps/novasightd/src/main.rs` — clap args, init logging, call `Application::bootstrap(&args.config).await?`, branch on `--check`, otherwise `run_until_shutdown().await`
- `rust/crates/novasight-runtime/src/lib.rs` — re-export `Application`

**Tests:** add to `rust/crates/novasight-runtime/tests/`:
- `application_bootstrap_loads_minimal_config.rs`
- `application_check_exits_with_zero_on_pass.rs`
- `application_run_until_shutdown_aborts_on_signal.rs` (uses tokio signal)

**Verification:**
- Old test count unchanged
- `novasightd --help` prints usage
- `novasightd --config /tmp/missing.toml` exits non-zero
- Old `relink_server` and `novasight-store` still untouched

**Commit message:**
`feat(daemon): wire novasightd main to Application bootstrap`

### Commit 5 — TCP API + WebSocket

**Add files:**
- `rust/crates/novasight-runtime/src/api_server.rs` — `ApiServer::new(handle, web_root, bind_addr)`, `run()` (TCP only this commit), `build_router()` (the shared router)
- `rust/crates/novasight-runtime/src/api/` — handler modules:
  - `status.rs` — `GET /api/v1/status`
  - `runtime.rs` — `POST /api/v1/runtime/{start,stop,emergency-stop,restart}`
  - `events.rs` — `GET /api/v1/events` (WebSocket)
  - `metrics.rs` — `GET /api/v1/metrics/current`
  - `static_files.rs` — `ServeDir` for `/opt/novasight/current/web`

**Modify:**
- `rust/crates/novasight-runtime/src/application.rs` — construct `ApiServer` from `RuntimeHandle`, spawn as `tokio::spawn` task in `run_until_shutdown`

**Tests:** add to `rust/crates/novasight-runtime/tests/`:
- `api_status_reflects_supervisor_state.rs`
- `api_runtime_start_returns_running_snapshot.rs`
- `api_runtime_stop_returns_stopped_snapshot.rs`
- `api_events_websocket_streams_state_changes.rs`
- `api_static_files_serves_index_html.rs`

**Verification:**
- `novasightd --config config/novasightd.example.yaml &` then `curl http://127.0.0.1:9070/api/v1/status` returns a JSON snapshot
- `curl -X POST http://127.0.0.1:9070/api/v1/runtime/start` returns a Running snapshot
- Old test count unchanged
- Old `relink_server` and `novasight-store` still untouched

**Commit message:**
`feat(api): expose supervisor via TCP HTTP + WebSocket on :9070`

### Commit 6 — Unix Socket + novasightctl

**Add files:**
- `rust/crates/novasight-runtime/src/transport.rs` — `TlsOrUnix` enum + `bind(router, listener)` adapter wrapping `axum::serve` over either TcpListener or tokio UnixListener
- `rust/crates/novasight-client/Cargo.toml` — add `tokio`, `serde`, `serde_json`, `anyhow` deps
- `rust/crates/novasight-client/src/lib.rs` — `NovaSightClient` (Unix Socket only in v1) with `status/start/stop/restart/emergency_stop/diagnose` methods
- `rust/apps/novasightctl/src/main.rs` — clap subcommand `Status|Start|Stop|Restart|EmergencyStop|Diagnose`, default socket `/run/novasight/novasightd.sock`

**Modify:**
- `rust/crates/novasight-runtime/src/application.rs` — bind both TCP and Unix listener on the same Router; pre-create `/run/novasight/` with safe perms on bootstrap

**Tests:** add:
- `rust/crates/novasight-client/tests/unix_socket_round_trip.rs`
- `rust/crates/novasight-runtime/tests/api_unix_socket_returns_status.rs`
- `rust/apps/novasightctl/tests/cli_parses_subcommands.rs`

**Verification:**
- `novasightd &` then `novasightctl status` returns the same JSON as `curl /api/v1/status`
- `novasightctl start` → `curl /api/v1/status` shows Running
- Old test count unchanged
- Old `relink_server` and `novasight-store` still untouched

**Commit message:**
`feat(daemon): expose supervisor over Unix Domain Socket for CLI`

### Commit 7 — Rename `novasight-store` → `novasight-config`

**Note:** This commit only renames the directory and crate name. It does **not** change the YAML format, the xattr/atomic-save behavior, or the call sites of the config repository. Format change is deferred to a later phase.

**Steps:**
1. `git mv rust/crates/novasight-store rust/crates/novasight-config`
2. Update `Cargo.toml` `name` field to `novasight-config`
3. Update `path = "../novasight-store"` references to `path = "../novasight-config"` everywhere
4. Update `extern crate novasight_store` and `use novasight_store::` references to `novasight_config::`
5. Update documentation references

**Verification:**
- `cargo test --workspace` passes
- Old test count unchanged (rename only)
- YAML format unchanged
- Atomic save / xattr / revision logic unchanged

**Commit message:**
`refactor(workspace): rename novasight-store to novasight-config (no behavior change)`

### Commit 8 — `Application` reads config via `novasight-config`

**Modify:**
- `rust/crates/novasight-runtime/src/application.rs` — `Application::bootstrap` calls `novasight_config::load(&path)?` instead of inline config loading
- `rust/crates/novasight-runtime/src/config.rs` — `AppConfig` now composes a `novasight_config::repository::YamlConfigRepository` snapshot

**Tests:** add:
- `application_bootstrap_reads_config_from_disk.rs`

**Verification:**
- Old test count unchanged
- `novasightd --config /tmp/missing.toml` exits non-zero with the same error as before
- `novasightd --check` runs preflight via the config repository

**Commit message:**
`feat(runtime): wire Application to the novasight-config repository`

### Commit 9 — Preflight check

**Add files:**
- `rust/crates/novasight-runtime/src/preflight.rs` — `PreflightReport { items: Vec<PreflightItem> }`, `run_preflight(config) -> PreflightReport`
- Each `PreflightItem` has `name: String`, `severity: Severity (Pass|Warn|Fail)`, `message: String`

**Checklist (per proposal section 3):**
- Config parse and validate
- Architecture is aarch64
- Jetson model in known list
- JetPack / L4T version detectable
- CUDA runtime / library present
- TensorRT library present
- GStreamer + NVIDIA plugins present
- Model file exists and is readable
- Web static root exists and is readable
- Unix socket parent dir exists or is creatable
- Data dir + log dir writable
- Capture device exists (Warn if missing)
- Output device / library exists (Warn if missing)

**Modify:**
- `rust/crates/novasight-runtime/src/application.rs` — `run_preflight_check()` calls `preflight::run_preflight(&self.config)`, prints human-readable summary, returns Ok(()) if no Fail items even with Warn items

**Tests:** add:
- `preflight_passes_when_config_loaded_and_dirs_writable.rs`
- `preflight_warns_when_capture_device_missing.rs`
- `preflight_fails_when_config_path_is_unreadable.rs`

**Verification:**
- Old test count unchanged
- `novasightd --check` exits 0 with Pass items, 0 with mixed Pass+Warn, non-zero with any Fail
- Old `relink_server` and `novasight-store` still untouched

**Commit message:**
`feat(daemon): add preflight check with Pass/Warn/Fail severity`

### Commit 10 — Phase 2 algorithm wiring into Fake Pipeline

**Modify:**
- `rust/crates/novasight-pipeline/src/fake/` — `FakeCapture` produces `FrameMeta`-shaped values; `FakeInference` runs through the existing `TargetingCore` + `ContinuousControl` from `novasight-core`; `FakeControl` quantizes via `PerAxisQuantizer`; `FakeDevice` writes to a `LatestCommandSlot`
- `rust/crates/novasight-pipeline/src/pipeline.rs` — `PipelineRuntime::start` now wires the four fake workers together via typed `LatestSlot<T>` between stages

**Tests:** add:
- `fake_pipeline_end_to_end_through_phase2_algorithms.rs` — confirms the Fake Pipeline produces typed `DeviceCommand` outputs through the algorithm stack

**Verification:**
- Old 139 algorithm tests still pass
- `cargo test --workspace` adds 1+ new test, no test removed
- Phase 2 fixture files are still consumed by `core` tests

**Commit message:**
`feat(pipeline): wire Phase 2 algorithm slice into Fake Pipeline workers`

### Commit 11 — Remove `novasight-deepstream-bridge`

**Steps:**
1. `git rm rust/crates/novasight-deepstream-bridge/`
2. Remove the member from `rust/Cargo.toml` `members`
3. Remove references in any top-level docs

**Verification:**
- `cargo test --workspace` passes
- `cargo build --workspace` succeeds
- `git grep novasight_deepstream_bridge` returns nothing

**Commit message:**
`remove unused DeepStream bridge prototype`

### Commit 12 — Remove `bins/relink_server`

**Steps:**
1. `git rm -r rust/bins/relink_server/`
2. Remove `bins/relink_server` from `rust/Cargo.toml` `members`
3. Update any docs that reference the old binary path
4. `novasightd` now is the only server

**Verification:**
- `cargo test --workspace` passes
- `cargo build --workspace` succeeds
- `novasightd --help` works
- The `RuntimeSnapshot`, `RuntimeCommand`, `RuntimeSupervisor` tests cover the daemon start/stop path

**Commit message:**
`remove(relay): drop relink_server now that novasightd owns the surface`

### Commit 13 — Full validation gate

This commit has no code changes; it is the final verification snapshot.

**Steps:**
- `cargo test --workspace` → all green
- `cargo clippy --workspace --all-targets -- -D warnings` → clean
- `cargo fmt --all -- --check` → clean
- `cargo bench -p novasight-pipeline` (if a bench is added) → numbers in the gate report
- `pnpm --dir web build` → 301 modules transformed
- Build `novasightd`, run it, exercise every CLI subcommand via `novasightctl`, then SIGTERM it cleanly
- Confirm no DeepStream / GStreamer / CUDA / TensorRT / kmNet crates got pulled into `novasight-pipeline` or `novasight-runtime`
- Re-run the Phase 2 parity script and confirm all 5 fixtures still pass

**Verification:** recorded in the final commit message body.

**Commit message:**
`chore(phase3): full validation gate report`

---

## Phase 3 Definition of Done

- `novasightd` is the only long-running server binary in the workspace
- `novasightctl` reaches the daemon over a Unix Domain Socket
- TCP HTTP and WebSocket are exposed on `0.0.0.0:9070` with the same Router, DTO, and RuntimeHandle
- `RuntimeSupervisor` is the only owner of `PipelineRuntime` lifecycle
- `PipelineRuntime` runs on dedicated `std::thread` workers; Tokio only owns the control plane
- `LatestSlot<T>` is the only data flow between capture and inference, and between inference and control
- HID commands flow over a small bounded `sync_channel`
- `novasight-config` (formerly `novasight-store`) keeps its atomic save, xattr protection, and revision behavior; format is unchanged in this phase
- Phase 2 algorithms stay in `novasight-core`; the Fake Pipeline calls them through the same `LatestSlot<T>` interface the real workers will use
- All 139 Phase 2 contract tests pass
- `preflight` reports `Pass` / `Warn` / `Fail` with the checklist from the proposal
- No `novasight-deepstream-bridge` or `relink_server` remains in the workspace
