# NovaSight Project Health Ledger

Last updated: 2026-08-10

This file is the current engineering-health authority for NovaSight. It records
the maintained product path, known risks, ownership boundaries, and the next
safe development work. Historical plans under `docs/superpowers/` are context,
not instructions, and must not override the current code or this ledger.

## Current Product Authority

NovaSight is a Jetson-first Rust application with a React Studio management UI.
The supported production path is:

```text
DeepStream capture
-> NVMM ROI crop/resize
-> TensorRT inference
-> DetectionBatch
-> target selection/tracking/prediction
-> continuous angular control
-> bounded device command
-> native kmNet output
```

Production behavior is owned by the Rust process. Python is not part of the
tracked product source or runtime. `novasightd` enables its `deepstream` feature
by default; non-Linux builds are development/reference environments and do not
prove the Jetson production composition.

The normal deliverable is the portable directory produced by
`novasight-packager`. Direct daemon startup, Vite development, reference native
libraries, and host-side tests are diagnostic/development paths.

## Current Workspace Snapshot

Measured from `develop-alpha` at `f87e6c7` before the development batch started
on 2026-08-10:

| Area | Current state | Health signal |
| --- | --- | --- |
| Rust workspace | 14 packages, about 47.6k source lines | Mainline is fully Rust-owned |
| React Studio | about 27.4k TypeScript/CSS lines | Builds cleanly; interaction coverage is missing |
| Tracked Python | 0 files | Legacy Python audit advice is retired |
| Rust host checks | 227 tests passed | Strong domain/reference evidence, not Jetson proof |
| Portable native contracts | 9 C++ tests passed | Latest-only and reference ABI contracts are covered |
| Frontend checks | typecheck, visual audit, and production build passed | Static/build health is good |
| Repository state | clean and aligned with `origin/develop-alpha` | Workspace hygiene is good |

This snapshot is evidence from the named revision, not a promise that later
unverified edits pass the same gates.

## Maintained Ownership Boundaries

| Module | Primary ownership | Boundary to preserve |
| --- | --- | --- |
| `novasight-core` | Domain types, freshness, targeting, tracking, prediction, control laws | No HTTP, GStreamer, TensorRT, database, or device transport knowledge |
| `novasight-pipeline` | Owned worker threads, latest-only slots, preview/crosshair hubs, control execution | Accept validated domain observations; keep queues bounded |
| `novasight-runtime` | Sole lifecycle actor, configuration transactions, model activation, immutable snapshots | One authority for start/stop/update and fail-closed output |
| `novasight-api` | HTTP/WebSocket projection and control surface | Delegate mutations to `RuntimeHandle`; do not become a second runtime |
| `novasight-store` | Config persistence, model catalog/deployments, license persistence | Keep filesystem/database authority explicit; avoid leaking broad store dependencies |
| `novasight-platform-jetson` | V4L2, DeepStream session, kmNet native transport, Jetson clock | Vendor/platform types stop before `DetectionBatch` |
| `novasight-tensorrt` | Narrow TensorRT C ABI and output decoding | Validate shapes/dtypes before unsafe/native execution |
| `novasight-deepstream-bridge` | Narrow owned metadata copy across the C ABI | Never retain vendor-owned metadata pointers |
| `apps/novasightd` | Production composition root | Linux + default DeepStream is the real product composition |
| `web` | Studio management and diagnostics | Never enter the real-time control loop |

## Current Safeguards

- Runtime commands are typed and owned by one supervisor actor.
- Runtime status is published as immutable watch snapshots.
- Urgent stop and supervisor drop latch physical output closed.
- Latest-only slots and runtime epochs reject stale replacement and old-session
  data.
- Detection freshness, target identity, device commissioning, and output enable
  state gate physical commands.
- DeepStream metadata crosses into Rust as bounded owned snapshots.
- TensorRT and preprocess native code are isolated behind versioned C ABI
  contracts.
- Configuration loading rejects unknown control fields and unsafe path identity
  changes.
- Cargo, web, package, logs, models, and local runtime outputs stay ignored.

## Active Findings

### P1: Verify the restored strict lint gate

`README.md` declares `cargo clippy --workspace --all-targets -- -D warnings` as
a development gate. At the 2026-08-10 audit it failed only on two prediction
helpers with ungrouped argument lists. The current development batch groups
projection and motion-classification inputs into typed internal contexts without
changing the algorithm sequence. Verification was intentionally not run during
that development-only batch.

Acceptance: the declared command exits successfully without a broad lint allow.

### P1: Bring the new host and Jetson quality gates online

The current development batch adds `rust-toolchain.toml` and
`.github/workflows/quality.yml`. Host checks are declared for GitHub-hosted
macOS/Ubuntu runners. The production job remains disabled until a self-hosted
Jetson runner is registered and `NOVASIGHT_JETSON_CI_ENABLED=true` is configured.
No workflow result exists yet, so automation remains `PARTIAL` until its first
successful run.

Portable host gate (macOS; production Linux modules remain Jetson-gated):

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
pnpm --dir web typecheck
pnpm --dir web visual:audit
pnpm --dir web build
```

Jetson gate:

```bash
cargo run -p novasight-packager -- --profile release
cd out/package/NovaSight
bin/novasightd --check
./NovaSight
```

The Jetson acceptance job must also confirm readiness, license status, model
contract, real DeepStream metadata, freshness/latency, capture start/stop, and
safe kmNet output behavior.

### P1: Keep production truth separate from portable reference evidence

macOS can validate domain logic, pipeline ownership, DeepStream pipeline-string
contracts, TensorRT reference ABI, and preprocess reference ABI. It cannot prove
DeepStream SDK loading, NVMM import, real TensorRT execution, V4L2 capture, or
kmNet hardware behavior. Those remain `UNKNOWN` until a Jetson receipt exists.

### P2: Add behavior coverage at external boundaries

Current Rust coverage is strongest in domain, configuration, pipeline, runtime,
and native-contract logic. The Axum router/daemon composition and Studio user
flows have little or no automated behavior coverage. Add coverage when test work
is explicitly in scope; do not inflate production modules with test-only seams.

Priority boundaries:

1. license and trusted-local-control middleware;
2. start/stop/emergency-stop HTTP behavior;
3. WebSocket shutdown and reconnect behavior;
4. configuration persistence failure/rollback;
5. Studio launch, save, retry, and runtime-disconnect flows.

### P2: Reduce concentrated ownership when related code is next touched

Do not split files solely to reduce line counts. Extract only around existing
semantic ownership:

- `StudioConsoleView.tsx`: page-level state hooks, runtime projection helpers,
  launch orchestration, and preview overlay;
- `novasight-runtime/src/supervisor.rs`: lifecycle transitions, model activation
  transaction, device diagnostics, and pipeline notice handling;
- `novasight-store/src/model_catalog.rs`: SQLite repository, filesystem scanner,
  deployment transaction, and response projection;
- `novasight-api/src/control.rs`: router construction, middleware, HTTP handlers,
  and WebSocket session handling.

Each extraction must preserve one runtime authority and avoid parallel sources
of truth.

### P2: Close supply-chain maintenance gaps

- Keep `Cargo.lock` and `web/pnpm-lock.yaml` tracked.
- Keep npm production audit and RustSec audit in automation.
- Add license/source policy only after the allowed proprietary dependency policy
  is written down.
- Replace deprecated `serde_yaml 0.9` through a dedicated config-compatibility
  migration; do not swap serializers without preserving existing YAML behavior.
- Audit oversized dependency boundaries with `cargo tree` and source usage
  before removing dependencies. No direct dependency is currently proven unused.

## Deferred Work

- Broad Studio or CSS rewrites without an active product requirement.
- Splitting crates only for aesthetic size targets.
- Removing DeepStream, TensorRT, kmNet, or native reference seams without current
  dependency and Jetson evidence.
- Renaming persisted configuration fields without a migration.
- Claiming zero-copy, latency, FPS, or production readiness from reference builds.
- Deleting local generated directories without an explicit cleanup request.

## Hard Stop Rules

- Do not reintroduce a Python runtime/backend beside the Rust authority.
- Do not let HTTP, WebSocket, or React state become a second runtime authority.
- Do not let vendor pointers or raw backend tensors escape the native/platform
  boundary.
- Do not reopen output after supervisor loss, stale observations, invalid model
  state, uncommissioned hardware, or device failure.
- Do not treat historical `docs/superpowers` plans as current requirements.
- Do not report a check as passed until its command has completed with exit code
  zero on the relevant platform.

## Decision Status

| Dimension | Status | Current conclusion |
| --- | --- | --- |
| Rust domain/runtime design | YES | Typed ownership and fail-closed seams are present |
| Host development baseline | PARTIAL | Tests/build passed at the recorded baseline; strict lint needed repair |
| Studio static health | YES | Typecheck, visual audit, and build passed at the recorded baseline |
| Studio behavior coverage | NO | No automated interaction suite is configured |
| Automated host CI | PARTIAL | Workflow is tracked; first successful run is pending |
| Jetson production readiness | UNKNOWN | Requires current hardware receipts |
| Dependency vulnerability status | PARTIAL | npm was clean; RustSec fetch was unavailable |
| Documentation authority | YES | This ledger supersedes the deleted Python-era audit content |
