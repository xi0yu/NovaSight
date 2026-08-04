# NovaSight Rust Backend Phase 2 Implementation Plan

> **Execution:** Use `subagent-driven-development` task by task. Each task starts with a failing behavioral test, ends with focused verification and review, and must not begin DeepStream or kmNet integration.

**Goal:** Replace the Phase 1 proportional replay placeholder with behaviorally verified Rust geometry, freshness, tracking, targeting, continuous control, quantization, residual, and latest-command algorithms using captured Python fixtures.

**Scope boundary:** Phase 2 is pure algorithm work. Keep the Python/C++ runtime, Axum routes, YAML/SQLite/License/model ownership, DeepStream bridge, GStreamer, TensorRT, CUDA, and every live device adapter unchanged. `RecordingPointerDevice` remains the only output sink. No production output gate may open.

**Compatibility rule:** Python exports versioned fixtures once; Rust tests consume checked-in fixtures without launching Python. Floating-point comparisons use field-specific tolerances recorded in the fixture schema. Target identity, state transitions, reset reasons, integer counts, generation, and residual outcomes are exact.

---

## Task 1: Capture Versioned Python Algorithm Fixtures

**Files:**
- Create: `scripts/export_phase2_replay_fixtures.py`
- Create: `tests/test_phase2_replay_fixture_export.py`
- Create: `rust/fixtures/phase2/schema.json`
- Create: `rust/fixtures/phase2/static-target.jsonl`
- Create: `rust/fixtures/phase2/moving-target.jsonl`
- Create: `rust/fixtures/phase2/target-switch-loss.jsonl`
- Create: `rust/fixtures/phase2/freshness-reset.jsonl`
- Create: `rust/fixtures/phase2/continuous-control-control.jsonl`

**Source authority:**
- `novasight/coordinates.py`, `novasight/roi.py`
- `novasight/runtime/freshness.py`, `tracker.py`, `target_selector.py`, `kalman.py`
- `novasight/control/algorithms/continuous_atan_predictive_v1/`
- `tests/test_roi.py`, `test_runtime_tracker.py`, `test_runtime_pipeline.py`, `test_continuous_atan_predictive_v1.py`, `test_mouse_control.py`

**Steps:**
1. Write an exporter test that fails because the fixture schema and deterministic exporter do not exist.
2. Define schema version `1` with explicit units and domains: epoch, generation, monotonic nanoseconds, model/ROI/control geometry, detections, trigger sample, target state/reason, control intermediates, integer output, residual, and tolerances.
3. Export canonical cases: center/off-center; shifted ROI; moving target; dropout/reacquire; target switch debounce; coordinate-space change; stale/duplicate observation; continuous response transition; sign inversion; saturation; fractional residual; terminal reset.
4. Run the exporter twice and assert byte-identical output. Reject NaN/Inf and unordered object maps.
5. Verify Python tests that own these contracts still pass.

**Verification:**
- `pytest -q tests/test_phase2_replay_fixture_export.py tests/test_roi.py tests/test_runtime_tracker.py tests/test_continuous_atan_predictive_v1.py`
- Commit: `test(phase2): capture Python algorithm replay fixtures`

---

## Task 2: Add Typed Units and Coordinate Geometry

**Files:**
- Create: `rust/crates/novasight-core/src/units.rs`
- Create: `rust/crates/novasight-core/src/geometry/mod.rs`
- Create: `rust/crates/novasight-core/src/geometry/types.rs`
- Create: `rust/crates/novasight-core/src/geometry/transform.rs`
- Create: `rust/crates/novasight-core/tests/geometry_contract.rs`
- Modify: `rust/crates/novasight-core/src/lib.rs`
- Modify: `rust/crates/novasight-core/src/perception/types.rs`

**Interfaces:**
- Newtypes distinguish model pixels, ROI pixels, control pixels, normalized coordinates, radians, counts, and monotonic durations.
- `CoordinateTransform` performs model ↔ ROI ↔ capture ↔ control conversions without implicit coordinate-space mixing.
- Validated rectangles guarantee finite values, positive dimensions, and complete containment.

**Steps:**
1. Write fixture-driven round-trip and shifted-ROI tests; include large integer geometry above exact `f32` range.
2. Implement validated typed units with checked constructors and `f64` internal geometry.
3. Implement transforms and explicit rounding only at named boundaries.
4. Migrate `Detection`/`DetectionBatch` geometry accessors without public raw-field escape hatches.
5. Verify no Axum/store/platform type enters core geometry.

**Verification:**
- `cargo test -p novasight-core --test geometry_contract`
- `cargo test -p novasight-core`
- Commit: `feat(core): add typed coordinate geometry`

---

## Task 3: Implement Freshness and Observation Admission

**Files:**
- Create: `rust/crates/novasight-core/src/freshness/mod.rs`
- Create: `rust/crates/novasight-core/src/freshness/gate.rs`
- Create: `rust/crates/novasight-core/tests/freshness_contract.rs`
- Modify: `rust/crates/novasight-core/src/lib.rs`
- Modify: `rust/crates/novasight-core/src/error.rs`

**Interfaces:**
- `FreshnessPolicy` stores positive capture, inference, trigger, and command-age bounds.
- `FreshnessGate` chooses the strictest positive bound and returns typed rejection reasons.
- Admission checks epoch, strictly increasing generation, monotonic timestamps, coordinate-space identity, and age before targeting/control.

**Steps:**
1. Write fixture-driven boundary tests for exactly-at-limit, one-nanosecond-over, stale trigger, duplicate generation, clock regression, and terminal epoch.
2. Implement pure admission logic with no system-clock reads; callers supply `now` from `Clock`.
3. Ensure every rejection resets downstream state through a typed reset reason.
4. Add property tests for nonnegative age and strictest-positive threshold selection.

**Verification:**
- `cargo test -p novasight-core --test freshness_contract`
- `cargo test -p novasight-core`
- Commit: `feat(core): enforce freshness admission invariants`

---

## Task 4: Port Tracker and TargetingCore

**Files:**
- Create: `rust/crates/novasight-core/src/tracking/mod.rs`
- Create: `rust/crates/novasight-core/src/tracking/assignment.rs`
- Create: `rust/crates/novasight-core/src/tracking/kalman.rs`
- Create: `rust/crates/novasight-core/src/tracking/tracker.rs`
- Create: `rust/crates/novasight-core/src/targeting/core.rs`
- Create: `rust/crates/novasight-core/tests/tracking_contract.rs`
- Create: `rust/crates/novasight-core/tests/targeting_contract.rs`
- Modify: `rust/crates/novasight-core/src/targeting/mod.rs`
- Modify: `rust/crates/novasight-core/src/lib.rs`

**Interfaces:**
- `TrackerConfig`, `TrackId`, `TrackState`, and `TrackedBatch` are core-owned typed values.
- Assignment is deterministic; equal costs use stable track/object identifiers.
- `TargetingCore` owns lock, debounce, preferred-class fallback, loss/reacquire, and explicit reset state.
- Lost tracks can never produce a control target.

**Steps:**
1. Write fixture-driven association and selection tests before implementation.
2. Port only the business invariants from Python, not its class hierarchy or mutable dictionaries.
3. Implement bounded history and checked Kalman fallback; no allocation proportional to unbounded runtime age.
4. Implement target switch/loss/reset transitions and deterministic tie-breaking.
5. Prove coordinate-space change, epoch change, and stale admission clear targeting history.

**Verification:**
- `cargo test -p novasight-core --test tracking_contract`
- `cargo test -p novasight-core --test targeting_contract`
- `cargo test -p novasight-core`
- Commit: `feat(core): port deterministic tracking and targeting`

---

## Task 5: Port Continuous ControlCore

**Files:**
- Create: `rust/crates/novasight-core/src/controller/atan.rs`
- Create: `rust/crates/novasight-core/src/controller/response_curve.rs`
- Create: `rust/crates/novasight-core/src/prediction/mod.rs`
- Create: `rust/crates/novasight-core/tests/control_trace_parity.rs`
- Modify: `rust/crates/novasight-core/src/control/mod.rs`
- Modify: `rust/crates/novasight-core/src/error.rs`

**Interfaces:**
- `ContinuousControl` consumes an admitted typed observation and returns a typed decision plus compatibility telemetry.
- Continuous response curve, velocity history, lead prediction, atan response, caps, deadzone, slew, recoil contribution, and reset reasons are explicit state-machine concepts.
- Algorithm state is epoch/target/coordinate-space scoped.

**Steps:**
1. Write JSONL fixture parity tests for every intermediate and exact state transition before implementation.
2. Define validated config structs; reject non-finite, negative, inverted, and inconsistent thresholds.
3. Implement bounded motion history and deterministic continuous response transitions.
4. Implement prediction and controller math in `f64`, recording explicit tolerance only for noncritical floating intermediates.
5. Reset derivative/history on epoch, target, coordinate-space, stale, duplicate, and terminal events.
6. Keep `ProportionalReplayControl` available only as a Phase 1 test adapter; Runtime integration switches explicitly in Task 7.

**Verification:**
- `cargo test -p novasight-core --test control_trace_parity`
- `cargo test -p novasight-core`
- Commit: `feat(core): port continuous predictive control`

---

## Task 6: Implement Quantizer, Residual, and LatestCommandSlot

**Files:**
- Create: `rust/crates/novasight-core/src/output/quantizer.rs`
- Create: `rust/crates/novasight-core/src/output/latest_command.rs`
- Create: `rust/crates/novasight-core/tests/output_contract.rs`
- Modify: `rust/crates/novasight-core/src/output/mod.rs`
- Modify: `rust/crates/novasight-core/src/telemetry/mod.rs`

**Interfaces:**
- Quantization owns fractional residual and uses documented rounding/sign rules.
- `LatestCommandSlot` has depth one, overwrite counters, expiry, epoch/generation provenance, and atomic take-or-reject semantics.
- Reset clears residual and pending command before a new epoch/target can emit.

**Steps:**
1. Write exact fixture tests for positive/negative halves, saturation, residual carry, sign inversion, deadzone, overwrite, expiry, and reset.
2. Implement checked `f64` → integer count conversion with no silent wrap or architecture-sized integer.
3. Implement capacity-one latest replacement and typed reject reasons.
4. Prove stale/old-epoch commands can never reach `PointerDevice`.

**Verification:**
- `cargo test -p novasight-core --test output_contract`
- `cargo test -p novasight-core`
- Commit: `feat(core): add exact command quantization and latest slot`

---

## Task 7: Integrate the Pure Algorithm Slice into RuntimeManager

**Files:**
- Create: `rust/crates/novasight-core/tests/runtime_algorithm_slice.rs`
- Modify: `rust/crates/novasight-core/src/runtime/session.rs`
- Modify: `rust/crates/novasight-core/src/runtime/manager.rs`
- Modify: `rust/crates/novasight-core/src/runtime/state.rs`
- Modify: `rust/crates/novasight-api/src/dto/runtime.rs`
- Modify: `rust/bins/relink_server/src/bootstrap.rs`

**Interfaces:**
- Pipeline becomes `ReplayPerceptionSource → FreshnessGate → Tracker → TargetingCore → ContinuousControl → Quantizer → LatestCommandSlot → RecordingPointerDevice`.
- RuntimeManager remains the only lifecycle owner; terminal faults still travel through its bounded event channel.
- API additions are telemetry-only and backward compatible; existing keys/types remain unchanged.

**Steps:**
1. Write an end-to-end fixture replay test that fails on the Phase 1 proportional adapter.
2. Inject validated algorithm configs/dependencies at session construction; do not let handlers or adapters call inner modules directly.
3. Preserve epoch revocation, cancellation responsiveness, post-fault join/clear, bounded watch, and replay pacing.
4. Compare Rust target state/reason, control intermediates, integer counts, residual, and reset edges against every fixture.
5. Run duplicate start, stop-during-control, fault, restart, slow WebSocket, and latest-command overwrite scenarios.
6. Confirm `RecordingPointerDevice` is still the only linked device adapter and output gate remains closed.

**Verification:**
- `cargo test -p novasight-core --test runtime_algorithm_slice`
- `cargo test -p novasight-core`
- `cargo test -p novasight-api`
- `cargo test -p relink-server`
- Commit: `feat(runtime): run verified Rust algorithm core in replay`

---

## Task 8: Phase 2 Regression and Performance Gate

**Files:**
- Create: `rust/crates/novasight-core/benches/algorithm_replay.rs`
- Modify only if required: `rust/crates/novasight-core/Cargo.toml`

**Steps:**
1. Run every checked-in Python fixture through Rust and produce a compact parity table.
2. Require exact equality for target identity/state/reason, reset edges, integer counts, generation, residual reset, and command ordering.
3. Record explicit per-field floating tolerances; reject any broad global epsilon.
4. Benchmark fixed fixtures for allocation count, batches/s, P50/P95/P99 per-batch latency, and bounded memory. No threshold may be weakened to make a regression pass.
5. Run workspace tests, Clippy, rustfmt, frontend build, clean-archive Cargo check, and the real replay HTTP/WebSocket/SIGTERM smoke.
6. Verify no DeepStream, GStreamer, CUDA, TensorRT, or kmNet dependency entered the algorithm slice.

**Verification:**
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo bench -p novasight-core --bench algorithm_replay`
- `pnpm --dir web build`
- Commit: `test(phase2): gate Rust algorithm parity and performance`

---

## Phase 2 Definition of Done

- Checked-in fixtures are deterministic, versioned, and generated from current Python behavior.
- Rust geometry, freshness, tracking, targeting, continuous control, quantization, residual, and latest-command semantics match fixtures.
- Noncritical floating tolerances are explicit; identity/state/reset/integer/residual outcomes are exact.
- RuntimeManager remains sole lifecycle owner; no post-stop/post-fault/post-switch command leaks.
- All channels/slots stay bounded and slow consumers do not backpressure runtime.
- `RecordingPointerDevice` remains the only output adapter; output gate cannot open.
- Existing HTTP/WebSocket/frontend contracts remain buildable and behaviorally compatible.
- No DeepStream or kmNet implementation begins in Phase 2.
- Phase 3 starts only after this gate passes and captured parity evidence is retained.
