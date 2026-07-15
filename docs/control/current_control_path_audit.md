# NovaSight Current Mouse Control Path Audit

Date: 2026-07-15

Status: live-code audit after promoting `dual_phase_atan_robust_predictive_v2` and retaining V1 as an isolated comparison implementation.

## Active Mainline

```text
DetectionBatch latest-only gate
-> FrameContext
-> basic candidate filter
-> RuntimeTracker (identity/association)
-> RuntimeTargetSelector
-> RawAimPointProjector using current bbox
-> dual_phase_atan_robust_predictive_v2
-> capacity-one latest-replace CommandScheduler
-> MouseCommandExecutor
-> KmNetExecutor.move(dx, dy)
```

The dedicated X estimator consumes the measured aim point. It retains four positions from one target, derives three adjacent `px/ms` speeds, takes their median, and applies a capture-dt adaptive EMA. Tracker Kalman state is not used as V2's predicted aim, so there is no double prediction.

The active algorithm bypasses:

```text
RuntimeService._mouse_observation_metadata legacy prediction
MouseController legacy deadzone/arrival/slew/rounding envelope
legacy CommandScheduler trajectory split path
```

## Per-Observation Ownership

`RuntimeService._control_intent_from_context()` determines trigger readiness before quantization, creates a typed algorithm observation, and calls the new algorithm exactly once. The resulting integers are wrapped in one `ControlIntent`. `ExecutorRegistry` forces:

```text
scheduler != None
direct_output = false
latest_replace = true
```

The observation call replaces the single pending complete integer command and never calls hardware. `RuntimeService.process_control_tick()` takes at most that one command and sends it through `MouseCommandExecutor`. A newer observation or clear event increments the delivery epoch, so a command already removed from the slot but still waiting for the device lock is discarded before the device call. Scheduler step limits equal V2's per-update limit, so this path never creates a multi-step trajectory or count debt.

## Source And Time Contract

- `capture_ts_ns`, `inference_end_ts_ns`, and `control_now_ns` must share the host monotonic domain.
- Capture age below zero, inference completion outside `[capture, control_now]`, stale age, or generation/frame rollback blocks the whole decision. A non-increasing capture timestamp resets prediction history and uses pure measured-position feedback for that otherwise valid observation.
- Global observation cursors survive target switches; target-local estimator/mode/quantizer state does not.
- Motion-estimator `dt_ms` is the adjacent same-target capture timestamp difference divided by `1_000_000`.
- Prediction uses the arithmetic mean of the three capture intervals as one reference frame, then multiplies filtered X velocity by that dt and configured `lead_frames`. Frame age only participates in stale-observation rejection.

## Coordinate Contract

Detection boxes and the dedicated estimator use current ROI coordinates. Trusted source geometry creates `CoordinateTransform`; the full-control center is mapped back into ROI coordinates for `e_meas`. The controller then maps ROI error through ROI/source scaling, FOV, radians, and calibrated counts.

If source geometry is unavailable or untrusted, the aim observation is invalid and no command is sent. A predicted-only or stale Track is also invalid as a control source.

## State Ownership

The new algorithm alone owns:

- FAR/NEAR selection with one measured-error threshold;
- four-position same-target history;
- three-segment median and time-adaptive EMA;
- spread/trend/detection/track-identity prediction confidence;
- reference dt, configured/effective lead frames, relative and absolute caps;
- measured-error zero-cross history;
- per-axis sub-count quantizer residual.

The runtime owns target selection, initial trigger readiness, algorithm calculation, reset edges, the capacity-one delivery slot, and telemetry publication. Immediately before the serialized device call, the registry verifies that no newer submission superseded the selected command, then `MouseCommandExecutor` rechecks the trigger snapshot, command deadline, and increasing generation. The delivery slot retains at most one complete command and no trajectory.

## Compatibility Algorithms

The remaining compatibility implementations stay selectable under their own config namespaces:

```text
control.algorithms.calibrated_angular
control.algorithms.universal_saturated
```

These generic controllers continue to use the existing `MouseController` envelope and legacy `scheduler_enabled` choice. Removed `ttbox_pid_atan` and `dual_phase_atan_predictive_v1` config blocks are discarded during legacy migration; an old configuration that still selects either ID is migrated to `dual_phase_atan_robust_predictive_v2`.

## Configuration Contract

```text
control.active_algorithm
control.algorithms.<algorithm_id>.*
```

The loader migrates the preceding `control.mode` and top-level algorithm blocks. Runtime Python aliases remain temporary compatibility accessors; serialized config and Studio edits use the isolated namespace.

## Direct Evidence

Implementation owners:

- `novasight/control/algorithms/dual_phase_atan_robust_predictive_v2/core.py`
- `novasight/control/algorithms/dual_phase_atan_robust_predictive_v2/motion_history.py`
- `novasight/runtime/service.py::_dual_phase_control_command`
- `novasight/executors/runtime.py::ExecutorRegistry.tick_pending`
- `novasight/config/runtime.py::ControlAlgorithmConfigs`

Focused integration tests prove that consecutive DetectionBatch results replace the pending command, one control tick sends only the newest frame, and a newer observation also supersedes an older command that has left the slot but has not acquired the device lock. A deterministic closed-loop regression also requires limited prediction to reduce post-warmup mean absolute lag versus the `lead_frames=0` feedback baseline.

## Remaining Blind Spots

The least-certain production value is still physical actuation delay. Successful device-send timestamps do not prove when the game consumes input or when the result appears in capture.

The largest control-model limitation is self-motion contamination: screen velocity combines target movement, manual view movement, and NovaSight's own prior output. V2 keeps prediction small and immediately removable; exact self-motion subtraction remains deferred until counts-to-visual timing is measured.

Capture resource/caps success also does not prove nonblack visual content. That requires a separate GPU content probe and is outside mouse-control ownership.
