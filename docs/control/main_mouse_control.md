# NovaSight Main Mouse Control

Date: 2026-07-13

Status: `dual_phase_atan_robust_predictive_v2` is the configured precise mainline. V1 and older algorithms remain isolated compatibility/A-B implementations.

## Mainline Route

```text
latest DetectionBatch
-> freshness and monotonic timestamp checks
-> TargetSelector / Tracker identity
-> raw bbox aim point
-> current measured ROI error
-> FAR / NEAR hysteresis from measured error
-> four same-target X positions / three segment velocities
-> median velocity / time-adaptive EMA
-> confidence-weighted bounded X prediction
-> complete source/FOV/counts projection
-> counts-domain Atan shaping
-> per-observation output clamp
-> truncating integer quantizer
-> MouseCommandExecutor
-> exactly one move(dx, dy)
```

The algorithm recalculates from the next real observation. It creates no trajectory plan, stores no unexecuted movement target, and has no Scheduler tick.

```text
frame 1 -> move(20, 3)
frame 2 -> move(14, 2)
frame 3 -> move(7, 1)
frame 4 -> move(2, 0)
```

`Kp + Atan + max_counts_per_update` already implement the incremental closed-loop approach. Splitting that increment again would create a stale open-loop tail and double-slow the response.

## Algorithm And Configuration Isolation

The serialized selector is `control.active_algorithm`; implementation-specific values live under `control.algorithms.<algorithm_id>`.

```yaml
control:
  active_algorithm: dual_phase_atan_robust_predictive_v2
  algorithms:
    dual_phase_atan_robust_predictive_v2:
      velocity:
        history_size: 4
        velocity_sample_count: 3
        smoothing_tau_ms: 30.0
      prediction:
        coefficient: 1.0
      atan:
        far:
          kp: 0.35
        near:
          kp: 0.15
    calibrated_angular:
      kp_x: 1.0
```

The source namespace is:

```text
novasight.control.algorithms.dual_phase_atan_robust_predictive_v2
```

Consequently, `prediction.coefficient`, `atan.far.kp`, and similarly named values in other algorithm namespaces are independent. V1 remains selectable without sharing mutable estimator, mode, prediction, or quantizer state with V2.

## Algorithm Differences

| Algorithm ID | Error/units | Prediction | Near behavior | Delivery |
| --- | --- | --- | --- | --- |
| `dual_phase_atan_robust_predictive_v2` | measured ROI px -> source px -> radians -> full counts -> counts-domain Atan | four positions, three `px/ms` velocities, median, dynamic EMA, confidence and caps | FAR/NEAR hysteresis; no movement deadzone | one direct integer command per observation |
| `dual_phase_atan_predictive_v1` | ROI error -> source angle -> full correction counts -> counts-domain Atan | constant-velocity Kalman compatibility baseline | FAR/NEAR hysteresis; no movement deadzone | one direct integer command per observation |
| `calibrated_angular` | full-space angular PD | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |
| `universal_saturated` | empirical pixel-domain saturated Atan | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |
| `ttbox_pid_atan` | empirical pixel-domain Atan; despite its historical name, not a complete PID | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |

The new algorithm bypasses the entire legacy `MouseController` envelope, so legacy Y prediction, deadzone, arrival state, slew limit, rounding residual, and Scheduler capacity cannot alter its result.

## Measured And Control Error

```text
e_meas = current measured aim - crosshair
e_ctrl.x = e_meas.x + safe_prediction_offset_x
e_ctrl.y = e_meas.y
```

`e_meas` owns FAR/NEAR selection, actual zero-cross detection, convergence, fallback, and telemetry. `e_ctrl` only feeds projection and Atan calculation. V2 never predicts Y.

Prediction is a small reversible addition to the current observation, not the primary controller:

```text
h_ms = clamp(frame_age_ms + actuation_delay_ms, 0, max_horizon_ms)
raw_offset_x = filtered_velocity_x_px_ms * h_ms
coefficient_offset_x = raw_offset_x * prediction_coefficient
weighted_offset_x = coefficient_offset_x * motion_confidence
allowed = min(mode_absolute_cap, mode_base_cap + mode_relative_cap * abs(e_meas.x))
safe_offset_x = clamp(weighted_offset_x, -allowed, allowed)
```

The velocity path is frozen to four current-target samples and three adjacent capture-time velocities. Their median rejects one-position spikes; EMA uses `alpha = 1 - exp(-dt_ms / tau_ms)`. Motion confidence also includes current detection and Tracker identity confidence. Stationary observations decay the previous velocity naturally—there is no forced-zero branch. A measured-error sign crossing clears only the opposite-direction fractional count; it does not damp or overwrite the motion estimate.

## Projection And Control Law

Detection coordinates are ROI coordinates. Trusted `CoordinateTransform` geometry identifies the crosshair in that ROI and maps the ROI error into the complete source projection.

```text
source_error = observation_error * roi_size / observation_size
focal_x = (source_width / 2) / tan(fov_x / 2)
theta = atan(source_error / focal)
full_error_counts = theta * counts_per_360 / (2*pi)
```

For the active FAR or NEAR phase:

```text
u = kp * scale_counts * atan(full_error_counts / scale_counts)
u = clamp(u, -max_counts_per_update, max_counts_per_update)
```

The motion estimator uses adjacent `capture_ts_ns` deltas converted to milliseconds; controller Kp remains the configured per-observation gain. No D term or velocity feed-forward is added.

## Integer Quantizer And Direct Executor

The quantizer retains only sub-count demand:

```text
accumulator += float_demand
integer_count = trunc(accumulator)
accumulator -= integer_count
```

Direction changes clear the opposite-direction residual. Trigger-inactive observations may continue updating the motion estimate, but the quantizer is cleared and cannot bank historical movement.

`MouseCommandExecutor` rechecks the trigger snapshot, freshness deadline, and monotonically increasing generation immediately before its serialized device invocation, then reports timing or failure. It does not:

- split a command into a trajectory;
- retain pending counts across observations;
- run an independent send tick;
- finish an old frame after a newer frame arrives.

V2 restricts configured per-observation limits to at most 127 counts. The executor additionally rejects non-integer or out-of-device-range values before calling the driver. Any unavoidable transport fragmentation belongs below the control algorithm, must complete promptly, and must remain discardable by newer input.

## Strict Blocks And Resets

The algorithm emits `(0, 0)` for future/stale observations, non-monotonic generation/frame results, invalid geometry, invalid or predicted-only targets, and incompatible timestamp domains. A regressed capture timestamp is excluded from velocity history and disables prediction for that observation, but valid measured-position feedback still runs and the following frame can start a new epoch. Target-local mode, four-point history, EMA state, confidence, and residual state reset on target or Tracker lifecycle changes. A history gap over `velocity.history_reset_gap_ms` starts a new one-point window.

Target loss, runtime restart, algorithm/config/calibration changes, and capture/inference restarts clear all control state. Releasing the trigger clears quantizer residual without creating historical debt.

## Telemetry

The decision trace includes identity/timestamps, aim/bbox, measured/control errors, FAR/NEAR state, four-point history count, three `px/ms` velocities, median/EMA/spread/confidence, prediction horizon/coefficient/caps/offset, full correction counts, float demand, integer command, quantizer residual, zero-cross state, and block reason, plus:

```text
delivery_mode: single_command_per_observation
scheduler_used: false
```

See `docs/control/dual_phase_atan_robust_predictive_v2.md` for the frozen V2 implementation contract and tuning boundaries.

## Remaining Physical Uncertainty

`prediction.actuation_delay_ms` still requires Jetson + device + game trace calibration. The estimator models target motion relative to the crosshair; it does not yet separate target motion, manual camera motion, and NovaSight-induced camera motion. Strict prediction caps make that limitation tolerable for V2 but do not remove it.
