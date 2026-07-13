# NovaSight Main Mouse Control

Date: 2026-07-13

Status: `dual_phase_atan_predictive_v1` is the configured mainline. Older algorithms remain isolated compatibility implementations.

## Mainline Route

```text
latest DetectionBatch
-> freshness and monotonic timestamp checks
-> TargetSelector / Tracker identity
-> raw bbox aim point
-> current real ROI error
-> FAR / NEAR hysteresis from real error
-> variable-dt X motion estimate
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
  active_algorithm: dual_phase_atan_predictive_v1
  algorithms:
    dual_phase_atan_predictive_v1:
      far:
        kp: 0.35
      near:
        kp: 0.15
    calibrated_angular:
      kp_x: 1.0
```

The source namespace is:

```text
novasight.control.algorithms.dual_phase_atan_predictive_v1
```

Consequently, `far.kp`, `near.kp`, and similarly named values in other algorithm namespaces are independent. Transitional Python properties and one-way config migration keep old configurations loadable; new serialized output uses only the explicit namespace.

## Algorithm Differences

| Algorithm ID | Error/units | Prediction | Near behavior | Delivery |
| --- | --- | --- | --- | --- |
| `dual_phase_atan_predictive_v1` | ROI real error -> source angle -> full correction counts -> counts-domain Atan | dedicated robust X-only estimator, confidence weighting, absolute/relative caps | FAR/NEAR hysteresis; no movement deadzone | one direct integer command per observation |
| `calibrated_angular` | full-space angular PD | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |
| `universal_saturated` | empirical pixel-domain saturated Atan | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |
| `ttbox_pid_atan` | empirical pixel-domain Atan; despite its historical name, not a complete PID | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |

The new algorithm bypasses the entire legacy `MouseController` envelope, so legacy Y prediction, deadzone, arrival state, slew limit, rounding residual, and Scheduler capacity cannot alter its result.

## Real And Control Error

```text
e_real = current raw aim - crosshair
e_ctrl.x = e_real.x + safe_prediction_offset_x
e_ctrl.y = e_real.y
```

`e_real` owns FAR/NEAR selection, actual zero-cross detection, convergence, reset decisions, and telemetry. `e_ctrl` only feeds the projection and Atan calculation. V1 never predicts Y.

Prediction is a small reversible addition to the current observation, not the primary controller:

```text
h = clamp(frame_age + actuation_delay, 0, max_horizon)
raw_offset_x = estimated_velocity_x * h
weight = mode_weight * motion_confidence * track_confidence
allowed = min(absolute_cap, base_cap + relative_cap * abs(e_real.x))
safe_offset_x = clamp(weight * raw_offset_x, -allowed, allowed)
```

NEAR mode prevents low-confidence prediction from changing the X control direction. Every actual `e_real(t) * e_real(t-1) < 0` zero-cross clears that axis's fractional residual, damps velocity, and starts a short prediction cooldown.

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
u = kp * atan_scale_counts * atan(full_error_counts / atan_scale_counts)
u = clamp(u, -max_counts_per_update, max_counts_per_update)
```

The motion estimator uses real capture `dt`; controller Kp remains the configured per-observation gain. No D term or velocity feed-forward is added.

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

If a device protocol has a smaller single-packet range, the controller limit should normally be configured to that range. Any unavoidable transport fragmentation belongs below the control algorithm, must complete promptly, and must remain discardable by newer input.

## Strict Blocks And Resets

The algorithm emits `(0, 0)` for future/stale/non-monotonic observations, invalid geometry, invalid or predicted-only targets, and incompatible timestamp domains. Global generation/frame/capture cursors remain monotonic across target switches; target-local mode, estimator, prediction cooldown, and residual state reset independently.

Target loss, runtime restart, algorithm/config/calibration changes, and capture/inference restarts clear all control state. Releasing the trigger clears quantizer residual without creating historical debt.

## Telemetry

The decision trace includes identity and timestamps, aim/bbox and real/control errors, FAR/NEAR state, variable `dt`, estimator innovation/velocity/confidence, prediction horizon/weight/caps/offset, full correction counts, float demand, integer command, quantizer residual, zero-cross state, block reason, and:

```text
delivery_mode: single_command_per_observation
scheduler_used: false
```

See `docs/control/dual_phase_atan_predictive_v1.md` for the frozen implementation contract and tuning boundaries.

## Remaining Physical Uncertainty

`prediction.actuation_delay_ms` still requires Jetson + device + game trace calibration. The estimator models target motion relative to the crosshair; it does not yet separate target motion, manual camera motion, and NovaSight-induced camera motion. Strict prediction caps make that limitation tolerable for V1 but do not remove it.
