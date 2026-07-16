# NovaSight Main Mouse Control

Date: 2026-07-15

Status: `dual_phase_atan_robust_predictive_v2` is the configured precise mainline. `ttbox_pid_atan` and `dual_phase_atan_predictive_v1` have been removed.

## Mainline Route

```text
latest DetectionBatch
-> freshness and monotonic timestamp checks
-> TargetSelector / Tracker identity
-> shared bbox aim point (`control.aim`, optional profile/class override)
-> one frozen `ControlReference` (verified HUD crosshair or geometry fallback)
-> current measured ROI error
-> FAR / NEAR selection from one measured-error threshold
-> four same-target X positions / three segment velocities
-> median velocity / time-adaptive EMA
-> confidence-weighted bounded X prediction
-> complete source/FOV/counts projection
-> counts-domain Atan shaping
-> per-observation output clamp
-> truncating integer quantizer
-> capacity-one latest-replace slot
-> MouseCommandExecutor
-> at most one move(dx, dy) per output tick
```

The algorithm recalculates from each real observation and creates no trajectory plan. Delivery retains at most one complete unsent command; a newer observation replaces it, and the independent output tick sends only the current command.

```text
frame 1 -> pending(20, 3)
frame 2 -> replace pending(14, 2)
output tick -> move(14, 2)
frame 3 -> pending(7, 1)
output tick -> move(7, 1)
```

`Kp + Atan + max_counts_per_update` already implement the incremental closed-loop approach. Splitting that increment again would create a stale open-loop tail and double-slow the response.

## Target Selection Score

Tracker identity remains class-consistent and Hungarian association is unchanged. `TargetSelector` ranks confirmed tracks with one normalized score:

```text
selection_score =
    normalized(
        class_weight * class_score
      + quality_weight * quality_score
      + distance_weight * distance_score
    )
```

The first class ID in `inference.detection_class_priority` receives `class_score=1.0`, the second receives `0.5`, and every remaining class receives `0.0`. The default `1,0,...` therefore prefers class 1 over class 0 while treating all other classes equally.

`quality_score` combines detection confidence and square-root visible size normalized against the median area of the same class in the current candidate set. Tracker quality additionally includes identity continuity and Kalman position sigma relative to bbox size. This keeps visible-size evidence without comparing a naturally small head box directly against a body box. `distance_score` is `1 - distance / selection_radius`, clamped to `[0, 1]`. The default component weights are `0.40 / 0.05 / 0.55`. Among fallback candidates of the same class, quality can only overturn distance inside roughly 9% of the selection radius; this makes the nearest body the normal result after a preferred head disappears while retaining confidence, size, and stability as close-range tie evidence.

The aim rule is shared by every controller: X is always bbox center and Y is `bbox_top + bbox_height * effective_y_ratio`. `control.aim.role_y_ratios` stores exactly three ratios (`head`, `body`, `other`), while `control.aim.class_roles.<detection_profile>.<class_id>` maps model classes to those roles. Unmapped and unknown classes use `other`. Candidate tracking and final control projection resolve the same effective ratio. Legacy shared or V2-local ratios are migrated to all three roles and are not read by the runtime.

Model detections are mapped back into ROI coordinates before selection. Selection radius uses 640-ROI reference pixels, while bbox area remains soft quality evidence rather than a hard rejection gate. This cannot prioritize a class-1 object that the model did not detect.

Target switching still requires composite-score advantage, Tracker continuity, and the configured confirmation delay. The score chooses a challenger; it does not bypass target-lock safety.

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
        smoothing_frames: 3.0
      prediction:
        lead_frames: 1.0
      atan:
        scale_counts: 1024.0
        far:
          kp: 0.90
        near:
          kp: 0.30
    calibrated_angular:
      kp_x: 1.0
```

The source namespace is:

```text
novasight.control.algorithms.dual_phase_atan_robust_predictive_v2
```

Consequently, `prediction.lead_frames`, `atan.far.kp`, and similarly named values in other algorithm namespaces are independent.

## Algorithm Differences

| Algorithm ID | Error/units | Prediction | Near behavior | Delivery |
| --- | --- | --- | --- | --- |
| `dual_phase_atan_robust_predictive_v2` | measured ROI px -> source px -> radians -> full counts -> counts-domain Atan | four positions, three `px/ms` velocities, median, dynamic EMA, confidence and caps | one NEAR threshold; all other error is FAR; no movement deadzone | capacity-one latest-replace slot; output tick sends newest |
| `calibrated_angular` | full-space angular PD | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |
| `universal_saturated` | empirical pixel-domain saturated Atan | legacy Tracker prediction | shared legacy deadzone/slew | legacy direct or Scheduler setting |

The new algorithm bypasses the legacy `MouseController` envelope, so legacy Y prediction, deadzone, arrival state, slew limit, rounding residual, and trajectory splitting cannot alter its result. Shared recoil configuration is explicitly copied into the V2 algorithm and is therefore the only shared output effect on this path.

## Measured And Control Error

```text
e_meas = current measured aim - crosshair
e_ctrl.x = e_meas.x + safe_prediction_offset_x
e_ctrl.y = e_meas.y
```

`e_meas` owns FAR/NEAR selection, actual zero-cross detection, convergence, fallback, and telemetry. `e_ctrl` only feeds projection and Atan calculation. V2 never predicts Y.

Prediction is a small reversible addition to the current observation, not the primary controller:

```text
dt_ref_ms = mean(dt12_ms, dt23_ms, dt34_ms)
raw_offset_x = filtered_velocity_x_px_ms * dt_ref_ms * lead_frames
weighted_offset_x = raw_offset_x * motion_confidence
allowed = min(mode_absolute_cap, mode_base_cap + mode_relative_cap * abs(e_meas.x))
safe_offset_x = clamp(weighted_offset_x, -allowed, allowed)
```

The velocity path uses four current-target samples and three adjacent capture-time velocities. Their velocity median rejects one-position spikes; their capture intervals use the arithmetic mean required by the frame-based prediction contract. EMA uses `alpha = 1 - exp(-latest_dt / (dt_ref * smoothing_frames))`. Motion confidence also includes current detection and Tracker identity confidence. Stationary observations decay the previous velocity naturally—there is no forced-zero branch. A measured-error sign crossing clears only the opposite-direction fractional count; it does not damp or overwrite the motion estimate.

Y target-velocity prediction remains disabled because apparent Y motion mixes target motion, recoil, manual input, and prior NovaSight output. When shared recoil is enabled, V2 instead adds one configured fixed reverse-Y count contribution per fresh accepted observation after the real-left-button delay. Fixed recoil owns a separate fractional residual and never runs from the independent output tick.

## Projection And Control Law

Detection coordinates are ROI coordinates. Trusted `CoordinateTransform` geometry identifies the crosshair in that ROI and maps the ROI error into the complete source projection.

```text
source_error = observation_error * roi_size / observation_size
focal_x = (source_width / 2) / tan(fov_x / 2)
theta = atan(source_error / focal)
full_error_counts = theta * counts_per_360 / (2*pi)
```

FAR and NEAR share one Atan scale. The active phase selects only Kp and its output limit:

```text
u = kp * scale_counts * atan(full_error_counts / scale_counts)
u = clamp(u, -max_counts_per_update, max_counts_per_update)
```

The motion estimator uses adjacent `capture_ts_ns` deltas converted to milliseconds; controller Kp remains the configured per-observation gain. No D term or velocity feed-forward is added.

## Integer Quantizer And Latest-Replace Delivery

The quantizer retains only sub-count demand:

```text
accumulator += float_demand
integer_count = trunc(accumulator)
accumulator -= integer_count
```

Direction changes clear the opposite-direction residual. Trigger-inactive observations may continue updating the motion estimate, but the quantizer is cleared and cannot bank historical movement.

The delivery scheduler retains one complete integer command. A newer observation replaces an unsent older command, including one already removed from the slot but still waiting for the device lock. The output tick does not split the command because its step limit equals V2's configured per-update maximum.

`MouseCommandExecutor` then rechecks the trigger snapshot, freshness deadline, and monotonically increasing generation immediately before its serialized device invocation. Together, the delivery path does not:

- split a command into a trajectory;
- merge or repay pending counts across observations;
- send a superseded command that has not entered the driver call.

An already-running driver call cannot be cancelled; latest-replace governs commands that have not entered that call.

V2 restricts configured per-observation limits to the signed 16-bit kmNet move range. The executor additionally rejects non-integer or out-of-device-range values before calling the driver. Any unavoidable transport fragmentation belongs below the control algorithm, must complete promptly, and must remain discardable by newer input.

## Strict Blocks And Resets

The algorithm emits `(0, 0)` for future/stale observations, non-monotonic generation/frame results, invalid geometry, invalid or predicted-only targets, and incompatible timestamp domains. A regressed capture timestamp is excluded from velocity history and disables prediction for that observation, but valid measured-position feedback still runs and the following frame can start a new epoch. Target-local mode, four-point history, EMA state, confidence, and residual state reset on target or Tracker lifecycle changes. A history gap over `velocity.history_reset_gap_ms` starts a new one-point window.

Target loss, runtime restart, algorithm/config/calibration changes, and capture/inference restarts clear all control state. Releasing the trigger clears quantizer residual without creating historical debt.

## Telemetry

The decision trace includes identity/timestamps, aim/bbox, measured/control errors, FAR/NEAR state, four-point history count, three `px/ms` velocities, median/EMA/spread/confidence, reference dt, configured/effective lead frames, prediction caps/offset, recoil feedforward, full correction counts, float demand, integer command, quantizer residual, zero-cross state, and block reason, plus:

```text
delivery_mode: latest_replace
scheduler_used: true
```

See `docs/control/dual_phase_atan_robust_predictive_v2.md` for the frozen V2 implementation contract and tuning boundaries. See `docs/control/crosshair_control_reference.md` for the optional visual HUD reference, confirmation policy, and geometry fallback.

## Remaining Physical Uncertainty

`prediction.lead_frames`, fixed recoil counts per observation, and caps still require Jetson + device + game trace calibration. The estimator models target motion relative to the crosshair; it does not yet separate target motion, manual camera motion, and NovaSight-induced camera motion. Strict prediction caps make that limitation tolerable for V2 but do not remove it.
