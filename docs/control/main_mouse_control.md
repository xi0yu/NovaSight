# NovaSight Main Mouse Control

Date: 2026-07-10

Status: frozen single-route implementation contract.

## Route

```text
DetectionBatch
-> target selection and continuity
-> current raw bbox
-> runtime Kalman update and future center prediction
-> observed and predicted bbox-relative aim points
-> trusted full control-space projection
-> P(predicted angle error) + D EMA(observed angle error derivative)
-> angle and angle-rate limits
-> rad-to-count calibration
-> fractional residual and feasible integer budget
-> replaceable Scheduler plan
-> kmNet
```

Kalman is the only position predictor. There is no LOS prediction pass after Kalman.

## Aim

```text
aim_x = bbox_left + bbox_width / 2
aim_y = bbox_top + bbox_height * aim_y_ratio
```

`aim_y_ratio` has range `0.00..1.00`, precision `0.01`, and default `0.22`.

Observed aim uses the current raw bbox. Predicted aim uses the Kalman future center and the same current bbox width, height, and `aim_y_ratio`.

## Projection

The aim point must be mapped from ROI coordinates to the complete control projection before angle conversion.

```text
focal_x = (control_width / 2) / tan(fov_x / 2)
focal_y = focal_x
error_x_rad = atan((aim_x - center_x) / focal_x)
error_y_rad = atan((aim_y - center_y) / focal_y)
```

`fov_x_deg` is restricted to `30.0..179.0`.

## Prediction

```text
frame_age_s = control_now_ts_ns - capture_ts_ns
horizon_s = frame_age_s + configured_actuation_delay_s
prediction_confidence = kalman_confidence * identity_confidence * velocity_confidence
predicted_center = raw_bbox_center + kalman_velocity * horizon_s * prediction_strength * prediction_confidence
```

`velocity_confidence` is an adaptive safety value based on successful counts sent during the recent 40ms window. It suppresses velocity extrapolation only; it does not disable the P controller.

```text
velocity_confidence = clamp(1 - sum(abs(dx) + abs(dy))_last_40ms / 80, 0, 1)
```

The 80-count zero-confidence threshold is an internal conservative constant, not a measured calibration.

## PD

```text
d_raw = (observed_error_t - observed_error_previous) / measurement_dt_s
d_ema = d_ema_alpha * d_raw + (1 - d_ema_alpha) * d_ema_previous
output_rad = Kp * predicted_error_rad + Kd * d_ema
```

The first observation, target switch, invalid `measurement_dt`, and first real observation after a predicted-only gap use `D=0`.

## Output

Per axis:

```text
limited_rad = clamp(output_rad, -max_output_rad, max_output_rad)
max_delta_rad = max_output_rate_rad_s * measurement_dt_s
rate_limited_rad = previous_rad + clamp(limited_rad - previous_rad, -max_delta_rad, max_delta_rad)
counts_float = rate_limited_rad * counts_per_360 / (2*pi)
```

Y inversion occurs after angle-to-count conversion.

Residual handling:

```text
total = counts_float + previous_fractional_residual
requested_integer = trunc(total)
fractional_residual = total - requested_integer
feasible_integer = clamp_to_scheduler_capacity(requested_integer)
```

Clamped integer counts are not stored as debt.

## Scheduler

The internal maximum plan duration is 24ms. Capacity is derived from the configured step interval:

```text
plan_step_capacity = floor(24ms / scheduler_interval_ms) + 1
max_budget_axis = scheduler_step_counts_axis * plan_step_capacity
```

Steps use cumulative rounding and conserve the feasible integer budget. The scheduler stores one current plan. New frame, target switch, direction change, trigger release, stale input, device failure, and runtime stop cancel pending steps.

Plan expiry includes one additional Scheduler interval as timing tolerance; this does not add another planned step or extend the 24ms step schedule.

## User Parameters

```text
control.aim.y_ratio
control.configured_actuation_delay_s
control.prediction_strength
control.prediction_x_enabled
control.prediction_y_enabled
control.kp_x
control.kp_y
control.kd_x
control.kd_y
control.d_ema_alpha
control.deadzone_px_x
control.deadzone_px_y
control.max_output_rad_x
control.max_output_rad_y
control.max_output_rate_rad_s_x
control.max_output_rate_rad_s_y
control.scheduler_step_counts_x
control.scheduler_step_counts_y
control.scheduler_interval_ms
control.target_fov_radius_px
control.min_confidence
control.target_switch_delay_ms
control.lost_target_timeout_ms
calibration.fov_x_deg
calibration.counts_per_360_x
calibration.counts_per_360_y
calibration.invert_y
```

## Legacy Configuration Migration

The runtime loader accepts the immediately preceding production schema and
migrates its aim ratio, configured delay, PD gains, derivative EMA, deadzone,
Scheduler interval/step limits, stale threshold, and calibration fields into
this route. `calibration.axis_sign_y` maps to `calibration.invert_y`.

`calibration.axis_sign_x=-1` is rejected with an explicit error because this
route has no X-axis inversion setting. Silently discarding it would reverse the
closed-loop control direction. Removed experimental fields that have no valid
single-route equivalent are discarded during this one-way in-memory migration.
Legacy `hardware.flip_dy` is also discarded: the preceding kmNet executor
accepted that setting but forced it to `False`, so mapping it to `invert_y`
would change actual device behavior during upgrade.

## Reset Rules

- Target switch: reset observed-error history, D EMA, output history, residual, and pending plan.
- Target unavailable: reset controller and cancel plan.
- Stale DetectionBatch: reset runtime control state and cancel plan.
- Trigger release: reset controller and cancel plan.
- Device error: cancel plan and enter Scheduler cooldown.
- Runtime stop or fatal error: reset all control state and cancel plan.

## Deferred

Self-motion subtraction, measured send-to-visual alignment, adaptive Kalman tuning, condition integral control, and pending-control observers are not part of this route.
