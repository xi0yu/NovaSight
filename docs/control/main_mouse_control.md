# NovaSight Main Mouse Control

Date: 2026-07-10

Status: frozen single-chain, dual-mode implementation contract.

## Route

```text
DetectionBatch
-> basic candidate filtering
-> Hungarian Tracker / Kalman
-> TargetSelector
-> observed raw aim and predicted future aim
-> trusted full control-space projection
-> predicted pixel error
-> exactly one mode controller
-> shared count protection and fractional residual
-> replaceable Scheduler plan
-> kmNet
```

Kalman is the only position predictor. There is no LOS prediction pass after Kalman.

## Aim And Prediction

```text
aim_x = bbox_left + bbox_width / 2
aim_y = bbox_top + bbox_height * control.aim.y_ratio

frame_age_s = control_now_ts_ns - capture_ts_ns
horizon_s = frame_age_s + configured_actuation_delay_s
predicted_aim = filtered_aim
  + kalman_velocity * horizon_s * prediction_strength * prediction_confidence
```

`aim.y_ratio` is clamped to `0.00..1.00`, rounded to `0.01`, and defaults to `0.22`. Recent successful device counts conservatively reduce prediction confidence; they are not a self-motion subtraction model.

## Mode Selection

`control.mode` accepts:

```text
universal_saturated
calibrated_angular
```

`ControllerFactory` creates one controller. The inactive controller is not evaluated and cannot contribute counts. A mode change recreates the controller, clears its history and fractional residual, and cancels the current Scheduler plan before the next `DetectionBatch`.

## Calibrated Angular

The aim point must be mapped from ROI coordinates to the complete control projection before angle conversion.

```text
focal_x = (control_width / 2) / tan(fov_x / 2)
fov_y = 2 * atan(tan(fov_x / 2) * control_height / control_width)
focal_y = (control_height / 2) / tan(fov_y / 2)

predicted_error_rad = atan(predicted_error_px / focal)
observed_error_rad = atan(observed_error_px / focal)
d_raw = (observed_error_rad_t - observed_error_rad_previous) / measurement_dt_s
d_ema = d_ema_alpha * d_raw + (1 - d_ema_alpha) * d_ema_previous
output_rad = Kp * predicted_error_rad + Kd * d_ema
limited_rad = clamp(output_rad, -max_angle_step_rad, max_angle_step_rad)
counts_float = limited_rad * counts_per_360 / (2*pi)
```

The first observation, target switch, and invalid measurement interval use `D=0`. D differentiates observed error only; prediction parameter changes cannot create a synthetic derivative spike.

Parameters:

```text
control.calibrated_angular.fov_x_deg
control.calibrated_angular.counts_per_360_x
control.calibrated_angular.counts_per_360_y
control.calibrated_angular.kp_x / kp_y
control.calibrated_angular.kd_x / kd_y
control.calibrated_angular.d_ema_alpha
control.calibrated_angular.max_angle_step_x_deg
control.calibrated_angular.max_angle_step_y_deg
```

## Universal Saturated

This mode maps full-control-space pixel error directly to counts and has no angle, PD, D EMA, FOV, or counts-per-360 dependency.

```text
counts_x = max_step_x_counts * (2/pi) * atan(error_x_px / response_scale_x_px)
counts_y = max_step_y_counts * (2/pi) * atan(error_y_px / response_scale_y_px)
```

Near the center it is approximately linear. At large error it approaches the configured maximum without exceeding it.

Parameters:

```text
control.universal_saturated.response_scale_x_px
control.universal_saturated.response_scale_y_px
control.universal_saturated.max_step_x_counts
control.universal_saturated.max_step_y_counts
```

## Shared Output

Both modes pass through the same sequence:

```text
mode counts
-> observed-error pixel deadzone
-> optional Y inversion
-> per-observation count slew
-> Scheduler-capacity feasible budget
-> fractional residual integer conversion
```

```text
slew_limited = previous_counts
  + clamp(requested_counts - previous_counts, -max_count_slew, max_count_slew)

total = feasible_counts + previous_fractional_residual
integer_budget = trunc(total)
fractional_residual = total - integer_budget
```

Clamped counts are discarded and never stored as hidden debt.

Shared parameters:

```text
control.shared.deadzone_x_px / deadzone_y_px
control.shared.max_count_slew_x / max_count_slew_y
control.shared.invert_y
control.scheduler_step_counts_x / scheduler_step_counts_y
control.scheduler_interval_ms
```

## Scheduler

The maximum plan duration is 24ms. Capacity is:

```text
plan_step_capacity = floor(24ms / scheduler_interval_ms) + 1
max_budget_axis = scheduler_step_counts_axis * plan_step_capacity
```

Cumulative rounding conserves the feasible integer budget. The Scheduler stores one current plan. A new frame, target switch, direction change, trigger release, stale input, device failure, runtime stop, or mode/config change cancels pending steps. Unexecuted counts are discarded.

## Migration And Removed Code

The loader performs one-way migration from the preceding flat angular schema into `control.calibrated_angular` and `control.shared`, selecting `calibrated_angular` to preserve behavior. Fresh configurations default to `universal_saturated`.

No production route exists for experimental-angle PID, integral control, magnetic assist, AimPoint position EMA, LOS angular prediction, LatencyCompensator, HID direct send, or alternate Scheduler queues.

## Deferred

Self-motion subtraction, measured send-to-visual alignment, adaptive Kalman tuning, pixel-domain D for universal mode, condition integral control, and pending-control observers are not part of this implementation.
