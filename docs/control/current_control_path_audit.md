# NovaSight Current Mouse Control Path Audit

Date: 2026-07-10

Status: current implementation audit. This document describes the only production mouse algorithm route.

## Current Mainline

```text
CapturedFrame / DetectionBatch
-> latest-only freshness gates
-> ROI detections
-> RuntimeTargetSelector
-> RuntimeTracker + runtime KalmanEstimator
-> current raw bbox
-> observed aim from raw bbox
-> Kalman future bbox center
-> predicted aim from future bbox
-> full control-space projection
-> P(predicted angle error) + D(observed angle error derivative EMA)
-> per-axis angle and angle-rate limits
-> calibrated floating-point counts
-> fractional residual
-> feasible integer observation budget
-> CommandScheduler bounded steps
-> KmNetExecutor
```

There is no `legacy`, `experimental_angle_pid`, LOS angular prediction, AimPoint EMA, or LatencyCompensator send path.

## Source And Time Contracts

- Capture sources assign monotonic `capture_ts_ns`; inference and control preserve it through `DetectionBatch` and `FrameContext`.
- `measurement_dt_s` is the capture timestamp difference between consecutive observations of the same target. It feeds only the observed-error derivative.
- `frame_age_s` is `control_now_ts_ns - capture_ts_ns`.
- `configured_actuation_delay_s` is a configured estimate, not a measured physical delay.
- Kalman extrapolation horizon is `frame_age_s + configured_actuation_delay_s`.
- Scheduler pacing uses `scheduler_interval_ms`; it is not used for target motion or D.

Implementation:

- `novasight/runtime/control_timing.py`
- `novasight/runtime/service.py::_record_control_timing`
- `novasight/runtime/service.py::_mouse_observation_metadata`

## Coordinate Contract

Detection boxes are ROI coordinates. `RawAimPointProjector` requires trusted capture geometry and a `CoordinateTransform`, then maps:

```text
ROI point -> capture point -> complete control projection point
```

The aim point is fixed as:

```text
observed_x = bbox_left + bbox_width / 2
observed_y = bbox_top + bbox_height * control.aim.y_ratio
```

`aim.y_ratio` is clamped to `[0.00, 1.00]`, rounded to two decimals, and defaults to `0.22`. No horizontal offset or position-time EMA exists.

The same bbox-relative anchor is applied to the Kalman-predicted bbox center. Prediction changes where the selected anchor is expected to move; it never changes the user's anchor ratio.

Implementation:

- `novasight/control/observation.py`
- `novasight/coordinates.py`
- `novasight/runtime/service.py::_control_frame_metadata`

## Prediction Contract

The runtime Kalman estimator is the only position predictor. It estimates screen-space bbox center and velocity. That state contains target motion, camera-induced motion, and detection noise; it is not target-world velocity.

For each axis:

```text
predicted_center
= raw_bbox_center
   + kalman_velocity
   * prediction_horizon_s
   * prediction_strength
   * prediction_confidence
```

The current raw bbox center is the prediction baseline; Kalman contributes only the future displacement. Therefore `prediction_strength=0` or zero prediction confidence returns exactly to the observed aim rather than retaining hidden Kalman position smoothing.

`prediction_confidence` combines Kalman/identity confidence with a conservative suppression factor derived from successful device counts sent in the recent 40ms window. Counts are recorded only after the device executor reports `sent=true`.

```text
recent_abs_counts_40ms = sum(abs(dx) + abs(dy))
velocity_confidence = clamp(1 - recent_abs_counts_40ms / 80, 0, 1)
```

The 80-count threshold is a conservative implementation constant, not a calibrated physical value.

The 20ms, 40ms, and 60ms executed-count windows are telemetry. They are not a self-motion subtraction model. Exact count-to-visual alignment remains deferred until send-to-visual delay is measured.

## PD Contract

The controller receives two points but produces one output:

```text
P input: predicted aim angle error
D input: derivative of observed aim angle error
```

```text
d_raw = (observed_error_t - observed_error_previous) / measurement_dt_s
d_ema = d_ema_alpha * d_raw + (1 - d_ema_alpha) * d_ema_previous
u = Kp * predicted_error + Kd * d_ema
```

The D term never differentiates predicted error. Prediction parameter changes therefore cannot create a synthetic D spike. A predicted-only/missing frame invalidates observed D history; the first real observation after the gap re-establishes the baseline with zero D.

Implementation: `novasight/control/mouse.py`.

## Counts And Scheduler Contract

```text
counts_float = limited_output_rad * counts_per_360 / (2*pi)
```

Y inversion is applied in the count mapper. Fractional residual stores only the sub-count fraction from integer conversion. Counts removed by feasible-budget clamping are discarded and never become hidden residual debt.

The scheduler plan capacity is derived from a fixed 24ms maximum plan duration and `scheduler_interval_ms`. Expiry adds one tick of timing tolerance so the last step scheduled at the 24ms boundary is not discarded. The controller caps each observation budget to:

```text
scheduler_step_counts_axis * plan_step_capacity
```

`CommandScheduler._split_steps()` uses cumulative rounding, so integer steps sum exactly to the feasible submitted budget and remain under the per-step limit.

New observations submit under the same runtime lock used by Scheduler ticks. A new frame replaces the pending plan before another old step can be consumed. Trigger release, target loss, stale input, calibration changes, runtime stop, and device errors clear pending steps.

Implementation:

- `novasight/control/scheduler.py`
- `novasight/executors/runtime.py`
- `novasight/runtime/service.py::process_frame`
- `novasight/runtime/service.py::process_control_tick`

## Removed Routes

The following old implementations were removed because they could create duplicate prediction, duplicate smoothing, or alternate send semantics:

```text
novasight/runtime/aim.py
novasight/control/angular.py
novasight/control/controller.py
novasight/control/latency_compensator.py
novasight/control/hid_output.py
novasight/core/control_loop.py
novasight/detection/kalman_estimator.py
novasight/detection/target_selector.py
```

Configuration no longer accepts strategy selection, experimental-angle fields, AimPoint EMA fields, legacy latency compensation fields, or frame-count target-switch fields.

## Current Answers

- Capture timestamp source: monotonic userspace frame receipt, preserved end to end.
- PD dt: adjacent same-target capture timestamp difference.
- Mouse output: one feasible integer budget per new observation, split by Scheduler.
- Repeated old observation control: no; control ticks consume pending steps only and never recalculate from `last_frame_context`.
- Old plan handling: every new observation replaces pending steps; unexecuted counts are discarded.
- Mainline count: one.

`KmNetExecutor.diagnostic_move()` remains available to the CLI and executor diagnostic API. It is an explicit hardware test command, not a detection-driven mouse algorithm route, and it does not share controller state with the production loop.

## Remaining Physical Uncertainty

The least certain value is the physical endpoint represented by `configured_actuation_delay_s`. Capture timestamps mark userspace receipt, and successful send timestamps do not reveal when the game consumes input or when that movement appears in a captured frame.

The largest remaining control-model limitation is self-motion contamination of Kalman screen velocity. Recent successful counts conservatively reduce prediction, but they do not estimate and subtract camera-induced screen motion. That compensation must remain out of the live predictor until send-to-visual alignment is measured.
