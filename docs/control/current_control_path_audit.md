# NovaSight Current Mouse Control Path Audit

Date: 2026-07-10

Status: current implementation audit. This document describes one production chain with two mutually exclusive control-mapping modes.

## Current Mainline

```text
CapturedFrame / DetectionBatch
-> latest-only freshness gates
-> ROI detections
-> class + confidence basic candidate filter
-> raw aim TrackObservation
-> RuntimeTracker Kalman prediction + normalized-distance gate + Hungarian assignment
-> ACTIVE tracks only
-> RuntimeTargetSelector
-> current raw bbox
-> observed aim from raw bbox
-> future aim from filtered Kalman aim + velocity
-> full control-space projection
-> predicted pixel error
-> ControllerFactory selects exactly one mapping:
   calibrated_angular: full-space angle -> observed D EMA -> PD -> counts
   universal_saturated: atan pixel saturation -> counts
-> shared deadzone, direction, count slew, and feasible budget
-> fractional residual
-> feasible integer observation budget
-> CommandScheduler bounded steps
-> KmNetExecutor
```

There is no `legacy`, `experimental_angle_pid`, LOS angular prediction, AimPoint EMA, LatencyCompensator, integral, or magnetic-assist send path. The two mapping modes cannot both execute for one observation.

The shipped Jetson config uses `source.default: capture`. Application startup restores the configured capture device in a background thread, then starts `RuntimePipeline` only after capture is available and the active TensorRT model reports `loaded=true`. kmNet connection is started independently and does not block API startup.

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

The same raw aim point is the Tracker measurement. Prediction moves that selected aim point through the Tracker's filtered position and velocity; it never chooses a different bbox-relative anchor.

Implementation:

- `novasight/control/observation.py`
- `novasight/coordinates.py`
- `novasight/runtime/service.py::_control_frame_metadata`

## Tracker Contract

The basic pre-Tracker filter applies only selectable class, the single minimum-confidence threshold, and mandatory finite bbox validation. FOV, candidate quality, class priority, target lock, and switch delay remain TargetSelector responsibilities after tracking.

Each observation freezes the current user aim point:

```text
aim_x = (bbox_x1 + bbox_x2) / 2
aim_y = bbox_y1 + bbox_height * aim_y_ratio
```

The Tracker Kalman state is `[aim_x, aim_y, vx, vy]`. Its `dt` is the difference between adjacent capture timestamps. Association predicts every retained track to the current capture timestamp, rejects class mismatches and pairs beyond the normalized distance gate, then solves the remaining global minimum-cost assignment with the Hungarian algorithm:

```text
normalized_distance = aim_distance / max(previous_bbox_height, detection_bbox_height)
cost = 0.75 * normalized_distance + 0.25 * (1 - IoU)
```

New tracks are `ACTIVE` immediately. An unmatched track becomes `LOST` and remains available only for future association. It is never emitted to TargetSelector or mouse control. A successful later match restores the same `track_id`; more than `max_missed_frames` consecutive misses removes the track.

The configured Kalman NIS, covariance, missing-time, identity-confidence, and prediction-confidence gates are also the gates used by the estimate consumed by mouse prediction. There is no separate permissive association-only Kalman configuration that silently bypasses those values.

Implementation:

- `novasight/runtime/candidates.py::BasicCandidateFilter`
- `novasight/runtime/tracker.py`
- `novasight/runtime/target_selector.py`

## Prediction Contract

The runtime Kalman estimator is the only position predictor. It estimates the screen-space aim point and velocity. That state contains target motion, camera-induced motion, and detection noise; it is not target-world velocity.

For each axis:

```text
predicted_aim
= filtered_aim
   + kalman_velocity
   * prediction_horizon_s
   * prediction_strength
   * prediction_confidence
```

The current filtered Kalman aim is the prediction baseline. Therefore `prediction_strength=0` disables future displacement but retains the Tracker's current measurement filtering. The raw bbox aim remains separately available for the observed D input.

`prediction_confidence` combines Kalman/identity confidence with a conservative suppression factor derived from successful device counts sent in the recent 40ms window. Counts are recorded only after the device executor reports `sent=true`.

```text
recent_abs_counts_40ms = sum(abs(dx) + abs(dy))
velocity_confidence = clamp(1 - recent_abs_counts_40ms / 80, 0, 1)
```

The 80-count threshold is a conservative implementation constant, not a calibrated physical value.

The 20ms, 40ms, and 60ms executed-count windows are telemetry. They are not a self-motion subtraction model. Exact count-to-visual alignment remains deferred until send-to-visual delay is measured.

## Controller Contracts

`control.mode` is either `calibrated_angular` or `universal_saturated`. `ControllerFactory` creates one implementation when the runtime starts or configuration changes. Runtime reconfiguration first clears the old Scheduler plan and controller state.

### Calibrated Angular

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

The D term never differentiates predicted error. Prediction parameter changes therefore cannot create a synthetic D spike. A missing frame emits no target and resets control state; the first restored real observation re-establishes the D baseline with zero derivative carry-over.

Implementation: `novasight/control/mouse.py`.

### Universal Saturated

```text
counts_axis
= max_step_axis_counts
  * (2 / pi)
  * atan(predicted_error_axis_px / response_scale_axis_px)
```

This mode has no FOV, counts-per-360, PD, D history, or angle state. Small errors are approximately linear and large errors approach the configured count limit.

## Counts And Scheduler Contract

```text
calibrated_angular:
  counts_float = limited_output_rad * counts_per_360 / (2*pi)

universal_saturated:
  counts_float = max_counts * (2/pi) * atan(error_px / response_scale_px)
```

Both modes then enter the same deadzone, Y inversion, count-slew, residual, and feasible-budget path. Fractional residual stores only the sub-count fraction from integer conversion. Counts removed by feasible-budget clamping are discarded and never become hidden residual debt.

The scheduler plan capacity is derived from a fixed 24ms maximum plan duration and `scheduler_interval_ms`. Expiry adds one tick of timing tolerance so the last step scheduled at the 24ms boundary is not discarded. The controller caps each observation budget to:

```text
scheduler_step_counts_axis * plan_step_capacity
```

`CommandScheduler._split_steps()` uses cumulative rounding, so integer steps sum exactly to the feasible submitted budget and remain under the per-step limit.

New observations only replace the single pending plan; they never call the device. The continuous Scheduler tick is the only production device-send owner. Scheduler mutation is serialized separately from device I/O, and the potentially blocking kmNet call does not hold the runtime control-state lock. A new frame replaces pending old steps; a step already handed to the device is recorded as executed or failed and is never placed back into residual debt. Trigger release, target loss, stale input, calibration changes, runtime stop, and device errors clear pending steps.

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
novasight/hardware/kmbox_net.py
novasight/hardware/factory.py
novasight/hardware/heartbeat.py
novasight/hardware/makcu.py
novasight/runtime/detection_batch_mailbox.py
novasight/capture/capture_loop.py
```

The current configuration schema exposes only `control.mode` plus nested `calibrated_angular`, `universal_saturated`, and `shared` settings. Experimental-angle fields, AimPoint EMA fields, legacy latency compensation fields, integral state, and magnetic-assist fields exist only in one-way load migration or are discarded.

## Current Answers

- Capture timestamp source: monotonic userspace frame receipt, preserved end to end.
- PD dt: adjacent same-target capture timestamp difference.
- Mouse output: one feasible integer budget per new observation, split by Scheduler.
- Repeated old observation control: no; control ticks consume pending steps only and never recalculate from `last_frame_context`.
- Old plan handling: every new observation replaces pending steps; unexecuted counts are discarded.
- Production chain count: one. Control-mapping modes: two, mutually exclusive.
- Device-send owner count: one, the Scheduler tick thread.
- Hardware trigger reads: cached kmNet monitor samples; control ticks do not issue synchronous button RPCs.

`KmNetExecutor.diagnostic_move()` remains available to the CLI and executor diagnostic API. It is an explicit hardware test command, not a detection-driven mouse algorithm route, and it does not share controller state with the production loop.

## Capture Content Evidence

The NVMM path proves resource type, caps, dimensions, frame/generation monotonicity, timestamp age, overwrite count, and inference execution. It does not by itself prove that pixels contain a valid nonblack source image. `GstResourceFrameSource` keeps `CapturedFrame.image=None` for inference and now receives a separate, leaky `preview_sink` branch from the same GStreamer pipeline. The branch holds only the latest CPU-readable preview sample; MJPEG mapping and encoding happen when a preview consumer requests it and never become TensorRT input.

Capture telemetry now reports:

```text
content_validation_status: not_integrated
content_validation_reason: NVMM zero-copy frame has no GPU luma/variance probe
preview_available: true after the preview branch emits its first snapshot
preview_reason: empty after successful MJPEG output
```

The content-validation values are deliberate unavailable states, not capture failures. The preview makes the real ROI visible to an operator, but automatic black/frozen-frame detection still requires a native GPU luma/variance or sampled-thumbnail probe and thresholds validated on the Jetson capture card. Until that exists, frame arrival and successful inference must not be described as proof that the source image is visually correct.

## Remaining Physical Uncertainty

The least certain value is the physical endpoint represented by `configured_actuation_delay_s`. Capture timestamps mark userspace receipt, and successful send timestamps do not reveal when the game consumes input or when that movement appears in a captured frame.

The largest remaining control-model limitation is self-motion contamination of Kalman screen velocity. Recent successful counts conservatively reduce prediction, but they do not estimate and subtract camera-induced screen motion. That compensation must remain out of the live predictor until send-to-visual alignment is measured.

The second physical uncertainty is capture content validity. No code path currently distinguishes a valid dark game scene from a disconnected capture-card black frame on the NVMM data plane.
