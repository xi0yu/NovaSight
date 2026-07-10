# NovaSight Predictive PIDv2 Implementation Plan

Date: 2026-07-10

Status: canonical staged plan for the `predictive_pid_v2` control path.

This document supersedes the original stage 2 and stage 3 definitions. Stage 0 and stage 1 remain complete. No later stage may treat screen-space displacement as pure target-world motion.

## 1. Correct State Definition

Two consecutive aim points provide screen line-of-sight velocity:

```text
measurement_dt_s
= current_capture_ts_ns - previous_capture_ts_ns

observed_screen_velocity_px_s
= (current_aim_px - previous_aim_px) / measurement_dt_s
```

This observation contains three components:

```text
observed_screen_velocity
= relative_target_motion
+ camera_induced_motion
+ detection_and_tracking_noise
```

The equation defines components, not fixed signs. Signs depend on image coordinates, mouse directions, calibration axis signs, and game camera conventions.

Under the common horizontal convention:

```text
screen x right = positive
mouse dx right = positive
camera turns right
stationary world target moves left on screen
```

Therefore positive executed X counts commonly produce negative screen displacement for a stationary target. This relationship must be verified by calibration and tests; it must not be guessed inside the estimator.

## 2. Canonical Terminology

The new path must use:

```text
observed_screen_velocity_px_s
raw_observed_vx_px_s
raw_observed_vy_px_s
filtered_observed_vx_px_s
filtered_observed_vy_px_s
velocity_confidence
configured_extra_prediction_delay_ms
```

The new path must not use:

```text
target_velocity
target_vx
target_vy
configured_actuation_delay_ms
estimated_actuation_delay_ms
```

`estimated_relative_target_velocity` is reserved for a future estimator that subtracts a time-aligned self-motion estimate. Legacy `Track.velocity_px_s` fields remain compatibility fields, but their documented meaning is screen-space velocity.

## 3. Revised Mainline

```text
DetectionBatch
-> TargetContinuityGuard
-> ScreenMotionEstimator
-> ObservedVelocityEmaFilter
-> ExecutedControlTelemetry
-> VelocityConfidenceModel
-> ConservativePredictionModel
-> ErrorProjector
-> PIDv2Controller
-> CountMapper
-> OutputLimiter
-> ControlPlanBuilder
-> ControlScheduler
-> HID / KMBOX
-> new observation closes the loop
```

Self-motion compensation remains shadow-only until command-to-frame alignment is measured.

## 4. Time Semantics

The first-version approximation is:

```text
frame_age_s
= control_now_ts_ns - capture_ts_ns

prediction_horizon_s
= frame_age_s + configured_extra_prediction_delay_s
```

`capture_ts_ns` currently marks userspace frame receipt, not the source image's physical presentation or capture-card sample time. `configured_extra_prediction_delay_ms` is additional prediction lead. It is not a claim that end-to-end actuation delay has been measured.

The physical quantity that later calibration should approximate is:

```text
physical_prediction_horizon
= visual_effect_time - source_image_time
```

## 5. Revised Stage Sequence

### Stage 0: Current Path Audit

Status: complete.

Artifact: `docs/control/current_control_path_audit.md`.

### Stage 1: Shared Time Model And Telemetry

Status: complete, with terminology corrected after review.

Implemented semantics:

```text
capture_ts_ns
inference_end_ts_ns
control_now_ts_ns
measurement_dt_ms
frame_age_ms
configured_extra_prediction_delay_ms
prediction_horizon_ms
```

The active mouse behavior remains legacy.

### Stage 2: Screen Line-of-Sight Velocity Estimation

Goal: estimate and smooth observed screen motion without claiming it is target-world velocity.

Add:

```text
ScreenMotionEstimator
ObservedVelocityEmaFilter
```

Required output:

```text
raw_observed_vx_px_s
raw_observed_vy_px_s
filtered_observed_vx_px_s
filtered_observed_vy_px_s
observed_velocity_valid
observed_velocity_reset_reason
```

Required guards:

```text
same stable target id
monotonic capture timestamp
min/max measurement dt
finite coordinates
maximum observed screen speed
target switch/loss reset
configuration/runtime reset
```

Stage 2 runs in shadow mode and must not alter control output.

### Stage 2.5: Executed Control Self-Motion Telemetry

Goal: record actual successful device sends needed to assess self-induced screen motion.

Only successful executor sends count. Do not substitute desired, planned, queued, cancelled, or remaining counts.

Add an append-only bounded time window of executed samples:

```text
ExecutedCountSample:
  scheduler_send_ts_ns
  device_send_start_ts_ns
  device_send_end_ts_ns
  executed_dx
  executed_dy
  source_frame_id
  source_target_id
  plan_id_optional
  step_index_optional
  executor_id
  send_succeeded
```

Recent-window aggregation uses `device_send_end_ts_ns` for successful sends. `scheduler_send_ts_ns` and device start/end timestamps remain separate so Scheduler wait and executor latency are observable. A legacy command without `plan_id` or `step_index` may leave those correlation fields unset; it must not invent values.

Observation telemetry:

```text
executed_counts_x_since_previous_observation
executed_counts_y_since_previous_observation
executed_counts_x_last_20ms
executed_counts_y_last_20ms
executed_counts_x_last_40ms
executed_counts_y_last_40ms
executed_counts_x_last_60ms
executed_counts_y_last_60ms
```

The previous-observation interval is:

```text
[previous_capture_ts_ns, current_capture_ts_ns]
```

The implementation must also retain recent windows because an executed command may not yet be visible in the current frame.

Stage 2.5 records data only. It must not subtract counts from position or velocity.

### Stage 3: Conservative Prediction

Goal: produce a bounded predicted reference point in shadow mode.

Formula:

```text
predicted_position
= observed_position
+ filtered_observed_screen_velocity
   * prediction_horizon
   * velocity_confidence
   * prediction_scale
```

Required properties:

```text
short bounded prediction horizon
prediction_scale uses a conservative value below 1.0 initially
fixed and bbox-relative prediction offset limits
stale observations disable prediction
target switch clears velocity
recent executed control suppresses velocity confidence
```

`velocity_confidence` uses actual executed-count windows and time-based thresholds, not a fixed number of frames. The initial model may be piecewise or clamped linear, but all threshold values remain provisional until trace validation.

Example shape, not a final calibration:

```text
recent_counts_magnitude = norm(executed_counts_last_40ms)
velocity_confidence
= clamp(1 - recent_counts_magnitude / confidence_zero_counts, 0, 1)
```

Stage 3 must not send the predicted position directly to the device.

### Stage 3.5: Self-Motion Compensation Shadow Mode

Goal: compare observed displacement with estimated camera-induced displacement without affecting control.

Counts to angle:

```text
camera_angle_rad
= executed_counts * 2*pi / counts_per_360
```

For a common horizontal sign convention, a camera-right turn moves a stationary target left. The actual sign must use calibration and a tested projection convention.

Small-angle screen displacement:

```text
self_delta_x_px ~= -focal_x_px * camera_angle_x_rad
```

General horizontal projection:

```text
x_prime
= focal_x_px
   * tan(atan(x / focal_x_px) - camera_angle_x_rad)

estimated_self_delta_x_px = x_prime - x
```

Estimated relative displacement:

```text
observed_delta_px = current_position_px - previous_position_px
estimated_relative_delta_px
= observed_delta_px - estimated_self_delta_px
```

Required telemetry:

```text
observed_delta_x_px
observed_delta_y_px
estimated_self_delta_x_px
estimated_self_delta_y_px
estimated_relative_delta_x_px
estimated_relative_delta_y_px
observed_screen_velocity_x_px_s
observed_screen_velocity_y_px_s
estimated_self_velocity_x_px_s
estimated_self_velocity_y_px_s
estimated_relative_velocity_x_px_s
estimated_relative_velocity_y_px_s
```

The main unresolved problem is deciding which executed counts have affected a given captured image. Until send-to-visual alignment is measured, `estimated_relative_velocity` must remain shadow-only and must not feed formal prediction.

### Stage 4: Error Projection

Implement `ErrorProjector` using one verified control coordinate space, horizontal FOV, derived vertical FOV, focal lengths, and `atan` pixel-to-angle mapping.

### Stage 5: PIDv2 P+D

Implement P plus derivative EMA with `Ki=0`. D is closed-loop damping based on angle error change over `measurement_dt_s`; it is not target velocity feed-forward.

### Stage 6: Count Mapping

Implement calibrated X/Y angle-to-count conversion, axis direction, scale, and fractional residual.

### Stage 7: Output Limits

Implement per-observation budget, per-tick limit, and count-output change-rate limit.

### Stage 8: Control Plan Builder

Split one observation budget into aligned integer X/Y steps whose sum exactly equals the limited budget.

### Stage 9: Control Scheduler

Use one replaceable current plan, never an accumulating command queue. Record every successful device send for stage 2.5 telemetry.

### Stage 10: Mainline Integration

Add `legacy | predictive_pid_v2`, defaulting initially to legacy. Enable the new path only after shadow and low-risk device verification.

## 6. Mandatory Offline Trajectories

### Stationary target, no mouse output

Expected:

```text
observed_screen_velocity ~= 0
prediction_offset ~= 0
```

### Stationary target, mouse moves right

Expected under the common sign convention:

```text
observed target moves left
raw_observed_vx_px_s < 0
velocity_confidence decreases
prediction does not fully interpret the displacement as target motion
```

### Target and camera both move right

Target-relative and camera-induced motion may partially cancel, so observed screen velocity may be near zero while target-world motion is not zero. The test proves screen velocity is not target-world velocity.

### Stationary target with alternating control output

Inspect observed velocity, confidence, prediction, D term, Scheduler output, center crossings, and sign reversals for self-excited oscillation.

## 7. Stage Gates

- Stage 2 may not expose fields named `target_velocity`.
- Stage 2.5 may count only successful device sends.
- Stage 3 prediction remains shadow-only until the four self-motion trajectories pass.
- Stage 3.5 remains shadow-only until executed counts can be aligned with visual feedback.
- No predicted point may bypass ErrorProjector, PIDv2, CountMapper, OutputLimiter, and Scheduler.
- Each stage ends with a report and explicit confirmation before the next stage begins.

## 8. Deferred Work

Still deferred:

```text
constant-acceleration prediction
complex behavior classification
adaptive PID
automatic parameter search
reinforcement learning
strong integral control
multi-target parallel control
automatic device-delay identification
formal pending-control observer in the live loop
game-physics trajectory simulation
```
