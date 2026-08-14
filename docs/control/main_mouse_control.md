# NovaSight Main Mouse Control

Date: 2026-08-04

Status: the production behavior is single-target aim prediction plus continuous
nonlinear Atan control.

For the current end-to-end production flow, start with
[`core_algorithm_flow.md`](core_algorithm_flow.md). This file keeps the detailed
mouse-control parameter contract.

## Mainline Route

```text
latest DetectionBatch
-> freshness and monotonic timestamp checks
-> TargetSelector / Tracker identity
-> shared bbox aim point
-> frozen crosshair/geometry reference
-> current measured ROI error
-> bounded four-point / three-segment velocity prediction
-> source/FOV/counts projection
-> continuous counts-domain Atan response
-> device-count limiter and truncating quantizer
-> capacity-one latest-replace slot
-> MouseCommandExecutor
-> at most one move(dx, dy) per output tick
```

The controller predicts only the current selected target. A new observation
replaces an unsent older command, so prediction changes the next aim error
instead of creating a queued trajectory.

## Target And Aim Inputs

Tracker identity preserves target continuity. `TargetSelector` ranks eligible
confirmed tracks using configured class priority and normalized distance. The
selected aim X is bbox center; aim Y is the configured role ratio for `head`,
`body` or `other`. Detection confidence is an eligibility threshold, not a
movement gain.

ROI detections are mapped through trusted geometry. The same resolved aim point
is used for selection diagnostics and final control, so the UI cannot display a
different target point from the controller.

## Active Control Law

```text
e_meas = current measured aim - crosshair
profile = classify_motion(last three aim-position segments)
v = profile_velocity(profile)
motion_strength = profile_strength(profile)
horizon = frame_age + actuation_delay + prediction_lead_ms
prediction = vector_clamp((v * horizon) * motion_strength, prediction_cap_px)
e_ctrl = e_meas + prediction

source_error = e_ctrl * roi_size / observation_size
theta = atan(source_error / focal_length)
full_counts = theta * counts_per_360 / (2*pi)

rho = hypot(full_counts_x, full_counts_y)
S = 256 counts
r = rho / S
curve = 1 - exp(-(r ^ response_curve_shape))
R = 1 + response_boost * curve * (0.35 + 0.65 * motion_strength)
response_gain = response_scale * R
u = response_gain * S * atan(full_counts / S)
u_x = clamp(u_x, -max_output_x_counts, max_output_x_counts)
u_y = clamp(u_y, -max_output_y_counts, max_output_y_counts)
```

`response_scale` is the base response strength. `response_boost` controls how
much extra strength appears as error grows. `response_curve_shape` controls when
that extra strength appears. `motion_strength` comes from the same motion profile
that gates prediction: static acquisition keeps 35% of the boost, stable motion
can use all of it, and reverse/peek/jittered motion receives little or none.
`S` is an internal fixed Atan scale and is not a user-facing configuration field.
Prediction is bounded before projection; the per-update count limit still owns
the final device output ceiling.

## Quantization And Latest-Replace Delivery

```text
accumulator += u
integer_count = trunc(accumulator)
accumulator -= integer_count
```

The dedicated limiter owns the per-update count ceiling, integer conversion and
fractional residual. Direction changes clear opposite-direction residual.
Trigger-inactive or blocked observations clear the limiter and cannot bank
historical movement.

The delivery slot retains one complete command. A newer observation replaces
an older unsent command. Immediately before the device call,
`MouseCommandExecutor` rechecks trigger state, freshness, generation and device
range. It does not merge pending counts or split one command into a trajectory.

## Production Configuration

```yaml
schema_version: 12
pipeline:
  p_response_scale: 0.20
  p_response_boost: 0.50
  p_response_curve_shape: 1.0
  max_output_x_counts: 127.0
  max_output_y_counts: 127.0
  prediction_enabled: true
  prediction_lead_ms: 16.0
  prediction_cap_px: 10.0
```

The Rust root schema is version 12 and uses `pipeline.prediction_enabled: true`.
Retired response fields are rejected rather than silently mapped into the new
control model.

## User-Facing Telemetry

The useful control display is limited to selected target/class, measured aim,
crosshair, current control error, prediction velocity,
prediction offset, projected full counts, Atan demand, integer command,
delivery state and block reason.

## Calibration Boundary

FOV, `counts_per_360`, response strength, response curve, Atan scale,
per-update limits and recoil counts need Jetson + kmNet + game trace
calibration. These values must be evaluated independently from capture/inference
latency and transport capacity.
