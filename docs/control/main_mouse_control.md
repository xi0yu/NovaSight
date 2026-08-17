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
-> hardware trigger continuous-hold delay
-> current measured ROI error
-> four-point / three-segment vector-medoid velocity prediction
-> source/FOV/counts projection
-> continuous counts-domain Atan response
-> truncating device-count quantizer
-> capacity-one latest-replace slot
-> recoil composition
-> fixed X/Y device-boundary clamp
-> PointerDevice::send
-> at most one move(dx, dy) per output tick
```

The controller predicts only the current selected target. A new observation
replaces an unsent older command, so prediction changes the next aim error
instead of creating a queued trajectory.

In hardware-trigger mode, capture, inference, tracking and target selection
continue while the button is held below `fire_delay_ms`, but the control worker
does not call `AimAlgorithm::step`. Once the held duration exceeds the threshold,
only the newest target observation starts a fresh prediction/control state.

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
v1, v2, v3 = adjacent aim-position velocities
v = vector_medoid(v1, v2, v3)
horizon = frame_age + actuation_delay + prediction_lead_ms
prediction = vector_clamp(v * horizon, prediction_cap_px)
e_ctrl = e_meas + prediction

source_error = e_ctrl * roi_size / observation_size
theta = atan(source_error / focal_length)
full_counts = theta * counts_per_360 / (2*pi)

rho = hypot(full_counts_x, full_counts_y)
S = 256 counts
r = rho / S
curve = 1 - exp(-(r ^ response_curve_shape))
R = 1 + response_boost * curve
response_gain = response_scale * R
u = response_gain * S * atan(full_counts / S)
tracking = truncating_quantize(u)
out_x = clamp(tracking_x, -max_output_x_counts, max_output_x_counts)
out_y = clamp(tracking_y + recoil_y, -max_output_y_counts, max_output_y_counts)
```

`response_scale` is the base response strength. `response_boost` controls how
much extra strength appears as error grows. `response_curve_shape` controls when
that extra strength appears. Prediction changes only the future aim point and
does not modify response gain.
`S` is an internal fixed Atan scale and is not a user-facing configuration field.
Prediction is bounded before projection; fixed per-axis limits own the final
device output ceiling after recoil is composed.

## Quantization And Latest-Replace Delivery

```text
accumulator += u
integer_count = trunc(accumulator)
accumulator -= integer_count
```

The quantizer owns integer conversion and fractional residual. Direction
changes clear opposite-direction residual. The only count ceiling is the final
fixed X/Y clamp after recoil composition.
Trigger-inactive or blocked observations clear the limiter and cannot bank
historical movement.

The delivery slot retains one complete command. A newer observation replaces
an older unsent command. Immediately before the device call,
the device worker rechecks trigger state, generation, output gate and device
range. It does not merge pending counts or split one command into a trajectory.

## Production Configuration

```yaml
schema_version: 16
pipeline:
  p_response_scale: 0.20
  p_response_boost: 0.50
  p_response_curve_shape: 1.0
  max_output_x_counts: 127.0
  max_output_y_counts: 127.0
  fire_delay_enabled: false
  fire_delay_ms: 0
  prediction_enabled: true
  prediction_lead_ms: 16.0
  prediction_cap_px: 10.0
```

The Rust root schema is version 16 and uses `pipeline.prediction_enabled: true`.
Retired response fields are rejected rather than silently mapped into the new
control model.

Timestamp-free replay uses a fixed five-frame identity-loss grace inside the
targeting module. Production control uses only
`pipeline.target_track_max_lost_age_ms`, so replay frame count is not a product
configuration parameter.

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
