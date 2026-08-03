# NovaSight Main Mouse Control

Date: 2026-07-26

Status: `dual_phase_atan_robust_predictive_v2` remains the serialized mainline
ID for compatibility. Its production behavior is single-target prediction plus
dual-phase Atan control.

## Mainline Route

```text
latest DetectionBatch
-> freshness and monotonic timestamp checks
-> TargetSelector / Tracker identity
-> shared bbox aim point
-> frozen crosshair/geometry reference
-> current measured ROI error
-> bounded four-point / three-segment velocity prediction
-> FAR / NEAR transition weight
-> source/FOV/counts projection
-> continuously blended counts-domain Atan response
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
v = average(last three aim-position segments)
horizon = frame_age + actuation_delay + adaptive_motion_lead
prediction = clamp((v * horizon + weak_acceleration_correction) * confidence_gate, motion_aware_cap)
e_ctrl = e_meas + prediction

source_error = e_ctrl * roi_size / observation_size
theta = atan(source_error / focal_length)
full_counts = theta * counts_per_360 / (2*pi)

w_far = smoothstep(distance, 0.75 * near_threshold, 1.25 * near_threshold)
kp = (1 - w_far) * near_kp + w_far * far_kp
u = kp * atan_scale * atan(full_counts / atan_scale)
limit = (1 - w_far) * near_limit + w_far * far_limit
u = clamp(u, -limit, limit)
```

The configured radial-error threshold is the center of a continuous transition,
not a hard mode switch. Outside the 75%-125% transition band, the original
NEAR or FAR response is unchanged. Prediction is bounded before projection;
the per-update count limit still owns the final device output ceiling.

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
schema_version: 9
pipeline:
  near_threshold_px: 12.0
  atan_scale_counts: 256.0
  far_kp: 0.30
  far_max_counts_per_update: 127.0
  near_kp: 0.20
  near_max_counts_per_update: 72.0
  prediction_enabled: true
  prediction_lead_frames: 1.0
  prediction_far_absolute_cap_px: 10.0
  prediction_near_absolute_cap_px: 3.0
```

The Rust root schema is version 9 and uses `pipeline.prediction_enabled: true`.
Older generated response profiles are migrated to the responsive baseline;
custom response profiles are left intact.

## User-Facing Telemetry

The useful control display is limited to selected target/class, measured aim,
crosshair, current control error, dominant FAR/NEAR region, prediction velocity,
prediction offset, projected full counts, Atan demand, integer command,
delivery state and block reason.

## Calibration Boundary

FOV, `counts_per_360`, FAR/NEAR Kp, Atan scale, per-update limits and recoil
counts need Jetson + kmNet + game trace calibration. These values must be
evaluated independently from capture/inference latency and transport capacity.
