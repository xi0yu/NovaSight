# NovaSight Main Mouse Control

Date: 2026-07-26

Status: `dual_phase_atan_robust_predictive_v2` remains the serialized mainline
ID for compatibility. Its production behavior is measured-error-only
dual-phase Atan control.

## Mainline Route

```text
latest DetectionBatch
-> freshness and monotonic timestamp checks
-> TargetSelector / Tracker identity
-> shared bbox aim point
-> frozen crosshair/geometry reference
-> current measured ROI error
-> FAR / NEAR transition weight
-> source/FOV/counts projection
-> continuously blended counts-domain Atan response
-> device-count limiter and truncating quantizer
-> capacity-one latest-replace slot
-> MouseCommandExecutor
-> at most one move(dx, dy) per output tick
```

The controller creates neither a future position nor a trajectory plan. A new
observation replaces an unsent older command. `Kp + Atan + per-update limit`
already forms the incremental closed-loop response; splitting it again would
create a stale open-loop tail.

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
e_ctrl = e_meas

source_error = e_ctrl * roi_size / observation_size
theta = atan(source_error / focal_length)
full_counts = theta * counts_per_360 / (2*pi)

u_near = near_kp * atan_scale * atan(full_counts / atan_scale)
u_far = far_kp * atan_scale * atan(full_counts / atan_scale)
w_far = smoothstep(distance, 0.75 * near_threshold, 1.25 * near_threshold)
u = (1 - w_far) * u_near + w_far * u_far
limit = (1 - w_far) * near_limit + w_far * far_limit
u = clamp(u, -limit, limit)
```

The configured radial-error threshold is the center of a continuous transition,
not a hard mode switch. Outside the 75%-125% transition band, the original
NEAR or FAR response is unchanged. No velocity estimate, position prediction,
D term or velocity feed-forward enters `e_ctrl` or `u`.
The old `predictive_v2` identifier and zero-valued prediction telemetry remain
only for compatibility.

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
control:
  active_algorithm: dual_phase_atan_robust_predictive_v2
  algorithms:
    dual_phase_atan_robust_predictive_v2:
      schema_version: 9
      prediction:
        enabled: false
      mode:
        near_threshold_px: 12.0
      atan:
        scale_counts: 256.0
        far:
          kp: 0.45
          max_counts_per_update: 127.0
        near:
          kp: 0.22
          max_counts_per_update: 72.0
```

The Rust root schema is version 6 and uses `pipeline.prediction_enabled: false`.
Both runtimes migrate older configurations to prediction disabled, reject an
attempt to enable it, and omit prediction/velocity tuning from public schemas.

## User-Facing Telemetry

The useful control display is limited to selected target/class, measured aim,
crosshair, current control error, dominant FAR/NEAR region, projected full counts, Atan
demand, integer command, delivery state and block reason. Prediction is shown
only as disabled. Internal compatibility fields are not presented as live
motion measurements.

## Calibration Boundary

FOV, `counts_per_360`, FAR/NEAR Kp, Atan scale, per-update limits and recoil
counts need Jetson + kmNet + game trace calibration. These values must be
evaluated independently from capture/inference latency and transport capacity.
