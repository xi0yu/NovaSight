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
-> FAR / NEAR phase selection
-> source/FOV/counts projection
-> counts-domain Atan response
-> per-observation clamp and integer quantizer
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

u = kp_phase * atan_scale * atan(full_counts / atan_scale)
u = clamp(u, -phase_limit, phase_limit)
```

One measured radial-error threshold chooses FAR or NEAR. No velocity estimate,
position prediction, D term or velocity feed-forward enters `e_ctrl` or `u`.
The old `predictive_v2` identifier and zero-valued prediction telemetry remain
only for compatibility.

## Quantization And Latest-Replace Delivery

```text
accumulator += u
integer_count = trunc(accumulator)
accumulator -= integer_count
```

Direction changes clear opposite-direction residual. Trigger-inactive or
blocked observations clear the quantizer and cannot bank historical movement.

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

The Rust root schema is version 5 and uses `pipeline.prediction_enabled: false`.
Both runtimes migrate older configurations to prediction disabled, reject an
attempt to enable it, and omit prediction/velocity tuning from public schemas.

## User-Facing Telemetry

The useful control display is limited to selected target/class, measured aim,
crosshair, current control error, FAR/NEAR phase, projected full counts, Atan
demand, integer command, delivery state and block reason. Prediction is shown
only as disabled. Internal compatibility fields are not presented as live
motion measurements.

## Calibration Boundary

FOV, `counts_per_360`, FAR/NEAR Kp, Atan scale, per-update limits and recoil
counts need Jetson + kmNet + game trace calibration. These values must be
evaluated independently from capture/inference latency and transport capacity.
