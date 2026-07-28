# Dual-Phase Atan Control

Formal algorithm ID: `dual_phase_atan_robust_predictive_v2`

Current production name: 双阶段 Atan 控制

The serialized ID is retained for configuration compatibility. Despite the
legacy name, the production controller does not estimate target velocity or
predict a future position. Old prediction fields remain readable only for safe
configuration migration and do not participate in mouse output.

## Production Data Path

```text
latest valid DetectionBatch
-> selected target and measured aim point
-> current measured error
-> FAR/NEAR transition weight from measured radial error
-> ROI/source projection and geometric atan
-> calibrated full correction counts
-> continuously blended counts-domain Atan response
-> device-count limiter and truncating fractional quantizer
-> capacity-one latest-replace slot
-> MouseCommandExecutor validation
-> kmNet move(dx, dy)
```

Every command is recalculated from the newest measured aim point. There is no
position extrapolation, D term, velocity feed-forward or algorithm-layer
trajectory. A new inference result replaces an older unsent command without
merging or repaying its counts.

## Projection And Atan Response

The two Atan operations serve different purposes:

```text
source_error = observation_error * roi_size / observation_size
focal_x = (source_width / 2) / tan(FOV_x / 2)
theta = atan(source_error / focal_x)
full_counts = theta * counts_per_360 / (2*pi)

w_far = smoothstep(distance, 0.75 * threshold, 1.25 * threshold)
K = (1 - w_far) * K_near + w_far * K_far
u = K * S_counts * atan(full_counts / S_counts)
limit = (1 - w_far) * near_limit + w_far * far_limit
u = clamp(u, -limit, limit)
```

The first Atan converts image displacement into view angle. The second is the
nonlinear response curve that compresses large device corrections. It is not a
derivative controller: no historical difference participates in `u`.

FAR and NEAR are two parameterizations of one response curve. The configured
radial-error threshold centers a cubic Smoothstep transition whose half-width
is 25% of that threshold. This removes the parameter jump without adding a
second gain stage or another tuning field. Both regions share one Atan scale;
only Kp and the per-update output limit differ. Neither is a movement deadzone.

## State And Integer Output

The active target identity and sub-count limiter residual are target-local.
Target switch/loss, Tracker rebuild, stale input, geometry or calibration
change and runtime restart clear state. No velocity history is built while
prediction is disabled.

```text
accumulator += u
integer = trunc(accumulator)
accumulator -= integer
```

Direction reversal clears an old-direction fraction. Trigger release and all
blocking conditions clear unsent fractions, so the controller cannot bank
movement for later output.

## Delivery And Safety

Every accepted observation yields at most one complete integer command for the
capacity-one latest-replace slot. The output tick takes only the newest command
and does not split it into a trajectory. `MouseCommandExecutor` rechecks the
trigger snapshot, freshness deadline, generation and signed 16-bit device range
before invoking the driver.

Shared recoil is independent from target prediction. When its configured time
interval is due, it adds one integer `+Y` contribution to the current command.
It never creates a second move, accumulates missed intervals, or owns a
fractional residual.

## Configuration Contract

- `prediction_enabled` / `prediction.enabled` must be `false` in production.
- Current configuration migrations force legacy prediction to `false`.
- Public configuration schemas do not expose velocity or prediction tuning.
- User-facing telemetry shows measured/control error and reports prediction as
  disabled; zero-valued legacy fields are wire-compatibility data, not active
  measurements.

The tuning surface is therefore limited to target/aim selection, projection
calibration, the FAR/NEAR threshold, Kp, Atan scale, per-update limits,
quantization/delivery safety and optional recoil.

## Physical Calibration Boundary

`counts_per_360`, FOV, FAR/NEAR Kp, Atan scale, per-update limits and recoil
counts require Jetson + kmNet + game trace calibration. Transport capacity must
not silently change controller authority, and prediction must not be used to
mask a base-control problem.
