# Continuous Atan Medoid Control

Formal algorithm ID: `continuous_atan_medoid_v2`

Current production name: 连续非线性 Atan 控制

The serialized ID is historical. Current production behavior uses one continuous
response model. Prediction is enabled by the production baseline. When enabled,
the controller estimates 2D aim-point velocity for the one selected TrackId and
advances that aim point by the measured frame age, configured prediction
actuation delay, and configured extra lead time. The velocity estimate is the
2D medoid of exactly three adjacent velocity segments from the latest four
same-target aim positions. There is no mean/latest fallback, confidence weight,
acceleration branch, or motion-profile classifier in the production predictor.

## Production Data Path

```text
latest valid DetectionBatch
-> selected target and measured aim point
-> current measured error
-> bounded 2D aim-point prediction for that same target
-> ROI/source projection and geometric atan
-> calibrated full correction counts
-> continuous counts-domain Atan response
-> truncating fractional device-count quantizer
-> recoil composition
-> fixed X/Y device-boundary clamp
-> capacity-one latest-replace slot
-> MouseCommandExecutor validation
-> kmNet move(dx, dy)
```

Every command is recalculated from the newest measured aim point. Prediction,
when enabled, only changes that one control reference; there is no D term or
algorithm-layer trajectory. A new inference result replaces an older unsent
command without merging or repaying its counts.

## Projection And Atan Response

The two Atan operations serve different purposes:

```text
source_error = observation_error * roi_size / observation_size
focal_x = (source_width / 2) / tan(FOV_x / 2)
theta = atan(source_error / focal_x)
full_counts = theta * counts_per_360 / (2*pi)

rho = hypot(full_counts_x, full_counts_y)
S_counts = 256
r = rho / S_counts
curve = 1 - exp(-(r ^ response_curve_shape))
R = 1 + response_boost * curve
K = response_scale * R
u = K * S_counts * atan(full_counts / S_counts)
```

The first Atan converts image displacement into view angle. The second is the
nonlinear response curve that compresses large device corrections. It is not a
derivative controller: no historical difference participates in `u`.

`response_scale` is the base response strength. `response_boost` controls the
bounded amount of extra strength available as normalized error grows.
`response_curve_shape` controls how early or late that extra strength appears.
Prediction changes the predicted error only; it does not modify this gain.
`S_counts` is the fixed internal Atan scale and is not user configurable.
`max_output_x_counts` and `max_output_y_counts` are fixed device-boundary limits
applied after recoil; they are not part of prediction.
`prediction_cap_px` is a vector cap on future aim-point displacement; it is not a
mouse-count limiter.

## State And Integer Output

The active target identity and sub-count limiter residual are target-local.
Target switch/loss, Tracker rebuild, stale input, timestamp discontinuity and
runtime restart clear target-relative state. A live configuration update only
updates the domain that changed; an output-limit change does not rebuild the
controller or clear its sub-count residual.

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
interval is due, it adds one integer `+Y` contribution to the newest safe
observation's tracking demand, including a zero tracking demand. It never
creates a second move, accumulates missed intervals, or owns a fractional
residual.

## Configuration Contract

- Prediction is enabled by the production baseline and can be disabled explicitly
  for a no-prediction control baseline.
- Enabling prediction activates both X and Y for the one selected TrackId.
- `prediction_lead_ms=0` still compensates measured frame age and actuation
  delay; it only removes the extra user lead. The module switch is the only way
  to disable prediction.
- Positive `prediction_lead_ms` is a direct time horizon in milliseconds. It no
  longer depends on frame interval, so unstable FPS and PGIE interval changes do
  not silently change the configured extra lead.
- Prediction output is the medoid aim-point velocity multiplied by the measured
  time horizon, then vector-capped. Detection confidence only admits or rejects
  a target upstream; it never rescales an admitted prediction.
- Runtime telemetry separates measured aim, predicted aim, prediction horizon,
  tracking command, recoil contribution, final fixed clamp and device receipt.

The tuning surface is therefore limited to target/aim selection, projection
calibration, response strength, response curve, prediction time/cap, fixed X/Y
output limits and optional recoil. The Atan scale and quantizer residual are
internal invariants, not user parameters.

## Physical Calibration Boundary

`counts_per_360`, FOV, response strength, response curve, prediction time,
fixed X/Y limits and recoil counts require Jetson + kmNet + game trace
calibration. Transport capacity must not silently change controller authority,
and prediction must not be used to mask a base-control problem.
