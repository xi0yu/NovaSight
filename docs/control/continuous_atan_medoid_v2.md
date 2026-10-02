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

S_counts = 256
q = clamp((t - entry_start) / entry_ramp_ms, 0, 1)
a = 3*q*q - 2*q*q*q; entry_ramp_ms=0: a=1
u = a * response_scale * S_counts * atan(full_counts / S_counts)
```

The first Atan converts image displacement into view angle. The second is the
nonlinear response curve that compresses large device corrections. It is not a
derivative controller: no historical difference participates in `u`.

`response_scale` is Kp, the constant normal response gain.
`entry_ramp_ms` controls how long it takes to rise from zero to Kp; zero
disables the ramp. The internal factor a is computed, not configurable.
There is no distance-dependent gain enhancement. Prediction changes only the
predicted error, not Kp.
`S_counts` is the fixed internal Atan scale and is not user configurable.
`max_output_x_counts` and `max_output_y_counts` are fixed device-boundary limits
applied after quantization; they are not part of prediction.
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

Zero tracking demand does not produce a device move, even while a trigger is held.

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
  tracking command, final fixed clamp and device receipt.

The tuning surface is therefore limited to target/aim selection, projection
calibration, response strength, entry ramp duration, prediction time/cap, fixed X/Y
output limits. The Atan scale and quantizer residual are
internal invariants, not user parameters.

## Physical Calibration Boundary

`counts_per_360`, FOV, response strength, entry ramp duration, prediction time,
fixed X/Y limits require Jetson + kmNet + game trace
calibration. Transport capacity must not silently change controller authority,
and prediction must not be used to mask a base-control problem.
