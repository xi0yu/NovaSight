# Dual-Phase Atan Control

Formal algorithm ID: `dual_phase_atan_robust_predictive_v2`

Current production name: 双阶段 Atan 控制

The serialized ID is retained for configuration compatibility. Prediction is
optional and disabled by default. When enabled, the controller estimates X/Y
screen velocity for the one selected TrackId and advances that aim point by the
measured frame age plus the configured number of capture intervals.
The velocity estimate starts from the arithmetic mean of the latest three
adjacent speed segments built from four same-target positions; the median is
kept only as diagnostic telemetry.

## Production Data Path

```text
latest valid DetectionBatch
-> selected target and measured aim point
-> current measured error
-> optional bounded X/Y prediction for that same target
-> FAR/NEAR transition weight from the predicted control-point radial error
-> ROI/source projection and geometric atan
-> calibrated full correction counts
-> continuously blended counts-domain Atan response
-> device-count limiter and truncating fractional quantizer
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

w_far = smoothstep(control_distance, 0.75 * threshold, 1.25 * threshold)
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
Prediction caps start from the measured error and can expand only toward the
configured absolute cap when the motion state is stable enough. A predicted
offset cannot recursively enlarge its own safety envelope beyond that budget.

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
interval is due, it adds one integer `+Y` contribution to the newest safe
observation's tracking demand, including a zero tracking demand. It never
creates a second move, accumulates missed intervals, or owns a fractional
residual.

## Configuration Contract

- Prediction is disabled by default and legacy migrations return it to the
  disabled state instead of silently changing physical output.
- Enabling prediction activates both X and Y for the one selected TrackId.
- `prediction_lead_frames=0` still compensates measured frame age; it adds no
  extra capture interval. The module switch is the only way to disable
  prediction.
- Positive `prediction_lead_frames` is an input to the internal adaptive
  horizon. The first complete three-segment window preserves the configured
  lead, stable continuous motion can receive a bounded half-frame bonus, and
  stop/reverse/peek states reduce extra lead before velocity is projected.
- Acceleration is only a weak correction on stable continuous motion. It is
  bounded to a small fraction of the velocity projection and is not applied to
  stationary, stop/reverse, peek, or first-window mean states.
- Prediction confidence is used as a motion-state gate. Stable continuous and
  stable mean windows can keep full projection strength, while low-confidence
  and peek patterns remain attenuated before the cap.
- Runtime telemetry separates measured aim, predicted aim, prediction horizon,
  tracking command, recoil contribution and final device receipt.

The tuning surface is therefore limited to target/aim selection, projection
calibration, the FAR/NEAR threshold, Kp, Atan scale, per-update limits,
quantization/delivery safety and optional recoil.

## Physical Calibration Boundary

`counts_per_360`, FOV, FAR/NEAR Kp, Atan scale, per-update limits and recoil
counts require Jetson + kmNet + game trace calibration. Transport capacity must
not silently change controller authority, and prediction must not be used to
mask a base-control problem.
