# Dual-Phase Atan Robust Predictive Control V2

Formal algorithm ID: `dual_phase_atan_robust_predictive_v2`

Formal name: 双阶段 Atan 非线性控制 + 同目标短窗稳健速度估计 + 小幅受限预测

## Frozen Data Path

```text
latest valid DetectionBatch
-> measured aim point
-> measured error and single-threshold FAR/NEAR selection
-> same-target four-position history
-> three adjacent capture-time velocities in px/ms
-> median velocity
-> time-adaptive EMA
-> motion confidence
-> bounded X prediction around the measured position
-> ROI/source projection and geometric atan
-> calibrated full correction counts
-> counts-domain control atan and per-update clamp
-> truncating fractional quantizer
-> capacity-one latest-replace slot
-> MouseCommandExecutor validation and device call on output tick
```

There is no algorithm-layer trajectory plan. The delivery scheduler retains one complete unsent command, and a new inference result replaces the older command without merging or repaying its counts. A command waiting for the device lock is rechecked against the latest submission epoch; a driver call already in progress cannot be cancelled.

## State Ownership

Velocity history contains exactly four `(aim_x, capture_ts_ns)` samples from one `target_id`. Target switch, target loss, Tracker create/restore notification, timestamp discontinuity, geometry/ROI change, runtime restart, calibration change, or a history gap greater than `history_reset_gap_ms` clears the history, filtered velocity, confidence, and prediction availability.

A capture timestamp regression is not admitted into the new history. If its clock domain and frame freshness are otherwise valid, that observation still runs pure measured-position feedback with prediction disabled, and establishes a new timestamp epoch for following frames. It cannot lock feedback until the old timestamp is exceeded.

Global generation/frame/capture cursors are not transferred into target-local state. A new target cannot inherit the old target's speed or fractional count.

## Robust Velocity

```text
v1 = (P2 - P1) / dt12_ms
v2 = (P3 - P2) / dt23_ms
v3 = (P4 - P3) / dt34_ms
v_median = median(v1, v2, v3)

dt_ref_ms = mean(dt12_ms, dt23_ms, dt34_ms)
alpha = 1 - exp(-latest_dt_ms / (dt_ref_ms * smoothing_frames))
v_filtered = (1 - alpha) * previous_filtered + alpha * v_median
```

The first three positions produce no prediction. The first complete window is confidence-warmed; a second confirmed window reaches full history quality. There is no stationary-target forced-zero branch. Repeated zero segment medians drive the EMA naturally toward zero.

One position spike normally creates two opposite extreme velocities. The median preserves the remaining normal segment, while median absolute spread reduces prediction confidence.

## Motion Confidence

```text
spread = median(abs(vi - v_median))
q_spread = 1 / (1 + spread / (spread_base + spread_relative * abs(v_median)))

trend_delta = abs(v_median - previous_filtered)
q_trend = 1 / (1 + trend_delta / (change_base + change_relative * abs(previous_filtered)))

q_motion = clamp(q_history * q_spread * q_trend
                 * q_detection * q_track_identity, 0, 1)
```

Fast acceleration, stop, reversal, or segment disagreement reduces prediction confidence but does not invalidate the current measured position or disable feedback control.

## Prediction

Prediction configuration uses dimensionless frame units. Capture timestamps still provide the physical dt needed for correct velocity under variable FPS and latest-only drops.

```text
dt_ref_ms = mean(dt12_ms, dt23_ms, dt34_ms)
raw = v_filtered_px_ms * dt_ref_ms * lead_frames
weighted = raw * q_motion
allowed = min(mode.absolute_cap_px,
              mode.base_cap_px + mode.relative_cap * abs(e_meas.x))
safe = clamp(weighted, -allowed, allowed)

e_ctrl.x = e_meas.x + safe
e_ctrl.y = e_meas.y
```

`lead_frames=0` provides the pure-feedback/shadow baseline. Increasing it cannot bypass confidence, absolute, relative, freshness, or X-only constraints. Frame age remains a freshness/rejection signal and is not a second hidden prediction multiplier.

## Y Feedback And Recoil

V2 does not extrapolate target Y velocity. When `control.shared.recoil_enabled` is true, RuntimeService reads the real left-button state even in target-driven trigger mode. After `recoil_start_delay_ms`, each fresh accepted target observation contributes the configured `recoil_y_counts_per_observation` in the effective reverse-Y device direction. The contribution has an independent fractional quantizer, so trigger release, target loss/switch, stale input, configuration change, or runtime reset cannot leave recoil debt inside visual Y feedback. There is no time-rate integration, ramp, curve, or output-tick recoil generator.

## Projection And Control Atan

The two Atan operations are different:

```text
source_error = observation_error * roi_size / observation_size
focal_x = (source_width / 2) / tan(FOV_x / 2)
theta = atan(source_error / focal_x)                 # geometric projection
full_counts = theta * counts_per_360 / (2*pi)

u = K_mode * S_counts * atan(full_counts / S_counts) # control shaping
u = clamp(u, -max_counts_per_update, max_counts_per_update)
```

FAR and NEAR are the only modes. A single measured radial-error threshold selects NEAR; every value above it is FAR. There is no enter/exit hysteresis and neither mode is a movement deadzone. Both phases share one `scale_counts`; only Kp and the per-update output limit differ. No D term or separate velocity feed-forward is present.

## Integer And Delivery Contract

```text
accumulator += u
integer = trunc(accumulator)
accumulator -= integer
```

Opposite demand clears an old-direction fraction. Trigger release, stale blocking, target switch/loss, geometry change, or runtime reset clears unsent fractions. Trigger-inactive observations never bank output counts.

Every accepted observation yields at most one complete integer command for the capacity-one latest-replace slot. An independent output tick takes the newest command; scheduler step limits equal V2's per-update maximum, so it is never split into a trajectory. V2 accepts the kmNet move range rather than applying the unrelated signed 8-bit HID limit; defaults are FAR 600 and NEAR 120 counts. `MouseCommandExecutor` still rejects non-integer counts and values outside the signed 16-bit device range before the driver call.

The frame-normalized profile uses FAR Kp `0.90`, NEAR Kp `0.30`, shared Atan scale `1024`, one configured lead frame, and a three-frame velocity smoothing window. FAR therefore has enough authority to reach the empirically verified kmNet range, while NEAR remains separately bounded.

The deterministic closed-loop test compares `lead_frames=0` against enabled limited prediction for a constant-velocity target and requires the predictive run to have lower post-warmup mean absolute error. This is a regression baseline, not a substitute for real-device A/B calibration.

## Main Remaining Calibration

`lead_frames`, fixed recoil counts per observation, `counts_per_360`, FOV, Kp, Atan scale, prediction caps, and confidence scales require real Jetson/device/game traces. The current estimator intentionally measures apparent target-to-crosshair screen motion and does not yet subtract manual or NovaSight-induced camera motion.
