# Dual-Phase Atan Predictive Control V1

Formal algorithm ID: `dual_phase_atan_predictive_v1`

Status: isolated compatibility/A-B baseline. The configured precise mainline is `dual_phase_atan_robust_predictive_v2`.

Formal name: 双阶段 Atan 非线性反馈控制 + 置信度加权受限预测 + 最新帧重计算

## Frozen Semantics

1. FAR and NEAR are the only phases. Real radial error selects the phase with hysteresis; neither phase is a deadzone.
2. `e_real` always remains available and owns correction, zero-cross, convergence, fallback, and trace semantics.
3. `e_ctrl.x = e_real.x + bounded_prediction`; `e_ctrl.y = e_real.y`. V1 predicts X only.
4. Prediction is a confidence-weighted, absolute-and-relative-capped offset on the current real position. It is never the primary controller.
5. The controller converts complete projection angle to full correction counts before applying counts-domain Atan shaping.
6. No additional velocity feed-forward or D term is present in V1.
7. The quantizer retains only a fractional count; the hardware command is always integer.
8. Every accepted inference result produces at most one immediate `move(dx, dy)`. No trajectory Scheduler participates.
9. The next inference result recalculates from the new picture. There is no old-frame remainder to replace, finish, or repay.

## Why Scheduler Splitting Is Forbidden

The output is already a current-cycle increment:

```text
full correction counts
-> phase Kp
-> Atan compression
-> per-update clamp
```

Splitting this increment over time would apply a second pacing layer after feedback gain, leaving unexecuted old-frame movement while the visual state changes. That produces double slowdown, reversal lag, near-target overshoot, duplicate prediction compensation, and an uncontrolled relationship between inference FPS and device-send cadence.

The allowed delivery abstraction is `MouseCommandExecutor`. Under one device lock it rechecks trigger/freshness/generation, rejects replay, calls the device once, and reports failure/timing. Its generation cursor is validation state, not a control trajectory.

## Control Equation

For phase `m`:

```text
source_error = roi_error * roi_size / observation_size
theta = atan(source_error / focal)
c_error = theta * counts_per_360 / (2*pi)

u = K_m * S_m * atan(c_error / S_m)
u = clamp(u, -max_counts_per_update_m, max_counts_per_update_m)
```

Estimator state transition uses the real capture timestamp delta. Controller Kp is the configured per-observation gain and is not silently changed by inference FPS.

## Prediction Equation

```text
h = clamp((control_now - capture_ts) + actuation_delay, 0, max_horizon)
raw = estimated_velocity_x * h
confidence = motion_confidence * track_confidence * direction_quality
weighted = phase_weight * confidence * raw
allowed = min(phase_abs_cap, phase_base_cap + phase_relative_cap * abs(e_real.x))
safe = clamp(weighted, -allowed, allowed)
```

NEAR low-confidence center crossing is reduced to zero control error. High-confidence crossing is limited to `near_cross_allow_px`. Every actual `e_real` sign crossing clears fractional residual, damps X velocity, and disables prediction for the configured cooldown frames.

## Output And Reset Contract

```text
accumulator += u
integer = trunc(accumulator)
accumulator -= integer
```

Opposite demand clears an old-direction fraction. Trigger inactive clears both accumulators. Target switch clears mode, estimator, cooldown, crossing history, and residuals without clearing the global observation cursor.

Blocked decisions always return zero integers. Timestamp-domain errors, stale observations, generation/frame/capture rollback, invalid geometry, target loss, stale/predicted-only tracks, runtime/config changes, and calibration changes cannot produce a device command.

## Initial Tuning Boundary

The tracked example config is the simulation starting point, not a production calibration. In particular:

- `actuation_delay_ms` must come from capture/send/visual-response traces;
- `counts_per_360` and FOV must match the active game/sensitivity profile;
- NEAR prediction caps must remain materially smaller than FAR caps;
- controller `max_counts_per_update` should normally fit one legal device packet.

## Acceptance Tests

Required traces cover stationary jitter at variable FPS, constant horizontal motion, a single 30-100 px detection jump, sudden reversal, target switch, abnormal timestamps, exact real-error zero-cross, trigger release, and one-command-per-observation runtime delivery.

Success means prediction lowers dynamic following error without materially increasing real zero-cross rate, outliers cannot create a large velocity/output, no Y prediction occurs, and no device call can originate from an older pending trajectory.
