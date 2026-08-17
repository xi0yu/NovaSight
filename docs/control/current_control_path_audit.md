# NovaSight Current Mouse Control Path Audit

Date: 2026-08-14

Status: current Rust mainline audit. This file replaces the retired Python
`RuntimeService / AngularPDController / CommandScheduler` description.

## Active mainline

```text
DetectionBatch
-> capacity-one latest slot
-> TargetingCore
-> AimAlgorithm
-> capacity-one OutputPlan slot
-> recoil composition
-> fixed X/Y device clamp
-> generation / gate / trigger recheck
-> PointerDevice::send
```

There is one control law and one physical-output seam. New observations replace
unsent older observations; the runtime does not split one command into a
trajectory or accumulate a queue of pending counts.

## Numeric ownership

`AimAlgorithm` owns:

- freshness and monotonic observation checks;
- same-target four-position history and three adjacent velocity vectors;
- vector-medoid velocity selection;
- `frame_age + prediction_actuation_delay + prediction_lead` horizon;
- prediction vector clamp;
- FOV/counts projection and continuous Atan response;
- truncating integer quantization with sub-count residual.

The pipeline device worker owns:

- current trigger, output-gate and generation rechecks;
- recoil +Y composition;
- one final fixed per-axis clamp;
- the actual `PointerDevice::send` call;
- typed output-delivery state.

The final device formula is:

```text
out_x = clamp(tracking_x, -max_output_x_counts, max_output_x_counts)
out_y = clamp(tracking_y + recoil_y, -max_output_y_counts, max_output_y_counts)
```

## Parameters that do not intervene

The current production path has no dynamic output limit, arrival radius,
residual-cap parameter, trajectory scheduler, FAR/NEAR branch, integral term or
confidence-based movement gain.

`target_track_max_age` is retired. Timestamp-free replay uses a fixed internal
five-frame fallback; production identity loss uses only
`target_track_max_lost_age_ms`.

Detection and track confidence remain targeting evidence. They are not inputs
to `AimSample` and do not multiply controller demand.

## Live configuration ownership

- Targeting changes replace `TargetingConfig` and fence commands from the old
  generation.
- Prediction/control changes replace `AimAlgorithmConfig` and take effect on
  the next observation.
- Fixed output limits are atomically read at the device seam and do not rebuild
  prediction state.
- Fire delay is sampled on each hardware-trigger rising edge and prevents
  `AimAlgorithm::step` until the continuous hold duration exceeds the threshold.
- Recoil configuration is versioned independently and applied without restarting
  capture or inference.
- Saved YAML and live runtime use the same canonical `pipeline.*` fields.

## UI contract

The main parameter page follows:

```text
触发方式 -> 开火延迟 -> 预测/算法 -> 压枪 -> 固定限幅 -> 输出
```

Ordinary target behavior exposes minimum confidence, switch delay and
millisecond loss grace. Association weights and Kalman tuning are expert-only.
Replay-only fallback values are not exposed.

## Remaining proof gap

Source inspection proves ownership and formulas, not physical timing. Jetson
validation must still measure real capture age, kmNet acceptance, visible
actuation delay and game-specific FOV/count calibration.
