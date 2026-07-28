# Fixed Y Recoil Plan

Date: 2026-07-16

Status: implemented.

## Contract

After the real left button has remained down for a configurable delay, every
fresh valid control observation adds one configurable fixed reverse-Y amount to
the existing visual Y feedback.

```text
if recoil enabled
and real left button is fresh and down
and left_hold_ms >= start_delay_ms
and current target is a fresh observed target:
    recoil_y = quantize(configured fixed counts per observation)
else:
    recoil_y = 0

final_y = clamp(quantized_visual_feedback_y + signed(recoil_y), active_mode_y_limit)
```

There is no recoil prediction, integral, curve, ramp, learning, output-tick
generator, or historical repayment.

## Why Per Observation

The V2 controller creates at most one current command per new DetectionBatch.
Binding fixed counts to that same observation preserves latest-replace semantics
and avoids sending the configured count at the independent 4ms tick rate.

The tradeoff is explicit:

```text
approximate recoil counts per second
    = fixed counts per observation * control observation FPS
```

This is simpler but not frame-rate invariant. Studio must display the current
control observation FPS and estimated counts/s next to the setting so the user
can see the consequence rather than treating the number as a universal rate.

## Configuration

Replace the current rate/ramp fields with:

```yaml
control:
  shared:
    recoil_enabled: false
    recoil_start_delay_ms: 0
    recoil_y_counts_per_observation: 0.0
```

`recoil_y_counts_per_observation` is non-negative and may be fractional. A
dedicated recoil quantizer accumulates only its sub-count residual, so values
such as 0.5 produce one device count every two accepted observations. Keeping
that residual separate from visual feedback lets release, target loss, stale
input, and send failure clear recoil immediately without deleting or repaying
visual error. The two integer contributions are added once and then clamped.

The reverse-Y device sign is derived once from the active effective Y direction.
Studio displays the resulting `+Y` or `-Y` direction; the amount field itself
remains non-negative.

## Reset Rules

Immediately output no recoil and clear its fractional residual on:

- real left-button release or stale/unavailable button state;
- target loss, target switch, predicted-only Track, or track rebuild;
- stale/non-monotonic DetectionBatch;
- runtime/capture/inference stop;
- algorithm, geometry, calibration, or recoil configuration change;
- kmNet disconnect or send block.

No release smoothing and no counts debt are allowed.

## Composition And Telemetry

Keep the decomposition visible:

```text
feedback_demand_y
recoil_y_counts_float
combined_demand_y
integer_command_y
driver_y_counts
recoil_left_hold_ms
recoil_active
recoil_block_reason
```

Studio derives `estimated_recoil_counts_s` from the live control-observation
FPS for presentation; it is not persisted as another controller state field.

This does not solve automatic weapon adaptation. It gives the user one stable,
understandable delay and one understandable strength parameter while preserving
enough evidence to diagnose over-pressure.

## Focused Tests

1. no output before `start_delay_ms`, fixed output at and after the boundary;
2. requires fresh real left button and a fresh observed target;
3. fractional setting quantizes deterministically and independently from visual feedback;
4. release/target switch/stale/config change clears residual immediately;
5. fixed recoil and visual feedback are separately reported before final clamp;
6. latest-replace still sends at most one command for one observation.
