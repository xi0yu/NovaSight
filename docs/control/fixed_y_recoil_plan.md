# Interval-Gated +Y Recoil

Date: 2026-07-28

Status: implemented in the Rust backend.

## Contract

Recoil is composed on the newest safe visual observation. It does not own an
independent timer or create a second device command:

```text
current tracking move (possibly zero)
        +
if real left button is down
and the optional target guard passes
and elapsed since the last successful recoil move >= interval_ms:
    + y_counts
        ↓
one combined kmNet move
```

The first eligible visual observation after a new press starts the cadence and does not
receive recoil until one complete interval has elapsed. A late command receives
one `+Y` contribution only; missed intervals are not accumulated or repaid.

The cadence advances only after the combined command is accepted by the pointer
adapter. A rejected send therefore leaves the contribution due for the next
eligible command.

## Configuration

```yaml
control:
  recoil:
    enabled: false
    require_target: true
    interval_ms: 16
    y_counts: 1
```

- `interval_ms`: minimum time between successful moves that contain recoil.
- `y_counts`: positive Y value added to the current command when due.
- `require_target`: requires a current valid target or the same locked identity
  to remain inside the existing Tracker loss-grace window. The loss grace does
  not authorize predicted-only tracking. Turning the option off allows a due
  recoil-only move when tracking demand is zero or no target is present; it
  still produces at most one move for that observation.

Schema 7 rate/ramp recoil settings cannot be converted without changing the
physical output. Migration to schema 8 therefore removes those retired fields
and closes `enabled`; the user must confirm the new interval and +Y values
before enabling recoil again.

## Safety And Performance

- Left-button release, reconnect state changes and live recoil reconfiguration
  reset the cadence.
- The final Y value is clamped to the signed 16-bit kmNet range.
- The targeting hot path no longer calculates recoil geometry or writes a
  recoil observation mutex.
- The device worker is event-driven and has no idle/recoil polling wake-up.
- Recoil-disabled and left-button-up zero-demand observations do not wake the
  device worker.
- Every accepted observation produces at most one combined device send.
