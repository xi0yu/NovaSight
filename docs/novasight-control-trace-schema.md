# NovaSight Control Trace Schema

Status: optional offline diagnostics for the current Rust controller.

The trace scorer is behind the `diagnostics` Cargo feature. It is not compiled
into the default production core and never participates in live control.

## AlgorithmTraceSample

Each sample mirrors one `AimResult`:

```text
generation: u64
capture_ts_ns: u64
control_now_ns: u64
target_id: u64 | null
target_valid: bool
observed_error_x_px: f64
observed_error_y_px: f64
emitted_counts_x: i32
emitted_counts_y: i32
emit_allowed: bool
frame_age_ms: f64
```

Capture and control timestamps use the host monotonic nanosecond domain.
`target_id` is null when the controller had no current valid target.
`emitted_counts_*` are the controller's integer tracking demand before recoil
composition and the final fixed device clamp.

## Live delivery evidence

Offline trace samples do not claim that a device accepted a command. Live
delivery evidence belongs to runtime status:

```text
vision.control.pipeline.output_delivery_state
executor.accepted_command_count
executor.last_accepted_dx
executor.last_accepted_dy
```

`output_delivery_state` is one of:

```text
idle
gate_closed
device_disabled
trigger_inactive
generation_fenced
no_movement
superseded
sent
send_failed
```

There are no `scheduler_used`, trajectory-splitting or pending-count-debt
fields in the current Rust mainline. New observations replace older unsent
`OutputPlan` values in a capacity-one latest slot.

## Fixture

The repository fixture is:

```text
crates/novasight-core/tests/fixtures/continuous-control-control.jsonl
```

It validates serialization and deterministic offline scoring only. It is not
Jetson, TensorRT or kmNet hardware proof.
