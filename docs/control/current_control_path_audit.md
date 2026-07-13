# NovaSight Current Mouse Control Path Audit

Date: 2026-07-13

Status: live-code audit after introducing `dual_phase_atan_predictive_v1` and removing trajectory scheduling from that algorithm.

## Active Mainline

```text
DetectionBatch latest-only gate
-> FrameContext
-> basic candidate filter
-> RuntimeTracker (identity/association)
-> RuntimeTargetSelector
-> RawAimPointProjector using current bbox
-> dual_phase_atan_predictive_v1
-> MouseCommandExecutor
-> KmNetExecutor.move(dx, dy)
```

The dedicated X estimator consumes the raw aim point. Tracker Kalman state is used for target identity confidence, not as the new algorithm's predicted aim, so there is no double prediction.

The active algorithm bypasses:

```text
RuntimeService._mouse_observation_metadata legacy prediction
MouseController legacy deadzone/arrival/slew/rounding envelope
CommandScheduler submit/tick/split path
```

## Per-Observation Ownership

`RuntimeService._control_intent_from_context()` determines trigger readiness before quantization, creates a typed algorithm observation, and calls the new algorithm exactly once. The resulting integers are wrapped in one `ControlIntent`. `ExecutorRegistry` forces:

```text
scheduler = None
direct_output = true
single_command_per_observation = true
```

The same observation call invokes the selected hardware executor once. `RuntimeService.process_control_tick()` is a no-op for this algorithm. Therefore there is no pending plan for a later tick to consume.

## Source And Time Contract

- `capture_ts_ns`, `inference_end_ts_ns`, and `control_now_ns` must share the host monotonic domain.
- Capture age below zero, inference completion outside `[capture, control_now]`, stale age, generation/frame rollback, or non-increasing capture time blocks the whole decision.
- Global observation cursors survive target switches; target-local estimator/mode/quantizer state does not.
- Motion-estimator `dt` is the adjacent same-target capture timestamp difference.
- Prediction horizon is current frame age plus configured actuation delay, capped by `max_horizon_ms`.

## Coordinate Contract

Detection boxes and the dedicated estimator use current ROI coordinates. Trusted source geometry creates `CoordinateTransform`; the full-control center is mapped back into ROI coordinates for `e_real`. The controller then maps ROI error through ROI/source scaling, FOV, radians, and calibrated counts.

If source geometry is unavailable or untrusted, the aim observation is invalid and no command is sent. A predicted-only or stale Track is also invalid as a control source.

## State Ownership

The new algorithm alone owns:

- FAR/NEAR hysteresis;
- raw-X constant-velocity state;
- innovation gates and outlier streak;
- prediction cooldown and caps;
- real-error zero-cross history;
- per-axis sub-count quantizer residual.

The runtime owns target selection, initial trigger readiness, algorithm calculation, reset edges, and telemetry publication. `MouseCommandExecutor` rechecks the trigger snapshot, command deadline, and increasing generation under the same lock that serializes the device call. It retains no movement amount or trajectory.

## Compatibility Algorithms

The legacy implementations remain selectable under their own config namespaces:

```text
control.algorithms.calibrated_angular
control.algorithms.universal_saturated
control.algorithms.ttbox_pid_atan
```

They continue to use the existing `MouseController` envelope and the legacy `scheduler_enabled` choice. Their behavior is deliberately not mixed into `dual_phase_atan_predictive_v1`.

## Configuration Contract

```text
control.active_algorithm
control.algorithms.<algorithm_id>.*
```

The loader migrates the preceding `control.mode` and top-level algorithm blocks. Runtime Python aliases remain temporary compatibility accessors; serialized config and Studio edits use the isolated namespace.

## Direct Evidence

Implementation owners:

- `novasight/control/algorithms/dual_phase_atan_predictive_v1/core.py`
- `novasight/runtime/service.py::_dual_phase_control_command`
- `novasight/executors/runtime.py::ExecutorRegistry._execute_direct`
- `novasight/config/runtime.py::ControlAlgorithmConfigs`

Focused integration tests prove that a DetectionBatch produces one hardware call during observation processing, the executor reports `mouse_command_executor`, no Scheduler exists even if the legacy switch is true, and a subsequent control tick emits nothing.

## Remaining Blind Spots

The least-certain production value is still physical actuation delay. Successful device-send timestamps do not prove when the game consumes input or when the result appears in capture.

The largest control-model limitation is self-motion contamination: screen velocity combines target movement, manual view movement, and NovaSight's own prior output. V1 keeps prediction small and immediately removable; exact self-motion subtraction remains deferred until counts-to-visual timing is measured.

Capture resource/caps success also does not prove nonblack visual content. That requires a separate GPU content probe and is outside mouse-control ownership.
