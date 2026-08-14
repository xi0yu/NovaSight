# NovaSight Current Mouse Control Path Audit

Date: 2026-08-04

Status: live-code audit for the current continuous nonlinear Atan control path.

## Active Mainline

```text
DetectionBatch latest-only gate
-> FrameContext
-> basic candidate filter
-> RuntimeTracker (identity/association)
-> RuntimeTargetSelector
-> RawAimPointProjector using current bbox
-> continuous nonlinear Atan control
-> capacity-one latest-replace CommandScheduler
-> MouseCommandExecutor
-> KmNetExecutor.move(dx, dy)
```

The dedicated 2D estimator consumes the measured aim point. It retains four
positions from one target, derives three adjacent `px/ms` velocity vectors, and
uses their medoid to reject a single-segment outlier. Tracker Kalman state is
not used as the final predicted aim, so there is no double prediction.

The active algorithm does not use:

```text
RuntimeService._mouse_observation_metadata prediction
MouseController deadzone/arrival/slew envelope
CommandScheduler trajectory split path
```

## Per-Observation Ownership

`RuntimeService._control_intent_from_context()` determines trigger readiness before quantization, creates a typed algorithm observation, and calls the new algorithm exactly once. The resulting integers are wrapped in one `ControlIntent`. `ExecutorRegistry` forces:

```text
scheduler != None
direct_output = false
latest_replace = true
```

The observation call replaces the single pending complete integer command and never calls hardware. `RuntimeService.process_control_tick()` takes at most that one command and sends it through `MouseCommandExecutor`. A newer observation or clear event increments the delivery epoch, so a command already removed from the slot but still waiting for the device lock is discarded before the device call. Scheduler step limits equal the controller's per-update limit, so this path never creates a multi-step trajectory or count debt.

## Source And Time Contract

- `capture_ts_ns`, `inference_end_ts_ns`, and `control_now_ns` must share the host monotonic domain.
- Capture age below zero, inference completion outside `[capture, control_now]`, stale age, or generation/frame rollback blocks the whole decision. A non-increasing capture timestamp resets prediction history and uses pure measured-position feedback for that otherwise valid observation.
- Global observation cursors survive target switches; target-local estimator/mode/quantizer state does not.
- Motion-estimator `dt_ms` is the adjacent same-target capture timestamp difference divided by `1_000_000`.
- When `prediction.enabled` is true, prediction estimates aim-point velocity
  from real adjacent capture timestamps. The horizon is
  `frame_age_ms + actuation_delay_ms + prediction_lead_ms`; velocity is
  multiplied by that time horizon, then strength-gated from motion confidence
  before `prediction_cap_px` vector limiting. Acceleration remains telemetry only
  and does not add a second correction path. Disabling prediction zeros every
  prediction offset and skips velocity-history updates.

## Coordinate Contract

Detection boxes and the dedicated estimator use current ROI coordinates. Trusted source geometry creates `CoordinateTransform`; the full-control center is mapped back into ROI coordinates for `e_meas`. The controller then maps ROI error through ROI/source scaling, FOV, radians, and calibrated counts.

If source geometry is unavailable or untrusted, the aim observation is invalid and no command is sent. A predicted-only or stale Track is also invalid as a control source.

## State Ownership

The control algorithm alone owns:

- continuous response gain from projected counts-domain error magnitude;
- four-position same-target history;
- three-segment medoid velocity for single-outlier rejection;
- spread/trend/detection/track-identity prediction confidence;
- reference dt, explicit `prediction_lead_ms`, and `prediction_cap_px`;
- per-axis sub-count quantizer residual.

The runtime owns target selection, initial trigger readiness, algorithm calculation, reset edges, the capacity-one delivery slot, and telemetry publication. Immediately before the serialized device call, the registry verifies that no newer submission superseded the selected command, then `MouseCommandExecutor` rechecks the trigger snapshot, command deadline, and increasing generation. The delivery slot retains at most one complete command and no trajectory.

## Configuration Contract

```text
pipeline.p_response_scale
pipeline.p_response_boost
pipeline.p_response_curve_shape
pipeline.max_output_x_counts
pipeline.max_output_y_counts
pipeline.prediction_lead_ms
pipeline.prediction_cap_px
```

The Atan scale `S` is fixed internally at 256 counts and is not part of the
user-facing configuration contract.

Runtime config and Studio edits use the same `pipeline.*` fields that are
composed into `AimAlgorithmConfig`.

## Direct Evidence

Implementation owners:

- `crates/novasight-core/src/controller/algorithm.rs`
- `crates/novasight-core/src/controller/control_law.rs`
- `crates/novasight-core/src/prediction/mod.rs`
- `crates/novasight-pipeline/src/runtime.rs`
- `crates/novasight-runtime/src/supervisor.rs`
- `crates/novasight-api/src/dto/runtime_status.rs`

`control_law.rs` is the only owner of the numeric formula from measured pixel
error and predicted displacement through projection, radial response scheduling,
and Atan demand. `algorithm.rs` owns bounded temporal state, calls that law
through `AimControlLaw::evaluate`, and applies only the configured X/Y output
ceilings before integer conversion. There is no arrival or visual-feedback stop
policy. Pipeline code adapts selected targets into
`AimSample`, applies live configuration, records `AimResult`, and delivers the
resulting device command.

Focused integration tests prove that consecutive DetectionBatch results replace the pending command, one control tick sends only the newest frame, and a newer observation also supersedes an older command that has left the slot but has not acquired the device lock. Prediction-disabled feedback remains the no-prediction baseline; `prediction_lead_ms=0` still compensates measured frame age and actuation delay, but adds no extra user lead.

## Remaining Blind Spots

The least-certain production value is still physical actuation delay. Successful device-send timestamps do not prove when the game consumes input or when the result appears in capture.

The largest control-model limitation is self-motion contamination: screen velocity combines target movement, manual view movement, and NovaSight's own prior output. The current prediction layer keeps prediction small and immediately removable; exact self-motion subtraction remains deferred until counts-to-visual timing is measured.

Capture resource/caps success also does not prove nonblack visual content. That requires a separate GPU content probe and is outside mouse-control ownership.
