# NovaSight Current Control Path Audit

Date: 2026-07-10

Scope: stage 0 audit for the mouse continuous closed-loop control rebuild. This document describes the current code path before changing control semantics.

## Summary

The current runtime already has a substantial control chain:

```text
CapturedFrame
-> LatestFrameBroker / latest_frame()
-> RuntimePipeline inference loop
-> DetectionBatch / FrameContext
-> RuntimeTargetSelector + RuntimeTracker + KalmanEstimator
-> AimPointGenerator + LatencyCompensator
-> ExperimentalAnglePidStrategy
-> AngularErrorMapper
-> AngularPDController
-> ControlOutputPolicy
-> CommandScheduler
-> KmNetExecutor
```

The current chain is not the requested new `predictive_pid_v2` mode. The main mismatch is scheduling semantics:

```text
Current:
new observation updates last_frame_context
continuous control tick repeatedly recomputes control from last_frame_context
each tick submits a new output to CommandScheduler

Requested mode A:
each new observation creates one bounded control budget/plan
Scheduler consumes that plan
next observation cancels remaining old plan and replaces it
```

## Existing Call Chain

### Capture to latest frame

- `CapturedFrame.capture_ts_ns` is a property over `CapturedFrame.ts_ns` in `novasight/capture/source.py:42-73`.
- `ImageFrameSource.read()` sets `ts_ns` from `time.monotonic_ns()` after synthetic frame pacing in `novasight/capture/source.py:172-194`.
- `OpenCvFrameSource.read()` sets `ts_ns` from userspace receive time after `cap.read()` in `novasight/capture/source.py:239-261`.
- `GstAppSinkFrameSource._pull_frame()` and `GstResourceFrameSource._pull_frame()` set `ts_ns` to `receive_ts_ns = time.monotonic_ns()`; GStreamer sample PTS is stored separately as `source_ts_ns/source_ts_kind` in `novasight/capture/source.py:342-385` and `novasight/capture/source.py:388-432`.
- `CaptureSession._publish_frame()` rejects capture timestamp rollback, increments a broker generation, replaces `_latest_frame`, and publishes a `FrameHandle` to `LatestFrameBroker` in `novasight/capture/session.py:194-264`.
- `LatestFrameBroker` is a single-slot latest-frame mailbox. It rejects stale generation/frame_id, releases overwritten handles, and clears the pending slot on acquire in `novasight/runtime/latest_frame.py:51-159`.

### Runtime inference to DetectionBatch

- `RuntimePipeline.start()` runs two threads: `novasight-inference-control` and `novasight-continuous-control` in `novasight/runtime/pipeline.py:64-90`.
- The inference thread prefers `LatestFrameBroker.acquire_latest(after_generation=...)`; fallback uses `wait_preview_frame(after_frame_id=...)` in `novasight/runtime/pipeline.py:188-248` and `novasight/runtime/pipeline.py:249-318`.
- Before inference, the pipeline rejects stale input frames using `FreshnessGate` in `novasight/runtime/pipeline.py:377-389`.
- `RuntimeService.process_captured_frame()` builds a `DetectionBatch` with `frame_id`, `generation`, `capture_ts_ns`, `inference_start_ts_ns`, `inference_end_ts_ns`, `publish_ts_ns`, age metrics, `clock_domain="monotonic"`, and `coordinate_space="roi"` in `novasight/runtime/service.py:1056-1325`.
- `RuntimeService.process_detection_batch()` is the external/producer-provided `DetectionBatch` path. It validates latest/freshness, ROI coordinate space, bbox contract, and stale age before converting to `FrameContext` in `novasight/runtime/service.py:419-596`.
- `DetectionBatchMailbox` exists as a latest-only single-slot batch mailbox in `novasight/runtime/detection_batch_mailbox.py:10-92`, but the current `RuntimePipeline` code path does not show a DetectionBatchMailbox consumer thread. It is a reusable seam, not currently the dominant runtime handoff.

### DetectionBatch to target state

- `DetectionBatch` includes the timing/latest fields needed for stage 1 auditing: `generation`, `publish_ts_ns`, `input_age_ms`, `result_age_ms`, `source_sequence`, `is_stale`, `clock_domain`, and `model_input_size` in `novasight/contracts.py:145-204`.
- `detection_batch_to_frame_context()` preserves `capture_ts_ns`, inference timestamps, generation, detections, tracks restored from metadata, and optional stage timestamps in `novasight/runtime/detection_batch.py:24-63`.
- `RuntimeTargetSelector.select()` filters candidates and routes through `RuntimeTracker` in `novasight/runtime/target_selector.py:57-240`.
- `RuntimeTracker` tracks `track_id`, class, score, bbox, state, hits/misses, `last_ts_ns`, Kalman estimator, and `estimate` in `novasight/runtime/tracker.py:53-70`.
- `RuntimeTracker.update()` uses `context.capture_ts_ns` as the frame timestamp, rejects repeat `frame_id` updates by returning unchanged tracker state, and updates/creates tracks in `novasight/runtime/tracker.py:110-214`.
- On matched updates, tracker computes finite-difference observed velocity from consecutive target centers over `now_ns - previous_ts_ns` and blends it into Kalman velocity with `apply_velocity_hint()` in `novasight/runtime/tracker.py:326-383`.
- `KalmanEstimator` stores position and velocity in `[x, y, vx, vy]`, uses timestamp-derived `dt`, and clamps prediction dt via `max_predict_dt_ms` in `novasight/runtime/kalman.py:43-255`.

### Target state to control output

- `RuntimeService.update_control_observation()` handles new observations. In `experimental_angle_pid`, it selects target, computes aim/latency metadata, and calls `control_strategy.observe()` only; it does not emit control for the new observation in that method in `novasight/runtime/service.py:955-1054`.
- Actual sending happens in `RuntimeService.process_control_tick()`, which reuses `self.last_frame_context` and calls `process_frame(context)` at `experimental_angle_control_hz` in `novasight/runtime/service.py:402-413`.
- `RuntimePipeline._control_loop()` calls `process_control_tick()` repeatedly while `control.strategy == "experimental_angle_pid"` in `novasight/runtime/pipeline.py:358-400`.
- `_control_intent_from_context()` gates on target selection, kmNet trigger, calibration fingerprint, and `control_allowed`, then calls `self.control_strategy.calculate()` in `novasight/runtime/service.py:1508-1820`.
- Trigger state is read from kmNet buttons unless `control.trigger_mode == "always"` in `novasight/runtime/service.py:1542-1550` and `novasight/runtime/service.py:2030-2054`.
- Target loss, disabled control, trigger inactive, calibration mismatch, stale DetectionBatch, and config calibration changes call `_clear_pending_commands()` and/or `_reset_runtime_control_state()` in `novasight/runtime/service.py:278-300`, `novasight/runtime/service.py:446-559`, `novasight/runtime/service.py:1512-1514`, and `novasight/runtime/service.py:1801-1826`.

### Error mapping, PID-like logic, counts, and output limiting

- The currently active strategy is `ExperimentalAnglePidStrategy`; `RuntimeService._create_control_strategy()` only accepts `experimental_angle_pid` and constructs that strategy in `novasight/runtime/service.py:2056-2118`.
- `ExperimentalAnglePidStrategy.calculate()` requires a `compensated_target` payload. Without it, it resets the angular controller and emits zero in `novasight/control/strategy.py:183-260`.
- `LatencyCompensator` calculates `measurement_age_ms = compute_ts_ns - capture_ts_ns`, applies velocity-based compensation using `compute_ts_ns + estimated_actuation_delay_ms - estimate.state_ts_ns`, and clamps by `max_compensation_ms` and `max_compensation_px` in `novasight/runtime/aim.py:195-383`.
- `AimPointGenerator` applies EMA to aim anchor points by track id in `novasight/runtime/aim.py:119-192`.
- `AngularErrorMapper` converts compensated control-space point to pixel error and radians using horizontal FOV, derived vertical FOV, and focal lengths in `novasight/control/angular.py:144-220`.
- `AngularPDController` is P+D, not full PID. It has no integral state; `ExperimentalAnglePidStrategy` accepts `ki/integral_limit` but deletes/ignores related legacy parameters when constructing `AngularPDConfig` in `novasight/control/strategy.py:44-181`.
- `AngularPDController.update()` calculates derivative from `error_rad` difference over `err.dt_s`, applies D EMA, applies zone gains, maps radians to counts via `counts_per_360_x/y`, applies axis sign, accumulates fractional residuals, clamps counts vector, and applies per-tick slew limit in `novasight/control/angular.py:266-409`.
- Current control `dt` for this strategy is not `measurement_dt_s`. `ExperimentalAnglePidStrategy` sets tracker debug `dt` to `1.0 / control_hz`, and `_debug_dt()` returns that or the same fallback, clamped to `[1/240, 1/15]` in `novasight/control/strategy.py:230-236` and `novasight/control/strategy.py:473-487`.

### Scheduler and device send

- `ExecutorRegistry.execute()` always applies `ControlOutputPolicy`, then submits to `CommandScheduler`; if scheduler emits a step immediately, it calls selected executor and records monotonic `device_send_start_ts_ns/end_ts_ns` in `novasight/executors/runtime.py:65-148`.
- `CommandScheduler` keeps latest pending command state. It cancels/replaces pending on trajectory generation, track change, new frame, direction change, expiry, rejection, and device error cooldown in `novasight/control/scheduler.py:22-184` and `novasight/control/scheduler.py:296-409`.
- `CommandScheduler._split_steps()` splits a command into integer steps whose sum equals the requested output, bounded by `max_step_x/y`, in `novasight/control/scheduler.py:415-489`.
- `KmNetExecutor.execute()` sends one relative move through kmNet `move`, `enc_move`, `move_auto`, or bezier APIs and records `driver_dx/driver_dy` in `novasight/executors/kmnet.py:195-296`.
- `HidOutputThread` can consume `scheduler.tick()` and send pending steps in `novasight/control/hid_output.py:68-115`, and `ExecutorRegistry.tick_pending()` can do the same in `novasight/executors/runtime.py:150-196`. However, current source search found no production instantiation/call of `HidOutputThread` or `tick_pending()`. In the active runtime path, pending tail steps are not independently consumed unless another integration path starts that consumer.

## Existing State Fields

### RuntimeService state

- `last_frame_context`
- `last_target`
- `last_control`
- `last_execution`
- `last_inference_status`
- `last_pipeline_timings`
- `_last_control_tick_ns`
- `target_selector`
- `aim_points`
- `latency_compensator`
- `_accepted_batch_generation`
- `_accepted_batch_capture_ts_ns`
- `stale_drop_count`
- `_runtime_calibration_signature`
- `_external_sensitivity_fingerprint/source/ts_ns`

### Tracker and estimator state

- `RuntimeTracker._tracks`
- `RuntimeTracker._last_frame_id`
- per-track `track_id`, `box`, `state`, `hits`, `misses`, `last_ts_ns`, `last_match_cost`, `last_mahalanobis`, `identity_confidence`, `switch_committed`
- per-track `KalmanEstimator` state `[x, y, vx, vy]`, covariance, `last_state_ts_ns`, `last_measurement_ts_ns`, `prediction_steps`

### Aim and control state

- `AimPointGenerator._last_by_track`
- `AimPointGenerator._last_raw_by_track`
- `AngularPDController.memory`: previous error, derivative EMA, fractional residual counts, initialized flag, active track id, calibration signature, control geometry signature, last emitted counts

### Scheduler/executor state

- `CommandScheduler._pending_steps`
- `_pending_parent`
- `_pending_created_s`
- `_pending_command_id`
- `_pending_expires_s`
- `_last_emit_s`
- `_cancelled_pending`
- `_expired_pending`
- `_throttled_since_emit`
- `_cooldown_until_s`
- `_last_cancel_reason`
- `_last_error`
- `KmNetExecutor.connected/monitoring/move_count/last_dx/last_dy/last_button_*`

## Existing Config Fields

### Calibration

- `calibration.fov_x_deg`
- `calibration.counts_per_360_x`
- `calibration.counts_per_360_y`
- `calibration.axis_sign_x`
- `calibration.axis_sign_y`
- `calibration.game_sensitivity_fingerprint`
- `calibration.projection_profile`

### Target continuity and tracking

- confidence/FOV/candidate quality fields under `control.*`
- target lock/switch fields
- tracker match/ambiguity/timeout fields
- Kalman noise/gating/prediction confidence fields

### Aim and prediction-like latency compensation

- `control.aim_horizontal_percent`
- `control.aim_ratio`
- `control.aim_offset_x_px`
- `control.aim_offset_y_px`
- `control.aim_ema_enabled`
- `control.aim_ema_alpha`
- `control.latency_compensation_enabled`
- `control.latency_compensation_scale`
- `control.latency_max_compensation_ms`
- `control.latency_reject_if_age_exceeds_ms`
- `control.latency_max_compensation_px`
- `control.latency_min_velocity_px_s`
- `control.latency_max_velocity_px_s`
- `control.latency_min_velocity_measurements`
- `control.latency_min_velocity_confidence`
- `control.latency_estimated_actuation_delay_ms`

### Active control and scheduler

- `control.strategy` currently only accepts `experimental_angle_pid`.
- `control.trigger_mode` accepts `hardware` or `always`.
- `control.output_mode` is kmNet-only.
- `control.command_interval_ms`
- `control.scheduler_command_ttl_ms`
- `control.scheduler_predicted_command_ttl_ms`
- `control.scheduler_cancel_on_new_frame`
- `control.scheduler_cancel_on_direction_change`
- `control.scheduler_cancel_on_track_change`
- `control.scheduler_max_step_x`
- `control.scheduler_max_step_y`
- `control.scheduler_queue_hard_limit`
- `control.scheduler_device_error_cooldown_ms`
- `control.experimental_angle_*` for Kp/Kd, D filter, zone gains, max angle, max counts, control Hz, and unused compatibility knobs.

`novasight/config/runtime.py:512-513` rejects any strategy other than `experimental_angle_pid`; there is no `legacy | predictive_pid_v2` algorithm switch yet.

## Existing Time Fields

### Present and mostly consistent

- `capture_ts_ns`: userspace monotonic receive time, stored as `CapturedFrame.ts_ns` and carried into `DetectionBatch` and `FrameContext`.
- `source_ts_ns/source_ts_kind`: optional source/GStreamer timestamp, not used as the main control clock.
- `inference_start_ts_ns`
- `inference_end_ts_ns`
- `publish_ts_ns`
- `input_age_ms`
- `result_age_ms`
- `frame_age_ms`: computed as `time.monotonic_ns() - capture_ts_ns` in several RuntimeService payloads.
- `control_now_ts_ns`: recorded in `last_control` and observation-only payloads.
- `device_send_start_ts_ns`
- `device_send_end_ts_ns`
- scheduler `created_ts_ns` and `expires_ts_ns`

### Present but semantically different from new plan

- Tracker/Kalman velocity dt uses target observation timestamps via `context.capture_ts_ns`.
- Angular D term dt uses control strategy `dt = 1 / control_hz`, not visual measurement dt.
- Latency compensation has `measurement_age_ms` and `compensation_ms`, but does not expose the requested `prediction_horizon_s = frame_age_s + actuation_delay_s` name.

### Missing as first-class fields for the new plan

- `measurement_dt_s` telemetry for consecutive same-target observations.
- `prediction_horizon_s` / `prediction_horizon_ms` telemetry.
- explicit `scheduler_dt_s` telemetry per emitted step.
- explicit separation of `control_dt` versus `measurement_dt`.
- explicit `configured_estimate` marker for actuation delay.

## Existing Send Path

```text
RuntimeService.process_control_tick()
-> RuntimeService.process_frame(last_frame_context)
-> _control_intent_from_context()
-> ExperimentalAnglePidStrategy.calculate()
-> ControlIntent
-> ExecutorRegistry.execute()
-> ControlOutputPolicy.apply()
-> CommandScheduler.submit()
-> KmNetExecutor.execute(step)
-> kmNet driver move/enc_move/move_auto/bezier
```

Important behavior:

- Runtime sends only if trigger and calibration gates allow.
- If not allowed, it clears scheduler pending.
- Scheduler can split a large command into steps.
- In the active execute path, a ready scheduler decision is sent immediately by `ExecutorRegistry.execute()`.
- There is no wired production consumer found for `CommandScheduler.tick()` pending-tail execution. Therefore split tail steps may be overwritten by the next control tick rather than consumed as a stable per-observation plan.

## Code To Preserve

- `CapturedFrame.capture_ts_ns` and the monotonic receive-time contract, while keeping `source_ts_ns` separate.
- `CaptureSession` timestamp rollback rejection and `LatestFrameBroker` single-slot latest-frame semantics.
- `DetectionBatch` timing/latest fields and validation.
- `RuntimeService.process_detection_batch()` freshness/ROI/stale guards.
- `DetectionBatchMailbox` as a reusable latest-only seam, though it may need integration.
- `RuntimeTargetSelector`, `RuntimeTracker`, and Kalman continuity machinery, unless replaced deliberately by the new `TargetContinuityGuard`.
- Coordinate transform payloads and ROI/control-space validation.
- `AngularErrorMapper` math for horizontal FOV -> focal length -> radian error.
- `AngularPDController` tests and logic for D EMA, residual counts, axis sign, counts mapping, angular/counts clamp, and slew limit, but likely split into separate `PIDv2Controller`, `CountMapper`, and `OutputLimiter` modules.
- `CommandScheduler` cancellation rules and integer step split tests, but its ownership model needs to become per-observation plan consumption.
- `ExecutorRegistry` device send timing telemetry.
- `KmNetExecutor` driver abstraction and button trigger read path.

## Code To Replace Or Refactor

- `RuntimePipeline._control_loop()` / `RuntimeService.process_control_tick()` repeated recomputation from `last_frame_context` conflicts with mandatory mode A.
- `ExperimentalAnglePidStrategy` should not stay the new algorithm boundary; it mixes prediction payload consumption, error projection, PD, count mapping, residuals, and output shaping.
- `LatencyCompensator` currently does prediction-like compensation from Kalman velocity and aim EMA. The new plan needs explicit `MotionEstimator`, `VelocityFilter`, and `PredictionModel`; first version should avoid reusing aim EMA as velocity smoothing.
- `AimPointGenerator` applies position EMA. The plan allows only speed EMA, D EMA, and optional weak output smoothing unless tracker/Kalman is already handling position. We need decide whether aim EMA remains enabled in `predictive_pid_v2`.
- `AngularPDController` uses `dt = 1/control_hz` for D, not `measurement_dt_s`; this must change for PIDv2.
- `ControlConfig.strategy` validation and UI schema need algorithm switch support: `legacy | predictive_pid_v2`, with a defined mapping from current `experimental_angle_pid`.
- `CommandScheduler` has a split queue but no active production `tick()` consumer in the current runtime path. It should be replaced or rewired as `ControlPlanBuilder + ControlScheduler`.
- The old `novasight/control/controller.py::AngularController` is not the active runtime controller and should not be extended for the new chain.

## Risk Points

1. Same observation can be repeatedly consumed.
   - `process_control_tick()` reuses `last_frame_context` at control Hz.
   - Tracker detects repeat frame and keeps state unchanged, but control is still recalculated and can be submitted again.
   - This is the strongest mismatch with the requested mode A.

2. Current PID dt is not measurement dt.
   - Tracker/Kalman velocity uses capture timestamps.
   - Angular D term uses `1/control_hz`.
   - The requested `measurement_dt_s` is not a first-class control input.

3. Scheduler split queue may not actually be consumed as a plan.
   - `CommandScheduler.tick()` and `HidOutputThread` exist.
   - No production wiring was found.
   - Runtime currently calls `scheduler.submit()` through `ExecutorRegistry.execute()` on each control tick.

4. The current prediction path is already partly complex.
   - Kalman + velocity hint + aim EMA + latency compensation are active.
   - The new first version wants simpler, explicit raw velocity -> velocity EMA -> constant velocity prediction.
   - Keeping both could create the forbidden multi-layer smoothing/prediction stack.

5. Strategy naming and fallback are incompatible with the new plan.
   - `control.strategy` only allows `experimental_angle_pid`.
   - There is no `legacy` fallback and no `predictive_pid_v2` switch.

6. DeepStream source files are currently absent under `novasight/deepstream/`.
   - The directory only contains `__pycache__`.
   - The active current path appears to be `nvmm_latest`/TensorRT plus latest frame broker and DetectionBatch construction in RuntimeService, not a live `novasight/deepstream/backend.py` source path.

7. `capture_ts_ns` is userspace receive time, not hardware exposure time or raw PTS.
   - This is internally monotonic and consistent.
   - It is not a physical sensor timestamp. Any future latency claims must keep that distinction.

8. `ControlOutputPolicy` and `AngularPDController` both have limits.
   - Stage 7 should avoid double-counting limits or hiding which layer clipped output.

9. `move_kind=bezier` / `move_ms` can delegate motion shaping to kmNet driver APIs.
   - This may conflict with scheduler-owned small-step plans unless explicitly disabled or modeled.

## Stage 0 End Questions

### Where is current `capture_ts` generated?

`capture_ts_ns` is generated as userspace monotonic receive time in FrameSource implementations:

- Image source: after synthetic pacing.
- OpenCV source: after `cap.read()`.
- GStreamer appsink/resource source: immediately after `try_pull_sample()`.

It is carried as `CapturedFrame.ts_ns -> capture_ts_ns -> RoiFrame.capture_ts_ns -> DetectionBatch.capture_ts_ns -> FrameContext.capture_ts_ns`.

### What is current PID dt?

For the active runtime controller, PID-like D dt is `1 / experimental_angle_control_hz`, clamped to `[1/240, 1/15]`. It is not consecutive detection `capture_ts` measurement dt.

Tracker/Kalman separately uses `capture_ts_ns` deltas for velocity estimation and prediction.

### Does current mouse output direct-send in one shot?

Not purely. Output goes through `ControlOutputPolicy` and `CommandScheduler`, and scheduler can split commands into bounded steps.

However, the active runtime send path emits only the scheduler decision returned by `submit()`. No production consumer was found for pending tail steps, so this is not a complete independent plan scheduler yet.

### Does current code repeatedly consume the same observation?

Yes. `update_control_observation()` records a new observation, while `process_control_tick()` repeatedly recalculates from `last_frame_context` until a newer observation replaces it or target/control gates clear it.

Scheduler cancellation reduces pending debt, but it does not prevent recomputing and resubmitting control from the same visual observation.

### Does current code have an old plan?

It has a latest pending command/step queue inside `CommandScheduler`, with TTL and cancellation rules. It is not a first-class `ControlPlan` tied one-to-one to a source observation with plan id, source capture timestamp, total budget, step schedule, and guaranteed replacement on new observation.

## Stage 0 Decision Needed Before Stage 1

Current phase: stage 0 audit complete.

Found facts:

- The existing active path is `experimental_angle_pid`, not `legacy | predictive_pid_v2`.
- It repeats control computation from `last_frame_context`.
- Its D dt is control-loop dt, not measurement dt.
- Scheduler is latest-pending, but not wired as a per-observation plan consumer.

Unconfirmed condition:

- Whether stage 1 should add the new timing model beside the current strategy in shadow/telemetry mode, or start renaming/refactoring the active strategy boundary immediately.

Risk if unconfirmed:

- Refactoring in place can break the only current kmNet runtime path.
- Adding the new chain beside it adds temporary duplication but preserves rollback and allows stage-by-stage validation.

Recommended options:

- Option A: Add `predictive_pid_v2` alongside current `experimental_angle_pid`, keep current strategy as legacy/current fallback, and make stage 1 telemetry shared by both.
- Option B: Refactor `experimental_angle_pid` in place into the requested modules and later rename it.
- Option C: Add only passive timing telemetry first, with no algorithm switch, then decide module boundaries in stage 2.

Recommended choice:

Option A. It matches the plan requirement for runtime switch and rollback, avoids breaking the active kmNet path, and gives each later stage a clean place to attach tests and telemetry.

Range I will not modify before confirmation:

- I will not change control behavior, strategy validation, scheduler behavior, or executor send behavior until stage 1 is confirmed.
