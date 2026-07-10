# NovaSight Stage 1 Time Model Report

Date: 2026-07-10

Scope: stage 1 only. This stage adds a shared observation timing model and telemetry. It does not change control calculation, Scheduler ownership, or HID/kmNet output behavior.

## Modified Files

- `novasight/runtime/control_timing.py`
  - Adds the control timing state and immutable timing snapshot.
- `novasight/runtime/service.py`
  - Records timing once for each new control observation.
  - Preserves `DetectionBatch.inference_end_ts_ns` in the captured-frame `FrameContext` path.
  - Adds complete timing fields to accepted and rejected DetectionBatch status payloads.
  - Emits a debug timing record for every DetectionBatch and valid target observation.
  - Resets timing continuity on target loss, runtime reset, and configuration update.
- `novasight/runtime/telemetry.py`
  - Adds a top-level `control_timing` telemetry section.
  - Falls back to the latest rejected DetectionBatch timing when no valid target snapshot exists.
- `novasight/runtime/__init__.py`
  - Exports the new timing model and snapshot types.
- `novasight/config/runtime.py` and `novasight/config/schema.py`
  - Use the canonical `configured_extra_prediction_delay_ms` name.
  - Migrate the legacy actuation-delay input key during config loading.
- `novasight/runtime/aim.py`
  - Uses `extra_prediction_delay_ms` internally so the active legacy path does not silently ignore the renamed configuration.
- `config/novasight.example.yaml`
  - Documents that the value is additional prediction lead, not measured end-to-end delay.
- `tests/test_control_timing.py`
  - Adds focused timing semantics and RuntimeService integration tests.

## New Types

### `ControlTimingSnapshot`

Carries one observation's distinct time values:

```text
capture_ts_ns
inference_end_ts_ns
control_now_ts_ns
measurement_dt_s
frame_age_s
configured_extra_prediction_delay_s
prediction_horizon_s
```

### `ControlTimingModel`

Maintains only the previous valid target id and capture timestamp needed for measurement time. It does not estimate velocity and does not calculate or send control output.

## Timing Semantics

```text
measurement_dt_s
= current_capture_ts_ns - previous_same_target_capture_ts_ns

frame_age_s
= max(0, control_now_ts_ns - capture_ts_ns)

prediction_horizon_s
= frame_age_s + configured_extra_prediction_delay_s
```

The model does not substitute control-loop or Scheduler tick time for `measurement_dt_s`.

Target switch, target loss, non-monotonic same-target capture time, runtime reset, and configuration update do not produce a synthetic measurement delta.

## Telemetry

Each latest accepted observation exposes:

```text
frame_id
target_id
capture_ts_ns
inference_end_ts_ns
control_now_ts_ns
measurement_dt_ms
frame_age_ms
configured_extra_prediction_delay_ms
extra_prediction_delay_source=configured_estimate
prediction_horizon_ms
```

Rejected stale/non-latest DetectionBatch payloads expose the same raw time chain with `target_id` and `measurement_dt_ms` unset. Rejected batches do not advance valid measurement continuity.

The `novasight.runtime.service` debug log emits `control_timing event=detection_batch` for every DetectionBatch. Accepted target observations additionally emit `event=target_observation` with resolved `target_id` and `measurement_dt_ms`.

## Replaced Or Removed Logic

- Replaced the captured-frame path's partial manual `FrameContext` construction with the existing `detection_batch_to_frame_context()` adapter so inference timestamps are preserved.
- Removed one unused local assignment around the existing strategy `observe()` call; the call itself remains unchanged.
- No PID, prediction, output limiting, Scheduler, executor, trigger, or kmNet behavior was replaced.

## Configuration

The canonical configuration key is:

```text
control.configured_extra_prediction_delay_ms
```

Telemetry labels it as `configured_estimate`. The existing default remains unchanged and is not treated as an optimal measured value. The legacy input key `control.latency_estimated_actuation_delay_ms` is accepted only by the config migration layer.

This terminology correction followed the self-motion review. The remaining authoritative stage plan is `docs/control/predictive_pid_v2_implementation_plan.md`.

The `legacy | predictive_pid_v2` runtime algorithm switch remains deferred. Enabling an unfinished strategy in stage 1 would misrepresent runtime behavior.

## Tests Added

- Capture delta is used for `measurement_dt_s`.
- Frame age is calculated independently from measurement delta.
- Prediction horizon equals frame age plus configured extra prediction delay.
- Target switch starts a new measurement sequence.
- RuntimeService publishes complete timing for consecutive same-target DetectionBatch observations.
- Telemetry exposes accepted observation timing.
- Telemetry preserves the time chain for rejected batches without advancing continuity.

## Verification Results

```text
./.venv/bin/ruff check novasight/runtime/control_timing.py novasight/runtime/service.py novasight/runtime/telemetry.py novasight/runtime/__init__.py tests/test_control_timing.py
All checks passed

./.venv/bin/python -m compileall -q novasight tests/test_control_timing.py
passed

./.venv/bin/python -m pytest tests/test_control_timing.py tests/test_runtime_pipeline.py::test_runtime_service_rejects_detection_batch_generation_and_capture_rollback tests/test_runtime_pipeline.py::test_runtime_service_rejects_stale_detection_batch_before_control -q
8 passed

./.venv/bin/python -m pytest tests/test_control_timing.py tests/test_config_runtime.py -q
94 passed

./.venv/bin/python -m pytest tests/test_runtime_pipeline.py -q
37 passed, 6 skipped

./.venv/bin/python -m pytest tests/test_control_trace.py tests/test_hardware_control.py::test_command_scheduler_cancels_pending_on_new_frame tests/test_hardware_control.py::test_command_scheduler_splits_large_command_into_steps tests/test_angular_control.py::test_angular_controller_derivative_uses_seconds -q
6 passed

./.venv/bin/python -m pytest -q --ignore=tests/test_frontend_studio_contract.py
251 passed, 8 skipped
```

The unfiltered full suite reported `253 passed, 8 skipped, 1 failed`. The failure is in `tests/test_frontend_studio_contract.py` because the current unrelated `web/src/features/studio/StudioConsoleView.tsx` worktree content no longer contains the test's expected `ROI 裁剪` heading. Stage 1 does not modify frontend files.

## Remaining Risks

1. `capture_ts_ns` remains userspace monotonic receive time, not physical sensor exposure time.
2. `configured_extra_prediction_delay_ms` is a static prediction lead, not a measured physical end-to-end delay.
3. `measurement_dt_ms` is available only after the selector provides two consecutive observations with the same valid track id.
4. The active `experimental_angle_pid` D term still uses control-loop dt. Stage 1 records the correct measurement dt but deliberately does not change PID behavior.
5. The active runtime still repeatedly recalculates control from `last_frame_context`; mode A scheduling is not implemented in this stage.

## Stage End Condition

Satisfied for stage 1:

- `capture_ts_ns` remains monotonic and is not regenerated after inference.
- `inference_end_ts_ns` reaches the observation timing model on both DetectionBatch paths.
- `measurement_dt_ms`, `frame_age_ms`, and `prediction_horizon_ms` have separate formulas and telemetry fields.
- `frame_age_ms` is non-negative.
- `prediction_horizon_ms >= frame_age_ms`.
- Existing mouse behavior, Scheduler behavior, and executor send behavior remain unchanged.

Stage 2 must not start until stage 1 is reviewed and confirmed.
