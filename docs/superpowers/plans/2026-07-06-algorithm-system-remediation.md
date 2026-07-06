# NovaSight Algorithm System Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NovaSight's production control path conform to the remediation baseline: `error_px -> error_rad -> angular PD -> calibrated counts -> scheduler`.

**Architecture:** The current code already contains an experimental angle PID path, but too much logic lives inside one strategy and some fallbacks violate the design document. The remediation will first extract a deep angular-control module with explicit calibration, angular error mapping, angular PD, residual counts, and invalid-state handling, then route `experimental_angle_pid` through it before expanding into a full scheduler and calibration profile model.

**Tech Stack:** Python 3, dataclasses, FastAPI runtime service, existing `RuntimeConfig`, existing `RuntimePipeline`, existing pytest suite.

---

### Task 1: Contract Mapping And Phase Boundary

**Files:**
- Add: `docs/superpowers/plans/2026-07-06-algorithm-system-remediation.md`
- Reference: `/Users/zhangxiaoyu/Downloads/NovaSight-算法系统整改设计书-简体中文版.md`

- [x] **Step 1: Record the first executable phase**

The first phase is intentionally narrow:

```text
Target/Track ROI bbox
-> Kalman/Hungarian observation state already in ExperimentalAnglePidStrategy
-> AngularErrorMapper using full source/control dimensions
-> AngularPDController using radians
-> calibrated counts with residual accumulation
-> MoveCommand for the current executor path
```

- [x] **Step 2: Record known gaps after phase 1**

The following are still required for full design-book compliance:

```text
1. Formal CalibrationProfile separate from RuntimeConfig.control.
2. Formal CommandScheduler with TTL, pending cancellation, splitting, and stale command cleanup.
3. Full TrackState / EstimatedState / CompensatedTarget data classes at runtime boundaries.
4. Replay logs and static guards proving no pixel-domain control reaches DeviceAdapter.
5. UI split between Calibration, UserConfig, RuntimeState, and AdaptiveParameters.
```

### Task 2: Angular Control Module

**Files:**
- Add: `novasight/control/angular.py`
- Modify: `novasight/control/__init__.py`
- Test: `tests/test_angular_control.py`

- [ ] **Step 1: Add an angular control module**

Create `CalibrationProfile`, `AngularErrorState`, `AngularPDConfig`, `AngularControlOutput`, `AngularErrorMapper`, and `AngularPDController`.

Required behavior:

```text
fov_x_rad = radians(fov_x_deg)
fov_y_rad = 2 * atan(tan(fov_x_rad / 2) * H / W)
focal_x = W / (2 * tan(fov_x_rad / 2))
focal_y = H / (2 * tan(fov_y_rad / 2))
error_x_rad = atan(error_x_px / focal_x)
error_y_rad = atan(error_y_px / focal_y)
counts_per_rad = counts_per_360 / (2 * pi)
```

The module must reject invalid control geometry instead of falling back to ROI dimensions.

- [ ] **Step 2: Add residual counts**

The controller must keep independent `residual_x_counts` and `residual_y_counts`:

```text
accum_x = counts_x_float + residual_x_prev
emit_x = trunc_toward_zero(accum_x)
residual_x_next = accum_x - emit_x
```

### Task 3: Route Experimental Strategy Through Angular Control

**Files:**
- Modify: `novasight/control/strategy.py`
- Test: `tests/test_angular_control.py`

- [ ] **Step 1: Replace inline focal/atan/counts code**

`ExperimentalAnglePidStrategy.calculate()` must call the new mapper and controller. It must not use ROI dimensions for focal length. If source/control dimensions are unavailable, it must return a zero command with an explicit invalid reason.

- [ ] **Step 2: Normalize Y sign location**

The mapper uses image coordinates:

```text
error_y_px = comp_y - center_y
```

Only `CalibrationProfile.axis_sign_y` changes device direction.

### Task 4: Runtime Metadata Contract

**Files:**
- Modify: `novasight/runtime/service.py`
- Test: `tests/test_runtime_service_roi.py` or a new focused test if existing local edits should not be touched.

- [ ] **Step 1: Ensure full source dimensions reach strategy metadata**

`_strategy_frame_metadata()` must include source/control width and height from the original capture frame, not only ROI size.

- [ ] **Step 2: Preserve capture timestamp**

`capture_ts_ns` remains the monotonic timestamp recorded by capture and must be passed into the strategy on observation and control ticks.

### Task 5: Verification And Commit

**Files:**
- Test files from previous tasks

- [ ] **Step 1: Run compile verification**

Run:

```bash
python3 -m compileall novasight
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest tests/test_angular_control.py tests/test_runtime_pipeline.py tests/test_capture_session.py -q
```

- [ ] **Step 3: Commit phase 1**

Run:

```bash
git add docs/superpowers/plans/2026-07-06-algorithm-system-remediation.md novasight/control/angular.py novasight/control/__init__.py novasight/control/strategy.py tests/test_angular_control.py
git commit -m "feat: enforce angular control contract"
```

---

## Coverage Notes

This first phase covers the core geometry contract, calibrated counts conversion, residual accumulation, invalid geometry rejection, and the current production strategy entry point. It intentionally does not claim the full design-book remediation is complete until CommandScheduler, CalibrationProfile persistence, replay logging, static architecture guards, and UI parameter separation are implemented.
