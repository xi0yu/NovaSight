# GStreamer ROI Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ROI-aware GStreamer appsink capture path that can deliver center-cropped ROI frames with source-coordinate metadata using GPU/NVMM candidates only.

**Architecture:** Capture candidates receive the runtime ROI size and build ROI-sized `nvvidconv left/right/top/bottom` pipelines. `CapturedFrame` carries optional ROI metadata. `center_roi_frame()` reuses an already-cropped ROI frame when metadata matches the configured size, avoiding duplicate crop work before inference.

**Tech Stack:** Python dataclasses, GStreamer pipeline string generation, existing FastAPI runtime config application, pytest, TypeScript build verification.

---

## Files

- Modify: `novasight/roi.py` for reusable center ROI region calculation and metadata reuse.
- Modify: `novasight/capture/source.py` to add optional source/ROI metadata on `CapturedFrame`.
- Modify: `novasight/capture/pipeline.py` to build ROI-aware appsink candidates.
- Modify: `novasight/capture/service.py` to pass runtime ROI into source creation.
- Modify: `novasight/api/app.py` and `novasight/api/routes_runtime.py` to keep `CaptureService.roi_size` synchronized.
- Test: `tests/test_roi.py`, `tests/test_capture_pipeline.py`, `tests/test_runtime_service_roi.py`, `tests/test_runtime_api.py`.

## Task 1: ROI Metadata Domain Behavior

- [x] Add failing tests in `tests/test_roi.py` for center region calculation and reuse of a pre-cropped ROI frame.
- [x] Run `pytest tests/test_roi.py -q` and confirm the new tests fail.
- [x] Add `center_roi_region()` and pre-cropped ROI reuse to `novasight/roi.py`.
- [x] Run `pytest tests/test_roi.py -q` and confirm it passes.

## Task 2: ROI-Aware Capture Pipeline Candidates

- [x] Add failing tests in `tests/test_capture_pipeline.py` proving `build_appsink_candidates(profile, roi_size=320)` emits ROI-sized caps and center crop coordinates.
- [x] Run the focused pipeline tests and confirm they fail.
- [x] Update `novasight/capture/pipeline.py` to accept `roi_size` and produce ROI output dimensions for appsink candidates.
- [x] Run focused pipeline tests and confirm they pass.

## Task 3: Capture Service ROI Wiring

- [x] Add failing tests proving `CaptureService` passes ROI size to the default source factory and config updates sync ROI size.
- [x] Run focused tests and confirm they fail.
- [x] Update `CaptureService`, `create_app()`, and `_apply_config()` to keep capture ROI size synchronized.
- [x] Run focused tests and confirm they pass.

## Task 4: Runtime No Double Crop

- [x] Add a failing runtime test proving a pre-cropped `CapturedFrame` is passed to inference as one `RoiFrame` with original source offsets.
- [x] Run focused runtime ROI tests and confirm they fail.
- [x] Update `CapturedFrame` metadata defaults and `center_roi_frame()` reuse logic.
- [x] Run focused runtime ROI tests and confirm they pass.

## Task 5: Verification And Commit

- [ ] Run `pytest -q`.
- [ ] Run `pnpm --dir web typecheck`.
- [ ] Run `pnpm --dir web build`.
- [ ] Run `git diff --check`.
- [ ] Commit and push to `origin/develop-alpha`.
