# GStreamer ROI Capture Design

## Goal

Move ROI closer to the Jetson capture path without changing NovaSight's core runtime boundary: inference receives one center ROI frame, detections are mapped back to source-frame coordinates, and browser preview remains lower priority.

## Current Findings

- Runtime config already has one ROI setting: `roi.size` with `640`, `480`, `320`, or `256`.
- `RuntimeService.process_captured_frame()` currently derives a CPU `RoiFrame` before inference.
- `CaptureService` currently receives only `CaptureConfig`, so the GStreamer source factory cannot build ROI-sized pipelines.
- `CapturedFrame` currently stores only the delivered frame size. If capture delivers an already-cropped ROI, runtime needs extra metadata to know the original source size and ROI origin.
- `TensorRtInferenceEngine` is still a placeholder. This phase must not pretend to have implemented CUDA/TensorRT bindings that are not present.

## Recommended Approach

Add an ROI-aware capture source path for GStreamer appsink while keeping CPU ROI fallback.

The intended data flow is:

```text
V4L2 camera
  -> GStreamer decode / convert
  -> center ROI crop in nvvidconv-capable pipeline where available
  -> CapturedFrame(image=ROI, source_width/source_height, roi_offset_x/roi_offset_y)
  -> RuntimeService
  -> center_roi_frame() reuses existing ROI frame instead of cropping again
  -> InferenceRuntime / TensorRT engine
```

## Scope

Implement in this phase:

- Compute center ROI region once from source dimensions and configured ROI size.
- Pass runtime `roi.size` into `CaptureService`.
- Build ROI-sized GStreamer appsink candidates when an ROI size is configured.
- Carry source-frame dimensions and ROI offset on `CapturedFrame`.
- Make `center_roi_frame()` reuse an already-cropped matching ROI frame.
- Keep CPU fallback behavior for full-frame sources and non-GStreamer paths.
- Add tests proving no double crop and correct coordinate offsets.

Do not implement in this phase:

- Real TensorRT CUDA input binding.
- DMA-BUF export into TensorRT.
- Manual ROI position, dynamic target ROI, or separate preview/inference ROI settings.
- Capture restart on ROI setting changes. Existing running captures continue until reconfigured; runtime CPU ROI still keeps inference semantically correct.

## Runtime Rules

- If a frame has ROI metadata and matches the requested ROI size, runtime must treat it as the canonical ROI frame.
- If a frame has no ROI metadata, runtime must center-crop from the delivered image as it does today.
- Detection mapping always uses `RoiFrame.offset_x` and `RoiFrame.offset_y`.
- Browser preview can render from the delivered ROI or CPU fallback ROI; it must not invoke inference.

## Risks

- GStreamer crop property support can vary by Jetson image and plugin version. Candidate fallback ordering must include CPU-compatible paths.
- If ROI config changes while capture is already running, the capture pipeline will not automatically rebuild. The runtime CPU ROI path preserves correctness, but GPU crop takes effect on the next capture reconfigure.
- The current TensorRT engine is not a real inference implementation, so end-to-end GPU buffer validation remains future work.

## Acceptance Criteria

- GStreamer appsink candidate strings include the configured ROI output size and center crop coordinates.
- Captured ROI frames preserve `source_width`, `source_height`, `roi_offset_x`, and `roi_offset_y`.
- Runtime inference receives a `RoiFrame` with the configured ROI size and correct source offsets without performing a second crop.
- Existing CPU full-frame ROI tests still pass.
- Full backend tests, frontend typecheck, and frontend build pass.
