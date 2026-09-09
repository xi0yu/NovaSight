# ROI Buffer Pipeline Design

## Decision

NovaSight should treat ROI as a first-class runtime buffer, not as a browser preview feature.

The target pipeline is:

```text
Capture
  -> Center ROI crop
  -> RoiFrame / ROI buffer
       -> TensorRT inference
       -> Preview / diagnostics / recording as lower-priority consumers
```

The primary product goal is capture plus inference stability. Browser preview is secondary and must never define, slow, or block the main capture/inference path.

## Goals

- Use one ROI setting for the system.
- Crop the center of the captured frame to one of:
  - `640x640`
  - `480x480`
  - `320x320`
  - `256x256`
- Feed TensorRT from the ROI, not from the full capture frame.
- Make preview show the same ROI when practical.
- Keep preview lower priority than inference.
- Preserve capture throughput by keeping capture ownership separate from ROI consumers.
- Prepare for GPU acceleration on Jetson without forcing a large rewrite later.

## Non-Goals

- Do not make browser preview the owner of ROI.
- Do not run TensorRT from the MJPEG preview path.
- Do not block capture on preview encoding.
- Do not queue old ROI frames for inference. Real-time behavior prefers latest frame over backlog.
- Do not implement unrelated Admin, recording, or logs features as part of ROI.

## Configuration

Use one system ROI config:

```yaml
roi:
  size: 640
  mode: center
```

Allowed `size` values:

```text
640, 480, 320, 256
```

`mode` is fixed to `center` for this phase. Future modes such as manual ROI or dynamic target ROI are out of scope.

## Core Data Model

Introduce a runtime ROI frame concept.

```python
@dataclass(frozen=True)
class RoiFrame:
    frame_id: int
    source_width: int
    source_height: int
    roi_size: int
    offset_x: int
    offset_y: int
    ts_ns: int
    pixel_format: str
    image: Any | None
    gpu_buffer: Any | None = None
```

Meaning:

- `frame_id`: source capture frame id.
- `source_width/source_height`: full captured frame size.
- `roi_size`: square ROI size after clamping.
- `offset_x/offset_y`: ROI origin in source-frame coordinates.
- `image`: CPU fallback image, used in first implementation or preview.
- `gpu_buffer`: future GPU/NVMM/CUDA buffer handle.

If the source frame is smaller than the configured ROI size, clamp ROI to the largest centered square that fits.

## Pipeline Shape

### Capture Thread

Capture remains responsible only for reading frames and publishing latest source frames.

```text
CaptureSession thread:
  source.read()
  publish latest CapturedFrame
```

It must not do:

- TensorRT inference
- JPEG encoding
- browser overlay drawing
- blocking ROI consumers
- long-running CPU preprocessing

### ROI Processor

The ROI processor consumes the latest captured frame and produces the latest ROI frame.

```text
CapturedFrame
  -> RoiProcessor
  -> latest RoiFrame
```

The ROI processor may run in the inference thread or in its own thread. It must use latest-frame semantics:

```text
capacity = 1
new ROI replaces old ROI
slow consumers drop old frames
```

### Inference Consumer

TensorRT consumes `RoiFrame`, not full `CapturedFrame`.

```text
RoiFrame
  -> TensorRT preprocess
  -> TensorRT inference
```

Detection coordinates from the model are ROI-local. Before control output, convert to full-frame coordinates:

```text
full_x = roi_x + offset_x
full_y = roi_y + offset_y
```

The `FrameContext` passed to plugins/control should make coordinate space explicit. For compatibility, the first implementation may map detections back to full-frame coordinates before constructing `FrameContext`.

### Preview Consumer

Preview should consume `RoiFrame` when available.

```text
RoiFrame
  -> low-priority MJPEG preview
```

Preview may be delayed or lower FPS. It must not block ROI processing or inference.

If ROI buffer sharing is not available yet, preview can independently crop from the latest captured frame using the same ROI config as a transitional fallback. This is acceptable only while GPU ROI buffer export is not implemented.

## GPU Acceleration Direction

The final Jetson target is GPU-side ROI preprocessing:

```text
NVMM / CUDA source buffer
  -> center crop
  -> resize to TensorRT input shape
  -> color convert
  -> normalize / NCHW
  -> TensorRT input buffer
```

A useful final form is one GPU preprocess stage that can produce:

```text
GPU ROI preprocess
  -> TensorRT input tensor
  -> optional preview ROI frame/buffer
```

This avoids duplicate ROI definitions while keeping preview lower priority.

## Implementation Phases

### Phase 1: CPU ROI Semantics

Purpose: validate behavior, config, coordinates, and UI.

- Add `roi.size` to runtime config schema.
- Implement center ROI calculation.
- Produce `RoiFrame` with CPU image from `CapturedFrame.image`.
- Feed inference from ROI frame.
- Map detections back to full-frame coordinates.
- Make preview use the same ROI config.
- Add tests for ROI clamping and coordinate offsets.

This phase may use CPU crop. It is not the final performance target.

### Phase 2: Inference-Side GPU Preprocess

Purpose: keep capture stable while moving ROI preprocessing off CPU.

- Keep `CaptureSession` ownership unchanged.
- Add TensorRT preprocessing boundary that accepts `RoiFrame` or source GPU buffer.
- Implement GPU crop/resize/normalize before TensorRT.
- Keep latest-frame semantics.
- Measure capture FPS separately from inference FPS.

### Phase 3: Preview From ROI Buffer

Purpose: make browser preview show the actual inference ROI without impacting inference.

- Export low-priority preview frame from ROI processor.
- Limit preview FPS.
- Drop preview frames when browser/JPEG encoding is slow.
- Keep inference consumer higher priority.

## UI Behavior

Add one ROI selector in Studio Config or Devices:

```text
ROI size: [640] [480] [320] [256]
```

Copy should be explicit:

```text
中心 ROI 同时用于 TensorRT 推理和浏览器预览。采集分辨率不变。
```

Preview panel should display:

```text
ROI 640x640 · center crop
```

The UI should not present separate `preview_roi` and `inference_roi` controls in this phase.

## Performance Rules

- Capture FPS is measured before ROI.
- Inference FPS is measured after ROI/TensorRT.
- Preview FPS is independent and lower priority.
- Slow preview must drop frames.
- Slow inference must consume the latest ROI, not build backlog.
- ROI processing must not happen inside the capture read loop.

## Risks

### CPU Fallback Can Mislead Performance

Phase 1 CPU crop is only for correctness. Label it as CPU fallback in diagnostics if exposed.

### Coordinate Space Bugs

ROI-local detections must be mapped back to source coordinates before control output unless the whole control stack becomes ROI-aware.

### GPU Buffer Lifetime

Preview must not hold the main inference buffer long enough to stall TensorRT. Use copy/export or low-priority buffer snapshots for preview.

### Model Input Shape Mismatch

`roi.size` is not necessarily the TensorRT input shape. The preprocess stage still needs resize to the model input shape.

## Acceptance Criteria

- Changing ROI size does not restart or slow capture.
- Capture metrics remain based on source frame acquisition.
- TensorRT receives ROI content rather than full frame content.
- Browser preview shows the same center ROI region.
- Detection/control coordinates remain correct in full-frame space.
- Preview slowdown does not reduce inference throughput.
- Tests cover ROI center crop offsets for `640`, `480`, `320`, and `256`.
