# ROI Buffer Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one center ROI pipeline setting that produces a reusable ROI frame for inference and preview without moving ROI work into the capture thread.

**Architecture:** Phase 1 implements CPU ROI semantics and tests while preserving the future GPU boundary. Capture continues to publish full `CapturedFrame`; runtime/inference derives `RoiFrame` from the latest captured frame; preview renders the same ROI. GPU/NVMM preprocessing remains a later implementation behind the same ROI processor interface.

**Tech Stack:** Python dataclasses, FastAPI routes/config schema, existing runtime/capture services, React/TypeScript Studio config UI, pytest.

---

## File Structure

- Create `novasight/roi.py`: ROI config constants, `RoiFrame`, center-crop helpers, coordinate mapping helpers.
- Modify `novasight/config/runtime.py`: add `roi` config section.
- Modify `novasight/config/schema.py`: expose `roi.size` in `/api/config/schema`.
- Modify `config/novasight.example.yaml`: document default ROI config.
- Modify `novasight/runtime/service.py`: derive ROI frame before inference and map detections back to full-frame coordinates.
- Modify `novasight/inference/contracts.py`: allow engines to accept ROI frames through the same structural interface.
- Modify `novasight/capture/preview.py`: render preview from ROI using the shared ROI config.
- Modify `novasight/api/routes_capture.py`: pass configured ROI size into preview rendering and show ROI metadata where practical.
- Modify `web/src/api.ts`: include `roi` config values through existing runtime config typing if needed.
- Modify `web/src/features/config/ConfigView.tsx`: no custom field hardcoding should be required if schema exposes `roi.size`; verify label/copy is clear.
- Add `tests/test_roi.py`: ROI crop, clamping, offsets, coordinate mapping.
- Add/update runtime/preview tests where necessary.

## Task 1: ROI Domain Model And Unit Tests

**Files:**
- Create: `novasight/roi.py`
- Create: `tests/test_roi.py`

- [ ] **Step 1: Write ROI tests**

Create `tests/test_roi.py`:

```python
import numpy as np

from novasight.capture.source import CapturedFrame
from novasight.roi import (
    ROI_SIZE_CHOICES,
    center_roi_frame,
    map_detection_to_source,
)
from novasight.plugins import Detection


def make_frame(width: int, height: int) -> CapturedFrame:
    image = np.arange(height * width * 3, dtype=np.uint8).reshape((height, width, 3))
    return CapturedFrame(
        frame_id=7,
        width=width,
        height=height,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=1.5,
        image=image,
    )


def test_allowed_roi_sizes_are_fixed():
    assert ROI_SIZE_CHOICES == (640, 480, 320, 256)


def test_center_roi_frame_crops_requested_square():
    frame = make_frame(width=1920, height=1080)
    roi = center_roi_frame(frame, requested_size=640)

    assert roi.frame_id == 7
    assert roi.source_width == 1920
    assert roi.source_height == 1080
    assert roi.roi_size == 640
    assert roi.offset_x == 640
    assert roi.offset_y == 220
    assert roi.image.shape == (640, 640, 3)
    np.testing.assert_array_equal(roi.image, frame.image[220:860, 640:1280])


def test_center_roi_frame_clamps_to_source_short_side():
    frame = make_frame(width=500, height=300)
    roi = center_roi_frame(frame, requested_size=640)

    assert roi.roi_size == 300
    assert roi.offset_x == 100
    assert roi.offset_y == 0
    assert roi.image.shape == (300, 300, 3)


def test_center_roi_frame_rejects_invalid_size():
    frame = make_frame(width=1920, height=1080)

    try:
        center_roi_frame(frame, requested_size=512)
    except ValueError as exc:
        assert "unsupported ROI size" in str(exc)
    else:
        raise AssertionError("expected invalid ROI size to fail")


def test_map_detection_to_source_adds_roi_offset():
    detection = Detection(cls="target", score=0.8, x=10, y=20, w=30, h=40)
    mapped = map_detection_to_source(detection, offset_x=640, offset_y=220)

    assert mapped.cls == "target"
    assert mapped.score == 0.8
    assert mapped.x == 650
    assert mapped.y == 240
    assert mapped.w == 30
    assert mapped.h == 40
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_roi.py -q`

Expected: fail because `novasight.roi` does not exist.

- [ ] **Step 3: Implement ROI model**

Create `novasight/roi.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from novasight.capture.source import CapturedFrame
from novasight.plugins import Detection


ROI_SIZE_CHOICES = (640, 480, 320, 256)


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


def normalize_roi_size(value: int) -> int:
    if int(value) not in ROI_SIZE_CHOICES:
        raise ValueError(f"unsupported ROI size: {value}")
    return int(value)


def center_roi_frame(frame: CapturedFrame, *, requested_size: int) -> RoiFrame:
    configured_size = normalize_roi_size(requested_size)
    roi_size = min(configured_size, int(frame.width), int(frame.height))
    offset_x = max(0, (int(frame.width) - roi_size) // 2)
    offset_y = max(0, (int(frame.height) - roi_size) // 2)
    image = _crop_image(frame.image, offset_x=offset_x, offset_y=offset_y, size=roi_size)
    return RoiFrame(
        frame_id=frame.frame_id,
        source_width=frame.width,
        source_height=frame.height,
        roi_size=roi_size,
        offset_x=offset_x,
        offset_y=offset_y,
        ts_ns=frame.ts_ns,
        pixel_format=frame.pixel_format,
        image=image,
    )


def map_detection_to_source(
    detection: Detection,
    *,
    offset_x: int,
    offset_y: int,
) -> Detection:
    return Detection(
        cls=detection.cls,
        score=detection.score,
        x=detection.x + offset_x,
        y=detection.y + offset_y,
        w=detection.w,
        h=detection.h,
    )


def _crop_image(image: Any, *, offset_x: int, offset_y: int, size: int) -> Any | None:
    if image is None:
        return None
    if hasattr(image, "shape"):
        import numpy as np

        return np.ascontiguousarray(image[offset_y : offset_y + size, offset_x : offset_x + size])
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.crop((offset_x, offset_y, offset_x + size, offset_y + size))
    except Exception:
        return None
    return None
```

- [ ] **Step 4: Run ROI tests**

Run: `pytest tests/test_roi.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add novasight/roi.py tests/test_roi.py
git commit -m "Add ROI frame domain model"
```

## Task 2: Runtime Config And Schema

**Files:**
- Modify: `novasight/config/runtime.py`
- Modify: `novasight/config/schema.py`
- Modify: `config/novasight.example.yaml`
- Add or modify: `tests/test_config_runtime.py`

- [ ] **Step 1: Inspect current config dataclasses**

Run: `sed -n '1,260p' novasight/config/runtime.py`

Use existing config style. Add a new dataclass:

```python
@dataclass(frozen=True)
class RoiConfig:
    size: int = 640
    mode: str = "center"
```

Add `roi: RoiConfig = field(default_factory=RoiConfig)` to `RuntimeConfig`.

- [ ] **Step 2: Add parser validation**

In `parse_runtime_config`, parse `roi.size` and reject values outside `(640, 480, 320, 256)`. Reject any `roi.mode` other than `"center"`.

Expected error examples:

```text
unsupported ROI size: 512
unsupported ROI mode: manual
```

- [ ] **Step 3: Expose schema fields**

Modify `novasight/config/schema.py` to add section:

```python
{
    "id": "roi",
    "label": "ROI",
    "fields": [
        {
            "path": "roi.size",
            "label": "中心 ROI",
            "type": "select",
            "options": ["640", "480", "320", "256"],
            "restart_required": False,
        },
    ],
}
```

Do not expose `roi.mode` yet because only center is supported.

- [ ] **Step 4: Update example config**

Add to `config/novasight.example.yaml`:

```yaml
roi:
  size: 640
  mode: center
```

- [ ] **Step 5: Add config tests**

Add tests to `tests/test_config_runtime.py`:

```python
from dataclasses import asdict

import pytest

from novasight.config import RuntimeConfig, parse_runtime_config
from novasight.config.schema import runtime_config_schema


def test_runtime_config_has_default_roi():
    config = RuntimeConfig()
    assert config.roi.size == 640
    assert config.roi.mode == "center"


def test_parse_runtime_config_accepts_allowed_roi_size():
    config = parse_runtime_config({"roi": {"size": 320, "mode": "center"}})
    assert config.roi.size == 320


def test_parse_runtime_config_rejects_invalid_roi_size():
    with pytest.raises(ValueError, match="unsupported ROI size"):
        parse_runtime_config({"roi": {"size": 512, "mode": "center"}})


def test_runtime_config_schema_exposes_roi_size():
    schema = runtime_config_schema(RuntimeConfig())
    roi_section = next(section for section in schema["sections"] if section["id"] == "roi")
    assert roi_section["fields"][0]["path"] == "roi.size"
    assert roi_section["fields"][0]["options"] == ["640", "480", "320", "256"]
```

Adjust imports if the existing test file already imports these helpers.

- [ ] **Step 6: Run config tests**

Run: `pytest tests/test_config_runtime.py -q`

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add novasight/config/runtime.py novasight/config/schema.py config/novasight.example.yaml tests/test_config_runtime.py
git commit -m "Add runtime ROI configuration"
```

## Task 3: Feed Inference From ROI Frame

**Files:**
- Modify: `novasight/runtime/service.py`
- Modify: `novasight/inference/contracts.py`
- Modify: `novasight/inference/runtime.py`
- Modify: `novasight/inference/tensorrt.py`
- Modify or add: `tests/test_runtime_service_roi.py`

- [ ] **Step 1: Add runtime service tests**

Create `tests/test_runtime_service_roi.py`:

```python
from dataclasses import dataclass

import numpy as np

from novasight.capture.source import CapturedFrame
from novasight.config.runtime import RuntimeConfig, RoiConfig
from novasight.inference.contracts import InferenceResult
from novasight.plugins import Detection
from novasight.runtime.service import RuntimeService


class EmptyModels:
    def get_active_deployment(self):
        return None


class EmptyPlugins:
    def __init__(self):
        self.last_context = None

    def process(self, context):
        self.last_context = context
        return type("Batch", (), {"control_intents": []})()


class EmptyExecutors:
    def status(self):
        return {"selected": "dry_run", "executors": {}}

    def execute(self, intent):
        return intent


class RoiAwareInference:
    def __init__(self):
        self.last_frame = None

    def infer(self, frame):
        self.last_frame = frame
        return InferenceResult(
            available=True,
            detections=[Detection(cls="target", score=0.9, x=10, y=20, w=30, h=40)],
            classes=["target"],
        )


def make_frame(width=1920, height=1080):
    return CapturedFrame(
        frame_id=1,
        width=width,
        height=height,
        pixel_format="BGR",
        ts_ns=1,
        capture_wait_ms=0.1,
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def test_runtime_infers_on_roi_and_maps_detection_to_source_coordinates():
    plugins = EmptyPlugins()
    inference = RoiAwareInference()
    runtime = RuntimeService(
        config=RuntimeConfig(roi=RoiConfig(size=640)),
        models=EmptyModels(),
        plugins=plugins,
        executors=EmptyExecutors(),
        inference=inference,
    )

    runtime.process_captured_frame(make_frame())

    assert inference.last_frame.roi_size == 640
    assert inference.last_frame.offset_x == 640
    assert inference.last_frame.offset_y == 220
    assert plugins.last_context.width == 1920
    assert plugins.last_context.height == 1080
    detection = plugins.last_context.detections[0]
    assert detection.x == 650
    assert detection.y == 240
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_runtime_service_roi.py -q`

Expected: fail because `RuntimeService` still passes full `CapturedFrame`.

- [ ] **Step 3: Update inference contracts**

Modify `novasight/inference/contracts.py` so `InferenceEngine.infer` accepts a structurally frame-like object. If the protocol currently requires `CapturedFrame`, change the annotation to `Any` or `CapturedFrame | RoiFrame` while avoiding import cycles.

Keep the runtime behavior unchanged.

- [ ] **Step 4: Update RuntimeService**

In `RuntimeService.process_captured_frame`:

1. Build ROI frame:

```python
roi_frame = center_roi_frame(frame, requested_size=self.config.roi.size)
```

2. Call inference with `roi_frame`.

3. If inference returns detections, map them back:

```python
map_detection_to_source(item, offset_x=roi_frame.offset_x, offset_y=roi_frame.offset_y)
```

4. Build `FrameContext` with full source frame dimensions:

```python
width=frame.width
height=frame.height
```

5. If ROI creation fails, fall back to existing empty context.

- [ ] **Step 5: Update TensorRT stub typing**

Modify `novasight/inference/tensorrt.py` and `novasight/inference/runtime.py` annotations to accept the same frame-like object. No real TensorRT implementation is required in this phase.

- [ ] **Step 6: Run runtime ROI tests**

Run: `pytest tests/test_runtime_service_roi.py -q`

Expected: pass.

- [ ] **Step 7: Run existing runtime tests**

Run: `pytest tests/test_runtime_pipeline.py tests/test_runtime_api.py tests/test_inference_runtime.py -q`

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add novasight/runtime/service.py novasight/inference/contracts.py novasight/inference/runtime.py novasight/inference/tensorrt.py tests/test_runtime_service_roi.py
git commit -m "Feed runtime inference from ROI frames"
```

## Task 4: Preview Uses Same ROI Config

**Files:**
- Modify: `novasight/capture/preview.py`
- Modify: `novasight/api/routes_capture.py`
- Modify or add: `tests/test_capture_preview.py`

- [ ] **Step 1: Add preview ROI test**

Update `tests/test_capture_preview.py` with:

```python
import numpy as np

from novasight.capture.preview import render_preview_frame
from novasight.capture.source import CapturedFrame


def test_preview_renders_center_roi_size():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = CapturedFrame(
        frame_id=1,
        width=1920,
        height=1080,
        pixel_format="BGR",
        ts_ns=1,
        capture_wait_ms=0.1,
        image=image,
    )

    preview = render_preview_frame(frame, roi_size=320)

    assert preview.size == (320, 320)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_capture_preview.py -q`

Expected: fail because `render_preview_frame` does not accept `roi_size`.

- [ ] **Step 3: Update preview renderer**

Modify `render_preview_frame`:

```python
def render_preview_frame(frame: CapturedFrame, *, runtime=None, fov_ratio=0.28, roi_size: int | None = None):
```

If `roi_size` is provided:

```python
roi_frame = center_roi_frame(frame, requested_size=roi_size)
image = roi_frame.image
width = roi_frame.roi_size
height = roi_frame.roi_size
```

Draw overlay on the ROI image. Do not run `runtime.inference.infer(frame)` inside preview. Preview must not trigger inference. If detections are needed later, they must come from a cached runtime result, not direct inference.

- [ ] **Step 4: Pass ROI config from capture route**

In `novasight/api/routes_capture.py`, get:

```python
roi_size = getattr(getattr(request.app.state, "config", None), "roi", None).size
```

Pass `roi_size=roi_size` into `_mjpeg_frames`, then into `render_preview_frame`.

- [ ] **Step 5: Run preview tests**

Run: `pytest tests/test_capture_preview.py -q`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add novasight/capture/preview.py novasight/api/routes_capture.py tests/test_capture_preview.py
git commit -m "Render preview from configured ROI"
```

## Task 5: Studio ROI Config UI Verification

**Files:**
- Modify only if needed: `web/src/features/config/ConfigView.tsx`
- Modify only if needed: `web/src/api.ts`

- [ ] **Step 1: Verify schema-driven UI displays ROI**

Because `ConfigView` renders `/api/config/schema`, no custom React field should be required if Task 2 added `roi.size`.

Run backend and frontend manually or use a schema test:

```bash
python3 -m novasight --host 127.0.0.1 --port 5174
pnpm --dir web dev
```

Expected Config page shows ROI section with center ROI select values `640`, `480`, `320`, `256`.

- [ ] **Step 2: Improve copy only if schema label is unclear**

If needed, change schema label from `"中心 ROI"` to `"中心 ROI 尺寸"` and keep options unchanged.

- [ ] **Step 3: Run frontend checks**

Run: `pnpm --dir web typecheck`

Expected: pass.

Run: `pnpm --dir web build`

Expected: pass.

- [ ] **Step 4: Commit if files changed**

If no files changed, do not create an empty commit. If copy changed:

```bash
git add novasight/config/schema.py
git commit -m "Clarify ROI config label"
```

## Task 6: Full Verification

**Files:**
- No planned edits.

- [ ] **Step 1: Run focused ROI tests**

Run:

```bash
pytest tests/test_roi.py tests/test_runtime_service_roi.py tests/test_capture_preview.py -q
```

Expected: pass.

- [ ] **Step 2: Run backend test suite**

Run:

```bash
pytest -q
```

Expected: pass.

- [ ] **Step 3: Run frontend checks**

Run:

```bash
pnpm --dir web typecheck
pnpm --dir web build
```

Expected: pass.

- [ ] **Step 4: Manual runtime smoke**

With a valid local license and backend running:

```bash
curl http://127.0.0.1:5174/api/config/schema
curl -X PUT http://127.0.0.1:5174/api/config \
  -H "Content-Type: application/json" \
  -d '{"roi":{"size":320,"mode":"center"}}'
```

Expected:

- schema includes ROI section.
- config accepts allowed values.
- invalid ROI values return 400.

## Self-Review

Spec coverage:

- One `roi.size` config is implemented.
- ROI buffer is modeled as a reusable runtime concept.
- Inference consumes ROI content.
- Preview uses the same ROI config.
- Capture thread ownership remains untouched.
- GPU acceleration remains a future boundary, not a fake implementation.

Placeholder scan:

- No `TBD`, `TODO`, or placeholder tasks are present.

Type consistency:

- `RoiFrame` is introduced in `novasight.roi`.
- Runtime service uses `RoiFrame` for inference and source-frame dimensions for plugin/control context.
- Preview receives `roi_size` explicitly from runtime config.
