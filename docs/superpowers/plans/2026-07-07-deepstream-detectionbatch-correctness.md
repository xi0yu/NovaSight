# DeepStream DetectionBatch Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepStream a complete, verifiable `DetectionBatch` producer where GPU/NVMM center ROI crop happens before model resize, and incorrect batches are rejected before algorithms or control can trust them.

**Architecture:** Treat DeepStream as a detection source, not as a generic ONNXRuntime/TensorRT engine. DeepStream owns capture, hardware decode, NVMM ROI crop, ROI resize, `nvinfer`, and tensor meta extraction; downstream runtime only receives NovaSight `DetectionBatch` / `FrameContext`. Correctness is guarded by a small number of high-value tests instead of broad test sprawl.

**Tech Stack:** Python, pytest, GStreamer/DeepStream concepts, `pyds` on Jetson, NovaSight `DetectionBatch`, model manifest, shared YOLO parser.

---

## Testing Policy

Keep tests intentionally small. Do not create dozens of files or exhaustive case matrices.

Use these four gates only:

1. ROI pipeline gate: proves DeepStream crops center ROI in NVMM before resize.
2. Tensor contract gate: proves tensor shape/layout can be validated against manifest before parsing.
3. Golden tensor gate: proves parsed `DetectionBatch` bbox/class/score/coordinate space are correct.
4. Runtime seam gate: proves downstream runtime consumes `DetectionBatch` without depending on DeepStream types.

Prefer adding tests to existing files:

- `tests/test_inference_runtime.py` for DeepStream pipeline, manifest, tensor, parser, backend behavior.
- `tests/test_runtime_pipeline.py` only if runtime loop behavior must be proven.

Do not add broad snapshot tests, UI tests, or per-format tests until Jetson evidence shows which formats must be production-supported.

## Implementation Order

Do not start by wiring DeepStream into runtime. The first implementation target is
to make prerequisites explicit and machine-checkable.

Use this order:

```text
P0. Environment and artifact readiness
P1. Manifest and deepstream.ini correctness
P2. DeepStream ROI crop/resize pipeline correctness
P3. Tensor meta contract and golden DetectionBatch correctness
P4. Runtime seam for verified DetectionBatch
P5. Jetson shell/probe evidence
P6. Runtime selection and A/B comparison
```

If P0 or P1 fails, stop and report the missing prerequisite. Do not silently fall
back to CPU or pretend DeepStream is active.

## File Structure

- Modify: `novasight/model_registry/scanner.py`
  - Keep artifact readiness status explicit: `ready`, `need_confirm`, `invalid`, `unsupported`.
  - Ensure DeepStream cannot start without a matching manifest and generated `deepstream.ini`.

- Modify: `novasight/model_registry/deepstream_config.py`
  - Keep `deepstream.ini` generation deterministic from manifest and engine path.
  - Ensure `output-tensor-meta=1` and `network-type=100` remain generated for tensor output.

- Modify: `novasight/api/routes_models.py`
  - Keep model preparation endpoints as the operator-facing way to confirm model semantics and generate `deepstream.ini`.
  - Do not make runtime guess class count, output layout, or parser semantics.

- Modify: `novasight/deepstream/pipeline_builder.py`
  - Keep DeepStream ROI crop and resize pipeline construction here.
  - Ensure center ROI can be represented by explicit `roi_left`, `roi_top`, `roi_size`.

- Modify: `novasight/deepstream/tensor_meta.py`
  - Add manifest/tensor validation before YOLO parsing.
  - Return or raise explicit errors for invalid tensor shape, missing dimensions, invalid timestamps, and invalid ROI geometry.

- Modify: `novasight/deepstream/backend.py`
  - Track correctness-oriented status fields: latest tensor count, latest batch count, latest error, postprocess timing, selected backend.
  - Continue publishing only `DetectionBatch`, not DeepStream-specific objects.

- Modify: `novasight/runtime/service.py`
  - Add a narrow method that accepts a verified `DetectionBatch` and converts it to `FrameContext`.
  - Keep tracker/control unaware of DeepStream.

- Modify: `tests/test_inference_runtime.py`
  - Add only critical DeepStream correctness tests.

- Modify: `tests/test_runtime_pipeline.py`
  - Add at most one seam test if needed.

## Task 0: Make Environment And Artifact Readiness Explicit

**Files:**
- Modify: `tests/test_inference_runtime.py`
- Modify: `novasight/model_registry/scanner.py` only if existing readiness behavior is incomplete.
- Modify: `novasight/model_registry/deepstream_config.py` only if generated config is incomplete.
- Modify: `novasight/api/routes_models.py` only if API readiness payload hides required state.

- [ ] **Step 1: Verify existing model readiness tests**

Run:

```bash
pytest tests/test_inference_runtime.py::test_model_scanner_reports_need_confirm_ready_and_invalid -q
```

Expected: pass.

- [ ] **Step 2: Confirm readiness statuses are operator-usable**

The model scan API/status must distinguish:

```text
ready        -> manifest exists, artifact hash matches, deepstream.ini exists or can be generated from matching manifest
need_confirm -> artifact exists, but manifest/model semantics are missing
invalid      -> manifest exists but does not match artifact or fingerprint
unsupported  -> artifact suffix or required metadata is unsupported
```

If these states already exist, do not add another test.

- [ ] **Step 3: Verify generated `deepstream.ini` contains tensor-output requirements**

Run the existing config generation test:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_config_is_generated_from_manifest -q
```

Expected generated config contains:

```text
[property]
model-engine-file=<absolute engine path>
batch-size=1
network-type=100
output-tensor-meta=1
output-blob-names=output0
```

- [ ] **Step 4: Fix only missing readiness behavior**

If readiness or config generation is incomplete, fix the smallest module that owns the behavior:

```text
artifact scan state        -> novasight/model_registry/scanner.py
deepstream.ini text        -> novasight/model_registry/deepstream_config.py
operator/API visibility    -> novasight/api/routes_models.py
```

Do not wire runtime DeepStream selection in this task.

- [ ] **Step 5: Run focused prerequisite verification**

Run:

```bash
pytest tests/test_inference_runtime.py::test_model_scanner_reports_need_confirm_ready_and_invalid tests/test_inference_runtime.py::test_deepstream_config_is_generated_from_manifest -q
```

Expected: pass.

## Task 1: Strengthen ROI Pipeline Correctness

**Files:**
- Modify: `tests/test_inference_runtime.py`
- Modify: `novasight/deepstream/pipeline_builder.py`

- [ ] **Step 1: Keep or update the existing ROI pipeline test**

Use the existing `test_deepstream_pipeline_builder_uses_nvmm_nvinfer_and_leaky_queues` as the only ROI pipeline unit test. It must assert the exact center ROI crop and model resize sequence.

Required assertions:

```python
assert "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true" in pipeline
assert "image/jpeg,width=1920,height=1080,framerate=120/1" in pipeline
assert "nvv4l2decoder mjpeg=1" in pipeline
assert "video/x-raw(memory:NVMM),format=I420" in pipeline
assert "nvvidconv left=720 right=1200 top=300 bottom=780" in pipeline
assert "video/x-raw(memory:NVMM),format=NV12,width=256,height=256" in pipeline
assert "nvinfer name=primary-infer" in pipeline
assert "videoconvert" not in pipeline
assert "video/x-raw,format=BGR" not in pipeline
```

- [ ] **Step 2: Run the ROI pipeline test**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_pipeline_builder_uses_nvmm_nvinfer_and_leaky_queues -q
```

Expected: pass.

- [ ] **Step 3: Fix only if the test fails**

If crop or resize order is wrong, update `build_deepstream_pipeline()` so the generated order is:

```text
nvv4l2decoder
-> video/x-raw(memory:NVMM)
-> nvvidconv left=<roi_left> right=<roi_right> top=<roi_top> bottom=<roi_bottom>
-> video/x-raw(memory:NVMM),format=NV12,width=<model_width>,height=<model_height>
-> nvstreammux
-> nvinfer
```

- [ ] **Step 4: Re-run the test**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_pipeline_builder_uses_nvmm_nvinfer_and_leaky_queues -q
```

Expected: pass.

## Task 2: Add Tensor Contract Validation

**Files:**
- Modify: `novasight/deepstream/tensor_meta.py`
- Modify: `tests/test_inference_runtime.py`

- [ ] **Step 1: Add one failing test for tensor shape mismatch**

Add this test near the existing DeepStream tensor tests:

```python
def test_deepstream_output_tensor_rejects_manifest_shape_mismatch(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    manifest = build_engine_manifest(
        model_id="combat",
        display_name="Combat",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )
    import numpy as np

    output = np.zeros((1, 7, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="tensor shape"):
        output_tensor_to_detection_batch(
            output,
            manifest=manifest,
            frame_id=12,
            capture_ts_ns=1000,
            inference_start_ts_ns=1200,
            inference_end_ts_ns=1800,
            roi_width=480,
            roi_height=480,
        )
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_output_tensor_rejects_manifest_shape_mismatch -q
```

Expected: fail until validation is added.

- [ ] **Step 3: Implement minimal validation**

In `novasight/deepstream/tensor_meta.py`, add a private validation helper:

```python
def _validate_output_tensor_contract(output: Any, manifest: ModelManifest) -> None:
    import numpy as np

    array = np.asarray(output)
    shape = [int(item) for item in array.shape]
    expected = [int(item) for item in manifest.output.shape]
    if not shape:
        raise ValueError("tensor shape is empty")
    if len(shape) != len(expected):
        raise ValueError(f"tensor shape {shape} does not match manifest output shape {expected}")
    for actual, wanted in zip(shape, expected):
        if actual != wanted:
            raise ValueError(f"tensor shape {shape} does not match manifest output shape {expected}")
```

Call it at the top of `output_tensor_to_detection_batch()` before `decode_nx6_detections()`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_output_tensor_rejects_manifest_shape_mismatch tests/test_inference_runtime.py::test_deepstream_output_tensor_reuses_shared_parser_and_returns_detection_batch -q
```

Expected: both pass.

## Task 3: Add One Golden Tensor Correctness Gate

**Files:**
- Modify: `tests/test_inference_runtime.py`
- Modify: `novasight/deepstream/tensor_meta.py` only if Task 2 reveals coordinate bugs.

- [ ] **Step 1: Keep the existing golden tensor test as the primary correctness test**

The existing `test_deepstream_output_tensor_reuses_shared_parser_and_returns_detection_batch` should remain the single golden tensor gate.

It must verify:

```python
assert batch.frame_id == 12
assert batch.capture_ts_ns == 1000
assert batch.coordinate_space == "roi"
assert batch.classes == ["0", "1", "2", "3"]
assert len(batch.detections) == 1
assert detection.cls == 0
assert detection.score == pytest.approx(0.9)
assert detection.x1 == pytest.approx(210.0)
assert detection.y1 == pytest.approx(180.0)
assert detection.x2 == pytest.approx(270.0)
assert detection.y2 == pytest.approx(300.0)
```

Do not add multiple variations unless a real bug appears.

- [ ] **Step 2: Run the golden test**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_output_tensor_reuses_shared_parser_and_returns_detection_batch -q
```

Expected: pass.

- [ ] **Step 3: Fix coordinate mapping only if needed**

If the test fails, keep the intended mapping:

```text
model 256x256 bbox
-> ROI 480x480 bbox
-> DetectionBatch(coordinate_space="roi")
```

Do not map to capture coordinates here. Capture/control/display mapping belongs downstream.

## Task 4: Add DeepStream Backend Correctness Status

**Files:**
- Modify: `novasight/deepstream/backend.py`
- Modify: `tests/test_inference_runtime.py`

- [ ] **Step 1: Add one status assertion to the existing backend publish test**

Extend `test_deepstream_backend_publishes_latest_detection_batch` with:

```python
status = backend.status()
assert status["selected"] == "deepstream"
assert status["published_batches"] == 1
assert status["last_frame_id"] == 12
assert status["last_error"] == ""
```

- [ ] **Step 2: Run the backend publish test**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_backend_publishes_latest_detection_batch -q
```

Expected: pass.

- [ ] **Step 3: Add minimal timing/status fields only if needed**

If status lacks correctness visibility, add fields without changing backend lifecycle:

```python
"selected": self.backend_id,
"running": running,
"published_batches": published,
"last_frame_id": last_result.frame_id if last_result else 0,
"last_error": self._last_error,
"pipeline": self.pipeline_description,
```

Do not add extensive telemetry in this task.

## Task 5: Add Runtime Seam For Verified DetectionBatch

**Files:**
- Modify: `novasight/runtime/service.py`
- Modify: `tests/test_runtime_pipeline.py`

- [ ] **Step 1: Add a narrow method to `RuntimeService`**

Add:

```python
def process_detection_batch(
    self,
    detection_batch: DetectionBatch,
    *,
    width: int,
    height: int,
) -> RuntimeFrameResult:
    if detection_batch.coordinate_space != "roi":
        self._clear_pending_commands("DETECTION_BATCH_COORDINATE_SPACE_INVALID")
        return self._empty_runtime_frame_result()
    context = FrameContext(
        frame_id=detection_batch.frame_id,
        width=int(width),
        height=int(height),
        detections=list(detection_batch.detections),
        classes=list(detection_batch.classes),
        capture_ts_ns=detection_batch.capture_ts_ns,
    )
    return self.update_control_observation(context)
```

This method is the seam between DeepStream and algorithms.

- [ ] **Step 2: Add one seam test**

Add one test that constructs a `DetectionBatch(coordinate_space="roi")`, passes it through `process_detection_batch()`, and asserts `RuntimeService.last_frame_context` receives the same frame id and detections. Keep this test small; do not test all tracking behavior here.

Example:

```python
def test_runtime_service_process_detection_batch_uses_roi_contract(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.control.trigger_mode = "always"
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: SimpleNamespace(
                executor_id="noop",
                sent=False,
                accepted=False,
                clipped=False,
                output_dx=0.0,
                output_dy=0.0,
                message="noop",
                metadata={},
            ),
        ),
    )
    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="roi",
    )

    service.process_detection_batch(batch, width=480, height=480)

    assert service.last_frame_context is not None
    assert service.last_frame_context.frame_id == 7
    assert service.last_frame_context.width == 480
    assert service.last_frame_context.height == 480
    assert len(service.last_frame_context.detections) == 1
```

- [ ] **Step 3: Run the seam test**

Run:

```bash
pytest tests/test_runtime_pipeline.py::test_runtime_service_process_detection_batch_uses_roi_contract -q
```

Expected: pass after method is added.

## Task 6: Reject Incorrect DetectionBatch Before Control

**Files:**
- Modify: `novasight/runtime/service.py`
- Modify: the same test file used in Task 5.

- [ ] **Step 1: Add one negative seam test**

Add only one negative test for wrong coordinate space:

```python
def test_runtime_service_rejects_non_roi_detection_batch(tmp_path) -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("invalid DetectionBatch must not execute"),
        ),
    )
    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="model",
    )

    result = service.process_detection_batch(batch, width=480, height=480)

    assert result.control_intents == []
    assert service.last_frame_context is None
```

- [ ] **Step 2: Run the two seam tests**

Run:

```bash
pytest tests/test_runtime_pipeline.py::test_runtime_service_process_detection_batch_uses_roi_contract tests/test_runtime_pipeline.py::test_runtime_service_rejects_non_roi_detection_batch -q
```

Expected: both pass.

## Task 7: Final Focused Verification

**Files:**
- No code changes unless verification exposes a bug.

- [ ] **Step 1: Run prerequisite readiness checks**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_config_is_generated_from_manifest -q
```

Expected: pass.

- [ ] **Step 2: Run the focused DeepStream correctness suite**

Run:

```bash
pytest tests/test_inference_runtime.py::test_deepstream_pipeline_builder_uses_nvmm_nvinfer_and_leaky_queues tests/test_inference_runtime.py::test_deepstream_output_tensor_reuses_shared_parser_and_returns_detection_batch tests/test_inference_runtime.py::test_deepstream_output_tensor_rejects_manifest_shape_mismatch tests/test_inference_runtime.py::test_deepstream_backend_publishes_latest_detection_batch -q
```

Expected: pass.

- [ ] **Step 3: Run runtime seam tests**

Run:

```bash
pytest tests/test_runtime_pipeline.py::test_runtime_service_process_detection_batch_uses_roi_contract tests/test_runtime_pipeline.py::test_runtime_service_rejects_non_roi_detection_batch -q
```

Expected: pass.

- [ ] **Step 4: Run existing related tests**

Run:

```bash
pytest tests/test_inference_runtime.py tests/test_runtime_pipeline.py -q
```

Expected: pass.

## Self-Review

Spec coverage:

- Environment and engine config readiness before runtime wiring: Task 0.
- DeepStream center ROI crop before resize: Task 1.
- GPU/NVMM path guarded at pipeline level: Task 1.
- `DetectionBatch` correctness before algorithm trust: Tasks 2, 3, 5, 6.
- Minimal tests only: Testing Policy and Tasks 1-7.
- Downstream does not depend on DeepStream types: Task 5.

Placeholder scan:

- No placeholder markers or unspecified "write tests" steps remain.

Type consistency:

- `DetectionBatch`, `Detection`, `FrameContext`, and `RuntimeFrameResult` names match existing project contracts.
- `process_detection_batch()` is introduced before tests depend on it.
