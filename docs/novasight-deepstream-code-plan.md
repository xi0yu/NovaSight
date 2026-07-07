# NovaSight DeepStream/NVMM Code Plan

## Goal

Move NovaSight from the current CPU appsink fallback path toward a production-capable Jetson path:

```text
v4l2src
→ jpegparse
→ nvv4l2decoder
→ NVMM
→ nvvidconv ROI/resize
→ nvstreammux
→ nvinfer
→ tensor meta
→ shared YOLO parser
→ DetectionBatch
```

The legacy backend remains available until Jetson A/B evidence proves the DeepStream backend is stable.

## Product Deployment Shape

NovaSight is split into a management host tier and a Jetson execution tier:

```text
Windows / Android Tablet / Android WebView / Linux Web / macOS Host
├── NovaSight Studio
├── model management
├── parameter configuration
├── logs and performance analysis
├── model training or ONNX export
└── remote Jetson management
          │
          │ WebSocket / REST / gRPC
          ▼
Jetson Orin Nano Super
├── JetPack / Ubuntu
├── GStreamer / DeepStream
├── NVMM zero-copy capture
├── CUDA
├── TensorRT
├── model inference
├── tracking and control algorithms
└── HID / KMBOX output
```

This split is a core product seam:

- The host tier owns user experience, model inventory, model confirmation, config editing, logs, dashboards, and remote commands.
- The Jetson tier owns the real-time loop: capture, decode, ROI/resize, inference, postprocess, tracking, target selection, Kalman/aim generation, controller, and HID output.
- The host tier must not become part of the real-time critical path. If the browser, tablet, or remote UI stalls, Jetson capture/inference/control must keep running.
- The network protocol should carry commands, configuration revisions, telemetry snapshots, logs, and model assets. It should not carry full-rate raw video unless explicitly in preview/debug mode.
- The Jetson runtime should expose a small interface: load/activate model, apply config revision, start/stop pipeline, stream telemetry, stream preview, and report health.
- Model training/export can happen on the host side, but the Jetson only runs confirmed artifacts with a persistent manifest and generated runtime config.

## Current Confirmation

Current default capture is `capture.memory=cpu`, which opens `GstAppSinkFrameSource`.
That path maps `Gst.Buffer` into Python, builds a NumPy BGR image, then preprocesses on CPU before TensorRT.

The desired fast path is not merely “appsink”. It is a DeepStream path where large image data stays in NVMM/GPU-side resources and Python only parses the small `output0` tensor.

## Non-Negotiable Invariants

- Do not delete the legacy CPU backend.
- Do not set DeepStream as default before Jetson evidence.
- Do not duplicate YOLO parser logic.
- Do not control HID/kmNet until DeepStream emits correct `DetectionBatch`.
- Do not trust `fpsdisplaysink` alone; count tensor meta, parsed detections, DetectionBatch, and frame age.
- Do not guess irreversible model semantics on every startup; write a persistent manifest after confirmation.

## Phase P0: Model Configuration Foundation

Completed:

- Shared parser at `novasight/inference/postprocess/yolo.py`.
- `ModelManifest` and deterministic model fingerprint.
- DeepStream `nvinfer` config text generator.
- Scan `data/models` for `.engine` and `.onnx`.
- Return per-artifact configuration status:
  - `ready`: manifest exists and matches artifact.
  - `need_confirm`: artifact can be fingerprinted, but manifest is missing.
  - `invalid`: manifest exists but does not match artifact.
  - `unsupported`: file suffix or required shape/parser information is not supported.
- Generate `model.manifest.json` and `deepstream.ini` only after model semantics are confirmed.
- `/api/models/scan` returns artifact configuration status. `GET` is supported for the no-parameter scan path; `POST` remains compatible.
- `/api/models/artifacts/{artifact_id}/deepstream/prepare` writes confirmed manifest and `deepstream.ini`.
- `/api/models/artifacts/{artifact_id}/deepstream/pipeline` builds the concrete NVMM/nvinfer pipeline from the confirmed manifest and ROI/capture parameters.

Next:

- Wire the frontend model confirmation flow to `scan`, `prepare`, and `pipeline`.
- On Jetson, verify generated `deepstream.ini` and pipeline with the real `.engine`.

## Phase P1: DeepStream Probe Backend

Build an experimental backend that produces `DetectionBatch` only:

```text
nvinfer tensor meta
→ extract output0
→ reshape to manifest output shape
→ decode_nx6_detections()
→ map model coords to ROI coords
→ DetectionBatch(coordinate_space="roi")
```

This phase does not touch Tracker, Kalman, Controller, or kmNet.

Current implementation:

- `novasight/deepstream/pipeline_builder.py` generates the target MJPEG hardware decode → NVMM → `nvvidconv` ROI/resize → `nvstreammux` → `nvinfer` pipeline.
- `novasight/deepstream/tensor_meta.py` converts an `output0` tensor into `DetectionBatch` through the shared YOLO parser.
- `novasight/deepstream/backend.py` adds an isolated `DeepStreamDetectionBackend` with dependency detection, start/stop lifecycle, latest-result semantics, and a probe publishing entry point.
- The backend remains opt-in and is not wired as the default runtime path.

Next:

- Validate `pyds` tensor metadata extraction on Jetson.
- Confirm DeepStream tensor layer shape and dtype match `model.manifest.json`.
- Count tensor meta FPS and DetectionBatch FPS before touching control.

## Phase P2: Jetson Measurement Gate

Measure and report:

```text
buffer_fps
tensor_meta_fps
postprocess_fps
detection_batch_fps
frame_age_ms P50/P95/P99
capture_to_infer_done_ms P50/P95/P99
CPU / GR3D / VIC / NVDEC / RSS / temperature
```

Acceptance target for continuing:

```text
tensor_meta_fps >= 115
detection_batch_fps >= 115
frame_age_p95 <= 20ms
postprocess_p95 <= 2ms
```

## Phase P3: Main Runtime Integration

Once DetectionBatch is correct and fast:

```text
DeepStream DetectionBatch
→ Tracker
→ TargetSelector
→ Kalman
→ AimPoint
→ Controller
```

The downstream runtime must not know whether the detection source is legacy TensorRT or DeepStream.

## Phase P4: Safe Model Switch

Model switching is lifecycle-managed:

```text
REQUEST_SWITCH
→ stop/pause pipeline
→ flush old tensor meta
→ clear tracker/kalman/target lock
→ load manifest + generated nvinfer config
→ restart pipeline
→ wait first tensor meta
→ mark active
```

No in-place `nvinfer` engine replacement until this safe path is proven.

## Immediate Implementation Slice

Implement the P0 model scan status module:

- `novasight/model_registry/scanner.py`
- Existing tests in `tests/test_inference_runtime.py`
- Optional API wiring after module behavior is stable

This slice creates product-visible truth before launching DeepStream: the app can tell whether each model is configured, needs confirmation, or is invalid.
