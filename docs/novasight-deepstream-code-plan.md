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
The current DeepStream fast path is deliberately scoped to MJPEG capture (`capture.pixel_format` empty, `MJPG`, or `MJPEG`) because the production target is `/dev/video0` MJPEG 1920×1080@120. Other capture formats must be added as explicit DeepStream pipeline variants instead of silently reusing the MJPEG pipeline.
This constraint is owned by `DeepStreamPipelineConfig` / `build_deepstream_pipeline()`, so API, runtime, CLI smoke, and tests share the same MJPEG contract.

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
  - `ready`: manifest exists and matches artifact. For TensorRT `.engine`, `deepstream.ini`
    must also match the manifest and point `model-engine-file` at the same engine.
  - `need_confirm`: artifact can be fingerprinted, but manifest or DeepStream config is missing.
  - `invalid`: manifest exists but does not match artifact.
  - `unsupported`: file suffix or required shape/parser information is not supported.
- Generate `model.manifest.json` and `deepstream.ini` only after model semantics are confirmed.
- `/api/models/scan` returns artifact configuration status. `GET` is supported for the no-parameter scan path; `POST` remains compatible.
  The same scan also syncs registry artifact status for the model page: `ready` only for a complete DeepStream artifact, `pending` for `need_confirm`, and `failed` for invalid artifacts.
- Uploading or rescanning a model refreshes the registry checksum for an existing artifact path. Uploaded TensorRT `.engine` files stay `pending` until DeepStream manifest/config preparation proves the artifact is runnable.
- `/api/models/artifacts/{artifact_id}/deepstream/prepare` writes confirmed manifest and `deepstream.ini`, then promotes the registry artifact from `pending` to `ready` only if the generated DeepStream artifact validates.
- `/api/models/artifacts/{artifact_id}/deepstream/pipeline` builds the concrete NVMM/nvinfer pipeline from the confirmed manifest and ROI/capture parameters.
  The pipeline preview endpoint accepts `pixel_format`; the current DeepStream pipeline rejects non-MJPEG values instead of returning a misleading MJPEG pipeline.
- The model page exposes `准备 DeepStream` for pending TensorRT `.engine` artifacts.
  It derives the NCHW input shape from the selected version, derives YOLO candidate count from strides 8/16/32, calls `/api/models/artifacts/{artifact_id}/deepstream/prepare`, and refreshes the artifact list so a validated artifact becomes publishable.
- Publishing a DeepStream `.engine` updates the manifest/config binding and clears the runtime pipeline. It is a prepared state, not a legacy `loaded=true` state; `nvinfer` loads the engine when the runtime is started.

Next:

- Add a more explicit advanced confirmation view if non-YOLO output contracts must be supported.
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
- `/api/runtime/start` can now create a `RuntimePipeline` backed by `DeepStreamDetectionBackend` when `inference.backend=deepstream`.
  This is still opt-in; the legacy capture/inference path remains available.
- When `inference.backend=deepstream`, `/api/capture/select` only selects and persists the capture profile from device capabilities.
  It must not start the legacy `CaptureService` appsink/cv2 session, because the DeepStream runtime owns `/dev/video0` and opens the real NVMM/nvinfer pipeline on `/api/runtime/start`.
- Runtime, model, capture, and source changes clear stale DeepStream pipeline objects instead of reusing an old GStreamer/nvinfer instance:
  - model publish/rollback;
  - DeepStream runtime config changes;
  - capture profile changes;
  - switching to an image source;
  - capture stop;
  - runtime stop.
  The next start rebuilds the pipeline from the current manifest, `deepstream.ini`, capture profile, and ROI.
- `novasight doctor deepstream-smoke` is the Jetson-side smoke gate for the generated `model.manifest.json` + `deepstream.ini` pair.
- The backend remains opt-in and is not the default runtime path.

Jetson smoke command:

```bash
novasight doctor deepstream-smoke \
  --manifest data/models/<model>/<version>/model.manifest.json \
  --nvinfer-config data/models/<model>/<version>/deepstream.ini \
  --device /dev/video0 \
  --capture-width 1920 \
  --capture-height 1080 \
  --fps 120 \
  --roi-left 720 \
  --roi-top 300 \
  --roi-size 480 \
  --io-mode 2 \
  --batched-push-timeout-us 0 \
  --seconds 10 \
  --report-json /tmp/novasight-deepstream-smoke.json
```

The command must fail loudly when `gi/Gst` or `pyds` is missing. It must not fall back to CPU capture.
It also prints the exact `engine`, `model_fingerprint`, `nvinfer_config_fingerprint`,
`model_input`, `model_output`, `io_mode`, and `batched_push_timeout_us` before starting
the pipeline, so a Jetson run can verify that the selected engine, manifest, generated
`deepstream.ini`, parser contract, and GStreamer runtime knobs are the same artifact pair.
When `--report-json` is provided, the same evidence is written to a structured report so Jetson
runs can be compared across models, ROI sizes, and GStreamer runtime knobs without relying on
terminal scrollback.
The structured report can be checked later with:

```bash
novasight doctor deepstream-smoke-report \
  --report-json /tmp/novasight-deepstream-smoke.json \
  --min-tensor-meta-fps 115 \
  --min-postprocess-fps 115 \
  --min-detection-batch-fps 115 \
  --max-frame-age-ms 20
```

This report check rejects incomplete or unsafe evidence, including missing tensor-meta counts,
missing model/config fingerprints, failed smoke result, unavailable DeepStream dependencies,
non-ROI `DetectionBatch` coordinates, non-`capture_to_tensor_meta_done` latency source, or
non-`gst_clock_base_time_pts` timestamp source. The optional threshold flags are the Jetson
measurement gate for deciding whether the DeepStream path is fast enough to feed control.
The smoke command and runtime startup also verify that `deepstream.ini` `model-engine-file`
resolves to the current manifest artifact engine. A matching config fingerprint alone is not
enough, because a stale config can otherwise point `nvinfer` at an older `.engine` after a
model switch.
Use `--io-mode` and `--batched-push-timeout-us` for explicit A/B runs; the defaults keep the
current low-latency path (`io-mode=2`, `batched-push-timeout=0`).
Runtime startup rejects non-MJPEG `capture.pixel_format` values while this backend uses the MJPEG hardware-decode pipeline.
DeepStream backend latency status is reported as `capture_to_tensor_meta_done`: from the frame timestamp to the nvinfer src-pad tensor-meta probe. It is not pure TensorRT kernel time.
Runtime `DetectionBatch` handling preserves this as `capture_to_tensor_meta_ms` in inference status and pipeline timings, so dashboards do not have to infer it from `engine_ms`.
For `DetectionBatch` sources, parser/NMS has already happened before runtime handoff; runtime therefore reports `stage_postprocess_ms=0` and exposes handoff wait separately as `stage_handoff_ms`.
`novasight doctor deepstream-smoke` prints both `latency_source` and `capture_to_tensor_meta_ms` for the same reason.
Runtime confidence and NMS thresholds override manifest defaults for the DeepStream Python tensor-meta parser. These thresholds are hot-updated on the live `DeepStreamDetectionBackend`; changing them must not rebuild the GStreamer/nvinfer pipeline because they only affect Python postprocess after tensor meta is produced.
Invalid `DetectionBatch` payloads are rejected before Tracker/Selector/Kalman/Controller. The runtime rejects non-ROI coordinate spaces, non-finite bbox values, scores outside `[0,1]`, non-positive boxes, and ROI-space boxes whose center falls outside the ROI. Rejection resets the control observation state so continuous control cannot keep driving from a stale target.
`DeepStreamDetectionBackend` only publishes and exposes `DetectionBatch` objects while the source is running and not in a terminal error state. Late tensor probes after stop, EOS, or a bus error must not update `last_result`, `published_batches`, or the control input stream.
DeepStream timestamp health is explicit. Backend status, runtime statistics, Studio, and `novasight doctor deepstream-smoke` expose `timestamp_source`; Jetson smoke acceptance now requires `gst_clock_base_time_pts`. `first_probe_offset_pts` and `observed_probe_time_invalid_pts` are fallback modes and must fail the smoke/report gate because frame age is not trustworthy enough for Kalman prediction and latency compensation.

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
timestamp_source
frame_age_ms P50/P95/P99
capture_to_infer_done_ms P50/P95/P99
CPU / GR3D / VIC / NVDEC / RSS / temperature
```

Acceptance target for continuing:

```text
timestamp_source == gst_clock_base_time_pts
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

## Current Immediate Slice

The current engineering slice is no longer basic model scanning; it is Jetson validation and lifecycle hardening:

- Keep the generated DeepStream pipeline opt-in and reproducible from the model manifest plus runtime config.
- Keep stale pipeline cleanup explicit across model switch, capture switch, source switch, runtime stop, and failed start paths.
- Use `novasight doctor deepstream-smoke` on Jetson to verify tensor-meta FPS, DetectionBatch FPS, frame age, and parser correctness before enabling control decisions from DeepStream output.
- Do not change Tracker, TargetSelector, Kalman, Controller, or kmNet defaults until a Jetson run proves `DetectionBatch(coordinate_space="roi")` is correct and fresh.
