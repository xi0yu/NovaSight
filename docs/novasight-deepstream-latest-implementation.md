# NovaSight DeepStream Capture + TensorRT Latest Baseline

This document is the implementation baseline for the production latest-only
capture/inference/control path.

Current default remains the compatible control mainline:

```text
GStreamer capture
-> CPU latest-frame slot
-> CUDA upload/preprocess when TensorRT is available
-> TensorRT single-frame inference
-> DetectionBatch
-> control
```

The compatible path is not zero-copy, but it preserves the two required
properties for the active control mainline: latest-only frame selection and GPU
TensorRT inference.

The target path must not become the default until the native NVMM exchange,
TensorRT loop, overload behavior, timestamp validation, and control freshness
acceptance are complete.

## Target Pipeline

```text
v4l2src
-> image/jpeg caps
-> jpegparse
-> nvv4l2decoder
-> nvvideoconvert / nvvidconv
-> video/x-raw(memory:NVMM)
-> GPU/VIC ROI crop
-> GPU/VIC resize
-> basic color conversion
-> LatestFrameExchange capacity=1
-> NovaSight TensorRT InferenceLoop
-> CUDA preprocess
-> TensorRT enqueueV3
-> CUDA postprocess / NMS
-> result timestamp validation
-> DetectionBatch latest mailbox
-> Tracker / Selector / Kalman
-> Angular control
-> replace-only Scheduler
```

DeepStream/GStreamer owns capture, NVIDIA hardware decode, NVMM memory
management, ROI crop, resize, basic colorspace conversion, and frame identity.

DeepStream/GStreamer does not own inference scheduling, TensorMeta output,
DeepStream tracker, inference queues, or control scheduling. The formal target
path does not require:

```text
nvstreammux -> nvinfer -> TensorMeta
```

## Admission Contract

The exchange is a latest-only frame handoff, not a queue.

Required invariants:

```text
pending_depth <= 1
inference_loop_max_inflight == 1
```

Operational rules:

- The capture pipeline publishes only ROI-ready frames into `LatestFrameExchange`.
- Publishing a new frame replaces the pending frame and releases the old
  resource.
- TensorRT inference is a single loop that pulls
  `acquire_latest_after(last_generation)` when ready.
- TensorRT does not consume ordinary FIFO input.
- Result timestamps are validated before a `DetectionBatch` can enter control.

## Native Module

The first production code is in:

```text
native/latest-frame-exchange/
```

The committed native layer contains a portable exchange state machine:

- `LatestFrameExchange`
- `FrameDescriptor`
- `FrameHandle`
- `ExchangeStatus`

It is intentionally resource-agnostic so replacement, acquire-after-generation,
capacity-one, and inference-loop jump behavior can be verified on development
machines. Jetson integration should wrap real `GstBuffer` / `NvBufSurface`
handles and publish them after ROI/resize/color conversion.

Build and test:

```bash
cmake -S native/latest-frame-exchange -B build/latest-frame-exchange
cmake --build build/latest-frame-exchange
ctest --test-dir build/latest-frame-exchange --output-on-failure
```

## Backend Naming

Long-term configuration should separate pipeline orchestration from inference
execution:

```text
pipeline.backend:
  legacy_latest
  deepstream_capture_latest
  deepstream_uncontrolled
  native_tensorrt_latest

inference.backend:
  onnxruntime
  native_tensorrt
```

The current repo still carries legacy names for compatibility:

- `tensorrt`: current default compatible GPU inference path.
- `nvmm_latest`: legacy name for the explicit target/experimental NVMM exchange
  path requiring native GPU support.
- `deepstream`: Full DeepStream push mode, experimental only.

Do not silently map one mode to another.

## Implementation Order

1. Preserve current `legacy_latest` and Full DeepStream push baselines.
2. Add frame identity and monotonic timestamp fields across probes.
3. Finish the native `LatestFrameExchange` and resource lifetime integration.
4. Publish ROI-ready NVMM frames into the exchange from the capture pipeline.
5. Implement the single-inflight TensorRT `InferenceLoop` that pulls latest.
6. Add CUDA preprocess, TensorRT `enqueueV3`, CUDA postprocess/NMS, and result
   timestamp validation.
7. Keep DetectionBatch and control delivery latest-only.
8. Run overload and fault acceptance before enabling the target path.

## Non-Negotiable Constraints

- Do not delete `legacy_latest`.
- Do not set Full DeepStream push mode as default.
- Do not use `nvinfer interval` or TensorMeta as the production scheduling path.
- Do not add ordinary FIFOs to the control mainline.
- Do not put heavy logic in Python probes.
- Do not directly subtract PTS from monotonic timestamps.
- Do not loosen stale thresholds to hide queueing.
- Do not mix scheduling, preprocessing numerics, and control algorithm changes
  in one commit.
