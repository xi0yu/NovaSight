# NovaSight DeepStream Latest Implementation Baseline

This document is the implementation baseline for the future production
`deepstream_latest` path.

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

`deepstream_latest` must not become the default until the native admission gate
and ACK path pass overload, timeout, and control freshness acceptance.

## Target Pipeline

```text
v4l2src
-> image/jpeg caps
-> jpegparse
-> nslatestgate
-> nvv4l2decoder
-> nvvideoconvert / nvvidconv
-> video/x-raw(memory:NVMM)
-> nvstreammux batch-size=1
-> nvdspreprocess or nvinfer internal preprocess
-> nvinfer interval=0
-> nsinferack
-> tensor postprocess
-> DetectionBatch latest mailbox
-> Tracker / Selector / Kalman
-> Angular control
-> replace-only Scheduler
```

For MJPEG input, `nslatestgate` is placed after `jpegparse` and before
`nvv4l2decoder`. The gate stores compressed independent JPEG buffers, so old
frames can be discarded before NVDEC/NVMM work is spent.

## Admission Contract

The gate is a single-credit latest admission controller, not a queue.

Required invariants:

```text
pending_depth <= 1
inflight_depth <= 1
ack_mismatch_total == 0
unexpected_ack_total == 0
```

Operational rules:

- The sink path only records a frame token and atomically replaces pending.
- The dispatch task may push one frame only when inference credit is available.
- `gst_pad_push()` return is not an inference completion signal.
- Credit is restored only by `nsinferack` after `nvinfer` output is observed.
- ACK timeout is fail-closed: stop control, clear pending, and rebuild the
  pipeline instead of releasing a second credit.
- FLUSH uses a new `session_epoch` so stale ACKs cannot unlock the new session.

## Native Module

The first production code is in:

```text
native/gst-novasight-latest/
```

The committed layer is a GStreamer-free core state machine:

- `NsLatestGateCore`
- `NsFrameToken`
- `NsGateMetrics`

It is intentionally independent of GStreamer so replacement, single-credit,
ACK, timeout, flush, EOS, and epoch behavior can be verified on development
machines before wrapping the core as `nslatestgate` and `nsinferack`.

Build and test:

```bash
cmake -S native/gst-novasight-latest -B build/ns-latest
cmake --build build/ns-latest
ctest --test-dir build/ns-latest --output-on-failure
```

## Backend Naming

Long-term configuration should separate pipeline orchestration from inference
execution:

```text
pipeline.backend:
  legacy_latest
  deepstream_latest
  deepstream_uncontrolled
  native_tensorrt_latest

inference.backend:
  onnxruntime
  native_tensorrt
  nvinfer
```

The current repo still carries legacy names for compatibility:

- `tensorrt`: current default compatible GPU inference path.
- `nvmm_latest`: explicit target/experimental path requiring native GPU support.
- `deepstream`: Full DeepStream push mode, experimental only.

Do not silently map one mode to another.

## Implementation Order

1. Preserve current `legacy_latest` and Full DeepStream push baselines.
2. Add frame identity and monotonic timestamp fields across probes.
3. Finish `nslatestgate` and `nsinferack` GStreamer elements around the tested
   core.
4. Validate MJPEG pre-decode gate before attaching ROI/mux/infer.
5. Attach NVMM ROI and `nvstreammux batch-size=1`.
6. Attach `nvinfer` and release credit only from `nsinferack`.
7. Keep DetectionBatch and control delivery latest-only.
8. Run overload and ACK-fault acceptance before enabling `deepstream_latest`.

## Non-Negotiable Constraints

- Do not delete `legacy_latest`.
- Do not set Full DeepStream push mode as default.
- Do not use `nvinfer interval` as a replacement for latest admission.
- Do not add ordinary FIFOs to the control mainline.
- Do not put heavy logic in Python probes.
- Do not directly subtract PTS from monotonic timestamps.
- Do not loosen stale thresholds to hide queueing.
- Do not mix scheduling, preprocessing numerics, and control algorithm changes
  in one commit.
