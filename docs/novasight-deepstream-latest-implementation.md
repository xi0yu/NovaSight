# NovaSight Capture + TensorRT Latest Baseline

> The `nvstreammux -> nvinfer -> C++ parser -> NvDsObjectMeta` implementation
> now lives in `docs/novasight-deepstream-object-mainline.md`. The older
> application-scheduled TensorRT design below remains architectural history and
> must not be mistaken for the currently selectable DeepStream backend.

This document is the implementation baseline for the production latest-only
capture/inference/control path.

NovaSight currently has three distinct paths:

- Current stable temporary mainline: compatible latest-only capture plus GPU
  TensorRT inference.
- Final GPU mainline: NVIDIA GStreamer / DeepStream data plane, NVMM ROI,
  GPU latest exchange, CUDA preprocess, and NovaSight-controlled TensorRT.
- Legacy Full DeepStream push path: `nvstreammux -> nvinfer -> TensorMeta`,
  retained only for experiments and A/B diagnostics.

## Current Stable Temporary Mainline

The default running path remains the compatible control mainline:

```text
GC553G2 / V4L2
-> GStreamer capture/decode
-> upstream leaky queue, capacity=1, drop old
-> appsink, internal capacity=1, drop old, sync=false
-> immediate callback handoff
-> CPU FrameHandle
-> CPU LatestFrameExchange, one pending_latest
-> TensorRT InferenceLoop pulls latest when idle
-> CPU preprocess host tensor
-> CUDA H2D upload
-> TensorRT single-frame inference
-> output sync
-> decode / NMS
-> DetectionBatch
-> freshness validation
-> control chain
```

The compatible path has a CPU-to-GPU upload, but it preserves the two required
properties for the active control mainline: latest-only frame selection and GPU
TensorRT inference.

The preferred CPU-readable appsink format is `BGRx`. The candidate list also
keeps `RGBA`, `RGB`, `BGR`, `NV12`, and `I420` fallbacks available for Jetson
images where one caps negotiation path is unstable. Host preprocessing converts
these formats to RGB/NCHW, normalizes to FP16/FP32, and uses letterbox padding
instead of stretching when the input and model aspect ratios differ.
TensorRT host input/output buffers prefer CUDA pinned host memory when the
runtime exposes an alloc/free pair, then fall back to ordinary NumPy buffers if
pinned allocation is unavailable.

Temporary-mainline acceptance focuses on freshness behavior, not absolute
minimum latency:

- frame age does not grow with runtime;
- there is no unbounded queue;
- the same `frame_id` is not inferred repeatedly;
- stale `DetectionBatch` objects never enter control;
- TensorRT only processes the newest available frame when the inference loop is
  idle.

## Final GPU Mainline

```text
GC553G2 / V4L2
-> NVIDIA GStreamer / DeepStream data plane
-> v4l2src
-> image/jpeg caps
-> jpegparse
-> nvv4l2decoder
-> NVMM NvBufSurface
-> nvvideoconvert / nvvidconv
-> center ROI, resize, basic video format normalization
-> video/x-raw(memory:NVMM)
-> upstream leaky boundary, capacity=1
-> GPU FrameHandle holding a strong GstBuffer reference
-> LatestFrameExchange, one pending_latest
-> TensorRT InferenceLoop pulls the newest generation when idle
-> CUDA preprocess: NVMM surface to model tensor
-> TensorRT enqueueV3
-> output completion event / stream synchronization
-> decode / NMS
-> result timestamp validation
-> DetectionBatch latest mailbox
-> Tracker / Selector / Kalman
-> Angular control
-> replace-only Scheduler
```

The final GPU path must not become the default until the native NVMM exchange,
TensorRT loop, overload behavior, timestamp validation, and control freshness
acceptance are complete.

The final GPU mainline uses the NVIDIA GStreamer / DeepStream GPU data plane,
but it does not adopt DeepStream's automatic per-frame inference scheduling.

NVIDIA GStreamer / DeepStream provides:

- source `GstBuffer` media timing such as PTS/DTS when present;
- buffer offset or source metadata when present;
- video format and dimensions;
- NVMM / `NvBufSurface`;
- hardware decode;
- ROI crop, resize, and basic video colorspace conversion.

NovaSight assigns at the application boundary:

- monotonic `frame_id`;
- `ingress_monotonic_ns`;
- `pipeline_epoch`;
- `source_sequence`.

Use `ingress_monotonic_ns` as the freshness and end-to-end latency baseline.
Do not treat GstBuffer PTS/DTS as NovaSight's absolute monotonic time. PTS may
be missing, invalid, non-monotonic, or in a media-clock domain that still needs
explicit mapping.

DeepStream/GStreamer does not own inference scheduling, TensorMeta output,
DeepStream tracker, inference queues, or control scheduling. The formal GPU
mainline does not require:

```text
nvstreammux -> nvinfer -> TensorMeta
```

## Legacy Full DeepStream Experimental Path

The old push-mode path remains available only for comparison and diagnostics:

```text
capture
-> decoder
-> ROI
-> nvstreammux
-> nvinfer
-> TensorMeta / Python postprocess
```

This path is not the default control mainline because `nvinfer` push scheduling
does not by itself prove latest-only frame admission. It may be used for A/B
latency evidence, model parser checks, and NVIDIA component diagnostics.

## Admission Contract

Latest-only is not only a NovaSight thread-exchange strategy. It must hold from
the GStreamer sink boundary all the way into TensorRT.

Required invariants:

```text
upstream_queue_depth <= 1
appsink_or_custom_sink_depth <= 1
pending_depth <= 1
inference_loop_max_inflight == 1
```

Operational rules:

- Every realtime GStreamer queue must have explicit capacity and old-frame drop
  behavior, for example `queue max-size-buffers=1 leaky=downstream`.
- appsink-style sinks must not accumulate samples; use the platform-supported
  equivalent of `max-buffers=1 drop=true sync=false`.
- Sink callbacks must immediately hand off the sample/frame and return. They
  must not run inference, postprocess, JSON, WebSocket sends, file writes, or
  heavy logging.
- The capture pipeline publishes only ROI-ready frames into
  `LatestFrameExchange`.
- Publishing a new frame replaces the pending frame and releases the old
  resource.
- TensorRT inference is a single loop that pulls
  `acquire_latest_after(last_generation)` when ready.
- TensorRT does not consume ordinary FIFO input.
- The exchange carries ROI-ready video surfaces, not TensorRT input tensors.
- CUDA preprocess must convert NVMM image surfaces into model input device
  buffers.
- `enqueueV3()` submission is not a readable `DetectionBatch`. Correct CUDA
  event or stream synchronization must prove output completion before decode/NMS
  reads model outputs.
- Result timestamps are validated before a `DetectionBatch` can enter control.

Ordinary FIFO creates historical task debt. With sustained capture faster than
inference, an unbounded FIFO causes result age to keep increasing. A fixed-size
FIFO that does not drop old frames can stabilize at a large fixed age. A
latest-only boundary avoids debt by skipping old frames:

```text
capture: 100 101 102 103 104 105 106 107 108
infer:   100 -------- 104 -------- 108
```

The exact frame numbers are not fixed. The only rule is that each idle
inference iteration takes the newest frame that has already arrived.

## Logical Double Buffering

`LatestFrameExchange` is logically a two-position boundary, but only one
position lives inside the exchange:

```text
current frame: owned by TensorRT InferenceLoop while inference is running
pending frame: owned by LatestFrameExchange and always replaceable
```

This is the correct source-frame model for `GstBuffer` / `NvBufSurface`
resources. The inference loop owns the current `FrameHandle`; the exchange owns
at most one pending latest `FrameHandle`; capture can keep replacing pending
without waiting for current inference to finish. The implementation must release
any replaced pending resource outside the exchange lock.

For Jetson integration, `FrameHandle` must hold a strong `GstBuffer` reference,
not only a raw `NvBufSurface*`. The native consumer may derive the current
`NvBufSurface` / CUDA access descriptor from the live buffer while consuming the
handle. Do not keep a temporary pointer obtained from a map operation after the
corresponding unmap unless that exact access pattern is proven valid for the
Jetson, DeepStream, and memory type in use.

Holding `GstBuffer` references also affects buffer pools. `LatestFrameExchange`
capacity 1 does not mean the NVMM buffer pool capacity is 1. The lower-level
pool must have enough bounded surfaces for decode, convert, sink delivery,
pending exchange ownership, and current inference ownership. Increasing pool
size must not be used to hide accumulated stale work.

Do not implement this as blind fixed A/B alternation. A waiting frame is valid
only when its `generation` is newer than the last consumed generation. If no
new frame arrived during the previous inference, the runtime must not re-infer
an older slot simply because the slot index toggled.

Physical A/B ping-pong is still useful one layer later, after CUDA preprocess
has produced NovaSight-owned TensorRT input buffers:

```text
LatestFrameExchange source frame
-> CUDA preprocess
-> TensorRT input slot A / TensorRT input slot B
-> TensorRT enqueueV3
```

Those tensor slots require an explicit state machine, for example `Empty`,
`Writing`, `Ready`, and `Inferencing`, with CUDA events or stream dependencies
between preprocess and TensorRT. A single atomic active-slot index is not a
valid synchronization model for asynchronous GPU work.

## NVMM Surface Is Not Tensor Input

NVMM output from DeepStream/GStreamer is normally a video surface, commonly
NV12, RGBA, or RGB. TensorRT models usually require a fixed-shape tensor such
as NCHW RGB FP16/FP32/INT8 with normalization and sometimes letterbox behavior.

The production path therefore includes a GPU-side preprocess step:

```text
NvBufSurface video frame
-> CUDA-accessible image data
-> RGB / layout conversion
-> HWC to CHW
-> uint8 to FP16 / FP32 / INT8
-> normalization and optional letterbox
-> TensorRT input device buffer
```

This may write a new GPU tensor buffer. That is expected. The goal is not a
literal no-write path from NVMM surface to TensorRT binding; the goal is to
avoid GPU -> CPU -> GPU crossings. GPU surface -> GPU tensor is the correct
low-latency boundary.

`NvBufSurface` mapping, CUDA/EGL access, synchronization, and lifetime must be
managed explicitly. Do not mark an NVMM video surface as a TensorRT-ready input
tensor unless CUDA preprocess has actually produced the model input buffer.

Responsibility split:

- `nvvideoconvert` / VIC: ROI crop, resize, and basic video format preparation
  such as NV12/RGBA where supported.
- NovaSight CUDA preprocess: letterbox, RGB channel order, HWC-to-CHW, FP16 /
  FP32 / INT8 conversion, normalization, and TensorRT binding writes.

Do not duplicate model-specific tensor work in both stages.

## Native Module

The first production code is in:

```text
native/latest-frame-exchange/
```

The Rust migration now owns the live resource seam directly:

```text
novasight-platform-jetson::deepstream::FrameLease
-> LatestFrameExchange (capacity one)
-> CudaFramePreprocessor
-> novasight-jetson-preprocess::DeviceTensor
```

`FrameLease` holds a strong `GstBuffer` reference and is cleared on runtime
stop after its sole publisher joins. `CudaFramePreprocessor` borrows that lease
for the complete synchronous native map/transform/kernel interval. The
`novasight-jetson-preprocess` module hides the legacy JSON C ABI, validates the
native readiness receipt and the exact RGB NCHW shape/dtype/nbytes result, and
owns `novasight_release_tensor` through `DeviceTensor` RAII. The linked adapter
is enabled only with the explicit Rust `cuda-preprocess` feature and expects
`libnovasight_preprocess.so` from `scripts/build_jetson_preprocess.sh`.

This integration reuses the existing production Jetson CUDA implementation in
`novasight_jetson_preprocess_native`; it does not introduce a second CUDA
kernel. The native ABI accepts both the legacy Python `appsink` source and the
Rust `deepstream_pad` source while keeping the strongly-held buffer alive.
RuntimeSupervisor consumption, TensorRT context ownership, and decode/NMS are
still separate later steps. Until all three are connected and verified on the
target Jetson, model probe `input_mode=latest` remains fail-closed with 409.

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
capture.backend:
  gst_cpu_latest
  nvmm_latest

preprocess.backend:
  cpu
  cuda

inference.backend:
  tensorrt
  nvmm_latest
```

The current repo still carries legacy names for compatibility:

- `gst_cpu_latest`: current default compatible capture bridge. It uses
  GStreamer/NVIDIA decode and `nvvidconv` ROI/resize, then appsink system
  memory plus CPU host preprocessing before TensorRT CUDA upload.
  `novasight doctor gst-cpu-latest-smoke` is the 60-second acceptance gate for
  this path; its saved report includes final evidence plus per-second
  `metric_samples` for capture FPS, broker depth, preprocessing/H2D/inference
  timing, host-frame copy cost, stale drops, control observation FPS, RSS memory, and
  `capture_ts_ns`/`inference_end_ts_ns`/`control_now_ts_ns` timing evidence.
- `tensorrt`: current default GPU inference path. CPU/ONNXRuntime fallback is
  not a runtime execution mode.
- `nvmm_latest`: explicit target/experimental NVMM exchange path requiring
  native GPU preprocess support.
- `deepstream`: Full DeepStream push mode, experimental only and not a default
  control mainline.

Do not silently map one mode to another.

## Implementation Order

1. Preserve current `legacy_latest` and Full DeepStream push baselines.
2. Add frame identity and monotonic timestamp fields across probes.
3. Finish the native `LatestFrameExchange` and resource lifetime integration.
4. Publish ROI-ready NVMM frames into the exchange from the capture pipeline.
5. Implement the single-inflight TensorRT `InferenceLoop` that pulls latest.
6. Add CUDA preprocess, TensorRT `enqueueV3`, output completion synchronization,
   decode/NMS, and result timestamp validation.
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
- Do not allow FIFO historical frame queues between GStreamer sink delivery and
  TensorRT.
- Do not let TensorRT have more than one active inference or a pending inference
  task queue.
- Do not publish stale results into control; stale results are telemetry only.
- Do not silently fall back to CPU while reporting that the NVMM GPU mode is
  running.
