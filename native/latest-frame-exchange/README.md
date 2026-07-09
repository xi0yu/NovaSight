# NovaSight LatestFrameExchange

This module is the native contract for the final GPU production route. The
current stable default may still use the compatible CPU latest-frame path with
GPU TensorRT inference until the Jetson NVMM integration is complete.

```text
GC553G2 / V4L2
-> NVIDIA GStreamer / DeepStream GPU data plane
-> NVIDIA hardware MJPEG decode
-> NVMM surface
-> GPU/VIC ROI crop
-> GPU/VIC resize / color conversion
-> upstream leaky sink boundary, capacity=1
-> LatestFrameExchange capacity=1
-> NovaSight TensorRT InferenceLoop
-> CUDA preprocess to model input tensor
-> TensorRT enqueueV3
-> output completion sync
-> decode / NMS
-> DetectionBatch
-> Tracker / Selector / Kalman / Control
```

NVIDIA GStreamer / DeepStream is responsible for capture, hardware decode,
NVMM memory management, ROI crop/resize, basic color conversion, and preserving
source media metadata such as GstBuffer PTS/DTS when present.

NovaSight assigns monotonic runtime identity at the application boundary:
`frame_id`, `ingress_monotonic_ns`, `pipeline_epoch`, and `source_sequence`.
Use those monotonic fields for freshness and control decisions. Treat GstBuffer
PTS/DTS as media diagnostics until the clock mapping is proven.

The final GPU route does not use DeepStream for:

- `nvstreammux`
- `nvinfer`
- TensorMeta
- DeepStream tracker
- inference queuing
- control scheduling

`LatestFrameExchange` is deliberately smaller than a queue. It stores only the
latest ROI-ready frame. A single TensorRT inference loop calls
`acquire_latest_after(last_generation)` when it is ready to run. If capture
produces frames faster than TensorRT can infer, newer frames replace older
pending frames and inference observes generation jumps.

Latest-only must also hold upstream of the exchange. A one-slot exchange cannot
repair stale samples that already accumulated inside GStreamer queues or an
appsink. Realtime capture boundaries must use explicit old-frame drop behavior:

```text
queue max-size-buffers=1 leaky=downstream
appsink max-buffers=1 drop=true sync=false
```

On newer GStreamer versions, use the equivalent supported leaky appsink
property. The rule is the same: sink-side internal depth is one and old frames
are dropped.

## Buffering Model

The source-frame layer uses logical double buffering:

```text
InferenceLoop-owned current frame
+
LatestFrameExchange pending latest frame
```

The current frame is held by the inference loop through its `FrameHandle`. The
pending frame is owned by the exchange and may be replaced repeatedly while the
current frame is still in use. This gives the required two-frame bound without
forcing fixed A/B slots onto `GstBuffer` / `NvBufSurface` resources that are
normally owned by a GStreamer buffer pool.

Jetson integration should make `FrameHandle` hold a strong `GstBuffer`
reference. Do not store only a raw `NvBufSurface*`, and do not carry a temporary
map pointer across the corresponding unmap unless that access is explicitly
validated for the deployed DeepStream and memory type. The consumer should
derive a current CUDA/NvBufSurface access descriptor while it owns the live
handle.

`LatestFrameExchange` capacity one is not the same thing as NVMM buffer-pool
capacity one. The underlying pool still needs a bounded number of surfaces for
decoder, converter, sink delivery, pending exchange ownership, and current
inference ownership. Do not hide backlog by endlessly increasing pool size.

Do not implement source-frame scheduling as unconditional A/B alternation. A
slot is inferable only when it contains a newer `generation` than the last
consumed frame. Repeating an older slot because it is "that slot's turn" breaks
latest-only semantics.

Physical A/B ping-pong belongs at the TensorRT input-buffer layer, where
NovaSight owns the CUDA device memory:

```text
LatestFrameExchange source frame
-> CUDA preprocess
-> TensorRT input slot A / slot B
-> TensorRT enqueueV3
```

Those tensor slots must be guarded by explicit states such as `Empty`,
`Writing`, `Ready`, and `Inferencing`, plus CUDA events or stream dependencies.
An atomic slot index alone is not enough because CPU visibility does not prove
asynchronous CUDA writes or TensorRT reads have completed.

The exchange carries video surfaces, not TensorRT input tensors. A typical
published resource is an NVMM `NvBufSurface` with formats such as NV12 or RGBA.
The inference loop must still run CUDA preprocess:

```text
NVMM video surface
-> CUDA-accessible image data
-> RGB / layout conversion
-> HWC to CHW
-> FP16 / FP32 / INT8 conversion
-> normalization / letterbox
-> TensorRT input device buffer
```

The target is no GPU-to-CPU-to-GPU boundary. A GPU-side write from image surface
to TensorRT input buffer is expected and correct.

Responsibility split:

- `nvvideoconvert` / VIC: ROI crop, resize, and basic video format preparation.
- NovaSight CUDA preprocess: letterbox, RGB channel order, HWC-to-CHW, FP16 /
  FP32 / INT8 conversion, normalization, and TensorRT input-buffer writes.

`enqueueV3()` submits work on a CUDA stream; output decode/NMS must wait on the
correct stream or event before reading model outputs. Decode/NMS may be CPU in
the compatible path and can move to CUDA later. Do not document CUDA NMS as
already required unless that implementation is present.

Build and test:

```bash
cmake -S native/latest-frame-exchange -B build/latest-frame-exchange
cmake --build build/latest-frame-exchange
ctest --test-dir build/latest-frame-exchange --output-on-failure
```

The current implementation is resource-agnostic and testable on development
machines. Jetson integration should wrap real `GstBuffer` / `NvBufSurface`
video surfaces with a release callback and publish them into this exchange
after ROI/resize/color conversion.
