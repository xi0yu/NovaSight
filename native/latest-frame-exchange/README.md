# NovaSight LatestFrameExchange

This module is the native contract for the corrected production route:

```text
GC553G2 / V4L2
-> GStreamer / DeepStream capture pipeline
-> NVIDIA hardware MJPEG decode
-> NVMM surface
-> GPU/VIC ROI crop
-> GPU/VIC resize / color conversion
-> LatestFrameExchange capacity=1
-> NovaSight TensorRT InferenceLoop
-> CUDA preprocess to model input tensor
-> DetectionBatch
-> Tracker / Selector / Kalman / Control
```

DeepStream is responsible for capture, hardware decode, NVMM memory management,
ROI crop/resize, basic color conversion, and frame identity/timestamps.

DeepStream is not responsible for:

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
