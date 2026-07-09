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

Build and test:

```bash
cmake -S native/latest-frame-exchange -B build/latest-frame-exchange
cmake --build build/latest-frame-exchange
ctest --test-dir build/latest-frame-exchange --output-on-failure
```

The current implementation is resource-agnostic and testable on development
machines. Jetson integration should wrap real `GstBuffer` / `NvBufSurface`
resources with a release callback and publish them into this exchange after
ROI/resize/color conversion.
