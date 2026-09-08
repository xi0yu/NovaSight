# CUDA raw YOLO postprocessing candidate

This is the shared compute candidate for the GPU pipeline experiment, not yet
connected to the production daemon. It accepts explicit FP32/FP16 raw YOLO
contracts with or without objectness, in `[C,N]` or `[N,C]` layout. It does not
claim decoded-NMS, EfficientNMS or Rockchip head support.

Decode, confidence filtering, stable score sorting, class-aware greedy NMS and
final packing all run on CUDA. The uniform confidence/IoU thresholds, best-class
selection, clipping, stable ties, per-class **post-NMS** Top-K and default strict
post-cluster threshold `> 0` follow the current raw parser / DS 7.1 combination.
There is no pre-NMS Top-K. The bounded result preserves the first 256 detections
in class order and reports the truncated count, matching the existing bridge's
prefix convention. Coordinate remapping and frame identity still belong to the
existing platform adapter and are not implemented by this tensor-only module.

Each instance owns persistent scratch, a completion event and pinned final-result
storage. The producer orders its work before `enqueue` on the same CUDA stream;
the input must stay alive through `wait`. A second outstanding frame is rejected.
Launch/completion failures invalidate the instance; no CPU fallback exists.
Only the 6,152-byte bounded result is copied to the host. The caller must consume
or copy it before reuse and retain its frame/epoch identity. The candidate does
not yet manage capture frames or decide between DeepStream and direct TensorRT.

Build and run on the Jetson, with the installed toolkit and an isolated build:

```sh
cmake -S native/yolo-postprocess -B /tmp/novasight-yolo-gpu-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build /tmp/novasight-yolo-gpu-build -j2
ctest --test-dir /tmp/novasight-yolo-gpu-build --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /tmp/novasight-yolo-gpu-build/yolo_gpu_contract_test
/tmp/novasight-yolo-gpu-build/yolo_gpu_contract_test --benchmark
```

For the measured five-class FP32 `[1,9,1344]` engine with 256px input:

```sh
/tmp/novasight-yolo-gpu-build/yolo_gpu_contract_test --engine /path/to/model.engine
nsys profile --trace=cuda --sample=none --cpuctxsw=none --output=/tmp/yolo-device \
  /tmp/novasight-yolo-gpu-build/yolo_gpu_contract_test --trace-engine /path/to/model.engine
```

This uses the original `native/tensorrt-runtime` implementation's additive device
API, not a second TensorRT loader. A producer event orders input readiness on the
GPU, inference and postprocessing share the engine's stream, and completion
releases the outstanding frame. The check also rejects overlapping host/device
execution and tests destruction with outstanding work. Input is all zero;
`--engine` compares against the existing host API plus the CPU reference, while
`--trace-engine` omits reference execution to isolate actual D2H transfers.
Neither mode exercises media input, `DetectionBatch` admission or control.

The native API also has an explicit, optional `capture_device_graph` call after
a completed warm-up. The engine owns an inference-only CUDA graph; changing input
addresses or entering the host API destroys it before the TensorRT context is
modified. Unsupported capture or graph execution errors stop this candidate.
The engine test compares raw outputs for different input addresses to catch
stale graph bindings. Only these extra checks use a constant nonzero input;
the timed engine loop remains zero input. CUDA Graph affects submission, not
the model, precision or postprocessing rules. The binding rule follows NVIDIA's
[TensorRT CUDA Graph guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html).

The comparison executable uses the installed DeepStream `NvDsInferContext`
preprocessed-input/device-output API, not the GStreamer `nvinfer` integration:

```sh
/tmp/novasight-yolo-gpu-build/deepstream_device_test /path/to/model.engine
nsys profile --trace=cuda --sample=none --cpuctxsw=none --output=/tmp/ds-device \
  /tmp/novasight-yolo-gpu-build/deepstream_device_test --trace /path/to/model.engine
```

**The installed DS 7.1 `Other` candidate fails the host-copy requirement.**
Raw tensor parity passed, but tracing found 120 copies of the 48,384-byte raw
output as well as 120 final-result copies. Its installed SDK source shows
`OtherPostprocessor::initResource()` returns without calling the base initializer
that reads `disableOutputHostCopy`. The flag therefore remains at its default.
No SDK library was patched. `deepstream_device_pass` only means that the device
output contract check ran; it is not hardware-execution or performance acceptance.
The direct TensorRT trace has the 120 final-result copies and no raw-output copies.

The bounded capture experiment keeps the measured MJPG 1080p120 input, ROI
`(800,380,320,320)`, 256px model, RGB normalization and thresholds. VIC produces
RGBA, CUDA reads the imported NVMM surface and packs normalized RGB CHW, then
the original TensorRT device API feeds the shared GPU postprocessor. GstBuffer
mapping reads the surface descriptor, not host image pixels. The sample and EGL
registration stay alive through the image-reading work. Unsupported EGL layouts
or GPU errors stop the experiment; there is no software image path.

```sh
/tmp/novasight-yolo-gpu-build/capture_device_test --self-test
# Run only while /dev/video0 and the product are idle; duration is 1..30 seconds.
/tmp/novasight-yolo-gpu-build/capture_device_test /path/to/model.engine 10
# Experimental graph submission; it did not consistently improve capture P95:
/tmp/novasight-yolo-gpu-build/capture_device_test /path/to/model.engine 10 --graph
# This fixture's 240 mode is 5000000/20833; count delivered/completed PTS with the probe.
/tmp/novasight-yolo-gpu-build/capture_device_test /path/to/model.engine 10 --fps=240
```

It has no control/device-output code and is not connected to the daemon.
The self-test checks all RGB values across a padded stride. Capture adds a
two-second warm-up; reported image-ready latency starts at appsink consumption,
not driver capture. With the existing P0 probe preloaded, the `gpu_candidate/result`
span allows `analyze_pipeline_trace.py --gpu-candidate DIRECTORY` to join JPEG
parser output to completed results by unique PTS. End-to-end capture age, image
quality against the original preprocessing, preview/crosshair and product
lifecycle still require separate validation. Per-frame EGL registration is
included in the measurement and remains an optimization candidate.

On this Jetson, Compute Sanitizer requires the `debug` group. Verification used
the existing `nvidia` account with that group for the diagnostic process only;
system group membership and driver settings were not changed. A tool error about
disabled debugging is not a successful memory/race check.

The contract test calls the existing parser source as its CPU decode oracle and
uses a sequential NMS reference with the installed SDK's verified rules. It checks
FP16/FP32, both layouts, objectness, greedy suppression chains, ties, clipping,
nonfinite values, exact thresholds, truncation, stream ordering and scratch reuse.
Bounds are explicit: at most 32,768 raw candidates, 1,024 classes, 16,384-pixel
dimensions and 256 post-NMS results per class. Unsupported contracts are rejected;
these limits are not evidence that all production models have been covered.

Benchmark inputs are synthetic and uploaded once before measurement. GPU wall
times cover host submission through final results; CUDA event times include the
queued postprocessing and final copy, and can include GPU idle time between host
submissions. The CPU comparator uses the real parser plus the test NMS reference,
not the entire nvinfer plugin. Neither is a capture/control or RK3588 benchmark.
Wall and event distributions currently come from separate GPU executions with
CPU reference work between them; they must not be added or read as same-frame
components. Measured results and their limits are recorded in the
[dated report](../../docs/pipeline-gpu-baseline-20260908.md).
