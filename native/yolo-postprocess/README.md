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
