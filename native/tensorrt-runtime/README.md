# NovaSight TensorRT Runtime

This module is the narrow native seam for Rust-owned TensorRT execution. The
production Jetson implementation performs real engine deserialization,
single-context `enqueueV3`, device-to-host output copies, and CUDA stream
synchronization. The portable reference implementation always fails closed and
exists only to verify the ABI on development hosts.

The runtime intentionally supports the control-mainline contract only:

- one `[1,3,H,W]` FP16 or FP32 device input;
- one to eight resolved FP16/FP32 outputs;
- caller-owned CUDA input memory;
- runtime-owned device outputs and pinned host mirrors;
- exactly one owner thread and one in-flight execution.

Build the portable ABI contract:

```bash
cmake -S native/tensorrt-runtime -B build/tensorrt-contract \
  -DNOVASIGHT_TENSORRT_IMPL=reference
cmake --build build/tensorrt-contract --parallel
ctest --test-dir build/tensorrt-contract --output-on-failure
```

Build the production library on Jetson:

```bash
scripts/build_tensorrt_runtime.sh
export NOVASIGHT_TENSORRT_RUNTIME_DIR="$PWD/build/jetson-native"
```

`novasight_tensorrt_execute` does not return until its CUDA stream has
synchronized. If an error occurs after enqueue, a scope guard drains the stream
before control returns across the C ABI, so Rust cannot release the input
`DeviceTensor` while TensorRT still reads it.
