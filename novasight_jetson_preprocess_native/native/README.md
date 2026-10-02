# NovaSight Jetson Native Preprocess Library

This directory contains the C ABI entrypoint expected by
`novasight_jetson_preprocess_native`.

Build the reference shared library:

```bash
cmake -S native -B build
cmake --build build
```

Load it from Python:

```bash
export NOVASIGHT_JETSON_NATIVE_LIBRARY=/path/to/libnovasight_preprocess.so
python -m novasight doctor jetson-bridge
```

The reference implementation is intentionally fail-closed. It validates the ABI,
checks the payload shape and supported resource contract, then reports
`jetson_cuda_preprocess_not_compiled`; it does not convert
DMABUF/NvBufSurface/EGL/CUDA resources into TensorRT input tensors.

The reference validator currently accepts:

- positive `frame_id` and `capture_ts_ns`
- `resource_kind=gstreamer_sample`
- `resource_memory=dmabuf` or `resource_memory=nvmm`
- `resource_source=appsink` for the legacy Python capture path, or
  `resource_source=deepstream_pad` while Rust holds the source `GstBuffer`
- `pixel_format=NV12`
- `dtype=float32` or `dtype=float16`
- a non-negative `dmabuf_fd` or a positive `gst_buffer_ptr`
- positive `width`, `height`, `source_width`, `source_height`
- `nchw=[N,C,H,W]` with channel count `1`, `3`, or `4`

A production Jetson implementation must keep the same symbols:

- `novasight_abi_version`
- `novasight_status_json`
- `novasight_prepare_tensor_json`
- `novasight_release_tensor`

`novasight_abi_version` must return
`NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION`, or `novasight_status_json` must
include the same `abi_version` field. The Python adapter rejects libraries
that do not explicitly report the expected ABI.

`novasight_status_json` is required for production readiness. A ready library
must explicitly report `available=true` or `ready=true`, `zero_copy=true`, and a
device `memory_space` such as `cuda_device`. A library that only exports
`novasight_prepare_tensor_json` is not allowed to start `capture.memory=nvmm`
inference, even if its prepare function could be called directly.

`novasight_prepare_tensor_json` must return a JSON object containing at least
`device_ptr`, `nbytes`, `zero_copy=true`, and `memory_space=cuda_device` for a
model-ready NCHW tensor on GPU memory.

## Implementation selection

The native CMake project intentionally separates ABI validation from production
Jetson preprocessing:

- `NOVASIGHT_JETSON_PREPROCESS_IMPL=reference` is the default portable
  fail-closed validator.
- `NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson_scaffold` builds a fail-closed ABI
  scaffold for build-system validation.
- `NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson` is the production mode. It requires
  Linux/aarch64, CUDA Toolkit, Jetson NvBufSurface/NvBufSurfTransform
  libraries, and a real source file provided through
  `NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE`.
  The production source may be C++ or CUDA (`.cpp` or `.cu`); CUDA language
  support is enabled only for this mode.

This repository includes a bundled Jetson CUDA production source at
`src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu`. Build it on a
Jetson target with:

```bash
cmake -S native -B build-jetson \
  -DNOVASIGHT_JETSON_PREPROCESS_IMPL=jetson \
  -DNOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE=$PWD/native/src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu
cmake --build build-jetson
```

The bundled source imports `dmabuf_fd` with NvBufSurface when available. When
GStreamer exposes NVMM without a DMABUF fd, either appsink or the Rust
DeepStream pad adapter may pass the live, strongly-held `GstBuffer*` as
`gst_buffer_ptr`. The native code maps it to `NvBufSurface` only for the
duration of the prepare call. It then
maps the surface to EGL/CUDA, converts NV12 to normalized NCHW FP32/FP16 device
memory, and returns a `release_token` for the CUDA allocation. It still must be
compiled and verified on the actual Jetson/GStreamer surface layout before
being treated as deployed zero-copy inference.

The application-level Jetson build check wraps the same CMake production mode
and then loads the resulting shared library through the ctypes bridge:

```bash
python -m novasight doctor jetson-zero-copy \
  --build-dir build/jetson-native \
  --device /dev/video0 \
  --pixel-format MJPG \
  --width 1920 \
  --height 1080 \
  --fps 120 \
  --roi-size 640 \
  --input-shape 1x3x640x640 \
  --tensorrt-engine data/models/<model>.engine \
  --require-tensorrt-engine \
  --report-json data/diagnostics/jetson-zero-copy.json
```

This is the preferred one-command acceptance gate. It first runs the native
build/status check, then automatically passes the built library into the
real-frame smoke check. Use the split commands below when you need to isolate a
compiler/linker problem from a capture/resource/import problem. When
`--report-json` is set, the command writes a machine-readable artifact with the
build and smoke phase exit codes, selected parameters, built library path, and
final `accepted` status. The report also stores each phase's stdout so the
artifact contains the actual `status_zero_copy`, `dmabuf_fd`, `gst_buffer_ptr`,
`preprocess_zero_copy`, TensorRT frame identity, TensorRT last-input,
TensorRT input-location evidence, and TensorRT output execution evidence
printed during acceptance. The one-shot command self-validates the
generated report before printing `zero_copy_acceptance: True`; phase exit codes
alone are not enough. If self-validation fails, the JSON report is left with
`accepted=false`, `reason=jetson_zero_copy_report_invalid`, and a
`validation_failures` list containing the exact verifier failures. A successful
report has `validation_failures=[]`. `--tensorrt-engine` is optional for early
native bridge debugging, but a final zero-copy inference acceptance record must
run the one-shot command with both `--tensorrt-engine` and
`--require-tensorrt-engine` so the command fails before build/smoke if TensorRT
evidence cannot be produced.

Validate the artifact before treating it as acceptance evidence:

```bash
python -m novasight doctor jetson-zero-copy-report \
  --report-json data/diagnostics/jetson-zero-copy.json \
  --require-tensorrt-engine
```

The report check rejects weak artifacts that only claim `accepted=true` without
an empty `reason`, empty `validation_failures`, timezone-qualified
`started_at/finished_at` values in chronological order, the production native
backend status, `gst-resource` NVMM capture backend, capture profile, monotonic
capture timestamp plus independent GStreamer source timestamp diagnostics, ROI
frame size, NV12 frame/source geometry, DMABUF, GPU resource, GPU
`PreparedTensorInput` frame identity, TensorRT input shape/dtype, bridge
readiness, device preprocess location, zero-copy preprocess, and device tensor
pointer/size/shape/dtype plus smoke probe owner release evidence required by
the production path. Critical stdout evidence keys must appear exactly once;
duplicate keys are rejected instead of letting a later line overwrite an earlier
wrong value. It also requires the report
parameters to include the capture format, source geometry, FPS, ROI size, input
shape, and dtype; verifies that each frame has either matching `dmabuf_fd`
values or matching `gst_buffer_ptr` values; verifies that `bridge_available` and
`bridge_native_ready` are true; verifies that `preprocess_location` is `device`
and the preprocess backend is not reference/scaffold; verifies that
`source_ts_kind` is only `gstreamer_pts`, `gstreamer_dts`, or empty, with
non-negative `source_ts_ns` required when the kind is present; verifies that
`tensor_device_ptr` and `tensor_nbytes` are positive integers; verifies that
`tensor_nbytes` equals the reported input shape multiplied by the dtype byte
width; and checks that reported capture profile/source size/ROI size/input
shape/input dtype/tensor shape/tensor dtype match the command parameters.
`smoke_device_owner_release` must be `released`, proving the native smoke probe
tensor owner/release token was cleaned up before the command returns.
Reference and scaffold backend evidence is rejected. When the report parameters
include `tensorrt_engine`, the verifier also requires
`tensorrt_available=True`, `tensorrt_last_input_mode=gpu_buffer`,
`tensorrt_last_input_resource_memory=nvmm`,
`tensorrt_preprocess_location=device`, `tensorrt_preprocess_zero_copy=True`,
and `tensorrt_input_location=device`. The TensorRT input contract must also be
a valid batch-1, 3-channel NCHW tensor shape with a supported FP32/FP16 dtype,
and TensorRT last-input resource handles must match both the captured resource
and the prepared resource through either `dmabuf_fd` or `gst_buffer_ptr`. It
also requires positive `frame_id`,
`capture_ts_ns`, `prepared_frame_id`, `prepared_capture_ts_ns`,
`tensorrt_frame_id`, `tensorrt_capture_ts_ns`,
`tensorrt_last_input_frame_id`, and `tensorrt_last_input_capture_ts_ns`
values, with the prepared and TensorRT frame fields matching the captured frame
fields. TensorRT output
evidence is also mandatory: `tensorrt_output_name` must be non-empty,
`tensorrt_output_shape` must contain positive dimensions,
`tensorrt_output_dtype` must be supported, `tensorrt_decoded_detections` must be
a non-negative integer, and `tensorrt_execute_enqueue_ms` plus
`tensorrt_d2h_enqueue_ms` must be non-negative timings. The verifier also
requires `tensorrt_device_owner_release=released`, proving the native device
tensor owner/release token was cleaned up after TensorRT execution. Keep
`--require-tensorrt-engine` enabled for final production acceptance; omit it
only while debugging the native bridge before TensorRT is attached.

On success, `jetson-zero-copy-report` prints a compact `evidence:` summary plus
individual `evidence.<field>: <value>` lines for the same key facts. Treat those
lines as the human-readable acceptance receipt: they should show the production
backend, actual `capture_device`, `capture_ts_ns` clock source,
`source_ts_kind`, NVMM appsink resource, DMABUF fd, zero-copy preprocess, owner
release, and TensorRT device-input evidence when a TensorRT engine is required.
The verifier rejects reports whose smoke `capture_device` does not match the
requested `parameters.device`, so accepted reports prove both the data path and
the device identity. Rejected one-shot reports should be triaged from
`validation_failures` before trusting any phase stdout.

```bash
python -m novasight doctor jetson-native-build \
  --build-dir build/jetson-native
```

It returns success only when CMake configure/build succeeds and the built
library reports native readiness with `zero_copy=true` and
`memory_space=cuda_device`. Use `--skip-load` only to debug compiler or linker
errors before the library is ready to be loaded.

The command fails before CMake when the selected production source is missing,
fails after build when `libnovasight_preprocess.so` was not created, and
rejects status hooks that report `available=true` without the device zero-copy
output contract. `libnovasight_jetson_preprocess_native.so` is still written as
a legacy alias for old local scripts.

After a successful build check, run the real-frame smoke check on the Jetson:

```bash
python -m novasight doctor jetson-native-smoke \
  --library build/jetson-native/libnovasight_preprocess.so \
  --device /dev/video0 \
  --pixel-format MJPG \
  --width 1920 \
  --height 1080 \
  --fps 120 \
  --roi-size 640 \
  --input-shape 1x3x640x640 \
  --tensorrt-engine data/models/<model>.engine
```

This command forces `capture.memory=nvmm`, opens the normal `CaptureService`
resource appsink path, waits for a real frame carrying a GPU-accessible
`FrameResource` with `dmabuf_fd`, calls the Jetson preprocessor through
`prepare_tensor()`, and requires a `DeviceTensor` result with a callable owner
release path. For the bundled ctypes bridge that means the native result must
return a positive `release_token` and export `novasight_release_tensor()`. When
`--tensorrt-engine` is provided, the same frame is then passed through
`TensorRtInferenceEngine.infer(frame)` and the smoke check requires
`gpu_buffer/nvmm` as the engine's last input plus `input_location=device` in the
TensorRT execution timings. It also prints `tensorrt_frame_id` and
`tensorrt_capture_ts_ns` so the report verifier can prove TensorRT consumed the
same captured frame, not just another buffer with a compatible shape. It also
prints TensorRT output name/shape/dtype, decoded detection count, execute
enqueue timing, and output D2H timing, so a smoke report cannot pass without
evidence that the engine executed and produced an output buffer. It is the
command that proves the production capture resource can actually cross the
native bridge and, with an engine, enter TensorRT without falling back to host
input.

`jetson` mode automatically compiles the support helper in
`src/jetson/novasight_jetson_preprocess_native_jetson_support.{h,cpp}`. The
helper handles payload parsing, contract validation, dtype/nbytes calculation,
and fail-closed status/error JSON for a production source. Native diagnostic
strings are JSON escaped before they cross the ABI boundary. It does not import
DMABUF/NvBufSurface resources or create a tensor.

See `IMPLEMENTATION.md` for the production resource and tensor contract. The
project must not use `reference` or `jetson_scaffold` as proof that
`capture.memory=nvmm` inference is zero-copy.
