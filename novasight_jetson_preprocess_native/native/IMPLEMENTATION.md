# Jetson Production Preprocess Implementation Contract

The Python bridge and the reference native library are not the production
zero-copy implementation. They define the ABI and fail closed. A production
Jetson source must be built with:

```bash
cmake -S native -B build-jetson \
  -DNOVASIGHT_JETSON_PREPROCESS_IMPL=jetson \
  -DNOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE=/path/to/jetson_impl.cpp
cmake --build build-jetson
```

`NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson` requires Linux/aarch64, CUDA Toolkit,
`nvbufsurface.h`, `libnvbufsurface`, and `libnvbufsurftransform`. CMake fails
if the real production source is not provided. This prevents the reference or
scaffold implementations from being mistaken for production NVMM inference.
The production source may be a C++ or CUDA source file (`.cpp` or `.cu`);
`jetson` mode enables the CUDA language with C++17/CUDA17 so a real NV12 to
NCHW device kernel can live in the production source. The portable `reference`
and `jetson_scaffold` modes do not require CUDA.

The repository now includes a bundled production source:

```bash
cmake -S native -B build-jetson \
  -DNOVASIGHT_JETSON_PREPROCESS_IMPL=jetson \
  -DNOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE=$PWD/native/src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu
cmake --build build-jetson
```

`novasight_jetson_preprocess_native_jetson_cuda.cu` implements the intended
DMABUF/NvBufSurface/EGL/CUDA path:

1. `NvBufSurfaceFromFd(dmabuf_fd)` imports the frame resource.
2. `NvBufSurfaceMapEglImage()` exposes an EGL image.
3. `cuGraphicsEGLRegisterImage()` and
   `cuGraphicsResourceGetMappedEglFrame()` expose CUDA access.
4. PITCH frames are read directly; ARRAY frames are copied device-to-device into
   linear NV12 planes.
5. A CUDA kernel resizes, converts NV12 to RGB, normalizes to `[0, 1]`, writes
   NCHW FP32/FP16 device memory, and returns `device_ptr/nbytes/release_token`.

This file is production source code, but it is not proof that the project has
completed Jetson zero-copy deployment until it builds and runs on the target
Jetson with the real `v4l2src -> NVMM appsink` buffers.

The preferred acceptance entrypoint is:

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

That command chains the build/status gate and the real-frame smoke gate. The
JSON report is the preferred artifact to attach to an acceptance record: it
records selected parameters, the built library path, each phase exit code, and
each phase stdout with the actual native status and real-frame smoke evidence,
plus the final `accepted` value. The one-shot command validates that report
before printing `zero_copy_acceptance: True`, so a build/smoke command that
returns 0 without the required stdout evidence is still rejected. Rejected
one-shot reports carry `accepted=false`,
`reason=jetson_zero_copy_report_invalid`, and a `validation_failures` array with
the exact verifier failures; successful reports carry an empty list.
`--tensorrt-engine` is optional while debugging the bridge, but a final
production zero-copy inference acceptance artifact must run with both
`--tensorrt-engine` and `--require-tensorrt-engine`; otherwise the one-shot
command must fail before build/smoke rather than producing a native-only
acceptance report. Validate that artifact before accepting it:

```bash
python -m novasight doctor jetson-zero-copy-report \
  --report-json data/diagnostics/jetson-zero-copy.json \
  --require-tensorrt-engine
```

The report verifier checks both phase exit codes and the required stdout
evidence: empty accepted `reason`, empty accepted `validation_failures`,
timezone-qualified `started_at/finished_at` values in chronological order,
native `status_available/status_zero_copy/status_memory_space`,
production `status_backend`, real-frame `capture_available=True`,
`gst-resource` capture backend, NVMM resource memory, capture profile, ROI frame
identity (`frame_id`), monotonic capture timestamp (`capture_ts_ns`), timestamp
source (`capture_ts_source=userspace_monotonic_receive`), independent GStreamer
source timestamp diagnostics (`source_ts_ns/source_ts_kind`), ROI frame size, ROI
offset, NV12 frame/source geometry, appsink resource source, `dmabuf_fd` or
`gst_buffer_ptr`, GPU `PreparedTensorInput`,
TensorRT input shape/dtype, bridge readiness, device preprocess location,
zero-copy preprocess, smoke probe owner release, and device tensor
pointer/size/shape/dtype. Critical stdout evidence keys must be unique; a report
with duplicate key/value evidence is rejected so a later correct line cannot
hide an earlier wrong value. It also
validates the evidence values: backend must be production `jetson_cuda`; report
parameters must include capture format, source geometry, FPS, ROI size, input
shape, and dtype; `capture_available` must be `True`; `frame_id` and
`capture_ts_ns` must be positive integers; `capture_ts_source` must be
`userspace_monotonic_receive`; `source_ts_ns/source_ts_kind` must be present as
diagnostic metadata and must not replace the control timestamp; `source_ts_kind`
is limited to `gstreamer_pts`, `gstreamer_dts`, or empty, with a non-negative
`source_ts_ns` required when a GStreamer kind is present; `roi_offset` must match
`parameters.roi_offset_x/roi_offset_y`; `resource_source` and
`prepared_resource_source` must both be `appsink`; `prepared_frame_id` and
`prepared_capture_ts_ns` must match the captured `frame_id/capture_ts_ns`; the
captured and prepared resources must expose matching `dmabuf_fd` or matching
`gst_buffer_ptr`;
`bridge_available/bridge_native_ready` must be true; `preprocess_location` must
be `device`; the preprocess backend must not be reference/scaffold;
`smoke_device_owner_release` must be `released`;
`tensor_device_ptr/tensor_nbytes` must be positive integers; `tensor_nbytes`
must equal input shape multiplied by dtype byte width; and
`capture_profile/source_size/frame_size/roi_offset/input_shape/input_dtype/tensor_shape/tensor_dtype`
must match the report parameters. Reference and scaffold backends are rejected
even if the report claims `accepted=true`. Final production acceptance must keep
`--require-tensorrt-engine` enabled; native-only reports without
`parameters.tensorrt_engine` are useful for bridge debugging but are not enough
to accept zero-copy inference. When `tensorrt_engine` is present in the report
parameters, the verifier also rejects reports unless TensorRT prints
`tensorrt_available=True`, `tensorrt_last_input_mode=gpu_buffer`,
`tensorrt_last_input_resource_memory=nvmm`,
`tensorrt_preprocess_location=device`, `tensorrt_preprocess_zero_copy=True`, and
`tensorrt_input_location=device`. It also validates that the TensorRT input
contract is a batch-1, 3-channel NCHW tensor with a supported FP32/FP16 dtype,
and that `tensorrt_last_input_dmabuf_fd` matches the captured `dmabuf_fd` and
the GPU `PreparedTensorInput` `prepared_dmabuf_fd`. Finally, it requires
`tensorrt_frame_id` to match `frame_id` and `tensorrt_capture_ts_ns` to match
`capture_ts_ns`, and requires TensorRT engine status
`tensorrt_last_input_frame_id/tensorrt_last_input_capture_ts_ns` to match the
same captured frame, so TensorRT cannot satisfy the report with a stale or
different frame. It also requires TensorRT output execution evidence:
non-empty `tensorrt_output_name`, positive-dimensional
`tensorrt_output_shape`, supported `tensorrt_output_dtype`, non-negative
`tensorrt_decoded_detections`, and non-negative
`tensorrt_execute_enqueue_ms/tensorrt_d2h_enqueue_ms`, plus
`tensorrt_device_owner_release=released` so native device tensor ownership is
not leaked after TensorRT binding execution. A successful `jetson-zero-copy-report`
run prints a compact `evidence:` summary plus individual
`evidence.<field>: <value>` lines for the production backend, actual
`capture_device`, capture timestamp source, GStreamer source timestamp
diagnostics, NVMM appsink resource, DMABUF, preprocess, owner release, and
TensorRT device-input evidence; this is the human-readable receipt to paste
into acceptance notes. The report verifier also rejects a smoke
`capture_device` that does not match `parameters.device`, so accepted reports
prove the requested device was the device actually opened. When one-shot
validation rejects a report, `validation_failures` is the authoritative machine
readable triage list. The split build command below is useful when compiler or
linker output needs to be debugged separately:

```bash
python -m novasight doctor jetson-native-build \
  --build-dir build/jetson-native
```

The command configures `NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson`, uses the
bundled `.cu` source by default, builds the shared library, then sets
`NOVASIGHT_JETSON_NATIVE_LIBRARY` to the build output and calls the ctypes
native status hook. It must print `status_available: True`,
`status_zero_copy: True`, and `status_memory_space: cuda_device` before
`capture.memory=nvmm` should be treated as production-ready.

`status_available: True` alone is not sufficient. The doctor command rejects a
missing production source, a successful build that does not create
`libnovasight_preprocess.so`, and any loaded status hook that does not
explicitly return `zero_copy=true` with `memory_space=cuda_device`. The build
also writes `libnovasight_jetson_preprocess_native.so` as a legacy alias.

The real-frame acceptance entrypoint is:

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

This command does not call the native bridge in isolation. It opens the normal
NVMM capture path, requires a GPU-accessible `FrameResource` with `dmabuf_fd`,
builds the same `PreparedTensorInput` used by TensorRT, and validates that the
native bridge returns a zero-copy `DeviceTensor`. With `--tensorrt-engine`, it
then calls `TensorRtInferenceEngine.infer(frame)` on the same frame and requires
the engine status/debug evidence to show `gpu_buffer/nvmm` input,
device-location preprocess, zero-copy preprocess, and TensorRT
`input_location=device`. The smoke output also records
`tensorrt_frame_id/tensorrt_capture_ts_ns` and the report verifier requires them
to match the captured `frame_id/capture_ts_ns`. The smoke output additionally
records TensorRT output name/shape/dtype, decoded detection count, execute
enqueue timing, and output D2H timing. This closes the gap between "native
preprocess produced a device tensor" and "TensorRT actually used that same
captured frame as the input binding and produced an output buffer."

The `jetson` CMake mode also compiles
`src/jetson/novasight_jetson_preprocess_native_jetson_support.cpp` and exposes
`novasight_jetson_preprocess_native_jetson_support.h` to the production source.
Use this helper for JSON payload parsing, contract validation, dtype/nbytes
calculation, and fail-closed status/error JSON. Diagnostic strings are JSON
escaped by the helper so driver paths, quotes, and multiline native errors do
not corrupt the ABI response. The helper does not import DMABUF/NvBufSurface
resources and does not create a tensor; the production source remains
responsible for the CUDA/NvBufSurface conversion.

## Build on Jetson (copy-paste recipe)

The Jetson production preprocess library depends on JetPack multimedia API
headers/libs that are not always preinstalled on a stock L4T image. Run this
once on the device (after `scripts/setup_jetson.sh` finishes), then rebuild.

```bash
# 1. Confirm dependency readiness in <1s without invoking cmake.
python -m novasight doctor jetson-preflight

# If "missing_components" lists anything, install it (the doctor prints the
# exact apt command):
sudo apt update
sudo apt install -y nvidia-l4t-jetson-multimedia-api libegl1 libegl-dev libgles2

# 2. Configure + build the production shared library.
python -m novasight doctor jetson-native-build \
  --build-dir build/jetson-native

# Expected artifacts:
#   build/jetson-native/libnovasight_preprocess.so          (canonical)
#   build/jetson-native/libnovasight_jetson_preprocess_native.so  (legacy alias)
ls -l build/jetson-native/libnovasight_preprocess.so

# 3. End-to-end smoke (real frame through the bridge -> TensorRT engine):
python -m novasight doctor jetson-native-smoke \
  --library build/jetson-native/libnovasight_preprocess.so \
  --device /dev/video0 \
  --pixel-format MJPG \
  --width 1920 --height 1080 --fps 30 \
  --roi-size 640 \
  --input-shape 1x3x640x640 \
  --tensorrt-engine data/models/<model>.engine
```

If `doctor jetson-native-build` itself prints an "apt_fix=" snippet into its
detail, run that snippet and re-run the doctor — it means
`nvidia-l4t-jetson-multimedia-api` is not installed in this L4T image.

If the headers/libs live in a non-standard path, override via env so the build
does not require apt:

```bash
export NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR=/path/to/include
export NOVASIGHT_NVBUFSURFACE_LIBRARY=/path/to/libnvbufsurface.so
export NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY=/path/to/libnvbufsurftransform.so
python -m novasight doctor jetson-native-build --build-dir build/jetson-native
```

The doctor configures with `NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson`,
`NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE=.../native/src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu`,
then `cmake --build`. A successful run must print `available: True`,
`status_zero_copy: True`, and `status_memory_space: cuda_device`. Anything else
means the bundled `.cu` was not linked against real nvbufsurface, even if the
.so file exists.

## Run with capture.memory=nvmm

Once `libnovasight_preprocess.so` is built, the NovaSight backend will find
it via `NOVASIGHT_JETSON_NATIVE_LIBRARY`, falling back to a built-in candidate
scan in `novasight_jetson_preprocess_native._library_path()` that already
probes `build/jetson-native/libnovasight_preprocess.so` from both cwd and
the repository root. For an explicit, prod-style setup:

```bash
cd /home/nvidia/NovaSight
export NOVASIGHT_JETSON_NATIVE_LIBRARY="$PWD/build/jetson-native/libnovasight_preprocess.so"

cp -n config/novasight.example.yaml config/novasight.yaml
# Edit config/novasight.yaml: set
#   capture.memory: nvmm           # appsink delivers dmabuf NvBufSurface
#   capture.device: /dev/video0    # or /dev/video1 for the second CSI
#   inference.backend: nvmm_latest # TensorRT engine that consumes DeviceTensor
#   inference.enabled: true
```

Start the backend:

```bash
python -m novasight --host 0.0.0.0 --port 5174
```

Open `http://<jetson-ip>:5174` and confirm via the device status route that
`gpu_preprocessor.available` is `true` and `gpu_preprocessor.native_ready`
is `true`. The 400 error `JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE` should be
gone; the pipeline is now producing real NCHW device tensors.

## Required Input

`novasight_prepare_tensor_json(payload_json,result_json,size)` receives JSON
metadata from Python. Python keeps the `Gst.Sample`/`GstBuffer` resource handle
alive while native preprocessing runs. The production source must use either
`dmabuf_fd` or `gst_buffer_ptr` plus the metadata below:

- `resource_kind=gstreamer_sample`
- `resource_memory=nvmm` or `dmabuf`
- `resource_source=appsink`
- `dmabuf_fd` as a non-negative file descriptor, or `gst_buffer_ptr` as a
  positive live `GstBuffer*` pointer
- `pixel_format=NV12`
- ROI dimensions: `width`, `height`
- source capture dimensions: `source_width`, `source_height`
- ROI offsets when present: `roi_offset_x`, `roi_offset_y`
- requested tensor layout: `nchw=[N,C,H,W]`
- tensor `dtype`, `float32` or `float16`, matching the TensorRT engine input binding

## Required Work

The production source must implement GPU-side ingest:

1. Import the DMABUF/NvBufSurface resource without mapping it into a CPU numpy
   image.
2. Crop/resize according to the ROI metadata and requested model input shape.
3. Convert NV12 to model-ready NCHW on device memory.
4. Normalize pixels according to the inference input contract.
5. Return a JSON result with `device_ptr`, `nbytes`, a positive
   `release_token`, `shape`, `dtype`, `backend`, `zero_copy=true`, and
   `memory_space=cuda_device`. The ctypes bridge rejects successful device
   tensor results that cannot be released deterministically.
6. Implement `novasight_release_tensor(release_token)` for every successful
   result. Non-zero release return codes are treated as runtime release
   failures and must not be reported as successful smoke/TensorRT evidence.
7. Implement `novasight_status_json` as a production readiness gate. It must
   report `available=true` or `ready=true`, `zero_copy=true`, and device
   `memory_space` only after the real DMABUF/NvBufSurface/EGL/CUDA conversion
   path is initialized enough to accept frames. Do not report ready from a
   validator, scaffold, or library that has not proven device output support.

The production source must not return success with dummy tensors, zero-filled
buffers, host pointers, or CPU-converted numpy data. If any GPU import,
transform, or allocation step fails, return a non-zero code with a concise
diagnostic JSON object.

## Timestamp Boundary

The frame timestamp used by control remains the Python userspace receive time:
`CapturedFrame.capture_ts_ns = time.monotonic_ns()` immediately after
`appsink.try_pull_sample()` returns. Native preprocess must not replace that
timestamp with GStreamer PTS/DTS or wall-clock time. Source PTS/DTS remains
diagnostic metadata only.
