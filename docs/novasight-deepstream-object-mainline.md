# NovaSight DeepStream Object-Meta GPU Image Mainline

## Current Decision

The selectable production candidate is now:

```text
V4L2 MJPEG
-> queue capacity=1, drop old
-> nvv4l2decoder
-> NVMM I420
-> queue capacity=1, drop old
-> nvvidconv ROI crop + resize
-> NVMM NV12
-> queue capacity=1, drop old
-> nvstreammux batch=1
-> nvinfer TensorRT FP16
-> NvDsInferParseNovaSight C++ decode
-> DeepStream cluster-mode=2 NMS
-> NvDsObjectMeta
-> DetectionBatchMailbox capacity=1
-> DetectionBatch
-> Tracker / Hungarian association / Kalman
-> TargetSelector / controller / Scheduler / kmNet
```

Configure it with:

```yaml
capture:
  backend: deepstream_nvinfer
  memory: nvmm
  device: /dev/video0
  pixel_format: MJPG
  width: 1920
  height: 1080
  fps: 120

inference:
  backend: deepstream_nvinfer
  deepstream_io_mode: 2
  deepstream_batched_push_timeout_us: 0
  deepstream_parser_library: build/deepstream-parser/libnovasight_parser.so
```

The dataclass default remains development-safe. The tracked Jetson example
selects `deepstream_nvinfer`; it fails closed when DeepStream, pyds, the parser
library, a complete capture profile, or an active validated engine is missing.

## What The Code Audit Found

Commit `4a3f01a` removed the previous `novasight/deepstream` implementation.
The removed backend copied `NvDsInferTensorMeta` into NumPy and called the
shared Python YOLO/NMS code. Restoring that code would violate the current
requirements.

The replacement probe reads only `NvDsObjectMeta` into caller-owned Rust
snapshots. Python is not present in the live perception path, and Rust never
maps an NVMM image or raw model tensor in this mode.

The downstream control boundary remains the existing NovaSight
`DetectionBatch`. Tracker, target selection, prediction, mouse control, and
kmNet do not import GStreamer, pyds, DeepStream, or TensorRT types.

## Latest-Only Semantics

Each explicit realtime queue is:

```text
max-size-buffers=1
max-size-bytes=0
max-size-time=0
leaky=downstream
```

This means the oldest waiting buffer is discarded when a newer buffer arrives.
The target semantic is:

```text
nvinfer processes frame 100
101 waits
102 replaces 101
103 replaces 102
nvinfer completes 100
the next admitted waiting frame is 103
```

An already executing nvinfer frame cannot be replaced. `nvstreammux` and
`nvinfer` still have bounded internal surface ownership; the Jetson smoke gate
must prove they do not create growing frame age. A pipeline string alone is not
proof of 60-second stability.

`DetectionBatchMailbox` is a separate capacity-one boundary. A newer batch
replaces the previous unconsumed batch; non-monotonic generations or capture
timestamps are rejected.

## Timestamp Contract

`v4l2src do-timestamp=true` produces pipeline-running-time PTS. The backend maps
it into the daemon's monotonic clock domain using:

```text
capture_ts_ns = pipeline_base_time + buffer_pts + gst_clock_to_monotonic_offset
```

The accepted source is named `gst_clock_base_time_pts`. Invalid PTS or the
first-probe offset fallback is visible in telemetry and is rejected before a
batch enters control.

Per batch:

```text
capture_ts_ns
inference_start_ts_ns   # nvinfer sink probe
inference_end_ts_ns     # nvinfer src probe, object metadata ready
publish_ts_ns           # DetectionBatch constructed
generation
frame_id
roi_size
model_input_size
```

`batch_age_ms` is always `publish_ts_ns - capture_ts_ns`.

## Parser And NMS Contract

The native parser supports one raw YOLO output with these layouts:

```text
[C,N]
[N,C]
[1,C,N]
[1,N,C]
```

`C` must be either `4 + class_count` or `5 + class_count`. Coordinates are
pixel-space `cx,cy,w,h`. The parser decodes boxes and class scores but does not
run NMS. DeepStream runs exactly one class-aware NMS pass through
`cluster-mode=2`.

Models that already contain Decode/NMS are rejected by the nvinfer config
generator. They require a separate explicit multi-output parser and
`cluster-mode=4`. Guessing their output structure or applying NMS twice is not
allowed.

## CPU/GPU Boundary

The following stays in NVIDIA image memory:

```text
MJPEG decode
ROI crop
resize
NV12 conversion
TensorRT input and inference
```

There is no appsink, BGR conversion, OpenCV resize, PIL preprocessing, CPU image
frame, or CPU-to-CUDA image re-upload in this mode.

The C++ parser and DeepStream clustering are CPU-side postprocessing unless a
future GPU parser/NMS plugin replaces them. Therefore the accurate claim is:

```text
CPU image processing: 0
CPU image round trip: 0
Python/NumPy NMS: 0
C++ metadata postprocess: present
```

## Build And Verify On Jetson

```bash
cargo build --release -p novasightd --features deepstream
cargo build --release -p novasightctl

target/release/novasightd \
  --config deploy/novasight.production.yaml \
  --check
```

Cargo builds the native parser for the daemon. The installed Rust daemon never
compiles native code at runtime: `novasightd --check` must prove that the
parser, TensorRT runtime, and DeepStream bridge are ABI compatible before
systemd starts capture.

## Runtime Failure Ownership

The Rust supervisor consumes explicit DeepStream fault/EOS events and also
polls the epoch-scoped `PerceptionSession` every 200 ms. The second path is
intentional: if the native owner thread panics before it can publish an event,
its finished join handle still turns the epoch into `Faulted`, closes ingress
and hardware output, and joins downstream workers. An unsolicited clean
perception stop is treated as a fault as well; only a supervisor-owned stop or
restart may end a live session without an error.

The 60-second gate checks:

- observed telemetry spans the requested run duration;
- the pipeline is still running at the final sample and reports no `last_error`;
- at least one `DetectionBatch` was published;
- parser decode calls are real;
- timestamp source is `gst_clock_base_time_pts`;
- mailbox capacity is one;
- sampled `pending_depth`, `published_batches`, and `last_frame_id` never
  violate latest-only monotonicity;
- batch-age P50/P95 telemetry is present;
- batch-age P50 is at most 15 ms;
- batch-age P95 is at most 30 ms;
- late-window frame age does not drift upward by more than 10 ms.

The report explicitly lists two current measurement gaps: nvinfer does not
expose per-frame TensorRT kernel time, and `cluster-mode=2` does not expose
per-frame NMS time. `nvinfer_total_ms` must not be relabeled as either value.

## Remaining Jetson Evidence

Mac verification covers Python compilation, native parser C++ syntax, pipeline
construction, parser/config contracts, latest-mailbox behavior, object-meta
coordinate scaling, and the complete existing test suite.

Only Jetson can prove:

- actual plugin/property compatibility for the installed DeepStream version;
- NVDEC/VIC/NVMM negotiation;
- real parser ABI compatibility;
- output binding layout of the selected engine;
- 60-second frame-age stability;
- P50/P95 latency and parser runtime under 1920x1080 at 120 FPS;
- device thermals and sustained clocks.
