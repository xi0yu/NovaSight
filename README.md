# NovaSight

NovaSight is a Jetson-first realtime vision console rebuilt from the old jetcam
prototype with cleaner model, configuration, plugin, executor, API, and UI
boundaries.

## Current Project Authority

Use `PROJECT_HEALTH_AUDIT.md` as the current cleanup and over-design ledger.
Older plans under `docs/superpowers/` are archive material unless promoted by a
current authority document.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Start the backend API:

```bash
python3 -m novasight --host 127.0.0.1 --port 5174
```

Start the web console in a second shell:

```bash
pnpm --dir web install
pnpm --dir web dev
```

Open http://127.0.0.1:5173 during development. Vite proxies `/api` and
`/healthz` to the backend on port 5174.

Run the full local checks:

```bash
pytest -q
pnpm --dir web typecheck
pnpm --dir web build
```

## Runtime Control API

The backend exposes runtime control endpoints for the dual-thread pipeline:

```bash
curl -X POST http://127.0.0.1:5174/api/license/activate \
  -H "Content-Type: application/json" \
  -d '{"key":"NOVASIGHT-TEST-MAX-ACCESS-2026"}'
curl http://127.0.0.1:5174/api/config
curl -X POST http://127.0.0.1:5174/api/runtime/start
curl -X POST http://127.0.0.1:5174/api/runtime/stop
```

`POST /api/runtime/start` opens the configured capture source, starts Thread A
for capture, starts Thread B for inference/control, and connects them through a
capacity-1 latest-frame queue. Runtime status is also streamed through:

```text
ws://127.0.0.1:5174/ws/status
```

Core thread crashes are handled fail-fast: NovaSight writes `logs/crash_*.log`,
records a `FATAL_ERROR` status snapshot, then exits the process.

Configuration is schema-driven and uses one runtime source of truth:

```bash
curl http://127.0.0.1:5174/api/config
curl http://127.0.0.1:5174/api/config/schema
```

The React settings page renders adjustable parameters from
`/api/config/schema`, then saves through `PUT /api/config`. The backend applies
the same config to `app.state.config`, capture, runtime, executors, and hardware
box construction so the UI and running program do not drift.

Frontend URL handling is centralized in `web/src/api.ts`. During development
Vite proxies `/api`, `/healthz`, and `/ws` to the backend. For split-host
deployments set:

```bash
VITE_NOVASIGHT_API_BASE=http://<jetson-ip>:5174
VITE_NOVASIGHT_WS_BASE=ws://<jetson-ip>:5174
```

Local license-key management is available through:

```bash
curl http://127.0.0.1:5174/api/license
curl -X POST http://127.0.0.1:5174/api/license/activate \
  -H "Content-Type: application/json" \
  -d '{"key":"NOVASIGHT-TEST-MAX-ACCESS-2026"}'
```

Operational APIs are blocked until a valid license is activated. The built-in
test key grants `test_max` permissions for Jetson bring-up. Production licenses
use signed `NS1.<payload>.<signature>` tokens with created time, activation time,
duration, tier, and feature permissions; plaintext keys are never returned by the
API.

## Jetson `deepstream_nvinfer` Mainline

The current GPU-image production candidate keeps decoded frames in NVMM and
lets DeepStream own TensorRT scheduling:

```text
GC553G2/V4L2 -> nvv4l2decoder -> nvvidconv ROI/resize
-> nvstreammux batch=1 -> nvinfer FP16
-> C++ YOLO parser -> DeepStream NMS -> NvDsObjectMeta
-> DetectionBatchMailbox capacity=1 -> DetectionBatch
-> Tracker/Selector -> isolated control algorithm
-> V2 MouseCommandExecutor direct send (legacy algorithms may use Scheduler)
-> kmNet
```

No image reaches appsink, NumPy, OpenCV, or Python postprocessing. The C++
parser and DeepStream NMS still perform small metadata work on CPU; this path
guarantees zero CPU image round trip, not zero CPU instructions.

Required config values:

```yaml
capture:
  backend: deepstream_nvinfer
  memory: nvmm
  latest_only: true
  appsink_max_buffers: 1
  queue_leaky: downstream
inference:
  backend: deepstream_nvinfer
  device: cuda
  require_gpu: true
  allow_cpu_fallback: false
  deepstream_io_mode: 2
  deepstream_batched_push_timeout_us: 0
  deepstream_parser_library: build/deepstream-parser/libnovasight_parser.so
runtime:
  freshness_threshold_ms: 55
  drop_stale_batches: true
  consume_latest_only: true
```

Start with the example config:

```bash
cp config/novasight.example.yaml config/novasight.yaml
python3 -m novasight --host 0.0.0.0 --port 5174
```

The example config starts the active TensorRT deployment through nvinfer. kmNet
auto-connect runs independently.
If no active model exists, the API remains available and the runtime reports an
explicit model-not-loaded startup reason instead of entering a false running state.

Build and run the 60-second hardware gate before enabling control:

```bash
scripts/setup_jetson.sh --pyds-wheel /path/to/pyds.whl
scripts/verify_deepstream_60s.py --seconds 60
```

Jetson setup builds `libnovasight_parser.so` by default. If the build artifact is
later removed, the DeepStream backend rebuilds it automatically on the next start.

Detailed contracts and current measurement gaps are in
`docs/novasight-deepstream-object-mainline.md`.

### Legacy CPU-bridge diagnostics

Run a 60-second capture smoke on Jetson:

```bash
python3 -m novasight capture-smoke --device /dev/video0 --seconds 60
```

For the full temporary runtime acceptance gate, run the live runtime smoke on
Jetson with either the active model already configured or an explicit TensorRT
engine:

```bash
python3 -m novasight doctor gst-cpu-latest-smoke \
  --device /dev/video0 \
  --seconds 60 \
  --tensorrt-engine data/models/combat/default/model.engine \
  --input-shape 1x3x640x640 \
  --dtype fp16 \
  --classes target \
  --report-json /tmp/novasight-gst-cpu-latest-smoke.json
```

Then re-validate the saved evidence with:

```bash
python3 -m novasight doctor gst-cpu-latest-smoke-report \
  --report-json /tmp/novasight-gst-cpu-latest-smoke.json
```

During the live run the command prints `metric_sample:` lines at the same
cadence used for the saved report, including capture FPS, broker counters,
host-frame copy cost, preprocess/H2D/inference timings, stale drops, control
observation FPS, and RSS memory.

The report must use `check: "gst-cpu-latest-smoke"` and include runtime
evidence for the latest-only pipeline, TensorRT GPU execution, metrics,
DetectionBatch timing fields, 60s duration, memory growth, and a
`metric_samples` list captured at roughly one-second cadence:

```json
{
  "schema_version": 1,
  "check": "gst-cpu-latest-smoke",
  "accepted": true,
  "exit_code": 0,
  "parameters": {
    "device": "/dev/video0",
    "seconds": 60,
    "tensorrt_engine": "data/models/combat/default/model.engine",
    "max_memory_growth_mb": 64
  },
  "evidence": {
    "capture_backend": "gst_cpu_latest:nvmm-mjpg-iomode2",
    "capture_memory": "system",
    "capture_fps": 119.4,
    "published_frames": 7164,
    "overwritten_frames": 420,
    "acquired_frames": 6744,
    "latest_frame_broker_max_pending_depth": 1,
    "latest_frame_age_ms": 4.5,
    "appsink_caps": "video/x-raw,format=BGRx,width=640,height=640",
    "actual_pipeline_string": "v4l2src ... queue max-size-buffers=1 ... leaky=downstream ... appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false",
    "inference_selected": "tensorrt",
    "inference_device": "cuda",
    "inference_require_gpu": true,
    "inference_allow_cpu_fallback": false,
    "tensorrt_engine": "data/models/combat/default/model.engine",
    "preprocess_backend": "cpu",
    "preprocess_ms": 1.3,
    "h2d_ms": 0.4,
    "host_frame_copy_ms": 0.2,
    "inference_ms": 5.2,
    "postprocess_ms": 0.1,
    "batch_age_ms": 11.8,
    "stale_drop_count": 3,
    "control_observe_fps": 58.0,
    "freshness_gate_rejected": false,
    "duration_s": 60.2,
    "memory_rss_start_mb": 312.0,
    "memory_rss_end_mb": 328.0,
    "memory_growth_mb": 16.0,
    "detection_batch": {
      "frame_id": 7164,
      "capture_ts_ns": 1000000,
      "inference_start_ts_ns": 1004000,
      "inference_end_ts_ns": 1009000,
      "control_now_ts_ns": 1020000,
      "model_input_size": [640, 640],
      "coordinate_space": "roi"
    }
  },
  "metric_samples": [
    {
      "elapsed_s": 1.0,
      "memory_rss_mb": 316.0,
      "capture_fps": 119.4,
      "published_frames": 7164,
      "latest_frame_broker_max_pending_depth": 1,
      "latest_frame_age_ms": 4.5,
      "preprocess_ms": 1.3,
      "h2d_ms": 0.4,
      "host_frame_copy_ms": 0.2,
      "inference_ms": 5.2,
      "batch_age_ms": 11.8,
      "stale_drop_count": 3,
      "control_observe_fps": 58.0,
      "inference_selected": "tensorrt",
      "inference_device": "cuda",
      "preprocess_backend": "cpu"
    }
  ]
}
```

Verify latest-only from status:

```bash
curl http://127.0.0.1:5174/api/runtime/state | python3 -m json.tool
```

Check these fields:

- `pipeline.latest_frame_broker.max_pending_depth == 1`
- `pipeline.latest_frame_broker.overwritten_frames` increases under overload
- `statistics.stale_drop_count` is present
- `statistics.batch_age_ms`, `statistics.preprocess_ms`, and `statistics.h2d_ms`
  are visible after inference
- `inference.execution_backend == "tensorrt"` or `inference.selected == "tensorrt"`
- `inference.allow_cpu_fallback == false`

The generated GStreamer CPU bridge uses:

```text
queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream
appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false
```

The preferred appsink caps are `video/x-raw,format=BGRx`, with fallback
candidates for `RGBA`, `RGB`, `BGR`, `NV12`, and `I420`. `appsink_caps` and
`actual_pipeline_string` are exposed in capture statistics after frames arrive.
Record the negotiated caps from Jetson smoke output before changing pixel
formats. CPU preprocess preserves aspect ratio with letterbox padding if a host
frame reaches TensorRT with non-matching model/input aspect ratio. TensorRT
host input/output buffers prefer CUDA pinned host memory when the runtime
exposes `cudaHostAlloc`/`cudaMallocHost` plus `cudaFreeHost`; otherwise they
fall back to ordinary NumPy host buffers and still use async H2D/D2H copies.

## Jetson Camera Diagnostics

Jetson capture uses the system GStreamer stack through `GstAppSink`. Use the
system Python environment and do not install pip OpenCV or pip `gi`:

```bash
cd ~/NovaSight
scripts/setup_jetson.sh
```

Build only the Jetson NVMM TensorRT preprocess bridge after pulling new code:

```bash
scripts/build_jetson_preprocess.sh --preflight
source build/jetson-native/novasight-native-env.sh
```

If DeepStream Python tensor-meta support is needed, pass the matching NVIDIA
`pyds` wheel. For DeepStream 7.1 on Jetson Python 3.10, use the
`pyds-1.2.0-cp310-cp310-linux_aarch64.whl` wheel from NVIDIA's
`deepstream_python_apps` v1.2.0 release:

```bash
scripts/setup_jetson.sh \
  --pyds-wheel ~/Downloads/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
```

Verify the system-provided GStreamer bindings and NVIDIA elements:

```bash
python3 - <<'PY'
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp
Gst.init(None)
print("Gst/GstApp ok")
PY

gst-inspect-1.0 nvvidconv
gst-inspect-1.0 nvv4l2decoder
```

Inspect `/dev/video0` capabilities:

```bash
python3 -m novasight doctor camera --device /dev/video0
```

For diagnostics, the optional `gst_cpu_latest` CPU bridge prefers:

```text
MJPG 1920x1080 @ 120
-> nvv4l2decoder
-> nvvidconv
-> video/x-raw,format=BGRx,width=<model_w>,height=<model_h>
-> appsink max-buffers=1 drop=true sync=false
```

The matching standalone GStreamer smoke command is:

```bash
gst-launch-1.0 -v \
  v4l2src device=/dev/video0 io-mode=2 ! \
  'image/jpeg,width=1920,height=1080,framerate=120/1' ! \
  queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! \
  jpegparse ! \
  nvv4l2decoder mjpeg=1 ! \
  nvvidconv ! \
  'video/x-raw,format=BGRx,width=320,height=320' ! \
  queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! \
  appsink sync=false max-buffers=1 drop=true
```

The production `nvmm_latest` path requires `capture.backend=nvmm_latest`,
`capture.memory=nvmm`, `preprocess.backend=cuda`, and
`inference.backend=nvmm_latest`.

Run a short real capture smoke test:

```bash
python3 -m novasight capture-smoke --device /dev/video0 --seconds 5
```

The smoke test reports selected profile, backend label, observed FPS,
frame period, capture wait, dropped frames, recoveries, and recent errors.

Start the backend so the React workbench can switch capture profiles and pull
the MJPEG preview stream:

```bash
cp config/novasight.example.yaml config/novasight.yaml
python3 -m novasight --host 0.0.0.0 --port 5174
```

Verify the capture API from the Jetson shell:

```bash
curl -X POST http://127.0.0.1:5174/api/license/activate \
  -H "Content-Type: application/json" \
  -d '{"key":"NOVASIGHT-TEST-MAX-ACCESS-2026"}'
curl http://127.0.0.1:5174/api/capture/state
curl "http://127.0.0.1:5174/api/capture/capabilities?device=/dev/video0"
curl -I http://127.0.0.1:5174/api/capture/stream.mjpg
```

Start the frontend in a second shell:

```bash
pnpm --dir web install
pnpm --dir web dev --host 0.0.0.0
```

Open `http://<jetson-ip>:5173`, switch to `采集`, click `刷新能力`, then apply
one of the listed profiles such as `MJPG 1920x1080 @ 120`. The preview panel
uses `/api/capture/stream.mjpg` and reconnects after every successful profile
switch. If a profile fails, the backend keeps the last healthy capture source
when possible and reports the error through `/api/capture/state`.

The default runtime config path is `config/novasight.yaml`. Keep model assets
and SQLite state under `data/`, which is ignored by git.

Hardware box config lives under `hardware` in `config/novasight.yaml`.
Use `kind: kmnet` with `host` and `port`, or `kind: makcu` with `serial_port`.
Local Win32 hooks and local screenshot paths are intentionally absent; trigger
state must come from hardware box input packets.
