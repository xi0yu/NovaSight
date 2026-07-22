# NovaSight

NovaSight is a Jetson-first realtime vision console rebuilt from the old jetcam
prototype with cleaner model, configuration, plugin, executor, API, and UI
boundaries.

## Current Project Authority

Use `PROJECT_HEALTH_AUDIT.md` as the current cleanup and over-design ledger.
Older plans under `docs/superpowers/` are archive material unless promoted by a
current authority document.

## Development

The Rust daemon is the canonical backend. The Python package remains in-tree
as a rollback path and for allowlisted offline model jobs.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Start the Rust backend API in explicit no-hardware mode:

```bash
cargo run --manifest-path rust/Cargo.toml -p novasightd -- \
  --config rust/config/novasightd.example.yaml --dry-run
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
cargo test --manifest-path rust/Cargo.toml --workspace
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
pytest -q
pnpm --dir web typecheck
pnpm --dir web build
```

The rollback backend remains directly runnable with
`python3 -m novasight --host 127.0.0.1 --port 5174`.

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
test key grants `test_max` permissions only in `--dry-run`. Hardware production
mode requires an RSA public key in `NOVASIGHT_LICENSE_PUBLIC_KEY`; there is no
environment override that enables the test key in production. Production
licenses use signed `NS1.<payload>.<signature>` tokens with created time,
duration, tier, and feature permissions. Rust re-verifies signed claims after
restart, enforces feature permissions server-side, and never returns the
activation token. The 0600 license file stores its hash plus the signed
payload/signature proof required for offline restart verification; it does not
store a plaintext `key` field. Offline license duration starts at the signed
creation time, so deleting and reactivating the same token cannot reset its
expiry.

## Jetson `deepstream_nvinfer` Mainline

The current GPU-image production candidate keeps decoded frames in NVMM and
lets DeepStream own TensorRT scheduling:

```text
GC553G2/V4L2 MJPEG -> nvv4l2decoder -> nvvidconv ROI/resize
                  or NV12/YUYV raw -> nvvidconv ROI/resize
-> nvstreammux batch=1 -> nvinfer FP16
-> C++ YOLO parser -> DeepStream NMS -> NvDsObjectMeta
-> DetectionBatchMailbox capacity=1 -> DetectionBatch
-> Tracker/Selector -> isolated control algorithm
-> V2 capacity-one latest-replace delivery (legacy algorithms use configured delivery)
-> MouseCommandExecutor safety validation
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
scripts/build_deepstream_parser.sh
scripts/build_deepstream_bridge.sh
cargo build --manifest-path rust/Cargo.toml -p novasightd --release --features deepstream
cargo build --manifest-path rust/Cargo.toml -p novasightctl --release
scripts/stage_jetson_release.sh
sudo build/jetson-release/scripts/install_jetson_release.sh
sudo /opt/novasight/current/bin/novasightd \
  --config /etc/novasight/novasight.yaml --check
```

The staged `bin/../lib` layout matches the daemon's production RUNPATH and the
`/opt/novasight/{bin,lib}` systemd layout. It also carries the retained Python
package used only by the daemon-owned kmNet helper and allowlisted offline model
job, so neither path depends on a source checkout or editable install. The
production template uses absolute `/opt/novasight` and `/var/lib/novasight`
paths; the development example remains repository-relative. Running the
unstaged Cargo binary is not a production check because the native bridge is
built outside its RUNPATH.

The installer writes each build to `/opt/novasight/releases/<release-id>` and
only then atomically switches `/opt/novasight/current`. It never overwrites an
existing `/etc/novasight/novasight.yaml`; the new template is written as
`novasight.yaml.dist` for an explicit operator merge. It reloads systemd but
does not start the service, so license/config/model checks remain a deliberate
deployment gate. Retained version directories keep rollback recoverable and
explicit; same-ID installs fail closed instead of mutating an immutable release.

The example config starts the active TensorRT deployment through nvinfer. kmNet
auto-connect runs independently.
If no active model exists, the API remains available and the runtime reports an
explicit model-not-loaded startup reason instead of entering a false running state.

Build and run the 60-second hardware gate before enabling control:

```bash
scripts/setup_jetson.sh --pyds-wheel /path/to/pyds.whl
scripts/verify_deepstream_60s.py --seconds 60
```

Jetson setup builds `libnovasight_parser.so` by default. The Rust daemon never
compiles native code at runtime: a missing parser, bridge, plugin, or ABI mismatch
makes `--check` and the systemd `ExecStartPre` fail closed. Put the production
public key at `/etc/novasight/license-public.pem` and reference it from
`/etc/novasight/novasight.env` with
`NOVASIGHT_LICENSE_PUBLIC_KEY_FILE=/etc/novasight/license-public.pem`; the unit
reads this environment file before preflight. `NOVASIGHT_LICENSE_PUBLIC_KEY`
remains available for environments that can safely supply the PEM text directly.

Detailed contracts and current measurement gaps are in
`docs/novasight-deepstream-object-mainline.md`.

### Rust-owned TensorRT path

The daemon also has an explicit `rust_tensor_rt` backend. It keeps capture,
decode, crop/resize, and batching in DeepStream/NVMM, but replaces `nvinfer`
with a latest-only Rust worker that owns the CUDA preprocess allocation,
TensorRT execution context/stream, host-output lifetime, YOLO decode/NMS, and
`DetectionBatch` admission:

```text
nvstreammux -> identity pad probe -> FrameLease
-> CUDA RGB NCHW preprocess -> TensorRT enqueueV3 + stream synchronization
-> bounded Rust decoder/NMS -> DetectionBatch -> existing control runtime
```

Build the three narrow native seams on the Jetson, then build the daemon with
the feature enabled:

```bash
scripts/build_deepstream_bridge.sh
scripts/build_jetson_preprocess.sh
scripts/build_tensorrt_runtime.sh
cargo build --manifest-path rust/Cargo.toml -p novasightd --release --features tensorrt
```

Select it without changing capture or downstream control configuration:

```yaml
inference:
  backend: rust_tensor_rt
```

Startup validates the active engine checksum and manifest, CUDA preprocess
semantics, named TensorRT output shape, and decoder contract before reporting
ready. The current Rust decoder intentionally supports raw YOLO and decoded
`[N,6]` boxes; EfficientNMS and Rockchip three-head manifests fail closed and
can continue using `deepstream_nvinfer`. The existing Python and nvinfer paths
remain available as compatibility and rollback routes.

### Rust control-plane contract

`novasightd` serves the same Axum router over TCP and its Unix control socket.
The production router exposes the persisted Rust configuration contract at
`GET /api/config/schema` and the supervisor-owned capture projection at
`GET /api/capture/state`; neither endpoint is backed by the replay-only
compatibility shim. Schema fields are restart-required unless the Rust daemon
has an explicit hot-apply owner. `control.output_enabled` is the first such
field: that safety gate is persisted and applied by
one typed supervisor command without disconnecting the device or stopping
capture, inference, targeting, or control calculation. Pausing drains the
latest command, and reopening accepts only a newer source generation, so a
command calculated during the pause cannot leak afterward.

`GET /api/runtime/state`, `GET /api/v1/status`, `/ws/status`, and
`novasightctl status` all read the same immutable runtime snapshot. The
snapshot restores the active model deployment from SQLite during daemon boot
and updates it only after a model switch has committed successfully. A catalog
read failure is reported separately from the valid “no active model” state.

Local administration uses that same router rather than a second in-process
control implementation:

```bash
novasightctl license activate --key-file /secure/path/license.key
novasightctl model projects
novasightctl model publish 1 4 --parser-preset auto
novasightctl capture capabilities --device /dev/video0
novasightctl device status
novasightctl device move 4 -2
novasightctl crosshair status
novasightctl crosshair learn
novasightctl motion status
novasightctl motion profiles
novasightctl motion builtin
```

License activation reads the key from a file so the secret is not exposed in
the process argument list. Model publish/rollback and the single raw diagnostic
move retain the API's compensated activation and supervisor ownership rules.
On Linux/Jetson, capture capability discovery calls the V4L2 enumeration
ioctls directly and returns only kernel-reported discrete format, size, and
frame-rate tuples; it does not spawn or parse `v4l2-ctl`.

When `crosshair.enabled` is true, the production DeepStream graph adds an
independent latest-only center crop that stays in NVMM until `nvjpegenc`. A
bounded Rust observer learns and persists the template under
`paths.data_dir/runtime/crosshair/template.json`; confirmed, fresh observations
replace the geometric center for both target selection and control only when
`crosshair.use_for_control` is also true. `GET /api/crosshair`,
`POST /api/crosshair/learn`, `DELETE /api/crosshair/template`, the Studio, and
`novasightctl crosshair` all read or mutate that same daemon-owned state.

Human trajectory profiles are likewise daemon-owned. The Studio and
`novasightctl motion` use the same `/api/motion/*` contract to create sessions,
append bounded high-frequency samples, train version-4 profiles, and atomically
switch the active immutable profile. Rust separates reaction delay from movement
time, rejects low-quality paths, fits Fitts timing plus normalized speed and
two-dimensional side curves, and stores them below
`paths.data_dir/motion/{sessions,profiles}` using atomic replacement. The active
profile shapes floating-point demand inside the control worker; FAR/NEAR
per-axis limits and count quantization remain authoritative afterward. Disabling
the profile is an exact return to static control, and the retained Python
implementation remains available only as rollback/reference.

Independent recoil is also owned by the Rust output scheduler. It integrates
`control.recoil` rates at the configured 1–10 ms device cadence, requires a
fresh left-button state and a fresh target observation, normalizes vertical
error by a bounded three-sample target-height median, and stops immediately on
release, target loss, or stale input. Tracking and recoil meet at one Y-axis
arbiter: two downward demands never stack, while upward tracking can still
cancel overshoot. The same immutable runtime snapshot publishes recoil state,
gate, rate, emitted counts, residual, observation age, and source generation to
the Studio.

The same snapshot carries the complete Rust dual-phase decision used by the
device lane: observed and predicted error, robust velocity window, confidence,
prediction caps, full correction counts, shaped floating-point demand, integer
command, and quantizer residual. Studio and CLI diagnostics therefore inspect
the actual daemon decision rather than recomputing display-only values.

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
