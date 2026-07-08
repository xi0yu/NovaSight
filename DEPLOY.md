# NovaSight Deployment

## Target

NovaSight is deployed as a Jetson-side runtime service. The Jetson owns capture,
DeepStream inference, target selection, control, and kmNet/HID output. Studio
clients connect over REST and WebSocket.

## Layout

- Application: `/opt/novasight`
- Runtime config: `/etc/novasight/novasight.yaml`
- Runtime data: `/var/lib/novasight`
- Runtime logs: `/var/log/novasight/novasight.log`
- Instance lock: `/run/novasight/instance.lock`
- systemd unit: `deploy/novasight.service`
- Optional nvtracker config: `deploy/deepstream-tracker-iou.yml`

## Install

```bash
sudo mkdir -p /opt/novasight /etc/novasight /var/lib/novasight
sudo cp -a . /opt/novasight
cd /opt/novasight
uv sync
sudo cp config/novasight.yaml /etc/novasight/novasight.yaml
sudo cp deploy/deepstream-tracker-iou.yml /etc/novasight/deepstream-tracker-iou.yml
sudo cp deploy/novasight.service /etc/systemd/system/novasight.service
sudo systemctl daemon-reload
sudo systemctl enable novasight
```

Set the production log directory in `/etc/novasight/novasight.yaml`:

```yaml
logging:
  level: INFO
  dir: /var/log/novasight
```

For Jetson DeepStream runs, point the runtime at the generated model files and
the deployed tracker config:

```yaml
inference:
  backend: deepstream
  deepstream_manifest_path: /var/lib/novasight/models/<model>/model.manifest.json
  deepstream_config_path: /var/lib/novasight/models/<model>/deepstream.ini
  deepstream_tracker_config_path: /etc/novasight/deepstream-tracker-iou.yml
```

## Jetson Prerequisites

- JetPack with DeepStream and TensorRT installed.
- `v4l2-ctl`, `gst-launch-1.0`, `tegrastats`, and the camera driver visible to
  the service user.
- kmNet/HID network access from the Jetson to the configured hardware endpoint.
- Model assets under `/var/lib/novasight/models` with generated manifest and
  DeepStream config files.
- Optional `nvtracker` low-level config deployed from
  `deploy/deepstream-tracker-iou.yml` or replaced by a Jetson-validated tracker
  config for the selected DeepStream tracker library.

Check the host before enabling the service:

```bash
v4l2-ctl --list-devices
gst-inspect-1.0 nvarguscamerasrc
gst-inspect-1.0 nvinfer
which tegrastats
```

## Start And Stop

```bash
sudo systemctl start novasight
sudo systemctl status novasight
sudo journalctl -u novasight -f
sudo systemctl stop novasight
```

## Watchdog

The service uses `Type=notify` and `WatchdogSec=10`. NovaSight sends `READY=1`
on app startup and sends `WATCHDOG=1` at half of `WATCHDOG_USEC` while the API
process is alive.

Logs are written to journald through `StandardOutput=journal` and
`StandardError=journal`. The runtime also writes rotating local logs to
`/var/log/novasight/novasight.log`; `deploy/novasight.service` declares
`LogsDirectory=novasight` so systemd creates that directory before startup.

## Hardware Checks

```bash
novasight doctor camera --device /dev/video0
uv run python scripts/nvmm_path_check.py --device /dev/video0 --resolution 1920x1080 --formats MJPG,NV12 --fps 60 --output /var/lib/novasight/nvmm-check.json
novasight doctor deepstream-smoke --manifest /var/lib/novasight/models/<model>/model.manifest.json --nvinfer-config /var/lib/novasight/models/<model>/deepstream.ini
uv run python scripts/render_detection_overlay.py --image /var/lib/novasight/evidence/frame.jpg --detections-json /var/lib/novasight/evidence/detection-batch.json --output /var/lib/novasight/evidence/overlay.jpg
novasight doctor kmnet --km-host <host> --km-port <port> --km-uuid <uuid>
```

## Runtime API Checks

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/api/device/capabilities
curl http://127.0.0.1:8000/api/v1/system/status
```

Protected API and WebSocket endpoints require a valid license.

## Performance Run

Run the Phase 9 telemetry probe for one hour after the runtime has been started
with the target capture source, model deployment, and HID executor:

```bash
uv run python tests/perf_test.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint /api/runtime/state \
  --duration-s 3600 \
  --interval-s 0.05 \
  --max-p95-ms 20 \
  --target-fps 60 \
  --fps-tolerance-pct 1 \
  --output /var/lib/novasight/perf-1h.json
```

Review the output:

- `e2e_latency_ms.p95` must be below `20.0`; below `15.0` is the preferred
  production target.
- `capture_fps.p50` should stay within `+-1%` of the configured target FPS.
- `covered_frames` should increase during the run.
- `dropped_frames` should not grow during a stable run.
- `stale_detection_samples` should remain near zero after warm-up.
- `passed` must be `true`; failed gates are listed under `gates`.

Keep the generated JSON beside the deployment logs for regression comparison.

## Fault Injection

Run these checks with HID output enabled only when the physical rig is safe:

```bash
# Camera disconnect: unplug the capture device, then watch state and logs.
uv run python scripts/fault_injection.py capture-loss --duration-s 30 --interval-s 0.5
journalctl -u novasight -f

# Inference overload: add load, then verify old frames are dropped instead of
# building unbounded latency.
uv run python scripts/fault_injection.py overload \
  --duration-s 120 \
  --interval-s 0.05 \
  --max-p95-ms 20 \
  --target-fps 60 \
  --fps-tolerance-pct 1

# Bad model import/build: submit an invalid ONNX and confirm the build job fails
# without stopping the API service.
uv run python scripts/fault_injection.py bad-model --model-id bad_model
```

Expected behavior:

- Capture loss must stop control/HID output and report an unavailable or
  degraded state.
- Runtime must continue serving health, status, and system endpoints.
- Model conversion errors must stay isolated to the model job/API response.

## Production Acceptance

Run the deployment self-check after installing the service file. Use
`--skip-api` before the API is running, then run it again with the service
started and a license header if the system API is protected:

```bash
uv run python scripts/deployment_check.py --skip-api
uv run python scripts/deployment_check.py --header "Authorization: Bearer <token>"
```

- `systemctl start novasight`, `systemctl stop novasight`, and restart on
  failure work through systemd.
- Watchdog notifications are active with `WatchdogSec=10`.
- A second service process cannot acquire `/run/novasight/instance.lock`.
- A 1-hour performance run satisfies the p95 latency and FPS gates above before
  a 24-hour soak is started.
- The 24-hour soak has no service crash, unbounded memory growth, or sustained
  stale detections.

Check the active instance lock through the system API:

```bash
curl -s http://127.0.0.1:8000/api/v1/system | jq .instance_lock
```

During the soak, record process memory from the system API:

```bash
watch -n 60 'curl -s http://127.0.0.1:8000/api/v1/system | jq .process'
```
