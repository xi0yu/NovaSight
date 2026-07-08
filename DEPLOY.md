# NovaSight Deployment

## Target

NovaSight is deployed as a Jetson-side runtime service. The Jetson owns capture,
DeepStream inference, target selection, control, and kmNet/HID output. Studio
clients connect over REST and WebSocket.

## Layout

- Application: `/opt/novasight`
- Runtime config: `/etc/novasight/novasight.yaml`
- Runtime data: `/var/lib/novasight`
- Instance lock: `/run/novasight/instance.lock`
- systemd unit: `deploy/novasight.service`

## Install

```bash
sudo mkdir -p /opt/novasight /etc/novasight /var/lib/novasight
sudo cp -a . /opt/novasight
cd /opt/novasight
uv sync
sudo cp config/novasight.yaml /etc/novasight/novasight.yaml
sudo cp deploy/novasight.service /etc/systemd/system/novasight.service
sudo systemctl daemon-reload
sudo systemctl enable novasight
```

## Jetson Prerequisites

- JetPack with DeepStream and TensorRT installed.
- `v4l2-ctl`, `gst-launch-1.0`, `tegrastats`, and the camera driver visible to
  the service user.
- kmNet/HID network access from the Jetson to the configured hardware endpoint.
- Model assets under `/var/lib/novasight/models` with generated manifest and
  DeepStream config files.

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

## Hardware Checks

```bash
novasight doctor camera --device /dev/video0
novasight doctor deepstream-smoke --manifest /var/lib/novasight/models/<model>/model.manifest.json --nvinfer-config /var/lib/novasight/models/<model>/deepstream.ini
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
  --output /var/lib/novasight/perf-1h.json
```

Review the output:

- `e2e_latency_ms.p95` must be below `20.0`; below `15.0` is the preferred
  production target.
- `capture_fps.p50` should stay within `+-1%` of the configured target FPS.
- `dropped_frames` should not grow during a stable run.
- `stale_detection_samples` should remain near zero after warm-up.

Keep the generated JSON beside the deployment logs for regression comparison.

## Fault Injection

Run these checks with HID output enabled only when the physical rig is safe:

```bash
# Camera disconnect: unplug the capture device, then watch state and logs.
watch -n 0.5 'curl -s http://127.0.0.1:8000/api/runtime/state'
journalctl -u novasight -f

# Inference overload: add load, then verify old frames are dropped instead of
# building unbounded latency.
stress-ng --cpu 4 --timeout 120s
uv run python tests/perf_test.py --duration-s 120 --interval-s 0.05

# Bad model import/build: submit an invalid ONNX and confirm the build job fails
# without stopping the API service.
curl -X POST http://127.0.0.1:8000/api/v1/models/import \
  -H 'Content-Type: application/json' \
  -d '{"source_path":"/tmp/not-a-model.onnx","model_id":"bad_model"}'
```

Expected behavior:

- Capture loss must stop control/HID output and report an unavailable or
  degraded state.
- Runtime must continue serving health, status, and system endpoints.
- Model conversion errors must stay isolated to the model job/API response.

## Production Acceptance

- `systemctl start novasight`, `systemctl stop novasight`, and restart on
  failure work through systemd.
- Watchdog notifications are active with `WatchdogSec=10`.
- A second service process cannot acquire `/run/novasight/instance.lock`.
- A 1-hour performance run satisfies the p95 latency and FPS gates above before
  a 24-hour soak is started.
- The 24-hour soak has no service crash, unbounded memory growth, or sustained
  stale detections.
