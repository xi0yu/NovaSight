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
- Instance guard: automatic Linux abstract socket (no filesystem entry)
- systemd unit: `deploy/novasight.service`
- Optional nvtracker config: `deploy/deepstream-tracker-iou.yml`

## Install

```bash
# Run on the Jetson host with the DeepStream SDK and Rust toolchain installed.
cargo build --release -p novasightd --features deepstream
cargo build --release -p novasightctl
sudo mkdir -p /opt/novasight/current/bin /etc/novasight /var/lib/novasight
sudo install -m 0755 target/release/novasightd /opt/novasight/current/bin/novasightd
sudo install -m 0755 target/release/novasightctl /opt/novasight/current/bin/novasightctl
sudo install -m 0640 deploy/novasight.production.yaml /etc/novasight/novasight.yaml
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
  backend: deepstream_nvinfer
  deepstream_manifest_path: /var/lib/novasight/models/<model>/model.manifest.json
  deepstream_parser_library: auto
  deepstream_nvinfer_config: /var/lib/novasight/models/<model>/deepstream.ini
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
v4l2-ctl --list-formats-ext --device /dev/video0
gst-inspect-1.0 nvinfer
/opt/novasight/current/bin/novasightd --config /etc/novasight/novasight.yaml --check
/opt/novasight/current/bin/novasightctl --help
```

## Runtime API Checks

```bash
curl http://127.0.0.1:5174/healthz
curl http://127.0.0.1:5174/api/device/capabilities
curl http://127.0.0.1:5174/api/v1/system/status
```

Protected API and WebSocket endpoints require a valid license.

## Fault Injection

Run these checks with HID output enabled only when the physical rig is safe.
Use the Rust daemon state endpoints and system logs as evidence:

- Capture loss must stop control/HID output and report an unavailable or
  degraded state.
- Runtime must continue serving health, status, and system endpoints.
- Model conversion errors must stay isolated to the model job/API response.

## Production Acceptance

Run the Rust daemon preflight after installing the service file, then check the
health endpoint after the service is running:

```bash
/opt/novasight/current/bin/novasightd --config /etc/novasight/novasight.yaml --check
curl -fsS http://127.0.0.1:5174/healthz
```

- `systemctl start novasight`, `systemctl stop novasight`, and restart on
  failure work through systemd.
- Watchdog notifications are active with `WatchdogSec=10`.
- A second service process cannot acquire the kernel-owned instance guard.
- Runtime telemetry remains fresh and bounded before a 24-hour soak is started.
- The 24-hour soak has no service crash, unbounded memory growth, or sustained
  stale detections.

During the soak, record process memory from the system API:

```bash
watch -n 60 'curl -s http://127.0.0.1:5174/api/v1/system | jq .process'
```
