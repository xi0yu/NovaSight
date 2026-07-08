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
