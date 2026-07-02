# NovaSight

NovaSight is a Jetson-first realtime vision console rebuilt from the old jetcam
prototype with cleaner model, configuration, plugin, executor, API, and UI
boundaries.

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

## Jetson Camera Diagnostics

Inspect `/dev/video0` capabilities:

```bash
python3 -m novasight doctor camera --device /dev/video0
```

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
one of the listed profiles such as `MJPG 1920x1080 @ 240`. The preview panel
uses `/api/capture/stream.mjpg` and reconnects after every successful profile
switch. If a profile fails, the backend keeps the last healthy capture source
when possible and reports the error through `/api/capture/state`.

The default runtime config path is `config/novasight.yaml`. Keep model assets
and SQLite state under `data/`, which is ignored by git.

Hardware box config lives under `hardware` in `config/novasight.yaml`.
Use `kind: kmnet` with `host` and `port`, or `kind: makcu` with `serial_port`.
Local Win32 hooks and local screenshot paths are intentionally absent; trigger
state must come from hardware box input packets.
