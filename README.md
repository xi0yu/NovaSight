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

The default runtime config path is `config/novasight.yaml`. Keep model assets
and SQLite state under `data/`, which is ignored by git.
