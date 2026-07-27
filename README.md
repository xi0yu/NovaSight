# NovaSight

NovaSight is a Jetson-first realtime vision application. The maintained product
path is the Rust backend plus the React Web UI; DeepStream, TensorRT, target
selection, prediction, control, and kmNet output are owned by the Rust process.

## Repository layout

```text
NovaSight/
├── Cargo.toml                 # Root Rust workspace
├── apps/
│   ├── novasightd/            # Backend daemon and HTTP/WebSocket API
│   └── novasightctl/          # Local command-line client
├── crates/                    # Rust domain and platform modules
├── native/                    # Narrow C/C++ seams for NVIDIA SDKs
├── config/
│   └── novasightd.example.yaml
└── web/                       # React Web UI
```

Cargo is intentionally rooted at the repository top level. Backend commands no
longer require entering a `rust/` directory or passing `--manifest-path`.

## Jetson development run

Install the JetPack/DeepStream development packages and Rust toolchain first.
From the repository root, create the user configuration once:

```bash
mkdir -p ~/.config/novasight
cp config/novasightd.example.yaml ~/.config/novasight/novasightd.yaml
```

Review capture, model, ROI, and kmNet values in that file, then run the real
Jetson backend in Cargo's development profile:

```bash
cargo run -p novasightd --features deepstream -- \
  --config ~/.config/novasight/novasightd.yaml
```

Live DeepStream perception is the normal Jetson path. It does not need a
`--live-perception` switch, a preparatory build script, or Python. Cargo builds
the native DeepStream/TensorRT integration required by the daemon.

To validate configuration and the HTTP API without opening capture or hardware:

```bash
cargo run -p novasightd -- \
  --config config/novasightd.example.yaml \
  --dry-run
```

The backend listens on `127.0.0.1:5174` by default. Debug builds accept the
built-in development license; release builds require the configured production
public key.

## Web development run

In a second terminal, from the same repository root:

```bash
pnpm --dir web install
pnpm --dir web dev --host 0.0.0.0
```

Open `http://<jetson-ip>:5173`. Vite proxies `/api`, `/healthz`, and `/ws` to
the backend on port 5174. When the Web UI and backend are on different hosts,
set `VITE_NOVASIGHT_API_BASE` and `VITE_NOVASIGHT_WS_BASE` explicitly.

## Development build checks

```bash
cargo check --workspace
pnpm --dir web typecheck
pnpm --dir web build
```

Local runtime databases, models, logs, generated native artifacts, Cargo
targets, and Web build output are ignored and must not be committed.
