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
├── .config/
│   └── novasight.yaml          # Generated local runtime configuration (Git-ignored)
└── web/                       # React Web UI
```

Cargo is intentionally rooted at the repository top level. Backend commands no
longer require entering a `rust/` directory or passing `--manifest-path`.

## Jetson development run

Install the JetPack/DeepStream development packages and Rust toolchain first.
On first start NovaSight creates the Git-ignored `.config/novasight.yaml` from
its bundled Jetson baseline and continues startup. Build the real Jetson
backend from the repository root, then start it through the daemon binary:

```bash
cargo build -p novasightd --features deepstream
target/debug/novasightd
```

After the first launch, review the generated capture, model, ROI, and kmNet
values through the Web UI or the local configuration file.

Live DeepStream perception is the normal Jetson path. It does not need a
`--live-perception` switch, a preparatory script, or Python. Cargo builds the
native DeepStream/TensorRT integration required by the daemon.

To validate configuration and the HTTP API without opening capture or hardware:

```bash
target/debug/novasightd --dry-run
```

Without `--config`, NovaSight reads exactly `.config/novasight.yaml` relative
to the directory where it was started, creating that one file when it is
missing. `--config` remains available only as an explicit path override and a
missing custom path still fails with `CONFIG_NOT_FOUND`. The backend listens on
`127.0.0.1:5174` by default.
Debug builds expose process-local development access through the Web UI and
`POST /api/license/temporary`. The daemon decides this from its compiled build
profile, makes no external authorization request, and writes no license file;
access ends when that daemon process exits. Release builds reject temporary
access, including release runs started with `--dry-run`, and require the
configured production public key for signed activation.

## Web development run

In a second terminal, from the same repository root:

```bash
pnpm --dir web install
pnpm --dir web dev --host 0.0.0.0
```

Open `http://<jetson-ip>:5173`. Vite proxies `/api`, `/healthz`, and `/ws` to
the backend on port 5174. When the Web UI and backend are on different hosts,
set `VITE_NOVASIGHT_API_BASE` and `VITE_NOVASIGHT_WS_BASE` explicitly.

## Local Control CLI

```bash
cargo build -p novasightctl
target/debug/novasightctl status
```

`novasightctl` talks to the same daemon authority as the Web UI. It is a
control surface, not a second backend.

## Development build checks

```bash
cargo check --workspace
pnpm --dir web typecheck
pnpm --dir web build
```

Local runtime databases, models, logs, generated native artifacts, Cargo
targets, and Web build output are ignored and must not be committed.
