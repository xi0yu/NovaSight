# NovaSight

NovaSight is a Jetson-first realtime vision application. The maintained product
path is the Rust backend plus the React Web UI; DeepStream, TensorRT, target
selection, prediction, control, and kmNet output are owned by the Rust process.

Current engineering health, active risks, and verification boundaries are kept
in [PROJECT_HEALTH_AUDIT.md](PROJECT_HEALTH_AUDIT.md). Historical plans under
`docs/superpowers/` are not current implementation authority.

## Repository layout

```text
NovaSight/
├── Cargo.toml                 # Root Rust workspace
├── apps/
│   ├── novasight/             # Portable user launcher
│   ├── novasight-packager/    # Author tool that builds portable packages
│   ├── novasightd/            # Backend daemon and HTTP/WebSocket API
│   └── novasightctl/          # Local command-line client
├── crates/                    # Rust domain and platform modules
├── native/                    # Narrow C/C++ seams for NVIDIA SDKs
├── out/                       # Local build output (Git-ignored)
└── web/                       # React Web UI
```

Cargo is intentionally rooted at the repository top level. Backend commands no
longer require entering a `rust/` directory or passing `--manifest-path`.

## Build Output Layout

NovaSight uses project-local output directories:

- Rust binaries and intermediate Cargo artifacts: `out/cargo`
- Web production build: `out/web`
- Portable product bundle: `out/package/NovaSight`
- Portable runtime configuration: `out/package/NovaSight/data/novasight.yaml`

`novasight`, `novasightd`, and `novasightctl` are normal compiled binaries, but
developers and users run NovaSight through the same portable package shape.
`out/cargo` is a build input; `out/package/NovaSight/NovaSight` is the runtime
entry.

## Build Package

Install the JetPack/DeepStream development packages and Rust toolchain first.
The Rust packager builds the three runtime binaries, builds the Web UI, seeds
`data/novasight.yaml`, copies model assets from workspace `data/models` into
the package when present, and validates the portable directory. It is an author
tool, not a user startup path. See
[Portable Package Guide](docs/portable-package-guide.md) for the programmer,
user, and test workflow split.

Developer package:

```bash
cargo run -p novasight-packager
```

Release package:

```bash
cargo run -p novasight-packager -- --profile release
```

Both package profiles build the same DeepStream/TensorRT production runtime;
`debug` only keeps unoptimized Rust binaries for author-side debugging.

Live DeepStream perception is the normal Jetson path. It does not need a
`--live-perception` switch, a preparatory script, or Python. Cargo builds the
native DeepStream/TensorRT integration required by the daemon.

## Portable Product Run

The normal user-facing shape is a portable directory, not a system service:

```text
out/package/NovaSight/
├── NovaSight
├── bin/
│   ├── novasightd
│   └── novasightctl
├── web/
├── data/
├── logs/
└── run/
```

Users start `NovaSight`. The launcher creates package-local `data`, `logs`,
and `run` directories, starts `bin/novasightd`, waits for `run/ready.json`, and
prints the Studio Web UI listener and LAN URLs. Portable packages listen on the
fixed `0.0.0.0:7351` address so a browser on another machine in the same LAN can
open the printed LAN URL. The
launcher also tries to open a desktop browser when one is available. It stays in
the foreground; pressing `Ctrl+C` stops the package-local daemon. It does not
install systemd units, write to `/etc`, `/usr`, `/var/lib`, or
`/run/novasight`, or require root.

Developer and user run:

```bash
cd out/package/NovaSight
./NovaSight
```

The launcher starts `bin/novasightd` with no startup arguments. The daemon uses
the package conventions directly: `data/novasight.yaml`, `web/`, and
`run/ready.json`.

When `web/index.html` exists, `novasightd` serves the Web UI and API from the
same origin. Direct `novasightd` startup remains a diagnostic path, not the
documented normal run path.

Debug builds expose process-local development access through the Web UI and
`POST /api/license/temporary`. The daemon decides this from its compiled build
profile, makes no external authorization request, and writes no license file;
access ends when that daemon process exits. Release builds reject temporary
access and require the configured production public key for signed activation.

## Developer Source Run

From a source workspace, the launcher uses the current frontend source by
default. It starts the internal Rust API and Vite together; Vite serves
`web/src` directly and does not build or read `out/web`. Cargo checks and
refreshes `novasightd` and `novasightctl` from the current Rust source before
they are started:

```bash
cargo run -p novasight
```

The source launcher uses the workspace root as the runtime root. Models,
configuration, logs, and run markers are read from the source tree:

- models: `data/models/`
- config: `data/novasight.yaml`
- logs: `logs/`
- ready/control socket: `run/`

The source launcher has no fallback to `out/web`: an old Web build can no longer
replace current frontend source. Only the user-facing packaged path
`out/package/NovaSight/NovaSight` serves build output created by the packaging
workflow.

## Frontend HMR Development

Frontend development has one process owner and never shares the product runtime
configuration file. After installing `web/node_modules`, the ordinary source
command refreshes the Rust binaries as needed and starts the internal daemon
and Vite:

```bash
cargo run -p novasight
```

`--frontend-dev` remains an explicit compatibility alias for the same mode.

Both processes read their endpoint roles from `deploy/studio-endpoints.json`.
Vite listens on `0.0.0.0:7351` with strict port ownership and proxies API,
health, and WebSocket traffic to the Rust daemon at `127.0.0.1:5174`. The
dedicated frontend-development configuration is initialized with the Rust
development defaults on first use. Product/source-launcher configuration stays
in `data/novasight.yaml` and is not read or rewritten by the HMR daemon. On a
headless Jetson, open `http://<Jetson-LAN-IP>:7351/` from another machine on the
same LAN. The launcher prints the resolved LAN URL and `Ctrl+C` stops both
processes. It does not install packages or build the Web UI. Cargo dependency
fingerprints make unchanged Rust startup checks cheap, while changed backend
source is rebuilt before launch instead of running an older daemon.

## Local Control CLI

```bash
cd out/package/NovaSight
bin/novasightctl license status
bin/novasightctl shutdown
```

`novasightctl` talks to the same daemon authority as the Web UI. It is a
control surface, not a second backend. It resolves `run/novasightd.sock` from
the portable package. Runtime `status` is still protected by the same license
gate as the Web API. `shutdown` is local-socket only and closes the package
daemon when the original launcher terminal is no longer available.

## Development build checks

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
pnpm --dir web typecheck
pnpm --dir web visual:audit
pnpm --dir web build
cargo run -p novasight-packager
```

The pinned Rust toolchain is declared in `rust-toolchain.toml`. The tracked
`quality` workflow repeats portable Rust/native and Studio gates on hosted
runners. Its Jetson production job remains disabled until a self-hosted runner
is registered and the repository variable `NOVASIGHT_JETSON_CI_ENABLED` is set
to `true`.

Local runtime databases, models, logs, generated native artifacts, Cargo
outputs, and Web build output are ignored and must not be committed.
