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
and `run` directories, starts `bin/novasightd`, waits for daemon IPC readiness,
then starts `bin/novasight-web` and prints the Chinese Studio LAN authenticated
address followed by the local and detected IPv4 network listener addresses.
Portable packages listen on the
fixed `0.0.0.0:7351` address so a browser on another machine in the same LAN can
open the printed authenticated LAN URL. On every launcher start, NovaSight creates a
256-bit caller access code in the owner-only `run/web-access-code` file.
The printed URL carries it only in the URL fragment; Studio removes the fragment
immediately and exchanges the code for a Web/API-process-local opaque HttpOnly session.
Anonymous TCP callers can reach only the static authentication screen and `/healthz`;
API and WebSocket traffic are rejected until the browser is authenticated. The
launcher also tries to open a desktop browser when one is available. It stays in
the foreground; pressing `Ctrl+C` stops the package-local Web/API server and daemon. It does not
install systemd units, write to `/etc`, `/usr`, `/var/lib`, or
`/run/novasight`, or require root.

Developer and user run:

```bash
cd out/package/NovaSight
./NovaSight
```

The launcher starts `bin/novasightd` with no startup arguments as the Unix-socket
single source of truth, then starts `bin/novasight-web` as the only TCP/static/API
boundary. Direct daemon startup remains a local diagnostic path and never opens
a LAN port. Diagnostic `novasight-web` startup and `novasight-web --check` must
provide a high-entropy `NOVASIGHT_WEB_ACCESS_CODE`; the normal launcher owns that
provisioning automatically. Set `NOVASIGHT_WEB_SECURE_COOKIE=true` only when
HTTPS terminates in front of the Web/API server. LAN IP literals and `localhost`
are accepted Web hosts by default; an intentional DNS/reverse-proxy name must
be listed exactly in comma-separated `NOVASIGHT_WEB_ALLOWED_HOSTS`. Origin and
Host must agree. The server-side session authenticates the caller,
but direct HTTP still belongs to a trusted LAN; untrusted or routed networks
must use an authenticated TLS reverse proxy.

Debug launchers generate a separate random 256-bit temporary license code for
each daemon start and pass it directly to that `novasightd` process without
writing it to disk. Temporary codes and signed licenses are submitted to
the same `POST /api/license/activate` endpoint. A matching temporary code writes
no license file and expires when that daemon process exits; restarting produces
a different code. Release builds never enable this credential and require the
configured production public key for signed activation.
Debug temporary access deliberately excludes `hardware_control`; physical
output always requires a valid signed license even in a development build.

## Developer Source Run

From a source workspace, the launcher uses the current frontend source by
default. It starts the Unix-only daemon, private Rust Web/API, and Vite together; Vite serves
`web/src` directly and does not build or read `out/web`. Cargo checks and
refreshes `novasightd`, `novasight-web`, and `novasightctl` from the current Rust source before
they are started:

```bash
cargo run -p novasight
```

The source launcher uses the workspace root as the runtime root. Models,
configuration, logs, and run markers are read from the source tree:

- models: `data/models/`
- config: `data/novasight.frontend-dev.yaml`
- logs: `logs/`
- ready/control socket: `run/`

The source launcher has no fallback to `out/web`: an old Web build can no longer
replace current frontend source. Only the user-facing packaged path
`out/package/NovaSight/NovaSight` serves build output created by the packaging
workflow.

## Frontend HMR Development

Frontend development has one process owner and never shares the product runtime
configuration file. After installing `web/node_modules`, the ordinary source
command refreshes the Rust binaries as needed and starts the daemon, Web/API,
and Vite:

```bash
cargo run -p novasight
```

The launcher detects the source workspace automatically; no startup mode or
runtime parameters are accepted from the user-facing command.

The Web/API and Vite processes read their endpoint roles from `deploy/studio-endpoints.json`.
Vite listens on `0.0.0.0:7351` with strict port ownership and proxies API,
health, and WebSocket traffic to `novasight-web` at `127.0.0.1:5174`; that
process reaches `novasightd` only through the Unix socket. The
dedicated frontend-development configuration is initialized with the Rust
development defaults on first use. The ordinary product configuration stays in
`data/novasight.yaml` and is not read or rewritten by the HMR daemon. On a
headless Jetson, open `http://<Jetson-LAN-IP>:7351/` from another machine on the
same LAN. The launcher prints the resolved authenticated LAN URL and `Ctrl+C`
stops all three processes. It does not install packages or build the Web UI. Cargo dependency
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
cargo metadata --locked --format-version 1 >/dev/null
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
pnpm --dir web typecheck
pnpm --dir web visual:audit
pnpm --dir web build
cargo run -p novasight-packager
```

The metadata gate fails when a manifest and `Cargo.lock` disagree, including a
deleted direct dependency that remains in the lockfile build graph. The pinned
Rust toolchain is declared in `rust-toolchain.toml`. The tracked `quality`
workflow repeats portable Rust/native and Studio gates on hosted runners. Set
`NOVASIGHT_JETSON_CI_ENABLED=true` to enable the independent Jetson release
build/safe-start job. Set `NOVASIGHT_JETSON_PRODUCTION_ACCEPTANCE_ENABLED=true`
and provision `NOVASIGHT_JETSON_MODEL_FIXTURE_DIR` to enable the protected real
camera/DeepStream receipt. See
[Jetson Production Acceptance](docs/jetson-production-acceptance.md).
The milestone commands, evidence identity, rollback boundary, and explicit
staging ledger are in the [D1A implementation runbook](docs/d1a-implementation-runbook.md).

Local runtime databases, models, logs, generated native artifacts, Cargo
outputs, and Web build output are ignored and must not be committed.
