# NovaSight

NovaSight is a Jetson-first realtime vision application. The maintained product
path is the Rust backend plus the React Web UI; DeepStream, TensorRT, target
selection, prediction, control, and kmNet output are owned by the Rust process.

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
`data/novasight.yaml`, and validates the portable directory. It is an author
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
prints the Web UI URL. It also tries to open a desktop browser when one is
available. It does not install systemd units, write to `/etc`, `/usr`,
`/var/lib`, or `/run/novasight`, or require root.

Developer and user run:

```bash
cd out/package/NovaSight
./NovaSight
```

The launcher starts `bin/novasightd` with no startup arguments. The daemon uses
the package conventions directly: `data/novasight.yaml`, `web/`, and
`run/ready.json`.

When `web/index.html` exists, `novasightd` serves the Web UI and API from the
same loopback origin. Direct `novasightd` startup remains a diagnostic path,
not the documented normal run path.

Debug builds expose process-local development access through the Web UI and
`POST /api/license/temporary`. The daemon decides this from its compiled build
profile, makes no external authorization request, and writes no license file;
access ends when that daemon process exits. Release builds reject temporary
access and require the configured production public key for signed activation.

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
daemon before rebuilds.

## Development build checks

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p novasight-packager
```

Local runtime databases, models, logs, generated native artifacts, Cargo
outputs, and Web build output are ignored and must not be committed.
