# NovaSight Portable Package Guide

## Goal

NovaSight uses one runtime shape for developers and users:

```text
NovaSight/
├── NovaSight
├── bin/
│   ├── novasightd
│   └── novasightctl
├── web/
├── data/
├── logs/
└── run/
```

Developers build this directory first, then run `NovaSight` from inside it.
Users receive the same directory and run the same entrypoint. Direct
`novasightd`, Vite dev server, and system service startup are diagnostic paths,
not the normal product path.

## Programmer Packaging

Developer package:

```bash
cargo run -p novasight-packager
```

Release package:

```bash
cargo run -p novasight-packager -- --profile release
```

Both profiles build the same DeepStream/TensorRT production runtime. Use
`debug` only when the author build needs unoptimized Rust binaries for local
diagnosis.

The packager does all required assembly work:

- builds `novasight`, `novasightd`, and `novasightctl`;
- builds the Web UI into `out/web`;
- creates `out/package/NovaSight`;
- copies binaries into `NovaSight` and `bin/`;
- copies Web assets into `web/`;
- seeds `data/novasight.yaml` from the packaged production template;
- copies model assets from workspace `data/models` into package `data/models`
  when they exist;
- creates `data/models`, `data/runtime/deepstream`, `logs`, and `run`;
- copies the bundled DeepStream tracker config;
- validates that the package has the required runtime files.

The generated package is the deliverable. Do not ask users to run Cargo, pnpm,
direct `novasightd`, or systemd commands.

Optional archive step after packaging:

```bash
cd out/package
tar -czf NovaSight-Jetson-aarch64.tar.gz NovaSight
```

## User Run

User receives `NovaSight/`, then starts only the launcher:

```bash
cd NovaSight
./NovaSight
```

The launcher owns startup orchestration:

- ensures `data`, `logs`, and `run` exist;
- starts `bin/novasightd`;
- lets the daemon use the package conventions `data/novasight.yaml`, `web/`,
  and `run/ready.json`;
- waits for the daemon to become healthy;
- prints the `0.0.0.0` Studio listener URL, prints a LAN URL for another browser
  on the same network, and tries to open a local desktop browser when one is
  available;
- stays in the foreground so `Ctrl+C` stops the package-local daemon.

NovaSight-owned runtime files stay inside the package:

- config and database: `data/`;
- models: `data/models/`;
- daemon logs: `logs/novasightd.log`;
- socket and readiness marker: `run/`.

Removing the `NovaSight/` directory removes NovaSight-owned state. The normal
portable path does not install systemd units, does not require root, and does
not write to `/etc`, `/usr`, `/var/lib`, `/var/log`, or `/run/novasight`.

When developing from the repository, place model assets in workspace
`data/models` before packaging. The packager copies `.engine`, `.onnx`, and
matching manifest files into `out/package/NovaSight/data/models`. Scripts and
notes in that directory are not copied into the user package.

## Developer Source Run

The source launcher does not build anything implicitly. Prepare its runtime
artifacts explicitly, then run the compiled launcher:

```bash
cargo build --locked -p novasight -p novasightd -p novasightctl
pnpm --dir web build
out/cargo/debug/novasight
```

When the launcher detects that it was started from Cargo's `out/cargo` artifact
directory, it treats the workspace root as the runtime root. It uses source-tree
`data/models`, `data/novasight.yaml`, `logs`, and `run`. It validates
`novasightd`, `novasightctl`, and `out/web/index.html` before starting the
daemon. Missing artifacts fail with an explicit message and never trigger a
hidden Cargo or Pnpm build.

For React HMR, the source launcher owns the API-only Rust daemon and Vite as one
development session:

```bash
out/cargo/debug/novasight --frontend-dev
```

Both processes resolve their endpoint roles from
`deploy/studio-endpoints.json`. This mode exposes Vite at `0.0.0.0:7351` and
keeps the Rust API at `127.0.0.1:5174`. It does not reuse or mutate
`data/novasight.yaml`, invoke Cargo, install packages, or build Web assets. The
launcher prints the LAN URL for a browser on another machine and `Ctrl+C` stops
both child processes.

## Same-Path Testing

Yes: testing can use the same startup mode as users. That should be the default
acceptance path.

Developer smoke test:

```bash
cargo run -p novasight-packager -- --profile release
cd out/package/NovaSight
./NovaSight
```

NovaSight does not expose test-only startup switches. If a test harness needs a
timeout or cleanup, it should supervise the same `./NovaSight` process
externally instead of changing the product command line.

After startup, verify from the package directory:

```bash
cat run/ready.json
bin/novasightctl license status
```

Stop the package-local daemon before rebuilding the same output directory.
Prefer `Ctrl+C` in the launcher terminal; if that terminal is gone, use:

```bash
bin/novasightctl shutdown
```

Use the fixed Studio port for HTTP checks on the target host:

```bash
curl -fsS http://127.0.0.1:7351/healthz
curl -fsS http://127.0.0.1:7351/ >/dev/null
```

For remote Studio access, open the printed
`NovaSight Studio Web UI (LAN): http://<target-lan-ip>:7351/` URL from a
browser on the same LAN. `0.0.0.0` is the bind address; remote browsers use the
target machine's LAN IP.

Runtime `status`, `start`, `emergency-stop`, and similar control commands are
behind the same license gate as the Web API. Before activation, use
`license status` as the unauthenticated package-socket check. Daemon
`shutdown` is local-socket only and does not require a browser session.

## Acceptance Checklist

- `cargo run -p novasight-packager -- --profile release` creates
  `out/package/NovaSight`.
- `out/package/NovaSight/NovaSight` starts the packaged daemon and writes
  `run/ready.json`.
- `run/ready.json` records the bound address and package-local control socket.
- The launcher prints the `0.0.0.0` listener URL and a LAN Studio URL when the
  package binds `0.0.0.0`.
- `web/index.html` is served by `novasightd` from the same origin as the API.
- `bin/novasightctl license status` connects through `run/novasightd.sock`.
- `Ctrl+C` in the launcher terminal, or `bin/novasightctl shutdown`, closes the
  package-local daemon before rebuilds.
- `logs/novasightd.log` is created inside the package.
- No NovaSight-owned runtime files are created outside the package directory.

Directly running `out/cargo/debug/novasightd`, separately running Vite, or
installing a system service can still help diagnose a narrow issue, but it is
not a passing product startup test.
