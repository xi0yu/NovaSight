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
cargo run -p novasight-packager -- --profile dev
```

Jetson release package:

```bash
cargo run -p novasight-packager -- --profile jetson-release
```

The packager does all required assembly work:

- builds `novasight`, `novasightd`, and `novasightctl`;
- builds the Web UI into `out/web`;
- creates `out/package/NovaSight`;
- copies binaries into `NovaSight` and `bin/`;
- copies Web assets into `web/`;
- seeds `data/novasight.yaml` from the packaged production template;
- creates `data/models`, `data/runtime/deepstream`, `logs`, and `run`;
- copies the bundled DeepStream tracker config;
- validates that the package has the required runtime files.

The generated package is the deliverable. Do not ask users to run Cargo, pnpm,
`novasightd --config`, or systemd commands.

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
- passes `--config data/novasight.yaml`;
- passes `--web-root web`;
- passes `--ready-file run/ready.json`;
- waits for the daemon to become healthy;
- opens the Web UI URL from `run/ready.json`.

NovaSight-owned runtime files stay inside the package:

- config and database: `data/`;
- models: `data/models/`;
- daemon logs: `logs/novasightd.log`;
- socket and readiness marker: `run/`.

Removing the `NovaSight/` directory removes NovaSight-owned state. The normal
portable path does not install systemd units, does not require root, and does
not write to `/etc`, `/usr`, `/var/lib`, `/var/log`, or `/run/novasight`.

## Same-Path Testing

Yes: testing can use the same startup mode as users. That should be the default
acceptance path.

Developer smoke test:

```bash
cargo run -p novasight-packager -- --profile dev
cd out/package/NovaSight
./NovaSight --dry-run --no-open
```

`--dry-run` keeps hardware output closed. `--no-open` avoids launching a
browser during automated checks. The launched process is still the same
portable entrypoint and the same packaged `bin/novasightd`.

After startup, verify from the package directory:

```bash
cat run/ready.json
bin/novasightctl license status
```

Use the URL in `run/ready.json` for HTTP checks:

```bash
curl -fsS http://127.0.0.1:<port>/healthz
curl -fsS http://127.0.0.1:<port>/ >/dev/null
```

Runtime `status`, `start`, `emergency-stop`, and similar control commands are
behind the same license gate as the Web API. Before activation, use
`license status` as the unauthenticated package-socket check.

## Acceptance Checklist

- `cargo run -p novasight-packager -- --profile dev` creates
  `out/package/NovaSight`.
- `out/package/NovaSight/NovaSight --dry-run --no-open` starts the packaged
  daemon and writes `run/ready.json`.
- `run/ready.json` contains a loopback URL and package-local control socket.
- `web/index.html` is served by `novasightd` from the same origin as the API.
- `bin/novasightctl license status` connects through `run/novasightd.sock`.
- `logs/novasightd.log` is created inside the package.
- No NovaSight-owned runtime files are created outside the package directory.

Directly running `out/cargo/debug/novasightd`, separately running Vite, or
installing a system service can still help diagnose a narrow issue, but it is
not a passing product startup test.
