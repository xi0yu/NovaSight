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
- starts `bin/novasightd` as a Unix-socket-only service;
- waits for daemon IPC readiness, then starts `bin/novasight-web` as the only
  TCP/static/API boundary;
- creates a 256-bit per-start Web access code, stores it in owner-only
  `run/web-access-code`, prints one Chinese authenticated LAN address, and tries
  to open a local desktop browser when one is available;
- stays in the foreground so `Ctrl+C` stops both package-local services.

NovaSight-owned runtime files stay inside the package:

- config and database: `data/`;
- models: `data/models/`;
- service logs: `logs/novasightd.log` and `logs/novasight-web.log`;
- socket and service readiness markers: `run/`.

The access credential is never sent as an HTTP URL path or query. It is placed
after `#` in the printed URL, consumed by Studio, immediately removed from the
address bar, and exchanged for an opaque HttpOnly, SameSite=Strict server-side
session owned by `novasight-web`. Restarting the Web/API server invalidates every
browser session. Do not share `run/web-access-code` beyond intended operators. Direct HTTP is
for a trusted LAN; use an authenticated TLS reverse proxy across untrusted or
routed networks. IP literals and `localhost` are accepted by default; list
intentional DNS names exactly in `NOVASIGHT_WEB_ALLOWED_HOSTS`. The Web/API server
rejects unapproved Host values and Origin/Host mismatches.

Source/debug launchers also create an independent per-start temporary license
code in memory and pass it directly to the current debug daemon process without
writing it to disk. It is entered in the same authorization form as a formal
signed license and never grants physical hardware output. Release packages do
not create or accept this temporary credential.

Removing the `NovaSight/` directory removes NovaSight-owned state. The normal
portable path does not install systemd units, does not require root, and does
not write to `/etc`, `/usr`, `/var/lib`, `/var/log`, or `/run/novasight`.

When developing from the repository, place model assets in workspace
`data/models` before packaging. The packager copies `.engine`, `.onnx`, and
matching manifest files into `out/package/NovaSight/data/models`. Scripts and
notes in that directory are not copied into the user package.

## Developer Source Run

Install the frontend dependencies and build the launcher itself once, then run
the compiled launcher without a startup mode argument:

```bash
cargo build --locked -p novasight
pnpm --dir web install
out/cargo/debug/novasight
```

When the launcher detects that it was started from Cargo's `out/cargo` artifact
directory, it treats the workspace root as the runtime root and starts the
frontend-development session. It uses source-tree `data/models`,
`data/novasight.frontend-dev.yaml`, `logs`, and `run`. Before launch it runs a
locked Cargo refresh for `novasightd`, `novasight-web`, and `novasightctl`,
verifies that Vite is already installed under `web/node_modules`, and then
starts the Unix-only daemon, private Web/API gateway, and Vite. It does not
install packages or create a production Web build.

For React HMR, the source launcher owns the daemon, Web/API gateway, and Vite as one
development session:

```bash
out/cargo/debug/novasight
```

The Web/API and Vite processes resolve their endpoint roles from
`deploy/studio-endpoints.json`. This mode exposes Vite at `0.0.0.0:7351` and
keeps the authenticated Web/API at `127.0.0.1:5174`; the daemon remains on its
Unix socket. It does not reuse or mutate
`data/novasight.yaml`, install packages, or build Web assets. The
launcher prints the authenticated LAN URL for a browser on another machine and
`Ctrl+C` stops all three child processes. The launcher selects this development layout from its own
artifact path; users do not pass a startup mode.

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

`/healthz` and static UI delivery intentionally remain public so operators can
distinguish reachability from authorization. Anonymous `/api/*` and `/ws/*`
requests return `AUTHENTICATION_REQUIRED`. API smoke tests must first POST the
current access code to `/api/auth/session` with a cookie jar, then attach the
returned `csrf_token` as `X-NovaSight-CSRF` on unsafe methods; production CI
does this in `scripts/ci/jetson-production-acceptance.sh`.

For remote Studio access, open the printed
`NovaSight Studio Web UI (LAN, authenticated): http://<target-lan-ip>:7351/#access=…` URL from a
browser on the same LAN. `0.0.0.0` is the bind address; remote browsers use the
target machine's LAN IP.

TCP runtime `status`, `start`, `emergency-stop`, license activation, and similar
control commands require an operator session first, then the applicable license
feature. Before activation, use
`license status` as the unauthenticated package-socket check. Daemon
`shutdown` is local-socket only and does not require a browser session.

## Acceptance Checklist

- `cargo run -p novasight-packager -- --profile release` creates
  `out/package/NovaSight`.
- `out/package/NovaSight/NovaSight` starts the packaged daemon and Web/API server.
- `run/novasightd-ready.json` records Unix IPC readiness; `run/ready.json`
  records the Web/API address and daemon socket.
- The launcher prints authenticated local/LAN Studio URLs when the Web/API
  server binds `0.0.0.0`, and the fragment is cleared after browser authentication.
- Anonymous TCP API and WebSocket callers are rejected; `/healthz` remains an
  explicit reachability-only endpoint.
- `web/index.html` is served by `novasight-web` from the same origin as the API.
- `bin/novasightctl license status` connects through `run/novasightd.sock`.
- `Ctrl+C` in the launcher terminal, or `bin/novasightctl shutdown`, closes the
  package-local daemon before rebuilds.
- `logs/novasightd.log` and `logs/novasight-web.log` are created inside the package.
- No NovaSight-owned runtime files are created outside the package directory.

Directly running `out/cargo/debug/novasightd`, separately running Vite, or
installing a system service can still help diagnose a narrow issue, but it is
not a passing product startup test.
