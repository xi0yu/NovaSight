# NovaSight Portable Deployment

## Target

NovaSight ships as a Jetson-side portable application. The Jetson owns capture,
DeepStream inference, target selection, control, and kmNet/HID output. Users
start the `NovaSight` launcher; it starts `novasightd` as a package-local child
process and opens the Web UI served by that daemon.

NovaSight does not install a system service by default. It should not write to
`/etc`, `/usr`, `/var/lib`, `/var/log`, or `/run/novasight` in the normal
portable path.

## Layout

```text
NovaSight/
├── NovaSight
├── bin/
│   ├── novasightd
│   └── novasightctl
├── web/
├── data/
│   ├── models/
│   ├── novasight.db
│   └── license.json
├── logs/
│   └── novasightd.log
└── run/
    ├── novasightd.sock
    └── ready.json
```

`data`, `logs`, and `run` are created by the launcher when missing. Deleting
the `NovaSight/` directory removes NovaSight-owned runtime state.

## Build

Run on the Jetson build host with DeepStream, TensorRT, Node, pnpm, and Rust
installed:

```bash
cargo build --release -p novasight
cargo build --release -p novasightd --features deepstream
cargo build --release -p novasightctl
pnpm --dir web build
```

Prepare the portable directory:

```bash
mkdir -p out/package/NovaSight/bin out/package/NovaSight/web
cp out/cargo/release/novasight out/package/NovaSight/NovaSight
cp out/cargo/release/novasightd out/package/NovaSight/bin/novasightd
cp out/cargo/release/novasightctl out/package/NovaSight/bin/novasightctl
cp -R out/web/. out/package/NovaSight/web/
```

## Run

Start:

```bash
cd out/package/NovaSight
./NovaSight
```

The launcher writes package-local paths into `data/novasight.yaml`, starts:

```bash
bin/novasightd \
  --config data/novasight.yaml \
  --web-root web \
  --ready-file run/ready.json
```

`novasightd` binds a loopback port, writes `run/ready.json`, and serves both
Studio and the API from that origin. The launcher prints and opens the URL.

## Local Control

```bash
cd out/package/NovaSight
bin/novasightctl license status
```

After license activation or debug temporary access:

```bash
bin/novasightctl status
bin/novasightctl emergency-stop
```

`novasightctl` resolves the package socket automatically when
`run/novasightd.sock` exists. Use `--socket` or `NOVASIGHT_CONTROL_SOCKET` only
for diagnostics.

## Hardware Prerequisites

- JetPack with DeepStream and TensorRT installed.
- `v4l2-ctl`, `gst-launch-1.0`, `tegrastats`, and the camera driver visible to
  the current user.
- kmNet/HID network access from the Jetson to the configured hardware endpoint.
- Model assets under `data/models` with generated manifest and DeepStream
  config files.
- Optional `nvtracker` low-level config from `deploy/deepstream-tracker-iou.yml`
  copied into the package or model runtime directory.

Check the host before a hardware run:

```bash
v4l2-ctl --list-devices
gst-inspect-1.0 nvarguscamerasrc
gst-inspect-1.0 nvinfer
which tegrastats
```

## Acceptance

```bash
cd out/package/NovaSight
bin/novasightd --config data/novasight.yaml --check
./NovaSight --no-open
bin/novasightctl license status
```

Expected result:

- No systemd unit is installed or enabled.
- No NovaSight-owned file is created outside the package directory.
- `run/ready.json` contains the loopback URL opened by the launcher.
- `logs/novasightd.log` contains daemon stdout/stderr.
- Web UI and API are served from the same loopback origin.
