# NovaSight Portable Deployment

## Target

NovaSight ships as a Jetson-side portable application. The Jetson owns capture,
DeepStream inference, target selection, control, and kmNet/HID output. Users
start the `NovaSight` launcher; it starts `novasightd` as a package-local child
process, prints the Web UI URL, and tries to open a desktop browser when one is
available. The launcher remains the foreground owner; `Ctrl+C` stops the
package-local daemon.

NovaSight does not install a system service by default. It should not write to
`/etc`, `/usr`, `/var/lib`, `/var/log`, or `/run/novasight` in the normal
portable path.

For the full programmer packaging, user running, and same-path testing guide,
see [Portable Package Guide](docs/portable-package-guide.md).

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
cargo run -p novasight-packager -- --profile release
```

The packager builds the Rust binaries with the DeepStream/TensorRT production
runtime, builds the Web UI, writes
`out/package/NovaSight`, and validates the required files. Developers use the
same package shape:

```bash
cargo run -p novasight-packager
```

`debug` and `release` package the same runtime capability. The difference is
only Rust optimization and debug symbol policy.

Runtime acceptance still uses the normal launcher command on the target host.

## Run

Start:

```bash
cd out/package/NovaSight
./NovaSight
```

The launcher starts `bin/novasightd` without startup arguments. The daemon uses
the package conventions directly: `data/novasight.yaml`, `web/`, and
`run/ready.json`.

Portable packages bind an ephemeral `0.0.0.0` port, write `run/ready.json`, and
serve both Studio and the API from that origin. The launcher prints the
`0.0.0.0` listener URL, prints a LAN URL for another machine on the same
network, and treats browser opening as best-effort.

## Local Control

```bash
cd out/package/NovaSight
bin/novasightctl license status
bin/novasightctl shutdown
```

After license activation or debug temporary access:

```bash
bin/novasightctl status
bin/novasightctl emergency-stop
```

`novasightctl` resolves the package socket automatically from
`run/novasightd.sock`. `shutdown` closes the package-local daemon when the
original launcher terminal is no longer available.

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
bin/novasightd --check
./NovaSight
bin/novasightctl license status
```

Expected result:

- No systemd unit is installed or enabled.
- No NovaSight-owned file is created outside the package directory.
- `run/ready.json` records the bound address and a package-local control socket.
- `logs/novasightd.log` contains daemon stdout/stderr.
- Web UI and API are served from the same origin.
