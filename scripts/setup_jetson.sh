#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_jetson.sh [--skip-build]

Installs Jetson system dependencies. The required DeepStream parser, metadata
bridge, Rust daemon, and thin CLI are built by default; pass --skip-build only
for dependency-only setup. The Python online backend has been removed; this
script does not create a Python runtime environment.
EOF
}

RUN_BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      RUN_BUILD=1
      shift
      ;;
    --skip-build)
      RUN_BUILD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

echo "==> Installing Jetson system packages"
sudo apt update
sudo apt install -y \
  git \
  cmake \
  build-essential \
  libglib2.0-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  \
  libegl1 \
  libegl-dev \
  libgles2 \
  \
  nvidia-l4t-jetson-multimedia-api

echo "==> Verifying JetPack multimedia API headers/libs for nvbufsurface"
for candidate in \
  /usr/src/jetson_multimedia_api/include/nvbufsurface.h \
  /usr/include/aarch64-linux-gnu/nvbufsurface.h \
  /usr/include/nvbufsurface.h \
; do
  if [[ -f "${candidate}" ]]; then
    echo "header_ok: ${candidate}"
    header_found=1
    break
  fi
done
: "${header_found:=0}"
if [[ "${header_found}" -ne 1 ]]; then
  echo "warning: nvbufsurface.h was not found. capture.memory=nvmm will not build."
  echo "         ensure nvidia-l4t-jetson-multimedia-api is installed and"
  echo "         /usr/src/jetson_multimedia_api/include/ is reachable."
fi
if ! ldconfig -p | grep -q 'libnvbufsurface\.so\|libnvbufsurftransform\.so'; then
  echo "warning: libnvbufsurface / libnvbufsurftransform were not found via ldconfig."
  echo "         try: sudo apt install --reinstall nvidia-l4t-jetson-multimedia-api"
fi

echo "==> Verifying DeepStream GStreamer elements"
gst-inspect-1.0 nvvidconv >/dev/null
gst-inspect-1.0 nvv4l2decoder >/dev/null
gst-inspect-1.0 nvstreammux >/dev/null
gst-inspect-1.0 nvinfer >/dev/null
echo "DeepStream GStreamer elements ok"

if [[ "${RUN_BUILD}" -eq 1 ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo is required to build the canonical Rust daemon" >&2
    echo "install a current Rust toolchain, then rerun scripts/setup_jetson.sh" >&2
    exit 1
  fi

  echo "==> Building DeepStream C++ parser and metadata bridge"
  scripts/build_deepstream_parser.sh build/deepstream-parser
  scripts/build_deepstream_bridge.sh build/deepstream-bridge

  echo "==> Building canonical Rust daemon and thin CLI"
  BUILD_REVISION="$(git rev-parse HEAD)"
  BUILD_DIRTY=false
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    BUILD_DIRTY=true
  fi
  NOVASIGHT_BUILD_REVISION="${BUILD_REVISION}" \
    NOVASIGHT_BUILD_DIRTY="${BUILD_DIRTY}" \
    cargo build -p novasightd --release --features deepstream
  cargo build -p novasightctl --release

  echo "==> Staging relocatable Jetson release layout"
  scripts/stage_jetson_release.sh build/jetson-release
fi

echo "==> NovaSight Jetson setup complete"
echo "Canonical Rust production preflight:"
echo "  build/jetson-release/bin/novasightd --config deploy/novasight.production.yaml --check"
echo "Thin local control client:"
echo "  build/jetson-release/bin/novasightctl --help"
