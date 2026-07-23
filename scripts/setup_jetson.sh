#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_jetson.sh [--pyds-wheel /path/to/pyds-*.whl] [--skip-build]

Creates a Jetson-friendly NovaSight virtual environment with system GStreamer
bindings visible through --system-site-packages. If a NVIDIA DeepStream pyds
wheel is provided, the script installs and verifies it. The required DeepStream
parser, metadata bridge, Rust daemon, and thin CLI are built by default; pass
--skip-build only for dependency-only setup. Python remains available for the
kmNet host and rollback operation.
EOF
}

PYDS_WHEEL=""
RUN_BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyds-wheel)
      PYDS_WHEEL="${2:-}"
      shift 2
      ;;
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
  python3-venv \
  python3-dev \
  python3-pip \
  python3-gi \
  python3-gst-1.0 \
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

echo "==> Creating .venv with system site packages"
/usr/bin/python3 -m venv .venv --system-site-packages
source .venv/bin/activate

echo "==> Installing Python dependencies"
python3 -m pip install -U pip setuptools wheel
python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true
python3 -m pip install -e ".[dev]" --no-deps
python3 -m pip install -r requirements-jetson.txt

if [[ -n "$PYDS_WHEEL" ]]; then
  echo "==> Installing DeepStream Python binding wheel: $PYDS_WHEEL"
  python3 -m pip install "$PYDS_WHEEL"
fi

echo "==> Verifying GStreamer Python bindings"
python3 - <<'PY'
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp
Gst.init(None)
print("Gst/GstApp ok")
PY

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
    cargo build --manifest-path rust/Cargo.toml -p novasightd --release --features deepstream
  cargo build --manifest-path rust/Cargo.toml -p novasightctl --release

  echo "==> Staging relocatable Jetson release layout"
  scripts/stage_jetson_release.sh build/jetson-release
fi

echo "==> Verifying optional legacy Python pyds binding"
if python3 - <<'PY'
import pyds
print("pyds ok", pyds)
PY
then
  python3 - <<'PY'
from novasight.deepstream.backend import check_deepstream_dependencies
print(check_deepstream_dependencies("build/deepstream-parser/libnovasight_parser.so"))
PY
else
  echo "pyds is not installed. The canonical Rust DeepStream path is unaffected;"
  echo "only the legacy Python object-meta fallback remains unavailable."
  echo "To enable that fallback, install the matching NVIDIA wheel, for example:"
  echo "  scripts/setup_jetson.sh --pyds-wheel /path/to/pyds-1.2.0-cp310-cp310-linux_aarch64.whl"
fi

echo "==> NovaSight Jetson setup complete"
echo "Canonical Rust production preflight:"
echo "  build/jetson-release/bin/novasightd --config rust/config/novasightd.example.yaml --check"
echo "Thin local control client:"
echo "  build/jetson-release/bin/novasightctl --help"
echo "Legacy Python fallback remains available with:"
echo "  source .venv/bin/activate"
echo "  python3 -m novasight --host 0.0.0.0 --port 5174"
