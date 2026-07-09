#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_jetson_preprocess.sh [options]

Builds the NovaSight Jetson native NVMM -> TensorRT preprocess shared library.
This script only compiles and load-checks the native preprocess .so; it does
not start capture, run inference smoke tests, or change runtime config.

Options:
  --build-dir DIR   CMake build directory. Default: build/jetson-native
  --source PATH     Override Jetson production CUDA/C++ source.
  --cmake PATH      CMake executable. Default: cmake
  --python PATH     Python executable. Default: .venv/bin/python3 or python3
  --preflight       Run `python -m novasight doctor jetson-preflight` first.
  --skip-load       Only compile/link; skip loading the built shared library.
  -h, --help        Show this help.

On success, export the printed NOVASIGHT_JETSON_NATIVE_LIBRARY before starting
the NovaSight backend.
EOF
}

BUILD_DIR="build/jetson-native"
SOURCE=""
CMAKE_BIN="${CMAKE:-cmake}"
PYTHON_BIN="${PYTHON:-}"
RUN_PREFLIGHT=0
SKIP_LOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --cmake)
      CMAKE_BIN="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --preflight)
      RUN_PREFLIGHT=1
      shift
      ;;
    --skip-load)
      SKIP_LOAD=1
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

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${BUILD_DIR}" ]]; then
  echo "--build-dir must not be empty" >&2
  exit 2
fi
if [[ "${BUILD_DIR}" = /* ]]; then
  BUILD_DIR_ABS="${BUILD_DIR}"
else
  BUILD_DIR_ABS="${ROOT_DIR}/${BUILD_DIR}"
fi

HOST_SYSTEM="$(uname -s)"
HOST_MACHINE="$(uname -m)"

build_args=(
  doctor
  jetson-native-build
  --build-dir "${BUILD_DIR_ABS}"
  --cmake "${CMAKE_BIN}"
)
if [[ -n "${SOURCE}" ]]; then
  build_args+=(--source "${SOURCE}")
fi
if [[ "${SKIP_LOAD}" -eq 1 ]]; then
  build_args+=(--skip-load)
fi

echo "==> NovaSight Jetson native preprocess build"
echo "repo: ${ROOT_DIR}"
echo "python: ${PYTHON_BIN}"
echo "build_dir: ${BUILD_DIR_ABS}"
echo "host: ${HOST_SYSTEM} ${HOST_MACHINE}"
echo "command: ${PYTHON_BIN} -m novasight doctor jetson-native-build ${build_args[*]:2}"

if [[ "${HOST_SYSTEM}" != "Linux" || ! "${HOST_MACHINE}" =~ ^(aarch64|arm64)$ ]]; then
  echo "available: False"
  echo "reason: jetson_native_build_requires_jetson"
  echo "detail: production NVMM TensorRT preprocess must be built on Jetson Linux/aarch64 with CUDA and NvBufSurface. This host cannot compile libnovasight_preprocess.so."
  exit 2
fi

if [[ "${RUN_PREFLIGHT}" -eq 1 ]]; then
  echo "==> Running Jetson preflight"
  "${PYTHON_BIN}" -m novasight doctor jetson-preflight
fi

"${PYTHON_BIN}" -m novasight "${build_args[@]}"

LIBRARY="${BUILD_DIR_ABS}/libnovasight_preprocess.so"
if [[ ! -f "${LIBRARY}" ]]; then
  echo "build completed but ${LIBRARY} was not produced" >&2
  exit 2
fi

ENV_FILE="${BUILD_DIR_ABS}/novasight-native-env.sh"
mkdir -p "${BUILD_DIR_ABS}"
printf 'export NOVASIGHT_JETSON_NATIVE_LIBRARY="%s"\n' "${LIBRARY}" > "${ENV_FILE}"

echo "==> Built Jetson native preprocess library"
echo "library: ${LIBRARY}"
echo "env_file: ${ENV_FILE}"
echo "export NOVASIGHT_JETSON_NATIVE_LIBRARY=\"${LIBRARY}\""
echo "To use it in this shell:"
echo "  source \"${ENV_FILE}\""
