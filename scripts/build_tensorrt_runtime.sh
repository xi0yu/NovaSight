#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${NOVASIGHT_JETSON_BUILD_DIR:-${ROOT_DIR}/build/jetson-native}"

if [[ "$(uname -s)" != "Linux" || ! "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  echo "available: False"
  echo "reason: tensorrt_runtime_build_requires_jetson"
  echo "detail: build native/tensorrt-runtime on Jetson Linux/aarch64"
  exit 2
fi

cmake -S "${ROOT_DIR}/native/tensorrt-runtime" -B "${BUILD_DIR}/tensorrt-runtime" \
  -DNOVASIGHT_TENSORRT_IMPL=jetson \
  -DNOVASIGHT_TENSORRT_BUILD_TESTS=OFF
cmake --build "${BUILD_DIR}/tensorrt-runtime" --parallel
cmake -E copy_if_different \
  "${BUILD_DIR}/tensorrt-runtime/libnovasight_tensorrt.so" \
  "${BUILD_DIR}/libnovasight_tensorrt.so"

echo "library: ${BUILD_DIR}/libnovasight_tensorrt.so"
echo "export NOVASIGHT_TENSORRT_RUNTIME_DIR=\"${BUILD_DIR}\""
