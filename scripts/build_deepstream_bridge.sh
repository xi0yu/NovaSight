#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build/deepstream-bridge}"
DEEPSTREAM_ROOT="${NOVASIGHT_DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream}"

cmake \
  -S "${ROOT_DIR}/native/deepstream-bridge" \
  -B "${BUILD_DIR}" \
  -DNOVASIGHT_DEEPSTREAM_ROOT="${DEEPSTREAM_ROOT}" \
  -DBUILD_TESTING=ON
cmake --build "${BUILD_DIR}" --parallel
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "bridge_library: ${BUILD_DIR}/libnovasight_deepstream_bridge.so"
