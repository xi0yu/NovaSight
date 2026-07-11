#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build/deepstream-parser}"
DEEPSTREAM_ROOT="${NOVASIGHT_DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream-7.1}"

cmake -S "${ROOT_DIR}/native/deepstream-parser" -B "${BUILD_DIR}" \
  -DNOVASIGHT_DEEPSTREAM_ROOT="${DEEPSTREAM_ROOT}"
cmake --build "${BUILD_DIR}" --parallel

echo "parser_library: ${BUILD_DIR}/libnovasight_parser.so"
