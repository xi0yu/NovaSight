#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${1:-${ROOT_DIR}/build/jetson-release}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "error: required release artifact is missing: $1" >&2
    exit 1
  fi
}

DAEMON="${ROOT_DIR}/rust/target/release/novasightd"
CLI="${ROOT_DIR}/rust/target/release/novasightctl"
BRIDGE="${ROOT_DIR}/build/deepstream-bridge/libnovasight_deepstream_bridge.so"
PARSER="${ROOT_DIR}/build/deepstream-parser/libnovasight_parser.so"

require_file "${DAEMON}"
require_file "${CLI}"
require_file "${BRIDGE}"
require_file "${PARSER}"
require_file "${ROOT_DIR}/scripts/model_ingress_job.py"
require_file "${ROOT_DIR}/deploy/novasight.service"

install -d "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/scripts" \
  "${STAGE_DIR}/deploy" "${STAGE_DIR}/share/novasight"
install -m 0755 "${DAEMON}" "${STAGE_DIR}/bin/novasightd"
install -m 0755 "${CLI}" "${STAGE_DIR}/bin/novasightctl"
install -m 0755 "${ROOT_DIR}/scripts/model_ingress_job.py" \
  "${STAGE_DIR}/scripts/model_ingress_job.py"
install -m 0644 "${BRIDGE}" "${STAGE_DIR}/lib/libnovasight_deepstream_bridge.so"
install -m 0644 "${PARSER}" "${STAGE_DIR}/lib/libnovasight_parser.so"
install -m 0644 "${ROOT_DIR}/deploy/novasight.service" \
  "${STAGE_DIR}/deploy/novasight.service"
install -m 0644 "${ROOT_DIR}/rust/config/novasightd.example.yaml" \
  "${STAGE_DIR}/share/novasight/novasight.example.yaml"

for optional in libnovasight_preprocess.so libnovasight_tensorrt.so; do
  source_path="${ROOT_DIR}/build/jetson-native/${optional}"
  if [[ -f "${source_path}" ]]; then
    install -m 0644 "${source_path}" "${STAGE_DIR}/lib/${optional}"
  fi
done

echo "release_root: ${STAGE_DIR}"
echo "daemon: ${STAGE_DIR}/bin/novasightd"
echo "cli: ${STAGE_DIR}/bin/novasightctl"
echo "libraries: ${STAGE_DIR}/lib"
