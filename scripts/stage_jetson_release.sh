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

DAEMON="${NOVASIGHT_STAGE_DAEMON:-${ROOT_DIR}/rust/target/release/novasightd}"
CLI="${NOVASIGHT_STAGE_CLI:-${ROOT_DIR}/rust/target/release/novasightctl}"
BRIDGE="${NOVASIGHT_STAGE_BRIDGE:-${ROOT_DIR}/build/deepstream-bridge/libnovasight_deepstream_bridge.so}"
PARSER="${NOVASIGHT_STAGE_PARSER:-${ROOT_DIR}/build/deepstream-parser/libnovasight_parser.so}"

require_file "${DAEMON}"
require_file "${CLI}"
require_file "${BRIDGE}"
require_file "${PARSER}"
require_file "${ROOT_DIR}/scripts/model_ingress_job.py"
require_file "${ROOT_DIR}/scripts/install_jetson_release.sh"
require_file "${ROOT_DIR}/scripts/deployment_check.py"
require_file "${ROOT_DIR}/scripts/release_manifest.py"
require_file "${ROOT_DIR}/deploy/novasight.service"
require_file "${ROOT_DIR}/deploy/novasight.production.yaml"
require_file "${ROOT_DIR}/novasight/executors/kmnet_host.py"
require_file "${ROOT_DIR}/novasight/executors/kmnet_loader.py"

install -d "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/scripts" \
  "${STAGE_DIR}/deploy" "${STAGE_DIR}/share/novasight"
install -m 0755 "${DAEMON}" "${STAGE_DIR}/bin/novasightd"
install -m 0755 "${CLI}" "${STAGE_DIR}/bin/novasightctl"
install -m 0755 "${ROOT_DIR}/scripts/model_ingress_job.py" \
  "${STAGE_DIR}/scripts/model_ingress_job.py"
install -m 0755 "${ROOT_DIR}/scripts/install_jetson_release.sh" \
  "${STAGE_DIR}/scripts/install_jetson_release.sh"
install -m 0755 "${ROOT_DIR}/scripts/deployment_check.py" \
  "${STAGE_DIR}/scripts/deployment_check.py"
install -m 0755 "${ROOT_DIR}/scripts/release_manifest.py" \
  "${STAGE_DIR}/scripts/release_manifest.py"
install -m 0644 "${BRIDGE}" "${STAGE_DIR}/lib/libnovasight_deepstream_bridge.so"
install -m 0644 "${PARSER}" "${STAGE_DIR}/lib/libnovasight_parser.so"
install -m 0644 "${ROOT_DIR}/deploy/novasight.service" \
  "${STAGE_DIR}/deploy/novasight.service"
install -m 0640 "${ROOT_DIR}/deploy/novasight.production.yaml" \
  "${STAGE_DIR}/share/novasight/novasight.production.yaml"
install -m 0644 "${ROOT_DIR}/rust/config/novasightd.example.yaml" \
  "${STAGE_DIR}/share/novasight/novasight.example.yaml"

# The online vision/control path is Rust. The daemon-owned kmNet crash-isolation
# helper and allowlisted offline model job still import the retained Python
# package, so a release must carry that exact code instead of depending on a
# source checkout or an ambient editable install.
install -d "${STAGE_DIR}/novasight"
cp -R "${ROOT_DIR}/novasight/." "${STAGE_DIR}/novasight/"
find "${STAGE_DIR}/novasight" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "${STAGE_DIR}/novasight" -type d -name __pycache__ -empty -delete

RELEASE_ID="${NOVASIGHT_RELEASE_ID:-}"
if [[ -z "${RELEASE_ID}" ]]; then
  RELEASE_ID="$(git -C "${ROOT_DIR}" rev-parse --short=12 HEAD)"
  if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=no)" ]]; then
    RELEASE_ID="${RELEASE_ID}.dirty"
  fi
fi
if [[ ! "${RELEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "error: invalid NOVASIGHT_RELEASE_ID: ${RELEASE_ID}" >&2
  exit 1
fi
printf '%s\n' "${RELEASE_ID}" > "${STAGE_DIR}/RELEASE_ID"

for optional in libnovasight_preprocess.so libnovasight_tensorrt.so; do
  source_path="${ROOT_DIR}/build/jetson-native/${optional}"
  if [[ -f "${source_path}" ]]; then
    install -m 0644 "${source_path}" "${STAGE_DIR}/lib/${optional}"
  fi
done

# Bind every staged payload file to this release before it can be installed.
# The installer verifies the same manifest before and after copying.
python3 "${STAGE_DIR}/scripts/release_manifest.py" generate "${STAGE_DIR}"

echo "release_root: ${STAGE_DIR}"
echo "daemon: ${STAGE_DIR}/bin/novasightd"
echo "cli: ${STAGE_DIR}/bin/novasightctl"
echo "libraries: ${STAGE_DIR}/lib"
echo "production_config: ${STAGE_DIR}/share/novasight/novasight.production.yaml"
echo "python_helpers: ${STAGE_DIR}/novasight"
echo "release_id: ${RELEASE_ID}"
echo "manifest: ${STAGE_DIR}/SHA256SUMS.json"
