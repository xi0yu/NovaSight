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

DAEMON="${NOVASIGHT_STAGE_DAEMON:-${ROOT_DIR}/target/release/novasightd}"
CLI="${NOVASIGHT_STAGE_CLI:-${ROOT_DIR}/target/release/novasightctl}"
BRIDGE="${NOVASIGHT_STAGE_BRIDGE:-${ROOT_DIR}/build/deepstream-bridge/libnovasight_deepstream_bridge.so}"
PARSER="${NOVASIGHT_STAGE_PARSER:-${ROOT_DIR}/build/deepstream-parser/libnovasight_parser.so}"
MANIFEST_NAME="SHA256SUMS"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

check_daemon_build() {
  local daemon="$1"
  local expected_revision="$2"
  local expected_dirty="$3"
  local output
  output="$("${daemon}" --build-info-json)"
  [[ "${output}" == *'"schema_version":1'* ]] || {
    echo "error: invalid novasightd build-info schema" >&2
    exit 1
  }
  [[ "${output}" == *'"binary":"novasightd"'* ]] || {
    echo "error: staged daemon is not novasightd" >&2
    exit 1
  }
  [[ "${output}" == *'"source_revision":"'"${expected_revision}"'"'* ]] || {
    echo "error: staged daemon source revision does not match ${expected_revision}" >&2
    exit 1
  }
  [[ "${output}" == *'"source_dirty":'"${expected_dirty}"* ]] || {
    echo "error: staged daemon dirty flag does not match ${expected_dirty}" >&2
    exit 1
  }
  [[ "${output}" == *'"target":"aarch64-'*linux* ]] || {
    echo "error: staged daemon target is not aarch64 Linux: ${output}" >&2
    exit 1
  }
  [[ "${output}" == *'"profile":"release"'* ]] || {
    echo "error: staged daemon is not a release build" >&2
    exit 1
  }
  [[ "${output}" == *'"features":["deepstream"]'* ]] || {
    echo "error: staged daemon was not built with the deepstream feature" >&2
    exit 1
  }
}

write_manifest() {
  local root="$1"
  if find "${root}" -type l -print -quit | grep -q .; then
    echo "error: release tree contains symlinks" >&2
    exit 1
  fi
  rm -f "${root}/${MANIFEST_NAME}" "${root}/SHA256SUMS.json"
  (
    cd "${root}"
    find . -type f \
      ! -name "${MANIFEST_NAME}" \
      ! -name ".complete" \
      -print \
      | LC_ALL=C sort \
      | while IFS= read -r path; do
          hash_file "${path#./}"
        done > "${MANIFEST_NAME}"
  )
}

require_file "${DAEMON}"
require_file "${CLI}"
require_file "${BRIDGE}"
require_file "${PARSER}"
require_file "${ROOT_DIR}/scripts/install_jetson_release.sh"
require_file "${ROOT_DIR}/deploy/novasight.service"
require_file "${ROOT_DIR}/deploy/novasight.production.yaml"

rm -rf "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/scripts" \
  "${STAGE_DIR}/deploy" "${STAGE_DIR}/share" \
  "${STAGE_DIR}/RELEASE_ID" "${STAGE_DIR}/${MANIFEST_NAME}" \
  "${STAGE_DIR}/SHA256SUMS.json"
install -d "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/scripts" \
  "${STAGE_DIR}/deploy" "${STAGE_DIR}/share/novasight"
install -m 0755 "${DAEMON}" "${STAGE_DIR}/bin/novasightd"
install -m 0755 "${CLI}" "${STAGE_DIR}/bin/novasightctl"
install -m 0755 "${ROOT_DIR}/scripts/install_jetson_release.sh" \
  "${STAGE_DIR}/scripts/install_jetson_release.sh"
install -m 0644 "${BRIDGE}" "${STAGE_DIR}/lib/libnovasight_deepstream_bridge.so"
install -m 0644 "${PARSER}" "${STAGE_DIR}/lib/libnovasight_parser.so"
install -m 0644 "${ROOT_DIR}/deploy/novasight.service" \
  "${STAGE_DIR}/deploy/novasight.service"
install -m 0640 "${ROOT_DIR}/deploy/novasight.production.yaml" \
  "${STAGE_DIR}/share/novasight/novasight.production.yaml"
install -m 0644 "${ROOT_DIR}/deploy/novasight.production.yaml" \
  "${STAGE_DIR}/share/novasight/novasight.example.yaml"

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

SOURCE_REVISION="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
SOURCE_DIRTY=false
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=no)" ]]; then
  SOURCE_DIRTY=true
fi
check_daemon_build "${STAGE_DIR}/bin/novasightd" "${SOURCE_REVISION}" "${SOURCE_DIRTY}"

for optional in libnovasight_tensorrt.so; do
  source_path="${ROOT_DIR}/build/jetson-native/${optional}"
  if [[ -f "${source_path}" ]]; then
    install -m 0644 "${source_path}" "${STAGE_DIR}/lib/${optional}"
  fi
done

# Bind every staged payload file to this release before it can be installed.
# The installer verifies the same manifest before and after copying.
write_manifest "${STAGE_DIR}"

echo "release_root: ${STAGE_DIR}"
echo "daemon: ${STAGE_DIR}/bin/novasightd"
echo "cli: ${STAGE_DIR}/bin/novasightctl"
echo "libraries: ${STAGE_DIR}/lib"
echo "production_config: ${STAGE_DIR}/share/novasight/novasight.production.yaml"
echo "release_id: ${RELEASE_ID}"
echo "manifest: ${STAGE_DIR}/${MANIFEST_NAME}"
