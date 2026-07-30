#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install_jetson_release.sh [--root /temporary/root] [--enable]

Installs a staged NovaSight release into a versioned /opt/novasight/releases
directory and atomically switches /opt/novasight/current. Existing production
configuration is preserved. --root installs into an isolated filesystem tree
and never invokes systemctl; it is intended for packaging tests.
EOF
}

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT=""
ENABLE_SERVICE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      DEST_ROOT="${2:-}"
      shift 2
      ;;
    --enable)
      ENABLE_SERVICE=1
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

if [[ -z "${DEST_ROOT}" && "$(id -u)" -ne 0 ]]; then
  echo "error: installing into the host filesystem requires root" >&2
  exit 1
fi
if [[ -n "${DEST_ROOT}" && "${DEST_ROOT}" != /* ]]; then
  echo "error: --root must be an absolute path" >&2
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "error: staged release file is missing: $1" >&2
    exit 1
  fi
}

manifest_check() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c SHA256SUMS
  else
    shasum -a 256 -c SHA256SUMS
  fi
}

verify_manifest() {
  local root="$1"
  require_file "${root}/SHA256SUMS"
  if find "${root}" -type l -print -quit | grep -q .; then
    echo "error: release tree contains symlinks" >&2
    exit 1
  fi
  (
    cd "${root}"
    manifest_check
  )
}

check_daemon_build() {
  local daemon="$1"
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

path_in_root() {
  printf '%s%s' "${DEST_ROOT}" "$1"
}

require_file "${SOURCE_ROOT}/RELEASE_ID"
require_file "${SOURCE_ROOT}/SHA256SUMS"
require_file "${SOURCE_ROOT}/bin/novasightd"
require_file "${SOURCE_ROOT}/bin/novasightctl"
require_file "${SOURCE_ROOT}/lib/libnovasight_deepstream_bridge.so"
require_file "${SOURCE_ROOT}/lib/libnovasight_parser.so"
require_file "${SOURCE_ROOT}/deploy/novasight.service"
require_file "${SOURCE_ROOT}/share/novasight/novasight.production.yaml"

verify_manifest "${SOURCE_ROOT}"
check_daemon_build "${SOURCE_ROOT}/bin/novasightd"

RELEASE_ID="$(tr -d '\r\n' < "${SOURCE_ROOT}/RELEASE_ID")"
if [[ ! "${RELEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "error: invalid RELEASE_ID: ${RELEASE_ID}" >&2
  exit 1
fi

OPT_ROOT="$(path_in_root /opt/novasight)"
RELEASES_DIR="${OPT_ROOT}/releases"
RELEASE_DIR="${RELEASES_DIR}/${RELEASE_ID}"
CURRENT_LINK="${OPT_ROOT}/current"
CONFIG_DIR="$(path_in_root /etc/novasight)"
UNIT_DIR="$(path_in_root /etc/systemd/system)"

install -d -m 0755 "${RELEASES_DIR}" "${CONFIG_DIR}" "${UNIT_DIR}"
if [[ -e "${RELEASE_DIR}" || -L "${RELEASE_DIR}" ]]; then
  echo "error: release is already installed: ${RELEASE_DIR}" >&2
  exit 1
fi

# Build the complete immutable version directory before switching `current`.
install -d -m 0755 "${RELEASE_DIR}"
cp -a "${SOURCE_ROOT}/bin" "${SOURCE_ROOT}/lib" "${SOURCE_ROOT}/scripts" \
  "${SOURCE_ROOT}/share" "${SOURCE_ROOT}/deploy" \
  "${RELEASE_DIR}/"
install -m 0644 "${SOURCE_ROOT}/RELEASE_ID" "${RELEASE_DIR}/RELEASE_ID"
install -m 0644 "${SOURCE_ROOT}/SHA256SUMS" "${RELEASE_DIR}/SHA256SUMS"
verify_manifest "${RELEASE_DIR}"
check_daemon_build "${RELEASE_DIR}/bin/novasightd"
install -m 0644 /dev/null "${RELEASE_DIR}/.complete"

install -m 0640 "${SOURCE_ROOT}/share/novasight/novasight.production.yaml" \
  "${CONFIG_DIR}/novasight.yaml.dist"
if [[ ! -e "${CONFIG_DIR}/novasight.yaml" ]]; then
  install -m 0640 "${SOURCE_ROOT}/share/novasight/novasight.production.yaml" \
    "${CONFIG_DIR}/novasight.yaml"
  echo "installed initial config: ${CONFIG_DIR}/novasight.yaml"
else
  echo "preserved existing config: ${CONFIG_DIR}/novasight.yaml"
fi

TEMP_LINK="${OPT_ROOT}/.current.${RELEASE_ID}.$$"
ln -s "releases/${RELEASE_ID}" "${TEMP_LINK}"
mv -Tf "${TEMP_LINK}" "${CURRENT_LINK}"

install -m 0644 "${SOURCE_ROOT}/deploy/novasight.service" \
  "${UNIT_DIR}/novasight.service"

if [[ -z "${DEST_ROOT}" ]]; then
  systemctl daemon-reload
  if [[ "${ENABLE_SERVICE}" -eq 1 ]]; then
    systemctl enable novasight.service
  fi
elif [[ "${ENABLE_SERVICE}" -eq 1 ]]; then
  echo "warning: --enable is ignored with --root" >&2
fi

echo "installed_release: ${RELEASE_ID}"
echo "current: ${CURRENT_LINK} -> releases/${RELEASE_ID}"
echo "config: ${CONFIG_DIR}/novasight.yaml"
echo "service is not started automatically; run novasightd --check before starting it"
