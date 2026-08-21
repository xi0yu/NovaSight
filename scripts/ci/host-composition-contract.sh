#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="$(mktemp -d "${TMPDIR:-/tmp}/novasight-host-composition.XXXXXX")"
config_path="$runtime_root/data/novasight.yaml"
daemon_log="$runtime_root/novasightd.log"
web_log="$runtime_root/novasight-web.log"
response_body="$runtime_root/response.json"
auth_body="$runtime_root/auth.json"
activation_body="$runtime_root/activation.json"
config_body="$runtime_root/config.json"
config_update_body="$runtime_root/config-update.json"
runtime_after_failure="$runtime_root/runtime-after-failure.json"
cookie_jar="$runtime_root/cookies.txt"
activation_trace="$runtime_root/activation.trace"
port="${NOVASIGHT_COMPOSITION_PORT:-17351}"
daemon_pid=""
web_pid=""
stage="preflight"

cleanup() {
  local exit_code="$?"
  set +e
  chmod 0755 "$runtime_root/data" 2>/dev/null || true
  if [[ -n "$web_pid" ]] && kill -0 "$web_pid" 2>/dev/null; then
    kill "$web_pid" 2>/dev/null || true
    wait "$web_pid" 2>/dev/null || true
  fi
  if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
    NOVASIGHT_CONTROL_SOCKET="$runtime_root/run/novasightd.sock" \
      "$repo_root/out/cargo/debug/novasightctl" shutdown >/dev/null 2>&1 \
      || kill "$daemon_pid" 2>/dev/null \
      || true
    wait "$daemon_pid" 2>/dev/null || true
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    echo "HOST_COMPOSITION_FAILED: stage=$stage" >&2
    [[ -f "$response_body" ]] && jq '.' "$response_body" >&2
    if [[ -f "$activation_trace" ]]; then
      grep -Ei 'cookie:|x-novasight-csrf:' "$activation_trace" \
        | sed -E 's/(cookie:).*/\1 <redacted>/I; s/(x-novasight-csrf:).*/\1 <redacted>/I' \
        >&2 \
        || true
    fi
    [[ -f "$daemon_log" ]] && tail -n 80 "$daemon_log" >&2
    [[ -f "$web_log" ]] && tail -n 80 "$web_log" >&2
  fi
  if [[ "$runtime_root" == "${TMPDIR:-/tmp}/novasight-host-composition."* ]]; then
    rm -rf "$runtime_root"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

for command_name in cargo chmod cmp curl jq openssl sed; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "HOST_COMPOSITION_FAILED: required command $command_name is missing" >&2
    exit 1
  fi
done

mkdir -p "$runtime_root/data/models" "$runtime_root/run"
cp "$repo_root/crates/novasight-config/src/bootstrap.yaml" "$config_path"
sed -i.bak -E "s/^  port: 7351$/  port: $port/" "$config_path"

cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  -p novasightd -p novasight-web -p novasightctl \
  --no-default-features --locked

access_code="$(openssl rand -hex 32)"
temporary_license_code="$(openssl rand -hex 32)"
jq -n --arg access_code "$access_code" '{access_code: $access_code}' >"$auth_body"
jq -n --arg key "$temporary_license_code" '{key: $key}' >"$activation_body"

(
  cd "$runtime_root"
  NOVASIGHT_TEMPORARY_LICENSE_CODE="$temporary_license_code" \
    "$repo_root/out/cargo/debug/novasightd" --config "$config_path" \
    >"$daemon_log" 2>&1
) &
daemon_pid=$!

for _ in $(seq 1 200); do
  [[ -S "$runtime_root/run/novasightd.sock" ]] && break
  if ! kill -0 "$daemon_pid" 2>/dev/null; then
    echo "HOST_COMPOSITION_FAILED: daemon exited before Unix-socket readiness" >&2
    exit 1
  fi
  sleep 0.05
done
if [[ ! -S "$runtime_root/run/novasightd.sock" ]]; then
  echo "HOST_COMPOSITION_FAILED: daemon Unix-socket readiness timed out" >&2
  exit 1
fi

(
  cd "$runtime_root"
  NOVASIGHT_WEB_ACCESS_CODE="$access_code" \
  NOVASIGHT_WEB_ROOT="$repo_root/web" \
    "$repo_root/out/cargo/debug/novasight-web" --config "$config_path" \
    >"$web_log" 2>&1
) &
web_pid=$!

for _ in $(seq 1 200); do
  if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$web_pid" 2>/dev/null; then
    echo "HOST_COMPOSITION_FAILED: Web/API exited before HTTP readiness" >&2
    exit 1
  fi
  sleep 0.05
done
if ! curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: Web/API HTTP readiness timed out" >&2
  exit 1
fi

stage="anonymous-rejection"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  "http://127.0.0.1:$port/api/runtime/state")"
if [[ "$status_code" != "401" ]] \
  || ! jq -e '.code == "AUTHENTICATION_REQUIRED"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: anonymous runtime request crossed the Web authentication gate" >&2
  exit 1
fi

stage="invalid-login"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  --data '{"access_code":"invalid-access-code"}' \
  "http://127.0.0.1:$port/api/auth/session")"
if [[ "$status_code" != "401" ]] \
  || ! jq -e '.code == "ACCESS_CODE_REJECTED"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: invalid Web access code was not rejected" >&2
  exit 1
fi

stage="valid-login"
curl -fsS -c "$cookie_jar" \
  -H 'Content-Type: application/json' \
  --data-binary "@$auth_body" \
  "http://127.0.0.1:$port/api/auth/session" >"$response_body"
csrf_token="$(jq -er 'select(.authenticated == true) | .csrf_token' "$response_body")"
if ! grep -q '^#HttpOnly_.*novasight_web_session' "$cookie_jar"; then
  echo "HOST_COMPOSITION_FAILED: login did not issue an HttpOnly Web session cookie" >&2
  exit 1
fi

stage="session-status"
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/auth/session" \
  | jq -e --arg csrf "$csrf_token" \
    '.authenticated == true and .role == "operator" and .csrf_token == $csrf' >/dev/null

stage="activation-csrf-rejection"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  --data-binary "@$activation_body" \
  "http://127.0.0.1:$port/api/license/activate")"
if [[ "$status_code" != "403" ]] \
  || ! jq -e '.code == "CSRF_REJECTED"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: license activation without CSRF was not rejected" >&2
  exit 1
fi

stage="temporary-license-activation"
curl -fsS -b "$cookie_jar" \
  --trace-ascii "$activation_trace" \
  -H 'Content-Type: application/json' \
  -H "x-novasight-csrf: $csrf_token" \
  --data-binary "@$activation_body" \
  "http://127.0.0.1:$port/api/license/activate" >"$response_body"
jq -e '
    .valid == true
    and .tier == "temporary"
    and .credential_format == "ephemeral_code"
    and (.features | index("runtime") != null)
    and (.features | index("hardware_control") == null)
  ' "$response_body" >/dev/null

stage="runtime-state"
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/runtime/state" >"$response_body"
jq -e '
    .pipeline.state == "stopped"
    and .capture.running == false
    and .vision.control.will_emit != true
    and .executor.executors.kmnet.runtime_connected == false
  ' "$response_body" >/dev/null

stage="config-revision-conflict"
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/config" >"$config_body"
config_revision="$(jq -er '.revision' "$config_body")"
stream_fps="$(jq -er '.limits.stream_fps' "$config_body")"
jq -n \
  --argjson expected_revision "$((config_revision + 1))" \
  --argjson value "$((stream_fps + 1))" \
  '{section: "limits", key: "stream_fps", value: $value, expected_revision: $expected_revision}' \
  >"$config_update_body"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  -H "x-novasight-csrf: $csrf_token" \
  --data-binary "@$config_update_body" \
  "http://127.0.0.1:$port/api/config")"
if [[ "$status_code" != "409" ]] \
  || ! jq -e '.code == "CONFIG_REVISION_CONFLICT"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: stale config revision was not rejected" >&2
  exit 1
fi
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/config" \
  | jq -e \
    --argjson revision "$config_revision" \
    --argjson stream_fps "$stream_fps" \
    '.revision == $revision and .limits.stream_fps == $stream_fps' >/dev/null

stage="config-persistence-failure"
cp "$config_path" "$runtime_root/config-before-failure.yaml"
jq -n \
  --argjson expected_revision "$config_revision" \
  --argjson value "$((stream_fps + 1))" \
  '{section: "limits", key: "stream_fps", value: $value, expected_revision: $expected_revision}' \
  >"$config_update_body"
chmod 0555 "$runtime_root/data"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  -H "x-novasight-csrf: $csrf_token" \
  --data-binary "@$config_update_body" \
  "http://127.0.0.1:$port/api/config")"
chmod 0755 "$runtime_root/data"
if [[ "$status_code" != "500" ]] \
  || ! jq -e '.code == "CONFIG_IO_ERROR"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: configuration write failure was not surfaced" >&2
  exit 1
fi
if ! cmp -s "$runtime_root/config-before-failure.yaml" "$config_path"; then
  echo "HOST_COMPOSITION_FAILED: failed configuration write changed the persisted file" >&2
  exit 1
fi
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/config" \
  | jq -e \
    --argjson revision "$config_revision" \
    --argjson stream_fps "$stream_fps" \
    '.revision == $revision and .limits.stream_fps == $stream_fps' >/dev/null

stage="runtime-start-stop"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H "x-novasight-csrf: $csrf_token" \
  -X POST "http://127.0.0.1:$port/api/runtime/start")"
if [[ "$status_code" != "204" ]]; then
  echo "HOST_COMPOSITION_FAILED: authenticated runtime start did not reach the daemon" >&2
  exit 1
fi
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/runtime/state" \
  | jq -e '.pipeline.state == "running" and .vision.control.will_emit != true' >/dev/null
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H "x-novasight-csrf: $csrf_token" \
  -X POST "http://127.0.0.1:$port/api/runtime/stop")"
if [[ "$status_code" != "200" ]] \
  || ! jq -e '
      .pipeline.state == "stopped"
      and .vision.control.will_emit != true
      and .executor.executors.kmnet.runtime_connected == false
    ' "$response_body" >/dev/null; then
  curl -sS -b "$cookie_jar" \
    "http://127.0.0.1:$port/api/runtime/state" >"$runtime_after_failure" \
    || true
  echo "HOST_COMPOSITION_FAILED: runtime stop did not confirm the safe stopped tuple" >&2
  [[ -f "$runtime_after_failure" ]] && jq '.' "$runtime_after_failure" >&2
  exit 1
fi

stage="emergency-stop-csrf-rejection"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" -X POST \
  "http://127.0.0.1:$port/api/runtime/emergency-stop")"
if [[ "$status_code" != "403" ]] \
  || ! jq -e '.code == "CSRF_REJECTED"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: emergency stop without CSRF was not rejected" >&2
  exit 1
fi

stage="emergency-stop"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H "x-novasight-csrf: $csrf_token" \
  -X POST "http://127.0.0.1:$port/api/runtime/start")"
if [[ "$status_code" != "204" ]]; then
  echo "HOST_COMPOSITION_FAILED: runtime did not enter running state before emergency stop" >&2
  exit 1
fi
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/runtime/state" \
  | jq -e '.pipeline.state == "running" and .vision.control.will_emit != true' >/dev/null
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H "x-novasight-csrf: $csrf_token" \
  -X POST "http://127.0.0.1:$port/api/runtime/emergency-stop")"
if [[ "$status_code" != "204" ]]; then
  echo "HOST_COMPOSITION_FAILED: authenticated emergency stop did not reach the daemon" >&2
  exit 1
fi
curl -fsS -b "$cookie_jar" \
  "http://127.0.0.1:$port/api/runtime/state" \
  | jq -e '
      .pipeline.state == "stopped"
      and .capture.running == false
      and .vision.output_trace.code == "runtime_stopped"
      and .vision.control.will_emit != true
      and .executor.executors.kmnet.runtime_connected == false
      and .executor.executors.kmnet.accepted_command_count == 0
    ' >/dev/null

stage="daemon-shutdown-authorization"
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" \
  -H "x-novasight-csrf: $csrf_token" \
  -X POST "http://127.0.0.1:$port/api/v1/daemon/shutdown")"
if [[ "$status_code" != "403" ]] \
  || ! jq -e '.code == "AUTHORIZATION_DENIED"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: Web operator reached the local-only daemon shutdown" >&2
  exit 1
fi

stage="daemon-disconnect"
NOVASIGHT_CONTROL_SOCKET="$runtime_root/run/novasightd.sock" \
  "$repo_root/out/cargo/debug/novasightctl" shutdown >/dev/null
for _ in $(seq 1 200); do
  if ! kill -0 "$daemon_pid" 2>/dev/null; then
    wait "$daemon_pid" 2>/dev/null || true
    daemon_pid=""
    break
  fi
  sleep 0.05
done
if [[ -n "$daemon_pid" ]]; then
  echo "HOST_COMPOSITION_FAILED: daemon did not stop through its local control client" >&2
  exit 1
fi

status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" "http://127.0.0.1:$port/api/runtime/state")"
if [[ "$status_code" != "502" ]] \
  || ! jq -e '.code == "DAEMON_UNAVAILABLE"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: daemon disconnect was not projected as an actionable gateway error" >&2
  exit 1
fi

stage="logout"
curl -fsS -b "$cookie_jar" -c "$cookie_jar" \
  -H "x-novasight-csrf: $csrf_token" \
  -X DELETE "http://127.0.0.1:$port/api/auth/session" \
  | jq -e '.authenticated == false and .csrf_token == null' >/dev/null

status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
  -b "$cookie_jar" "http://127.0.0.1:$port/api/runtime/state")"
if [[ "$status_code" != "401" ]] \
  || ! jq -e '.code == "AUTHENTICATION_REQUIRED"' "$response_body" >/dev/null; then
  echo "HOST_COMPOSITION_FAILED: logged-out Web session still reached the API" >&2
  exit 1
fi

echo "HOST_COMPOSITION_PASS anonymous_rejection=verified login=verified session=verified csrf=verified temporary_license=verified config_revision_conflict=verified config_persistence_failure=verified runtime_start_stop=verified emergency_stop=verified local_shutdown_denied=verified daemon_disconnect=verified logout=verified daemon_transport=unix"
