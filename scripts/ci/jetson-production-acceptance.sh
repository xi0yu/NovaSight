#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:-build}"
if [[ "$mode" != "build" && "$mode" != "production" ]]; then
  echo "usage: $0 [build|production]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
package_root="$repo_root/out/package/NovaSight"
license_public_key="$repo_root/testdata/license-public.pem"
license_key="$repo_root/testdata/license-signed.key"
access_code="${NOVASIGHT_WEB_ACCESS_CODE:-}"

receipt_dir="${NOVASIGHT_ACCEPTANCE_RECEIPT_DIR:-$repo_root/out/jetson-acceptance/$mode}"
acceptance_result="failed"
acceptance_stage="preflight"
acceptance_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fixture_identity="not-applicable"
acceptance_tmp=""
daemon_log=""
web_log=""
daemon_pid=""
web_pid=""

assert_safe_stopped_file() {
  jq -e '
        .pipeline.state == "stopped"
        and .capture.running == false
        and (.semantic.daemon_instance_id | type == "string" and length > 0)
        and .vision.output_trace.code == "runtime_stopped"
        and .vision.control.will_emit != true
        and .executor.executors.kmnet.runtime_connected == false
        and .executor.executors.kmnet.accepted_command_count == 0
      ' "$1" >/dev/null
}

cleanup() {
  local exit_code="$?"
  set +e
  if [[ -n "$web_pid" ]] && kill -0 "$web_pid" 2>/dev/null; then
    kill "$web_pid" 2>/dev/null || true
    wait "$web_pid" 2>/dev/null || true
  fi
  if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
    (cd "$package_root" && bin/novasightctl shutdown >/dev/null 2>&1) || kill "$daemon_pid" 2>/dev/null || true
    wait "$daemon_pid" 2>/dev/null || true
  fi
  mkdir -p "$receipt_dir"
  [[ -n "$daemon_log" && -f "$daemon_log" ]] && cp "$daemon_log" "$receipt_dir/novasightd.log"
  [[ -n "$web_log" && -f "$web_log" ]] && cp "$web_log" "$receipt_dir/novasight-web.log"
  [[ -f "${runtime_body:-}" ]] && cp "$runtime_body" "$receipt_dir/final-runtime.json"
  package_identity="unavailable"
  if command -v sha256sum >/dev/null 2>&1 && [[ -x "$package_root/bin/novasightd" && -x "$package_root/bin/novasight-web" ]]; then
    package_identity="$(sha256sum "$package_root/bin/novasightd" "$package_root/bin/novasight-web" | sha256sum | cut -d ' ' -f 1)"
  fi
  commit_sha="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
  platform_arch="$(uname -m 2>/dev/null || echo unknown)"
  platform_kernel="$(uname -sr 2>/dev/null || echo unknown)"
  jetson_release="$(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo unavailable)"
  daemon_instance_id="unavailable"
  final_sequence="-1"
  strict_safe_confirmed="false"
  if command -v jq >/dev/null 2>&1 && [[ -f "${runtime_body:-}" ]]; then
    daemon_instance_id="$(jq -r '.semantic.daemon_instance_id // "unavailable"' "$runtime_body")"
    final_sequence="$(jq -r '.semantic.snapshot_sequence // -1' "$runtime_body")"
    if assert_safe_stopped_file "$runtime_body"; then
      strict_safe_confirmed="true"
    fi
  fi
  if command -v jq >/dev/null 2>&1; then
    jq -n \
    --argjson schema_version 1 \
    --arg result "$acceptance_result" \
    --arg stage "$acceptance_stage" \
    --arg mode "$mode" \
    --arg commit_sha "$commit_sha" \
    --arg package_identity "$package_identity" \
    --arg fixture_identity "$fixture_identity" \
    --arg platform_arch "$platform_arch" \
    --arg platform_kernel "$platform_kernel" \
    --arg jetson_release "$jetson_release" \
    --arg daemon_instance_id "$daemon_instance_id" \
    --argjson final_sequence "$final_sequence" \
    --argjson strict_safe_confirmed "$strict_safe_confirmed" \
    --arg started_at "$acceptance_started_at" \
    --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson exit_code "$exit_code" \
    '{schema_version: $schema_version, result: $result, stage: $stage, mode: $mode, commit_sha: $commit_sha, package_identity: $package_identity, fixture_identity: $fixture_identity, platform: {arch: $platform_arch, kernel: $platform_kernel, jetson_release: $jetson_release}, runtime: {daemon_instance_id: $daemon_instance_id, final_sequence: $final_sequence, strict_safe_confirmed: $strict_safe_confirmed}, unproven: ["physical kmNet movement receipt"], started_at: $started_at, completed_at: $completed_at, exit_code: $exit_code}' \
    >"$receipt_dir/receipt.json"
  else
    printf '{"schema_version":1,"result":"failed","stage":"preflight","mode":"%s","exit_code":%s}\n' "$mode" "$exit_code" >"$receipt_dir/receipt.json"
  fi
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "NovaSight Jetson acceptance: $acceptance_result ($mode), receipt: out/jetson-acceptance/$mode/receipt.json" >>"$GITHUB_STEP_SUMMARY"
  fi
  [[ -n "$acceptance_tmp" && -d "$acceptance_tmp" ]] && rm -rf "$acceptance_tmp"
}
trap cleanup EXIT

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "JETSON_ACCEPTANCE_FAILED: expected aarch64 runner" >&2
  exit 1
fi
if [[ ! -f /etc/nv_tegra_release ]] && ! command -v tegrastats >/dev/null 2>&1; then
  echo "JETSON_ACCEPTANCE_FAILED: NVIDIA Jetson runtime marker is missing" >&2
  exit 1
fi
for command_name in cargo cmake curl jq pnpm rustc sha256sum; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "JETSON_ACCEPTANCE_FAILED: required command $command_name is missing" >&2
    exit 1
  fi
done
if [[ ${#access_code} -lt 32 ]]; then
  echo "JETSON_ACCEPTANCE_FAILED: NOVASIGHT_WEB_ACCESS_CODE must contain at least 32 bytes" >&2
  exit 1
fi
if [[ ! -f "$license_public_key" || ! -f "$license_key" ]]; then
  echo "JETSON_ACCEPTANCE_FAILED: tracked license acceptance fixtures are missing" >&2
  exit 1
fi
acceptance_stage="host-and-package"

cd "$repo_root"
cargo metadata --locked --format-version 1 >/dev/null
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
pnpm --dir web install --frozen-lockfile
pnpm --dir web typecheck
pnpm --dir web test:unit:ci
pnpm --dir web visual:audit
pnpm --dir web build

if [[ "$mode" == "production" ]]; then
  model_fixture_dir="${NOVASIGHT_JETSON_MODEL_FIXTURE_DIR:-}"
  if [[ -z "$model_fixture_dir" || ! -d "$model_fixture_dir" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: NOVASIGHT_JETSON_MODEL_FIXTURE_DIR must name the provisioned TensorRT model fixture" >&2
    exit 1
  fi
  if [[ ! -c /dev/video0 ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: production capture fixture /dev/video0 is unavailable" >&2
    exit 1
  fi
  mkdir -p data/models
  existing_model_entry="$(find data/models -mindepth 1 -print -quit)"
  if [[ -n "$existing_model_entry" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: checkout data/models must be empty before fixture provisioning" >&2
    exit 1
  fi
  fixture_manifest="$model_fixture_dir/novasight-fixture.json"
  if [[ ! -f "$fixture_manifest" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: model fixture must include novasight-fixture.json" >&2
    exit 1
  fi
  engine_count="$(find "$model_fixture_dir" -type f -name '*.engine' | wc -l | tr -d ' ')"
  if [[ "$engine_count" != "1" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: model fixture must contain exactly one TensorRT .engine file" >&2
    exit 1
  fi
  source_engine="$(find "$model_fixture_dir" -type f -name '*.engine' -print -quit)"
  manifest_engine="$(jq -er 'select(.schema_version == 1) | .engine' "$fixture_manifest")"
  manifest_sha="$(jq -er '.sha256 | select(test("^[0-9a-f]{64}$"))' "$fixture_manifest")"
  actual_sha="$(sha256sum "$source_engine" | cut -d ' ' -f 1)"
  if [[ "$manifest_engine" != "$(basename "$source_engine")" || "$manifest_sha" != "$actual_sha" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: fixture manifest does not match the single Engine" >&2
    exit 1
  fi
  fixture_identity="$actual_sha"
  cp -R "$model_fixture_dir/." data/models/
fi

cargo run --locked -p novasight-packager -- --profile release
if [[ ! -x "$package_root/NovaSight" || ! -x "$package_root/bin/novasightd" || ! -x "$package_root/bin/novasight-web" ]]; then
  echo "JETSON_ACCEPTANCE_FAILED: release package is incomplete" >&2
  exit 1
fi

acceptance_tmp="$(mktemp -d "${TMPDIR:-/tmp}/novasight-jetson-acceptance.XXXXXX")"
cookie_jar="$acceptance_tmp/cookies.txt"
daemon_log="$acceptance_tmp/novasightd.log"
web_log="$acceptance_tmp/novasight-web.log"
response_body="$acceptance_tmp/response.json"
auth_body="$acceptance_tmp/auth.json"
activation_body="$acceptance_tmp/license.json"
daemon_pid=""
web_pid=""
csrf_token=""

jq -n --arg access_code "$access_code" '{access_code: $access_code}' >"$auth_body"
jq -Rn --rawfile key "$license_key" '{key: ($key | sub("[\\r\\n]+$"; ""))}' >"$activation_body"

start_stack() {
  : >"$daemon_log"
  : >"$web_log"
  (
    cd "$package_root"
    NOVASIGHT_LICENSE_PUBLIC_KEY_FILE="$license_public_key" \
      bin/novasightd >"$daemon_log" 2>&1
  ) &
  daemon_pid=$!
  for _ in $(seq 1 300); do
    if [[ -f "$package_root/run/novasightd-ready.json" ]]; then
      break
    fi
    if ! kill -0 "$daemon_pid" 2>/dev/null; then
      echo "JETSON_ACCEPTANCE_FAILED: daemon exited before IPC readiness" >&2
      tail -n 80 "$daemon_log" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ ! -f "$package_root/run/novasightd-ready.json" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: daemon IPC readiness timed out" >&2
    tail -n 80 "$daemon_log" >&2
    exit 1
  fi
  (
    cd "$package_root"
    NOVASIGHT_WEB_ACCESS_CODE="$access_code" \
      bin/novasight-web >"$web_log" 2>&1
  ) &
  web_pid=$!
  for _ in $(seq 1 300); do
    if curl -fsS http://127.0.0.1:7351/healthz >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$web_pid" 2>/dev/null; then
      echo "JETSON_ACCEPTANCE_FAILED: Web/API exited before readiness" >&2
      tail -n 80 "$web_log" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "JETSON_ACCEPTANCE_FAILED: Web/API readiness timed out" >&2
  tail -n 80 "$web_log" >&2
  exit 1
}

stop_stack() {
  if [[ -n "$web_pid" ]] && kill -0 "$web_pid" 2>/dev/null; then
    kill "$web_pid"
    wait "$web_pid" 2>/dev/null || true
    web_pid=""
  fi
  (cd "$package_root" && bin/novasightctl shutdown)
  for _ in $(seq 1 100); do
    if ! kill -0 "$daemon_pid" 2>/dev/null; then
      wait "$daemon_pid" 2>/dev/null || true
      daemon_pid=""
      return 0
    fi
    sleep 0.1
  done
  echo "JETSON_ACCEPTANCE_FAILED: daemon shutdown timed out" >&2
  exit 1
}

pair_and_activate() {
  local status_code
  status_code="$(curl -sS -o "$response_body" -w '%{http_code}' http://127.0.0.1:7351/api/runtime/state)"
  if [[ "$status_code" != "401" ]] || ! jq -e '.code == "AUTHENTICATION_REQUIRED"' "$response_body" >/dev/null; then
    echo "JETSON_ACCEPTANCE_FAILED: anonymous API request was not rejected by Web authentication" >&2
    exit 1
  fi

  status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data '{"access_code":"invalid-invalid-invalid-invalid"}' \
    http://127.0.0.1:7351/api/auth/session)"
  if [[ "$status_code" != "401" ]] || ! jq -e '.code == "ACCESS_CODE_REJECTED"' "$response_body" >/dev/null; then
    echo "JETSON_ACCEPTANCE_FAILED: invalid Web access code was not rejected" >&2
    exit 1
  fi

  curl -fsS -c "$cookie_jar" \
    -H 'Content-Type: application/json' \
    --data-binary "@$auth_body" \
    http://127.0.0.1:7351/api/auth/session >"$response_body"
  csrf_token="$(jq -er 'select(.authenticated == true) | .csrf_token' "$response_body")"
  if ! grep -q '^#HttpOnly_.*novasight_web_session' "$cookie_jar"; then
    echo "JETSON_ACCEPTANCE_FAILED: Web session cookie is not HttpOnly" >&2
    exit 1
  fi

  status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
    -b "$cookie_jar" -X POST \
    http://127.0.0.1:7351/api/runtime/emergency-stop)"
  if [[ "$status_code" != "403" ]] || ! jq -e '.code == "CSRF_REJECTED"' "$response_body" >/dev/null; then
    echo "JETSON_ACCEPTANCE_FAILED: unsafe Web request without CSRF was not rejected" >&2
    exit 1
  fi

  status_code="$(curl -sS -o "$response_body" -w '%{http_code}' \
    -b "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X POST \
    http://127.0.0.1:7351/api/v1/daemon/shutdown)"
  if [[ "$status_code" != "403" ]] || ! jq -e '.code == "AUTHORIZATION_DENIED"' "$response_body" >/dev/null; then
    echo "JETSON_ACCEPTANCE_FAILED: Web operator could reach local-only daemon shutdown" >&2
    exit 1
  fi

  curl -fsS -b "$cookie_jar" -c "$cookie_jar" \
    -H 'Content-Type: application/json' \
    -H "X-NovaSight-CSRF: $csrf_token" \
    --data-binary "@$activation_body" \
    http://127.0.0.1:7351/api/license/activate \
    | jq -e '.valid == true and (.features | index("runtime") != null)' >/dev/null
}

assert_safe_stopped_state() {
  curl -fsS -b "$cookie_jar" http://127.0.0.1:7351/api/runtime/state >"$response_body"
  assert_safe_stopped_file "$response_body"
}

start_stack
pair_and_activate
acceptance_stage="runtime-safety"
assert_safe_stopped_state
curl -fsS -o /dev/null -b "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X POST http://127.0.0.1:7351/api/runtime/emergency-stop
assert_safe_stopped_state

if [[ "$mode" == "production" ]]; then
  acceptance_stage="production-runtime"
  catalog_body="$acceptance_tmp/catalog.json"
  registration_body="$acceptance_tmp/registration.json"
  publish_body="$acceptance_tmp/publish.json"
  runtime_body="$acceptance_tmp/runtime.json"
  curl -fsS -b "$cookie_jar" 'http://127.0.0.1:7351/api/models/catalog?force=true' >"$catalog_body"
  model_path="$(jq -er '[.. | objects | select(.type? == "model" and .kind? == "engine")][0].relative_path' "$catalog_body")"
  jq -n --arg relative_path "$model_path" '{relative_path: $relative_path}' >"$registration_body"
  curl -fsS -b "$cookie_jar" \
    -H 'Content-Type: application/json' -H "X-NovaSight-CSRF: $csrf_token" \
    --data-binary "@$registration_body" \
    http://127.0.0.1:7351/api/models/catalog/register >"$response_body"
  project_id="$(jq -er '.project.id' "$response_body")"
  artifact_id="$(jq -er '.artifact.id' "$response_body")"
  jq -n --argjson artifact_id "$artifact_id" \
    '{artifact_id: $artifact_id, parser_preset: "auto"}' >"$publish_body"
  curl -fsS -b "$cookie_jar" \
    -H 'Content-Type: application/json' -H "X-NovaSight-CSRF: $csrf_token" \
    --data-binary "@$publish_body" \
    "http://127.0.0.1:7351/api/models/projects/$project_id/publish" \
    | jq -e '.report.applied == true and .report.rolled_back == false' >/dev/null

  stop_stack
  (
    cd "$package_root"
    NOVASIGHT_LICENSE_PUBLIC_KEY_FILE="$license_public_key" \
      bin/novasightd --check
  )
  (
    cd "$package_root"
    NOVASIGHT_WEB_ACCESS_CODE="$access_code" bin/novasight-web --check
  )

  rm -f "$cookie_jar"
  start_stack
  pair_and_activate
  curl -fsS -o /dev/null -b "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X POST http://127.0.0.1:7351/api/runtime/start

  receipt_ready=false
  for _ in $(seq 1 900); do
    curl -fsS -b "$cookie_jar" http://127.0.0.1:7351/api/runtime/state >"$runtime_body"
    if jq -e '
      .pipeline.state == "running"
      and .capture.running == true
      and .inference.running == true
      and .inference.metadata_extractions > 0
      and .inference.published_batches > 0
      and .statistics.detection_batch_consumed_counter > 0
      and .statistics.inference_latency_samples > 0
      and (.statistics.detection_data_age_ms != null)
      and (.statistics.detection_data_age_ms <= .statistics.detection_freshness_threshold_ms)
      and .executor.executors.kmnet.accepted_command_count == 0
    ' "$runtime_body" >/dev/null; then
      receipt_ready=true
      break
    fi
    if jq -e '.pipeline.state == "faulted" or .inference.terminal_error == true' "$runtime_body" >/dev/null; then
      echo "JETSON_ACCEPTANCE_FAILED: production runtime faulted" >&2
      jq '.' "$runtime_body" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ "$receipt_ready" != "true" ]]; then
    echo "JETSON_ACCEPTANCE_FAILED: real DeepStream metadata/freshness receipt timed out" >&2
    jq '.' "$runtime_body" >&2
    exit 1
  fi

  curl -fsS -o /dev/null -b "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X POST http://127.0.0.1:7351/api/runtime/stop
  for _ in $(seq 1 200); do
    curl -fsS -b "$cookie_jar" http://127.0.0.1:7351/api/runtime/state >"$runtime_body"
    if assert_safe_stopped_file "$runtime_body"; then
      break
    fi
    sleep 0.1
  done
  assert_safe_stopped_file "$runtime_body"

  curl -fsS -o /dev/null -b "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X POST http://127.0.0.1:7351/api/runtime/start
  for _ in $(seq 1 300); do
    curl -fsS -b "$cookie_jar" http://127.0.0.1:7351/api/runtime/state >"$runtime_body"
    if jq -e '.pipeline.state == "running" and .capture.running == true' "$runtime_body" >/dev/null; then
      break
    fi
    sleep 0.1
  done
  curl -fsS -o /dev/null -b "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X POST http://127.0.0.1:7351/api/runtime/emergency-stop
  for _ in $(seq 1 200); do
    curl -fsS -b "$cookie_jar" http://127.0.0.1:7351/api/runtime/state >"$runtime_body"
    if assert_safe_stopped_file "$runtime_body"; then
      break
    fi
    sleep 0.1
  done
  assert_safe_stopped_file "$runtime_body"
fi

curl -fsS -b "$cookie_jar" -c "$cookie_jar" -H "X-NovaSight-CSRF: $csrf_token" -X DELETE \
  http://127.0.0.1:7351/api/auth/session \
  | jq -e '.authenticated == false' >/dev/null
status_code="$(curl -sS -o "$response_body" -w '%{http_code}' -b "$cookie_jar" http://127.0.0.1:7351/api/runtime/state)"
if [[ "$status_code" != "401" ]]; then
  echo "JETSON_ACCEPTANCE_FAILED: cleared Web session still accesses the API" >&2
  exit 1
fi

stop_stack
acceptance_stage="complete"
acceptance_result="passed"
echo "JETSON_ACCEPTANCE_PASS mode=$mode web_auth=verified csrf=verified daemon_transport=unix package=release output=fail_closed"
