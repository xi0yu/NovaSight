# Jetson Production Acceptance

NovaSight has two independent Jetson receipts. A release build must not depend
on a camera or model fixture merely to prove that the product compiles and
starts fail-closed. Production readiness must not be inferred from that build;
it requires real DeepStream metadata from the protected hardware job.

## Runner contract

Both jobs require a self-hosted runner labelled `self-hosted`, `Linux`, `ARM64`,
and `jetson`, with the pinned Rust toolchain, CMake, curl, jq, pnpm, JetPack,
DeepStream, TensorRT, and the native kmNet build dependencies installed.

Set repository variable `NOVASIGHT_JETSON_CI_ENABLED=true` for the release
build/safe-start job. This job:

1. verifies `Cargo.lock` against every manifest with `cargo metadata --locked`;
2. runs Rust formatting, strict Clippy, workspace tests, Studio type checks, and the frontend unit/contract suite;
3. builds the release portable package on aarch64;
4. starts the real release daemon and Web/API gateway with output disabled;
5. proves anonymous TCP API rejection, bad-code rejection, valid browser
   session creation, CSRF enforcement, signed license activation, emergency
   stop, the strict stopped/disconnected/output-blocked tuple, session logout,
   and local-socket-only daemon shutdown.

Set `NOVASIGHT_JETSON_PRODUCTION_ACCEPTANCE_ENABLED=true` and configure the
protected `jetson-production` GitHub environment for the production receipt.
The runner variable `NOVASIGHT_JETSON_MODEL_FIXTURE_DIR` must point outside the
checkout to an approved TensorRT engine fixture with its matching runtime
metadata. The fixture directory must contain exactly one `.engine` and a
`novasight-fixture.json` manifest:

```json
{"schema_version":1,"engine":"approved.engine","sha256":"<64 lowercase hex>"}
```

The manifest filename and SHA-256 must match that Engine. The checkout `data/models` directory must be empty before the fixture
is copied, preventing stale local models from turning an unknown build into a
false pass. The configured capture device is `/dev/video0`.

The production job additionally:

1. registers and publishes the fixture through the public API;
2. stops both services and runs `bin/novasightd --check` plus
   `bin/novasight-web --check` against the persisted active deployment;
3. starts capture/inference and waits for real input, metadata extraction,
   published/consumed `DetectionBatch` values, latency samples, and a detection
   age inside the configured freshness threshold;
4. performs ordinary stop/restart and independent emergency stop, requiring
   both to finish with the same strict stopped tuple (`runtime_stopped`,
   `will_emit != true`, kmNet disconnected, no accepted commands);
5. asserts that kmNet accepted-command count remains zero because
   `control.output_enabled` is false.

The last assertion is the automated safe-output receipt: real perception may
run, but CI cannot emit a physical device command. A physical kmNet acceptance
must use an isolated target fixture, an operator-approved GitHub environment,
and a separately recorded diagnostic-move/monitor receipt. It is intentionally
not part of pull-request CI.

## Required evidence

A production claim needs the successful workflow URL, commit SHA, Jetson/L4T
identity, package profile, `--check` line, and final
`JETSON_ACCEPTANCE_PASS mode=production` receipt. Host checks or the
`mode=build` receipt alone leave Jetson production readiness `UNKNOWN`.

Every Jetson job uploads `out/jetson-acceptance/<mode>` even on failure. Its
versioned `receipt.json` binds result and exit code to the commit SHA, package
binary identity, approved fixture hash, and timestamps; daemon/Web logs and the
last runtime snapshot are retained when available. The workflow Job Summary
links the corresponding receipt path.
