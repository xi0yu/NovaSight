# Jetson Production Acceptance

NovaSight has two independent Jetson receipts. A release build must not depend
on a camera or model fixture merely to prove that the product compiles and
starts fail-closed. Production readiness must not be inferred from that build;
it requires real DeepStream metadata from the protected hardware job.

The ordinary quality workflow defines `Linux ARM64 portable contracts` on
GitHub's `ubuntu-24.04-arm` hosted runner. For a given SHA, a successful job
builds and preflights the fail-closed host-preview daemon, then runs the
workspace without default hardware features. That receipt proves native ARM64
Rust/Linux compatibility for the SHA without using a developer-owned machine.
It cannot prove NVIDIA SDK headers/libraries, DeepStream plugins, TensorRT
execution, NVMM, a real camera, or kmNet hardware.

## Runner contract

Both jobs require a self-hosted runner labelled `self-hosted`, `Linux`, `ARM64`,
and `jetson`, with the pinned Rust toolchain, CMake, curl, jq, pnpm, JetPack,
DeepStream, TensorRT, and the native kmNet build dependencies installed.

## Optional Jetson CI setup

No setup is required for the GitHub-hosted ARM64 gate. The steps below are only
for the two hardware-specific Jetson receipts.

1. In the GitHub repository, add a Linux ARM64 self-hosted runner and run the
   generated registration commands on the dedicated Jetson build account.
2. Add the custom `jetson` label and install the runner as a service so ordinary
   pushes do not depend on an interactive terminal.
3. Install the dependencies in the runner contract above and verify that the
   runner account can read the camera and NVIDIA runtime without using root.
4. Set repository variable `NOVASIGHT_JETSON_CI_ENABLED=true` after the build
   runner is ready.
5. Create the protected `jetson-production` environment, configure required
   reviewers, and set its `NOVASIGHT_JETSON_MODEL_FIXTURE_DIR` variable to the
   approved fixture outside the checkout. Set repository variable
   `NOVASIGHT_JETSON_PRODUCTION_ACCEPTANCE_ENABLED=true` only when that protected
   receipt should be available.

The host jobs need no private runner. They run automatically for pull requests
and pushes to `develop-alpha`, or manually from the `quality` workflow. The same
manual run is available from a configured GitHub CLI:

```bash
gh workflow run quality.yml --ref develop-alpha
gh run list --workflow quality.yml --branch develop-alpha \
  --event workflow_dispatch --limit 1 --json databaseId,url
gh run watch <run-id> --exit-status
```

Both Jetson enable variables may remain true once the runner is stable. The
production job still waits for the protected environment approval and never
enables physical kmNet output. Because this is a public repository, Jetson jobs
never run pull-request code: they accept only a direct `develop-alpha` push or
an authorized manual workflow dispatch. Pull requests use GitHub-hosted runners
for host checks.

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
The environment variable `NOVASIGHT_JETSON_MODEL_FIXTURE_DIR` must point outside
the checkout to an approved TensorRT engine fixture with its matching runtime
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
