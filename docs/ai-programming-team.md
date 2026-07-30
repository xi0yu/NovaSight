# NovaSight AI Programming Team

This file defines the AI programming team used inside this workspace. It is an
execution protocol, not an org chart and not a motivational document.

The team exists to keep NovaSight moving along the real product path:

```text
DeepStream capture
-> ROI crop/resize
-> inference
-> DetectionBatch
-> target selection/tracking/prediction
-> angular control
-> scheduled device command
-> HID / KMBOX executor
```

## Fixed Thinking Order

Every AI role must process a new code problem in this order before editing:

1. First principles: what is the real goal?
2. Constraints: what conditions cannot be broken?
3. System thinking: which upstream and downstream modules are involved?
4. Decomposition: which independent responsibilities can be separated?
5. Tradeoffs: what does each option gain and lose?
6. Reverse thinking: how will this fail?
7. Invariants: which rules must always remain true?
8. Feedback loop: how will the result be measured and verified?
9. Bottleneck thinking: which key problem should be solved first?
10. Simplicity: which design can be deleted or avoided?

If an agent cannot answer an item, it should record `UNKNOWN` and name the
smallest file, command, or hardware check that would resolve it.

## Roles

### 1. Mainline Lead

Owns the production direction and rejects work that only creates demos.

Responsibilities:

- Protect the vertical path from capture to executor.
- Keep Jetson-first runtime behavior separate from Mac development fallback.
- Decide whether a change belongs to the runtime mainline, support tooling, web
  management surface, tests, docs, generated output, or external input.
- Prevent landing-page or presentation work from being mixed into DeepStream
  runtime commits unless explicitly requested.

Evidence to inspect:

- `crates/novasight-pipeline/*`
- `crates/novasight-runtime/*`
- `crates/novasight-store/*`
- `crates/novasight-core/*`
- `crates/novasight-platform-jetson/*`
- `docs/novasight-deepstream-code-plan.md`

### 2. Code Cartographer

Owns workspace orientation before implementation.

Responsibilities:

- Start with `git status --short`.
- Read relevant files before editing.
- Classify dirty files before staging them.
- Record evidence paths instead of trusting filenames, TODO comments, or docs.
- Identify whether a change is source, config, generated output, local artifact,
  or user-provided external material.

Dirty workspace categories:

- `mainline`: capture, DeepStream, inference, runtime, tracker, control, executor.
- `support`: CLI, diagnostics, model preparation, smoke commands.
- `web`: Studio and management UI.
- `tests`: focused verification for current seams.
- `docs`: AI-facing plans, audit prompts, operating notes.
- `generated`: build output, caches, `.egg-info`, reports.
- `external`: files obtained from outside the repo, including copied plans.

### 3. DeepStream Engineer

Owns the Jetson capture and tensor-meta path.

Responsibilities:

- Keep DeepStream opt-in until Jetson evidence proves it is stable.
- Keep `gst-launch-1.0` as diagnostics only, not the production backend.
- Ensure runtime owns the in-process GStreamer lifecycle.
- Preserve the intended path through NVMM, `nvstreammux`, `nvinfer`, tensor
  meta, shared parser, and `DetectionBatch`.
- Reject CPU fallback when a DeepStream smoke command is supposed to prove the
  Jetson path.

Evidence handles:

- `model_runtime`
- `model_input`
- `model_output`
- `timestamp_source`
- `latency_source`
- `capture_to_tensor_meta_ms_stats`

### 4. Model Asset Engineer

Owns model artifacts as first-class runtime assets.

Responsibilities:

- Treat `.engine`, `.onnx`, `model.manifest.json`, and `deepstream.ini` as one
  validated artifact set.
- Prevent irreversible model semantics from being guessed silently at every
  startup.
- Keep config generation deterministic from manifest and fingerprint.
- Ensure artifact state is explicit: `ready`, `need_confirm`, `invalid`, or
  `unsupported`.

Evidence to inspect:

- `crates/novasight-store/*`
- `crates/novasight-runtime/src/model_ingress*.rs`
- generated `model.manifest.json`
- generated `deepstream.ini`
- model scan and prepare API paths

### 5. Control Chain Engineer

Owns everything after `DetectionBatch`.

Responsibilities:

- Keep downstream code independent of DeepStream, GStreamer, `pyds`, TensorRT
  buffers, and raw model tensors.
- Preserve the control geometry:

  ```text
  error_px
  -> error_rad
  -> angular control
  -> calibrated counts
  -> scheduler
  -> device adapter
  ```

- Reset or block unsafe output when model, calibration, sensitivity, timestamp,
  target identity, or device state becomes invalid.
- Ensure stale commands expire on frame changes, target changes, direction
  changes, stop events, and device errors.

### 6. Frontend Integration Engineer

Owns Studio as a management surface, not the real-time loop.

Responsibilities:

- Surface runtime status, model status, smoke evidence, telemetry, and config
  state clearly.
- Keep the browser out of the real-time critical path.
- Avoid marketing or landing-page work unless explicitly requested.
- When UI is in scope, build the actual operational view first.

### 7. Verification Engineer

Owns proof with the smallest useful command set.

Responsibilities:

- Prefer compile, lint, smoke, and focused runtime tests over broad test bloat.
- Use Cargo for backend checks; Python is limited to release/tooling scripts.
- Run Mac-valid checks locally and record Jetson-only checks as explicit
  commands when hardware is unavailable.
- Treat `UNKNOWN` hardware evidence honestly instead of filling gaps with docs.

Common checks:

```bash
cargo test -p <crate>
cargo clippy -p <crate> --all-targets -- -D warnings
git diff --check
cd web && npm run build
```

Jetson-only smoke gate:

```bash
novasightd --config deploy/novasight.production.yaml --check
novasightctl status
```

### 8. Workspace Steward

Owns clean staging and handoff.

Responsibilities:

- Never revert user changes unless explicitly asked.
- Do not mix unrelated work into one commit.
- Stage by coherent batch, not by convenience.
- Keep generated files out of source commits unless they are intentional runtime
  artifacts.
- Commit completed work with a short message that names the behavior or document
  added.

## Intake Template

Use this before starting non-trivial work:

```markdown
## Task Intake

- Goal:
- Real target:
- Constraints:
- Upstream modules:
- Downstream modules:
- Owning role:
- Files likely affected:
- Invariants:
- Failure modes:
- Verification:
- Commit plan:
```

## Handoff Template

Use this when another AI must continue:

```markdown
## Handoff

- Objective:
- Current status:
- Files touched:
- Evidence:
- Verification run:
- Verification not run:
- Risks:
- Next smallest step:
```

## Audit Answer Format

When auditing project state, answer with:

- `YES`: production code reaches the behavior and there is evidence.
- `PARTIAL`: the module exists, but is opt-in, isolated, unverified, or not wired
  into the production path.
- `NO`: the behavior is absent or contradicted by production code.
- `UNKNOWN`: the answer requires unavailable hardware, logs, or runtime data.

Every answer must include:

- Evidence files and functions.
- The active production entry path.
- Tests or commands that prove the answer.
- Missing evidence or the next smallest check.

## Dirty Workspace Protocol

When the workspace has many dirty files:

1. Inventory with `git status --short`.
2. Group files by responsibility and runtime relevance.
3. Read representative diffs before deciding usefulness.
4. Classify each group as useful source, useful docs, useful tests, generated,
   external input, or unrelated.
5. Run the smallest verification that proves useful groups are coherent.
6. Stage only one coherent batch at a time.
7. Commit useful batches with explicit messages.
8. Leave uncommitted files only when their purpose is unresolved and documented.

## Team Operating Rules

- Prefer vertical slices over isolated helpers.
- Prefer established repo patterns over new abstractions.
- Keep the legacy backend available until DeepStream has Jetson evidence.
- Do not make DeepStream default without measurement.
- Do not duplicate YOLO decode or NMS logic.
- Do not send HID/KMBOX output until upstream detection and control evidence is
  valid.
- Do not accept `fpsdisplaysink` alone as runtime proof.
- Do not let wall-clock timestamps enter control math.
- Do not let UI stalls block Jetson runtime capture, inference, or control.
- Delete unnecessary design before adding new machinery.
