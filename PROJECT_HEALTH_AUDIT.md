# NovaSight Project Health Audit

Date: 2026-07-08

Purpose: give future agents and developers one current cleanup ledger for
NovaSight. This is not a product roadmap. It is a health audit for deciding what
to keep, merge, delete, defer, or verify before adding more design.

## Unified Decisions

These decisions are treated as current project law unless the owner explicitly
changes them:

- Jetson-first runtime. macOS is a development fallback.
- The product mainline is:

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

- DeepStream remains opt-in until Jetson smoke evidence proves correctness and
  speed.
- The legacy CPU/backend path stays available until DeepStream has A/B evidence.
- `DetectionBatch` is the downstream seam. Tracker, selector, Kalman, control,
  scheduler, and device adapters should not depend on GStreamer, `pyds`, raw
  tensors, or DeepStream types.
- Model assets are first-class runtime assets: `.engine` or `.onnx`,
  `model.manifest.json`, generated `deepstream.ini`, and fingerprints belong
  together.
- HID/KMBOX output is blocked unless detection freshness, coordinate contract,
  calibration, scheduler, and device state are valid.
- Host/UI is a management surface, not part of the real-time loop.
- Audit answers use `YES / PARTIAL / NO / UNKNOWN` with evidence paths and the
  smallest next verification.

## Snapshot

Measured from the current checkout:

| Area | Current count | Health signal |
| --- | ---: | --- |
| Python source files | 121 | Medium-sized backend; most modules are reasonable, but several files are too large. |
| Python source lines | 31,639 | Large enough that hidden coupling is now likely. |
| Tests | 10 files, 12,676 lines | Strong evidence base, but concentrated in a few very large files. |
| Tracked docs | 33 files | Useful history, but too many plan/spec files compete with current truth. |
| `docs/superpowers` plans/specs | 10,266 lines | Archive value is high; day-to-day authority should be reduced. |
| Git-tracked generated artifacts | 0 found for `.audit`, `.egg-info`, caches, logs, or `data` | Good. Keep this invariant. |

Largest current files:

| File | Lines | Health note |
| --- | ---: | --- |
| `tests/test_inference_runtime.py` | 7,004 | Too concentrated; split by model/runtime/DeepStream/API behavior when touched. |
| `web/src/styles.css` | 5,679 | High visual maintenance risk; split only when UI work is in scope. |
| `novasight/main.py` | 3,508 | CLI/doctor logic is too broad; prime refactor candidate. |
| `novasight/runtime/service.py` | 2,782 | Runtime orchestration is too dense; extract status/report helpers before adding features. |
| `web/src/features/studio/StudioConsoleView.tsx` | 2,422 | UI surface likely mixes several panels; defer until UI work resumes. |
| `novasight/api/routes_models.py` | 1,637 | Model route owns too many responsibilities; split API schemas and orchestration helpers. |

## KEEP

Keep these modules and documents as current authority.

| Item | Keep reason | Boundary |
| --- | --- | --- |
| `docs/ai-programming-team.md` | Defines agent operating protocol and dirty workspace categories. | Update only when team rules change. |
| `docs/novasight-ai-purpose-audit-questions.md` | Canonical AI audit checklist. | Use for evidence questions, not design prose. |
| `docs/novasight-deepstream-code-plan.md` | Current DeepStream/NVMM implementation plan and invariants. | Keep aligned with code after DeepStream changes. |
| `README.md` | User entrypoint for dev, runtime API, Jetson diagnostics. | Should stay shorter than detailed audit docs. |
| `novasight/deepstream/*` | Current DeepStream backend seam. | Still opt-in and evidence-gated. |
| `novasight/runtime/*` | Runtime mainline state, pipeline, reconfiguration, tracking, telemetry. | Needs slimming, not deletion. |
| `novasight/model_registry/*` | Model assets are first-class; registry is core. | Split large API wrappers before changing store semantics. |
| `novasight/control/*` and `novasight/executors/*` | Control intent and device execution are separate, which is correct. | Keep output safety checks explicit. |
| `novasight/capture/session.py` | Useful seam for threaded capture integration. | Prefer it over ad hoc capture loops. |
| `config/novasight.example.yaml` | Shared config reference. | Must match runtime schema. |
| `deploy/*` | Deployment support. | Verify on Jetson before treating as production-ready. |

## MERGE

These are not deletion targets yet. They should be merged or collapsed when the
next related feature touches them.

| Item | Merge target | Why |
| --- | --- | --- |
| `docs/superpowers/plans/*` and `docs/superpowers/specs/*` | A smaller `docs/archive/` or one indexed history file | Historical plans are useful, but agents should not treat them as current authority. |
| `code_audit.md` and `docs/superpowers/plans/2026-07-04-product-health-audit.md` | This file, then archive the older audits | Avoid multiple competing health-audit truths. |
| `deepseek_project.md` | Keep as external/input plan, but summarize accepted decisions into current docs | It is useful source material, not current project law by itself. |
| `novasight/main.py` doctor commands | `novasight/doctor/*` or focused command modules | CLI/doctor code is too large and hard to review. |
| `novasight/api/routes_models.py` request/response models | `novasight/api/model_schemas.py` or route-local helpers | The route file mixes schemas, registry calls, DeepStream prepare, and runtime state. |
| `novasight/runtime/service.py` status/report construction | `novasight/runtime/status.py`, `telemetry.py`, or `recorder.py` | RuntimeService should orchestrate, not format every report payload. |
| `tests/test_inference_runtime.py` | Split by backend, model binding, preprocessing, DeepStream behavior | One huge test file slows review and obscures failure ownership. |
| `web/src/styles.css` | Feature-local styles or tokenized component styles | Only do this when touching UI; do not churn CSS just for cleanup. |

## DELETE

Do not delete source code blindly. These are safe deletion candidates after a
quick `git grep`/smoke check, or immediate deletion targets if they reappear as
untracked dirty files.

| Item | Delete condition | Reason |
| --- | --- | --- |
| `.audit/` | If present as untracked/local output | Disposable analysis output. |
| `*.egg-info/` | If present as untracked/local output | Generated package metadata. |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | If present as untracked/local output | Tool caches. |
| `logs/novasight.log` and crash logs | If untracked/local output | Runtime output, not source. |
| `data/novasight.db`, `data/license.json`, model binaries under `data/` | If untracked/local state | Local runtime state and assets. Keep model fixtures only if explicitly added. |
| Duplicate health/audit docs | After this file becomes the active health ledger | Reduce conflicting advice for agents. |
| Landing-page work from runtime commits | If mixed into DeepStream/runtime changes | Landing is outside the runtime mainline unless explicitly requested. |

Concrete source code deletion candidates to verify before removing:

| Candidate | Verify first | Expected outcome |
| --- | --- | --- |
| `novasight/detection/*` | Check whether runtime still imports these old target/Kalman/ROI helpers. | Delete or mark legacy if `novasight/runtime/*` fully replaced them. |
| `novasight/pipelines/*` | Check whether these are diagnostics only or still used by doctor/runtime commands. | Keep as doctor support or merge into `novasight/deepstream/*`. |
| `novasight/inference/decoders/*` | Confirm whether shared YOLO parser makes decoder wrappers redundant. | Keep only if runtime selection needs them. |
| Old `experimental_*` naming in config/control | Verify migration and UI compatibility. | Rename only with config migration; do not break existing configs casually. |

## DEFER

These are real concerns but not the next bottleneck.

| Item | Why defer |
| --- | --- |
| Large UI cleanup | Runtime/DeepStream evidence is currently more important than UI polish. |
| Full CSS architecture rewrite | High churn, low immediate runtime value. |
| Non-MJPEG DeepStream variants | Current target is `/dev/video0` MJPEG 1920x1080@120. Add other formats after the main path is proven. |
| Removing the legacy CPU backend | Explicitly forbidden until Jetson evidence proves DeepStream stability. |
| Making DeepStream default | Requires Jetson smoke reports meeting timestamp, FPS, frame-age, and postprocess thresholds. |
| Broad test-suite reshaping | Split files opportunistically when touching related behavior. |
| Native Jetson preprocess expansion | Useful later, but do not let it distract from DeepStream tensor-meta evidence. |

## VERIFY

These need evidence before the project can claim full health.

| Question | Current status | Smallest verification |
| --- | --- | --- |
| Is DeepStream the active production detection path? | PARTIAL | Run `/api/runtime/start` with `inference.backend=deepstream` on Jetson and confirm runtime status fields. |
| Does tensor meta extraction match the manifest shape/dtype? | UNKNOWN without Jetson `pyds` run | `novasight doctor deepstream-smoke --report-json ...` on Jetson. |
| Does the DeepStream path sustain control-ready freshness? | UNKNOWN | Check smoke report with tensor-meta FPS, DetectionBatch FPS, frame age, and latency thresholds. |
| Is `DetectionBatch` the single downstream seam? | PARTIAL/VERIFY | Search downstream runtime/control for backend-specific imports and raw tensor usage. |
| Are stale commands always cancelled on source/model/target/device changes? | PARTIAL | Focused scheduler/runtime tests around restart, model switch, and device error. |
| Can model switch never mix old tensors/tracker state with new model outputs? | PARTIAL | Exercise publish/rollback/start/stop path with status and tracker reset assertions. |
| Are health/status payloads truthful about selected backend, not only availability? | PARTIAL | Compare `/api/status`, runtime state, and DeepStream backend status during both legacy and DeepStream runs. |

## Current Health Scorecard

| Dimension | Rating | Why |
| --- | --- | --- |
| Product direction | GOOD | Decisions are clear and consistent across current docs. |
| Runtime safety | MEDIUM | Many safeguards exist, but verification depends on focused runtime and Jetson checks. |
| DeepStream readiness | MEDIUM-LOW | Code path exists and is opt-in; Jetson tensor-meta proof remains the gate. |
| Module boundaries | MEDIUM | Boundaries exist, but large files show orchestration is getting too dense. |
| Documentation health | MEDIUM-LOW | Good current docs, too much historical plan mass. |
| Test health | MEDIUM | Strong coverage signal, but test files are too concentrated. |
| Workspace hygiene | GOOD | No tracked generated/cache/runtime artifacts found in the checked categories. |

## Next Cleanup Order

1. Make this file the current health ledger.
2. Move or label old `docs/superpowers` plans/specs as archive so agents do not
   treat them as live instructions.
3. Split `novasight/main.py` only along doctor command boundaries.
4. Split `novasight/api/routes_models.py` by schemas, registry operations, and
   runtime side effects.
5. Extract status/report payload construction from `novasight/runtime/service.py`
   before adding new runtime features.
6. Verify whether `novasight/detection/*` and `novasight/pipelines/*` are still
   current source or historical support code.
7. Run Jetson DeepStream smoke evidence before promoting any DeepStream defaults.

## Hard Stop Rules

- Do not delete legacy runtime paths until DeepStream has measured Jetson
  evidence.
- Do not rename `experimental_angle_*` config keys without a migration plan.
- Do not merge landing-page work into runtime/control commits.
- Do not add new abstractions until an item in `MERGE`, `DELETE`, or `VERIFY`
  has been resolved.
- Do not claim project health from docs alone; use code paths, commands, status
  payloads, and Jetson reports.
