# D1A Implementation Runbook

D1A is released by evidence, not by the presence of UI code. The browser and
host checks below can prove contracts and presentation. Only the two Jetson
jobs can prove the aarch64 package and real camera/DeepStream behavior.

## Milestone gates

| Gate | Platform | Command | Required evidence | Exit criterion |
| --- | --- | --- | --- | --- |
| M0 contracts | host | `pnpm --dir web typecheck && pnpm --dir web test:unit:ci` | passing TypeScript plus unit/contract report | runtime/auth/theme contracts pass |
| M1 safety | Jetson CI | `scripts/ci/jetson-production-acceptance.sh build` | `jetson-build-receipt` artifact | strict stopped tuple and LAN boundary pass |
| M2/M3a Overview | host browser + Jetson CI | `pnpm --dir web test:e2e`, then Jetson build job | Playwright report and receipt bound to the same SHA | `?page=overview` remains opt-in |
| M3b default flip | host browser + Jetson CI | change only `DEFAULT_CONSOLE_PAGE`, repeat M2/M3a gates | a new receipt bound to the flipped SHA | only then may Overview become default |
| M4 surfaces | host | typecheck, unit tests, `pnpm --dir web visual:audit` | per-surface diff/checklist | no internal evidence dominates primary tasks |
| M5 models/errors | host browser | unit tests and browser journey | active/no-op/reconcile plus bounded incident UI | daemon/runtime remains authoritative |
| M6 production | protected Jetson | `scripts/ci/jetson-production-acceptance.sh production` | `jetson-production-receipt` artifact | real metadata/freshness and final strict safety tuple pass |

Failure never advances the next gate. Re-run using the same commit only after
fixing an environmental failure; any source change creates a new SHA and needs
a new receipt. M3b can be reverted independently. Runtime safety and API
contracts must roll forward.

## Current accepted-file ledger

- Runtime identity/safety: `crates/novasight-api/src/control.rs`,
  `crates/novasight-api/src/dto/runtime_status.rs`, `web/src/contracts/`,
  `web/src/features/runtime/`, `web/src/App.tsx`.
- Studio/UI: `web/src/features/studio/`, `web/src/features/models/`,
  `web/src/features/license/`, `web/src/styles.css`,
  `web/src/design/tokens.css`.
- Verification/release: `web/e2e/`, `web/vitest.config.ts`,
  `web/playwright.config.ts`, `.github/workflows/quality.yml`,
  `scripts/ci/jetson-production-acceptance.sh`,
  `docs/jetson-production-acceptance.md`.

The checkout may contain older user-owned pending changes. Never stage D1A
with `git add -A`; review and stage explicit paths after checking
`git diff --cached --name-only`.

## Current boundary

The default remains `capture`. Local macOS checks do not run or build the Rust
product. Until current-SHA Jetson build and production receipts exist, Jetson
production readiness is `UNKNOWN`.
