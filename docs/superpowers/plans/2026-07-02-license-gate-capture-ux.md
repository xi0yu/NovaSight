# License Gate and Capture UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate operational APIs behind local license activation, improve diagnostic logging, and make the capture workbench easier to use on high-capability capture cards.

**Architecture:** The backend keeps local activation state in `data/license.json`; middleware allows only health and license endpoints before activation. The license store accepts a built-in maximum-permission test key now and supports future signed license tokens with created/activated/duration metadata. The React app starts at a Chinese license gate, then loads the console after activation.

**Tech Stack:** FastAPI middleware, Python standard logging, optional RSA signature verification through `cryptography`, React + TypeScript, Vite.

---

### Task 1: Backend License Model and Gate

**Files:**
- Modify: `novasight/license.py`
- Modify: `novasight/api/app.py`
- Modify: `novasight/api/routes_runtime.py`
- Test: `tests/test_runtime_api.py`

- [x] Add structured license status fields: configured, valid, tier, features, created_at, activated_at, expires_at, duration.
- [x] Accept `NOVASIGHT-TEST-MAX-ACCESS-2026` as a max-permission test key.
- [x] Add middleware that returns 401 for protected API and websocket paths when no valid license exists.
- [x] Keep `/healthz`, `/api/license`, `/api/license/activate`, and `/api/config/schema` open.

### Task 2: Logging and Capture Error Clarity

**Files:**
- Modify: `novasight/api/routes_capture.py`
- Modify: `novasight/api/routes_runtime.py`
- Modify: `web/src/api.ts`

- [x] Log capture select request parameters and failure `last_error`.
- [x] Log license activation, clear, and failures.
- [x] Make frontend API errors prefer `last_error`, `reason`, then `detail`.

### Task 3: Frontend License Gate and Capture Grouping

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [x] Load license status before operational API calls.
- [x] Show a Chinese activation screen when not licensed.
- [x] Provide a one-click test max-access key fill.
- [x] Group capture capabilities by pixel format to reduce visual noise.

### Task 4: Verification

**Files:**
- Modify: `tests/test_runtime_api.py`

- [x] Cover protected API rejection before activation and test license activation.
- [x] Run `pytest -q`.
- [x] Run `pnpm --dir web typecheck`.
- [x] Run `pnpm --dir web build`.
