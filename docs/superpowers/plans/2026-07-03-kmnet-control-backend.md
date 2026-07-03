# KmNet Control Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize kmNet output and add a practical proportional control strategy without migrating the old project's experimental algorithm baggage.

**Architecture:** `ControlIntent` carries optional movement metadata. `ControlOutputPolicy` preserves that metadata after limit checks. `KmNetExecutor` becomes a stateful kmNet adapter supporting init/monitor/raw/auto/bezier/trace/buttons. `RuntimeService` uses real hardware trigger state when available and only uses diagnostic auto-trigger when no hardware box is configured.

**Tech Stack:** Python dataclasses, existing runtime config parser, kmNet Python extension when present, NovaSight control/executor abstractions.

---

## Files

- Modify: `novasight/config/runtime.py`
  - Add kmNet `uuid`, `monitor_port`, `flip_dy`.
  - Add control move mode, move duration, trace, deadzone, near/far speed.
- Modify: `novasight/config/schema.py`
  - Expose the new kmNet and proportional fields to frontend schema.
- Modify: `novasight/contracts.py`
  - Add optional `move_kind`, `move_ms`, `trace_ms`, `bezier_ctrl` to `ControlIntent`.
- Modify: `novasight/control/output.py`
  - Preserve move metadata in `ControlOutput`.
- Modify: `novasight/control/strategy.py`
  - Add `ProportionalStrategy` inspired by the clean `jetvision` implementation.
- Modify: `novasight/control/__init__.py`
  - Export `ProportionalStrategy`.
- Modify: `novasight/executors/kmnet.py`
  - Add init/monitor/raw/auto/bezier/trace/click/status support.
- Modify: `novasight/executors/runtime.py`
  - Construct `KmNetExecutor` from runtime config.
  - Include executor status details.
- Modify: `novasight/runtime/service.py`
  - Use real box input state when hardware is configured.
  - Use `ProportionalStrategy` when configured.
  - Pass move metadata to `ControlIntent`.

## Task 1: Runtime config and metadata

- [x] Add compatible config fields with defaults.
- [x] Add schema fields.
- [x] Add metadata fields to `ControlIntent` and `ControlOutput`.
- [x] Verify `python3 -m compileall -q novasight`.

## Task 2: Proportional strategy

- [x] Implement FOV/deadzone/near-far speed/EMA/atan count mapping.
- [x] Return move metadata for raw/auto/bezier and trace.
- [x] Export the strategy.

## Task 3: kmNet executor

- [x] Initialize kmNet with host/port/uuid.
- [x] Start monitor when monitor port is configured.
- [x] Implement raw, auto, bezier, trace, left/right.
- [x] Expose status fields: available, connected, move_count, last_dx, last_dy, last_error.

## Task 4: Runtime service trigger and strategy routing

- [x] Remove unconditional fake left trigger when hardware exists.
- [x] Preserve diagnostic active trigger only when no hardware box exists.
- [x] Route `control.strategy = proportional`.
- [x] Pass move metadata from `MoveCommand` into `ControlIntent`.

## Completion

- [x] Run `python3 -m compileall -q novasight`.
- [x] Run `pnpm --dir web build`.
- [x] Run `git diff --check`.
- [ ] Commit and push to `origin/develop-alpha`.
