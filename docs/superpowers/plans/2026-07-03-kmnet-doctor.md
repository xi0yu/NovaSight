# KmNet Doctor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Jetson-friendly `python -m novasight doctor kmnet` command that verifies kmNet import, init, monitor, button state, and optional movement without starting the web app.

**Architecture:** Reuse `KmNetExecutor` as the single kmNet integration point. The CLI builds the executor from explicit command-line parameters, prints structured human-readable lines, optionally sends a tiny movement, and exits non-zero when the driver or connection is unavailable.

**Tech Stack:** Python argparse CLI, existing `novasight.executors.KmNetExecutor`, no new tests or scripts.

---

## Files

- Modify: `novasight/main.py`
  - Add `doctor kmnet` subcommand.
  - Add CLI options for host, port, uuid, monitor-port, move dx/dy, and no-move.
- Modify: `novasight/executors/kmnet.py`
  - Add `read_buttons()` for monitor verification.
  - Add `diagnostic_move(dx, dy)` helper using the same send path.
  - Keep status reporting explicit and safe.
- Modify: `docs/superpowers/plans/2026-07-03-kmnet-doctor.md`
  - Track execution.

## Task 1: Executor diagnostic helpers

- [x] Add `read_buttons()` returning a dict with left/right availability.
- [x] Add `diagnostic_move(dx, dy)` returning `ExecutionResult`.
- [x] Ensure unavailable driver returns a useful message instead of raising.

## Task 2: CLI command

- [x] Add `python -m novasight doctor kmnet --host ... --port ...`.
- [x] Print driver availability, connection state, monitor state, function support, button state, and movement result.
- [x] Exit `0` when kmNet is available and connected, otherwise `2`.

## Completion

- [x] Run `python3 -m compileall -q novasight`.
- [x] Run `python -m novasight doctor kmnet --port 0 --no-move`.
- [x] Run `git diff --check`.
- [ ] Commit and push to `origin/develop-alpha`.
