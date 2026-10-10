# Inference Settings Runtime Link Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the basic settings inference panel accurately reflect the active model, runtime loading state, input shape, class names, and threshold synchronization.

**Architecture:** Runtime state becomes the single source of truth. Backend active model payload includes version metadata and `RuntimeService.update_config()` defensively synchronizes inference thresholds. Frontend basic settings uses that runtime payload to show a practical inference checklist rather than a detached parameter form.

**Tech Stack:** Python runtime service, model registry state payload, React + TypeScript settings UI.

---

## Files

- Modify: `novasight/runtime/service.py`
  - Include model version in `active_model`.
  - Synchronize inference thresholds in `update_config()` when the service owns an inference runtime.
- Modify: `novasight/config/schema.py`
  - Rename consumers inference label from TensorRT-specific copy to model inference.
- Modify: `web/src/api.ts`
  - Add optional `version` to `ActiveModel`.
- Modify: `web/src/features/devices/DevicesView.tsx`
  - Show current model version, input shape, class count, backend, loaded status, and threshold effect.
  - Show a practical readiness checklist: input ready, model runnable, runtime loaded, thresholds active.
  - Keep threshold editing in the same panel and link users back to model repository when no runnable model is bound.

## Task 1: Backend runtime state consistency

- [x] Add active version metadata to `RuntimeService._active_model()`.
- [x] Make `RuntimeService.update_config()` call `self.inference.configure(...)` when possible.
- [x] Change config schema consumer label to `模型推理`.
- [x] Verify with `python3 -m compileall -q novasight`.

## Task 2: Frontend active model typing

- [x] Add optional `version: ModelVersion | null` to `ActiveModel`.
- [x] Keep existing active model consumers compatible.

## Task 3: Basic settings inference panel

- [x] Display active model name, artifact kind, input shape, class count, and class preview.
- [x] Add readiness checklist based on capture/input, artifact kind, runtime loaded, and inference enabled.
- [x] Clarify that confidence/NMS updates affect the currently loaded model runtime.
- [x] Show a stronger action when no runnable ONNX/Engine is bound.
- [x] Verify with `pnpm --dir web build`.

## Completion

- [x] Run `python3 -m compileall -q novasight`.
- [x] Run `pnpm --dir web build`.
- [x] Run `git diff --check`.
- [ ] Commit and push to `origin/develop-alpha`.
