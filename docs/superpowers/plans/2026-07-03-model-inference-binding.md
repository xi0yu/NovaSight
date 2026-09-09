# Model Inference Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model selection in the model repository visibly bind to runtime inference, with clear success/failure feedback after publish and rollback.

**Architecture:** Keep the model registry as the persistence layer and `InferenceRuntime` as the runtime loading authority. The publish/rollback API will return both deployment information and current inference status. The frontend model repository will present model artifacts as user-facing runnable choices and display whether the selected model is actually loaded by the inference runtime.

**Tech Stack:** FastAPI model routes, SQLite-backed model registry, NovaSight inference runtime, React + TypeScript model repository UI.

---

## Files

- Modify: `novasight/api/routes_models.py`
  - Return inference status from publish and rollback.
  - Load the rolled-back artifact into inference runtime.
- Modify: `web/src/api.ts`
  - Add `ModelPublishResponse` with `deployment` and `inference`.
  - Update `publishModel()` and `rollbackModel()` return types.
- Modify: `web/src/features/models/ModelsView.tsx`
  - Show inference load status after publish/rollback.
  - Make artifact rows explain ONNX/Engine auto backend selection.
  - Disable publishing non-runnable `.pt` artifacts with a user-facing reason.
- Verify:
  - `python3 -m compileall -q novasight`
  - `pnpm --dir web build`
  - `git diff --check`

## Task 1: Backend publish/rollback runtime feedback

- [x] Add helper `_inference_status(request)` returning `request.app.state.inference.status()`.
- [x] Change publish response to `{ "deployment": ..., "inference": ... }`.
- [x] Change rollback to load `deployment.artifact_id` into inference runtime and return `{ "deployment": ..., "inference": ... }`.
- [x] If rollback points to a non-runnable artifact, call `disable()` with a clear reason.

## Task 2: Frontend API typing

- [x] Add `ModelPublishResponse` type.
- [x] Update `publishModel()` to return `ModelPublishResponse`.
- [x] Update `rollbackModel()` to return `ModelPublishResponse`.

## Task 3: Model repository UX binding

- [x] Store last inference load feedback from publish/rollback.
- [x] Display status, backend, loaded flag, and reason in the current model panel.
- [x] Change publish button labels from generic “设为当前” to “用于推理”.
- [x] Disable `.pt` publish buttons and show that `.pt` must be converted/exported before inference.
- [x] Explain `.onnx` and `.engine` backend selection by suffix.

## Completion

- [x] Run `python3 -m compileall -q novasight`.
- [x] Run `pnpm --dir web build`.
- [x] Run `git diff --check`.
- [ ] Commit and push to `origin/develop-alpha`.
